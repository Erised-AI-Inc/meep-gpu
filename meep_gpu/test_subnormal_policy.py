"""The float32 subnormal policy: the flag, the three executors, and the audit.

This suite runs on the DEV LAPTOP, which is arm64 with neither CuPy nor Triton
installed. That is not a limitation to work around here — it is one of the facts
under test. MEEP's ``_set_zero_subnormals`` is compiled under
``#if HAVE_IMMINTRIN_H`` (x86 SSE), so on this machine ``mp.set_zero_subnormals(True)``
is a NO-OP and the ``"flush"`` policy is unattainable for the host path. The
package must say so, loudly, rather than run a different policy than the one it
was asked for, and :func:`test_flush_is_refused_on_a_host_where_meeps_knob_is_a_no_op`
is that assertion.

THE SAME MACHINE DECIDES THE DEFAULT, AND ONLY ONE SIDE OF THAT IS REACHABLE HERE.
The default request is ``match_meep``, answered by MEASURING what MEEP left this
host's FPU doing rather than by recognizing a machine string, so on this laptop it
resolves to ``"keep"`` — verified against real MEEP, in a subprocess, because
``import meep`` initializes MPI for the whole session. The flushing half of the
same rule cannot be produced on an arm64 host at all; what is pinned for it is the
MECHANISM (the resolution follows ``backends.subnormals_flushed()``, whatever it
says) plus a negative control that renaming ``platform.machine()`` changes no
answer. The x86 half is the conformance gate's business, on device.

The device halves are tested WITHOUT a device: the CuPy leg's decisions are made
on option tuples and directory names (both pure), and the Triton leg's are made on
LLVM IR text and PTX text (both pure). Those are the parts that decide whether the
policy is uniform; the parts that need a GPU are covered by
``parity/meep_gpu/gate_subnormal_policy.py``, whose case builders are exercised
here so a broken case cannot reach the device silently.

AND THE IR IS HANDED TO A REAL LLVM PARSER. The Triton fixtures below are the
shape Triton 3.1.0 actually emits — no attribute group, a ``!dbg`` metadata
attachment — because an earlier revision's fixtures were shapes Triton never
emits, all 59 tests passed, and the mechanism aborted the process on the first
real compile (``LLVM ERROR: expected '{' in function body``, SIGABRT, which no
``except`` can see). ``llvmlite`` is used where it is importable to settle the
question with LLVM's own parser rather than with a regex, and the misplacement is
kept as a NEGATIVE CONTROL so a test that could not fail is impossible here.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import meep_gpu
from meep_gpu import backends, subnormal_policy as sp

API_DIR = Path(__file__).resolve().parent.parent
GATE_PATH = API_DIR / "parity" / "meep_gpu" / "gate_subnormal_policy.py"


@pytest.fixture(autouse=True)
def _fresh_policy():
    """Every test starts with no policy installed AND with the FPU where it found it.

    THE SECOND HALF IS NEW AND IT IS NOT TIDINESS. ``_reset_for_tests`` clears this
    module's bookkeeping; it does not put the floating-point environment back,
    because an ATTAINED install deliberately stands — a process that asked to flush
    and got it should keep flushing. That is right in production and wrong in a
    suite: the moment the non-x86 fenv lever landed, any test that installed
    ``flush`` left the FPU flushing for every test after it, and two later tests
    that compute an exact IEEE keep-word failed IN A FULL-FILE RUN while passing in
    isolation. Measured 2026-09-10 on exactly that pair.

    A test that leaks process state into its neighbours is the one failure a green
    isolated run cannot see, so the entry state is measured and restored here rather
    than remembered by each test. ``_drive_host_fpu`` is the same operation the
    rollback path uses, and it is symmetric by construction.
    """
    sp._reset_for_tests()
    flushing_at_entry = backends.subnormals_flushed()
    try:
        yield
    finally:
        sp._reset_for_tests()
        if backends.subnormals_flushed() != flushing_at_entry:
            sp._drive_host_fpu(flushing_at_entry)


def _a_host_with_no_working_lever(monkeypatch):
    """Stand in for a host where NEITHER host lever can flush. Returns MEEP's stub.

    WHY THE TESTS BELOW NEED IT, and it is a change of PREMISE rather than of
    expectation. They were written on this arm64 laptop when ``flush`` was
    genuinely unattainable here: MEEP's ``set_zero_subnormals`` is the
    ``#if HAVE_IMMINTRIN_H`` no-op on this architecture and there was no second
    lever, so "install flush and watch it be refused" needed no arrangement at all.
    The Metal dispatch table cannot exist without ``flush`` — MPS flushes natively
    and offers nothing else — so ``subnormal_policy`` gained the Darwin fenv lever,
    and on this machine the refusal these tests are about no longer happens.

    THE REFUSAL IS STILL A REAL BRANCH and it is still the one that matters: a host
    with neither lever must refuse rather than half-apply. So the platform is
    ARRANGED instead of assumed — MEEP's knob a no-op, the fenv lever answering
    "did not work" — and the tests keep measuring what they always measured, on
    every architecture rather than on one.
    """
    module, state = _fake_meep(monkeypatch, effective=False)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lambda want_flush: False)
    return module, state


@pytest.fixture(scope="module")
def gate():
    """Import the conformance gate by path — ``parity/`` is not a package."""
    spec = importlib.util.spec_from_file_location("gate_subnormal_policy", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The flag: resolution order, validation, and the shadowing trap
# ---------------------------------------------------------------------------


def test_the_default_asks_what_meep_does_here_rather_than_naming_a_policy(monkeypatch):
    """The default is a QUESTION — ``match_meep`` — and never an installable name.

    Stock MEEP calls ``set_zero_subnormals(true)`` at init and never unsets it
    (``src/mympi.cpp:188``, call site ``:219``, through
    ``initialize::initialize() -> setup()``), but that function is compiled under
    ``#if HAVE_IMMINTRIN_H`` — x86 SSE — so MEEP flushes on some hosts and keeps on
    others. A constant ``"flush"`` default would make the startup call this module
    is designed for a HARD REFUSAL wherever MEEP's own knob is a no-op, measured:
    an install with no argument raised ``SubnormalPolicyUnattainable`` on this
    laptop. So the default names the question and :func:`resolve_match_meep`
    answers it; ``get_subnormal_policy`` hands back the ANSWER, which is always
    something an executor can be driven to.
    """
    monkeypatch.delenv(sp.POLICY_ENV, raising=False)
    assert sp.default_policy() == sp.MATCH_MEEP
    assert sp.MATCH_MEEP not in sp.POLICIES        # not a policy bytes can be cut under
    assert sp.MATCH_MEEP in sp.REQUESTABLE         # but one a caller may ask for
    record = sp.policy_resolution()
    assert record["requested"] == sp.MATCH_MEEP and record["source"] == "default"
    assert record["policy"] in sp.POLICIES
    assert sp.resolve_policy() == (record["policy"], "default")
    assert sp.get_subnormal_policy() == record["policy"]


def test_the_default_is_resolved_by_measurement_and_not_by_the_machine_string(monkeypatch):
    """The negative control for the whole change: rename the machine, get the same answer.

    The previous implementation was a platform table — ``platform.machine()`` in
    ``{x86_64, amd64, i386, i686, x86}`` meant flush — so under this monkeypatch it
    resolved to ``"flush"`` on this laptop and the install then refused, because
    the FPU it had claimed to know about had not moved. Nothing here may consult
    that string: the answer comes from what MEEP actually left the FPU doing.
    """
    monkeypatch.delenv(sp.POLICY_ENV, raising=False)
    honest = sp.resolve_match_meep()
    for machine in ("x86_64", "AMD64", "i686", "aarch64", "wildly-unknown"):
        monkeypatch.setattr(sp.platform, "machine", lambda machine=machine: machine)
        record = sp.resolve_match_meep()
        assert record["policy"] == honest["policy"], machine
        assert record["measured"] == honest["measured"], machine
        assert record["machine"] == machine                 # recorded, and only recorded
        assert sp.resolve_policy()[0] == honest["policy"], machine


def test_match_meep_resolves_by_reading_the_fpu_meeps_own_init_left(monkeypatch):
    """Both answers, from the same read — the x86 side reachable no other way here.

    This laptop is arm64, where MEEP's knob is a no-op, so the flushing half of the
    rule cannot be produced on it. What is pinned is the MECHANISM: with MEEP in
    ``sys.modules``, the resolution is whatever ``backends.subnormals_flushed()``
    says, and the record names that measurement rather than a platform.
    """
    _module, state = _fake_meep(monkeypatch, effective=False)

    state["flushing"] = True                      # what an x86 host reads after import
    record = sp.resolve_match_meep()
    assert record["policy"] == sp.FLUSH
    assert record["measured"] is True and record["flushing"] is True
    assert record["mechanism"].startswith("backends.subnormals_flushed()")
    assert "resolved to 'flush'" in record["why"] and "flushing" in record["why"]
    assert record["fallback"] is None and record["reasons"] == []

    state["flushing"] = False                     # what this host reads after import
    record = sp.resolve_match_meep()
    assert record["policy"] == sp.KEEP
    assert record["measured"] is True and record["flushing"] is False
    assert "resolved to 'keep'" in record["why"] and "keeping" in record["why"]


def test_match_meep_resolves_to_keep_against_real_meep_on_this_host():
    """The real thing, in a subprocess: import MEEP, then read what it left.

    In-process fakes pin what the module DOES with a measurement; this pins the
    measurement itself, on the machine the suite runs on. It is a subprocess
    because ``import meep`` initializes MPI for the whole session, which this suite
    deliberately never does.

    Measured here (arm64, MEEP 1.33.0): flushing False before ``import meep``,
    False after, and False again after ``mp.set_zero_subnormals(True)`` — the
    ``#if HAVE_IMMINTRIN_H`` no-op. THE X86 HALF IS NOT VERIFIABLE ON THIS MACHINE;
    the test above pins the mechanism that produces it.
    """
    if importlib.util.find_spec("meep") is None:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("meep", "the default resolves by measuring what MEEP "
                                       "itself left this host's FPU doing")
    source = (
        "import json, meep\n"
        "from meep_gpu import subnormal_policy as sp\n"
        "print('RECORD ' + json.dumps(sp.resolve_match_meep()))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], cwd=str(API_DIR), text=True,
        capture_output=True, timeout=300,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
    assert completed.returncode == 0, completed.stderr[-2000:]
    line = [entry for entry in completed.stdout.splitlines()
            if entry.startswith("RECORD ")]
    assert line, completed.stdout[-2000:]
    record = json.loads(line[0][len("RECORD "):])
    assert record["meep_imported"] is True
    assert record["measured"] is True, record
    assert record["attributable_to_meep"] is True
    # The RULE, on whatever host this is: the answer follows the read, both ways.
    assert record["policy"] == (sp.FLUSH if record["flushing"] else sp.KEEP), record
    if record["flushing"]:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "meep_keeps_subnormals",
            f"MEEP flushes on this {record['machine']} host: the keeping half of the "
            f"rule is not reachable here, and the flushing half is the conformance "
            f"gate's business on device")
    # And what it actually is here, recorded rather than assumed.
    assert record["policy"] == sp.KEEP
    assert record["flushing"] is False
    assert "MEEP's own initialization left this host keeping" in record["why"]


def _run_meep_free(snippet: str) -> dict:
    """Run ``snippet`` in a FRESH interpreter that has not imported MEEP.

    The unmeasurable-case contracts are statements about a process in which MEEP
    is absent from ``sys.modules``. That is a PROCESS-GLOBAL precondition, and in
    a full-suite run some earlier test file has always imported MEEP — asserting
    the precondition in-process makes these tests order-dependent (measured: they
    passed alone and failed after 3,400 tests). A subprocess is the only honest
    home for the claim; the snippet must print one JSON object to stdout.
    """
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    env.pop(sp.POLICY_ENV, None)  # the DEFAULT is under test, not the override
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, (
        f"meep-free probe failed rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_MEEP_FREE_PRELUDE = (
    "import json, sys\n"
    "sys.path.insert(0, '.')\n"
    "from meep_gpu import subnormal_policy as sp\n"
    "assert sp._meep_module() is None, 'fresh interpreter unexpectedly has meep'\n"
)


def test_an_unmeasurable_default_states_its_fallback_instead_of_picking_a_side():
    """No MEEP in the process — a standalone engine user — is not a licence to guess.

    The module may not import MEEP to find out (that would initialize MPI as a side
    effect of choosing a flag), so the question has no answer there. The record says
    ``measured=False``, names the fallback AS a fallback, says why it is that one,
    and says what to do to get the question answered instead. Runs in a fresh
    interpreter because the claim is about a MEEP-free process, which the suite's
    own process stops being as soon as any other file imports MEEP.
    """
    record = _run_meep_free(_MEEP_FREE_PRELUDE +
                            "print(json.dumps(sp.resolve_match_meep()))\n")
    assert record["measured"] is False
    assert record["flushing"] is None
    assert record["attributable_to_meep"] is False
    assert record["fallback"] == sp.MATCH_MEEP_FALLBACK == sp.KEEP
    assert record["policy"] == sp.MATCH_MEEP_FALLBACK
    assert len(record["reasons"]) == 1
    assert "not measurable" in record["reasons"][0]
    assert "initializes MPI" in record["reasons"][0]
    assert "could NOT be measured" in record["why"]
    assert "FALLBACK, not a measurement of MEEP" in record["why"]
    assert "attainable on every host" in record["why"]   # why keep and not flush


def test_the_unmeasurable_case_says_which_unmeasurable_case_it_is():
    """"MEEP is not installed" and "MEEP is not imported yet" need different actions."""
    both = _run_meep_free(
        _MEEP_FREE_PRELUDE +
        "installed = sp.resolve_match_meep()\n"
        "sp._meep_installed = lambda: False\n"
        "absent = sp.resolve_match_meep()\n"
        "print(json.dumps({'installed': installed, 'absent': absent}))\n")
    record = both["installed"]
    assert record["meep_installed"] is True         # it is installed in this env
    assert "has not been imported in this process" in record["reasons"][0]

    absent = both["absent"]
    assert absent["policy"] == sp.MATCH_MEEP_FALLBACK
    assert "not installed in this environment" in absent["reasons"][0]
    assert "no MEEP behaviour on this host to match" in absent["reasons"][0]


def test_the_default_is_attainable_on_this_host_without_meep_imported():
    """The default must never be a policy this host cannot deliver.

    This is the assertion behind the deferred startup wiring: whatever the default
    resolves to here must install, on this machine, in a process that has not
    imported MEEP — which is the case where the answer is the fallback, so the
    fallback is under test too. A fresh interpreter for the same reason as above;
    it also keeps ``install_subnormal_policy(strict=True)``'s FPU mutation out of
    the suite's own process.
    """
    stamp = _run_meep_free(
        _MEEP_FREE_PRELUDE +
        "stamp = sp.install_subnormal_policy(strict=True)\n"
        "stamp['_default'] = sp.default_policy()\n"
        "print(json.dumps(stamp, default=str))\n")
    assert stamp["requested"] == stamp["_default"] == sp.MATCH_MEEP
    assert stamp["resolved"] == sp.KEEP
    assert stamp["unattained"] == []
    assert stamp["executors"]["host"]["attained"] is True


def test_the_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv(sp.POLICY_ENV, "keep")
    assert sp.resolve_policy() == ("keep", "environment")
    assert sp.get_subnormal_policy() == "keep"
    # And the request is recorded as an answer, not as a question: nothing was
    # measured, so nothing may claim to have been.
    assert sp.policy_resolution() == {"requested": "keep", "source": "environment",
                                      "policy": "keep", "resolution": None}


def test_an_explicit_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(sp.POLICY_ENV, "keep")
    assert sp.resolve_policy("flush") == ("flush", "argument")


def test_an_explicit_request_is_untouched_by_what_this_host_measures(monkeypatch):
    """``match_meep`` may not leak into either explicit answer, in either direction.

    The whole point of the flag is that a user can demand the policy their
    comparison is against even where it disagrees with this host — so an explicit
    request must not be re-measured, re-derived or downgraded. (Whether it is
    ATTAINABLE is a separate verdict, taken by ``install_host_policy``.)
    """
    _module, state = _fake_meep(monkeypatch, effective=False)
    for measured in (True, False):
        state["flushing"] = measured
        for requested in (sp.FLUSH, sp.KEEP):
            record = sp.policy_resolution(requested)
            assert record["policy"] == record["requested"] == requested
            assert record["resolution"] is None, "an explicit request was measured"
            assert record["source"] == "argument"
    monkeypatch.setenv(sp.POLICY_ENV, sp.MATCH_MEEP)
    assert sp.policy_resolution("flush")["policy"] == sp.FLUSH


def test_match_meep_can_be_asked_for_by_name_through_the_environment(monkeypatch):
    """It is a request like any other — the default is just the one nobody typed."""
    monkeypatch.setenv(sp.POLICY_ENV, "  MATCH_MEEP \n")
    record = sp.policy_resolution()
    assert record["requested"] == sp.MATCH_MEEP and record["source"] == "environment"
    assert record["policy"] in sp.POLICIES
    assert record["resolution"]["why"]
    assert sp.policy_resolution(sp.MATCH_MEEP)["source"] == "argument"


def test_the_policy_name_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv(sp.POLICY_ENV, "  KEEP \n")
    assert sp.resolve_policy() == ("keep", "environment")


def test_an_empty_environment_value_falls_through_to_the_default(monkeypatch):
    """An exported-but-empty variable is "unset", not "the empty policy"."""
    monkeypatch.setenv(sp.POLICY_ENV, "")
    record = sp.policy_resolution()
    assert record["requested"] == sp.default_policy() and record["source"] == "default"
    assert sp.resolve_policy() == (record["policy"], "default")


def test_an_unknown_policy_is_refused_by_name_and_says_where_it_came_from(monkeypatch):
    monkeypatch.setenv(sp.POLICY_ENV, "ftz")
    with pytest.raises(ValueError) as excinfo:
        sp.resolve_policy()
    message = str(excinfo.value)
    assert "'ftz'" in message and "environment" in message
    assert "flush" in message and "keep" in message
    assert "match_meep" in message and "measures which of those MEEP" in message


def test_the_getter_does_not_shadow_the_module_it_lives_in():
    """``meep_gpu.get_subnormal_policy``, never ``meep_gpu.subnormal_policy``.

    The package documents this trap for ``harminv``: binding the lowercase name
    on the package would make ``import meep_gpu.subnormal_policy as x`` hand back
    a function instead of the module.
    """
    assert isinstance(meep_gpu.subnormal_policy, types.ModuleType)
    assert meep_gpu.get_subnormal_policy is sp.get_subnormal_policy
    assert not isinstance(getattr(meep_gpu, "subnormal_policy"), types.FunctionType)


def test_the_package_exports_the_flag_at_package_level():
    for name in ("set_subnormal_policy", "get_subnormal_policy",
                 "install_subnormal_policy", "policy_stamp", "default_policy",
                 "resolve_match_meep",
                 "SubnormalPolicyUnattainable", "SubnormalPolicyLocked",
                 "FLUSH", "KEEP", "MATCH_MEEP"):
        assert name in meep_gpu.__all__, name
        assert hasattr(meep_gpu, name), name


# ---------------------------------------------------------------------------
# The arm64 no-op: the refusal that is the whole reason this is a flag
# ---------------------------------------------------------------------------


def _fake_meep(monkeypatch, *, effective: bool):
    """A MEEP stand-in whose knob either works or is the aarch64 no-op.

    Real MEEP is deliberately NOT imported by this suite: importing it starts MPI
    for the whole session. The arm64 no-op is measured elsewhere (with MEEP
    imported, ``mp.set_zero_subnormals(True)`` leaves ``subnormals_flushed()``
    False on this machine); what is pinned here is what the module DOES with that
    measurement.
    """
    state = {"calls": [], "flushing": backends.subnormals_flushed()}

    def set_zero_subnormals(flag):
        state["calls"].append(bool(flag))
        if effective:
            state["flushing"] = bool(flag)

    module = types.ModuleType("meep")
    module.set_zero_subnormals = set_zero_subnormals
    monkeypatch.setitem(sys.modules, "meep", module)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: state["flushing"])
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: state["flushing"])
    return module, state


def test_flush_is_refused_on_a_host_where_meeps_knob_is_a_no_op(monkeypatch):
    """The aarch64 case: the knob is called, nothing changes, and that is a REFUSAL.

    ``_set_zero_subnormals`` is compiled under ``#if HAVE_IMMINTRIN_H`` (x86 SSE),
    so on this laptop it does nothing — measured with real MEEP imported:
    ``flushing`` False before the call and False after, on machine 'arm64'. The
    verdict comes from that measurement and never from a platform table.
    """
    _module, state = _fake_meep(monkeypatch, effective=False)
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        report = sp.install_host_policy(sp.FLUSH, strict=False)
    assert state["calls"] == [True]          # the knob WAS asked
    assert report["attained"] is False       # and the measurement disagreed
    assert report["flushing_after"] is False
    assert any("HAVE_IMMINTRIN_H" in reason for reason in report["reasons"])
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_host_policy(sp.FLUSH, strict=True)
    assert "UNATTAINABLE" in str(excinfo.value)
    assert "'host'" in str(excinfo.value)


def test_flush_is_attained_where_the_knob_actually_moves_the_fpu(monkeypatch):
    """The x86 case, which no assertion on this laptop could otherwise reach."""
    _module, state = _fake_meep(monkeypatch, effective=True)
    report = sp.install_host_policy(sp.FLUSH, strict=True)
    assert report["attained"] is True
    assert report["flushing_before"] is False and report["flushing_after"] is True
    assert report["mechanism"] == "meep.set_zero_subnormals"
    # And the exception the host cannot deliver under flush is RECORDED, not hidden.
    assert [entry["ops"] for entry in report["exceptions"]] == [
        sp.EXECUTOR_OP_EXCEPTIONS["host_negation"]["ops"]]


def test_installing_a_policy_never_imports_meep(monkeypatch):
    """``backends.py``: "none may import meep at all".

    Not a style rule here — ``import meep`` initializes MPI, so an install that
    imported it would join a communicator and print ``Using MPI version 3.1``
    as a side effect of setting a flag. Measured on an earlier revision, which
    did exactly that from ``meep_gpu.set_subnormal_policy()``.

    Asserted rather than observed, so the answer does not depend on whether some
    OTHER module in the session already imported MEEP: the import machinery is
    trapped for the duration, and ``sys.modules`` is emptied of MEEP (restored by
    ``monkeypatch``) so the install is put in exactly the position where an
    importing implementation would import.
    """
    real_import_module = importlib.import_module

    def refuse_meep(name, *args, **kwargs):
        assert name.partition(".")[0] != "meep", (
            "the policy imported MEEP, which initializes MPI as a side effect")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", refuse_meep)
    monkeypatch.delitem(sys.modules, "meep", raising=False)
    # ``default_policy()`` is now the one request that WANTS to know what MEEP
    # does, so this is the install most tempted to import it — and the one that
    # must instead record that it could not measure.
    stamp = sp.install_subnormal_policy(sp.default_policy(), strict=True)
    assert "meep" not in sys.modules
    assert sp._meep_module() is None
    assert stamp["requested"] == sp.MATCH_MEEP
    assert stamp["match_meep"]["measured"] is False


def test_a_refused_install_puts_the_host_fpu_back(monkeypatch):
    """A refusal must not leave the process in the refused policy's mode.

    ``install_host_policy`` runs FIRST and mutates the process MXCSR. An earlier
    revision restored only the bookkeeping, so a refused ``"flush"`` left the
    process flushing every float32 op with ``policy_is_installed()`` False —
    bytes attributable to neither policy, which is the exact state the refusal
    exists to prevent. Invisible on arm64 (the knob is a no-op there), so the
    knob is a stand-in that works.
    """
    _module, state = _fake_meep(monkeypatch, effective=True)
    # A cache directory cut under keep is refused under flush — the refusal fires
    # AFTER the host leg has already flipped the FPU, which is the ordering that
    # made the leak reachable.
    monkeypatch.setenv("CUPY_CACHE_DIR", f"/tmp/cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_x")
    with pytest.raises(sp.SubnormalPolicyUnattainable):
        sp.install_subnormal_policy(sp.FLUSH, cupy=types.SimpleNamespace(), strict=True)
    assert sp.policy_is_installed() is False
    assert state["calls"] == [True, False], "the FPU was never driven back"
    assert state["flushing"] is False, "the FPU was left in the refused policy's mode"
    assert sp._STATE["executors"]["host"]["fpu_restored_to"] is False


def test_a_refused_install_puts_cupys_reduction_accelerators_back(monkeypatch, tmp_path):
    """Same rule as the FPU: no process-wide residue under an uninstalled policy.

    Switching CUB off changes no bytes under either policy — the NVRTC route obeys
    whichever one is in force — but an unexplained performance residue left by a
    REFUSED install is a defect whatever it costs.
    """
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda module: False)
    with pytest.raises(sp.SubnormalPolicyUnattainable):
        sp.install_cupy_policy(sp.FLUSH, cupy, strict=True)
    assert state["reduction_accelerators"] == [], "the switch-off is what is under test"
    sp._reset_for_tests()
    assert state["reduction_accelerators"] == [_FAKE_CUB], (
        "the rollback must hand CuPy back what its GETTER returned. Measured on "
        "device: handing back the report's stringified copy raised "
        "ValueError: Unknown accelerator: 1 from inside the refusal handling, "
        "which replaced the refusal with a traceback and left CUB switched off")


def test_a_rollback_that_cannot_restore_still_lets_the_refusal_through(
        monkeypatch, tmp_path):
    """Cleanup runs inside the handling of a refusal, so it must never raise.

    Measured on device: the rollback raised while restoring CuPy's accelerators
    and the process died with that traceback instead of the refusal — the one
    message the caller needed, naming CUPY_ACCELERATORS, was buried under
    "During handling of the above exception". The residue is reported as a
    warning; the refusal is what reaches the caller.
    """
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    with pytest.warns(RuntimeWarning, match="CUPY_ACCELERATORS"):
        sp.install_cupy_policy(sp.FLUSH, cupy, strict=False)

    def hostile(names):
        raise ValueError(f"Unknown accelerator: {names[0] if names else ''}")

    # The rollback holds the setter it captured at install time, so the hostile
    # one has to be put where the rollback will actually reach it.
    _setter, before = sp._STATE["_cupy_accelerators"]
    sp._STATE["_cupy_accelerators"] = (hostile, before)
    with pytest.warns(RuntimeWarning, match="reduction accelerators back"):
        assert sp._restore_cupy_accelerators() is False
    assert "Unknown accelerator" in sp._STATE["_cupy_accelerator_restore_failed"]


def test_keep_is_attainable_on_this_host():
    report = sp.install_host_policy(sp.KEEP, strict=True)
    assert report["attained"] is True
    assert report["flushing_after"] is False
    assert backends.subnormals_flushed() is False


def test_a_non_strict_refusal_warns_and_records_rather_than_raising(monkeypatch):
    """``strict=False`` is the "loudly declare" half of refuse-or-declare."""
    _a_host_with_no_working_lever(monkeypatch)
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        stamp = sp.install_subnormal_policy(sp.FLUSH, strict=False)
    assert stamp["executors"]["host"]["attained"] is False
    assert "host" in stamp["unattained"]


def test_an_unattained_host_stops_the_device_legs_even_when_not_strict(monkeypatch):
    """``strict=False`` must not BUILD the half-applied policy.

    An earlier revision warned that the host could not flush and then installed
    Triton's flush hooks anyway, producing a process where the host keeps and
    Triton flushes — measured to diverge at step 1 with 92 differing floats of
    2880 across 22 arrays. Declaring the failure is the point of ``strict=False``;
    proceeding past it is not.
    """
    _a_host_with_no_working_lever(monkeypatch)
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        stamp = sp.install_subnormal_policy(sp.FLUSH, strict=False)
    for executor in ("cupy", "triton"):
        assert stamp["executors"][executor]["installed"] is False
        assert stamp["executors"][executor]["mechanism"] == "not attempted"
        assert stamp["executors"][executor]["attained"] is False
    assert stamp["unattained"] == ["cupy", "host", "triton"]
    assert sp.policy_is_installed() is False


def test_host_policy_reasons_names_meep_when_meep_is_unreachable():
    reasons = sp.host_policy_reasons(sp.FLUSH, None, before=False, after=False)
    assert len(reasons) == 1
    assert "set_zero_subnormals" in reasons[0]


def test_host_policy_reasons_is_empty_when_the_measurement_agrees():
    assert sp.host_policy_reasons(sp.KEEP, object(), before=False, after=False) == []
    assert sp.host_policy_reasons(sp.FLUSH, object(), before=True, after=True) == []


# ---------------------------------------------------------------------------
# The lock: one policy per process
# ---------------------------------------------------------------------------


def test_installing_the_same_policy_twice_is_idempotent():
    first = sp.install_subnormal_policy(sp.KEEP)
    second = sp.install_subnormal_policy(sp.KEEP)
    assert first["requested"] == second["requested"] == "keep"


def test_a_second_different_policy_is_refused_not_swapped():
    """Binaries and cache entries already exist under the first one."""
    sp.install_subnormal_policy(sp.KEEP)
    with pytest.raises(sp.SubnormalPolicyLocked) as excinfo:
        sp.install_subnormal_policy(sp.FLUSH)
    assert "already installed" in str(excinfo.value)
    assert sp.get_subnormal_policy() == "keep"


def test_a_refusal_that_installed_no_device_seam_leaves_the_process_unlocked(monkeypatch):
    """Otherwise the refused policy would lock out the one the caller can have."""
    _a_host_with_no_working_lever(monkeypatch)
    with pytest.raises(sp.SubnormalPolicyUnattainable):
        sp.install_subnormal_policy(sp.FLUSH, strict=True)
    assert sp._STATE["policy"] is None
    sp.install_subnormal_policy(sp.KEEP)  # must not raise SubnormalPolicyLocked
    assert sp.get_subnormal_policy() == "keep"


# ---------------------------------------------------------------------------
# The optional-dependency contract
# ---------------------------------------------------------------------------


def test_the_module_imports_nothing_optional_at_module_level():
    """cupy, triton and meep are all optional; none may be a module-level import."""
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    forbidden = {"cupy", "triton", "meep"}
    offenders = []
    for node in tree.body:
        roots = set()
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots = {node.module.partition(".")[0]}
        if roots & forbidden:
            offenders.append(f"line {node.lineno}: {sorted(roots & forbidden)}")
    assert not offenders, "module-level optional imports:\n" + "\n".join(offenders)


def test_the_module_is_importable_with_every_optional_dependency_blocked(monkeypatch):
    for name in ("cupy", "triton", "meep"):
        monkeypatch.setitem(sys.modules, name, None)
    # Executed under its real dotted name so ``from . import backends`` resolves,
    # but never registered in ``sys.modules`` — nothing this suite holds is rebound.
    spec = importlib.util.spec_from_file_location("meep_gpu.subnormal_policy",
                                                  Path(sp.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module is not sp
    assert module.resolve_policy("keep") == ("keep", "argument")
    assert module.audit_ptx(_PTX_KEEP, module.KEEP,
                            reflect_ftz_governed=False)["violations"] == []


def test_absent_cupy_and_triton_report_attained_without_installing(monkeypatch):
    """A missing optional dependency is not a policy failure — it has no executor."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    cupy_report = sp.install_cupy_policy(sp.KEEP)
    triton_report = sp.install_triton_policy(sp.FLUSH)
    for report in (cupy_report, triton_report):
        assert report["attained"] is True
        assert report["installed"] is False
        assert "not installed" in report["mechanism"]


def test_no_subnormal_float_literal_in_the_new_module():
    """The same rule ``test_backends`` applies to the package, restated here.

    A subnormal spelled as a literal becomes ``0.0`` in ``co_consts`` when the
    module is compiled in a flushed process — and this module's whole job is to
    be correct in exactly that process.
    """
    smallest_normal = sys.float_info.min
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
        and 0.0 < abs(node.value) < smallest_normal
    ]
    assert not offenders, "subnormal float literals:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# The CuPy leg: option tuples and cache directories, both pure
# ---------------------------------------------------------------------------


def test_the_strip_removes_both_spellings_and_never_adds_ftz_false():
    """NVRTC hard-errors on a duplicate ``-ftz`` option, measured in both orders."""
    options = ("-std=c++17", "-ftz=true", "--fmad=false", "--ftz=true")
    stripped = sp._strip_ftz(options)
    assert stripped == ("-std=c++17", "--fmad=false")
    assert not any("ftz" in option for option in stripped)


def test_the_strip_leaves_an_option_tuple_without_ftz_alone():
    options = ("--fmad=false",)
    assert sp._strip_ftz(options) == options


def test_keep_demands_a_policy_suffixed_cupy_cache_dir():
    """CuPy's cache key is computed ABOVE the seam, so the directory is the key."""
    assert sp.cupy_cache_reasons(sp.KEEP, "") != []
    assert sp.cupy_cache_reasons(sp.KEEP, None) != []
    assert sp.cupy_cache_reasons(sp.KEEP, "/tmp/plain_cache") != []
    assert sp.cupy_cache_reasons(sp.KEEP, "/tmp/cupy_cache_ftz_stripped_2417") == []


def test_flush_refuses_a_cache_dir_cut_under_keep():
    """The reverse direction: a stale keep cache would serve stripped binaries."""
    reasons = sp.cupy_cache_reasons(sp.FLUSH, "/tmp/cupy_cache_ftz_stripped_2417")
    assert len(reasons) == 1 and sp.CUPY_CACHE_POLICY_TOKEN in reasons[0]
    assert sp.cupy_cache_reasons(sp.FLUSH, "") == []
    assert sp.cupy_cache_reasons(sp.FLUSH, "/tmp/plain_cache") == []


#: CuPy's accelerator names and the integer values its getter hands back.
_FAKE_ACCELERATOR_VALUES = {"cub": 1, "cutensor": 2}
_FAKE_CUB = _FAKE_ACCELERATOR_VALUES["cub"]


def _fake_cupy(monkeypatch, tmp_path, accelerators=(_FAKE_CUB,)):
    """A CuPy stand-in exposing every seam the policy touches.

    ``accelerators`` is what CuPy IMPORTED with, which is the only thing that
    decides where a reduction is dispatched — see
    :func:`test_flush_refuses_because_the_runtime_switch_off_does_not_move_the_dispatch`.
    Pass ``()`` for a CuPy that came up under ``CUPY_ACCELERATORS=''``.

    Three of them, all load-bearing: the compiler front end (the option strip),
    ``_core._accelerator`` (the CUB reduction path, which no compile option can
    reach) and ``_util.clear_memo`` (the in-process kernel cache, which is above
    the disk cache and defeats a late install). Scoped through ``monkeypatch`` and
    never written into ``sys.modules``, so ``conftest``'s impostor guard is
    untouched.
    """
    calls = []
    state = {"reduction_accelerators": list(accelerators), "memo_cleared": 0}

    def compile_using_nvrtc(source, options=(), *args, **kwargs):
        calls.append(tuple(options))
        return "cubin"

    def set_reduction_accelerators(names):
        # CuPy's asymmetry, measured on the GPU host (CuPy 13.5.1):
        # ``get_reduction_accelerators()`` returns plain ints (``[1]``) and the
        # setter accepts an int or a NAME ('cub') — but not the int's ``str``.
        # A fake that round-tripped strings would have let the rollback bug
        # through, which is exactly what happened.
        resolved = []
        for name in names:
            if isinstance(name, int) and not isinstance(name, bool):
                resolved.append(name)
            elif name in _FAKE_ACCELERATOR_VALUES:
                resolved.append(_FAKE_ACCELERATOR_VALUES[name])
            else:
                raise ValueError(f"Unknown accelerator: {name}")
        state["reduction_accelerators"] = resolved

    compiler = types.SimpleNamespace(compile_using_nvrtc=compile_using_nvrtc)
    cuda = types.SimpleNamespace(compiler=compiler)
    accelerator = types.SimpleNamespace(
        get_reduction_accelerators=lambda: list(state["reduction_accelerators"]),
        set_reduction_accelerators=set_reduction_accelerators)
    core = types.SimpleNamespace(_accelerator=accelerator)
    util = types.SimpleNamespace(
        clear_memo=lambda: state.__setitem__("memo_cleared", state["memo_cleared"] + 1))
    cupy = types.SimpleNamespace(cuda=cuda, _core=core, _util=util)
    return cupy, compiler, calls, state


def test_keep_wraps_the_compiler_seam_and_counts_what_it_stripped(monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, calls, _state = _fake_cupy(monkeypatch, tmp_path)

    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["installed"] is True
    assert report["wrapped_entry_points"] == ["compile_using_nvrtc"]

    compiler.compile_using_nvrtc("src", ("--fmad=false", "-ftz=true"))
    compiler.compile_using_nvrtc("src", ("--fmad=false",))
    assert calls == [("--fmad=false",), ("--fmad=false",)]
    assert sp.cupy_strip_counters() == {"calls": 2, "removed": 1}


def test_flush_leaves_cupy_compiles_native_with_no_option_override(monkeypatch, tmp_path):
    """No policy wrapper at all under flush — CuPy's own ``-ftz=true`` survives.

    The seam is still wrapped by ``backends.guard_kernel_compilation``, which is a
    different concern (the HOST fenv during a compile) and carries a different
    marker; what must not appear is a wrapper that touches the OPTIONS.
    """
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, compiler, calls, _state = _fake_cupy(monkeypatch, tmp_path, accelerators=())

    report = sp.install_cupy_policy(sp.FLUSH, cupy)
    assert report["attained"] is True and report["installed"] is False
    assert getattr(compiler.compile_using_nvrtc, sp._POLICY_MARK, None) is None
    compiler.compile_using_nvrtc("src", ("--fmad=false", "-ftz=true"))
    assert calls == [("--fmad=false", "-ftz=true")]


def test_flush_refuses_because_the_runtime_switch_off_does_not_move_the_dispatch(
        monkeypatch, tmp_path):
    """The one CuPy path no compile option reaches, and no in-process call either.

    CuPy 13.5.1's default reduction accelerator is a CUB binary built when CuPy
    was built. ``set_reduction_accelerators([])`` returns cleanly and the getter
    then reports ``[]`` — and the dispatch does not move. Measured on device
    (the GPU host, RTX A6000, the conformance gate's own operand) with the whole
    process flushing AND the list reported empty: ``cp.sum``, ``cp.max``,
    ``cp.add.reduce`` and ``ndarray.sum`` over one ``2^-135`` among 4095 zeros all
    returned ``0x00004000`` with ZERO NVRTC compiles, while the same buffer sliced
    ``[::2]`` — a shape CUB declines — compiled a kernel and returned
    ``0x00000000``. So ``before`` is the verdict and ``after`` is decoration, and
    a CuPy that imported with CUB cannot reach ``flush`` at all.

    The gate caught this in the field before this test existed: ``gate_flush.json``
    from the 2026-08-13 device run failed exactly two cells,
    ``cupy_reduction/reduce_sum`` and ``reduce_max``, while the install reported
    ``attained=True``.
    """
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert "CUPY_ACCELERATORS" in str(excinfo.value)
    assert "zero NVRTC compiles" in str(excinfo.value)
    report = sp.executor_report("cupy")
    assert report["reduction_accelerators"]["before"] == [str(_FAKE_CUB)]
    assert report["reduction_accelerators"]["runtime_setter_changes_dispatch"] is False


def test_flush_attains_when_cupy_imported_with_no_reduction_accelerator(
        monkeypatch, tmp_path):
    """``CUPY_ACCELERATORS=''`` before the import is the only route, and it works.

    Measured in the same device session, in a second process with the variable
    set: every reduction compiled through NVRTC — options carrying CuPy's own
    ``-ftz=true`` — and every one returned ``0x00000000``.
    """
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path, accelerators=())
    report = sp.install_cupy_policy(sp.FLUSH, cupy)
    assert report["attained"] is True
    assert report["reduction_accelerators"]["before"] == []
    assert state["reduction_accelerators"] == []


def test_keep_leaves_cub_alone_because_cub_already_keeps(monkeypatch, tmp_path):
    """Measured: CUB kept the subnormal even with the rest of the process flushing.

    So under ``"keep"`` it already agrees, and switching it off would trade a
    matching path for a slower one.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path)
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert state["reduction_accelerators"] == [_FAKE_CUB]
    assert report["reduction_accelerators"]["after"] == [str(_FAKE_CUB)]


def test_flush_refuses_when_the_cub_path_cannot_be_reached(monkeypatch, tmp_path):
    """A reduction path outside the policy is a refusal, not a footnote."""
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    del cupy._core._accelerator.set_reduction_accelerators
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert "CUPY_ACCELERATORS" in str(excinfo.value)


def test_installing_the_policy_drops_cupys_in_process_kernel_memo(monkeypatch, tmp_path):
    """A kernel compiled BEFORE the install would otherwise keep its old policy.

    Measured: ``cp.multiply`` compiled before a ``"keep"`` install still flushed
    afterwards while the install reported ``attained=True``, because CuPy memoizes
    the compiled module in process — above the disk cache and above the seam the
    strip installs at. The first kernel compiled after the install obeyed the
    policy, which is what identified the memo as the culprit.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path)
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert state["memo_cleared"] == 1
    assert report["kernel_memo"]["cleared"] is True


def test_flush_refuses_when_the_compile_time_fenv_guard_could_not_be_installed(
        monkeypatch, tmp_path):
    """``guard_kernel_compilation`` matters MORE under flush, not less.

    Under flush the HOST is deliberately flushing, which is exactly the state in
    which NVRTC cannot parse ``__FLT_DENORM_MIN__`` in CCCL's ``<cuda/std/limits>``
    — so a guard that could not be installed converts a startup refusal into a
    mid-run compile failure inside a header. Reporting ``attained=True`` there,
    which an earlier revision did unconditionally, is the worst of both.
    """
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, accelerators=())
    seen = []
    monkeypatch.setattr(backends, "guard_kernel_compilation",
                        lambda module: seen.append(module) or True)
    report = sp.install_cupy_policy(sp.FLUSH, cupy)
    assert seen == [cupy]
    assert report["compile_guard_installed"] is True and report["attained"] is True

    sp._reset_for_tests()
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda module: False)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert "FE_DFL_ENV" in str(excinfo.value)


def test_uninstalling_removes_the_strip_and_leaves_the_backends_guard_alone(
        monkeypatch, tmp_path):
    """Restore what this module installed, and nothing that it did not.

    The seam ends up two layers deep — ``backends``' fenv guard underneath, the
    policy strip on top — and only the top one belongs to this module. After an
    uninstall, ``-ftz=true`` must reach the compiler again (the strip is gone) and
    the guard's marker must still be there (the guard is not this module's to
    remove).
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, calls, _state = _fake_cupy(monkeypatch, tmp_path)
    sp.install_cupy_policy(sp.KEEP, cupy)
    assert getattr(compiler.compile_using_nvrtc, sp._POLICY_MARK, None) == sp.KEEP

    sp.uninstall_subnormal_policy()
    assert getattr(compiler.compile_using_nvrtc, sp._POLICY_MARK, None) is None
    assert getattr(compiler.compile_using_nvrtc, backends._GUARD_MARK, False) is True
    compiler.compile_using_nvrtc("src", ("-ftz=true",))
    assert calls == [("-ftz=true",)]


def test_a_second_different_policy_at_the_cupy_seam_is_refused(monkeypatch, tmp_path):
    """The strip installs under keep only, so a later flush would report "native"
    while every compile is still being stripped."""
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    sp.install_cupy_policy(sp.KEEP, cupy)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    with pytest.raises(sp.SubnormalPolicyLocked) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert "compile_using_nvrtc" in str(excinfo.value)


def test_the_strip_composes_with_the_backends_guard_without_stacking(monkeypatch, tmp_path):
    """The composition rule, in both orders — this is the reason for functools.wraps.

    ``backends.guard_kernel_compilation`` decides "already guarded?" by looking
    for its own marker on whatever sits at the seam. ``functools.wraps`` copies
    ``__dict__``, so each wrapper inherits the other's marker and neither stacks
    a second layer — which matters because ``resolve_backend`` calls the guard on
    EVERY GPU resolution.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))

    def depth(function):
        n = 0
        while getattr(function, "__wrapped__", None) is not None:
            n += 1
            function = function.__wrapped__
        return n

    # guard first, then the policy strip.
    cupy, compiler, calls, _state = _fake_cupy(monkeypatch, tmp_path)
    assert backends.guard_kernel_compilation(cupy) in (True, False)
    sp.install_cupy_policy(sp.KEEP, cupy)
    layered = depth(compiler.compile_using_nvrtc)
    backends.guard_kernel_compilation(cupy)
    sp.install_cupy_policy(sp.KEEP, cupy)
    assert depth(compiler.compile_using_nvrtc) == layered
    compiler.compile_using_nvrtc("src", ("-ftz=true",))
    assert calls == [()]

    # policy strip first, then the guard.
    sp._reset_for_tests()
    cupy2, compiler2, calls2, _state2 = _fake_cupy(monkeypatch, tmp_path)
    sp.install_cupy_policy(sp.KEEP, cupy2)
    layered2 = depth(compiler2.compile_using_nvrtc)
    backends.guard_kernel_compilation(cupy2)
    backends.guard_kernel_compilation(cupy2)
    assert depth(compiler2.compile_using_nvrtc) == layered2
    compiler2.compile_using_nvrtc("src", ("-ftz=true",))
    assert calls2 == [()]


def test_a_keep_install_without_a_policy_cache_dir_refuses_before_touching_cupy(
        monkeypatch, tmp_path):
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    original = compiler.compile_using_nvrtc
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    assert sp.CUPY_CACHE_POLICY_TOKEN in str(excinfo.value)
    assert compiler.compile_using_nvrtc is original


def test_the_cache_refusal_obeys_strict_like_every_other_refusal(monkeypatch, tmp_path):
    """``strict`` decides raise-vs-warn for ALL of them, or it decides nothing.

    The documented contract is "raise, or emit a RuntimeWarning and record
    attained=False". An earlier revision raised on this one path regardless of
    ``strict``, which made every caller's ``strict=False`` a coin flip depending
    on which refusal fired first.
    """
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    original = compiler.compile_using_nvrtc
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        report = sp.install_cupy_policy(sp.KEEP, cupy, strict=False)
    assert report["attained"] is False and report["installed"] is False
    assert sp.CUPY_CACHE_POLICY_TOKEN in report["reasons"][0]
    assert compiler.compile_using_nvrtc is original
    assert sp.policy_stamp()["unattained"] == ["cupy"]


# ---------------------------------------------------------------------------
# The Triton leg: LLVM IR text and PTX text, both pure
# ---------------------------------------------------------------------------

_LLIR = '''; ModuleID = 'LLVMDialectModule'
target triple = "nvptx64-nvidia-cuda"

define void @pml_curl_step(ptr addrspace(1) %0, i32 %1) local_unnamed_addr #0 {
  %3 = fmul float 1.0, 2.0
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { mustprogress nofree "no-trapping-math"="true" }
attributes #1 = { nocallback nofree nosync nounwind }
'''

#: THE SHAPE TRITON 3.1.0 ACTUALLY EMITS: no attribute group on the define, and a
#: ``!dbg`` metadata attachment. Captured from a real ``make_llir`` on the GPU host.
#: Every fixture in an earlier revision lacked the metadata attachment, all 59
#: tests passed, and the injector produced IR that aborted LLVM on first contact.
_LLIR_TRITON_SHAPE = sp.TRITON_LLIR_SHAPE


def _llvm_rejects(text):
    """LLVM's parse error for this IR, or None. Needs a real LLVM parser.

    Declared as a resource rather than skipped quietly: ``llvmlite`` is optional
    for the package, so a host without it loses the definitive half of this
    verification and that loss must be visible in the run summary.
    """
    if sp._llvm_parser() is None:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("llvmlite", "LLVM's own parser is what makes the "
                                           "IR-shape checks definitive")
    return sp.parse_llvm_error(text)


def test_the_injected_attribute_lands_where_llvm_accepts_it_on_real_triton_ir():
    """The placement, on the shape Triton emits, checked by LLVM's own parser.

    Function attributes come BEFORE metadata attachments in LLVM's grammar.
    Putting ``#N`` before the body brace instead — i.e. after ``!dbg !7`` — is
    rejected with ``expected '{' in function body``, and the rejection is
    ``report_fatal_error``: an abort, exit 134, from inside Triton's compiler,
    which no ``except`` clause and no ``strict=False`` can soften.
    """
    text, counts = sp.inject_denormal_attribute(_LLIR_TRITON_SHAPE)
    define = [line for line in text.splitlines() if line.startswith("define ")][0]
    assert counts["groups_minted"] == 1
    assert define.index("#0") < define.index("!dbg"), define
    assert "local_unnamed_addr #0 !dbg !7 {" in define

    # And the grammar slot itself, settled by LLVM rather than by reading it:
    parseable = "define void @k(float %0) local_unnamed_addr !foo !0 {\n  ret void\n}\n\n!0 = !{}\n"
    assert _llvm_rejects(parseable) is None                       # the baseline parses
    assert _llvm_rejects(sp.inject_denormal_attribute(parseable)[0]) is None


def test_the_injector_and_the_reflect_rewrite_on_recorded_real_triton_ir():
    """The strongest evidence available without a GPU: real ``make_llir`` output.

    Four LLVM IR modules recorded from real Triton 3.1.0 compiles of this engine's
    own curl and PML kernels. For each: LLVM parses it, LLVM parses it after the
    injection, LLVM parses it after the keep-policy reflect rewrite, and LLVM
    REJECTS the previous placement — measured on all four, ``expected '{' in
    function body``. All four also carry the ``nvvm-reflect-ftz`` flag, which is
    what makes the keep rewrite reach a real kernel rather than only a fixture.
    """
    recorded = sorted((API_DIR / "parity" / "meep_gpu" / "results"
                       / "triton_feasibility_2026-08-09" / "triton_ptx").glob("*.llir"))
    if not recorded:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("triton_feasibility_llir",
                               "recorded real Triton IR lives under gitignored results/")
    for path in recorded:
        text = path.read_text(encoding="utf-8")
        define = [line for line in text.splitlines() if line.startswith("define ")][0]
        assert "!dbg" in define, path.name      # the shape the fixtures used to miss
        assert "#" not in define.split("(")[0], path.name

        injected, counts = sp.inject_denormal_attribute(text)
        assert counts == {"defines": 1, "groups_extended": 0, "groups_minted": 1}, path.name
        assert _llvm_rejects(text) is None, path.name
        assert _llvm_rejects(injected) is None, path.name

        kept, rewrites = sp.set_reflect_ftz(text, 0)
        assert rewrites == 1, path.name
        assert _llvm_rejects(kept) is None, path.name

        brace = define.rfind("{")
        misplaced = (text.replace(define, f"{define[:brace].rstrip()} #9 {define[brace:]}")
                     + f"\nattributes #9 = {{ {sp.TRITON_DENORMAL_ATTRIBUTE} }}\n")
        error = _llvm_rejects(misplaced)
        assert error is not None and "expected '{' in function body" in error, path.name


def test_the_old_placement_is_still_rejected_by_llvm(monkeypatch):
    """The negative control: a test that cannot fail proves nothing.

    This is the exact text the previous injector produced — the attribute group
    after the metadata attachment — and LLVM must refuse it. If this ever parses,
    the check above has stopped discriminating.
    """
    misplaced = ("define void @k(float %0) local_unnamed_addr !foo !0 #0 {\n"
                 "  ret void\n}\n\n!0 = !{}\n"
                 f"attributes #0 = {{ {sp.TRITON_DENORMAL_ATTRIBUTE} }}\n")
    error = _llvm_rejects(misplaced)
    assert error is not None
    assert "expected '{' in function body" in error


def test_the_install_time_verification_is_what_refuses_a_broken_injector(monkeypatch):
    """"The hook is installed" and "the policy is attained" are different claims.

    An earlier revision reported ``attained=True`` for a mechanism that aborted
    the process on its first compile, because its verdict was structural — was
    ``add_stages`` wrapped — and nothing ever ran the rewrite. The verdict is now
    a measurement over Triton-shaped IR, taken BEFORE the hook is armed.
    """
    good = sp.verify_injection()
    assert good["ok"] is True
    assert good["checks"][0]["ok"] is True

    # Keep is measured too, because keep now EDITS the IR as well.
    assert sp.verify_reflect_rewrite()["ok"] is True
    assert sp.verify_reflect_rewrite("define void @k() {\n ret void\n}\n")["ok"] is False

    monkeypatch.setattr(sp, "inject_denormal_attribute",
                        lambda text: (text.replace("!dbg !7 {", "!dbg !7 #9 {"),
                                      {"defines": 1, "groups_extended": 0, "groups_minted": 1}))
    broken = sp.verify_injection()
    assert broken["ok"] is False
    assert any("misplaced" in reason for reason in broken["reasons"])

    backend = _install_fake_triton(monkeypatch, sp.FLUSH)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
            sp.install_triton_policy(sp.FLUSH, strict=True)
        assert "misplaced" in str(excinfo.value)
        assert backend.add_stages is original_stages, "the hook was armed anyway"
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_the_denormal_attribute_extends_the_existing_group_without_replacing_it():
    text, counts = sp.inject_denormal_attribute(_LLIR)
    assert counts == {"defines": 1, "groups_extended": 1, "groups_minted": 0}
    line = [entry for entry in text.splitlines() if entry.startswith("attributes #0")][0]
    assert 'denormal-fp-math-f32"="preserve-sign,preserve-sign"' in line
    assert '"no-trapping-math"="true"' in line   # the composition, not a replacement
    assert "mustprogress nofree" in line
    # The group the DEFINE does not reference is left alone.
    other = [entry for entry in text.splitlines() if entry.startswith("attributes #1")][0]
    assert "denormal-fp-math-f32" not in other


def test_the_denormal_attribute_is_minted_for_a_define_with_no_group():
    source = "define void @k(ptr %0) {\n  ret void\n}\n"
    text, counts = sp.inject_denormal_attribute(source)
    assert counts == {"defines": 1, "groups_extended": 0, "groups_minted": 1}
    assert "define void @k(ptr %0) #0 {" in text
    assert f"attributes #0 = {{ {sp.TRITON_DENORMAL_ATTRIBUTE} }}" in text


def test_the_injection_is_idempotent():
    once, _ = sp.inject_denormal_attribute(_LLIR)
    twice, counts = sp.inject_denormal_attribute(once)
    assert twice == once
    assert counts["groups_extended"] == 0 and counts["groups_minted"] == 0


def test_every_define_gets_the_attribute_when_there_are_several():
    source = (
        "define void @a(ptr %0) #0 {\n ret void\n}\n"
        "define void @b(ptr %0) #2 {\n ret void\n}\n"
        "define void @c(ptr %0) {\n ret void\n}\n"
        "attributes #0 = { nounwind }\n"
        "attributes #2 = { nofree }\n"
    )
    text, counts = sp.inject_denormal_attribute(source)
    assert counts["defines"] == 3
    assert counts["groups_extended"] == 2
    assert counts["groups_minted"] == 1
    assert "define void @c(ptr %0) #3 {" in text
    for group in ("#0", "#2", "#3"):
        line = [entry for entry in text.splitlines() if entry.startswith(f"attributes {group} ")][0]
        assert "denormal-fp-math-f32" in line, group


def test_an_attribute_group_with_a_trailing_comment_is_extended_not_duplicated():
    """Two definitions of one group is invalid IR, and the regex decided it.

    The old pattern anchored on ``}`` at END of line, so a definition carrying
    anything after the brace was not recognized as a definition at all — and the
    define's reference to it then fell through to the "mint one" branch, appending
    a SECOND ``attributes #0 = ...``. Latent (LLVM's printer emits no trailing
    comments) but the same class as the placement bug: IR treated as line-shaped
    text, with the shape assumptions checked only against hand-written fixtures.
    """
    source = ("define void @k(ptr %0) #0 {\n ret void\n}\n"
              "attributes #0 = { nounwind } ; printed with a comment\n")
    text, counts = sp.inject_denormal_attribute(source)
    definitions = [line for line in text.splitlines() if line.startswith("attributes #0")]
    assert len(definitions) == 1, definitions
    assert counts == {"defines": 1, "groups_extended": 1, "groups_minted": 0}
    assert "denormal-fp-math-f32" in definitions[0]
    assert definitions[0].endswith("; printed with a comment")


def test_a_hash_inside_a_quoted_name_is_not_an_attribute_group_reference():
    """``#`` is legal inside a quoted symbol or section name; ``#0`` there is not
    a group, and treating it as one would leave that function unattributed."""
    source = 'define void @k(ptr %0) section "a#b" {\n ret void\n}\n'
    text, counts = sp.inject_denormal_attribute(source)
    assert counts["groups_minted"] == 1
    assert 'section "a#b" #0 {' in text


def test_the_reflect_flag_is_what_makes_keep_actually_ieee():
    """``"keep"`` was not IEEE-754 for ``tl.math.div_rn``, and the fix is one flag.

    Triton calls ``set_nvvm_reflect_ftz`` unconditionally, so libdevice resolves
    to its FTZ variants: measured, ``tl.math.div_rn`` emits ``div.rn.ftz.f32``
    natively, and ``nonlinear_constitutive_step`` carries 24 of them. Rewriting
    the module flag to 0 — the same llir-stage string the flush policy already
    edits — gives ``div.rn.f32`` on the same kernel through the same seam.
    """
    assert 'i32 4, !"nvvm-reflect-ftz", i32 1' in _LLIR_TRITON_SHAPE
    kept, rewrites = sp.set_reflect_ftz(_LLIR_TRITON_SHAPE, 0)
    assert rewrites == 1
    assert 'i32 4, !"nvvm-reflect-ftz", i32 0' in kept
    assert sp.set_reflect_ftz(kept, 1)[0] == _LLIR_TRITON_SHAPE
    # A module without the flag reports zero rewrites rather than pretending.
    assert sp.set_reflect_ftz("define void @k() {\n ret void\n}\n", 0)[1] == 0
    assert _llvm_rejects(kept.replace("!dbg !7 ", "")) is None


_PTX_KEEP = """
//
.visible .entry k(
	.param .u64 k_param_0
)
{
	.reg .f32 	%f<8>;
	ld.global.f32 	%f1, [%rd1];
	mul.rn.f32 	%f3, %f1, %f2;
	add.f32 	%f4, %f3, %f1;
	fma.rn.f32 	%f5, %f1, %f2, %f3;
	neg.f32 	%f6, %f5;
	div.rn.ftz.f32 	%f7, %f6, %f1;   // the unconditional nvvm-reflect baseline
	cvt.rn.f32.f64 	%f8, %fd1;
	st.global.f32 	[%rd2], %f7;
	ret;
}
"""

_PTX_FLUSH = _PTX_KEEP.replace("mul.rn.f32", "mul.rn.ftz.f32") \
                      .replace("add.f32", "add.ftz.f32") \
                      .replace("fma.rn.f32", "fma.rn.ftz.f32") \
                      .replace("neg.f32", "neg.ftz.f32")


def test_the_ptx_audit_counts_only_f32_arithmetic():
    """``ld``/``st``/``cvt`` are transport or conversion, not governed arithmetic."""
    audit = sp.audit_ptx(_PTX_KEEP, sp.KEEP)
    assert audit["audited"] == 5           # mul, add, fma, neg, div
    assert set(audit["by_opcode"]) == {
        "mul.rn.f32", "add.f32", "fma.rn.f32", "neg.f32", "div.rn.ftz.f32"}
    assert "ld.global.f32" not in audit["by_opcode"]
    assert "st.global.f32" not in audit["by_opcode"]
    assert "cvt.rn.f32.f64" not in audit["by_opcode"]


def test_keep_now_audits_the_libdevice_family_because_it_governs_it():
    """The exemption was a choice, and the choice has changed.

    While ``nvvm-reflect-ftz`` was out of reach, ``div.rn.ftz.f32`` under keep was
    a fact to record rather than a violation to refuse. Keep now rewrites that
    flag, so the same instruction IS a violation — and the exemption survives only
    for IR that carried no such flag to rewrite, where refusing would refuse a
    kernel over a knob that was not there.
    """
    governed = sp.audit_ptx(_PTX_KEEP, sp.KEEP)
    assert [line.split()[0] for _n, line in governed["violations"]] == ["div.rn.ftz.f32"]

    ungoverned = sp.audit_ptx(_PTX_KEEP, sp.KEEP, reflect_ftz_governed=False)
    assert ungoverned["violations"] == []
    assert ungoverned["with_ftz"] == 1


def test_the_audit_reads_predicated_instructions(monkeypatch):
    """``@%p1 mul.rn.f32`` is an INSTRUCTION, not a directive.

    Skipping every line starting with ``@`` made the auditor blind to exactly the
    class it exists to catch: an f32 op that keeps subnormals among flushing
    neighbours. Not reachable in today's PTX — the 26-kernel census found no
    predicated f32 arithmetic — but the auditor's contract is "every generated
    PTX", and it was not true.
    """
    for line in ("\t@%p1 mul.rn.f32 \t%f3, %f1, %f2;",
                 "\t@!%p2 fma.rn.f32 \t%f4, %f1, %f2, %f3;"):
        audit = sp.audit_ptx(line, sp.FLUSH)
        assert audit["audited"] == 1, line
        assert len(audit["violations"]) == 1, line
        assert sp.audit_ptx(line.replace(".f32", ".ftz.f32"), sp.FLUSH)["violations"] == []
    # A directive still is one.
    assert sp.audit_ptx("\t.reg .f32 %f<8>;\n", sp.FLUSH)["audited"] == 0


def test_the_audit_reads_the_packed_f32x2_opcodes():
    """``add.rn.f32x2`` is f32 arithmetic; matching the type token by equality
    against ``f32`` dropped it silently."""
    audit = sp.audit_ptx("\tadd.rn.f32x2 \t%fd3, %fd1, %fd2;\n", sp.FLUSH)
    assert audit["audited"] == 1 and len(audit["violations"]) == 1
    assert sp.audit_ptx("\tadd.rn.ftz.f32x2 \t%fd3, %fd1, %fd2;\n",
                        sp.FLUSH)["violations"] == []


def test_flush_refuses_ptx_that_kept():
    audit = sp.audit_ptx(_PTX_KEEP, sp.FLUSH)
    assert audit["missing_ftz"] == 4
    assert {line.split()[0] for _n, line in audit["violations"]} == {
        "mul.rn.f32", "add.f32", "fma.rn.f32", "neg.f32"}


def test_flush_accepts_ptx_where_every_audited_instruction_flushed():
    audit = sp.audit_ptx(_PTX_FLUSH, sp.FLUSH)
    assert audit["violations"] == []
    assert audit["audited"] == 5 and audit["with_ftz"] == 5


def test_keep_refuses_ptx_where_a_governed_instruction_gained_ftz():
    audit = sp.audit_ptx(_PTX_FLUSH, sp.KEEP)
    assert {line.split()[0] for _n, line in audit["violations"]} == {
        "mul.rn.ftz.f32", "add.ftz.f32", "fma.rn.ftz.f32", "neg.ftz.f32",
        "div.rn.ftz.f32"}
    assert {line.split()[0] for _n, line in
            sp.audit_ptx(_PTX_FLUSH, sp.KEEP, reflect_ftz_governed=False)["violations"]} == {
        "mul.rn.ftz.f32", "add.ftz.f32", "fma.rn.ftz.f32", "neg.ftz.f32"}


def test_the_known_gap_is_what_the_audit_exists_to_catch():
    """A plain ``/`` lowers to ``div.full.f32`` as inline asm; LLVM never sees it.

    Under flush that one instruction would keep subnormals while every neighbour
    flushed — a new internal divergence, which is worse than not flushing at all.
    """
    ptx = _PTX_FLUSH + "\n\tdiv.full.f32 \t%f9, %f1, %f2;\n"
    audit = sp.audit_ptx(ptx, sp.FLUSH)
    assert len(audit["violations"]) == 1
    assert audit["violations"][0][1].startswith("div.full.f32")
    # Under keep the same instruction is what keep WANTS: no .ftz anywhere.
    assert sp.audit_ptx(ptx.replace("ftz.", ""), sp.KEEP)["violations"] == []


def test_a_comment_mentioning_an_opcode_is_not_an_instruction():
    assert sp.audit_ptx("\t// mul.rn.f32 is not here\n", sp.FLUSH)["audited"] == 0


def test_the_triton_cache_needs_no_token_once_the_policy_is_in_the_hash():
    """Triton's key includes ``backend.hash()``; CuPy's has no such hook.

    BOTH policies need the fold now: keep stopped being Triton's native behaviour
    when it started rewriting ``nvvm-reflect-ftz``, so a keep process may no
    longer share a cache with a pre-policy one.
    """
    for policy in (sp.FLUSH, sp.KEEP):
        assert sp.triton_cache_reasons(policy, "", hash_folded=True) == []
        assert sp.triton_cache_reasons(policy, "", hash_folded=False) != []
        assert sp.triton_cache_reasons(policy, "/tmp/plain", hash_folded=False) != []
        assert sp.triton_cache_reasons(policy, f"/tmp/triton_{policy}",
                                       hash_folded=False) == []


def test_the_cache_key_carries_the_mechanism_and_not_only_the_policy_name():
    """Measured: four different injected attribute strings compiled into ONE
    ``TRITON_CACHE_DIR`` all returned the first variant's PTX, because the key
    never changed. A change to what the injection DOES must change the key."""
    assert sp.mechanism_digest(sp.FLUSH) != sp.mechanism_digest(sp.KEEP)
    baseline = sp.mechanism_digest(sp.FLUSH)
    try:
        sp.TRITON_DENORMAL_ATTRIBUTE = '"nvptx-f32ftz"="true"'
        assert sp.mechanism_digest(sp.FLUSH) != baseline
    finally:
        sp.TRITON_DENORMAL_ATTRIBUTE = (
            '"denormal-fp-math-f32"="preserve-sign,preserve-sign"')
    assert sp.mechanism_digest(sp.FLUSH) == baseline


class _FakeCUDABackend:
    """Enough of Triton's ``CUDABackend`` for the two hooks to be exercised.

    The ``llir`` stage returns the shape Triton 3.1.0 really emits, so a wrapper
    that only works on hand-written IR cannot pass here. ``llir_text`` is a class
    attribute so a test can hand the stage a different module — the flagless case
    is one the policy must handle without refusing a kernel.
    """

    llir_text = _LLIR_TRITON_SHAPE

    def hash(self):
        return "nvidia-86-ptx83"

    def add_stages(self, stages, options):
        stages["llir"] = lambda src, metadata: type(self).llir_text
        stages["ptx"] = lambda src, metadata: metadata["ptx"]


def _install_fake_triton(monkeypatch, policy):
    backend = _FakeCUDABackend
    monkeypatch.setattr(sp, "_cuda_backend_class", lambda: backend)
    # Only ``triton`` is faked present. Every other lookup goes to the real
    # ``find_spec``: blanket-faking it hid ``llvmlite`` and disarmed the very
    # verification these tests exist to exercise.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "triton" else real_find_spec(name))
    fake_triton = types.ModuleType("triton")
    fake_triton.__version__ = "3.1.0"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    return backend


def test_flush_injects_the_attribute_and_folds_the_policy_into_the_cache_key(monkeypatch):
    backend = _install_fake_triton(monkeypatch, sp.FLUSH)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        assert report["installed"] is True and report["hash_folded"] is True
        assert report["attained"] is True
        assert report["verification"]["ok"] is True
        assert backend().hash() == (
            f"nvidia-86-ptx83-{sp.TRITON_HASH_PREFIX}flush-{sp.mechanism_digest(sp.FLUSH)}")
        stages = {}
        backend().add_stages(stages, options=None)
        emitted = stages["llir"]("src", {})
        assert "denormal-fp-math-f32" in emitted
        assert _llvm_rejects(emitted.replace("!dbg !7 ", "")) is None
        assert sp.triton_counters()["llir_compiles"] == 1
        assert sp.triton_counters()["llir_groups_minted"] == 1
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_keep_rewrites_the_reflect_flag_and_folds_its_own_cache_key(monkeypatch):
    """Keep is no longer "native", so it may not share the native cache key."""
    backend = _install_fake_triton(monkeypatch, sp.KEEP)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        report = sp.install_triton_policy(sp.KEEP)
        assert report["hash_folded"] is True and report["attained"] is True
        assert backend().hash() == (
            f"nvidia-86-ptx83-{sp.TRITON_HASH_PREFIX}keep-{sp.mechanism_digest(sp.KEEP)}")
        stages = {}
        backend().add_stages(stages, options=None)
        emitted = stages["llir"]("src", {})
        assert "denormal-fp-math-f32" not in emitted        # no attribute under keep
        assert 'i32 4, !"nvvm-reflect-ftz", i32 0' in emitted  # but libdevice stops flushing
        assert sp.triton_counters()["reflect_rewrites"] == 1
        assert sp.triton_counters()["reflect_flag_absent"] == 0
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_a_kernel_whose_ir_carries_no_reflect_flag_is_not_refused_over_it(monkeypatch):
    """The exemption's remaining job: a knob that was not there to turn.

    When the llir stage finds no ``nvvm-reflect-ftz`` flag, keep cannot govern
    libdevice, so ``div.rn.ftz.f32`` in that kernel's PTX is recorded rather than
    refused — and the counter says which case this was.
    """
    backend = _install_fake_triton(monkeypatch, sp.KEEP)
    original_hash, original_stages = backend.hash, backend.add_stages
    monkeypatch.setattr(backend, "llir_text", "define void @k() {\n ret void\n}\n")
    try:
        sp.install_triton_policy(sp.KEEP)
        stages = {}
        backend().add_stages(stages, options=None)
        stages["llir"]("src", {})
        assert sp.triton_counters()["reflect_flag_absent"] == 1
        assert sp.triton_counters()["reflect_rewrites"] == 0
        assert stages["ptx"]("src", {"ptx": _PTX_KEEP}) == _PTX_KEEP  # not refused
        # And with the flag present the same PTX IS refused, which is what makes
        # the exemption a scoped fallback rather than a hole.
        sp._reset_for_tests()
        monkeypatch.setattr(backend, "llir_text", _LLIR_TRITON_SHAPE)
        sp.install_triton_policy(sp.KEEP)
        stages = {}
        backend().add_stages(stages, options=None)
        stages["llir"]("src", {})
        with pytest.raises(sp.SubnormalPolicyUnattainable):
            stages["ptx"]("src", {"ptx": _PTX_KEEP})
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_a_second_different_policy_at_the_triton_seam_is_refused(monkeypatch):
    """``install_triton_policy`` is public, so the umbrella lock does not cover it.

    Measured on an earlier revision: ``install_triton_policy('keep')`` then
    ``('flush')`` returned ``attained=True`` for flush while the keep stage
    wrapper was still in place — and folded ``-flush`` into the cache key, so
    keep-compiled binaries were written under the flush namespace.
    """
    backend = _install_fake_triton(monkeypatch, sp.KEEP)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        sp.install_triton_policy(sp.KEEP)
        assert getattr(backend.add_stages, sp._POLICY_MARK) == sp.KEEP
        with pytest.raises(sp.SubnormalPolicyLocked) as excinfo:
            sp.install_triton_policy(sp.FLUSH)
        assert "already" in str(excinfo.value)
        assert getattr(backend.add_stages, sp._POLICY_MARK) == sp.KEEP
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_resetting_the_policy_puts_the_triton_class_back(monkeypatch):
    """Forgetting the bookkeeping while leaving the wrappers is worse than either.

    The wrappers live on the ``CUDABackend`` CLASS and survive a state reset, so
    the next install returned a stamp naming a policy the bytes would not obey —
    and the autouse fixture in this file made that the live path on any
    Triton-bearing host, not a hypothetical.
    """
    backend = _install_fake_triton(monkeypatch, sp.FLUSH)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        sp.install_triton_policy(sp.FLUSH)
        assert backend.add_stages is not original_stages
        sp._reset_for_tests()
        assert backend.add_stages is original_stages
        assert backend.hash is original_hash
        assert getattr(backend.add_stages, sp._POLICY_MARK, None) is None
        # And the next policy therefore installs cleanly rather than locking.
        report = sp.install_triton_policy(sp.KEEP)
        assert report["attained"] is True
        assert getattr(backend.add_stages, sp._POLICY_MARK) == sp.KEEP
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_installing_the_policy_drops_tritons_in_process_kernel_caches(monkeypatch):
    """``JITFunction.run`` keys its own cache ABOVE ``compile()``.

    Measured: a kernel launched before a ``"flush"`` install produced identical
    bytes after it and every counter read zero, because the hash fold separates
    only the DISK cache. Clearing the in-process caches is what makes "installed"
    mean "in force".
    """
    class _JITFunction:
        def __init__(self):
            self.cache = {0: {"key_a": object(), "key_b": object()}}

    module = types.ModuleType("triton.runtime.jit")
    module.JITFunction = _JITFunction
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", module)
    kernel = _JITFunction()
    dropped = sp._clear_triton_jit_caches()
    assert dropped["entries_dropped"] >= 2
    assert kernel.cache[0] == {}


def test_the_ptx_stage_refuses_a_compile_that_defies_the_policy(monkeypatch):
    backend = _install_fake_triton(monkeypatch, sp.FLUSH)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        sp.install_triton_policy(sp.FLUSH)
        stages = {}
        backend().add_stages(stages, options=None)
        assert stages["ptx"]("src", {"ptx": _PTX_FLUSH}) == _PTX_FLUSH
        with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
            stages["ptx"]("src", {"ptx": _PTX_KEEP})
        assert "div.full.f32" in str(excinfo.value)   # the refusal names the gap
        assert sp.triton_counters()["ptx_violations"] == 4
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_the_triton_hooks_are_idempotent(monkeypatch):
    backend = _install_fake_triton(monkeypatch, sp.FLUSH)
    original_hash, original_stages = backend.hash, backend.add_stages
    try:
        sp.install_triton_policy(sp.FLUSH)
        once = backend().hash()
        sp.install_triton_policy(sp.FLUSH)
        assert backend().hash() == once
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


# ---------------------------------------------------------------------------
# The artifact stamp
# ---------------------------------------------------------------------------


def test_the_keep_stamp_still_satisfies_the_gates_policy_predicate(monkeypatch, tmp_path):
    """Continuity, checked by CALLING the predicate rather than by paraphrasing it.

    ``gate_triton_complex.probe_record_policy_reasons`` is the thing that decides
    whether an artifact may license an expansion, and its substantive clauses are
    about the strip COUNTERS, not the name: an earlier version of this test
    asserted the name plus two ``isinstance`` checks and passed on a stamp the
    real predicate refuses ("zero NVRTC compiles under a fresh cache").
    """
    gate_path = API_DIR / "parity" / "meep_gpu" / "gate_triton_complex.py"
    spec = importlib.util.spec_from_file_location("gate_triton_complex", gate_path)
    complex_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(complex_gate)

    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    sp.install_subnormal_policy(sp.KEEP, cupy=cupy)

    # A stamp with no compiles behind it is REFUSED — by the real predicate.
    assert complex_gate.probe_record_policy_reasons({"subnormal_policy": sp.policy_stamp()})

    compiler.compile_using_nvrtc("src", ("--fmad=false", "-ftz=true"))
    stamp = sp.policy_stamp()
    assert complex_gate.probe_record_policy_reasons({"subnormal_policy": stamp}) == []
    assert stamp["policy"] == "ieee_keep_ftz_stripped"
    assert stamp["nvrtc_calls"] == 1 and stamp["ftz_removed"] == 1
    assert stamp["requested"] == "keep"
    assert stamp["resolved"] == "keep"
    assert stamp["match_meep"] is None, "an explicit request measured nothing"
    assert stamp["resolved_from"] == "argument"
    assert stamp["machine"]


def test_the_stamp_carries_the_question_the_answer_and_the_measurement(monkeypatch):
    """A reader must be able to tell WHY this process is on the policy it is on.

    ``requested`` is what was asked (``match_meep`` for a default startup),
    ``resolved`` is what the executors were driven to, and ``match_meep`` is the
    measurement that connects them — what was read, through what, and why it landed
    there. A bare value would leave "the default" indistinguishable from "somebody
    typed keep".
    """
    _module, state = _fake_meep(monkeypatch, effective=True)
    state["flushing"] = True                      # an x86 host after MEEP's own init
    stamp = sp.install_subnormal_policy(strict=True)
    assert stamp["requested"] == sp.MATCH_MEEP
    assert stamp["resolved"] == sp.FLUSH
    assert stamp["policy"] == sp.FLUSH_POLICY_NAME
    assert stamp["resolved_from"] == "default"
    measurement = stamp["match_meep"]
    assert measurement["measured"] is True
    assert measurement["flushing"] is True
    assert measurement["mechanism"].startswith("backends.subnormals_flushed()")
    assert "resolved to 'flush'" in measurement["why"]
    assert "MEEP's own initialization left this host flushing" in measurement["why"]

    sp._reset_for_tests()
    monkeypatch.delitem(sys.modules, "meep")      # the standalone-user process
    state["flushing"] = False                     # a host that keeps, so keep installs
    fallback = sp.install_subnormal_policy(strict=True)["match_meep"]
    assert fallback["measured"] is False and fallback["fallback"] == sp.KEEP
    assert "could NOT be measured" in fallback["why"]


def test_an_uninstalled_stamp_carries_no_certification_grade_name(monkeypatch):
    """A resolved policy is a preference; a stamp is a certification.

    Measured on an earlier revision: with ``MEEP_GPU_SUBNORMAL_POLICY=keep`` and
    nothing installed, ``policy_stamp()['policy']`` read ``ieee_keep_ftz_stripped``
    — the exact string ``fingerprints.json`` ties nine certifications to — with
    zero NVRTC calls and zero options stripped.
    """
    monkeypatch.setenv(sp.POLICY_ENV, "keep")
    stamp = sp.policy_stamp()
    assert stamp["policy"] == sp.UNINSTALLED_POLICY_NAME
    assert stamp["policy"] not in (sp.CUPY_KEEP_POLICY_NAME, sp.FLUSH_POLICY_NAME)
    assert stamp["would_be_policy"] == sp.CUPY_KEEP_POLICY_NAME
    assert stamp["installed"] is False
    sp.install_subnormal_policy(sp.KEEP)
    assert sp.policy_stamp()["policy"] == sp.CUPY_KEEP_POLICY_NAME


def test_the_flush_stamp_names_a_different_policy(monkeypatch):
    _module, _state = _fake_meep(monkeypatch, effective=True)
    stamp = sp.install_subnormal_policy(sp.FLUSH)
    assert stamp["policy"] == sp.FLUSH_POLICY_NAME != "ieee_keep_ftz_stripped"
    assert stamp["installed"] is True


def test_the_stamp_lists_every_executor_that_did_not_attain_the_policy(monkeypatch):
    _a_host_with_no_working_lever(monkeypatch)
    with pytest.warns(RuntimeWarning):
        stamp = sp.install_subnormal_policy(sp.FLUSH, strict=False)
    assert stamp["unattained"] == ["cupy", "host", "triton"]
    assert set(stamp["executors"]) == {"host", "cupy", "triton"}


# ---------------------------------------------------------------------------
# The conformance gate's own case builders
# ---------------------------------------------------------------------------


def test_every_case_is_built_from_bit_patterns_not_literals(gate):
    """A float literal would be baked to 0.0 in a flushed process and pass hollowly."""
    for case in gate.build_cases():
        assert case["a"].dtype == np.uint32, case["op"]
        assert case["b"].dtype == np.uint32, case["op"]
        assert len(case["a"]) == len(case["b"]) == gate.N_LANES, case["op"]


def test_every_case_lane0_keep_word_is_the_exact_ieee_result(gate):
    """The gate's expected words are checked here against real IEEE arithmetic.

    The laptop keeps subnormals (asserted), so plain NumPy IS the IEEE oracle for
    lane 0 — which makes this a check of the CASE TABLE, not of an executor.
    """
    assert backends.subnormals_flushed() is False
    for case in gate.build_cases():
        bits = gate.exec_host_numpy(case)
        if bits is None:
            assert case["kind"] in ("fma", "cumsum"), case["op"]
            continue
        assert int(bits[0]) == int(case["keep_lane0"]), case["op"]


def test_the_scan_case_takes_its_verdict_from_lane_one(gate):
    """Lane 0 of a cumulative sum performs no addition — it is a pass-through.

    A verdict that counted it would read ``mixed`` on a perfectly uniform flush,
    which is the one answer that means "this measurement decides nothing".
    """
    case = {c["op"]: c for c in gate.build_cases()}["cumsum_scan"]
    assert case["verdict_from"] == 1 and case["triton_only"] is True
    # The keep word IS the exact IEEE cumulative sum at that lane: 2 * 2^-136.
    running = np.cumsum(case["a"].view(np.float32).astype(np.float64))
    assert np.float32(running[1]).view(np.uint32) == case["keep_lane0"]
    # A flushed scan whose lane 0 survived must still read "flush", not "mixed".
    flushed = np.zeros(gate.N_LANES, dtype=np.uint32)
    flushed[0] = gate.BITS_2P_M136
    assert gate.verdict(flushed, case)["verdict"] == "flush"


def test_the_scan_case_is_the_only_triton_only_case(gate):
    """Everything else must be measurable on more than one executor.

    A case only one executor can run cannot detect a DISAGREEMENT between
    executors, which is what this gate is for; cumsum earns the exception because
    it is an op class the shipped kernels genuinely use and no other executor has
    a single-operation counterpart for.
    """
    triton_only = [c["op"] for c in gate.build_cases() if c.get("triton_only")]
    assert triton_only == ["cumsum_scan"]


def test_every_governed_case_distinguishes_the_two_policies_where_it_is_governed(gate):
    """A case whose keep word equals its flush word could never detect anything.

    Per EXECUTOR, because the classification is per executor: ``-x`` is governed
    on the device and not on the host, so the invariant "governed implies the two
    words differ" only means anything against an executor the case is governed on.
    """
    for case in gate.build_cases():
        for executor in ("host_numpy", "cupy_ufunc", "triton"):
            keep = gate.expected_lane0(case, sp.KEEP, executor)
            flush = gate.expected_lane0(case, sp.FLUSH, executor)
            governed = gate.expected_verdict(case, sp.FLUSH, executor) == "flush"
            assert (keep != flush) == governed, (case["op"], executor)


def test_the_host_does_not_flush_a_negation_and_the_gate_knows_it(gate):
    """x86 FTZ/DAZ are MXCSR bits over SSE ARITHMETIC; ``-x`` is a sign flip.

    Measured on the GPU host with the FPU flushing: ``np.negative(0x00004000)`` ->
    ``0x80004000``, while ``np.multiply``/``np.subtract``/``np.divide`` on the same
    operand all returned zero. An earlier gate had no executor axis and therefore
    FAILED a correctly installed flush policy on this one cell.
    """
    case = {c["op"]: c for c in gate.build_cases()}["neg_subnormal"]
    assert gate.expected_verdict(case, sp.FLUSH, "host_numpy") == "keep"
    assert gate.expected_lane0(case, sp.FLUSH, "host_numpy") == case["keep_lane0"]
    # The device flushes it, and the two device lowerings disagree in the SIGN:
    # neg.ftz.f32 gives -0, while Triton's ``-x`` is ``sub.rn.f32 0, x`` and gives +0.
    assert gate.expected_verdict(case, sp.FLUSH, "cupy_ufunc") == "flush"
    assert gate.expected_lane0(case, sp.FLUSH, "cupy_ufunc") == gate.SIGN
    assert gate.expected_lane0(case, sp.FLUSH, "triton") == 0x00000000


def test_the_host_flush_expectations_are_the_words_x86_actually_produced(gate):
    """The case table, pinned against a measurement rather than against a model.

    These are the lane-0 words the host leg produced on the GPU host (x86_64, MEEP
    1.33.0) with the flush policy installed and ``subnormals_flushed()`` measured
    True. Six flushed; ``neg`` did not, and that one cell is why the expectations
    carry an executor axis at all.
    """
    measured = {
        "mul_result_subnormal": 0x00000000,
        "sub_cancellation_subnormal": 0x00000000,
        "mul_subnormal_input": 0x00000000,
        "add_signed_zero": 0x00000000,
        "div_rn_result_subnormal": 0x00000000,
        "neg_subnormal": 0x80004000,          # xorps; FTZ/DAZ never see it
        "select_transport": 0x00004000,       # transport keeps under both policies
    }
    cases = {case["op"]: case for case in gate.build_cases()}
    for op, word in measured.items():
        assert gate.expected_lane0(cases[op], sp.FLUSH, "host_numpy") == word, op


def test_a_cell_that_matches_the_class_but_not_the_word_still_fails(gate):
    """The verdict alone cannot tell ``-0`` from ``+0``; both read "flush".

    An earlier revision computed the word match and then ignored it, so two
    executors producing different bytes for the same operation both passed.
    """
    case = {c["op"]: c for c in gate.build_cases()}["neg_subnormal"]
    plus_zero = np.zeros(gate.N_LANES, dtype=np.uint32)
    minus_zero = np.full(gate.N_LANES, gate.SIGN, dtype=np.uint32)
    good = gate.judge(minus_zero, case, sp.FLUSH, "cupy_ufunc")
    bad = gate.judge(plus_zero, case, sp.FLUSH, "cupy_ufunc")
    assert good["agrees"] is True
    assert bad["verdict_agrees"] is True and bad["lane0_agrees"] is False
    assert bad["agrees"] is False


def test_the_governed_cases_cover_every_f32_op_class_the_kernels_use(gate):
    """add, sub, mul, neg, fma, div, comparison — plus transport (tl.where).

    The shipped kernels use ``tl.math.fma`` (11 sites), ``tl.math.div_rn`` (5),
    ``tl.where`` (455), ``!=`` as a store mask (``conductivity.py:307``, 48
    ``setp.neu.f32`` in one kernel) and plain ``+ - *``; no shipped kernel spells a
    plain ``/`` — and none uses ``tl.cumsum``, which is why the scan case is
    marked ``shipped=False`` rather than claimed as coverage.
    """
    cases = gate.build_cases()
    kinds = {case["kind"] for case in cases}
    assert {"add", "sub", "mul", "neg", "fma", "div", "select", "cmp"} <= kinds
    scan = [case for case in cases if case["kind"] == "cumsum"]
    assert [case.get("shipped", True) for case in scan] == [False]


def test_the_comparison_case_is_classified_by_word_not_by_structure(gate):
    """Its result is a flag (1.0 / 0.0), and neither word is subnormal."""
    case = {c["op"]: c for c in gate.build_cases()}["cmp_subnormal_against_zero"]
    assert case["verdict_mode"] == "word"
    ones = np.full(gate.N_LANES, gate.BITS_ONE, dtype=np.uint32)
    zeros = np.zeros(gate.N_LANES, dtype=np.uint32)
    assert gate.verdict(ones, case)["verdict"] == "keep"
    assert gate.verdict(zeros, case)["verdict"] == "flush"
    mixed = ones.copy()
    mixed[3] = 0
    assert gate.verdict(mixed, case)["verdict"] == "mixed"


def test_the_transport_case_is_not_compared_against_the_policy(gate):
    """``selp.f32`` and ``ld.global.f32`` carry no ``.ftz`` under any policy."""
    cases = {case["op"]: case for case in gate.build_cases()}
    transport = cases["select_transport"]
    assert gate.expected_verdict(transport, sp.FLUSH, "triton") == "keep"
    assert gate.expected_verdict(transport, sp.KEEP, "triton") == "keep"
    governed = cases["mul_result_subnormal"]
    assert gate.expected_verdict(governed, sp.FLUSH, "triton") == "flush"
    assert gate.expected_verdict(governed, sp.KEEP, "triton") == "keep"


def test_every_leg_the_gate_offers_maps_to_an_executor_and_a_matrix_row(gate):
    """``--legs host`` must install the host executor and nothing else.

    Measured: the gate's own documented laptop invocation exited 2 on a host that
    merely had CuPy importable, because the install was unconditional and CuPy's
    cache-directory refusal fired for a leg that was never requested.
    """
    assert set(gate.LEG_EXECUTORS) == set(gate.ALL_LEGS) == set(gate.LEG_ROWS)
    assert gate.LEG_EXECUTORS["host"] == "host"
    assert {gate.LEG_EXECUTORS[leg] for leg in gate.ALL_LEGS} == {"host", "cupy", "triton"}


def test_the_verdict_classifier_reads_raw_words_and_reports_mixed(gate):
    case = gate.build_cases()[0]
    kept = np.full(gate.N_LANES, gate.BITS_2P_M135, dtype=np.uint32)
    flushed = np.zeros(gate.N_LANES, dtype=np.uint32)
    mixed = kept.copy()
    mixed[7] = 0
    assert gate.verdict(kept, case)["verdict"] == "keep"
    assert gate.verdict(kept, case)["lane0_matches_keep"] is True
    assert gate.verdict(flushed, case)["verdict"] == "flush"
    assert gate.verdict(mixed, case)["verdict"] == "mixed"
    assert gate.verdict(mixed, case)["n_kept"] == gate.N_LANES - 1


def test_a_signed_zero_still_counts_as_flushed(gate):
    """``neg.ftz.f32`` flushes to -0, not +0; a classifier that missed that would
    report ``mixed`` on a perfectly uniform flush."""
    case = {case["op"]: case for case in gate.build_cases()}["neg_subnormal"]
    signed_zero = np.full(gate.N_LANES, gate.SIGN, dtype=np.uint32)
    assert gate.verdict(signed_zero, case)["verdict"] == "flush"


def test_the_rawkernel_leg_uses_the_engines_own_options(gate):
    """Not a convenient option set — the one the array path actually compiles with."""
    assert gate.ENGINE_RAWKERNEL_OPTIONS == ("--fmad=false",)


def test_the_reduction_case_hides_one_subnormal_among_zeros(gate):
    case = gate.reduction_case()
    assert case["buffer"].dtype == np.uint32
    assert int((case["buffer"] != 0).sum()) == 1
    assert int(case["keep_word"]) == gate.BITS_2P_M135
    assert int(case["flush_word"]) == 0
    # It must not claim CuPy's options reach this path: measured, the default
    # reduction is a CUB binary that kept the subnormal with the rest of the
    # process flushing, which is why the policy switches the accelerator off.
    assert "-ftz=true" not in case["why"]
    assert "CUB" in case["why"]


def test_a_gate_run_that_measured_nothing_does_not_read_pass(gate, tmp_path):
    """A green artifact with missing rows is worse than a red one.

    Measured: the gate returned ``PASS`` with exit 0 while 5 of the 6 executors it
    enumerates never ran, the Triton leg's absence was recorded NOWHERE (it was
    nested inside "did CuPy import"), and the stamp asserted those executors had
    attained the policy.
    """
    out = tmp_path / "gate.json"
    code = gate.run(sp.KEEP, str(out), ("host", "triton"))
    record = json.loads(out.read_text())
    assert record["verdict"] == "INCOMPLETE" and code == 3
    assert record["legs_without_cells"] == ["triton"]
    assert "triton" in record["skipped"]
    assert record["executors_installed"] == ["host", "triton"]

    code = gate.run(sp.KEEP, str(out), ("host",))
    record = json.loads(out.read_text())
    assert record["verdict"] == "PASS" and code == 0
    assert record["legs_without_cells"] == []
    assert record["executors_installed"] == ["host"]


# ---------------------------------------------------------------------------
# The non-x86 host lever, and the fourth executor
# ---------------------------------------------------------------------------
#
# WHY BOTH ARRIVED TOGETHER. The Metal kernel table can be certified under exactly
# one float32 subnormal policy — MPS flushes denormals natively and exposes no
# lever, so ``keep`` is unattainable on that executor — and Apple silicon is where
# MPS lives. Without a host lever the two halves of a Metal run would be
# permanently unable to agree, which is not a policy split this module may install:
# it is a refusal. The fenv lever is what makes ``flush`` attainable on the host
# half, and the ``mps`` arm is what makes the device half report rather than assume.


def test_the_fenv_lever_is_tried_only_after_meeps_knob_and_only_if_it_is_needed(
        monkeypatch):
    """MEEP'S KNOB FIRST, ALWAYS — on x86 nothing about this changes.

    ``match_meep`` means matching MEEP, so the mechanism the certified families were
    cut under has to be the one that runs where it works. The second lever is
    reached only when the MEASUREMENT says the first did not move the FPU.
    """
    calls = []
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv",
                        lambda want_flush: calls.append(want_flush) or True)
    _module, _state = _fake_meep(monkeypatch, effective=True)
    report = sp.install_host_policy(sp.FLUSH, strict=True)
    assert report["attained"] is True
    assert report["mechanism"] == "meep.set_zero_subnormals"
    assert calls == [], "the fenv lever was tried on a host whose knob already worked"


def test_the_fenv_lever_carries_the_run_when_meeps_knob_is_the_arm64_no_op(
        monkeypatch):
    """The arm64 case, and the mechanism string names BOTH halves.

    A record that said only ``meep.set_zero_subnormals`` would describe a call that
    did nothing; one that said only the lever would hide that MEEP's own knob was
    asked first. Both are in the string because both happened.
    """
    module, state = _fake_meep(monkeypatch, effective=False)

    def lever(want_flush):
        state["flushing"] = bool(want_flush)
        return True

    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lever)
    report = sp.install_host_policy(sp.FLUSH, strict=True)
    assert report["attained"] is True
    assert report["mechanism"].startswith("meep.set_zero_subnormals (no-op here)")
    assert sp.FENV_MECHANISM_PREFIX in report["mechanism"], report["mechanism"]
    assert module.set_zero_subnormals is not None


def test_a_fenv_symbol_this_platform_does_not_export_is_a_refusal_not_a_crash(
        monkeypatch):
    """``_FE_DFL_DISABLE_DENORMS_ENV`` is a Darwin export; elsewhere it is absent.

    A lever that raised on a platform without the symbol would turn "this host
    cannot flush" — a refusal the ladder handles — into an exception inside a
    module whose whole contract is to decide rather than to raise.
    """
    class _NoSymbol:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(sp.backends, "_math_library", lambda: _NoSymbol())
    assert sp._drive_host_fpu_by_fenv(True) is False


def test_a_lever_that_moves_the_fpu_without_attaining_the_policy_restores_it(
        monkeypatch):
    """THE THIRD STATE THAT BELONGS TO NOBODY, and the reason the restore is unconditional.

    A header reading cannot settle whether an environment object carries the
    denormal bits on this ABI, so the lever installs a candidate and MEASURES. If
    the measurement disagrees, the saved environment goes back immediately — a
    process left in a mode neither the request nor the previous policy describes is
    worse than the refusal, because nothing reports it.
    """
    calls = {"set": 0, "restored": 0}
    saved_marker = object()

    class _Library:
        _FE_DFL_DISABLE_DENORMS_ENV = None

        def fegetenv(self, address):
            return 0

        def fesetenv(self, address):
            calls["set"] += 1
            return 0

    library = _Library()
    monkeypatch.setattr(sp.backends, "_math_library", lambda: library)
    monkeypatch.setattr(sp.ctypes, "addressof", lambda value: id(value))
    monkeypatch.setattr(sp.ctypes.c_char, "in_dll",
                        classmethod(lambda cls, lib, name: saved_marker))
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: False)
    assert sp._drive_host_fpu_by_fenv(True) is False
    # Two calls: the candidate, then the saved environment put back.
    assert calls["set"] == 2, calls


def test_the_mps_executor_is_asked_only_when_it_is_named(monkeypatch):
    """``executors`` NAMES THE ARMS TO DRIVE and ``mps`` is not in the default tuple.

    A process with no MPS device has no business importing the Metal package to be
    told about an arm it never asked for, so the fourth executor is opt-in — and a
    Metal dispatch asks for it by name.
    """
    _module, state = _fake_meep(monkeypatch, effective=True)
    stamp = sp.install_subnormal_policy(sp.FLUSH, strict=False)
    assert "mps" not in stamp["executors"], sorted(stamp["executors"])
    assert state["flushing"] is True


def test_the_mps_arm_attains_flush_and_refuses_keep_by_name(monkeypatch):
    """The device half of the Metal table's one policy, through the real arm.

    MPS flushes float32 denormals natively and both denormal pragma spellings are
    compile errors on this toolchain, so ``flush`` is ATTAINED with no action and
    ``installed`` is False — no seam is wrapped and nothing has to be put back, which
    is why a refused sibling's rollback must not be blocked by this arm. ``keep`` is
    a refusal BY NAME rather than a silent downgrade.
    """
    from meep_gpu.metal_kernels import subnormal as metal_subnormal

    report = metal_subnormal.install_mps_policy(sp.FLUSH, strict=False)
    assert report["attained"] is True
    assert report["installed"] is False
    assert report["reasons"] == []
    refused = metal_subnormal.install_mps_policy(sp.KEEP, strict=False)
    assert refused["attained"] is False
    assert any("unattainable on the MPS executor" in reason
               for reason in refused["reasons"]), refused["reasons"]


def test_a_metal_dispatch_drives_host_and_mps_and_refuses_keep_on_both(monkeypatch):
    """``executors=("host", "mps")`` is what rung 8bM asks for, driven end to end.

    THE ONE-POLICY RULE IS UNCHANGED HERE; only the VALUE is this table's. Under
    ``flush`` both arms attain and ``unattained`` is empty; under ``keep`` the mps
    arm refuses by name and the whole install is refused, which is exactly what
    stops a Metal run from being certified under a split.
    """
    _module, state = _fake_meep(monkeypatch, effective=True)
    stamp = sp.install_subnormal_policy(sp.FLUSH, executors=("host", "mps"),
                                        strict=False)
    assert stamp["unattained"] == [], stamp["unattained"]
    assert set(stamp["executors"]) == {"host", "mps"}
    sp._reset_for_tests()
    state["flushing"] = backends.subnormals_flushed()
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_subnormal_policy(sp.KEEP, executors=("host", "mps"), strict=True)
    assert "'mps'" in str(excinfo.value), str(excinfo.value)
