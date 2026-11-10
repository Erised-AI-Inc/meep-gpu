"""The float32 subnormal policy: the flag, the three executors, and the audit.

This suite runs on the DEV LAPTOP, which is arm64 with neither CuPy nor Triton
installed. That is not a limitation to work around here — it is one of the facts
under test. MEEP's ``_set_zero_subnormals`` is compiled under
``#if HAVE_IMMINTRIN_H`` (x86 SSE), so on this machine ``mp.set_zero_subnormals(True)``
is a NO-OP and the ``"flush"`` policy is unattainable for the host path. The
package must say so, loudly, rather than run a different policy than the one it
was asked for, and :func:`test_flush_is_refused_on_a_host_where_meeps_knob_is_a_no_op`
is that assertion.

The same machine decides the default, and only one side of that is reachable here.
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

And the IR is handed to a real LLVM parser. The Triton fixtures below are the
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
from meep_gpu.test_backends import (
    CUPY_14_COMPILER,
    MOVED_SEAM_COMPILER,
    NVRTC_FRONT_END,
    cache_name,
    cupy_compiler_module,
)

API_DIR = Path(__file__).resolve().parent.parent
GATE_PATH = API_DIR / "parity" / "meep_gpu" / "gate_subnormal_policy.py"


@pytest.fixture(autouse=True)
def _fresh_policy(monkeypatch):
    """Every test starts with no policy installed AND with the FPU where it found it.

    The second half is new and it is not tidiness. ``_reset_for_tests`` clears this
    module's bookkeeping; it does not put the floating-point environment back,
    because an ATTAINED install deliberately stands — a process that asked to flush
    and got it should keep flushing. That is right in production and wrong in a
    suite: the moment the non-x86 fenv lever landed, any test that installed
    ``flush`` left the FPU flushing for every test after it, and two later tests
    that compute an exact IEEE keep-word failed in a full-file run while passing in
    isolation. Measured 2026-09-10 on exactly that pair.

    A test that leaks process state into its neighbours is the one failure a green
    isolated run cannot see, so the entry state is measured and restored here rather
    than remembered by each test. ``_drive_host_fpu`` is the same operation the
    rollback path uses, and it is symmetric by construction.
    """
    sp._reset_for_tests()
    flushing_at_entry = backends.subnormals_flushed()
    # And with no CuPy CACHE POINTER, where CuPy is installed. A dispatch earlier in
    # the session points ``CUPY_CACHE_DIR`` at its keep-policy directory for the rest
    # of the process (``point_cupy_cache_at_keep_policy``), and a ``flush`` install
    # here then refuses that directory by name. Measured in the NVIDIA reference
    # environment, in a run where a keep dispatch came earlier. A test that needs a
    # directory names one itself (:func:`_keep_policy_cupy_cache`).
    if importlib.util.find_spec("cupy") is not None:
        monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    try:
        yield
    finally:
        sp._reset_for_tests()
        if backends.subnormals_flushed() != flushing_at_entry:
            sp._drive_host_fpu(flushing_at_entry)


def _a_host_with_no_working_lever(monkeypatch):
    """Stand in for a host where NEITHER host lever can flush. Returns MEEP's stub.

    Why the tests below need it, and it is a change of PREMISE rather than of
    expectation. They were written on this arm64 laptop when ``flush`` was
    genuinely unattainable here: MEEP's ``set_zero_subnormals`` is the
    ``#if HAVE_IMMINTRIN_H`` no-op on this architecture and there was no second
    lever, so "install flush and watch it be refused" needed no arrangement at all.
    The Metal dispatch table cannot exist without ``flush`` — MPS flushes natively
    and offers nothing else — so ``subnormal_policy`` gained the Darwin fenv lever,
    and on this machine the refusal these tests are about no longer happens.

    The refusal is still a real branch and it is still the one that matters: a host
    with neither lever must refuse rather than half-apply. So the platform is
    ARRANGED instead of assumed — MEEP's knob a no-op, the fenv lever answering
    "did not work" — and the tests keep measuring what they always measured, on
    every architecture rather than on one.
    """
    module, state = _fake_meep(monkeypatch, effective=False)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lambda want_flush: False)
    # The x86-64 glibc lever is the MEEP-less path and a stand-in MEEP is imported
    # here, so it is not reached; it is stubbed too so "neither lever" stays true
    # on an x86-64 Linux host whatever the ladder's order becomes.
    monkeypatch.setattr(sp, "_drive_host_fpu_by_mxcsr", lambda want_flush: False)
    return module, state


def _keep_policy_cupy_cache(monkeypatch, tmp_path) -> None:
    """Where CuPy is installed, a private ``CUPY_CACHE_DIR`` that ``keep`` may use.

    A direct ``keep`` install drives the CuPy executor too, and refuses by name
    without a policy-suffixed private directory (:func:`sp.cupy_cache_reasons`):
    dispatch derives one, a direct install does not. The tests that install ``keep``
    themselves name one first, carrying :data:`sp.CUPY_CACHE_POLICY_TOKEN`, fresh and
    empty. Where CuPy is not installed nothing is set: the CuPy leg is not driven
    there, and those hosts run these tests exactly as before.
    """
    if importlib.util.find_spec("cupy") is None:
        return
    monkeypatch.setenv("CUPY_CACHE_DIR",
                       str(tmp_path / f"cupy-cache-{sp.CUPY_CACHE_POLICY_TOKEN}"))


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
    ``#if HAVE_IMMINTRIN_H`` no-op. The x86 half is not verifiable on this machine;
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


def test_the_default_is_attainable_on_this_host_without_meep_imported(monkeypatch, tmp_path):
    """The default must never be a policy this host cannot deliver.

    This is the assertion behind the deferred startup wiring: whatever the default
    resolves to here must install, on this machine, in a process that has not
    imported MEEP — which is the case where the answer is the fallback, so the
    fallback is under test too. A fresh interpreter for the same reason as above;
    it also keeps ``install_subnormal_policy(strict=True)``'s FPU mutation out of
    the suite's own process.
    """
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
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


def test_installing_a_policy_never_imports_meep(monkeypatch, tmp_path):
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
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
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


def test_installing_the_same_policy_twice_is_idempotent(monkeypatch, tmp_path):
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
    first = sp.install_subnormal_policy(sp.KEEP)
    second = sp.install_subnormal_policy(sp.KEEP)
    assert first["requested"] == second["requested"] == "keep"


def test_a_second_different_policy_is_refused_not_swapped(monkeypatch, tmp_path):
    """Binaries and cache entries already exist under the first one."""
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
    sp.install_subnormal_policy(sp.KEEP)
    with pytest.raises(sp.SubnormalPolicyLocked) as excinfo:
        sp.install_subnormal_policy(sp.FLUSH)
    assert "already installed" in str(excinfo.value)
    assert sp.get_subnormal_policy() == "keep"


def test_a_refusal_that_installed_no_device_seam_leaves_the_process_unlocked(monkeypatch,
                                                                             tmp_path):
    """Otherwise the refused policy would lock out the one the caller can have."""
    _a_host_with_no_working_lever(monkeypatch)
    with pytest.raises(sp.SubnormalPolicyUnattainable):
        sp.install_subnormal_policy(sp.FLUSH, strict=True)
    assert sp._STATE["policy"] is None
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
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


def _fake_cupy(monkeypatch, tmp_path, accelerators=(_FAKE_CUB,), shape="13", source=""):
    """A CuPy stand-in exposing every seam the policy touches.

    ``accelerators`` is what CuPy IMPORTED with, which is the only thing that
    decides where a reduction is dispatched — see
    :func:`test_flush_refuses_because_the_runtime_switch_off_does_not_move_the_dispatch`.
    Pass ``()`` for a CuPy that came up under ``CUPY_ACCELERATORS=''``.

    Three of them, all load-bearing: the compiler module (the option strip at the
    front end its compile path calls, CuPy ``shape``'s stand-in from
    ``test_backends``; ``calls`` logs the option tuple every front end receives),
    ``_core._accelerator`` (the CUB reduction path, which no compile option can
    reach) and ``_util.clear_memo`` (the in-process kernel cache, which is above
    the disk cache and defeats a late install). Scoped through ``monkeypatch`` and
    never written into ``sys.modules``, so ``conftest``'s impostor guard is
    untouched.
    """
    calls = []
    state = {"reduction_accelerators": list(accelerators), "memo_cleared": 0}

    def front_end(name, source, options):
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

    compiler = cupy_compiler_module(shape, front_end, tmp_path / "cupy_default_cache", source)
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
    assert report["wrapped_entry_points"] == [
        "compile_using_nvrtc", "compile_using_nvcc", "compile_using_hipcc"]

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


def _fake_kernel_module(*, ufunc_memo="_kernel_memo",
                        elementwise_memo="_elementwise_kernel_memo"):
    """``cupy._core._kernel`` as far as the per-object drop reads it.

    The two classes carry their memo NAME at class level, as CuPy's compiled classes
    do (a ``readonly dict`` attribute is a descriptor on the class), and each
    instance holds its own dict of compiled kernels. Pass ``None`` for a name to
    stand in for a CuPy whose class no longer has that attribute.
    """
    namespace = {}
    for class_name, attribute in (("ufunc", ufunc_memo),
                                  ("ElementwiseKernel", elementwise_memo)):
        body = {attribute: None} if attribute else {}
        namespace[class_name] = type(class_name, (), body)
    return types.SimpleNamespace(**namespace)


def test_keep_drops_the_kernels_a_ufunc_and_an_elementwise_kernel_hold_themselves(
        monkeypatch, tmp_path):
    """``clear_memo`` is not the whole drop: a ufunc keeps its kernels on the object.

    Measured on the GPU host (CuPy 13.5.1): a float32 ``cp.multiply`` compiled before
    a keep install returned 0.0 for 1e-30 * 1e-10 after ``cupy.clear_memo()``, while
    the install reported ``attained=True``; a lift compiles exactly that signature
    before dispatch installs the policy, and stepping reuses it. Emptying every live
    ufunc's ``_kernel_memo`` and every ``ElementwiseKernel``'s
    ``_elementwise_kernel_memo`` is what made the same call return 9.99995e-41.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, state = _fake_cupy(monkeypatch, tmp_path)
    kernel = _fake_kernel_module()
    cupy._core._kernel = kernel
    multiply = kernel.ufunc()
    multiply._kernel_memo = {(0, "float32,float32->float32", (), False): object(),
                             (0, "float64,float64->float64", (), False): object()}
    elementwise = kernel.ElementwiseKernel()
    elementwise._elementwise_kernel_memo = {(0, "float32"): object()}

    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["attained"] is True, report["reasons"]
    assert multiply._kernel_memo == {} and elementwise._elementwise_kernel_memo == {}
    memo = report["kernel_memo"]
    assert state["memo_cleared"] == 1
    assert memo["walked"] is True and memo["cleared"] is True
    assert memo["entries_dropped"] == {"ufunc._kernel_memo": 2,
                                       "ElementwiseKernel._elementwise_kernel_memo": 1}


@pytest.mark.parametrize("missing", ["ufunc", "ElementwiseKernel"])
def test_a_cupy_whose_per_object_memo_has_moved_is_refused_by_name(
        monkeypatch, tmp_path, missing):
    """A memo the drop cannot find is a policy that is not in force, so it is refused.

    The defect this drop closes was a policy reported attained while pre-install
    kernels stayed reachable. A CuPy whose classes no longer carry the attribute
    would reproduce it silently, so the install names the attribute and refuses.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    cupy._core._kernel = _fake_kernel_module(
        ufunc_memo=None if missing == "ufunc" else "_kernel_memo",
        elementwise_memo=None if missing == "ElementwiseKernel"
        else "_elementwise_kernel_memo")
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    attribute = "_kernel_memo" if missing == "ufunc" else "_elementwise_kernel_memo"
    assert f"cupy._core._kernel.{missing} has no {attribute}" in str(excinfo.value)
    assert sp.executor_report("cupy")["attained"] is False


def test_uninstalling_the_strip_drops_the_kernels_compiled_under_it(monkeypatch, tmp_path):
    """The mirror of a late install: stripped kernels must not outlive the strip.

    After an uninstall the seam compiles natively again (CuPy's own ``-ftz=true``),
    so a ufunc kernel compiled while the strip was in place would otherwise keep
    subnormals in a process that no longer carries the policy.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path)
    kernel = _fake_kernel_module()
    cupy._core._kernel = kernel
    sp.install_cupy_policy(sp.KEEP, cupy)
    multiply = kernel.ufunc()
    multiply._kernel_memo = {(0, "float32,float32->float32", (), False): object()}
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    restored = sp.uninstall_subnormal_policy()
    assert multiply._kernel_memo == {}
    assert restored["cupy_kernels_dropped"] == 1


@pytest.mark.parametrize("shape", ["13", "14"])
def test_keep_folds_its_token_into_cupys_cache_key_and_uninstall_takes_it_out(
        shape, monkeypatch, tmp_path):
    """The key is computed above the strip, so the keep token goes into the key itself.

    ``extra_source`` is hashed into CuPy's key and handed to no compiler. With the
    token folded in, a keep compile cannot read a binary CuPy compiled natively into
    the same directory before the install — measured on device: in a keep directory
    minted before the process started, the memo drop alone left 0 of 3 float32
    products keeping, because the recompile read the pre-install binary back.

    Through the compile path's own caller, in both shapes: CuPy 14 dropped the
    ``cache_dir`` parameter, so a token folded by POSITION would land in ``backend``
    and every keep compile would raise "Invalid backend".
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    cache.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape=shape)
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["cache_key_folded"] is True
    compiler._compile_module_with_cache("src", ("--fmad=false",))
    compiler._compile_module_with_cache("src", ("--fmad=false",), extra_source="headers")
    options = ("--fmad=false", "-ftz=true")
    assert sorted(os.listdir(cache)) == sorted([
        cache_name("src", options, sp.CUPY_KEEP_CACHE_KEY_TOKEN),
        cache_name("src", options, "headers\n" + sp.CUPY_KEEP_CACHE_KEY_TOKEN)])

    sp.uninstall_subnormal_policy()
    compiler._compile_module_with_cache("src", ("--fmad=false",))
    assert cache_name("src", options, None) in os.listdir(cache)


@pytest.mark.parametrize("shape", ["13", "14"])
def test_keep_strips_the_front_end_the_compile_path_calls(shape, monkeypatch, tmp_path):
    """The strip must be CALLED by the compile path, not merely installed.

    CuPy 14's compile path calls ``_compile_using_nvrtc_no_warning``; a strip on
    ``compile_using_nvrtc`` there installs, reports ``installed``, and leaves CuPy's
    own ``-ftz=true`` on every compile.
    """
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, calls, _state = _fake_cupy(monkeypatch, tmp_path, shape=shape)
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["attained"] is True and report["installed"] is True
    assert report["wrapped_entry_points"][0] == NVRTC_FRONT_END[shape]
    compiler._compile_module_with_cache("src", ("--fmad=false",))
    compiler._compile_module_with_cache("src2", ("--fmad=false",), backend="nvcc")
    assert calls == [("--fmad=false",), ("--fmad=false",)]
    assert sp.cupy_strip_counters() == {"calls": 2, "removed": 2}
    # A direct call of CuPy 14's public front end is stripped once, through the private one.
    compiler.compile_using_nvrtc("src3", ("-ftz=true",))
    assert calls[-1] == () and sp.cupy_strip_counters() == {"calls": 3, "removed": 3}


def test_a_late_cache_dir_reaches_cupy_14_which_reads_it_once_at_import(monkeypatch, tmp_path):
    """CuPy 14 fixes its cache directory when ``cupy.cuda.compiler`` is imported.

    The keep policy's private directory is set AFTER the import (dispatch moves it at
    the freeze, and its repair moves it again), so on that shape the every-compile
    read is put back: a compile caches where the variable points NOW, a restore of
    the variable is followed too, and an uninstall puts CuPy's own backend back.
    """
    start = tmp_path / "cupy_cache_at_import"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(start))
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="14")
    original = compiler._kernel_cache_backend
    moved = sp.point_cupy_cache_at_keep_policy(cupy)
    assert moved["after"] == f"{start}-{sp.CUPY_CACHE_POLICY_TOKEN}"
    assert moved["cupy_reads_it"]["reads"] == "every compile"
    assert moved["cupy_reads_it"]["replaced_backend_directory"] == str(start)
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["attained"] is True
    assert report["cache_dir_read"]["reads"] == "every compile"
    compiler._compile_module_with_cache("src", ())
    assert len(os.listdir(moved["after"])) == 1 and not os.listdir(start)

    monkeypatch.setenv("CUPY_CACHE_DIR", str(start))  # the gate's restore of the variable
    compiler._compile_module_with_cache("src2", ())
    assert len(os.listdir(start)) == 1

    sp.uninstall_subnormal_policy()
    assert compiler._kernel_cache_backend is original


def test_cupy_13_needs_no_new_reader_because_it_reads_the_variable_every_compile(
        monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="13")
    report = sp.install_cupy_policy(sp.KEEP, cupy)
    assert report["cache_dir_read"] == {
        "reads": "every compile", "reasons": [],
        "mechanism": "cupy.cuda.compiler.get_cache_dir(), CuPy's own"}
    assert not hasattr(compiler, sp.CUPY_CACHE_BACKEND)


def test_keep_refuses_a_cupy_14_cache_backend_it_did_not_build_by_name(monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="14")

    class SharedCache:
        def load(self, name):
            return None

        def save(self, name, cubin, source):
            pass

    custom = SharedCache()
    compiler._kernel_cache_backend = custom
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    assert "CuPy's kernel cache backend is SharedCache, not CuPy's own " \
           "_DiskKernelCacheBackend" in str(excinfo.value)
    assert compiler._kernel_cache_backend is custom


def test_keep_refuses_by_name_when_the_compile_path_calls_no_known_front_end(
        monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, compiler, calls, _state = _fake_cupy(
        monkeypatch, tmp_path, shape="14", source=MOVED_SEAM_COMPILER)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    message = str(excinfo.value)
    assert "the -ftz=true strip cannot be installed where CuPy compiles" in message
    assert "_compile_with_cache_cuda calls no compiler front end" in message
    assert sp.executor_report("cupy")["installed"] is False
    for name in ("_nvrtc_build", "_nvcc_build", "compile_using_nvrtc",
                 "_compile_with_cache_cuda"):
        assert getattr(getattr(compiler, name), sp._POLICY_MARK, None) is None, name


def test_flush_names_the_front_end_it_could_not_guard(monkeypatch, tmp_path):
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    cupy, _compiler, _calls, _state = _fake_cupy(
        monkeypatch, tmp_path, accelerators=(), shape="14", source=MOVED_SEAM_COMPILER)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert ("the compiler front ends CuPy's compile path calls could not be found: "
            "cupy.cuda.compiler._compile_with_cache_cuda calls no compiler front end"
            in str(excinfo.value))


def test_keep_refuses_a_front_end_whose_options_it_cannot_find(monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    swapped = CUPY_14_COMPILER.replace(
        'def _compile_using_nvrtc_no_warning(source, options=(), arch=None, filename="kern.cu",',
        'def _compile_using_nvrtc_no_warning(options, source=(), arch=None, filename="kern.cu",')
    assert swapped != CUPY_14_COMPILER
    cupy, compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="14", source=swapped)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    assert ("cupy.cuda.compiler._compile_using_nvrtc_no_warning takes ['options', 'source'] "
            "first" in str(excinfo.value))
    assert getattr(compiler._compile_using_nvrtc_no_warning, sp._POLICY_MARK, None) is None


def test_keep_refuses_a_compile_path_whose_key_input_it_cannot_find(monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    renamed = CUPY_14_COMPILER.replace("extra_source", "key_source")
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="14", source=renamed)
    with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
        sp.install_cupy_policy(sp.KEEP, cupy)
    assert ("cupy.cuda.compiler._compile_with_cache_cuda has no extra_source parameter"
            in str(excinfo.value))
    assert sp.executor_report("cupy")["cache_key_folded"] is False


def test_a_second_policy_is_refused_at_cupy_14s_front_end_too(monkeypatch, tmp_path):
    cache = tmp_path / f"cupy_cache_{sp.CUPY_CACHE_POLICY_TOKEN}_test"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache))
    cupy, _compiler, _calls, _state = _fake_cupy(monkeypatch, tmp_path, shape="14")
    sp.install_cupy_policy(sp.KEEP, cupy)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    with pytest.raises(sp.SubnormalPolicyLocked) as excinfo:
        sp.install_cupy_policy(sp.FLUSH, cupy)
    assert "_compile_using_nvrtc_no_warning" in str(excinfo.value)


#: Run in a child process: the suite's own process may already hold a keep-policy
#: ``cp.multiply`` kernel, and the defect is about what a kernel compiled BEFORE the
#: install does after it. The CuPy cache directory is a fresh one, given either as a
#: plain directory (the install moves to a keep sibling) or as a keep directory
#: minted before the first compile, as every route-campaign leg sets it — so the
#: pre-install compile writes CuPy's own ``-ftz=true`` binary INTO the directory the
#: policy then uses.
_LATE_INSTALL_CHILD = r"""
import json
import sys
import numpy as np
import cupy as cp
from meep_gpu import subnormal_policy as sp

if sys.argv[1] == "keep_directory":
    sp.point_cupy_cache_at_keep_policy(cupy=cp)
x = cp.full(8, np.float32(1e-30), dtype=cp.float32)
y = cp.full(8, np.float32(1e-10), dtype=cp.float32)
before = float(cp.asnumpy(cp.multiply(x, y))[0])
cache = sp.point_cupy_cache_at_keep_policy(cupy=cp)
report = sp.install_cupy_policy("keep", cupy=cp)
after = float(cp.asnumpy(cp.multiply(x, y))[0])
operator = float(cp.asnumpy(x * y)[0])
print(json.dumps({"before": before, "after": after, "operator": operator,
                  "attained": report["attained"], "reasons": report["reasons"],
                  "memo": report["kernel_memo"], "cache": cache.get("after")},
                 default=str))
"""


@pytest.mark.requires_resource("cupy")
@pytest.mark.parametrize("directory", ["plain_directory", "keep_directory"])
def test_a_float32_ufunc_compiled_before_a_keep_install_keeps_subnormals_after_it(
        tmp_path, directory):
    """On device: the install reaches a ufunc kernel that was compiled before it.

    The shipped route's order — the lift compiles ``cupy_multiply__float32_float32_float32``
    under CuPy's own ``-ftz=true``, then dispatch moves ``CUPY_CACHE_DIR`` and installs
    keep — reproduced on one ``x * y``. Before the install 1e-30 * 1e-10 flushes to
    0.0 (the precondition: without it this test could not see the defect); after it
    the same signature must return the IEEE word 0x000116c2, 9.99995e-41, through
    ``cp.multiply`` and through the ``*`` operator, with no compiled kernel left
    alive by the drop. ``keep_directory`` is the configuration the route campaign
    runs in, where the pre-install binary sits in the policy's own directory; the
    memo drop alone measured 0 of 3 there, and the cache-key token is what reaches it.
    """
    if not backends.cupy_available():
        pytest.skip("CuPy is not installed or no CUDA device is visible")
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("MEEP_GPU_")}
    environment["CUPY_CACHE_DIR"] = str(tmp_path / "cupy_cache")
    root = str(Path(meep_gpu.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [root] + [p for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p])
    completed = subprocess.run([sys.executable, "-c", _LATE_INSTALL_CHILD, directory],
                               env=environment, cwd=str(tmp_path), capture_output=True,
                               text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = float(np.array([0x000116C2], dtype=np.uint32).view(np.float32)[0])
    assert result["before"] == 0.0, (
        f"precondition: a float32 product compiled before any install must flush "
        f"(CuPy appends -ftz=true to every NVRTC compile); it returned "
        f"{result['before']!r}, so this host cannot show the defect")
    assert result["attained"] is True, result["reasons"]
    assert result["after"] == expected, (
        f"cp.multiply compiled before the keep install returned {result['after']!r} "
        f"after it, not {expected!r}: the pre-install kernel is still reachable "
        f"({result['memo']})")
    assert result["operator"] == expected, result
    assert result["memo"]["walked"] is True, result["memo"]
    assert result["memo"]["entries_dropped"]["ufunc._kernel_memo"] >= 1, result["memo"]
    assert result["memo"]["live_compiled_kernels_after_drop"] == 0, result["memo"]


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

#: The shape Triton 3.1.0 ACTUALLY EMITS: no attribute group on the define, and a
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
                        lambda text, attribute=None: (
                            text.replace("!dbg !7 {", "!dbg !7 #9 {"),
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


#: What a simulated LLVM does with each denormal spelling. Measured on the real
#: ones (Triton's own ``ptx`` stage, the synthetic FMA module): 3.1.0 flushes under
#: the string attribute and ABORTS on ``denormal_fpenv``; 3.8.0 ignores the string
#: attribute and flushes under ``denormal_fpenv``. The other two worlds are the ones
#: the selection must survive: nothing flushes, and a later LLVM that ignores the
#: string attribute without reading the first-class one.
_LLVM_WORLDS = {
    "3.1": {"string": "flushes", "fpenv": "aborts"},
    "3.8": {"string": "ignored", "fpenv": "flushes"},
    "neither": {"string": "ignored", "fpenv": "ignored"},
    "no_fpenv": {"string": "ignored", "fpenv": "aborts"},
}

_FAKE_TRITON_FILE = "/fake/site-packages/triton/__init__.py"


def _spelling_in(text):
    """Which denormal spelling this LLVM IR text carries, if any."""
    if "denormal_fpenv" in text:
        return "fpenv"
    if '"denormal-fp-math-f32"' in text:
        return "string"
    return None


def _fake_lowering(text, world):
    """The PTX the simulated LLVM emits for an FMA in a module carrying ``text``'s spelling."""
    spelling = _spelling_in(text)
    ftz = spelling is not None and _LLVM_WORLDS[world].get(spelling) == "flushes"
    return (".version 8.3\n.target sm_80\n.visible .entry meep_gpu_denormal_probe(\n)\n{\n"
            f"\tfma.rn{'.ftz' if ftz else ''}.f32 \t%f4, %f1, %f2, %f3;\n\tret;\n}}\n")


def _fake_probe_runner(world, calls, *, file=_FAKE_TRITON_FILE, ptx=None, returncode=None):
    """Stands in for :func:`sp._run_denormal_probe`: what the probe CHILD would print
    and return on a simulated LLVM, with no process started. ``calls`` records the
    spelling of every module the probe was asked to measure."""
    def run(module_text):
        spelling = _spelling_in(module_text)
        calls.append(spelling)
        if world == "unmeasured":
            return {"returncode": 1, "stdout": "",
                    "stderr": "ModuleNotFoundError: No module named 'triton'"}
        started = sp._DENORMAL_PROBE_STARTED + json.dumps({"version": world, "file": file})
        if _LLVM_WORLDS[world].get(spelling) == "aborts":
            return {"returncode": -6, "stdout": started + "\n",
                    "stderr": "LLVM ERROR: failed to parse IR: unterminated attribute group"}
        body = _fake_lowering(module_text, world) if ptx is None else ptx
        result = sp._DENORMAL_PROBE_RESULT + json.dumps(
            {"version": world, "file": file, "ptx": body})
        return {"returncode": 0 if returncode is None else returncode,
                "stdout": started + "\n" + result + "\n", "stderr": ""}
    return run


class _FakeCUDABackend:
    """Enough of Triton's ``CUDABackend`` for the two hooks to be exercised.

    The ``llir`` stage returns the shape Triton 3.1.0 really emits, so a wrapper
    that only works on hand-written IR cannot pass here. ``llir_text`` is a class
    attribute so a test can hand the stage a different module — the flagless case
    is one the policy must handle without refusing a kernel. The ``ptx`` stage
    returns ``metadata["ptx"]`` when a test hands it PTX, and otherwise LOWERS its
    input the way the simulated LLVM of ``world`` would: ``.ftz`` only when the
    text carries a spelling that LLVM reads. So a wrapper that injects a spelling
    the LLVM ignores fails the audit here exactly as it does on the device.
    """

    llir_text = _LLIR_TRITON_SHAPE
    world = "3.1"

    def hash(self):
        return "nvidia-86-ptx83"

    def add_stages(self, stages, options):
        stages["llir"] = lambda src, metadata: type(self).llir_text
        stages["ptx"] = lambda src, metadata: (
            metadata["ptx"] if "ptx" in metadata else _fake_lowering(src, type(self).world))


def _install_fake_triton(monkeypatch, policy, world="3.1"):
    """A fake Triton on a simulated LLVM (``world``, see :data:`_LLVM_WORLDS`).

    The flush install's spelling probe runs in a child process on a real host;
    here it is answered by :func:`_fake_probe_runner` for the same world, and the
    spellings it was asked about land in ``backend.probe_calls``.
    """
    backend = _FakeCUDABackend
    calls = []
    monkeypatch.setattr(backend, "world", world)
    monkeypatch.setattr(backend, "probe_calls", calls, raising=False)
    monkeypatch.setattr(sp, "_run_denormal_probe", _fake_probe_runner(world, calls))
    monkeypatch.setattr(sp, "_cuda_backend_class", lambda: backend)
    # Only ``triton`` is faked present. Every other lookup goes to the real
    # ``find_spec``: blanket-faking it hid ``llvmlite`` and disarmed the very
    # verification these tests exist to exercise.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "triton" else real_find_spec(name))
    fake_triton = types.ModuleType("triton")
    fake_triton.__version__ = "3.1.0"
    fake_triton.__file__ = _FAKE_TRITON_FILE
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
# The flush spelling: measured per Triton, in a child process
# ---------------------------------------------------------------------------

#: The base tree's digests (git archive 57130c67 + the 1.0.1 series to this
#: point), computed once there. Keep's must never move, and flush's must not
#: move wherever the string attribute is the spelling selected.
_BASE_KEEP_DIGEST = "d2e88ea25a6c"
_BASE_FLUSH_DIGEST = "ac55eb37b54a"


def _restoring(backend):
    return backend.hash, backend.add_stages


def test_the_string_attribute_is_selected_where_llvm_reads_it_and_nothing_moves(monkeypatch):
    """Triton 3.1.0 through 3.7.1: the string attribute flushes, so it is selected,
    the second spelling is never handed to an LLVM (it aborts 3.1.0's), and the
    flush cache key is the one the base tree computed."""
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="3.1")
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        assert report["attained"] is True
        assert backend.probe_calls == ["string"]          # fpenv never measured
        record = report["denormal_spelling"]
        assert record["selected"] == sp.TRITON_DENORMAL_ATTRIBUTE
        assert [v["outcome"] for v in record["verdicts"]] == ["flushes"]
        assert sp.mechanism_digest(sp.FLUSH) == _BASE_FLUSH_DIGEST
        assert backend().hash().endswith(f"flush-{_BASE_FLUSH_DIGEST}")
        stages = {}
        backend().add_stages(stages, options=None)
        emitted = stages["llir"]("src", {})
        assert '"denormal-fp-math-f32"="preserve-sign,preserve-sign"' in emitted
        assert "denormal_fpenv" not in emitted
        assert "fma.rn.ftz.f32" in stages["ptx"](emitted, {})   # audited, passes
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_denormal_fpenv_is_selected_where_llvm_ignores_the_string_and_is_what_compiles(monkeypatch):
    """Triton 3.8.0: the string attribute is read and IGNORED (``fma.rn.f32``), so
    the first-class spelling is measured next, flushes, and is selected — and it is
    the spelling every compile then carries. A hook that measured it but still
    injected the string attribute fails the audit on this simulated LLVM exactly
    as on the device."""
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="3.8")
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        assert report["attained"] is True
        assert backend.probe_calls == ["string", "fpenv"]
        record = report["denormal_spelling"]
        assert record["selected"] == sp.TRITON_DENORMAL_FPENV_ATTRIBUTE
        assert [v["outcome"] for v in record["verdicts"]] == ["ignored", "flushes"]
        assert record["verdicts"][0]["fma"] == ["fma.rn.f32"]
        assert record["verdicts"][1]["fma"] == ["fma.rn.ftz.f32"]
        assert sp.TRITON_DENORMAL_FPENV_ATTRIBUTE in report["mechanism"]
        # The key carries the selected spelling, and differs from the string's.
        fpenv_digest = sp.mechanism_digest(sp.FLUSH, sp.TRITON_DENORMAL_FPENV_ATTRIBUTE)
        assert fpenv_digest != _BASE_FLUSH_DIGEST
        assert backend().hash() == (
            f"nvidia-86-ptx83-{sp.TRITON_HASH_PREFIX}flush-{fpenv_digest}")
        stages = {}
        backend().add_stages(stages, options=None)
        emitted = stages["llir"]("src", {})
        assert "denormal_fpenv(float: preservesign)" in emitted
        assert "denormal-fp-math-f32" not in emitted
        ptx = stages["ptx"](emitted, {})                  # lowered by the simulated LLVM
        assert "fma.rn.ftz.f32" in ptx
        assert sp.triton_counters()["ptx_violations"] == 0
        assert sp.policy_stamp()["executors"]["triton"]["denormal_spelling"]["selected"] == (
            sp.TRITON_DENORMAL_FPENV_ATTRIBUTE)
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_when_no_spelling_flushes_the_audit_still_refuses_by_name(monkeypatch):
    """Neither spelling flushes: nothing is selected, the string attribute is
    injected as before, the install is not turned into a raise (a caller's flush
    install keeps working and the CUDA table serves), and every Triton compile is
    refused by the audit — naming both measured spellings."""
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="neither")
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        assert report["attained"] is True and report["installed"] is True
        record = report["denormal_spelling"]
        assert record["selected"] is None
        assert record["attribute"] == sp.TRITON_DENORMAL_ATTRIBUTE
        assert [v["outcome"] for v in record["verdicts"]] == ["ignored", "ignored"]
        assert backend().hash().endswith(f"flush-{_BASE_FLUSH_DIGEST}")
        stages = {}
        backend().add_stages(stages, options=None)
        emitted = stages["llir"]("src", {})
        assert '"denormal-fp-math-f32"' in emitted
        with pytest.raises(sp.SubnormalPolicyUnattainable) as excinfo:
            stages["ptx"](emitted, {})
        message = str(excinfo.value)
        assert "Denormal spelling probe" in message
        assert sp.TRITON_DENORMAL_ATTRIBUTE in message
        assert sp.TRITON_DENORMAL_FPENV_ATTRIBUTE in message
        assert message.count("ignored") >= 2
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_a_spelling_the_llvm_cannot_parse_is_an_aborted_child_never_a_selection(monkeypatch):
    """An LLVM that ignores the string attribute AND cannot parse ``denormal_fpenv``
    aborts the probe child. The parent reads that as "aborted", selects nothing,
    and is still alive to say so."""
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="no_fpenv")
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        verdicts = report["denormal_spelling"]["verdicts"]
        assert [v["outcome"] for v in verdicts] == ["ignored", "aborted"]
        assert verdicts[1]["returncode"] == -6
        assert "unterminated attribute group" in verdicts[1]["detail"]
        assert report["denormal_spelling"]["selected"] is None
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_an_unmeasured_probe_never_reaches_the_second_spelling(monkeypatch):
    """If the child cannot even measure the string attribute, nothing says this
    LLVM can parse ``denormal_fpenv``, so it is never measured: the string
    attribute is injected exactly as before and the audit decides."""
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="3.1")
    calls = []
    monkeypatch.setattr(sp, "_run_denormal_probe", _fake_probe_runner("unmeasured", calls))
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.FLUSH)
        assert calls == ["string"]
        record = report["denormal_spelling"]
        assert [v["outcome"] for v in record["verdicts"]] == ["unmeasured"]
        assert record["selected"] is None
        assert record["attribute"] == sp.TRITON_DENORMAL_ATTRIBUTE
        assert "No module named 'triton'" in record["verdicts"][0]["detail"]
        assert report["attained"] is True
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


@pytest.mark.parametrize("how", ["no_fma", "other_triton", "died_after_result"])
def test_a_probe_that_did_not_measure_an_fma_on_this_triton_selects_nothing(monkeypatch, how):
    """Three ways a probe can return "no violations" without having measured
    anything: PTX with no f32 FMA in it (an empty audit has no violations), PTX
    from a different Triton than this process imports, and a child that printed a
    result and then died. Each is "unmeasured", and none is a selection."""
    _install_fake_triton(monkeypatch, sp.FLUSH, world="3.8")
    calls = []
    runner = {
        "no_fma": _fake_probe_runner("3.8", calls, ptx=".visible .entry k()\n{\n\tret;\n}\n"),
        "other_triton": _fake_probe_runner("3.8", calls, file="/elsewhere/triton/__init__.py"),
        "died_after_result": _fake_probe_runner("3.8", calls, returncode=-11),
    }[how]
    monkeypatch.setattr(sp, "_run_denormal_probe", runner)
    record = sp.select_denormal_spelling()
    assert [v["outcome"] for v in record["verdicts"]] == ["unmeasured"], record
    assert record["selected"] is None
    assert calls == ["string"]


def test_a_probe_verdict_carries_no_path_from_the_child(monkeypatch):
    """A verdict lands in the policy stamp, and the stamp in gate artifacts: the
    child's message is kept, the host's paths in it are not."""
    _install_fake_triton(monkeypatch, sp.FLUSH, world="3.8")

    def failed(module_text):
        return {"returncode": 1, "stdout": "", "stderr": (
            'Traceback (most recent call last):\n  File "/opt/probe-env/lib/x.py", line 1\n'
            "FileNotFoundError: [Errno 2] No such file or directory: '/opt/probe-env/bin/ptxas'")}

    monkeypatch.setattr(sp, "_run_denormal_probe", failed)
    verdict = sp.measure_denormal_spelling(sp.TRITON_DENORMAL_ATTRIBUTE)
    assert verdict["outcome"] == "unmeasured"
    assert "FileNotFoundError" in verdict["detail"] and "<path>" in verdict["detail"]
    assert "/opt" not in json.dumps(verdict) and "probe-env" not in json.dumps(verdict)
    calls = []
    monkeypatch.setattr(sp, "_run_denormal_probe", _fake_probe_runner(
        "3.8", calls, file="/opt/probe-env/elsewhere/triton/__init__.py"))
    elsewhere = sp.measure_denormal_spelling(sp.TRITON_DENORMAL_ATTRIBUTE)
    assert elsewhere["outcome"] == "unmeasured"
    assert "probe-env" not in json.dumps(elsewhere)


def test_keep_measures_no_spelling_and_its_report_and_key_are_unchanged(monkeypatch):
    """Keep injects no attribute, so it must not pay for (or depend on) the probe:
    a probe runner that raises is never reached, keep's report carries the same
    keys as before, and keep's cache key is the base tree's."""
    backend = _install_fake_triton(monkeypatch, sp.KEEP)

    def must_not_run(module_text):
        raise AssertionError("the spelling probe ran under keep")

    monkeypatch.setattr(sp, "_run_denormal_probe", must_not_run)
    original_hash, original_stages = _restoring(backend)
    try:
        report = sp.install_triton_policy(sp.KEEP)
        assert report["attained"] is True
        assert "denormal_spelling" not in report
        assert sorted(report) == sorted([
            "executor", "requested", "attained", "installed", "triton_version",
            "cache_dir", "hash_folded", "verification", "in_process_caches_dropped",
            "exceptions", "mechanism", "reasons"])
        assert sp.mechanism_digest(sp.KEEP) == _BASE_KEEP_DIGEST
        assert sp.mechanism_digest(sp.KEEP, None) == _BASE_KEEP_DIGEST
        assert backend().hash().endswith(f"keep-{_BASE_KEEP_DIGEST}")
        assert "_triton_denormal" not in sp._STATE
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_the_spelling_is_measured_once_per_process(monkeypatch):
    backend = _install_fake_triton(monkeypatch, sp.FLUSH, world="3.8")
    original_hash, original_stages = _restoring(backend)
    try:
        first = sp.select_denormal_spelling()
        assert sp.select_denormal_spelling() is first
        sp.install_triton_policy(sp.FLUSH)
        sp.install_triton_policy(sp.FLUSH)
        assert backend.probe_calls == ["string", "fpenv"]
        sp._reset_for_tests()                       # an uninstall forgets it
        assert "_triton_denormal" not in sp._STATE
        sp.select_denormal_spelling()
        assert backend.probe_calls == ["string", "fpenv", "string", "fpenv"]
    finally:
        backend.hash, backend.add_stages = original_hash, original_stages


def test_denormal_fpenv_is_injected_once_and_composes_with_the_string_attribute():
    """Idempotent by the attribute's own name: a second injection writes nothing,
    and a group that already carries the string attribute still gains the
    first-class one (the string form alone is ignored where fpenv is selected)."""
    fpenv = sp.TRITON_DENORMAL_FPENV_ATTRIBUTE
    once, counts = sp.inject_denormal_attribute(_LLIR_TRITON_SHAPE, attribute=fpenv)
    assert counts["groups_minted"] == 1
    twice, again = sp.inject_denormal_attribute(once, attribute=fpenv)
    assert twice == once and again["groups_extended"] == 0 and again["groups_minted"] == 0
    assert once.count("denormal_fpenv") == 1
    both, extended = sp.inject_denormal_attribute(
        sp.inject_denormal_attribute(_LLIR_TRITON_SHAPE)[0], attribute=fpenv)
    assert extended["groups_extended"] == 1
    group = [line for line in both.splitlines() if line.startswith("attributes #0")][0]
    assert '"denormal-fp-math-f32"' in group and group.count("denormal_fpenv") == 1
    define = [line for line in once.splitlines() if line.startswith("define ")][0]
    assert define.index("#0") < define.index("!dbg")       # the same placement


def test_the_probe_module_lands_the_attribute_where_llvm_accepts_it():
    """The probe module gets its group from the injector, so it exercises the
    placement every kernel gets; with the string spelling it parses under
    llvmlite's LLVM too, a third LLVM beside Triton 3.1.0's and 3.8.0's."""
    text, counts = sp.inject_denormal_attribute(sp.TRITON_DENORMAL_PROBE_LLIR)
    assert counts == {"defines": 1, "groups_extended": 0, "groups_minted": 1}
    assert "@meep_gpu_denormal_probe(ptr addrspace(1) %0, float %1, float %2, float %3) #0 {" in text
    assert "@llvm.fma.f32" in text
    assert _llvm_rejects(text) is None


def test_verification_never_hands_denormal_fpenv_to_llvmlite():
    """llvmlite carries its own LLVM, which can predate ``denormal_fpenv`` — 0.47.0's
    LLVM 20.1.8 rejects it, measured — so verifying the selected spelling with it
    would refuse flush on Triton 3.8.0 on every host that has numba installed. The
    parse check runs on the string spelling (the grammar slot is the same); the
    placement check runs on the selected one."""
    fpenv = sp.TRITON_DENORMAL_FPENV_ATTRIBUTE
    verification = sp.verify_injection(attribute=fpenv)
    assert verification["ok"] is True, verification
    placement = verification["checks"][0]
    assert placement["ok"] is True and "#0" in placement["define"]
    if sp._llvm_parser() is not None:
        assert verification["verified_by"].startswith("llvmlite parse + placement")
        assert "spelling probe" in verification["verified_by"]
        error = _llvm_rejects(sp.inject_denormal_attribute(sp._PARSEABLE_SHAPE, attribute=fpenv)[0])
        if error is None:
            pytest.skip("this llvmlite's LLVM reads denormal_fpenv; the hazard is absent here")
        assert "attribute group" in error


#: A stand-in Triton package for the probe CHILD: the module paths the child
#: imports, a ``ptx`` stage that lowers by spelling — and that calls ``os.abort()``
#: on a spelling its "LLVM" cannot parse, as Triton 3.1.0's does through
#: ``report_fatal_error``. ``FAKE_TRITON_WORLD`` picks the LLVM: ``string`` reads
#: the string attribute only (and has 3.1.0's two-argument ``add_stages``),
#: ``fpenv`` reads ``denormal_fpenv`` only (3.8.0's three-argument one).
_FAKE_TRITON_PACKAGE = {
    "triton/__init__.py": '__version__ = "0.0+fake"\n',
    "triton/backends/__init__.py": "",
    "triton/backends/compiler.py": (
        "class GPUTarget:\n"
        "    def __init__(self, backend, arch, warp_size):\n"
        "        self.backend, self.arch, self.warp_size = backend, arch, warp_size\n"),
    "triton/compiler/__init__.py": "",
    "triton/compiler/compiler.py": (
        "import os\n"
        "class Language:\n"
        "    TRITON = 'triton'\n"
        "def lower(src):\n"
        "    world = os.environ['FAKE_TRITON_WORLD']\n"
        "    if 'denormal_fpenv' in src and world == 'string':\n"
        "        os.abort()\n"
        "    ftz = ('denormal_fpenv' in src and world == 'fpenv') or (\n"
        "        '\"denormal-fp-math-f32\"' in src and world == 'string')\n"
        "    return ('.visible .entry meep_gpu_denormal_probe()\\n{\\n\\tfma.rn'\n"
        "            + ('.ftz' if ftz else '') + '.f32 \\t%f4, %f1, %f2, %f3;\\n}\\n')\n"
        "class Backend:\n"
        "    def __init__(self, target):\n"
        "        self.target = target\n"
        "    def parse_options(self, options):\n"
        "        return dict(options)\n"
        "    def add_stages(self, stages, options, *language):\n"
        "        if (len(language) == 1) != (os.environ['FAKE_TRITON_WORLD'] == 'fpenv'):\n"
        "            raise TypeError('add_stages() takes a different number of arguments')\n"
        "        stages['ptx'] = lambda src, metadata: lower(src)\n"
        "def make_backend(target):\n"
        "    return Backend(target)\n"),
    "triton/_C/__init__.py": "",
    "triton/_C/libtriton.py": (
        "class llvm:\n"
        "    @staticmethod\n"
        "    def init_targets():\n"
        "        pass\n"),
}


def _fake_triton_on_disk(monkeypatch, tmp_path, world):
    site = tmp_path / "site"
    for relative, text in _FAKE_TRITON_PACKAGE.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(site)] + [entry for entry in [os.environ.get("PYTHONPATH")] if entry]))
    monkeypatch.setenv("FAKE_TRITON_WORLD", world)
    spec = importlib.util.spec_from_file_location(
        "triton", site / "triton" / "__init__.py",
        submodule_search_locations=[str(site / "triton")])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "triton", module)
    return module


def test_the_real_probe_child_survives_an_llvm_abort_and_selects_by_measurement(monkeypatch, tmp_path):
    """The child process itself, end to end, on a stand-in Triton on disk: an
    abort inside the child's "LLVM" is read as "aborted" (and the parent lives),
    an "LLVM" that reads the string attribute selects it without ever being
    handed ``denormal_fpenv``, and one that reads only ``denormal_fpenv`` selects
    that — through both ``add_stages`` arities."""
    if sys.flags.ignore_environment:
        pytest.skip("the child is started with -E like this interpreter, so PYTHONPATH cannot reach it")
    _fake_triton_on_disk(monkeypatch, tmp_path, "string")
    aborted = sp.measure_denormal_spelling(sp.TRITON_DENORMAL_FPENV_ATTRIBUTE)
    assert aborted["outcome"] == "aborted", aborted
    assert aborted["returncode"] not in (0, None)
    record = sp.select_denormal_spelling()
    assert [v["outcome"] for v in record["verdicts"]] == ["flushes"], record
    assert record["selected"] == sp.TRITON_DENORMAL_ATTRIBUTE

    sp._reset_for_tests()
    _fake_triton_on_disk(monkeypatch, tmp_path, "fpenv")
    record = sp.select_denormal_spelling()
    assert [v["outcome"] for v in record["verdicts"]] == ["ignored", "flushes"], record
    assert record["selected"] == sp.TRITON_DENORMAL_FPENV_ATTRIBUTE
    assert record["verdicts"][1]["triton"] == "0.0+fake"


# ---------------------------------------------------------------------------
# The artifact stamp
# ---------------------------------------------------------------------------


def test_the_stamp_carries_the_question_the_answer_and_the_measurement(monkeypatch, tmp_path):
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
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
    fallback = sp.install_subnormal_policy(strict=True)["match_meep"]
    assert fallback["measured"] is False and fallback["fallback"] == sp.KEEP
    assert "could NOT be measured" in fallback["why"]


def test_an_uninstalled_stamp_carries_no_certification_grade_name(monkeypatch, tmp_path):
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
    _keep_policy_cupy_cache(monkeypatch, tmp_path)
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
# Why both arrived together. The Metal kernel table can be certified under exactly
# one float32 subnormal policy — MPS flushes denormals natively and exposes no
# lever, so ``keep`` is unattainable on that executor — and Apple silicon is where
# MPS lives. Without a host lever the two halves of a Metal run would be
# permanently unable to agree, which is not a policy split this module may install:
# it is a refusal. The fenv lever is what makes ``flush`` attainable on the host
# half, and the ``mps`` arm is what makes the device half report rather than assume.


def test_the_fenv_lever_is_tried_only_after_meeps_knob_and_only_if_it_is_needed(
        monkeypatch):
    """MEEP's knob first, always — on x86 nothing about this changes.

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
    """The third state that belongs to nobody, and the reason the restore is unconditional.

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


# ---------------------------------------------------------------------------
# The x86-64 glibc MXCSR lever: flush and keep on Linux without MEEP
# ---------------------------------------------------------------------------
#
# On x86 MEEP's own switch is the mechanism whenever MEEP is imported. A process
# that does not import MEEP (the Tier A replay of MEEP identity blocks it on
# purpose) had no lever that reached ``flush`` on Linux, because glibc exports no
# environment object with the denormal bits set. The lever sets or clears MXCSR
# FTZ|DAZ through ``fegetenv``/``fesetenv`` on a mirror of glibc's x86_64
# ``fenv_t``. The tests that need the real FPU run on an x86-64 glibc host only;
# the ones that pin the ladder, the layout and the guards run everywhere.

#: glibc's x86_64 ``fenv_t`` as the compiler reads ``bits/fenv.h`` on the Linux
#: reference host (glibc 2.31, GCC 9.4): a program printing ``sizeof`` and
#: ``offsetof`` reads 32, 28 (``__mxcsr``) and 20 (``__data_offset``).
_GLIBC_X86_64_FENV_T = {"sizeof": 32, "mxcsr": 28, "data_offset": 20}


def _require_x86_64_glibc():
    applies, abi = sp.glibc_x86_64_abi()
    if not applies:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("x86_64_glibc_host", f"the MXCSR lever's real-FPU "
                                                    f"tests need an x86-64 glibc host: {abi}")


def _thread_mxcsr():
    """This thread's MXCSR, read through ``fegetenv`` on the mirrored structure."""
    library = backends._math_library()  # noqa: SLF001
    buffer = sp.ctypes.create_string_buffer(backends._FENV_BYTES)  # noqa: SLF001
    assert library.fegetenv(sp.ctypes.addressof(buffer)) == 0
    return int(sp._GlibcX8664Fenv.from_buffer(buffer).mxcsr)  # noqa: SLF001


def _meep_not_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "meep", raising=False)


def test_the_mirrored_fenv_t_is_glibcs_x86_64_header_layout():
    """The structure the lever writes into, against the header's own numbers."""
    layout = sp._GlibcX8664Fenv  # noqa: SLF001
    assert sp.ctypes.sizeof(layout) == _GLIBC_X86_64_FENV_T["sizeof"]
    assert layout.mxcsr.offset == _GLIBC_X86_64_FENV_T["mxcsr"]
    assert layout.data_offset.offset == _GLIBC_X86_64_FENV_T["data_offset"]
    assert sp.ctypes.sizeof(layout) <= backends._FENV_BYTES  # noqa: SLF001
    assert sp.MXCSR_FTZ_DAZ == 0x8040


def test_the_mxcsr_lever_drives_flush_and_keep_without_meep_and_measures_both(monkeypatch):
    """x86-64 glibc, MEEP not imported: both policies attain, by the MXCSR lever.

    The verdict is the measurement: the install reads ``backends.subnormals_flushed``
    and the test reads all four probes (FTZ and DAZ, float32 and float64) through
    ``meep_identity.subnormal_state``, plus the MXCSR word itself, whose other
    control bits (exception masks, rounding) must come back as they were.
    """
    _require_x86_64_glibc()
    _meep_not_imported(monkeypatch)
    from meep_gpu import meep_identity  # noqa: PLC0415

    assert sp.install_host_policy(sp.KEEP, strict=True)["attained"] is True
    entry = _thread_mxcsr()
    flush = sp.install_host_policy(sp.FLUSH, strict=True)
    assert flush["attained"] is True and flush["flushing_after"] is True, flush
    assert flush["mechanism"] == sp._MXCSR_MECHANISM[True]  # noqa: SLF001
    assert flush["installed"] is True, "the lever moved the FPU and the record must say so"
    assert meep_identity.subnormal_state()["mode"] == "flush"
    assert _thread_mxcsr() & 0x8040 == 0x8040
    keep = sp.install_host_policy(sp.KEEP, strict=True)
    assert keep["attained"] is True and keep["flushing_before"] is True, keep
    assert keep["mechanism"] == sp._MXCSR_MECHANISM[False]  # noqa: SLF001
    assert meep_identity.subnormal_state()["mode"] == "keep"
    word = _thread_mxcsr()
    assert word & 0x8040 == 0
    assert word & 0x7F80 == entry & 0x7F80, (hex(entry), hex(word))


def test_the_mxcsr_lever_reaches_the_calling_thread_only(monkeypatch):
    """MXCSR is per thread: a thread already running keeps its own state.

    MEEP's switch reaches every OpenMP thread through a parallel loop; this lever
    reaches the thread that calls it, which is the thread that steps the array
    path. A thread created after the call starts from a copy of its creator's state.
    """
    _require_x86_64_glibc()
    _meep_not_imported(monkeypatch)
    import threading  # noqa: PLC0415

    from meep_gpu import meep_identity  # noqa: PLC0415

    seen = {}
    go, done = threading.Event(), threading.Event()

    def running_before():
        go.wait(30)
        seen["before"] = meep_identity.subnormal_state()["mode"]
        done.set()

    def created_after():
        seen["after"] = meep_identity.subnormal_state()["mode"]

    assert sp.install_host_policy(sp.KEEP, strict=True)["attained"] is True
    early = threading.Thread(target=running_before)
    early.start()
    assert sp.install_host_policy(sp.FLUSH, strict=True)["attained"] is True
    go.set()
    assert done.wait(30)
    early.join(30)
    late = threading.Thread(target=created_after)
    late.start()
    late.join(30)
    assert meep_identity.subnormal_state()["mode"] == "flush"
    assert seen == {"before": "keep", "after": "flush"}, seen


def test_the_mxcsr_lever_restores_the_entry_word_when_the_measurement_disagrees(monkeypatch):
    """x86-64 glibc, real FPU: the candidate goes in, the stand-in measurement says
    no, and the entry environment is back — FTZ|DAZ as they were on entry."""
    _require_x86_64_glibc()
    _meep_not_imported(monkeypatch)
    from meep_gpu import meep_identity  # noqa: PLC0415

    entry_mode = meep_identity.subnormal_state()["mode"]
    entry = _thread_mxcsr()
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: entry & 0x8040 == 0x8040)
    assert sp._drive_host_fpu_by_mxcsr(entry & 0x8040 != 0x8040) is False  # noqa: SLF001
    assert "restored" in sp._MXCSR_LEVER_LAST["outcome"]  # noqa: SLF001
    assert _thread_mxcsr() & 0xFFC0 == entry & 0xFFC0
    assert meep_identity.subnormal_state()["mode"] == entry_mode


def _fake_glibc(monkeypatch, word):
    """An x86-64 glibc stand-in: ``fegetenv`` writes ``word`` at the mirrored
    ``__mxcsr`` offset, and ``fesetenv`` records the word each call installs."""
    installed = []

    class _Library:
        def fegetenv(self, address):
            sp.ctypes.memset(address, 0, sp.ctypes.sizeof(sp._GlibcX8664Fenv))  # noqa: SLF001
            sp.ctypes.c_uint.from_address(address + _GLIBC_X86_64_FENV_T["mxcsr"]).value = word
            return 0

        def fesetenv(self, address):
            installed.append(sp.ctypes.c_uint.from_address(
                address + _GLIBC_X86_64_FENV_T["mxcsr"]).value)
            return 0

    monkeypatch.setattr(sp, "glibc_x86_64_abi", lambda: (True, "a stand-in x86-64 glibc"))
    monkeypatch.setattr(sp.backends, "_math_library", lambda: _Library())
    return installed


def test_the_mxcsr_lever_puts_the_saved_word_back_when_the_measurement_disagrees(monkeypatch):
    """Every host: the candidate is the entry word with FTZ|DAZ set, and the second
    ``fesetenv`` installs the entry word itself."""
    installed = _fake_glibc(monkeypatch, 0x1F80)
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: False)
    assert sp._drive_host_fpu_by_mxcsr(True) is False  # noqa: SLF001
    assert installed == [0x9FC0, 0x1F80], [hex(w) for w in installed]
    installed.clear()
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: True)
    assert sp._drive_host_fpu_by_mxcsr(True) is True  # noqa: SLF001
    assert installed == [0x9FC0]


def test_the_mxcsr_lever_writes_nothing_when_the_word_is_not_an_mxcsr(monkeypatch):
    """Reserved bits set means the offset is not this C library's ``__mxcsr``:
    ``ldmxcsr`` would fault, so nothing is installed and the lever refuses."""
    installed = _fake_glibc(monkeypatch, 0xDEAD1F80)
    assert sp._drive_host_fpu_by_mxcsr(True) is False  # noqa: SLF001
    assert installed == []
    assert "reserved bits" in sp._MXCSR_LEVER_LAST["outcome"]  # noqa: SLF001


def test_a_meep_less_flush_on_an_abi_with_no_lever_refuses_by_name(monkeypatch):
    """Neither MEEP-less lever applies: ``flush`` refuses, naming the ABI; ``keep``,
    which the host already does, still attains."""
    _meep_not_imported(monkeypatch)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lambda want_flush: False)
    monkeypatch.setattr(sp, "glibc_x86_64_abi",
                        lambda: (False, "this process is a stand-in ABI with no lever"))
    if backends.subnormals_flushed():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("host_fpu_mode", "the test needs a host that keeps at entry")
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        report = sp.install_host_policy(sp.FLUSH, strict=False)
    assert report["attained"] is False and report["installed"] is False, report
    assert "a stand-in ABI with no lever" in report["reasons"][0]
    assert "a stand-in ABI with no lever" in report["mechanism"]
    with pytest.raises(sp.SubnormalPolicyUnattainable, match="stand-in ABI"):
        sp.install_host_policy(sp.FLUSH, strict=True)
    assert sp.install_host_policy(sp.KEEP, strict=True)["attained"] is True


def test_the_mxcsr_lever_is_not_tried_when_meep_is_imported(monkeypatch):
    """With MEEP imported its switch is the mechanism; a thread it did not move is
    refused, never moved behind MEEP's back by the MEEP-less lever."""
    calls = []
    _module, state = _fake_meep(monkeypatch, effective=False)
    state["flushing"] = False  # a host that keeps, whatever this process entered in
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lambda want_flush: False)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_mxcsr",
                        lambda want_flush: calls.append(want_flush) or True)
    with pytest.warns(RuntimeWarning, match="UNATTAINABLE"):
        report = sp.install_host_policy(sp.FLUSH, strict=False)
    assert report["attained"] is False
    assert calls == [], "the MEEP-less lever ran in a process that imported MEEP"


def test_a_meep_less_install_already_in_the_requested_state_writes_nothing(monkeypatch):
    """Keep on a host that keeps, no MEEP: no lever runs and the record is the one a
    keep install made before the MXCSR lever existed (no mechanism, nothing installed)."""
    _meep_not_imported(monkeypatch)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_fenv", lambda want_flush: False)
    monkeypatch.setattr(sp, "_drive_host_fpu_by_mxcsr",
                        lambda want_flush: pytest.fail("the MXCSR lever ran with nothing to move"))
    monkeypatch.setattr(sp.backends, "subnormals_flushed", lambda: False)
    report = sp.install_host_policy(sp.KEEP, strict=True)
    assert report["attained"] is True and report["installed"] is False, report
    assert report["mechanism"] == "none (MEEP is not imported in this process)"


def test_the_mxcsr_lever_declines_off_its_abi_without_touching_the_c_library(monkeypatch):
    """Off x86-64 glibc the lever returns False before resolving any symbol."""
    monkeypatch.setattr(sp, "glibc_x86_64_abi", lambda: (False, "not the mirrored ABI"))
    monkeypatch.setattr(sp.backends, "_math_library",
                        lambda: pytest.fail("the lever resolved the C library off its ABI"))
    assert sp._drive_host_fpu_by_mxcsr(True) is False  # noqa: SLF001
    assert sp._MXCSR_LEVER_LAST["outcome"] == "not run: not the mirrored ABI"  # noqa: SLF001


def test_the_mps_executor_is_asked_only_when_it_is_named(monkeypatch):
    """``executors`` names the arms to drive and ``mps`` is not in the default tuple.

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

    The one-policy rule is unchanged here; only the VALUE is this table's. Under
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
