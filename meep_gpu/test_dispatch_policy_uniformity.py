"""The uniformity rule, measured on plans where CuPy and Triton are really governed.

WHY THIS FILE EXISTS, and it is not "more coverage of the same thing".
``test_dispatch_subnormal_gate`` pins that the shipped opt-in path installs the
certified policy or refuses by name. It did so, for its whole life, with
``governed_executors == ['host']`` on every test where the gate PERMITTED
dispatch — the two device executors whose disagreement IS the defect were absent
from the governed set in exactly the tests that dispatched, and present only in
the two negative controls that refuse. The headline test therefore proved the
HOST FPU had been driven and nothing at all about CuPy and Triton: the same shape
of gap as the "the subnormal policy question is settled" claim that this whole
round exists to correct.

The cause was structural. The gate decided governance with
``importlib.util.find_spec`` and read ANY exception from it as "absent" — and
``find_spec`` raises ``ValueError`` on a module object whose ``__spec__`` is
``None``, which a bare ``types.ModuleType`` (the file's own Triton stub) has. An
executor dropped from ``governed`` can never trip the ``ungoverned`` refusal,
because that list is built OVER ``governed``; so the disagreement was FAIL-OPEN
and the whole suite ran inside it.

So the doubles here are GOVERNABLE: the real ``install_triton_policy`` is driven
onto :func:`governable_triton`'s backend class, wraps its ``add_stages`` and
``hash``, and reports ``attained`` from its own measurement. A test that dispatches
here has genuinely put a Triton under the policy. The CuPy leg answers honestly
that a laptop with no CuPy compiles nothing — which is a true "nothing to govern",
not a dropped executor, and the artifact now says which.

NOTHING HERE LAUNCHES A KERNEL. The device half is
``parity/meep_gpu/gate_dispatch_end_to_end.py`` and ``probe_ftz_ptx_census.py``.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.subnormal_policy as policy_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.launch import TritonStepPlan


# ---------------------------------------------------------------------------
# Doubles that can actually be governed
# ---------------------------------------------------------------------------


def governable_triton(monkeypatch, version: str = "3.1.0"):
    """A Triton double the REAL ``install_triton_policy`` can be driven onto.

    A FRESH backend class per call, because the install wraps ``add_stages`` and
    ``hash`` on the CLASS: a shared one would carry one test's wrappers into the
    next. ``_cuda_backend_class`` is patched rather than a whole
    ``triton.backends`` package faked, for the reason the policy module's own
    tests give — a blanket fake hides ``llvmlite`` and disarms the verification
    the install refuses on.

    Returns the backend class, so a test can assert the hooks landed on it.
    """

    class _CUDABackend:
        """Enough of Triton's ``CUDABackend`` for both hooks to be exercised."""

        def hash(self):
            return "nvidia-86-ptx83"

        def add_stages(self, stages, options):
            stages["llir"] = lambda src, metadata: metadata.get("llir", "")
            stages["ptx"] = lambda src, metadata: metadata["ptx"]

    monkeypatch.setattr(policy_module, "_cuda_backend_class", lambda: _CUDABackend)
    module = types.ModuleType("triton")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "triton", module)
    return _CUDABackend


def cupy_like_grid():
    xp = types.SimpleNamespace()
    xp.__name__ = "cupy"
    xp.__version__ = "13.5.1"
    return types.SimpleNamespace(xp=xp)


class CountingPlan:
    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, drive=None, guard=None):
        self.runs += 1

    def warm(self):
        pass


def step_plan(plans=None, selected=None, reasons=None):
    plans = dict(plans or {})
    selected = dict(selected or {})
    reasons = dict(reasons or {})
    for slot in launch_module.STEP_ORDER:
        if slot not in plans and slot not in reasons:
            reasons[slot] = (f"no consulted product admitted {slot}",)
    return TritonStepPlan(plans, [], reasons, selected)


def plan_on_a_governed_host(monkeypatch, plan=None, fields=None):
    """One configuration freeze with a real, governable Triton behind rung 4."""
    governable_triton(monkeypatch)
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", lambda f, p, **kwargs: (
        plan if plan is not None else step_plan({"step_B": CountingPlan("b")},
                                                {"step_B": "PML"})))
    return fastpath.plan_fast_path(fields or object(), object(), cupy_like_grid())


def gate_of(record):
    return record["subnormal"]["gate"]


@pytest.fixture(autouse=True)
def _clean_process(monkeypatch, tmp_path):
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    monkeypatch.delenv(policy_module.POLICY_ENV, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    # A cache directory per test, so the provenance marker one test writes cannot
    # decide the next one's verdict and no test touches the real ~/.cupy tree.
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    yield
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None


# ---------------------------------------------------------------------------
# The gap this file was written for: a DISPATCHING plan with the device
# executors in the governed set
# ---------------------------------------------------------------------------


def test_a_dispatching_plan_governs_triton_and_says_it_drove_it(monkeypatch):
    """THE HEADLINE, and the one the old headline could not make.

    A plan that DISPATCHES, with Triton in ``governed_executors`` and its install
    report showing the policy was really driven onto the backend class — not a
    process where the executor was quietly dropped and the host alone was moved.
    """
    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    gate = gate_of(plan.report())
    assert "triton" in gate["governed_executors"], gate["governed_executors"]
    assert gate["attained"]["triton"] is True
    report = policy_module.executor_report("triton")
    assert report["installed"] is True and report["attained"] is True
    # THE HOOKS LANDED ON THE CLASS THE COMPILER WOULD INSTANTIATE, which is the
    # difference between "the install returned" and "Triton is under the policy":
    # the mechanism digest in the cache key is what separates this policy's
    # binaries from the other one's on disk.
    backend = policy_module._cuda_backend_class()
    assert policy_module.mechanism_digest("keep") in backend().hash()


def test_every_governed_executor_is_named_in_the_artifact(monkeypatch):
    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    gate = gate_of(plan.report())
    assert set(gate["governed_executors"]) == {"host", "cupy", "triton"}, (
        "the executors rungs 3 and 4 established are the ones that must be driven")
    assert set(gate["attained"]) == set(gate["governed_executors"])
    assert all(gate["attained"].values()), gate["attained"]
    assert gate["not_installed_in_this_process"] == []


def test_a_present_triton_is_governed_even_when_find_spec_will_not_answer(monkeypatch):
    """THE FAIL-OPEN, reconstructed exactly.

    ``find_spec('triton')`` raises ``ValueError('triton.__spec__ is None')`` for a
    module object with no spec, and the previous revision read that as "absent" —
    dropping from the governed set an executor rung 4 had just imported and
    certified, then dispatching. Measured then: ``governed_executors == ['host']``,
    ``executor_report('triton') == {}``, ``refused_because is None``, DISPATCHED.
    """
    import importlib.util

    with pytest.raises(ValueError):
        # The precondition, asserted rather than assumed: this is the state the
        # old gate silently read as "nothing to govern".
        governable_triton(monkeypatch)
        importlib.util.find_spec("triton")

    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    gate = gate_of(plan.report())
    assert "triton" in gate["governed_executors"]
    assert policy_module.executor_report("triton") != {}


def test_a_triton_that_cannot_be_governed_refuses_instead_of_dispatching(monkeypatch):
    """The other half: present, certified at rung 4, and NOT drivable.

    A bare module answers rung 4's version check and exposes no compiler backend.
    The old gate dropped it and dispatched; this one refuses by name, which is the
    fail-CLOSED direction on the same input.
    """
    module = types.ModuleType("triton")
    module.__version__ = "3.1.0"
    monkeypatch.setitem(sys.modules, "triton", module)
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", lambda f, p, **kwargs: step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "triton" in reason and "keep" in reason
    assert "internal error" not in reason
    assert not policy_module.policy_is_installed(), (
        "a refused install must leave the process on no policy at all")


def test_the_cupy_leg_is_governed_and_answers_for_itself(monkeypatch):
    """Governed, and its report READ — the fix is not "assume CuPy is fine".

    On this laptop the honest answer is that CuPy compiles nothing here, and that
    is what the leg says. What changed is that the leg RAN and is on the record:
    the defect was ``executor_report('cupy') == {}`` on a dispatching plan.
    """
    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert "cupy" in gate_of(plan.report())["governed_executors"]
    report = policy_module.executor_report("cupy")
    assert report != {}, "the CuPy leg was dropped from the governed set again"
    assert report["requested"] == "keep"


# ---------------------------------------------------------------------------
# CUPY_CACHE_DIR: the pointer, and what a refusal does with it
# ---------------------------------------------------------------------------


def test_a_refused_install_puts_the_cupy_cache_pointer_back(monkeypatch, tmp_path):
    """THE LEAK. A refusal used to leave the variable pointing at — and having
    created — the keep-policy directory, with the strip NOT installed, so the rest
    of the process wrote CuPy's own ``-ftz=true`` binaries into the keep policy's
    private cache. CuPy's cache key is computed above the strip seam, so the next
    process deriving that same deterministic path is served them under the keep
    policy's name, with no NVRTC call to notice."""
    mine = str(tmp_path / "my-cupy-cache")
    monkeypatch.setenv("CUPY_CACHE_DIR", mine)

    def refuse(*args, **kwargs):
        raise policy_module.SubnormalPolicyUnattainable(
            "the 'keep' subnormal policy is UNATTAINABLE for the 'cupy' executor")

    monkeypatch.setattr(policy_module, "install_subnormal_policy", refuse)
    assert plan_on_a_governed_host(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    assert "UNATTAINABLE" in record["refused_because"]
    assert not policy_module.policy_is_installed()
    assert os.environ["CUPY_CACHE_DIR"] == mine, (
        "the refusal left this process compiling into the keep policy's cache")
    cache = gate_of(record)["cupy_cache_dir"]
    assert cache["restored_to"] == mine and cache["restored_because"]


def test_a_successful_install_keeps_the_pointer_moved(monkeypatch, tmp_path):
    """The other direction, and it matters as much: with the strip installed this
    process's CuPy compiles ARE stripped, so they belong in the policy's own
    directory. Putting the pointer back would be the poisoning, not the fix."""
    mine = str(tmp_path / "my-cupy-cache")
    monkeypatch.setenv("CUPY_CACHE_DIR", mine)
    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert os.environ["CUPY_CACHE_DIR"] == mine + "-" + policy_module.CUPY_CACHE_POLICY_TOKEN
    assert policy_module.cupy_cache_reasons("keep", os.environ["CUPY_CACHE_DIR"]) == []


def test_a_cache_directory_this_package_did_not_mint_is_refused(monkeypatch, tmp_path):
    """PROVENANCE. The token in the NAME is a rule about the name; it is not
    evidence about the binaries inside, and nothing checked that. A directory
    holding entries with no marker may hold CuPy's own -ftz=true output, and a
    disk-cache hit serves it with no compile and no counter."""
    poisoned = tmp_path / f"cache-{policy_module.CUPY_CACHE_POLICY_TOKEN}"
    poisoned.mkdir()
    (poisoned / "e3b0c442.cubin").write_bytes(b"not ours")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(poisoned))

    assert policy_module.cupy_cache_reasons("keep", str(poisoned)) == [], (
        "the naming rule accepts it — which is exactly why provenance is separate")
    provenance = policy_module.keep_policy_cache_provenance(str(poisoned))
    assert provenance["reasons"], "an unmarked, non-empty cache is unknown"

    assert plan_on_a_governed_host(monkeypatch) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "provenance" in reason and str(poisoned) in reason
    assert "Delete that directory" in reason, "a refusal a user cannot act on"


def test_a_directory_this_package_minted_is_reused_without_complaint(monkeypatch, tmp_path):
    """The marker is what makes the refusal above a check rather than a wall."""
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cache"))
    first = policy_module.point_cupy_cache_at_keep_policy()
    assert first["provenance"]["marked"] is True and first["provenance"]["reasons"] == []
    chosen = first["after"]
    with open(os.path.join(chosen, "some.cubin"), "wb") as handle:
        handle.write(b"ours")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cache"))
    again = policy_module.point_cupy_cache_at_keep_policy()
    assert again["after"] == chosen
    assert again["provenance"]["reasons"] == []
    marker = policy_module.keep_policy_cache_marker(chosen)
    assert json.load(open(marker, encoding="utf-8"))["policy"] == "keep"


# ---------------------------------------------------------------------------
# The freeze is not the whole run
# ---------------------------------------------------------------------------


def test_a_policy_uninstalled_after_the_freeze_stops_the_dispatch(monkeypatch):
    """MEASURED BEFORE THE FIX: the plan kept answering True after
    ``uninstall_subnormal_policy()`` had put CuPy's compiler seam back — so
    already-compiled Triton kernels went on keeping while every new CuPy compile
    carried CuPy's own ``-ftz=true`` again. The shipped split, from inside a
    certified plan."""
    fields = object()
    counting = CountingPlan("b")
    plan = plan_on_a_governed_host(
        monkeypatch, step_plan({"step_B": counting}, {"step_B": "PML"}), fields=fields)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.dispatch("step_B", fields) is True and counting.runs == 1

    policy_module.uninstall_subnormal_policy()
    assert plan.dispatch("step_B", fields) is False, (
        "the plan kept dispatching under a policy that is no longer in force")
    assert counting.runs == 1, "a kernel ran after its licence lapsed"
    assert plan.report()["licence_lapses"]["step_B"] == 1


def test_a_policy_swapped_after_the_freeze_stops_the_dispatch(monkeypatch):
    fields = object()
    plan = plan_on_a_governed_host(
        monkeypatch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}),
        fields=fields)
    assert plan is not None
    policy_module.uninstall_subnormal_policy()
    policy_module.install_subnormal_policy("flush", strict=False, executors=("host",))
    assert plan.dispatch("step_B", fields) is False


def test_the_licence_lapse_reaches_stderr(monkeypatch, capsys):
    fields = object()
    plan = plan_on_a_governed_host(
        monkeypatch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}),
        fields=fields)
    assert plan is not None
    capsys.readouterr()
    policy_module.uninstall_subnormal_policy()
    plan.dispatch("step_B", fields)
    err = capsys.readouterr().err
    assert "no longer in force" in err and "step path array" in err


# ---------------------------------------------------------------------------
# What a refusal below the gate leaves behind
# ---------------------------------------------------------------------------


def test_a_refusal_below_the_gate_says_the_policy_stayed_installed(monkeypatch, capsys):
    """Rung 8b installs; rung 9 can still refuse. The process then keeps the
    certification's policy for its whole life — the ARRAY path's arithmetic
    included, which on an x86 host silently moves the whole run onto IEEE-754 —
    while the line a user reads says "step path array"."""
    def will_not_compile(plan, drive=None):
        raise RuntimeError("this kernel will not compile")

    monkeypatch.setattr(fastpath, "warm_plan", will_not_compile)
    assert plan_on_a_governed_host(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    assert "no slot is left carrying a kernel" in record["refused_because"]
    assert policy_module.policy_is_installed(), (
        "the precondition of this test: the gate ran before the warm pass")
    outlived = gate_of(record)["policy_outlived_the_refusal"]
    assert outlived["policy"] == "keep"
    assert outlived["installed_by"].startswith("dispatch")
    assert outlived["not_rolled_back_because"]
    line = [text for text in capsys.readouterr().err.splitlines()
            if text.startswith("meep_gpu: step path")][0]
    assert "STAYS IN FORCE" in line


def test_an_escape_after_the_gate_keeps_the_record_the_ladder_built(monkeypatch):
    """The one path on which the artifact made a FALSE NEGATIVE claim about
    coverage: an exception below rung 8b minted a fresh base record, whose
    subnormal block hardcodes "the configuration was refused before any kernel
    could run" — with a policy installed and CUPY_CACHE_DIR already moved."""
    def explode(*args, **kwargs):
        raise ValueError("something no rung expects")

    # AFTER the gate and outside the warm pass's own per-slot handler — the shape
    # of an escape that reaches ``plan_fast_path``'s wrapper with a policy live.
    monkeypatch.setattr(fastpath, "_record_slots", explode)
    assert plan_on_a_governed_host(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    assert "internal error" in record["refused_because"]
    assert policy_module.policy_is_installed()
    assert "not_read" not in record["subnormal"], (
        "the artifact claimed nothing was installed while a policy was in force")
    assert gate_of(record)["policy_in_force"] == "keep"


# ---------------------------------------------------------------------------
# Attribution, and the stderr line
# ---------------------------------------------------------------------------


def test_the_second_freeze_still_credits_dispatch_for_its_own_install(monkeypatch):
    """The driver re-plans after every ``invalidate_fast_path()``, so a long run
    emits far more second-and-later freezes than first ones; every one of them
    used to credit "something other than this freeze" for a policy dispatch had
    installed itself."""
    assert plan_on_a_governed_host(monkeypatch) is not None
    first = gate_of(fastpath.last_dispatch_report())["installed_by"]
    assert plan_on_a_governed_host(monkeypatch) is not None
    second = gate_of(fastpath.last_dispatch_report())["installed_by"]
    assert first == "dispatch, at the configuration freeze"
    assert second == "dispatch, at an earlier configuration freeze in this process"


def test_a_caller_installed_policy_is_still_credited_to_the_caller(monkeypatch):
    """The other side of the same string, which ``probe_dispatch_policy_gate``
    reads as its evidence that the caller installed it."""
    governable_triton(monkeypatch)
    policy_module.install_subnormal_policy("keep", strict=True)
    assert plan_on_a_governed_host(monkeypatch) is not None, (
        fastpath.last_dispatch_report()["refused_because"])
    installed_by = gate_of(fastpath.last_dispatch_report())["installed_by"]
    assert "dispatch" not in installed_by


def test_an_uninstall_between_freezes_drops_the_attribution(monkeypatch):
    """A flag would still read True here; the epoch does not."""
    backend = governable_triton(monkeypatch)
    assert plan_on_a_governed_host(monkeypatch) is not None
    policy_module.uninstall_subnormal_policy()
    monkeypatch.setattr(policy_module, "_cuda_backend_class", lambda: backend)
    policy_module.install_subnormal_policy("keep", strict=True)
    assert plan_on_a_governed_host(monkeypatch) is not None, (
        fastpath.last_dispatch_report()["refused_because"])
    assert "dispatch" not in gate_of(fastpath.last_dispatch_report())["installed_by"]


def test_a_second_distinct_refusal_still_reaches_stderr(monkeypatch, capsys):
    """The announcement was keyed on the literal "refused", so a process whose
    first freeze refused for an unrelated reason swallowed every later refusal —
    including the policy's, the one this round exists for."""
    import meep_gpu.triton_kernels as package
    import numpy

    monkeypatch.setattr(package, "plan_step", lambda f, p, **kwargs: step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    # THE FIRST REFUSAL HAS TO BE THE BACKEND'S ON EVERY MACHINE, and since the
    # kernel table became a choice that is a statement about the HARDWARE rather
    # than about the engine: a NumPy engine on a Mac with an MPS device is now a
    # CANDIDATE for the Metal table and is admitted, so on this laptop the ladder
    # walks past rung 3 and the refusal this test needs never happens. Pinning the
    # device read is what keeps the test about the ANNOUNCEMENT mechanism — which is
    # what it is for — instead of about which machine ran it.
    if hasattr(fastpath, "metal_hardware_present"):
        monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)
    numpy_grid = types.SimpleNamespace(xp=numpy)
    assert fastpath.plan_fast_path(object(), object(), numpy_grid) is None
    first = capsys.readouterr().err
    assert "backend is not CuPy" in first

    monkeypatch.setattr(policy_module, "policy_is_installed", lambda: True)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "flush")
    assert plan_on_a_governed_host(monkeypatch) is None
    second = capsys.readouterr().err
    assert "subnormal policy" in second, (
        "the policy refusal never reached the user who does not open the artifact")


# ---------------------------------------------------------------------------
# The host leg, read from the FPU rather than from the record
# ---------------------------------------------------------------------------


def test_a_host_whose_fpu_moved_after_the_install_is_caught(monkeypatch):
    """``install_host_policy``'s report is a statement about the moment it ran.
    ``mp.set_zero_subnormals`` is public MEEP API and a second library setting
    MXCSR needs none, so the FPU can move under a passing report — and the block's
    own comment says to verify the STATE rather than the REPORT."""
    from meep_gpu import backends

    real = backends.subnormals_flushed
    moved = {"yet": False}
    real_install = policy_module.install_subnormal_policy

    def install_then_move(*args, **kwargs):
        # The install itself must see a truthful FPU — it is the thing whose
        # report the gate is about to over-trust — so the move happens strictly
        # after it returns.
        stamp = real_install(*args, **kwargs)
        moved["yet"] = True
        return stamp

    monkeypatch.setattr(policy_module, "install_subnormal_policy", install_then_move)
    monkeypatch.setattr(backends, "subnormals_flushed",
                        lambda: True if moved["yet"] else real())
    assert plan_on_a_governed_host(monkeypatch) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "host FPU is flushing" in reason and "after it" in reason


def test_the_freeze_records_what_the_fpu_actually_reads(monkeypatch):
    plan = plan_on_a_governed_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert gate_of(plan.report())["host_flushing_now"] is False


# ---------------------------------------------------------------------------
# The policy module's own verdict logic
# ---------------------------------------------------------------------------


def test_an_undropped_cupy_kernel_memo_is_not_attained(monkeypatch, tmp_path):
    """"Installed" means "in force", and the drop is the mechanism that makes it
    so: this module's own note records ``cp.multiply`` compiled BEFORE a keep
    install still flushing afterwards, with the install reporting attained=True.
    The drop's result was recorded and then left out of ``reasons``."""
    monkeypatch.setenv("CUPY_CACHE_DIR",
                       str(tmp_path / f"c-{policy_module.CUPY_CACHE_POLICY_TOKEN}"))
    compiler = types.SimpleNamespace()
    for name in policy_module._COMPILER_ENTRY_POINTS:
        setattr(compiler, name, lambda *a, **k: None)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(compiler=compiler))
    monkeypatch.setattr(policy_module, "_clear_cupy_kernel_memo",
                        lambda cupy: {"cleared": False,
                                      "mechanism": "cupy._util exposes no clear_memo"})
    with pytest.raises(policy_module.SubnormalPolicyUnattainable) as excinfo:
        policy_module.install_cupy_policy("keep", fake, strict=True)
    assert "memo was NOT dropped" in str(excinfo.value)
    assert policy_module.executor_report("cupy")["attained"] is False


def test_an_install_that_dies_some_other_way_is_still_rolled_back(monkeypatch):
    """The bookkeeping is written BEFORE anything is driven, and only
    ``SubnormalPolicyUnattainable`` used to roll it back — so any other exception
    left the process LOCKED to a policy no executor had been driven to, with
    ``policy_stamp()`` certifying it as 'ieee_keep_ftz_stripped'."""
    def explode(*args, **kwargs):
        raise OSError("the cache directory is write-but-not-read")

    monkeypatch.setattr(policy_module, "install_cupy_policy", explode)
    with pytest.raises(OSError):
        policy_module.install_subnormal_policy("keep", strict=True)
    assert not policy_module.policy_is_installed()
    assert policy_module.policy_stamp()["policy"] == policy_module.UNINSTALLED_POLICY_NAME
    # And the process is not locked out of choosing again.
    policy_module.install_subnormal_policy("flush", strict=False, executors=("host",))


def test_the_epoch_changes_on_every_transition():
    """One integer, two jobs: who installed the policy in force, and whether the
    one a plan was gated on is still it."""
    start = policy_module.policy_epoch()
    policy_module.install_subnormal_policy("keep", strict=True, executors=("host",))
    after_install = policy_module.policy_epoch()
    policy_module.uninstall_subnormal_policy()
    after_uninstall = policy_module.policy_epoch()
    assert start != after_install != after_uninstall
    assert after_uninstall != start
