"""The rule: dispatch never runs two executors under disagreeing subnormal policies.

WHY THIS FILE EXISTS. Every per-kernel and composition gate in this package
installs one float32 subnormal policy before its first compile, so both executors
are driven to the same one for the whole gate. That is correct for certifying
arithmetic and it is exactly the condition a user's run did not satisfy: nothing
in the shipped package called ``install_subnormal_policy``, so CuPy compiled with
its own unconditional ``-ftz=true`` and Triton compiled with no ftz attribute, and
the driver-route gate measured 6 of 7 dispatching cases diverging INSIDE the
8-64-step window the families are certified over. The split was read straight out
of the emitted code: 19 of 19 CuPy f32 PTX instructions carrying ``.ftz`` against
0 of 186 for Triton, on the same sub-step.

So what is pinned here is not "a policy is recorded". It is:

* the shipped opt-in path either INSTALLS the certified policy or REFUSES by name
  — there is no third outcome, and in particular no dispatched run with no policy;
* the install lands BEFORE anything compiles, because a policy installed after a
  compile does not reach the binary that already exists;
* a process that pre-installed a DIFFERENT policy is refused rather than swapped;
* every refusal is a named reason plus the array path, never an exception;
* and the guard catches a split that is DRIVEN APART deliberately, rather than
  only reporting the happy path.

NOTHING HERE LAUNCHES A KERNEL. These run on the NumPy laptop. The device half —
the shipped path byte-identical to the kill-switched array path with no harness
installing anything, and the PTX census showing both executors on one policy — is
``parity/meep_gpu/gate_dispatch_end_to_end.py`` and ``probe_ftz_ptx_census.py``.
"""

from __future__ import annotations

import sys
import types

import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.subnormal_policy as policy_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.launch import TritonStepPlan


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_process(monkeypatch):
    """A cold policy state and an opted-in ladder around every test.

    The policy module keeps PROCESS state — that is the point of it — so a test
    that installs and does not put it back decides the next test's verdict. The
    reset is the module's own full uninstall, seams included.
    """
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    monkeypatch.delenv(policy_module.POLICY_ENV, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    yield
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()


class StubBackend(types.SimpleNamespace):
    """Stands in for the ``cupy`` module without importing or needing it."""


def cupy_like_grid():
    xp = StubBackend()
    xp.__name__ = "cupy"
    xp.__version__ = "13.5.1"
    return types.SimpleNamespace(xp=xp)


def stub_triton(monkeypatch, version="3.1.0"):
    """A Triton the REAL install can be driven onto — see ``governable_triton``.

    IT WAS A BARE ``types.ModuleType``, whose ``__spec__`` is ``None``, and the
    gate's ``find_spec``-based governance read that raise as "absent". Every test
    in this file that DISPATCHED therefore ran with ``governed_executors ==
    ['host']``: the two executors whose disagreement is the defect were never
    driven, and the headline proved only that the host FPU had moved. The gate now
    governs what rungs 3 and 4 established, so the double has to be drivable.
    """
    from meep_gpu.test_dispatch_policy_uniformity import governable_triton

    governable_triton(monkeypatch, version)
    return sys.modules["triton"]


def step_plan(plans=None, selected=None, reasons=None):
    plans = dict(plans or {})
    selected = dict(selected or {})
    reasons = dict(reasons or {})
    for slot in launch_module.STEP_ORDER:
        if slot not in plans and slot not in reasons:
            reasons[slot] = (f"no consulted product admitted {slot}",)
    return TritonStepPlan(plans, [], reasons, selected)


class CountingPlan:
    def __init__(self, name):
        self.name = name
        self.runs = 0
        self.warms = 0

    def run(self, drive=None, guard=None):
        self.runs += 1

    def warm(self):
        self.warms += 1


def install_composer(monkeypatch, plan):
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step",
                        lambda fields, pml, **kwargs: plan)


def plan_on_a_cupy_host(monkeypatch, plan=None):
    """One configuration freeze on a host that answers "cupy" and has Triton."""
    stub_triton(monkeypatch)
    install_composer(monkeypatch, plan if plan is not None else step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    return fastpath.plan_fast_path(object(), object(), cupy_like_grid())


def gate_of(record):
    return record["subnormal"]["gate"]


# ---------------------------------------------------------------------------
# The shipped path: install or refuse, and nothing else
# ---------------------------------------------------------------------------


def test_the_shipped_opt_in_path_installs_the_certified_policy(monkeypatch):
    """THE HEADLINE. ``MEEP_GPU_DISPATCH=1`` and nothing else: a policy is on.

    Before this gate the same call dispatched with ``policy_is_installed()``
    False, which is the configuration the driver-route gate measured diverging.
    """
    assert not policy_module.policy_is_installed()
    plan = plan_on_a_cupy_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    gate = gate_of(fastpath.last_dispatch_report())
    assert gate["installed_by"].startswith("dispatch")
    assert gate["required_policy"] == fastpath.CERTIFICATION_SUBNORMAL_POLICY
    assert policy_module.policy_is_installed()
    assert policy_module.get_subnormal_policy() == "keep"


def test_the_installed_policy_is_the_one_the_families_were_certified_under():
    """One constant governs both, so a re-certification moves the gate with it."""
    assert fastpath.CERTIFICATION_SUBNORMAL_POLICY in policy_module.POLICIES
    for arm, entry in fastpath.ARM_CERTIFICATION.items():
        assert fastpath._certification_for(arm)["certification_policy"] == (
            fastpath.CERTIFICATION_SUBNORMAL_POLICY), arm


def test_the_artifact_records_the_policy_who_installed_it_and_where_it_landed(monkeypatch):
    """R: "the artifact must record the policy installed, by whom, and whether it
    was attained on each executor" — all three, from the record alone."""
    plan = plan_on_a_cupy_host(monkeypatch)
    assert plan is not None
    gate = gate_of(plan.report())
    assert gate["required_policy"] == "keep"
    assert gate["installed_by"]
    assert gate["policy_in_force"] == "keep"
    assert set(gate["attained"]) == set(gate["governed_executors"])
    assert all(gate["attained"].values()), gate["attained"]
    assert gate["rule"].startswith("dispatch never runs two executors")


def test_a_refused_configuration_never_installs_anything(monkeypatch):
    """Installing moves the process's arithmetic. A plan that was never going to
    dispatch must not pay that, which is why the gate is the LAST rung."""
    calls = []
    monkeypatch.setattr(policy_module, "install_subnormal_policy",
                        lambda *a, **k: calls.append(a) or {})
    reasons = {slot: (f"{slot}: nothing covers this grid",)
               for slot in launch_module.STEP_ORDER}
    assert plan_on_a_cupy_host(monkeypatch, step_plan(reasons=reasons)) is None
    assert calls == [], "a refused configuration installed a policy anyway"
    assert not policy_module.policy_is_installed()


def test_the_install_lands_before_the_first_kernel_compiles(monkeypatch):
    """THE ORDERING CONSTRAINT, measured rather than argued.

    Both device executors memoize compiled kernels in process, so a policy
    installed after a compile is not seen by the binary that already exists. The
    warm pass is where this seam compiles; the install must precede it.
    """
    order = []
    monkeypatch.setattr(policy_module, "install_subnormal_policy",
                        lambda *a, **k: order.append("install") or {})
    monkeypatch.setattr(policy_module, "policy_is_installed",
                        lambda: "install" in order)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "keep")
    monkeypatch.setattr(policy_module, "executor_report",
                        lambda name: {"executor": name, "attained": True})
    monkeypatch.setattr(fastpath, "warm_plan",
                        lambda plan, drive=None: order.append("warm") or None)
    assert plan_on_a_cupy_host(monkeypatch) is not None
    assert order == ["install", "warm"], order


# ---------------------------------------------------------------------------
# A process that already made this decision
# ---------------------------------------------------------------------------


def test_a_preinstalled_matching_policy_is_accepted_and_not_reinstalled(monkeypatch):
    # ALL THREE executors, because that is what "the caller already made this
    # decision" has to mean: a caller who drove only the host has left the two
    # device executors on their own defaults, which is the split, and
    # ``test_an_executor_left_ungoverned_is_caught`` is where that case belongs.
    # This test used to install the host alone and pass — because the gate was
    # dropping cupy and triton from the governed set, so there was nothing left to
    # be missing.
    stub_triton(monkeypatch)
    policy_module.install_subnormal_policy("keep", strict=True)
    calls = []
    monkeypatch.setattr(policy_module, "install_subnormal_policy",
                        lambda *a, **k: calls.append(a) or {})
    plan = plan_on_a_cupy_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    gate = gate_of(plan.report())
    assert calls == [], "an already-correct policy was installed a second time"
    assert "dispatch" not in gate["installed_by"]
    assert gate["caller_installed_policy"] == "keep"


def test_a_preinstalled_different_policy_refuses_by_name(monkeypatch):
    """THE DECISION, pinned: refuse, never swap.

    Uniform-flush is not a split — but it is not the certification either, and
    the module cannot swap in place because binaries compiled under the first
    policy are still reachable (CuPy's cache key is computed above the strip
    seam). So dispatch takes the array path and says why.
    """
    monkeypatch.setattr(policy_module, "policy_is_installed", lambda: True)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "flush")
    assert plan_on_a_cupy_host(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    reason = record["refused_because"]
    assert "'flush'" in reason and "'keep'" in reason
    assert "fresh process" in reason
    assert record["step_path"] == "array" and record["decision"] == "refused"
    assert gate_of(record)["caller_installed_policy"] == "flush"


def test_the_refusal_reaches_the_stderr_line_a_user_actually_reads(monkeypatch, capsys):
    monkeypatch.setattr(policy_module, "policy_is_installed", lambda: True)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "flush")
    assert plan_on_a_cupy_host(monkeypatch) is None
    line = [text for text in capsys.readouterr().err.splitlines()
            if text.startswith("meep_gpu: step path")][0]
    assert "step path array" in line and "subnormal policy" in line


# ---------------------------------------------------------------------------
# The opt-out: "do not install" is offered, "do not care" is not
# ---------------------------------------------------------------------------


def test_the_opt_out_refuses_when_nothing_is_installed_and_names_the_variable(monkeypatch):
    monkeypatch.setenv(fastpath.SUBNORMAL_INSTALL_SWITCH, "0")
    assert plan_on_a_cupy_host(monkeypatch) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert f"{fastpath.SUBNORMAL_INSTALL_SWITCH}=0" in reason
    assert "install_subnormal_policy" in reason, (
        "a refusal a user cannot act on is a refusal that will be worked around")
    assert not policy_module.policy_is_installed(), (
        "the opt-out installed a policy anyway")


def test_the_opt_out_dispatches_when_the_caller_installed_the_policy_itself(monkeypatch):
    monkeypatch.setenv(fastpath.SUBNORMAL_INSTALL_SWITCH, "0")
    stub_triton(monkeypatch)
    policy_module.install_subnormal_policy("keep", strict=True)
    plan = plan_on_a_cupy_host(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert gate_of(plan.report())["install_permitted"] is False


def test_the_opt_out_cannot_reach_a_split(monkeypatch):
    """It says "do not install", never "do not check" — the distinction is the
    whole reason the switch is safe to offer."""
    monkeypatch.setenv(fastpath.SUBNORMAL_INSTALL_SWITCH, "0")
    monkeypatch.setattr(policy_module, "policy_is_installed", lambda: True)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "flush")
    assert plan_on_a_cupy_host(monkeypatch) is None


# ---------------------------------------------------------------------------
# The negative controls: drive them apart and watch the guard fire
# ---------------------------------------------------------------------------


def test_an_executor_left_ungoverned_is_caught(monkeypatch, tmp_path):
    """THE SHIPPED DEFECT, reconstructed: one executor on the policy, one on its
    own default. A caller may drive a SUBSET — ``executors=("host","cupy")``
    leaves Triton compiling with no ftz attribute while CuPy is stripped — and
    that is the split with an extra step in front of it.
    """
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cache"))
    stub_triton(monkeypatch)
    policy_module.install_subnormal_policy("keep", strict=True,
                                           executors=("host", "cupy"))
    assert policy_module.policy_is_installed()
    assert policy_module.executor_report("triton") == {}, (
        "the fixture must leave Triton undriven for this to measure anything")
    assert plan_on_a_cupy_host(monkeypatch) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "triton" in reason and "own default" in reason


def test_an_unattained_executor_is_caught_with_its_own_reason(monkeypatch):
    """Installed is not attained. The module records WHY a leg did not take, and
    the refusal carries that text rather than a verdict of its own."""
    monkeypatch.setattr(policy_module, "policy_is_installed", lambda: True)
    monkeypatch.setattr(policy_module, "get_subnormal_policy", lambda: "keep")
    monkeypatch.setattr(policy_module, "executor_report", lambda name: {
        "executor": name, "attained": name != "cupy",
        "reasons": ["CUPY_CACHE_DIR is unset: the -ftz=true strip changes "
                    "binaries without changing CuPy's cache keys"],
    })
    assert plan_on_a_cupy_host(monkeypatch) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "cupy" in reason and "CUPY_CACHE_DIR is unset" in reason


def test_an_install_that_raises_is_a_named_refusal_not_an_error(monkeypatch):
    """Every rung of this ladder degrades to the array path. The policy rung is
    not an exception, and ``SubnormalPolicyUnattainable`` is the raise it is most
    likely to see — flush on a host whose CuPy imported CUB, measured."""
    def refuse(*args, **kwargs):
        raise policy_module.SubnormalPolicyUnattainable(
            "the 'keep' subnormal policy is UNATTAINABLE for the 'cupy' executor")

    monkeypatch.setattr(policy_module, "install_subnormal_policy", refuse)
    assert plan_on_a_cupy_host(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    assert "UNATTAINABLE" in record["refused_because"]
    assert "internal error" not in record["refused_because"], (
        "the ladder turned a policy refusal into an escape")
    assert record["step_path"] == "array"


def test_the_gate_never_lets_a_dispatched_plan_run_without_a_policy(monkeypatch):
    """The invariant behind all of it, stated once as a property.

    Whatever the process state, a returned plan implies a policy in force equal
    to the certification's — over every combination of the two knobs and the two
    pre-existing states this gate distinguishes.
    """
    for opt_out in ("0", None):
        for preinstalled in (None, "keep", "flush"):
            policy_module._reset_for_tests()
            with monkeypatch.context() as patch:
                if opt_out is None:
                    patch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
                else:
                    patch.setenv(fastpath.SUBNORMAL_INSTALL_SWITCH, opt_out)
                if preinstalled is not None:
                    patch.setattr(policy_module, "policy_is_installed", lambda: True)
                    patch.setattr(policy_module, "get_subnormal_policy",
                                  lambda p=preinstalled: p)
                plan = plan_on_a_cupy_host(patch)
                if plan is None:
                    assert fastpath.last_dispatch_report()["refused_because"]
                    continue
                assert policy_module.get_subnormal_policy() == (
                    fastpath.CERTIFICATION_SUBNORMAL_POLICY), (
                    f"dispatched with opt_out={opt_out} preinstalled={preinstalled}")


# ---------------------------------------------------------------------------
# The CuPy cache directory the policy needs (subnormal_policy's own addition)
# ---------------------------------------------------------------------------


def test_the_derived_cache_dir_satisfies_the_rule_that_demands_it(monkeypatch):
    """Derived by the module that owns the token rule, and CHECKED against it —
    not against a second copy of the convention written here."""
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    chosen = policy_module.keep_policy_cache_dir()
    assert policy_module.CUPY_CACHE_POLICY_TOKEN in chosen
    assert policy_module.cupy_cache_reasons("keep", chosen) == []


def test_a_compliant_cache_dir_is_left_exactly_as_the_caller_set_it(monkeypatch, tmp_path):
    mine = str(tmp_path / f"mine-{policy_module.CUPY_CACHE_POLICY_TOKEN}")
    monkeypatch.setenv("CUPY_CACHE_DIR", mine)
    assert policy_module.keep_policy_cache_dir() == mine


def test_a_noncompliant_cache_dir_keeps_the_callers_location(monkeypatch, tmp_path):
    """The caller chose WHERE; the policy only adds the separation. Replacing
    their directory outright would move a cache they may have sized on purpose."""
    mine = str(tmp_path / "somewhere-i-chose")
    monkeypatch.setenv("CUPY_CACHE_DIR", mine)
    chosen = policy_module.keep_policy_cache_dir()
    assert chosen.startswith(mine)
    assert policy_module.cupy_cache_reasons("keep", chosen) == []


def test_pointing_the_cache_at_the_policy_sets_the_variable_and_reports_it(monkeypatch, tmp_path):
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "before"))
    record = policy_module.point_cupy_cache_at_keep_policy()
    assert record["changed"] is True
    assert record["writable"] is True
    import os

    assert os.environ["CUPY_CACHE_DIR"] == record["after"]
    assert os.path.isdir(record["after"])
    assert policy_module.cupy_cache_reasons("keep", record["after"]) == []
