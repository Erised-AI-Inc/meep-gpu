"""The flush dispatch licence on the NVIDIA tables: admitted per arm, on evidence.

WHAT CHANGED. Until 0.10 every NVIDIA family's dispatch licence was ``keep`` only:
rung 8b refused any plan in a process whose caller had installed ``flush``, so a
user comparing against an unmodified x86 MEEP (which flushes) got the array path,
and the certification round's flush legs measured the array path against itself.
:data:`fastpath.TABLE_DISPATCH_POLICIES` now names the policies a table may
dispatch under, and :func:`fastpath.policy_evidence_reasons` admits one that is
not the table's install default only when EVERY arm the plan dispatches has a run
on this GPU's compute capability whose :data:`fastpath.CERTIFIED_POLICIES` names
it. The rebind tools write that field from released legs; nothing else does.

WHAT DID NOT CHANGE, and is pinned here so it cannot move by accident:

* dispatch still INSTALLS the table's install default (``keep`` on NVIDIA,
  ``flush`` on Metal) when nothing is installed;
* Metal admits ``flush`` only;
* a caller-installed policy the table never dispatches under is refused by name;
* a caller-installed admitted policy WITHOUT evidence for every arm is refused
  by name, naming the arm, and nothing compiles.

NOTHING HERE LAUNCHES A KERNEL. The device half is the route gate's ``flush`` /
``cuda_flush`` legs, which the round runs after the rebind has written the
evidence (``parity/meep_gpu/rounds/1.0/RUNBOOK.md``).
"""

from __future__ import annotations

import sys
import types

import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.subnormal_policy as policy_module
from meep_gpu import backends
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.launch import TritonStepPlan


# ---------------------------------------------------------------------------
# The admitted-policy table
# ---------------------------------------------------------------------------


def test_the_nvidia_tables_admit_both_policies_and_install_keep():
    for table in ("triton", fastpath.CUDA_TABLE):
        assert fastpath.TABLE_SUBNORMAL_POLICY[table] == "keep"
        assert set(fastpath.TABLE_DISPATCH_POLICIES[table]) == {"keep", "flush"}


def test_metal_admits_flush_only():
    assert fastpath.TABLE_SUBNORMAL_POLICY[fastpath.METAL_TABLE] == "flush"
    assert tuple(fastpath.TABLE_DISPATCH_POLICIES[fastpath.METAL_TABLE]) == ("flush",)


def test_every_table_admits_its_own_install_default():
    for table, policy in fastpath.TABLE_SUBNORMAL_POLICY.items():
        assert policy in fastpath.TABLE_DISPATCH_POLICIES[table], table


def test_the_evidence_field_is_a_run_field():
    """It describes ONE run on one capability, so it must live under ``runs[cc]``;
    an entry-level copy would be refused by the per-capability shape check."""
    assert fastpath.CERTIFIED_POLICIES in fastpath.RUN_FIELDS


# ---------------------------------------------------------------------------
# policy_evidence_reasons, against a stubbed certification lookup
# ---------------------------------------------------------------------------


def _lookup(entries):
    """A ``_certification_for`` stand-in: ``entries[arm]`` or an unmapped entry."""

    def certification_for(arm, table="triton", *, capability=None):
        entry = dict(entries.get(arm) or {"family": "unmapped", "gate": "unmapped"})
        if entry.get("capability") not in (None, capability):
            entry.pop("capability")
            entry.pop(fastpath.CERTIFIED_POLICIES, None)
        return entry

    return certification_for


def _run(gate, capability="8.6", policies=("keep", "flush")):
    return {"family": gate, "gate": gate, "capability": capability,
            fastpath.CERTIFIED_POLICIES: {p: {"artifact_sha256": "0" * 64}
                                          for p in policies}}


def test_the_install_default_needs_no_evidence(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({}))
    assert fastpath.policy_evidence_reasons(["PML"], "triton", "8.6", "keep") == []
    assert fastpath.policy_evidence_reasons(
        ["PML"], fastpath.METAL_TABLE, None, "flush") == []


def test_a_policy_the_table_never_dispatches_under_is_refused(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({}))
    reasons = fastpath.policy_evidence_reasons(["PML"], fastpath.METAL_TABLE,
                                               None, "keep")
    assert len(reasons) == 1 and "never dispatches" in reasons[0]


def test_flush_with_evidence_for_every_arm_is_admitted(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({
        "PML": _run("g_pml"), "fused pair B": _run("g_pair")}))
    assert fastpath.policy_evidence_reasons(
        ["PML", "fused pair B"], "triton", "86", "flush") == []


def test_one_arm_without_flush_evidence_refuses_and_is_named(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({
        "PML": _run("g_pml"), "fused pair B": _run("g_pair", policies=("keep",))}))
    reasons = fastpath.policy_evidence_reasons(
        ["PML", "fused pair B"], "triton", "8.6", "flush")
    assert len(reasons) == 1
    assert reasons[0].startswith("fused pair B:") and "'g_pair'" in reasons[0]
    assert "['keep']" in reasons[0]


def test_a_run_that_records_no_policy_field_certifies_only_the_default(monkeypatch):
    """Every run bound before the field existed was cut under the install default,
    so an absent field must NARROW the licence, never widen it."""
    bare = {"family": "g", "gate": "g", "capability": "8.6"}
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({"PML": bare}))
    reasons = fastpath.policy_evidence_reasons(["PML"], "triton", "8.6", "flush")
    assert len(reasons) == 1 and "only the install default" in reasons[0]


def test_evidence_on_another_capability_does_not_count(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for",
                        _lookup({"PML": _run("g_pml", capability="9.0")}))
    reasons = fastpath.policy_evidence_reasons(["PML"], "triton", "8.6", "flush")
    assert len(reasons) == 1 and "no live run on compute capability 8.6" in reasons[0]


def test_an_unreadable_capability_refuses_every_arm(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({"PML": _run("g")}))
    reasons = fastpath.policy_evidence_reasons(["PML", "PML"], "triton", None, "flush")
    assert len(reasons) == 1 and "could not be read" in reasons[0]


def test_an_unmapped_arm_is_refused(monkeypatch):
    monkeypatch.setattr(fastpath, "_certification_for", _lookup({}))
    reasons = fastpath.policy_evidence_reasons(["pending label"], fastpath.CUDA_TABLE,
                                               "8.6", "flush")
    assert len(reasons) == 1 and reasons[0].startswith("pending label:")


# ---------------------------------------------------------------------------
# Rung 8b, end to end through plan_fast_path on a governable host
# (the doubles are test_dispatch_policy_uniformity's, restated so this file
# stands alone)
# ---------------------------------------------------------------------------


def governable_triton(monkeypatch, version: str = "3.1.0"):
    class _CUDABackend:
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


def step_plan(plans, selected):
    reasons = {}
    for slot in launch_module.STEP_ORDER:
        if slot not in plans:
            reasons[slot] = (f"no consulted product admitted {slot}",)
    return TritonStepPlan(dict(plans), [], reasons, dict(selected))


def plan_on_a_governed_host(monkeypatch):
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", lambda f, p, **kwargs: step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    return fastpath.plan_fast_path(object(), object(), cupy_like_grid())


def gate_of(record):
    return record["subnormal"]["gate"]


def install_caller_flush():
    """Install ``flush`` as the caller would, or skip by the resource this host lacks.

    A caller's ``flush`` is attained only where ``subnormal_policy._drive_host_fpu`` can
    drive this thread's FPU to FTZ/DAZ (an already-imported MEEP's
    ``set_zero_subnormals``, or a host lever such as Darwin's
    ``FE_DFL_DISABLE_DENORMS_ENV``). Where none applies, the install records
    ``attained=False``, no caller policy is in force and dispatch installs ``keep``,
    so the caller-flush gate these tests pin is never reached. Measured here, after the
    install, not read off the platform.
    """
    policy_module.install_subnormal_policy("flush", strict=False)
    if not backends.subnormals_flushed():
        from conftest import requires_resource_skip  # noqa: PLC0415 - the root conftest

        requires_resource_skip("host_fpu_mode", "this host's FPU could not be driven to "
                               "flush, so no caller flush policy is in force")


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
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    yield
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None


def test_nothing_installed_still_installs_keep(monkeypatch):
    governable_triton(monkeypatch)
    assert plan_on_a_governed_host(monkeypatch) is not None, (
        fastpath.last_dispatch_report().get("refused_because"))
    gate = gate_of(fastpath.last_dispatch_report())
    assert gate["required_policy"] == "keep"
    assert gate["installed_by"] == "dispatch, at the configuration freeze"
    assert "policy_evidence" not in gate
    assert set(gate["admitted_policies"]) == {"keep", "flush"}


def test_a_caller_flush_without_evidence_refuses_naming_the_arm(monkeypatch):
    governable_triton(monkeypatch)
    install_caller_flush()
    assert plan_on_a_governed_host(monkeypatch) is None
    report = fastpath.last_dispatch_report()
    gate = gate_of(report)
    assert gate["policy_evidence"]["policy"] == "flush"
    assert gate["policy_evidence"]["arms"] == ["PML"]
    assert gate["policy_evidence"]["missing"]
    assert "admits 'flush' only where every arm" in report["refused_because"]
    assert "PML" in report["refused_because"]


def test_a_caller_flush_with_evidence_for_every_arm_dispatches_under_flush(monkeypatch):
    governable_triton(monkeypatch)
    asked = {}

    def evidence(arms, table, capability, policy):
        asked.update(arms=list(arms), table=table, policy=policy)
        return []

    monkeypatch.setattr(fastpath, "policy_evidence_reasons", evidence)
    install_caller_flush()
    plan = plan_on_a_governed_host(monkeypatch)
    report = fastpath.last_dispatch_report()
    assert plan is not None, report.get("refused_because")
    gate = gate_of(report)
    assert asked == {"arms": ["PML"], "table": "triton", "policy": "flush"}
    assert gate["required_policy"] == "flush"
    assert gate["caller_installed_policy"] == "flush"
    assert gate["policy_evidence"]["missing"] == []
    assert gate["policy_in_force"] == "flush"
    assert "dispatch" not in gate["installed_by"]


def test_the_admitted_licence_lapses_if_the_policy_moves_after_the_freeze(monkeypatch):
    """The consult compares the policy in force with the policy the GATE VERIFIED
    (flush here), not with the table's install default."""
    backend = governable_triton(monkeypatch)
    monkeypatch.setattr(fastpath, "policy_evidence_reasons", lambda *a: [])
    policy_module.install_subnormal_policy("flush", strict=False)
    fields = object()
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", lambda f, p, **kwargs: step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    plan = fastpath.plan_fast_path(fields, object(), cupy_like_grid())
    assert plan is not None, fastpath.last_dispatch_report().get("refused_because")
    assert plan.dispatch("step_B", fields) is True
    policy_module.uninstall_subnormal_policy()
    monkeypatch.setattr(policy_module, "_cuda_backend_class", lambda: backend)
    policy_module.install_subnormal_policy("keep", strict=False, executors=("host",))
    assert plan.dispatch("step_B", fields) is False
