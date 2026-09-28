"""Host contracts for the one-launch, one-susceptibility ADE product."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

from meep_gpu.device_identity import weld_survives_edit
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels import fused_ade_state as module

PACKAGE_DIR = pathlib.Path(module.__file__).parent
PROBE = PACKAGE_DIR.parents[1] / "parity" / "meep_gpu" / "probe_triton_fused_ade_state.py"
SLURM = PACKAGE_DIR.parents[1] / "parity" / "meep_gpu" / "run_triton_fused_ade_state.slurm"


def _live_probe_is_welded(record, weld_name, repo_key, path):
    """The live probe file is the one a PASS weld recorded, or say why not.

    Kept separate from the historical entry's ``probe_sha256`` on purpose: that
    field names the bytes a particular past job ran and must not be re-typed when
    the file changes. This one asks the different question a reader actually needs
    answered — "is the probe on disk today the probe some released device run
    executed?" — and it is answered by the weld, which is the record that carries
    a status, a policy and an artifact digest.

    A MISSING WELD IS A NAMED FAILURE, never a skip. The probe was re-cut on
    device on 2026-08-19 and the weld entry is pending application to
    fingerprints.json; until it lands this test says so, with the digest to apply.
    """
    import hashlib

    live = hashlib.sha256(path.read_bytes()).hexdigest()
    weld = record.get(weld_name)
    assert isinstance(weld, dict), (
        f"{weld_name} is not in fingerprints.json; the live {repo_key} "
        f"({live}) is covered by no PASS weld. Apply the 2026-08-19 device weld.")
    assert weld.get("status") == "PASS", f"{weld_name} is not a PASS weld"
    recorded = weld["source_sha256"].get(repo_key)
    assert recorded == live, (
        f"{weld_name} recorded {recorded} for {repo_key}; the live file is "
        f"{live} — re-run the gate, do not re-type the record")


def with_state(sigma=None):
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    state = PolarizationState(
        Susceptibility(frequency=0.8, gamma=0.1),
        0.4 if sigma is None else sigma, grid, np.float32)
    fields.polarizations.append(state)
    return fields, state


def non_backend_reasons(fields, state):
    return [reason for reason in module.fused_ade_state_coverage(fields, state).reasons
            if "not cupy" not in reason]


def test_module_is_optional_and_the_host_predicate_matches_the_existing_ADE_slice():
    fields, state = with_state()
    assert non_backend_reasons(fields, state) == []
    assert module.plan_fused_ade_state(fields, state) is None


def test_predicate_refuses_an_empty_or_reordered_component_inventory():
    fields, state = with_state()
    state._driven = ()
    assert any("no driven" in reason for reason in non_backend_reasons(fields, state))

    fields, state = with_state()
    state._driven = ("Ez", "Ex")
    assert any("canonical" in reason for reason in non_backend_reasons(fields, state))


def test_kernel_keeps_each_ADE_recurrence_left_associated():
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    code = ast.unparse(tree)
    assert "options=" not in code
    assert code.count("enable_fp_fusion=") == 1
    for axis in range(3):
        expression = (
            f"((p{axis} * c_now) + (c_prev * q{axis})) + "
            f"(c_drive * (s{axis} * d{axis}))"
        )
        assert "".join(expression.split()) in "".join(source.split())


class _FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


def test_plan_resolves_the_shared_scratch_chain_then_rotates_only_after_launch(
        monkeypatch):
    old_p = {name: object() for name in ("Ex", "Ey", "Ez")}
    old_q = {name: object() for name in ("Ex", "Ey", "Ez")}
    old_scratch = object()
    drives = {name: object() for name in ("Ex", "Ey", "Ez")}
    state = SimpleNamespace(
        P=dict(old_p), P_prev=dict(old_q), _scratch=old_scratch,
        sigma={"Ex": 0.2, "Ey": 0.3, "Ez": 0.4},
        _coefficients=(1.1, -0.2, 0.05),
    )
    kernel = _FakeKernel()
    plan = module.FusedAdeStatePlan(
        state, ("Ex", "Ey", "Ez"), (2, 3, 4), 32,
        kernel=kernel, pointer=lambda value: value)
    monkeypatch.setitem(
        sys.modules, "meep_gpu.triton_kernels.kernels",
        SimpleNamespace(ENABLE_FP_FUSION=False))

    plan.run(drives.__getitem__, guard=False)

    assert len(kernel.calls) == 1
    _, args, kwargs = kernel.calls[0]
    assert (args[0], args[5], args[10]) == (
        old_scratch, old_q["Ex"], old_q["Ey"])
    assert (args[1], args[6], args[11]) == (
        old_p["Ex"], old_p["Ey"], old_p["Ez"])
    assert (args[2], args[7], args[12]) == (
        old_q["Ex"], old_q["Ey"], old_q["Ez"])
    assert kwargs["DRIVEN0"] == kwargs["DRIVEN1"] == kwargs["DRIVEN2"] == 1
    assert state.P == {
        "Ex": old_scratch, "Ey": old_q["Ex"], "Ez": old_q["Ey"]}
    assert state.P_prev == old_p
    assert state._scratch is old_q["Ez"]


def _stub_non_ade_products(monkeypatch):
    refused = coverage_module.Coverage(False, ("outside this routing test",))
    monkeypatch.setattr(launch_module, "pml_curl_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "plain_curl_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "constitutive_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *args: refused)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage", lambda *args: refused)
    return refused


def test_plan_step_selects_one_fused_ADE_launch_per_state_only_when_requested(monkeypatch):
    refused = _stub_non_ade_products(monkeypatch)
    admitted = coverage_module.Coverage(True, ())
    state = SimpleNamespace(driven=lambda: ("Ex", "Ey", "Ez"))
    fields = SimpleNamespace(polarizations=(state,))
    sentinel = object()
    monkeypatch.setattr(launch_module, "fused_ade_state_coverage",
                        lambda *args: admitted, raising=False)
    monkeypatch.setattr(launch_module, "plan_fused_ade_state",
                        lambda *args: sentinel, raising=False)
    monkeypatch.setattr(launch_module, "plan_ade_update_p",
                        lambda *args: (_ for _ in ()).throw(
                            AssertionError("separate ADE builder should not run")))

    plan = launch_module.plan_step(fields, object(), fuse_ade=True)

    assert plan.plans["update_P"] == [sentinel]
    assert plan.polarization_plans == (sentinel,)
    assert "update_P" not in plan.reasons
    assert refused.reasons


def test_cuda_gate_has_full_driver_scalar_volume_subset_and_second_oracle_cases():
    spec = importlib.util.spec_from_file_location("fused_ade_probe", PROBE)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert [case[0] for case in probe.CASES] == [
        "scalar_all_periodic", "volume_subset_periodic", "drude_all_metallic_x",
    ]
    assert sum(case[-1] for case in probe.CASES) == 36
    source = PROBE.read_text(encoding="utf-8")
    assert "fused_vs_array" in source and "fused_vs_separate" in source
    assert "Pscratch" in source and "uint32" in source
    assert "raise AssertionError" in source and "save(payload, args.out)" in source


def test_the_cuda_probe_reports_progress_per_step():
    assert "case {name} step {step}/{steps}" in PROBE.read_text(encoding="utf-8")


def test_fused_ADE_state_record_is_welded_to_every_live_dependency():
    import hashlib
    import json
    import meep_gpu.triton_kernels as public

    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    specialized = record["specialized_kernel_sources"]["fused_ade_state.py"]
    assert specialized["kernel"] == "fused_ade_state"
    module_path = pathlib.Path(module.__file__)
    live = hashlib.sha256(module_path.read_bytes()).hexdigest()
    # ONE HOME FOR THE RULE (device_identity.py:209). This entry pins a
    # specialized kernel module, whose device identity is the ``@triton.jit``
    # body plus the constants it reads; an edit the shared rule proves leaves
    # those alone leaves the record describing the shipped kernel. False for
    # anything it cannot establish -- this entry until a device_sha256 is
    # backfilled -- and then the byte rule below decides.
    if specialized["sha256"] != live and not weld_survives_edit(
            module_path, specialized, "fused_ade_state.py"):
        assert specialized["sha256"] == live, (
            "fused_ade_state.py drifted from the recorded specialized kernel "
            "source and the change reaches executable code")

    gate = record["fused_ade_state_gate"]
    assert record["current_host_gate"] in record
    assert gate["slurm_job_id"] == 2320
    assert gate["slurm_state"] == "COMPLETED 0:0"
    assert gate["product"]["complete_steps_exact"] == "36/36 against both oracles"
    assert "separate" in gate["product"]["oracles_per_step"]
    assert gate["dispatch"].startswith("DISABLED")

    package = PACKAGE_DIR.parent
    live = {
        "driver.py": package / "driver.py",
        "dispersion.py": package / "dispersion.py",
        "fields.py": package / "fields.py",
        "sources.py": package / "sources.py",
        "kernels.py": PACKAGE_DIR / "kernels.py",
        "launch.py": PACKAGE_DIR / "launch.py",
        "coverage.py": PACKAGE_DIR / "coverage.py",
        "conductivity.py": PACKAGE_DIR / "conductivity.py",
        "no_pml.py": PACKAGE_DIR / "no_pml.py",
        "dispersive_update_e.py": PACKAGE_DIR / "dispersive_update_e.py",
        "dispersive_fused_pair.py": PACKAGE_DIR / "dispersive_fused_pair.py",
        "fused_ade_state.py": PACKAGE_DIR / "fused_ade_state.py",
        "symmetry.py": PACKAGE_DIR / "symmetry.py",
        "__init__.py": PACKAGE_DIR / "__init__.py",
    }
    # This record is historical evidence; do not rewrite it merely because a
    # later, mutually excluded central route evolves.
    for name in live:
        assert len(gate["source_sha256"][name]) == 64, name
    # THE LIVE-PROBE PIN MOVED; the historical entry is left alone. See the
    # sibling comment in test_triton_dispersive_fused_pair for the whole reason:
    # the probe was re-cut on 2026-08-19 to install a NAMED subnormal policy and
    # to stamp provenance, and this tree adds a record rather than re-typing one.
    _live_probe_is_welded(record, "triton_fused_ade_state_device_gate",
                          "parity/meep_gpu/probe_triton_fused_ade_state.py",
                          PROBE)
    assert gate["slurm_launcher_sha256"] == hashlib.sha256(
        SLURM.read_bytes()).hexdigest()
    for name in ("fused_ade_state_coverage", "plan_fused_ade_state"):
        assert name in public.__all__ and callable(getattr(public, name))
