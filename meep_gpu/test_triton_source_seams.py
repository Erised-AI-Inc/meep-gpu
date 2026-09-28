"""Laptop contracts for the Triton cross-sub-step source seams.

The CUDA gate owns byte identity with real injected currents.  These tests own the
fail-closed family partition and the shape of that gate, and require neither CuPy
nor Triton.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest

from meep_gpu import deposit_repair
from meep_gpu.driver import FdtdDriver
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels.coverage import fused_pair_coverage

HERE = pathlib.Path(__file__).resolve()
PROBE = HERE.parents[1] / "parity" / "meep_gpu" / "probe_triton_source_seams.py"
SLURM = HERE.parents[1] / "parity" / "meep_gpu" / "run_triton_source_seams.slurm"


def _driver(*sources):
    driver = FdtdDriver(
        cell_size=(2.0, 2.0, 0.0), resolution=8.0, dimensions=2,
        force_complex_fields=False,
    )
    driver.setup_pml({"x": 2, "y": 2})
    for source in sources:
        driver.add_source(source)
    return driver


def _source(component: str, *, integrated: bool = False):
    return {
        "component": component,
        "frequency": 0.7,
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 0.0, 0.0),
        "is_integrated": integrated,
    }


@pytest.mark.parametrize(
    ("sources", "b_refuses", "d_refuses"),
    (
        ((), False, False),
        ((_source("Ez"),), False, True),
        ((_source("Hx"),), True, False),
        ((_source("Ez"), _source("Hx")), True, True),
        ((_source("Ez", integrated=True),), False, True),
        ((_source("Hx", integrated=True),), True, False),
    ),
)
def test_real_source_objects_partition_the_two_fusion_seams(
    monkeypatch, sources, b_refuses, d_refuses,
):
    """Only the family injected inside a pair's seam may refuse that pair.

    THE PARTITION IS STILL THE THING UNDER TEST, and it is asked twice.

    * With the deposit repair NOT declared -- the state every product shipped in
      before ``deposit_repair`` was wired -- the refusal must fall exactly on the
      pair whose own seam carries the injection and on no other. That is the table
      below, unchanged, and it is what a mis-assigned seam would break.
    * With it declared, which is what ``coverage.py`` now ships, every one of these
      rows is CARRIED instead: each source is a real ``VolumeSource`` that publishes
      the index it writes, and these fields are diagonal, linear and PML-backed, so
      ``deposit_repair.repairable`` admits them. The admission is the whole point of
      the flag; asserting it here is what keeps a silent regression to the old
      refusals visible.
    """
    driver = _driver(*sources)
    try:
        inventory = tuple(driver._sources)

        monkeypatch.setattr(coverage_module, "CARRIES_DEPOSIT_REPAIR", False)
        b_reasons = fused_pair_coverage(
            driver.fields, driver.pml, "B", inventory).reasons
        d_reasons = fused_pair_coverage(
            driver.fields, driver.pml, "D", inventory).reasons
        assert any("is magnetic" in reason for reason in b_reasons) is b_refuses
        assert any("is electric" in reason for reason in d_reasons) is d_refuses
        assert all("cupy" in reason or "source" in reason for reason in b_reasons)
        assert all("cupy" in reason or "source" in reason for reason in d_reasons)

        monkeypatch.setattr(coverage_module, "CARRIES_DEPOSIT_REPAIR", True)
        carried_b = fused_pair_coverage(
            driver.fields, driver.pml, "B", inventory).reasons
        carried_d = fused_pair_coverage(
            driver.fields, driver.pml, "D", inventory).reasons
        assert not [reason for reason in carried_b if "cupy" not in reason], carried_b
        assert not [reason for reason in carried_d if "cupy" not in reason], carried_d
    finally:
        driver.close()


def test_the_shipped_declaration_is_the_one_the_carried_leg_above_asserts():
    """The monkeypatch above proves what the flag does; this proves which way it ships.

    Without this, both legs would keep passing if ``coverage.py`` silently reverted to
    ``CARRIES_DEPOSIT_REPAIR = False`` -- the test would then be measuring a value it
    supplies itself.
    """
    assert coverage_module.CARRIES_DEPOSIT_REPAIR is True


def test_a_source_that_hides_its_deposit_index_is_still_refused():
    """The flag admits a seam the repair can carry, not every seam.

    ``deposit_repair.save`` needs the (ix, iy, iz) the injection writes. A source that
    does not publish one cannot be saved and restored across the launch, and admitting
    it would fuse over a deposit that is then never repaired.
    """
    driver = _driver()
    try:
        opaque = SimpleNamespace(field_type="B")
        reasons = fused_pair_coverage(
            driver.fields, driver.pml, "B", (opaque,)).reasons
        assert any("does not publish the index it writes" in reason
                   for reason in reasons), reasons
    finally:
        driver.close()


def test_an_offdiagonal_row_is_refused_because_its_constitutive_half_is_a_stencil():
    """The precondition the repair rests on, asserted where the flag is now True.

    ``update_E`` on an off-diagonal chi1inv reads the PARTNER components' volumes at
    shifted indices, so a repair confined to the deposit points cannot reconstruct it.
    ``deposit_repair.repairable`` refuses it by name and the reason reaches this
    predicate rather than being assumed away by the declaration.
    """
    driver = _driver(_source("Ez"))
    try:
        offdiagonal = SimpleNamespace(
            grid=driver.fields.grid, has_offdiagonal_epsilon=True,
            has_nonlinearity=False,
            **{"f_w_E" + axis: getattr(driver.fields, "f_w_E" + axis)
               for axis in "xyz"})
        reasons = deposit_repair.seam_source_reasons(
            offdiagonal, tuple(driver._sources), "D",
            undeclared="undeclared", refusal=lambda index, source: "refused",
            carries_repair=True)
        assert any("stencil" in reason for reason in reasons), reasons
        assert "refused" not in reasons, reasons
    finally:
        driver.close()


def test_an_undeclared_inventory_refuses_both_pairs():
    driver = _driver()
    try:
        for pair in ("B", "D"):
            reasons = fused_pair_coverage(driver.fields, driver.pml, pair).reasons
            assert any("source set was not declared" in reason for reason in reasons)
    finally:
        driver.close()


def test_the_cuda_probe_covers_positive_refusal_and_integrated_source_routes():
    spec = importlib.util.spec_from_file_location("triton_source_seam_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = {case[0]: case for case in module.CASES}
    assert set(cases) == {
        "source_free", "electric", "magnetic", "electric_and_magnetic",
        "integrated_electric", "integrated_magnetic",
    }
    # THE CLEAN SEAM keeps the sentinel in the second slot; the seam that carries a
    # deposit holds the repair, on the pair whose OWN family was injected and on that
    # pair only. A case that expected NoopPlan where a deposit lands would be asserting
    # byte identity produced by a refused fusion, which measures nothing.
    assert cases["source_free"][3]["step_B"] == "FusedPairPlan"
    assert cases["source_free"][3]["update_H"] == "NoopPlan"
    assert cases["source_free"][3]["step_D"] == "FusedPairPlan"
    assert cases["source_free"][3]["update_E"] == "NoopPlan"

    assert cases["electric"][3]["step_B"] == "FusedPairPlan"
    assert cases["electric"][3]["update_H"] == "NoopPlan"
    assert cases["electric"][3]["step_D"] == "LeadingRepairPlan"
    assert cases["electric"][3]["update_E"] == "TrailingRepairPlan"

    assert cases["magnetic"][3]["step_B"] == "LeadingRepairPlan"
    assert cases["magnetic"][3]["update_H"] == "TrailingRepairPlan"
    assert cases["magnetic"][3]["step_D"] == "FusedPairPlan"

    assert cases["electric_and_magnetic"][3] == {
        "step_B": "LeadingRepairPlan", "update_H": "TrailingRepairPlan",
        "step_D": "LeadingRepairPlan", "update_E": "TrailingRepairPlan",
    }
    # An integrated current reaches the field on a different route (it withdraws its
    # standing dipole before the curl); it still deposits in the same seam.
    assert cases["integrated_electric"][3] == cases["electric"][3]
    assert cases["integrated_magnetic"][3] == cases["magnetic"][3]
    assert all(case[2] >= 8 for case in cases.values())


def test_the_cuda_probe_refuses_a_vacuous_repair():
    """A repaired case that repaired nothing would be byte-identical for the wrong
    reason, so the probe counts the points ``apply`` touched and fails on zero."""
    probe = PROBE.read_text(encoding="utf-8")
    assert "repairs_seen" in probe
    assert "the repair touched no point" in probe
    assert "no deposit point was ever repaired" in probe


def test_the_cuda_probe_is_self_enforcing():
    probe = PROBE.read_text(encoding="utf-8")
    assert "raise AssertionError" in probe
    assert "differing_floats" in probe and "uint32" in probe
    assert "save(payload, args.out)" in probe
    assert "case {name} step {step}/{steps}" in probe


def test_the_source_seam_record_is_welded_to_the_injection_and_plan_bytes():
    """The gate includes the driver seam, not only the kernels it selected."""
    import hashlib
    import json

    package = HERE.parent
    record = json.loads(
        (package / "triton_kernels" / "fingerprints.json").read_text(
            encoding="utf-8"))
    gate = record["source_seam_gate"]
    # RE-CUT 2026-08-27 AND NO LONGER A SLURM JOB. probe_triton_source_seams.py is the
    # gate that measures the in-seam deposit repair, so it moved when that repair went
    # live in triton_kernels/coverage.py (CARRIES_DEPOSIT_REPAIR False -> True) and the
    # record was owed a fresh run. That run was launched directly on the GPU host rather
    # than through Slurm, so the job fields describe HISTORY and are spelled that way
    # instead of being left to read as though 2318 had produced today's bytes.
    assert gate["_historical_slurm_job_id"] == 2318
    assert gate["_historical_slurm_state"] == "COMPLETED 0:0"
    assert "RE-RUN on 2026-08-27" in gate["_superseded_2026-08-27"]
    assert gate["real_engine_route"]["complete_steps_exact"] == "62/62"
    assert gate["real_engine_route"]["nonvacuous_source_cases"].startswith("5/5")
    assert gate["real_engine_route"]["same_family_refusals"].startswith("6/6")
    assert gate["dispatch"].startswith("DISABLED")

    sources = {
        "driver.py": package / "driver.py",
        "sources.py": package / "sources.py",
        "kernels.py": package / "triton_kernels" / "kernels.py",
        "launch.py": package / "triton_kernels" / "launch.py",
        "coverage.py": package / "triton_kernels" / "coverage.py",
        "conductivity.py": package / "triton_kernels" / "conductivity.py",
        "no_pml.py": package / "triton_kernels" / "no_pml.py",
        "dispersive_update_e.py": (
            package / "triton_kernels" / "dispersive_update_e.py"),
        "dispersive_fused_pair.py": (
            package / "triton_kernels" / "dispersive_fused_pair.py"),
        "fused_ade_state.py": (
            package / "triton_kernels" / "fused_ade_state.py"),
        "symmetry.py": package / "triton_kernels" / "symmetry.py",
        "__init__.py": package / "triton_kernels" / "__init__.py",
    }
    # This is historical evidence: later, fail-closed routes may legitimately
    # change the shared planner without retroactively changing this gate's bytes.
    # The current-gate check lives in test_triton_kernels instead.
    for name in sources:
        assert len(gate["source_sha256"][name]) == 64, name
    assert gate["probe_sha256"] == hashlib.sha256(PROBE.read_bytes()).hexdigest()
    assert gate["slurm_launcher_sha256"] == hashlib.sha256(
        SLURM.read_bytes()).hexdigest()
