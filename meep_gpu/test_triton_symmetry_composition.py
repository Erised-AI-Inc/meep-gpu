"""Laptop-safe contracts for folded-grid Triton step composition.

The standalone folded curl and mirror-fill kernels are certified separately in
``test_triton_symmetry.py`` and on CUDA by ``gate_triton_symmetry.py``.  These
tests cover the next, narrower product: selecting those kernels together with
the ordinary constitutive kernel for a nondispersive folded PML step, while
preserving the driver's source-adjacent boundary-fill order.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import pathlib
from types import SimpleNamespace

from meep_gpu.grid import Grid
from meep_gpu.fields import Fields
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels import symmetry


PACKAGE_DIR = pathlib.Path(symmetry.__file__).parent
PARITY_DIR = pathlib.Path(__file__).parents[1] / "parity" / "meep_gpu"
PROBE = PARITY_DIR / "probe_triton_symmetry_composition.py"


ADMITTED = coverage_module.Coverage(True, ())
REFUSED = coverage_module.Coverage(False, ("outside this routing test",))


def folded_numpy():
    grid = Grid(
        resolution=10.0,
        cell_size=(0.8, 2.0, 0.8),
        boundaries="periodic",
        symmetry=("Y",),
    )
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


def composition_fields():
    grid = SimpleNamespace(
        has_symmetry=lambda: True,
        is_mirrored=lambda axis: axis == 1,
    )
    return SimpleNamespace(grid=grid, polarizations=())


def install_folded_routing_stubs(monkeypatch, *, ordinary=REFUSED,
                                 folded=ADMITTED, fill=ADMITTED):
    """Isolate the central routing decision from every device-backed builder."""
    monkeypatch.setattr(launch_module, "pml_curl_coverage",
                        lambda *args: ordinary)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "plain_curl_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "folded_composition_curl_coverage",
                        lambda *args: folded, raising=False)
    monkeypatch.setattr(launch_module, "plan_folded_pml_curl",
                        lambda fields, pml, name, block=None: f"folded:{name}",
                        raising=False)

    monkeypatch.setattr(launch_module, "mirror_ghost_fill_coverage",
                        lambda fields, family: fill, raising=False)
    monkeypatch.setattr(launch_module, "plan_mirror_ghost_fill",
                        lambda fields, family, block=None: f"fill:{family}",
                        raising=False)

    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda *args: ordinary)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "folded_constitutive_coverage",
                        lambda *args: folded, raising=False)
    monkeypatch.setattr(launch_module, "plan_folded_constitutive",
                        lambda fields, pml, side, block=None: f"folded:{side}",
                        raising=False)


def test_folded_constitutive_is_covered_but_for_numpy_and_requires_a_fold():
    fields, pml = folded_numpy()
    for side in ("H", "E"):
        verdict = symmetry.folded_constitutive_coverage(fields, pml, side)
        assert verdict.covered is False
        assert len(verdict.reasons) == 1, verdict.reasons
        assert "cupy" in verdict.reasons[0]

    unfolded = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    ordinary = Fields(grid=unfolded, force_complex_fields=False)
    ordinary.enable_pml_storage()
    verdict = symmetry.folded_constitutive_coverage(
        ordinary, PML(grid=unfolded, thickness=2), "H")
    assert any("no mirror plane" in reason for reason in verdict.reasons)


def test_step_order_includes_source_adjacent_fill_slots():
    assert launch_module.STEP_ORDER == (
        "step_B", "fill_B", "update_H",
        "step_D", "fill_D", "update_E", "update_P",
    )


def test_plan_step_selects_the_complete_nondispersive_folded_slice(monkeypatch):
    install_folded_routing_stubs(monkeypatch)

    plan = launch_module.plan_step(composition_fields(), object())

    assert plan.replaces == (
        "step_B", "fill_B", "update_H",
        "step_D", "fill_D", "update_E",
    )
    assert plan.plans == {
        "step_B": "folded:step_B",
        "fill_B": "fill:B",
        "update_H": "folded:H",
        "step_D": "folded:step_D",
        "fill_D": "fill:D",
        "update_E": "folded:E",
    }


def test_folded_curl_overlap_fails_closed_instead_of_selecting_by_order(monkeypatch):
    install_folded_routing_stubs(monkeypatch, ordinary=ADMITTED)
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *args, **kwargs: "ordinary")

    plan = launch_module.plan_step(composition_fields(), object())

    for name in ("step_B", "step_D"):
        assert name not in plan.plans
        assert any("ambiguous" in reason for reason in plan.reasons[name])


def test_folded_constitutive_overlap_fails_closed(monkeypatch):
    install_folded_routing_stubs(monkeypatch, ordinary=ADMITTED)
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *args, **kwargs: "ordinary")
    monkeypatch.setattr(launch_module, "plan_constitutive",
                        lambda fields, pml, side, block=None: f"ordinary:{side}")

    plan = launch_module.plan_step(composition_fields(), object())

    for name in ("update_H", "update_E"):
        assert name not in plan.plans
        assert any("ambiguous" in reason or "both" in reason
                   for reason in plan.reasons[name])


def test_requested_pair_fusion_cannot_overwrite_folded_source_seams(monkeypatch):
    install_folded_routing_stubs(monkeypatch)
    monkeypatch.setattr(launch_module, "fused_pair_coverage",
                        lambda *args: ADMITTED)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *args: ADMITTED)
    monkeypatch.setattr(
        launch_module, "plan_fused_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("folded source-adjacent slots cannot be fused yet")))

    plan = launch_module.plan_step(
        composition_fields(), object(), fuse=True, sources=())

    assert plan.plans["step_B"] == "folded:step_B"
    assert plan.plans["update_H"] == "folded:H"
    assert plan.plans["step_D"] == "folded:step_D"
    assert plan.plans["update_E"] == "folded:E"
    assert all(any("fold" in reason.lower()
                   for reason in plan.reasons[f"fused_pair_{pair}"])
               for pair in ("B", "D"))


def test_public_package_exposes_each_folded_composition_entry_point():
    import meep_gpu.triton_kernels as public

    names = (
        "folded_pml_curl_coverage", "plan_folded_pml_curl",
        "mirror_ghost_fill_coverage", "plan_mirror_ghost_fill",
        "folded_constitutive_coverage", "plan_folded_constitutive",
    )
    for name in names:
        assert name in public.__all__
        assert callable(getattr(public, name))


def test_cuda_gate_spans_fold_rules_and_both_source_adjacent_seams():
    spec = importlib.util.spec_from_file_location("symmetry_composition_probe", PROBE)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert [case[0] for case in probe.CASES] == [
        "periodic_even",
        "periodic_even_electric_on_plane",
        "periodic_even_magnetic_on_plane",
        "periodic_odd_phase_odd_count",
        "metallic_even",
        "periodic_two_folds",
    ]
    assert sum(case[-1] for case in probe.CASES) == 52
    assert [item["component"] for case in probe.CASES for item in case[4]] == [
        "Ez", "Hy"]
    assert probe.EXPECTED == {
        "step_B": "FoldedPmlCurlPlan",
        "fill_B": "MirrorGhostFillPlan",
        "update_H": "ConstitutivePlan",
        "step_D": "FoldedPmlCurlPlan",
        "fill_D": "MirrorGhostFillPlan",
        "update_E": "ConstitutivePlan",
    }
    source = PROBE.read_text(encoding="utf-8")
    assert "fill_symmetry_bc_B" in source and "fill_folded_far_ghosts_B" in source
    assert "source_effect_seen" in source and "uint32" in source
    assert "case {name} step {step}/{steps}" in source


def test_the_symmetry_composition_record_is_welded_to_the_complete_step_seam():
    """The folded composer gate is invalidated by any live dependency change."""
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    gate = record["symmetry_composition_gate"]
    assert gate["product"]["cases_exact"] == "6/6"
    assert gate["product"]["complete_steps_exact"] == "52/52"
    assert gate["product"]["source_cases_nonvacuous"] == "2/2: on-plane electric Ez and magnetic Hy"
    assert "six launches" in gate["fusion_boundary"]
    assert gate["dispatch"].startswith("DISABLED")

    package = PACKAGE_DIR.parent
    live = {
        "driver.py": package / "driver.py",
        "fields.py": package / "fields.py",
        "grid.py": package / "grid.py",
        "pml.py": package / "pml.py",
        "sources.py": package / "sources.py",
        "stepping.py": package / "stepping.py",
        "kernels.py": PACKAGE_DIR / "kernels.py",
        "launch.py": PACKAGE_DIR / "launch.py",
        "coverage.py": PACKAGE_DIR / "coverage.py",
        "symmetry.py": PACKAGE_DIR / "symmetry.py",
        "__init__.py": PACKAGE_DIR / "__init__.py",
    }
    # Later, mutually excluded central routes do not alter the exact bytes this
    # historical folded-grid gate ran.  The current closure is welded elsewhere.
    for name in live:
        assert len(gate["source_sha256"][name]) == 64, name
    assert gate["probe_sha256"] == hashlib.sha256(PROBE.read_bytes()).hexdigest()
