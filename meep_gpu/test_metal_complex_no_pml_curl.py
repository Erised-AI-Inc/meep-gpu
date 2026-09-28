"""Host contracts for Metal's complex/Bloch lossless no-PML curl family.

These tests deliberately use an identity residency and an array-path callback:
they establish planner, ABI and lifecycle parity without replacing the native
MPS byte gate. The family is wired only after that native gate and the separate
whole-step residency gate have established the device path.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_fields as expansion_predicate
from meep_gpu.metal_kernels import complex_no_pml_curl as family
from meep_gpu.metal_kernels import complex_no_pml_conductive as shared_predicate
from meep_gpu.pml import PML


class HostResidency:
    """Identity mirrors make plan argument ordering observable on this host."""

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def probe():
    return {"backend": "numpy", "patterns": {
        name: "FMA_V1" for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left")}}


def build(*, complex_storage=True, bloch=True, seed=71):
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries="periodic",
                k_point=(0.2, -0.125, 0.0) if bloch else (0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            real = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            imag = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            array[...] = real + 1j * imag if complex_storage else real
    return fields, PML(grid=grid, thickness=0)


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(shared_predicate, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, expected_targets):
    def run(*args):
        targets, sources = args[:3], args[3:6]
        assert all(target is expected for target, expected in zip(targets, expected_targets))
        accessor = getattr(actual, "get_E" if sub_step == "step_B" else "get_H")
        assert all(source is accessor(name)
                   for source, name in zip(
                       sources, family.SUB_STEPS[sub_step]["sources"]))
        # Positions 6:12 are ABI placeholders only; lossless source never reads
        # them, and binding targets makes accidental host copies impossible.
        assert all(value is target for value, target in zip(args[6:9], targets))
        assert all(value is target for value, target in zip(args[9:12], targets))
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, family.SUB_STEPS[sub_step]["targets"]):
            target[...] = getattr(reference, name)
    return run


def _unresolvable(grid, pml):  # noqa: ARG001
    """What the array path does with a grid it will not resolve.

    ``coverage._boundary_kinds`` turns exactly this into ``None`` (coverage.py:105-121),
    which is the input the predicate has to REFUSE rather than crash on.
    """
    raise RuntimeError("the array path will not resolve this grid")


def test_lossless_source_reuses_phase_stencil_but_compiles_out_all_coefficients():
    source = family.complex_no_pml_curl_source(
        (0, 0, 0), False, (1, 0, 0), "FMA_V1")
    body = source[source.index("kernel void"):]
    assert "b_x = wx ? c_mul(b_x, px) : b_x;" in body
    assert "f0[ii] = f0[ii] - curl0;" in body
    assert "f1[ii] = f1[ii] - curl1;" in body
    assert "f2[ii] = f2[ii] - curl2;" in body
    assert "cf0[ii]" not in body and "ci0[ii]" not in body
    assert "[[buffer(19)]]" in body and "[[buffer(20)]]" not in body


def test_coverage_admits_both_complex_lossless_curl_sides(backend_ready):
    fields, pml = build()
    for slot in family.SUB_STEPS:
        verdict = family.metal_complex_no_pml_curl_coverage(
            fields, pml, slot, HostResidency(), probe())
        assert verdict.covered, verdict.reasons


def test_complex_bloch_two_lossless_curl_cycles_match_the_array_path(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    b = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_B", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", (actual.Bx, actual.By, actual.Bz))})
    d = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_D", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", (actual.Dx, actual.Dy, actual.Dz))})
    assert b is not None and d is not None
    for _ in range(2):
        b.run()
        d.run()
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert np.array_equal(getattr(actual, name).view(np.uint32),
                              getattr(reference, name).view(np.uint32)), name
    assert b.launches == d.launches == 2


def test_step_d_binds_derived_h_through_the_live_b_mirrors(backend_ready):
    fields, pml = build()
    residency = HostResidency()
    plan = family.plan_metal_complex_no_pml_curl(
        fields, pml, "step_D", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert plan is not None
    assert {"Bx", "By", "Bz"}.issubset(residency.arrays)
    assert not {"Hx", "Hy", "Hz"}.intersection(residency.arrays)
    assert plan.volumes == ("Dx", "Dy", "Dz", "Bx", "By", "Bz")


def test_an_unreadable_boundary_rule_is_a_refusal_and_not_a_crash(monkeypatch):
    """UNREADABLE IS A REFUSAL, NEVER A CRASH — measured on an otherwise REAL grid.

    ``coverage._boundary_kinds`` answers ``None`` for a grid the array path will not
    resolve, and ``complex_no_pml_conductive._base_reasons`` — the clause list this
    family shares with the other complex no-PML curls, the off-diagonal E and the
    fused conductive pair — reads that triple TWICE: once for the covered ghost rule
    and once for per-axis Bloch legality. The grid here is real, so ``bloch_phase``
    is callable and the SECOND read is genuinely reached; a guard on the first read
    alone would still raise here.
    """
    fields, pml = build()
    monkeypatch.setattr(stepping, "_boundary_kinds", _unresolvable)
    verdict = family.metal_complex_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency(), probe())
    assert verdict.covered is False
    assert any("the per-axis boundary rule is unreadable" in reason
               for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "complex64 storage"),
    ("active_pml", "active PML"),
    ("conductive", "conductivity is installed on Bx"),
    ("missing_probe", "probe artifact"),
])
def test_distinct_products_are_refused_by_name(backend_ready, monkeypatch, mutation, needle):
    fields, pml = build()
    record = probe()
    if mutation == "real":
        fields, pml = build(complex_storage=False, bloch=False)
    elif mutation == "active_pml":
        pml = PML(grid=fields.grid, thickness=1)
    elif mutation == "conductive":
        fields.set_b_conductivity(np.full(fields.grid.shape, np.float32(0.18)))
    elif mutation == "missing_probe":
        record = None
        monkeypatch.delenv(expansion_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    verdict = family.metal_complex_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_family_is_wired_after_native_and_whole_step_evidence():
    for slot in family.SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == "complex no-PML curl"
        assert arm.wired


def test_registry_and_lazy_package_surface_include_the_family():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    assert family.FAMILY in registry.FAMILY_MODULES
    assert family.FAMILY in package.__all__
    assert package.complex_no_pml_curl is family


def test_native_gate_is_incremental_and_arms_the_lossless_failure_modes():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_complex_no_pml_curl.py")
    source = gate.read_text(encoding="utf-8")
    assert "flush=True" in source and "os.fsync" in source
    assert "curl_subtraction_reversed" in source
    assert "curl_grouping_flattened" in source
    assert "bloch_x_rotation_dropped" in source
