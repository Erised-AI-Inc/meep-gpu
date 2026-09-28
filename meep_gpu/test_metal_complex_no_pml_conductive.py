"""Host contracts for Metal's complex/Bloch no-PML conductive curl family."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.metal_kernels import complex_no_pml_conductive as family
from meep_gpu.pml import PML


class HostResidency:
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


def build(*, sides=("B", "D"), complex_storage=True, bloch=True, seed=57):
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries="periodic", k_point=(0.2, -0.125, 0.0) if bloch
                else (0.0, 0.0, 0.0), xp=np)
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
    sigma = np.full(grid.shape, np.float32(0.18))
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    return fields, PML(grid=grid, thickness=0)


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, expected_targets):
    def run(*args):
        targets, condfac, condinv = args[:3], args[6:9], args[9:12]
        assert tuple(targets) == tuple(expected_targets)
        for index, name in enumerate(family.SUB_STEPS[sub_step]["targets"]):
            if actual.condfac_for(name) is None:
                assert condfac[index] is targets[index]
                assert condinv[index] is targets[index]
            else:
                assert condfac[index] is actual.condfac_for(name)
                assert condinv[index] is actual.condinv_for(name)
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


def test_source_keeps_bloch_rotation_and_three_ordered_complex_passes():
    source = family.complex_conductive_no_pml_curl_source(
        (0, 0, 0), False, (1, 0, 0), (True, False, True), "FMA_V1")
    assert "device float2*" in source
    assert "b_x = wx ? c_mul(b_x, px) : b_x;" in source
    assert "float2 value0 = f0[ii];" in source
    assert "value0 = c_mul_field_left(value0, cf0[ii]);" in source
    assert "value0 = value0 - curl0;" in source
    assert "value0 = c_mul_field_left(value0, ci0[ii]);" in source
    assert "f1[ii] = f1[ii] - curl1;" in source
    assert "[[buffer(19)]]" in source and "[[buffer(20)]]" not in source


@pytest.mark.parametrize(("sides", "expected"), [
    (("B",), {"step_B": (True, True, True), "step_D": (False, False, False)}),
    (("D",), {"step_B": (False, False, False), "step_D": (True, True, True)}),
])
def test_complex_per_target_conductivity_is_split_by_curl_side(
        backend_ready, sides, expected):
    fields, pml = build(sides=sides)
    for slot in family.SUB_STEPS:
        plan = family.plan_metal_complex_conductive_no_pml_curl(
            fields, pml, slot, HostResidency(), probe=probe(),
            functions={shaders.CONTRACT_OFF: lambda *args: None})
        assert plan is not None
        assert plan.conductive == expected[slot]


def test_complex_bloch_two_curl_cycles_match_the_array_path(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    plan_b = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_B", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", [actual.Bx, actual.By, actual.Bz])})
    plan_d = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_D", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", [actual.Dx, actual.Dy, actual.Dz])})
    assert plan_b is not None and plan_d is not None
    for _ in range(2):
        plan_b.run()
        plan_d.run()
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert np.array_equal(getattr(actual, name).view(np.uint32),
                              getattr(reference, name).view(np.uint32))
    assert plan_b.launches == plan_d.launches == 2


def test_conductive_step_d_binds_derived_h_through_the_live_b_mirrors(backend_ready):
    fields, pml = build()
    residency = HostResidency()
    plan = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_D", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert plan is not None
    assert {"Bx", "By", "Bz"}.issubset(residency.arrays)
    assert not {"Hx", "Hy", "Hz"}.intersection(residency.arrays)
    assert plan.volumes == ("Dx", "Dy", "Dz", "Bx", "By", "Bz")


def test_complex_mixed_target_flags_bind_only_the_lossy_components(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.18))
    fields.set_b_conductivity({"Bx": None, "By": sigma, "Bz": None})
    fields.set_d_conductivity({"Dx": sigma, "Dy": None, "Dz": None})
    b = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_B", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_D", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    assert b.conductive == (False, True, False)
    assert d.conductive == (True, False, False)


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
    verdict = family.metal_complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency(), probe())
    assert verdict.covered is False
    assert any("the per-axis boundary rule is unreadable" in reason
               for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "complex64 storage"), ("active_pml", "active PML"),
    ("lossless", "no curl target"), ("missing_probe", "probe artifact"),
])
def test_distinct_products_are_refused_by_name(backend_ready, monkeypatch, mutation, needle):
    fields, pml = build(sides=(() if mutation == "lossless" else ("B", "D")))
    record = probe()
    if mutation == "real":
        fields, pml = build(complex_storage=False, bloch=False)
    elif mutation == "active_pml":
        pml = PML(grid=fields.grid, thickness=1)
    elif mutation == "missing_probe":
        record = None
        monkeypatch.delenv(shared_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    verdict = family.metal_complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_after_direct_and_complete_step_evidence():
    for slot in family.SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == "complex conductive no-PML curl"
        assert arm.wired


def test_native_gate_is_incremental_and_arms_conductive_failure_modes():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_complex_no_pml_conductive.py")
    source = gate.read_text(encoding="utf-8")
    assert "flush=True" in source and "os.fsync" in source
    assert "condfac_condinv_swapped" in source
    assert "target_store_dropped" in source
    assert "bloch_x_rotation_dropped" in source
