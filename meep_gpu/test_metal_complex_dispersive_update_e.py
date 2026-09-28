"""Host contract for complex/Bloch PML dispersive Metal ``update_E``."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_dispersive_update_e as family
from meep_gpu.metal_kernels import complex_dispersive_spine as spine
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.pml import PML


class HostResidency:
    def __init__(self):
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        prior = self.arrays.get(name)
        if prior is not None and prior is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def probe():
    return {"backend": "numpy", "patterns": {
        name: "FMA_V1" for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left")}}


def build(*, poles=2, seed=761):
    grid = Grid(resolution=8.0, cell_size=(2.0, 1.5, 1.0), courant=0.35,
                boundaries="periodic", k_point=(0.2, -0.1, 0.15), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    states = []
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(0.9 + 0.1 * index, 0.06, LORENTZIAN),
            {"Ex": 0.2 + 0.01 * index, "Ey": 0.0, "Ez": 0.3 + 0.01 * index},
            grid, np.complex64)
        fields.polarizations.append(state)
        states.append(state)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    rng = np.random.default_rng(seed)
    def values(scale):
        return (rng.uniform(-scale, scale, grid.shape).astype(np.float32)
                + 1j * rng.uniform(-scale, scale, grid.shape).astype(np.float32))
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = values(0.4)
    for state in states:
        for component in state.driven():
            state.P[component][...] = values(0.2)
            state.P_prev[component][...] = values(0.2)
    return fields, pml, states


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])
    monkeypatch.setattr(shared_predicate, "_metal_backend_reasons", lambda grid: [])


def _host_function(count, axis):
    def run(out, fw, displacement, inverse, *args):
        *poles, kps, kms, nx, ny, nz, n_elem = args
        assert out.shape == (nx, ny, nz) and out.size == n_elem
        source = displacement.copy()
        for pole in poles[:count]:
            source = source - pole
        src = source * inverse
        previous = fw.copy()
        fw[...] = src
        coefficient_shape = [1, 1, 1]
        coefficient_shape[axis] = -1
        value = out.copy()
        value = value + kps.reshape(coefficient_shape) * src
        value = value - kms.reshape(coefficient_shape) * previous
        out[...] = value
    return run


def test_source_uses_both_measured_complex_by_real_orientations():
    source = family.complex_dispersive_e_source(2, 2, "FMA_V1")
    body = source[source.index("kernel void complex_dispersive_e_component"):]
    assert "float2 prev = fw[idx];" in body
    assert "c_mul_field_left(source, inv_e[idx])" in body
    assert "c_mul_coefficient_left(kps[k], src)" in body
    assert "c_mul_coefficient_left(kms[k], prev)" in body
    assert "source = source - p1[idx];" in body


def test_complex_bloch_pml_product_is_admitted(backend_ready):
    fields, pml, _ = build()
    verdict = family.metal_complex_dispersive_e_coverage(
        fields, pml, HostResidency(), probe())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("inactive", "active PML"), ("offdiag", "off-diagonal"),
    ("missing_probe", "probe artifact"), ("no_poles", "no susceptibility"),
    ("conductive", "conductivity"),
])
def test_distinct_products_refuse_by_name(backend_ready, monkeypatch, mutation, needle):
    fields, pml, _ = build()
    record = probe()
    if mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.ones(fields.grid.shape, np.float32)}}
    elif mutation == "missing_probe":
        record = None
        monkeypatch.delenv(shared_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    elif mutation == "conductive":
        fields.set_d_conductivity(np.full(fields.grid.shape, np.float32(0.2)))
    else:
        fields.polarizations.clear()
    verdict = family.metal_complex_dispersive_e_coverage(fields, pml, HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_plan_matches_complex_reference_across_ade_rotation(backend_ready):
    reference, pml, _ = build()
    actual, actual_pml, states = build()
    functions = {(shaders.CONTRACT_OFF, count, axis, "FMA_V1"): _host_function(count, axis)
                 for count in (0, 2) for axis in range(3)}
    plan = family.plan_metal_complex_dispersive_e(
        actual, actual_pml, HostResidency(), probe=probe(), functions=functions)
    assert plan is not None
    for _ in range(3):
        stepping.update_E(reference, pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
        for state in states:
            state.update(actual.drive_field, actual.grid.dt)
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        assert np.array_equal(getattr(reference, name).view(np.uint32),
                              getattr(actual, name).view(np.uint32)), name
    assert plan.launches == 9


def test_pole_aware_complex_pml_product_wires_e_and_its_three_slot_spine(
        backend_ready):
    fields, pml, _ = build()
    for slot in ("step_B", "step_D"):
        verdict = spine.metal_complex_dispersive_pml_curl_coverage(
            fields, pml, slot, HostResidency(), probe())
        assert verdict.covered, (slot, verdict.reasons)
    magnetic = spine.metal_complex_dispersive_pml_magnetic_coverage(
        fields, pml, HostResidency(), probe())
    assert magnetic.covered, magnetic.reasons

    fields.set_d_conductivity(np.full(fields.grid.shape, np.float32(0.2)))
    verdict = spine.metal_complex_dispersive_pml_curl_coverage(
        fields, pml, "step_D", HostResidency(), probe())
    assert not verdict.covered
    assert any("conductivity" in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_only_with_the_complete_pole_aware_spine():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == "complex dispersive PML E"
    assert arm.wired
    spine_arms = [spec for spec in arms.registered()
                  if spec.family == spine.FAMILY]
    assert {(spec.slot, spec.label, spec.wired) for spec in spine_arms} == {
        ("step_B", "complex dispersive PML curl", True),
        ("step_D", "complex dispersive PML curl", True),
        ("update_H", "complex dispersive PML magnetic", True),
    }


def test_native_gate_is_incremental_and_arms_the_pml_dispersive_faults():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_complex_dispersive_update_e.py")
    source = gate.read_text(encoding="utf-8")
    assert "flush()" in source and "os.fsync" in source
    assert "second_pole_dropped" in source
    assert "fw_store_dropped" in source
    assert "pml_kps_kms_order_reversed" in source
