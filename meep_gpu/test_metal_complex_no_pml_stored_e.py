"""Host contract for complex/Bloch no-PML stored-E Metal update_E."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.metal_kernels import complex_no_pml_stored_e as family
from meep_gpu.pml import PML


class HostResidency:
    def __init__(self):
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


def build(*, poles=2, bloch=True, complex_storage=True, seed=41):
    grid = Grid(resolution=6.0, cell_size=(1.0, 0.75, 0.5), courant=0.3,
                boundaries="periodic", k_point=(0.2, -0.1, 0.15) if bloch else (0, 0, 0),
                xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    def values(scale):
        real = rng.uniform(-scale, scale, grid.shape).astype(np.float32)
        if not complex_storage:
            return real
        return real + 1j * rng.uniform(-scale, scale, grid.shape).astype(np.float32)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = values(0.4)
    states = []
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(1.0 + index * 0.1, 0.05, LORENTZIAN),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.complex64 if complex_storage else np.float32)
        fields.polarizations.append(state)
        states.append(state)
        for component in state.driven():
            for array in (state.P[component], state.P_prev[component]):
                array[...] = values(0.2)
    return fields, PML(grid=grid, thickness=0), states


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _host_function(count):
    def run(out, displacement, inverse, *args):
        *poles, n_elem = args
        assert out.size == n_elem
        source = displacement.copy()
        for pole in poles[:count]:
            source = source - pole
        out[...] = source * inverse
    return run


def test_source_uses_float2_and_only_the_probed_field_left_orientation():
    source = family.complex_stored_e_source(2, "FMA_V1")
    assert "device float2*" in source and "[[buffer(11)]]" in source
    assert "source = source - p0[idx];" in source
    assert "source = source - p1[idx];" in source
    assert "c_mul_field_left(source, inv_e[idx])" in source
    body = source[source.index("kernel void complex_stored_e_component"):]
    assert "c_mul_coefficient_left" not in body


def test_complex_bloch_no_pml_product_is_admitted(backend_ready):
    fields, pml, _ = build()
    verdict = family.metal_complex_stored_e_coverage(
        fields, pml, HostResidency(), probe())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "complex64 storage"), ("pml", "active PML"),
    ("offdiag", "off-diagonal"), ("missing_probe", "probe artifact"),
])
def test_distinct_products_refuse_by_name(backend_ready, monkeypatch, mutation, needle):
    fields, pml, _ = build()
    record = probe()
    if mutation == "real":
        fields, pml, _ = build(complex_storage=False, bloch=False)
    elif mutation == "pml":
        pml = PML(grid=fields.grid, thickness=1)
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.ones(fields.grid.shape, np.float32)}}
    else:
        record = None
        monkeypatch.delenv(shared_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    verdict = family.metal_complex_stored_e_coverage(fields, pml, HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_plan_matches_complex_reference_across_pointer_rotation(backend_ready):
    reference, pml, _ = build()
    actual, actual_pml, states = build()
    functions = {(shaders.CONTRACT_OFF, count, "FMA_V1"): _host_function(count)
                 for count in (0, 2)}
    plan = family.plan_metal_complex_stored_e(
        actual, actual_pml, HostResidency(), probe=probe(), functions=functions)
    assert plan is not None
    physical = {id(array) for state in states for component in state.driven()
                for array in (state.P[component], state.P_prev[component], state._scratch)}
    for _ in range(3):
        stepping.update_E(reference, pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
        for state in states:
            state.update(actual.drive_field, actual.grid.dt)
    for component in ("Ex", "Ey", "Ez"):
        assert np.array_equal(getattr(reference, component).view(np.uint32),
                              getattr(actual, component).view(np.uint32))
    observed = {id(array) for state in states for component in state.driven()
                for array in (state.P[component], state.P_prev[component], state._scratch)}
    assert observed == physical


def test_arm_is_wired_after_direct_and_complete_step_evidence():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == "complex no-PML stored E"
    assert arm.wired


def test_native_gate_is_incremental_and_arms_ordered_constitutive_faults():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_complex_no_pml_stored_e.py")
    source = gate.read_text(encoding="utf-8")
    assert "flush=True" in source and "os.fsync" in source
    assert "second_pole_dropped" in source
    assert "ordered_subtraction_reversed" in source
    assert "inverse_epsilon_omitted" in source
