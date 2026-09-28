"""Host contract for the real PML dispersive Metal ``update_E`` family."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import dispersive_update_e as family
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


def build(*, poles=2, kind=LORENTZIAN, seed=307):
    grid = Grid(resolution=8.0, cell_size=(2.0, 1.5, 1.0), courant=0.35,
                boundaries="periodic", xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    states = []
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(0.9 + 0.1 * index, 0.06 + 0.01 * index,
                           kind if index % 2 == 0 else DRUDE),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.float32)
        fields.polarizations.append(state)
        states.append(state)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    rng = np.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
    for state in states:
        for component in state.driven():
            for array in (state.P[component], state.P_prev[component]):
                array[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml, states


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _host_function(count, axis):
    def run(out, fw, displacement, inverse, *args):
        *poles, kps, kms, nx, ny, nz, n_elem = args
        assert out.size == n_elem and out.shape == (nx, ny, nz)
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


def test_source_keeps_ordered_poles_prestored_auxiliary_and_two_accumulations():
    source = family.dispersive_e_source(2, 1)
    assert "float prev = fw[idx];" in source
    assert "source = source - p0[idx];" in source
    assert "source = source - p1[idx];" in source
    assert "fw[idx] = src;" in source
    assert "value = value + kps[j] * src;" in source
    assert "value = value - kms[j] * prev;" in source
    assert "value + (kps[j] * src - kms[j] * prev)" not in source


@pytest.mark.parametrize("kind", (LORENTZIAN, DRUDE))
def test_real_pml_dispersive_product_is_admitted(backend_ready, kind):
    fields, pml, _ = build(kind=kind)
    verdict = family.metal_dispersive_e_coverage(fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("inactive", "active PML"), ("no_storage", "PML storage mode"),
    ("offdiag", "off-diagonal"), ("no_poles", "no susceptibility"),
])
def test_distinct_products_refuse_by_name(backend_ready, mutation, needle):
    fields, pml, _ = build()
    if mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif mutation == "no_storage":
        fields._pml_active = False
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.ones(fields.grid.shape, np.float32)}}
    else:
        fields.polarizations.clear()
    verdict = family.metal_dispersive_e_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_plan_matches_reference_and_resolves_p_after_ade_rotation(backend_ready):
    reference, pml, _ = build(poles=2)
    actual, actual_pml, states = build(poles=2)
    functions = {(shaders.CONTRACT_OFF, count, axis): _host_function(count, axis)
                 for count in (0, 2) for axis in range(3)}
    plan = family.plan_metal_dispersive_e(
        actual, actual_pml, HostResidency(), functions=functions)
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
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        assert np.array_equal(getattr(reference, name).view(np.uint32),
                              getattr(actual, name).view(np.uint32)), name
    observed = {id(array) for state in states for component in state.driven()
                for array in (state.P[component], state.P_prev[component], state._scratch)}
    assert observed == physical
    assert plan.launches == 9


def test_arm_is_wired_after_device_and_whole_step_evidence():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == "dispersive PML E"
    assert arm.wired
