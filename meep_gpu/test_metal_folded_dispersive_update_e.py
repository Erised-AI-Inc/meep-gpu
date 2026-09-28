"""Host contracts for the Metal folded real PML dispersive ``update_E`` arm."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import dispersive_update_e as ordinary
from meep_gpu.metal_kernels import folded_dispersive_update_e as family
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


def build(*, poles=2, kind=LORENTZIAN, seed=741):
    grid = Grid(resolution=8.0, cell_size=(1.75, 2.0, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic", symmetry=(Mirror("Y", 1),),
                xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    for index in range(poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.9 + index * 0.1, 0.05 + index * 0.01,
                           kind if index % 2 == 0 else DRUDE),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.float32))
    fields.enable_pml_storage()
    thickness = ((2, 2), (0, 2), (0, 0))
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey",
                 "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr("meep_gpu.metal_kernels.symmetry._metal_backend_reasons",
                        lambda grid: [])


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
        out[...] = out + kps.reshape(coefficient_shape) * src
        out[...] = out - kms.reshape(coefficient_shape) * previous
    return run


@pytest.mark.parametrize("kind", (LORENTZIAN, DRUDE))
def test_folded_real_dispersive_product_is_admitted(backend_ready, kind):
    fields, pml = build(kind=kind)
    verdict = family.folded_dispersive_e_coverage(fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons


def test_folded_predicate_reuses_the_fold_contract_but_not_the_unfolded_arm(backend_ready):
    fields, pml = build()
    assert family.folded_dispersive_e_coverage(fields, pml, HostResidency()).covered
    ordinary_verdict = ordinary.metal_dispersive_e_coverage(fields, pml, HostResidency())
    assert any("mirror plane" in reason for reason in ordinary_verdict.reasons)


@pytest.mark.parametrize("mutation, needle", [
    ("no_poles", "no susceptibility"),
    ("offdiag", "own fused stencil"),
    ("inactive", "no active PML"),
    ("unfolded", "no mirror plane"),
])
def test_intersection_boundaries_refuse_by_name(backend_ready, mutation, needle):
    fields, pml = build()
    if mutation == "no_poles":
        fields.polarizations.clear()
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.ones(fields.grid.shape,
                                                               np.float32)}}
    elif mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    else:
        grid = Grid(resolution=8.0, cell_size=(1.75, 2.0, 0.0), dimensions=2,
                    courant=0.35, boundaries="periodic", xp=np)
        fields.grid = grid
    verdict = family.folded_dispersive_e_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_shared_builder_matches_stepping_over_the_folded_stored_extent(backend_ready):
    reference, pml = build(poles=2)
    actual, actual_pml = build(poles=2)
    functions = {(shaders.CONTRACT_OFF, count, axis): _host_function(count, axis)
                 for count in (0, 1, 2) for axis in range(3)}
    plan = family.plan_folded_dispersive_e(
        actual, actual_pml, HostResidency(), functions=functions)
    assert isinstance(plan, ordinary.MetalDispersiveEPlan)
    for _ in range(3):
        stepping.update_E(reference, pml)
        plan.run()
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        assert np.array_equal(getattr(reference, name).view(np.uint32),
                              getattr(actual, name).view(np.uint32)), name
    assert plan.launches == 9


def test_arm_is_wired_in_the_experimental_composer_only():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == family.LABEL
    assert arm.wired
