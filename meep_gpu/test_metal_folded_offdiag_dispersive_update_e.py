"""Host contracts for Metal's folded tensor-row dispersive intersection."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import arms, device
from meep_gpu.metal_kernels import folded_offdiag_dispersive_update_e as family
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


def build(*, counts: Sequence[int] = (2, 1, 2), rows: bool = True,
          fold: bool = True, seed: int = 906) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=(2.0, 1.5, 1.25), courant=0.35,
                boundaries="periodic", symmetry=(Mirror("Y", 1),) if fold else (),
                xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: np.float32(1.0) / value for name, value in epsilon.items()}
    tensor_rows = ({"Ex": {"Ey": rng.uniform(-0.2, 0.2, shape).astype(np.float32),
                            "Ez": rng.uniform(-0.2, 0.2, shape).astype(np.float32)},
                    "Ey": {"Ez": rng.uniform(-0.2, 0.2, shape).astype(np.float32)}}
                   if rows else None)
    fields.set_epsilon_volumes(epsilon, inverse, tensor_rows)
    for index in range(max(counts)):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + 0.1 * index, 0.06 + 0.01 * index, "lorentzian"),
            {name: 0.18 + 0.02 * index if counts[axis] > index else 0.0
             for axis, name in enumerate(("Ex", "Ey", "Ez"))}, grid, np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (0, 2) if fold else (2, 2), (2, 2)))
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey",
                 "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
    return fields, pml


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(
        "meep_gpu.metal_kernels.folded_offdiag_update_e._metal_backend_reasons",
        lambda grid: [])


def test_the_packed_signature_uses_all_31_available_bindings_once():
    source = family.folded_offdiag_dispersive_source(
        (1, 1, 1, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), (8, 8, 8))
    assert family.BINDING_COUNT == device.MAX_BUFFER_BINDINGS == 31
    assert source.count("[[buffer(") == family.BINDING_COUNT
    assert "p[7u * n_elem + uint(index)]" in source
    assert source.index("p[0u * n_elem + uint(index)]") < source.index(
        "p[7u * n_elem + uint(index)]")


def test_partner_stencil_reads_ordered_d_minus_p_not_raw_d():
    source = family.folded_offdiag_dispersive_source(
        (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), (2, 1, 2))
    assert "dmp1(g1, p1, i * nyz + dj * nzi + k, n_elem)" in source
    assert "float near_00 = dmp1(g1, p1, ii, n_elem)" in source
    assert "at_y ? -dn_00 : dn_00" in source


def test_folded_tensor_dispersive_product_and_max_pole_count_are_admitted(backend_ready):
    fields, pml = build(counts=(family.MAX_POLES,) * 3)
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("no_rows", "no off-diagonal"),
    ("no_poles", "no susceptibility"),
    ("unfolded", "no mirror plane"),
])
def test_intersection_boundaries_refuse_by_name(backend_ready, mutation, needle):
    fields, pml = build(rows=mutation != "no_rows", fold=mutation != "unfolded")
    if mutation == "no_poles":
        fields.polarizations.clear()
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_builder_allocates_three_persistent_max_pole_packs(backend_ready):
    fields, pml = build(counts=(2, 1, 2))
    residency = HostResidency()
    plan = family.plan_folded_offdiag_dispersive(
        fields, pml, residency, contract_variants=())
    assert plan is not None
    assert plan.counts == (2, 1, 2)
    assert tuple(plan._packs) == ("Ex", "Ey", "Ez")
    assert all(pack.shape == (family.MAX_POLES, plan.n_elem)
               for pack in plan._packs.values())
    assert residency.arrays["row:Ey:Ex"] is fields.inverse_epsilon_for("Ey")
    assert residency.arrays["row:Ez:Ex"] is fields.inverse_epsilon_for("Ez")


def test_arm_uses_a_device_side_live_pole_refresh_when_composed():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == family.LABEL
    assert arm.wired
    source = inspect.getsource(family.MetalFoldedOffdiagDispersivePlan._refresh_packs)
    assert "tensor_for_host(state.P[component])" in source
