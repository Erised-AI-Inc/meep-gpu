"""Host contract for the real PML dispersive Metal ``update_E`` family.

The kernel steps MEEP's per-chunk constitutive form on the chunk record it binds:
the accumulation in MEEP's grouping where the component's own (dsigw) chunk holds,
``E = src`` elsewhere, ``f_w`` touched only on the former.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import (
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
    membership_from_record,
)
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import dispersive_update_e as family
from meep_gpu.pml import PML


class HostResidency:
    device = "cpu"

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


def _record(fields, pml):
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


def _host_function(count, axis):
    """The kernel's arithmetic on the host: the per-chunk select on the bound record."""
    def run(out, fw, displacement, inverse, *args):
        *poles, kps, kms, nx, ny, nz, n_elem, record = args
        assert out.size == n_elem and out.shape == (nx, ny, nz)
        line = membership_from_record(np.asarray(record), True, axis, out.shape[axis])
        shape = [1, 1, 1]
        shape[axis] = -1
        dw = np.broadcast_to(line.reshape(shape), out.shape)
        source = displacement.copy()
        for pole in poles[:count]:
            source = source - pole
        src = source * inverse
        previous = fw.copy()
        accumulated = out + ((kps.reshape(shape) * src) - (kms.reshape(shape) * previous))
        fw[dw] = src[dw]
        out[...] = np.where(dw, accumulated, src)
    return run


def test_source_keeps_ordered_poles_and_selects_meeps_form_per_chunk():
    source = family.dispersive_e_source(2, 1)
    for line in ("    const bool dw = in_h_y;",
                 "    float prev = dw ? fw[idx] : 0.0f;",
                 "    float source = d_in[idx];",
                 "    source = source - p0[idx];",
                 "    source = source - p1[idx];",
                 "    float src = source * inv_e[idx];",
                 "    if (dw) { fw[idx] = src; }",
                 "    float value = src;",
                 "    if (dw) { value = e_out[idx] + ((kps[j] * src) - (kms[j] * prev)); }",
                 "    e_out[idx] = value;"):
        assert line in source, line
    assert source.index("source - p0") < source.index("source - p1")
    # the previous package spelling: an unconditional pre-store load and two separate
    # accumulations into E
    assert "value = value + kps[j] * src;" not in source
    assert "    float prev = fw[idx];" not in source


def test_the_record_binds_last_and_the_tail_is_the_base_constitutive_select():
    source = family.dispersive_e_source(1, 0)
    assert source.count("[[buffer(") == family.BINDING_COUNT
    assert f"constant ChunkBounds& cb [[buffer({family.RECORD_BUFFER})]]" in source
    flags = shaders.constitutive_membership_flags("E")
    for axis in range(3):
        assert f"const bool dw = {flags[axis]};" in family.dispersive_e_source(1, axis)


def test_the_builder_takes_the_record_as_a_required_keyword(backend_ready):
    fields, pml, _ = build()
    with pytest.raises(TypeError):
        family.plan_metal_dispersive_e(fields, pml, HostResidency(), functions={})
    with pytest.raises(ChunkLayoutRefusal, match="plan_metal_dispersive_e"):
        family.plan_metal_dispersive_e(fields, pml, HostResidency(), functions={},
                                       chunk_bounds=None)


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
        actual, actual_pml, HostResidency(), functions=functions,
        chunk_bounds=_record(actual, actual_pml))
    assert plan is not None
    stepping._region_plan(actual, actual_pml)  # the plan the polarizations read
    split = stepping._region_plan(reference, pml)
    assert all(split.spec("constitutive", c).mode not in ("outside_only", "inside_only")
               for c in ("Ex", "Ey", "Ez")), "the fixture does not exercise both forms"
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
