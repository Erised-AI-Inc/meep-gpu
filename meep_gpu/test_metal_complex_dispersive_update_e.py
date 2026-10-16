"""Host contract for complex/Bloch PML dispersive Metal ``update_E``."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import part_products, stepping
from meep_gpu.chunk_map import (
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
    membership_from_record,
)
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_dispersive_update_e as family
from meep_gpu.metal_kernels import complex_dispersive_spine as spine
from meep_gpu.metal_kernels import complex_fields as shared_predicate
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


def _record(fields, pml):
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


def _host_function(count, axis):
    """The kernel's arithmetic on the host: every product one per part, the select on
    the bound record."""
    def run(out, fw, displacement, inverse, *args):
        *poles, kps, kms, nx, ny, nz, n_elem, record = args
        assert out.shape == (nx, ny, nz) and out.size == n_elem
        line = membership_from_record(np.asarray(record), True, axis, out.shape[axis])
        shape = [1, 1, 1]
        shape[axis] = -1
        dw = np.broadcast_to(line.reshape(shape), out.shape)
        source = displacement.copy()
        for pole in poles[:count]:
            source = source - pole
        src = part_products.scale(inverse, source)
        previous = fw.copy()
        accumulated = out + (part_products.scale(kps.reshape(shape), src)
                             - part_products.scale(kms.reshape(shape), previous))
        fw[dw] = src[dw]
        out[...] = np.where(dw, accumulated, src)
    return run


def test_source_selects_meeps_form_per_chunk_with_part_wise_products():
    source = family.complex_dispersive_e_source(2, 2)
    body = source[source.index("kernel void complex_dispersive_e_component"):]
    for line in ("    const bool dw = in_h_z;",
                 "    float2 prev = dw ? fw[idx] : float2(0.0f, 0.0f);",
                 "    float2 src = c_mul_field_left(source, inv_e[idx]);",
                 "    if (dw) { fw[idx] = src; }",
                 "    float2 value = src;",
                 "    if (dw) { value = e_out[idx] + (c_mul_coefficient_left(kps[k], src) "
                 "- c_mul_coefficient_left(kms[k], prev)); }",
                 "    e_out[idx] = value;",
                 "    source = source - p1[idx];"):
        assert line in body, line
    assert "value = value + c_mul_coefficient_left" not in body
    assert body.count("[[buffer(") == family.BINDING_COUNT
    assert f"constant ChunkBounds& cb [[buffer({family.RECORD_BUFFER})]]" in body


def test_the_tail_is_the_complex_base_constitutive_kernels_own_lines():
    """The per-component lines of complex_fields' E constitutive, renamed."""
    certified = shared_predicate.bloch_constitutive_source("E")
    for axis, coordinate in ((0, "i"), (1, "j"), (2, "k")):
        line = (f"    if (dw{axis}) {{ a{axis} = f{axis}[ii] + (c_mul_coefficient_left("
                f"kp_{axis}, src{axis}) - c_mul_coefficient_left(km_{axis}, prev{axis})); }}")
        assert line in certified
        renamed = (line.replace(f"dw{axis}", "dw").replace(f"a{axis} =", "value =")
                   .replace(f"f{axis}[ii]", "e_out[idx]")
                   .replace(f"kp_{axis}", f"kps[{coordinate}]")
                   .replace(f"km_{axis}", f"kms[{coordinate}]")
                   .replace(f"src{axis}", "src").replace(f"prev{axis}", "prev"))
        assert renamed in family.complex_dispersive_e_source(1, axis)


def test_the_builder_takes_the_record_as_a_required_keyword(backend_ready):
    fields, pml, _ = build()
    with pytest.raises(TypeError):
        family.plan_metal_complex_dispersive_e(fields, pml, HostResidency(), functions={})
    with pytest.raises(ChunkLayoutRefusal, match="plan_metal_complex_dispersive_e"):
        family.plan_metal_complex_dispersive_e(fields, pml, HostResidency(), functions={},
                                               chunk_bounds=None)


def test_complex_bloch_pml_product_is_admitted(backend_ready):
    fields, pml, _ = build()
    verdict = family.metal_complex_dispersive_e_coverage(
        fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("inactive", "active PML"), ("offdiag", "off-diagonal"),
    ("no_poles", "no susceptibility"),
    ("conductive", "conductivity"),
])
def test_distinct_products_refuse_by_name(backend_ready, mutation, needle):
    fields, pml, _ = build()
    if mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.ones(fields.grid.shape, np.float32)}}
    elif mutation == "conductive":
        fields.set_d_conductivity(np.full(fields.grid.shape, np.float32(0.2)))
    else:
        fields.polarizations.clear()
    verdict = family.metal_complex_dispersive_e_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_plan_matches_complex_reference_across_ade_rotation(backend_ready):
    reference, pml, _ = build()
    actual, actual_pml, states = build()
    functions = {(shaders.CONTRACT_OFF, count, axis): _host_function(count, axis)
                 for count in (0, 2) for axis in range(3)}
    plan = family.plan_metal_complex_dispersive_e(
        actual, actual_pml, HostResidency(), functions=functions,
        chunk_bounds=_record(actual, actual_pml))
    assert plan is not None
    stepping._region_plan(actual, actual_pml)  # the plan the polarizations read
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
            fields, pml, slot, HostResidency())
        assert verdict.covered, (slot, verdict.reasons)
    magnetic = spine.metal_complex_dispersive_pml_magnetic_coverage(
        fields, pml, HostResidency())
    assert magnetic.covered, magnetic.reasons

    fields.set_d_conductivity(np.full(fields.grid.shape, np.float32(0.2)))
    verdict = spine.metal_complex_dispersive_pml_curl_coverage(
        fields, pml, "step_D", HostResidency())
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
