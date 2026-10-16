"""Host merge bar for the PyTorch/MPS Lorentz/Drude ``update_P`` family.

The device bytes live in ``parity/meep_gpu/gate_metal_ade_update_p.py``.  These
tests pin the parts a device comparison cannot establish by itself: MEEP's
isotropic expression on its four float32 constants, the per-part complex form, the
per-chunk drive select and its record binding, scalar/volume sigma
specializations, all-or-nothing slot coverage, live three-buffer rotation,
residency declarations, and registry composition.  No device is required.
"""

from __future__ import annotations

import os
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
from meep_gpu.dispersion import (
    DRUDE,
    LORENTZIAN,
    PolarizationState,
    RealnumConstants,
    Susceptibility,
)
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, device, launch as metal_launch
from meep_gpu.metal_kernels import ade_update_p as family
from meep_gpu.metal_kernels.complex_dispersive_update_e import MAX_POLES
from meep_gpu.pml import PML


class HostResidency:
    """Torch-free mirror registry for builder and rotation tests."""

    device = "cpu"

    def __init__(self) -> None:
        self._arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self._arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self._arrays[name] = host
        return host

    @property
    def names(self):
        return tuple(sorted(self._arrays))


def _record(fields, pml):
    """The one-rank chunk record of the fixture's grid and layer."""
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


def _kernel(volume: bool, select_axis):
    """A host transcription of the kernel, built from the array path's own products.

    Every product goes through :func:`part_products.scale` (one product per stored
    part), so a complex fixture steps MEEP's two real passes exactly as the shader
    does; the drive select decodes the bound record on the host.
    """
    def run(out, now, previous, *args):
        if volume:
            sigma, drive, two_minus_od, gamma1, omega0dtsqr, gamma1inv, n_elem = args[:7]
        else:
            drive, sigma, two_minus_od, gamma1, omega0dtsqr, gamma1inv, n_elem = args[:7]
            sigma = np.float32(sigma)
        assert out.size == n_elem
        w = drive
        if select_axis is not None:
            fw, ny, nz, record = args[7:]
            assert (ny, nz) == drive.shape[1:]
            line = membership_from_record(np.asarray(record), True, select_axis,
                                          drive.shape[select_axis])
            shape = [1, 1, 1]
            shape[select_axis] = -1
            w = np.where(line.reshape(shape), fw, drive)
        word = np.float32
        part_products.scale(word(two_minus_od), now, out=out)
        out -= part_products.scale(word(gamma1), previous)
        drive_term = part_products.scale(sigma, w)
        part_products.scale(word(omega0dtsqr), drive_term, out=drive_term)
        out += drive_term
        part_products.scale(word(gamma1inv), out, out=out)

    return run


def _functions():
    return {
        ("off", volume, dtype, axis): _kernel(volume, axis)
        for volume in (False, True)
        for dtype in ("float32", "complex64")
        for axis in (None, 0, 1, 2)
    }


def build(*, active_pml: bool = False, complex_storage: bool = False,
          kind: str = LORENTZIAN, volume_sigma: bool = False,
          states: int = 1, counts=(1, 1, 1), seed: int = 20260818):
    grid = Grid(
        resolution=8.0,
        cell_size=(1.0, 0.75, 0.75),
        courant=0.35,
        boundaries="periodic",
        k_point=(0.15, -0.1, 0.05) if complex_storage else (0.0, 0.0, 0.0),
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    # One cell of layer per face on a (8, 6, 6) grid: every component's constitutive
    # line has both forms, so the drive select reads f_w and the stored E.
    pml = PML(grid=grid, thickness=1 if active_pml else 0)
    dtype = np.complex64 if complex_storage else np.float32
    rng = np.random.default_rng(seed)
    built = []
    for state_index in range(states):
        sigma = {}
        for axis, component in enumerate(("Ex", "Ey", "Ez")):
            if not counts[axis]:
                sigma[component] = 0.0
            elif volume_sigma:
                sigma[component] = np.linspace(
                    0.15 + 0.01 * state_index,
                    0.55 + 0.02 * axis,
                    np.prod(grid.shape),
                    dtype=np.float32,
                ).reshape(grid.shape)
            else:
                sigma[component] = 0.25 + 0.05 * axis + 0.01 * state_index
        state = PolarizationState(
            Susceptibility(1.0 + 0.05 * state_index, 0.1, kind),
            sigma,
            grid,
            dtype,
        )
        fields.polarizations.append(state)
        built.append(state)

    for component in ("Ex", "Ey", "Ez"):
        for prefix in (("f_w_", "") if active_pml else ("",)):
            array = getattr(fields, prefix + component)
            low, high = (-0.4, 0.4) if prefix else (0.6, 0.9)
            array.real[...] = rng.uniform(low, high, grid.shape).astype(np.float32)
            if np.iscomplexobj(array):
                array.imag[...] = rng.uniform(low, high, grid.shape).astype(np.float32)

    for state in built:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                array = table[component]
                array.real[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
                if np.iscomplexobj(array):
                    array.imag[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml, built


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

def test_the_real_body_is_meeps_expression_on_the_four_constants():
    for dtype, value_type in (("float32", "float"), ("complex64", "float2")):
        source = family.ade_source(dtype, sigma_is_volume=True)
        assert f"device {value_type}*       p_out" in source
        assert "if (idx >= n_elem) { return; }" in source
        assert "thread_position_in_grid" in source
        for name in family.CONSTANT_NAMES:
            assert f"constant float&              {name}" in source
        for retired in ("c_now", "c_prev", "c_drive"):
            assert retired not in source
    real = family.ade_source("float32", sigma_is_volume=True)
    assert ("p_out[idx] = gamma1inv * (((two_minus_od * p) - (gamma1 * q)) + "
            "(omega0dtsqr * (s * w)));") in real
    assert "((p * c_now) + (c_prev * q))" not in real


def test_the_complex_body_is_two_real_passes_with_no_complex_helper():
    """MEEP's DOCMP2: each product a real word times one part, the sums per part."""
    source = family.ade_source("complex64", sigma_is_volume=True)
    for line in ("    float2 a = two_minus_od * p;",
                 "    float2 b = gamma1 * q;",
                 "    float2 sw = s * w;",
                 "    float2 d = omega0dtsqr * sw;",
                 "    p_out[idx] = gamma1inv * ((a - b) + d);"):
        assert line in source, line
    assert "c_mul" not in source
    assert "fma" not in source


def test_scalar_and_volume_sigma_are_distinct_binding_products():
    scalar = family.ade_source("float32", sigma_is_volume=False)
    volume = family.ade_source("float32", sigma_is_volume=True)
    assert "device const float*          sigma" not in scalar
    assert "constant float&              sigma" in scalar
    assert "device const float*          sigma" in volume
    assert "constant float&              sigma" not in volume


def test_without_an_absorber_the_drive_is_the_stored_e_and_no_record_binds():
    source = family.ade_source("float32", sigma_is_volume=False)
    assert "    float w = drive[idx];" in source
    assert "ChunkBounds" not in source
    assert " fw " not in source and "fw[" not in source
    assert source.count("[[buffer(") == 10


@pytest.mark.parametrize("dtype", ("float32", "complex64"))
@pytest.mark.parametrize("axis,coordinate,name", ((0, "i", "in_h_x"),
                                                  (1, "j", "in_h_y"),
                                                  (2, "k", "in_h_z")))
def test_under_an_absorber_the_drive_is_meeps_per_chunk_select(dtype, axis,
                                                               coordinate, name):
    """``w = f_w ? f_w : f`` on the component's own half-integer membership line."""
    value_type = "float2" if dtype == "complex64" else "float"
    source = family.ade_source(dtype, sigma_is_volume=True, select_axis=axis)
    assert f"    const bool dw = {name};" in source
    assert f"    {value_type} w = dw ? fw[idx] : drive[idx];" in source
    assert f"({coordinate}) >= cbh_" in source
    assert "cbi_" not in source.split("kernel void", 1)[1], (
        "the drive reads the integer lattice; the E side's own axis is half-integer")
    assert f"constant ChunkBounds& cb [[buffer({family.RECORD_BUFFER})]]" in source
    assert source.count("[[buffer(") == family.RECORD_BUFFER + 1


def test_the_select_line_is_the_base_constitutive_kernels_own_predicate():
    from meep_gpu.metal_kernels import shaders

    flags = shaders.constitutive_membership_flags("E")
    for axis in range(3):
        source = family.ade_source("float32", True, select_axis=axis)
        assert f"const bool dw = {flags[axis]};" in source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("active_pml", (False, True))
@pytest.mark.parametrize("complex_storage", (False, True))
@pytest.mark.parametrize("kind", (LORENTZIAN, DRUDE))
@pytest.mark.parametrize("volume_sigma", (False, True))
def test_supported_product_is_admitted(
        backend_ready, active_pml, complex_storage, kind, volume_sigma):
    fields, pml, _states = build(
        active_pml=active_pml,
        complex_storage=complex_storage,
        kind=kind,
        volume_sigma=volume_sigma,
        counts=(1, 0, 1),
    )
    verdict = family.metal_ade_update_p_coverage(
        fields, pml, device.Residency())
    assert verdict.covered, verdict.reasons


def test_the_predicate_never_asks_drive_field_for_a_pointer(backend_ready, monkeypatch):
    """A Fields under a layer with no region plan installed is admitted.

    ``Fields.drive_field`` refuses by name there (and composes a pooled host buffer
    for a split component elsewhere); the plan reads E and f_w in place instead.
    """
    fields, pml, _states = build(active_pml=True)
    assert fields.region_plan is None
    monkeypatch.setattr(type(fields), "drive_field",
                        lambda self, component: pytest.fail("drive_field consulted"))
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons
    plan = family.plan_metal_ade_update_p(
        fields, pml, HostResidency(), functions=_functions(),
        chunk_bounds=_record(fields, pml))
    assert plan is not None


def test_an_active_layer_requires_pml_storage_for_the_select(backend_ready):
    fields, pml, _states = build(active_pml=True)
    fields._pml_active = False
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("the drive select would read no f_w" in reason
               for reason in verdict.reasons), verdict.reasons


def test_a_missing_auxiliary_drive_is_refused_by_name(backend_ready):
    fields, pml, _states = build(active_pml=True, counts=(1, 0, 0))
    fields.f_w_Ex = None
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("drive array f_w_Ex is not allocated" in reason
               for reason in verdict.reasons), verdict.reasons


def test_an_invalid_kind_is_refused_by_name(backend_ready):
    fields, pml, states = build()
    states[0].susceptibility = type("Bad", (), {"kind": "noisy"})()
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("kind 'noisy'" in reason for reason in verdict.reasons)


def test_one_refused_state_refuses_the_whole_update_p_slot(backend_ready):
    fields, pml, states = build(states=2)
    word = np.float32
    states[1].constants = RealnumConstants(word(1.0), word(np.nan), word(0.25), word(1.0))
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("polarization 1" in reason and "gamma1=nan is not finite" in reason
               for reason in verdict.reasons), verdict.reasons


def test_a_constant_that_is_not_a_float32_word_is_refused_by_name(backend_ready):
    fields, pml, states = build()
    constants = states[0].constants
    states[0].constants = constants._replace(omega0dtsqr=float(constants.omega0dtsqr) + 1e-12)
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("omega0dtsqr" in reason and "not a float32 word" in reason
               for reason in verdict.reasons), verdict.reasons


def test_the_retired_triple_is_not_read(backend_ready):
    """The predicate reads ``PolarizationState.constants``; no ``_coefficients`` exists."""
    fields, pml, states = build()
    assert not hasattr(states[0], "_coefficients")
    states[0].constants = (1.0, 2.0, 3.0)
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("four realnum constants" in reason for reason in verdict.reasons)


def test_no_live_pole_is_a_refusal_not_a_vacuous_kernel(backend_ready):
    fields, pml, _states = build(counts=(0, 0, 0))
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("no driven component" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("active_pml", (False, True))
@pytest.mark.parametrize("complex_storage", (False, True))
@pytest.mark.parametrize("volume_sigma", (False, True))
def test_plan_rotates_live_buffers_and_matches_the_reference(
        backend_ready, active_pml, complex_storage, volume_sigma):
    reference, pml, _ = build(
        active_pml=active_pml, complex_storage=complex_storage,
        volume_sigma=volume_sigma, states=2, counts=(1, 0, 1))
    actual, actual_pml, states = build(
        active_pml=active_pml, complex_storage=complex_storage,
        volume_sigma=volume_sigma, states=2, counts=(1, 0, 1))
    if active_pml:
        stepping._region_plan(reference, pml)
    residency = HostResidency()
    plan = family.plan_metal_ade_update_p(
        actual, actual_pml, residency, functions=_functions(),
        chunk_bounds=_record(actual, actual_pml))
    assert plan is not None
    if active_pml:
        assert {"Ex", "f_w_Ex", "Ez", "f_w_Ez"} <= set(residency.names)
        assert plan.chunk_bounds is not None
    else:
        assert not any(name.startswith("f_w_") for name in residency.names)
        assert plan.chunk_bounds is None
    initial_ids = [
        tuple(id(state.P[c]) for c in state.driven())
        + tuple(id(state.P_prev[c]) for c in state.driven())
        + (id(state._scratch),)
        for state in states
    ]
    for _ in range(4):
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
    assert plan.runs == 4
    assert plan.launches == 4 * sum(len(state.driven()) for state in states)
    for expected_state, actual_state, physical_ids in zip(
            reference.polarizations, states, initial_ids):
        current_ids = (
            tuple(id(actual_state.P[c]) for c in actual_state.driven())
            + tuple(id(actual_state.P_prev[c]) for c in actual_state.driven())
            + (id(actual_state._scratch),)
        )
        assert sorted(current_ids) == sorted(physical_ids)
        for component in actual_state.driven():
            assert np.array_equal(
                expected_state.P[component].view(np.uint32),
                actual_state.P[component].view(np.uint32))
            assert np.array_equal(
                expected_state.P_prev[component].view(np.uint32),
                actual_state.P_prev[component].view(np.uint32))


def test_the_layer_fixture_splits_every_driven_component():
    """The PML rotation case reads both drives: each component has both forms."""
    fields, pml, _ = build(active_pml=True, counts=(1, 0, 1))
    plan = stepping._region_plan(fields, pml)
    for component in ("Ex", "Ez"):
        assert plan.spec("constitutive", component).mode not in (
            "outside_only", "inside_only"), component


def test_the_builder_takes_the_record_as_a_required_keyword(backend_ready):
    fields, pml, _ = build(active_pml=True)
    with pytest.raises(TypeError):
        family.plan_metal_ade_update_p(fields, pml, HostResidency(),
                                       functions=_functions())
    with pytest.raises(ChunkLayoutRefusal, match="plan_metal_ade_update_p"):
        family.plan_metal_ade_update_p(fields, pml, HostResidency(),
                                       functions=_functions(), chunk_bounds=None)


def test_without_an_absorber_the_record_is_not_bound(backend_ready):
    fields, pml, _ = build(active_pml=False)
    plan = family.plan_metal_ade_update_p(fields, pml, HostResidency(),
                                          functions=_functions(), chunk_bounds=None)
    assert plan is not None
    assert all(entry.select_axis is None for entry in plan.entries)


def test_refused_builder_never_compiles(backend_ready, monkeypatch):
    fields, pml, states = build()
    word = np.float32
    states[0].constants = RealnumConstants(word(np.inf), word(0.0), word(1.0), word(1.0))
    monkeypatch.setattr(
        family, "compile_ade",
        lambda *args, **kwargs: pytest.fail("refused builder compiled"))
    assert family.plan_metal_ade_update_p(
        fields, pml, HostResidency(), chunk_bounds=_record(fields, pml)) is None


def test_family_is_registered_once_on_update_p():
    matches = [spec for spec in arms.registered("update_P")
               if spec.family == family.FAMILY]
    assert len(matches) == 1
    assert matches[0].label == "ADE update_P"
    assert matches[0].wired


def test_composer_selects_update_p_and_requires_the_array_update_e_sync(
        backend_ready, monkeypatch):
    """ADE fills update_P while update_E stays on the array path, so the mirror
    handoff this test exists to exercise is a real one.

    A pole count above MAX_POLES refuses update_E on the kernel's own cap rather
    than on an environmental clause (an unset probe or subnormal policy), so the
    array-path handoff is guaranteed by the configuration itself.
    """
    poles = MAX_POLES + 1
    fields, pml, _states = build(active_pml=True, complex_storage=True,
                                 states=poles)
    monkeypatch.setattr(
        family, "compile_ade",
        lambda dtype, volume, contract, select_axis=None:
        _functions()[(contract, volume, dtype, select_axis)])
    residency = HostResidency()
    unsafe = metal_launch.plan_step(fields, pml, residency, sources=())
    assert unsafe.selected["update_P"] == "ADE update_P"
    assert "update_E" not in unsafe.selected, unsafe.selected
    assert any(f"more than MAX_POLES={MAX_POLES}" in reason
               for reason in unsafe.reasons.get("update_E", ())), \
        unsafe.reasons.get("update_E")
    assert not unsafe.residency.covered
    assert any("update_E runs on the array path" in reason
               for reason in unsafe.residency.reasons)

    fields, pml, _states = build(active_pml=True, complex_storage=True,
                                 states=poles)
    residency = HostResidency()
    safe = metal_launch.plan_step(
        fields, pml, residency, sources=(), synced=("step_B", "update_E"))
    assert safe.selected["update_P"] == "ADE update_P"
    assert safe.residency.covered, safe.residency.reasons


def test_gate_exists_and_records_incremental_source_bound_evidence():
    gate = os.path.join(os.path.dirname(__file__), os.pardir, "parity", "meep_gpu",
                        "gate_metal_ade_update_p.py")
    source = open(os.path.abspath(gate), encoding="utf-8").read()
    for needle in ("flush=True", "jsonl", "source_provenance", "mutation",
                   "previous_folded_three_term_form", "drive_select_removed"):
        assert needle in source
