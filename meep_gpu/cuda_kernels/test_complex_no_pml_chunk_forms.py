"""The CUDA complex no-absorber lifters transcribe the array path's forms, read off their
text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. Three products lift the complex
no-absorber curl (``complex_no_pml_kernels``, MEEP's left-to-right stencil with its
partner-presence forms and the D curl's Bloch connection) into one launch each:

- ``no_pml_complex_fused_electric_pair``: ``step_D`` and ``update_E`` (plain and
  conductive curl tails, the pole bank subtracted in registration order);
- ``complex_no_pml_three_slot_dispersive_weld``: ``step_D``, ``update_E`` and
  ``update_P``, one launch per component with the host rotation between;
- ``complex_no_pml_offdiag_fused_electric_pair``: ``step_D``, ``zero_metal_D`` and the
  off-diagonal ``update_E`` (MEEP's double row per part, per-chunk presence, two
  connection tables), written to scratch and rotated in.

Each is LAUNCHED THROUGH ITS OWN LAUNCHER, loaded with NumPy wearing CuPy's name, whose
compiled kernel is replaced by ``test_complex_chunk_forms``' interpreter, which EXECUTES
the emitted device text in float32 NumPy -- its helpers too, from their own text -- so the
launcher's argument order and its record, connection and presence bindings are exercised
with the text. Every array the products write, and every one they must leave alone, is
compared word for word (sign of zero included) with the array path's own pass sequence on
a copy of the same seeded full-volume state.

The cases cover a generic Bloch phase on two or three axes (the unfused connection), the
Brillouin edge (the per-part negation), metallic walls, a two-dimensional grid (the
partner-presence forms on the invariant axis), the conductive tail, one and two poles with
volume and uniform sigma, the off-diagonal corpus row mask, and planted signed zeros of
both signs. Each case asserts that the products moved what they must move (a vacuity
floor). A battery of planted defects -- each a rewrite of one emitted text -- must change
a word.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import numpy
import pytest

from .. import part_products, stepping
from ..chunk_map import ChunkLayout, chunk_bounds_record
from ..fields import Fields
from ..grid import Grid
from . import complex_emitter
from .test_complex_chunk_forms import (F32, InterpretedKernel, _NumpyWearingCupysName,
                                       bind_launcher, bits, differing, fused_oracle,
                                       launcher)

__all__ = ["launcher"]  # the interpreter-backed complex_pml_kernels fixture, shared


#: The array path's pass sequence each product replaces (the in-seam passes between
#: step_D and update_E included: a fold-free no-absorber grid runs them as no-ops or as
#: the wall clear).
D_TO_E_PASSES = ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
                 "update_E")
THREE_SLOT_PASSES = D_TO_E_PASSES + ("update_P",)

#: The volumes compared on every case; the products write D and E (and P), and must leave
#: B and H as they were.
STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: name -> (cell, dimensions, boundaries, k_point, conductive, poles, uniform sigma).
#: NO CASE CARRIES A WALL: the D -> E pair and the three-slot weld built on it refuse a
#: walled run by name (``covers_no_pml_complex_fused_electric_pair``'s wall clause), so a
#: wall here would compare a launch the predicate never makes; the off-diagonal weld,
#: which carries ``zero_metal_D``, takes walled cases below.
CASES: Dict[str, Tuple] = {
    "np_bloch3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.2, 0.31, 0.0), False, 0,
                   False),
    # The Brillouin edge on x (a negation) and a generic phase on y, on a 2-D grid: the
    # z partner is absent, so two components take MEEP's one-partner forms.
    "np_edge2d_pole": ((10.0, 9.0, 0.0), 2, ("periodic",) * 3, (0.05, 0.13, 0.0), False,
                       1, False),
    "np_k_zero3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.0, 0.0, 0.0), False, 0,
                    False),
    "np_conductive_poles3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.2, 0.0, 0.29),
                              True, 2, False),
    "np_conductive2d_uniform": ((10.0, 9.0, 0.0), 2, ("periodic",) * 3, (0.0, 0.21, 0.0),
                                True, 1, True),
    "np_uniform_sigma3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.0, 0.17, 0.0), False,
                           2, True),
}


def build(case: str, plant_zeros: bool, seed: int = 31):
    """A seeded complex no-absorber run: every stored word nonzero (or a planted zero)."""
    from ..dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    cell, dimensions, boundaries, k_point, conductive, poles, uniform = CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    if conductive:
        fields.set_d_conductivity(rng.uniform(0.0, 0.4, size=shape).astype(F32))
        fields.set_b_conductivity(rng.uniform(0.0, 0.4, size=shape).astype(F32))
    for index in range(poles):
        sigma = (F32(0.35 + 0.1 * index) if uniform
                 else rng.uniform(0.1, 0.6, size=shape).astype(F32))
        state = PolarizationState(Susceptibility(frequency=1.1 + index, gamma=0.05),
                                  sigma, grid, fields._field_dtype())
        fields.polarizations.append(state)

    def _random():
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        return part_products.from_parts(parts[0], parts[1], numpy.complex64)

    for name in present(fields, STATE):
        getattr(fields, name)[...] = _random()
    for state in fields.polarizations:
        for component in state.P:
            state.P[component][...] = _random() * numpy.complex64(1e-2)
            state.P_prev[component][...] = _random() * numpy.complex64(1e-2)
    return fields, grid


def record_of(grid) -> numpy.ndarray:
    """The one-rank record of an unlayered grid: only its partner-presence word is read."""
    return numpy.ascontiguousarray(chunk_bounds_record(ChunkLayout.one_rank(grid, None)),
                                   dtype=numpy.int32)


def present(fields, names) -> Tuple[str, ...]:
    """The named volumes this run allocates (without a layer H may not be stored)."""
    return tuple(name for name in names if getattr(fields, name, None) is not None)


def snapshot(fields) -> Dict[str, numpy.ndarray]:
    """Every compared word: the fields and each polarization's two stored P sets."""
    out = {name: numpy.array(getattr(fields, name), copy=True)
           for name in present(fields, STATE)}
    for index, state in enumerate(fields.polarizations):
        for component in sorted(state.P):
            out[f"P{index}_{component}"] = numpy.array(state.P[component], copy=True)
            out[f"Pprev{index}_{component}"] = numpy.array(state.P_prev[component],
                                                           copy=True)
    return out


def restore(fields, state) -> None:
    for name in present(fields, STATE):
        getattr(fields, name)[...] = state[name]
    for index, polarization in enumerate(fields.polarizations):
        for component in polarization.P:
            polarization.P[component][...] = state[f"P{index}_{component}"]
            polarization.P_prev[component][...] = state[f"Pprev{index}_{component}"]


def moved(start, want, names) -> bool:
    return all(int(numpy.count_nonzero(bits(want[n]) != bits(start[n]))) > 0 for n in names)


@pytest.fixture()
def host(launcher, monkeypatch):
    """NumPy wearing CuPy's name, and the interpreter-backed ``complex_pml_kernels``."""
    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    return launcher


PARAMETERS = [(case, plant) for case in sorted(CASES) for plant in (False, True)]
IDS = [f"{c}-{'zeros' if p else 'random'}" for c, p in PARAMETERS]


# =============================================================================
# 1. The D -> E pair: step_D and update_E in one launch
# =============================================================================

def _pair_launch(fields, grid, host, text=None):
    from . import no_pml_complex_fused_electric_pair as family  # noqa: PLC0415

    arm = family.no_pml_complex_curl_arm(fields, "step_D")
    flags, values = host.bloch_phase_arguments(grid, True)
    kernel = InterpretedKernel(text if text is not None else
                               family.no_pml_complex_fused_electric_pair_source(
                                   arm == "conductive"))
    family.launch_no_pml_complex_fused_electric_pair(
        fields, host.complex_boundary_codes(grid), flags, values, float(grid.dt / grid.dx),
        arm, kernel=kernel, chunk_bounds=record_of(grid))
    return kernel


def test_no_pair_case_carries_a_wall_the_pair_refuses():
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    for case in CASES:
        _fields, grid = build(case, False)
        assert not any(zero_metal_axes(grid)), case


@pytest.mark.parametrize("case,plant", PARAMETERS, ids=IDS)
def test_the_fused_pair_text_is_the_array_path_word_for_word(case, plant, host):
    fields, grid = build(case, plant)
    start = snapshot(fields)
    fused_oracle(D_TO_E_PASSES, fields, None)
    want = snapshot(fields)
    restore(fields, start)
    kernel = _pair_launch(fields, grid, host)
    assert kernel.launches == 1
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant,
                                                {k: v for k, v in diff.items() if v})
    assert moved(start, want, ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"))


# =============================================================================
# 2. The three-slot weld: step_D, update_E and update_P, one launch per component
# =============================================================================

POLE_PARAMETERS = [(case, plant) for case, plant in PARAMETERS if CASES[case][5] > 0]
POLE_IDS = [f"{c}-{'zeros' if p else 'random'}" for c, p in POLE_PARAMETERS]


def _three_slot_launch(fields, grid, host, monkeypatch, transform=None):
    from . import complex_no_pml_three_slot_dispersive_weld as family  # noqa: PLC0415

    launches = []

    def _get_kernel(conductive, component_index, count, kinds):
        text = family.three_slot_no_pml_complex_source(conductive, component_index,
                                                       count, tuple(kinds))
        kernel = InterpretedKernel(transform(text) if transform else text)
        launches.append(kernel)
        return kernel

    monkeypatch.setattr(family, "_get_kernel", _get_kernel)
    report = family.run_three_slot_no_pml_complex(fields, grid,
                                                  chunk_bounds=record_of(grid))
    return report, launches


@pytest.mark.parametrize("case,plant", POLE_PARAMETERS, ids=POLE_IDS)
def test_the_three_slot_weld_text_is_the_array_path_word_for_word(case, plant, host,
                                                                  monkeypatch):
    fields, grid = build(case, plant)
    start = snapshot(fields)
    fused_oracle(THREE_SLOT_PASSES, fields, None)
    want = snapshot(fields)
    restore(fields, start)
    _report, launches = _three_slot_launch(fields, grid, host, monkeypatch)
    assert len(launches) == 3 and all(kernel.launches == 1 for kernel in launches)
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant,
                                                {k: v for k, v in diff.items() if v})
    assert moved(start, want, ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"))
    assert moved(start, want, [n for n in want if n.startswith("P0_")])


# =============================================================================
# 3. The off-diagonal no-absorber weld: step_D, zero_metal_D, the off-diagonal update_E
# =============================================================================

#: name -> (cell, dimensions, boundaries, k_point, row mask)
WELD_CASES: Dict[str, Tuple] = {
    "wn_bloch3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.23, -0.17, 0.35),
                   (1, 1, 1, 1, 1, 1)),
    # The corpus row that puts this family on the board: one row pair, a phase on x, y.
    "wn_corpus3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.3892, 0.1597, 0.0),
                    (1, 0, 0, 1, 0, 0)),
    "wn_walls3d": ((7.0, 8.0, 9.0), 3, ("metallic", "metallic", "periodic"),
                   (0.0, 0.0, 0.19), (1, 1, 1, 1, 1, 1)),
    # The Brillouin edge on y (a negation) beside a wall on x.
    "wn_edge_y_wall_x": ((7.0, 8.0, 9.0), 3, ("metallic", "periodic", "periodic"),
                         (0.0, 0.0625, 0.0), (1, 1, 1, 1, 1, 1)),
}

WELD_STATE = ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Bx", "By", "Bz", "Hx", "Hy", "Hz")


def build_weld(case: str, plant_zeros: bool, seed: int = 37):
    from .coverage import OFFDIAG_ROW_SLOTS  # noqa: PLC0415

    cell, dimensions, boundaries, k_point, mask = WELD_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(F32)
        epsilon[component] = values
        inverse[component] = (F32(1.0) / values).astype(F32)
    rows: Dict[str, Dict[str, Any]] = {}
    for slot, (row, partner) in enumerate(OFFDIAG_ROW_SLOTS):
        if mask[slot]:
            rows.setdefault(row, {})[partner] = rng.uniform(
                -0.45, 0.45, size=shape).astype(F32)
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    for name in present(fields, WELD_STATE):
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    return fields, grid


def _weld_launch(fields, grid, text=None):
    """The family's own launcher on the interpreter, with every host fact its own
    resolver derives; returns the interpreted kernel."""
    from . import complex_no_pml_offdiag_fused_electric_pair as family  # noqa: PLC0415
    from . import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
    from . import complex_offdiag_update_e as offdiag  # noqa: PLC0415
    from .coverage import (offdiag_row_mask, offdiag_row_volumes,  # noqa: PLC0415
                           offdiag_wall_mask_flags)

    codes = family.complex_no_pml_offdiag_fused_electric_pair_codes(grid)
    phases = family.complex_no_pml_offdiag_fused_electric_pair_phases(grid)
    record, kept = family.complex_no_pml_offdiag_fused_electric_pair_chunk_arguments(
        fields, None, codes)
    wall_mask = offdiag_wall_mask_flags(grid)
    weights = offdiag.mirror_ghost_weights(grid)
    fills = weld.fill_plan(grid)
    family._validate_launch_arguments(codes, wall_mask, fills)
    mask = offdiag.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    nan = numpy.complex64(complex(numpy.nan, numpy.nan))
    scratch = {name: numpy.full_like(getattr(fields, name), nan)
               for name in family.SCRATCH_VOLUMES}
    kernel = InterpretedKernel(text if text is not None else family.kernel_source(mask))
    family.launch_complex_no_pml_offdiag_fused_electric_pair(
        fields, scratch, codes, wall_mask, phases, weights, fills,
        float(grid.dt / grid.dx), mask, rows, kept,
        numpy.asarray(record, dtype=numpy.int32), kernel=kernel)
    family.rotate_into_fields(fields, scratch)
    return kernel


WELD_PARAMETERS = [(case, plant) for case in sorted(WELD_CASES) for plant in (False, True)]


@pytest.mark.parametrize("case,plant", WELD_PARAMETERS,
                         ids=[f"{c}-{'zeros' if p else 'random'}" for c, p in WELD_PARAMETERS])
def test_the_offdiag_weld_text_is_the_array_path_word_for_word(case, plant, host,
                                                               monkeypatch):
    from . import complex_no_pml_offdiag_fused_electric_pair as family  # noqa: PLC0415

    bind_launcher(family, host, monkeypatch)
    fields, grid = build_weld(case, plant)
    start = {name: numpy.array(getattr(fields, name), copy=True)
             for name in present(fields, WELD_STATE)}
    fused_oracle(D_TO_E_PASSES, fields, None)
    want = {name: numpy.array(getattr(fields, name), copy=True)
             for name in present(fields, WELD_STATE)}
    for name, values in start.items():
        getattr(fields, name)[...] = values
    kernel = _weld_launch(fields, grid)
    assert kernel.launches == 1
    got = {name: numpy.array(getattr(fields, name), copy=True)
             for name in present(fields, WELD_STATE)}
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant,
                                                {k: v for k, v in diff.items() if v})
    assert moved(start, want, ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"))


def test_each_weld_case_binds_the_connection_kinds_it_is_built_to_reach(host):
    from . import complex_no_pml_offdiag_fused_electric_pair as family  # noqa: PLC0415

    kinds = {case: family.complex_no_pml_offdiag_fused_electric_pair_phases(
        build_weld(case, False)[1]) for case in WELD_CASES}
    phase, negate = complex_emitter.CONNECT_PHASE, complex_emitter.CONNECT_NEGATE
    assert kinds["wn_bloch3d"]["pair_kinds"] == (phase, phase, phase)
    assert kinds["wn_edge_y_wall_x"]["pair_kinds"] == (0, negate, 0)
    assert kinds["wn_bloch3d"]["pair"] != kinds["wn_bloch3d"]["own"]


def test_the_two_dimensional_cases_reach_the_one_partner_forms():
    """The record's partner-presence word marks the invariant z axis on the 2-D cases, so
    the stencil's one-partner arms run there."""
    for case in ("np_edge2d_pole", "np_conductive2d_uniform"):
        _fields, grid = build(case, False)
        assert int(grid.shape[2]) == 1 and grid.is_invariant(2)
        assert not grid.is_invariant(0) and not grid.is_invariant(1)


# =============================================================================
# 4. Planted defects: each a rewrite of one emitted text, each must change a word
# =============================================================================

#: defect -> (product, case, old text, new text)
DEFECTS: Dict[str, Tuple[str, str, str, str]] = {
    # KERNELS.md section 7.1, inverted: the previous package grouping of the bracket.
    "pair_stencil_regrouped": (
        "pair", "np_bloch3d", "cf_sub(cf_add(cf_sub(sf, f1), f2), ss)",
        "cf_add(cf_sub(sf, f1), cf_sub(f2, ss))"),
    # Section 7.2 (C5): the full bracket over an absent partner.
    "pair_presence_ignored": (
        "pair", "np_edge2d_pole", "    if (pr1 && pr2) return", "    if (1) return"),
    # The window read without its Bloch connection: MEEP's not-owned word on plane 0 of
    # a phased axis served as the stored one.
    "pair_window_unconnected": (
        "pair", "np_bloch3d", "    return cconnect(cf_load(g, idx), ia, ph, phase);",
        "    return cf_load(g, idx);"),
    "three_slot_stencil_regrouped": (
        "three_slot", "np_conductive_poles3d", "cf_sub(cf_add(cf_sub(sf, f1), f2), ss)",
        "cf_add(cf_sub(sf, f1), cf_sub(f2, ss))"),
    "three_slot_presence_ignored": (
        "three_slot", "np_edge2d_pole", "    if (pr1 && pr2) return", "    if (1) return"),
    "weld_stencil_regrouped": (
        "weld", "wn_bloch3d", "cf_sub(cf_add(cf_sub(sf, f1), f2), ss)",
        "cf_add(cf_sub(sf, f1), cf_sub(f2, ss))"),
    "weld_pair_window_unconnected": (
        "weld", "wn_corpus3d",
        "    return cconnect(resolve_D_at(comp, index, weld), pa, pph, pphase);",
        "    return resolve_D_at(comp, index, weld);"),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_defect_changes_a_word(defect, host, monkeypatch):
    from . import complex_no_pml_offdiag_fused_electric_pair as weld_family  # noqa: PLC0415
    from . import no_pml_complex_fused_electric_pair as pair_family  # noqa: PLC0415
    from .coverage import offdiag_row_mask  # noqa: PLC0415

    product, case, old, new = DEFECTS[defect]

    def rewrite(text: str) -> str:
        assert text.count(old) >= 1, (defect, "the anchor is not in the emitted text")
        return text.replace(old, new)

    if product == "weld":
        bind_launcher(weld_family, host, monkeypatch)
        fields, grid = build_weld(case, True)
        names = present(fields, WELD_STATE)
        start = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
        fused_oracle(D_TO_E_PASSES, fields, None)
        want = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
        for name, values in start.items():
            getattr(fields, name)[...] = values
        text = weld_family.kernel_source(weld_family._offdiag.normalized_row_mask(
            offdiag_row_mask(fields)))
        _weld_launch(fields, grid, text=rewrite(text))
        got = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
    else:
        fields, grid = build(case, True)
        start = snapshot(fields)
        fused_oracle(THREE_SLOT_PASSES if product == "three_slot" else D_TO_E_PASSES,
                     fields, None)
        want = snapshot(fields)
        restore(fields, start)
        if product == "pair":
            arm = pair_family.no_pml_complex_curl_arm(fields, "step_D")
            _pair_launch(fields, grid, host, text=rewrite(
                pair_family.no_pml_complex_fused_electric_pair_source(
                    arm == "conductive")))
        else:
            _three_slot_launch(fields, grid, host, monkeypatch, transform=rewrite)
        got = snapshot(fields)
    assert sum(differing(want, got).values()) > 0, defect
