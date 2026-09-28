"""Laptop tests for the FOLDED off-diagonal ``update_E`` Triton family.

The merge bar for ``triton_kernels/folded_offdiag_update_e.py``: everything that
can be decided without a GPU, without Triton, and without a byte gate. What the
compiled kernel does is the device gate's question
(``parity/meep_gpu/gate_triton_folded_offdiag.py``) and no test here claims
anything about it.

These tests exist because each one pins a fact that is a SILENT WRONG ANSWER if
it drifts, not because the lines need covering:

* the restated constants really equal the sources they were restated from
  (``MIRROR_SOURCE_INDEX``, the four ghost codes, ``DEFAULT_BLOCK``, the row
  tables) — a code that means 2 here and 3 there is a plane of wrong values;
* the mirror ghost weight really collapses to ``-phase`` for every partner on
  every axis and both plane phases — the collapse the kernel's single runtime
  scalar per axis depends on;
* the predicate refuses what the docstring says it refuses, and the two
  disjointness seams hold from BOTH sides;
* ``None`` means refused: wherever the predicate answers covered, the plan
  builder must return a plan rather than raise;
* the reachability statement agrees with a direct measurement against
  ``stepping.update_E``;
* the transcription IS ``stepping.update_E``, byte for byte, on real folded
  objects — a compressed form of the gate's reference leg, so a regression in
  the array path or in the transcription fails the ordinary test run rather than
  waiting for a device.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from . import stepping
from .fields import Fields, IYEE_SHIFTS, mirror_parity
from .grid import Grid, Mirror
from .pml import PML
from .triton_kernels import coverage as coverage_module
from .triton_kernels import folded_offdiag_update_e as folded
from .triton_kernels import offdiag_update_e as offdiag
from .triton_kernels import symmetry as symmetry_module

E_NAMES = ("Ex", "Ey", "Ez")
D_NAMES = ("Dx", "Dy", "Dz")
FW_NAMES = tuple("f_w_" + n for n in E_NAMES)
ALL_NAMES = E_NAMES + FW_NAMES + D_NAMES
AXES = "xyz"


# ---------------------------------------------------------------------------
# Fabric
# ---------------------------------------------------------------------------

def _build(fold="X", phase=1, cell=(2.0, 1.2, 1.2), boundaries=None,
           courant=0.35, rows=None, seed=3, dimensions=3):
    grid = Grid(resolution=10.0, cell_size=cell, courant=courant,
                symmetry=tuple(Mirror(letter, phase) for letter in fold),
                boundaries=boundaries, dimensions=dimensions, xp=np)
    fields = Fields(grid=grid)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)
    epsilon = {n: np.full(shape, v, np.float32)
               for n, v in zip(E_NAMES, (2.0, 2.5, 3.0))}
    inverse = {n: np.full(shape, 1.0 / v, np.float32)
               for n, v in zip(E_NAMES, (2.0, 2.5, 3.0))}
    if rows is None:
        rows = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                      for partner in E_NAMES if partner != row}
                for row in E_NAMES}
    else:
        rows = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                      for partner in partners}
                for row, partners in rows.items()}
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    fields.enable_pml_storage()
    for name in ALL_NAMES:
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    thickness = tuple(
        (0, 2) if grid.is_mirrored(axis)
        else ((2, 2) if shape[axis] >= 6 else (0, 0)) for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _face(axis, index):
    return (slice(None),) * axis + (index,)


def _reference_update_e(fields, pml, state, kinds=None, down="parity", up="zero"):
    """The transcription, with one knob per ghost arm for the control legs."""
    grid = fields.grid
    kinds = tuple(stepping._boundary_kinds(grid, pml)) if kinds is None else kinds
    phases = stepping._mirror_phases(grid)
    walls = offdiag.wall_mask_axes(grid)
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        constitutive = volumes[target] * fields.inverse_epsilon_for(target)
        rows = fields.chi1inv_offdiagonal_for(target)
        total = None
        for offset in (1, 2):
            partner_axis = (own_axis + offset) % 3
            coefficient = rows.get("E" + AXES[partner_axis])
            if coefficient is None:
                continue
            values = volumes["E" + AXES[partner_axis]]
            shifted = np.roll(values, 1, axis=partner_axis)
            kind = kinds[partner_axis]
            if kind == stepping.METALLIC:
                shifted[_face(partner_axis, 0)] = 0
            elif kind == stepping.MIRROR:
                if down == "metallic":
                    shifted[_face(partner_axis, 0)] = 0
                elif down != "wrap":
                    weight = mirror_parity("D" + AXES[partner_axis], partner_axis,
                                           int(phases[partner_axis]))
                    shifted[_face(partner_axis, 0)] = weight * values[
                        _face(partner_axis, folded.MIRROR_SOURCE_INDEX)]
            product = (values + shifted) * coefficient
            up_shifted = np.roll(product, -1, axis=own_axis)
            if kinds[own_axis] in (stepping.METALLIC, stepping.MIRROR):
                if not (kinds[own_axis] == stepping.MIRROR and up == "wrap"):
                    up_shifted[_face(own_axis, -1)] = 0
            term = 0.25 * (product + up_shifted)
            total = term if total is None else total + term
        if total is not None:
            iyee = IYEE_SHIFTS[target]
            for axis in range(3):
                if iyee[axis] == 0 and walls[axis]:
                    total[_face(axis, 0)] = 0
            constitutive = constitutive + total
        fw = state["f_w_" + target]
        previous = fw.copy()
        fw[...] = constitutive
        state[target] += getattr(pml, f"kps_{AXES[own_axis]}_h") * fw
        state[target] -= getattr(pml, f"kms_{AXES[own_axis]}_h") * previous


def _bytes(source):
    getter = (source.__getitem__ if isinstance(source, dict)
              else lambda n: getattr(source, n))
    return b"".join(np.asarray(getter(n)).tobytes() for n in E_NAMES + FW_NAMES)


# ---------------------------------------------------------------------------
# Restated constants
# ---------------------------------------------------------------------------

def test_restated_constants_equal_their_sources():
    """A code that means one thing here and another there is a plane of wrong
    values, not a crash — so every restated constant is pinned to its home."""
    assert folded.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX
    assert folded.MIRROR_SOURCE_INDEX == symmetry_module.MIRROR_SOURCE_INDEX
    assert (folded.CODE_PERIODIC, folded.CODE_METALLIC,
            folded.CODE_MIRROR_METALLIC, folded.CODE_MIRROR_PERIODIC) == (
        symmetry_module.CODE_PERIODIC, symmetry_module.CODE_METALLIC,
        symmetry_module.CODE_MIRROR_METALLIC, symmetry_module.CODE_MIRROR_PERIODIC)
    assert folded.MIRROR_CODES == (folded.CODE_MIRROR_METALLIC,
                                   folded.CODE_MIRROR_PERIODIC)
    # The unfolded codes must keep kernels.py's values so a plan built here and
    # a plan built there index the same table.
    assert (folded.CODE_PERIODIC, folded.CODE_METALLIC) == (0, 1)
    assert folded.E_TERMS is offdiag.E_TERMS
    assert folded.ROW_SLOTS is offdiag.ROW_SLOTS
    assert folded.WALL_MASK_AXES is offdiag.WALL_MASK_AXES
    assert folded.DEFAULT_BLOCK == offdiag.DEFAULT_BLOCK
    assert folded.HALF_INTEGER is True


def test_row_slots_cover_every_offdiagonal_pair_in_cycle_order():
    """Six slots, each an (row, partner) pair with partner != row, in MEEP's
    cycle_direction order X -> Y -> Z (vec.hpp:586; stepping.py:1235-1237)."""
    assert len(folded.ROW_SLOTS) == 6
    for index, (row, partner) in enumerate(folded.ROW_SLOTS):
        own_axis = index // 2
        offset = index % 2 + 1
        assert row == E_NAMES[own_axis]
        assert partner == E_NAMES[(own_axis + offset) % 3]


def test_wall_mask_axes_table_matches_iyee_shifts():
    """The mask loops the axes on which the component's Yee shift is 0,
    ascending (stepping.py:1279-1283)."""
    for index, term in enumerate(folded.E_TERMS):
        expected = tuple(axis for axis in range(3)
                         if IYEE_SHIFTS[term[0]][axis] == 0)
        assert folded.WALL_MASK_AXES[index] == expected


def test_partner_axes_are_never_the_rows_own_axis():
    """The whole reachability statement rests on this: axis ``a`` is a partner
    axis of row ``E_c`` exactly when ``a != c``."""
    for index, (row, partner) in enumerate(folded.ROW_SLOTS):
        own_axis = E_NAMES.index(row)
        partner_axis = E_NAMES.index(partner)
        assert partner_axis != own_axis


# ---------------------------------------------------------------------------
# The ghost weight
# ---------------------------------------------------------------------------

def test_mirror_ghost_weight_collapses_to_minus_phase_everywhere():
    """``_shift_down`` passes ``'D' + AXIS_NAMES[partner_axis]`` on
    ``axis = partner_axis`` (stepping.py:1243-1245), and every D component has
    Yee shift 1 on its own axis — so ``mirror_parity`` collapses to ``-phase``
    for every partner and every axis. One signed scalar per folded axis is the
    kernel's entire parity input, and this is why."""
    for axis in range(3):
        for phase in (1, -1):
            component = "D" + AXES[axis]
            assert IYEE_SHIFTS[component][axis] == 1
            assert mirror_parity(component, axis, phase) == -phase


def test_mirror_ghost_weights_are_read_through_mirror_parity():
    for phase in (1, -1):
        fields, _pml = _build(fold="Y", phase=phase, cell=(1.2, 2.0, 1.2))
        weights = folded.mirror_ghost_weights(fields.grid)
        assert weights[1] == float(-phase)
        assert weights[0] == weights[2] == 1.0   # unfolded axes never read it


def test_mirror_parity_is_phase_times_one_minus_two_iyee():
    """The identity :mod:`symmetry` derives its single signed constexpr from
    (symmetry.py:368-374), asserted exhaustively rather than trusted."""
    for component, shifts in IYEE_SHIFTS.items():
        for axis in range(3):
            for phase in (1, -1):
                assert mirror_parity(component, axis, phase) == \
                    phase * (1 - 2 * shifts[axis])


# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------

def test_module_imports_and_answers_without_triton():
    """The predicate and the builders must answer on the laptop that is the
    merge bar, whether or not the optional dependency is installed."""
    fields, pml = _build()
    verdict = folded.folded_offdiag_constitutive_coverage(fields, pml)
    assert verdict.covered is False           # NumPy is not cupy
    assert folded.plan_folded_offdiagonal_constitutive(fields, pml) is None
    assert folded.explain_folded_offdiag(fields, pml).reasons


def test_kernel_accessor_is_diagnosable_when_triton_is_absent():
    if folded.folded_offdiag_constitutive_step is not None:
        pytest.skip("triton is installed; the absent branch is the laptop's")
    with pytest.raises(ImportError, match="triton"):
        folded.folded_offdiag_constitutive_step_kernel()


def test_shared_clause_helpers_still_exist_in_their_modules():
    """Named as data so a rename in the module that defines a helper fails here
    rather than silently widening coverage."""
    lookup = {"coverage": coverage_module, "symmetry": symmetry_module,
              "offdiag_update_e": offdiag}
    for entry in folded.SHARED_CLAUSES:
        module_name, attribute = entry.split(".")
        assert hasattr(lookup[module_name], attribute), entry


# ---------------------------------------------------------------------------
# Refusals and the disjointness seams
# ---------------------------------------------------------------------------

def _reasons(fields, pml):
    return folded.folded_offdiag_constitutive_coverage(fields, pml).reasons


def test_refuses_a_run_with_no_surviving_row_slot():
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {}
    assert any("no off-diagonal chi1inv row survived" in r
               for r in _reasons(fields, pml))


def test_refuses_complex_storage_toward_the_folded_complex_family():
    fields, pml = _build()
    fields.force_complex_fields = True
    assert any("force_complex_fields" in r for r in _reasons(fields, pml))


def test_refuses_a_row_volume_that_aliases_an_output():
    """Reachable through the PUBLIC installer, which keeps the caller's array
    without copying (fields.py:1296) — so the predicate must refuse it, not the
    builder alone."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": fields.Ez}}
    assert any("aliases output" in r for r in _reasons(fields, pml))


def test_refuses_an_inactive_pml():
    fields, _pml = _build()
    assert any("no active PML layer" in r for r in _reasons(fields, None))


def test_conductivity_is_not_a_clause():
    """Deliberately: a conductivity changes the CURL sub-steps only, never the
    constitutive one (coverage.py:128-130). Inheriting the folded curl's
    refusal would silently narrow an independent sub-step."""
    fields, pml = _build()
    assert not any("conductivit" in r.lower() for r in _reasons(fields, pml))


def test_the_two_parent_predicates_still_refuse_this_family():
    """The disjointness seams, from both sides. If either stops refusing, two
    predicates admit the same run and the composer picks by branch order."""
    fields, pml = _build()
    parent_offdiag = offdiag.offdiag_constitutive_coverage(fields, pml)
    assert not parent_offdiag.covered
    assert any("fold" in r or "mirror" in r for r in parent_offdiag.reasons)
    parent_folded = symmetry_module.folded_constitutive_coverage(fields, pml, "E")
    assert not parent_folded.covered
    assert any("off-diagonal" in r for r in parent_folded.reasons)


def test_composition_verdict_requires_a_real_fold():
    """The standalone predicate admits an unfolded grid so the gate can prove
    reduction; the composition verdict must not, or routing would depend on
    composer order (symmetry.py:713-722)."""
    fields, pml = _build(fold="")
    reasons = folded.folded_offdiag_composition_coverage(fields, pml).reasons
    assert any("no mirror plane is active" in r for r in reasons)
    folded_fields, folded_pml = _build(fold="X")
    assert not any("no mirror plane is active" in r for r in
                   folded.folded_offdiag_composition_coverage(
                       folded_fields, folded_pml).reasons)


# ---------------------------------------------------------------------------
# The plan builder's backstops
# ---------------------------------------------------------------------------

def _arrays(shape):
    rng = np.random.default_rng(17)
    out = {name: rng.uniform(-0.4, 0.4, shape).astype(np.float32)
           for name in ALL_NAMES}
    out.update({"inv_eps_" + name: np.full(shape, 0.4, np.float32)
                for name in E_NAMES})
    return out


def _flat(shape):
    return {f"{stem}_{axis}": np.full(shape[i], 0.9, np.float32)
            for i, axis in enumerate(AXES) for stem in ("kps", "kms")}


def _plan(shape=(8, 8, 8), codes=(2, 0, 0), walls=(0, 0, 0),
          weights=(-1.0, 1.0, 1.0), rows=None):
    return folded.plan_folded_offdiagonal_constitutive_from_arrays(
        _arrays(shape), _flat(shape),
        {"Ey": {"Ex": np.full(shape, 0.1, np.float32)}} if rows is None else rows,
        codes, walls, weights)


def test_plan_builds_from_bare_arrays():
    plan = _plan()
    assert plan.boundary_codes == (2, 0, 0)
    assert plan.ghost_weights == (-1.0, 1.0, 1.0)
    assert plan.row_mask == (0, 0, 0, 1, 0, 0)
    assert plan.launch_grid[0] * plan.block >= plan.n_elem


def test_plan_refuses_a_folded_axis_that_is_also_wall_masked():
    """``_mask_metallic_wall_coupling`` abstains on a mirrored axis
    (stepping.py:1282); zeroing the fold plane costs 2.0e-02 by the array
    path's own measurement."""
    with pytest.raises(ValueError, match="folded AND wall-masked"):
        _plan(codes=(2, 0, 0), walls=(1, 0, 0))


def test_plan_refuses_a_ghost_weight_that_is_not_plus_or_minus_one():
    with pytest.raises(ValueError, match="ghost weight"):
        _plan(weights=(0.5, 1.0, 1.0))


def test_plan_refuses_a_folded_axis_too_thin_for_the_ghost_row():
    with pytest.raises(ValueError, match="mirror ghost images stored row"):
        _plan(shape=(2, 8, 8), codes=(2, 0, 0))


def test_plan_refuses_an_unknown_boundary_code():
    with pytest.raises(ValueError, match="boundary codes"):
        _plan(codes=(4, 0, 0))


def test_plan_refuses_an_all_dead_row_mask_toward_the_element_wise_family():
    with pytest.raises(ValueError, match="no row slot survives"):
        _plan(rows={})


def test_plan_refuses_an_input_that_aliases_an_output():
    shape = (8, 8, 8)
    arrays = _arrays(shape)
    arrays["Dx"] = arrays["Ez"]
    with pytest.raises(ValueError, match="aliases"):
        folded.plan_folded_offdiagonal_constitutive_from_arrays(
            arrays, _flat(shape),
            {"Ey": {"Ex": np.full(shape, 0.1, np.float32)}},
            (2, 0, 0), (0, 0, 0), (-1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("letter,axis", [("X", 0), ("Y", 1), ("Z", 2)])
def test_reachability_matches_a_direct_measurement(letter, axis):
    """The mirror arm changes bytes exactly when the folded axis is the PARTNER
    axis of a live row SLOT. Measured by flipping ONLY the ghost code on the same
    grid, the same coefficients and the same seeds.

    THE THIRD CASE IS THE SLOT-LEVEL COUNTEREXAMPLE and it is why this test is
    parametrised over row SHAPES rather than over rows: a single slot
    ``E_{a+1} <- E_{a+2}`` leaves a live row that is not ``E_a`` while no live
    slot takes ``E_a`` as its partner, so the fold enters NEITHER role. A
    row-level predicate answers 'reachable' there and is wrong; without this
    case the suite could not tell the two readings apart.
    """
    cell = tuple(2.0 if i == axis else 1.2 for i in range(3))
    own_row = E_NAMES[axis]
    next_row = E_NAMES[(axis + 1) % 3]
    far_row = E_NAMES[(axis + 2) % 3]
    for rows, expect_reachable, why in (
            ({own_row: [next_row, far_row]}, False,
             "every live slot is in row E_a, whose own axis the fold is"),
            ({next_row: [far_row, own_row]}, True,
             "row E_{a+1} carries both partners, one of them E_a"),
            ({next_row: [far_row]}, False,
             "SLOT-LEVEL COUNTEREXAMPLE: the live row is not E_a, and no live "
             "slot takes E_a as its partner"),
            ({next_row: [own_row]}, True,
             "the single live slot takes E_a as its partner")):
        fields, pml = _build(fold=letter, cell=cell, rows=rows)
        reachable, notes = folded.mirror_arm_is_reachable(fields.grid, fields)
        assert reachable is expect_reachable, (why, notes)
        kinds = list(stepping._boundary_kinds(fields.grid, pml))

        def drive(active):
            state = {n: getattr(fields, n).copy() for n in ALL_NAMES}
            for _ in range(3):
                _reference_update_e(fields, pml, state, kinds=tuple(active))
            return _bytes(state)

        swapped = list(kinds)
        swapped[axis] = stepping.METALLIC
        assert (drive(kinds) == drive(swapped)) is (not expect_reachable), why


# ---------------------------------------------------------------------------
# The transcription IS the array path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fold,phase,cell,boundaries,courant", [
    ("X", 1, (2.0, 1.2, 1.2), None, 0.35),        # folded PERIODIC, even count
    ("X", 1, (2.1, 1.2, 1.2), None, 0.35),        # odd full count
    ("X", -1, (2.0, 1.2, 1.2), None, 0.35),       # odd plane
    ("Y", 1, (1.2, 2.0, 1.2), "metallic", 0.5),   # folded METALLIC
    ("Z", 1, (1.2, 1.2, 2.0), None, 0.3125),      # a third axis, NP2 Courant
    ("XY", 1, (2.0, 2.0, 1.2), None, 0.35),       # two planes at once
    ("XYZ", 1, (2.0, 2.0, 2.0), None, 0.35),      # THREE planes: every partner
    ("XYZ", -1, (2.0, 2.0, 2.0), "metallic", 0.5),  # axis folded, both ends
])
def test_transcription_is_bit_identical_to_stepping_update_e(
        fold, phase, cell, boundaries, courant):
    """The compressed form of the gate's reference leg: a regression in the
    array path OR in the transcription fails the ordinary test run rather than
    waiting for a device."""
    fields, pml = _build(fold=fold, phase=phase, cell=cell,
                         boundaries=boundaries, courant=courant)
    state = {n: getattr(fields, n).copy() for n in ALL_NAMES}
    for _ in range(3):
        stepping.update_E(fields, pml)
        _reference_update_e(fields, pml, state)
        for name in D_NAMES:
            state[name][...] = getattr(fields, name)
    assert _bytes(state) == _bytes(fields)


@pytest.mark.parametrize("down,up,mask", [
    ("metallic", "zero", False),   # the certified kernel's arm
    ("wrap", "zero", False),       # the periodic wrap
])
def test_the_certified_ghost_arms_are_not_the_folded_ones(down, up, mask):
    """Non-vacuity: if the certified kernel's DOWN arm reproduced a fold, this
    family would be a redundant kernel. It does not."""
    fields, pml = _build(fold="X", cell=(2.0, 1.2, 1.2))
    shipped = {n: getattr(fields, n).copy() for n in ALL_NAMES}
    control = {n: getattr(fields, n).copy() for n in ALL_NAMES}
    _reference_update_e(fields, pml, shipped)
    _reference_update_e(fields, pml, control, down=down, up=up)
    assert _bytes(shipped) != _bytes(control)


def test_the_own_axis_up_ghost_is_an_exact_zero_on_a_fold():
    """``_offdiagonal_terms`` calls ``_shift_up`` with four arguments
    (stepping.py:1248-1249) — no ``component``, no ``reflect_row`` — so both
    mirror terminations take stepping.py:1828-1830. Pinned as an ENGINE fact,
    because it is the one place this family's transcription differs from what a
    reader of ``_shift_up``'s docstring would expect."""
    fields, pml = _build(fold="X", cell=(2.0, 1.2, 1.2))
    assert fields.grid.is_mirrored(0) and not fields.grid.is_metallic(0)
    assert stepping._far_reflect_rows(fields.grid)[0] is not None
    values = fields.Dx
    shifted = stepping._shift_up(np, values, 0, stepping.MIRROR,
                                 stepping._bloch_phases(
                                     fields.grid,
                                     stepping._boundary_kinds(fields.grid, pml),
                                     values, pml)[0])
    assert not np.any(shifted[_face(0, -1)])


def _negative_zero_census(state, names) -> int:
    """Stored words that are NEGATIVE ZERO — not merely negative.

    The distinction is the whole clause. ``np.signbit`` over every word counts
    ordinary negative numbers too, which a needle case has in quantity, so a
    sign-bit tally is >0 for reasons that have nothing to do with the class the
    discipline names (a defect visible only in the sign of a zero). Counting
    ``(x == 0) & signbit(x)`` is what makes census 0 mean 'this case cannot see
    that class'."""
    total = 0
    for name in names:
        array = np.asarray(state[name])
        total += int(np.count_nonzero((array == 0) & np.signbit(array)))
    return total


def test_zero_init_alone_is_vacuous_and_the_row_two_needle_is_not():
    """The case-discipline clause, measured rather than assumed, in THREE parts.

    1. With every array zero the fold plane stays at zero and a sign-flipped
       ghost is INVISIBLE — vacuous, not coverage.
    2. Seeding stored row ``MIRROR_SOURCE_INDEX`` and nothing else makes the
       mirror ghost the only path to a nonzero fold-plane E, and the flip
       becomes byte-visible.
    3. BUT a ``+0.0`` background still carries a NEGATIVE-ZERO CENSUS OF 0: the
       row sum's ``diag + total`` add annihilates the ghost's parity sign
       (``+0.0 + -0.0 == +0.0``). Only a ``-0.0`` background keeps both addends
       negative, so the sign survives into stored ``f_w``. That is why the
       gate's zero-init class seeds negative zero — the census meets the clause
       by construction rather than by relabelling a sign-bit tally.
    """
    fields, pml = _build(fold="X", cell=(2.0, 1.2, 1.2))
    shape = tuple(fields.grid.shape)

    def run(seed_row_two, background=0.0):
        for name in ALL_NAMES:
            getattr(fields, name)[...] = np.full(shape, background, np.float32)
        if seed_row_two:
            plane = np.random.default_rng(2).uniform(
                0.2, 0.6, shape[1:]).astype(np.float32)
            for name in D_NAMES:
                getattr(fields, name)[_face(0, folded.MIRROR_SOURCE_INDEX)] = plane
        shipped = {n: getattr(fields, n).copy() for n in ALL_NAMES}
        flipped = {n: getattr(fields, n).copy() for n in ALL_NAMES}
        _reference_update_e(fields, pml, shipped)
        _reference_update_e(fields, pml, flipped, down="metallic")
        census = _negative_zero_census(shipped, E_NAMES + FW_NAMES)
        plane_energy = max(float(np.max(np.abs(shipped[n][_face(0, 0)])))
                           for n in E_NAMES)
        return _bytes(shipped) != _bytes(flipped), census, plane_energy

    visible_plain, census_plain, energy_plain = run(False)
    assert not visible_plain and census_plain == 0 and energy_plain == 0.0, (
        "an all-zero seed must be recorded VACUOUS, not as coverage")

    visible_needle, census_positive, energy_needle = run(True)
    assert visible_needle, "the row-2 needle must make the ghost byte-visible"
    assert energy_needle > 0.0
    assert census_positive == 0, (
        "a +0.0 background annihilates the parity sign in the row sum; if this "
        "ever becomes nonzero the gate's reason for seeding -0.0 has changed")

    visible_signed, census_signed, energy_signed = run(True, background=-0.0)
    assert visible_signed and energy_signed > 0.0
    assert census_signed > 0, (
        "census 0 is vacuous by the case discipline: with a -0.0 background "
        "the stored row sum must keep negative zeros for a sign-bit defect to "
        "be visible at all")


# ---------------------------------------------------------------------------
# The gate's own harness — the parts that must not drift silently
# ---------------------------------------------------------------------------
#
# The gate is the arbiter of bytes and needs a device to say anything about
# them. What it does NOT need a device for is its own machinery: a mutation that
# stopped matching the kernel source, a sweep that stopped covering a
# termination, a second transcription that drifted from the pinned one. Those
# fail here, in the ordinary test run, on the day the kernel is edited.

PARITY_DIR = (pathlib.Path(__file__).resolve().parents[1]
              / "parity" / "meep_gpu")


def _gate_module():
    if str(PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(PARITY_DIR))
    return importlib.import_module("gate_triton_folded_offdiag")


def test_the_gate_sweep_covers_every_axis_of_the_case_discipline():
    """The enumeration is data, so what it covers is assertable without running
    it: BOTH terminations, BOTH plane phases, a fold on each of X/Y/Z, TWO
    planes at once, ZERO folded axes (the reduction row), a REDUCED shape, a
    non-power-of-two Courant, both inverse-epsilon forms, and the needle
    classes."""
    gate = _gate_module()
    cases = gate.sweep_cases()
    codes = [tuple(case["codes"]) for case in cases]
    assert any(gate.C_MIRROR_PERIODIC in c for c in codes), "no folded PERIODIC"
    assert any(gate.C_MIRROR_METALLIC in c for c in codes), "no folded METALLIC"
    phases = {p for case in cases for p in case["phases"] if p is not None}
    assert phases == {1, -1}, f"both plane phases required, got {phases}"
    for axis in range(3):
        assert any(c[axis] in folded.MIRROR_CODES for c in codes), (
            f"no case folds axis {axis}")
    assert any(sum(1 for x in c if x in folded.MIRROR_CODES) == 2
               for c in codes), "no two-plane case"
    assert any(all(x not in folded.MIRROR_CODES for x in c) for c in codes), (
        "no ZERO-fold case: the reduction row has no witness")
    assert any(1 in tuple(case["shape"]) for case in cases), "no reduced shape"
    assert any(case["courant"] is not None
               and case["courant_is_power_of_two"] is False
               for case in (gate.one_sweep_case(spec, False) for spec in cases
                            if spec["coefficients"] == "layer")), (
        "the sweep carries no NON-POWER-OF-TWO Courant")
    assert {case["inv_form"] for case in cases} == {"distinct", "aliased"}
    amplitudes = {case["amplitude"] for case in cases}
    assert {"normal", "cancellation", "zero_init_row2",
            "signed_zero_row2"} <= amplitudes
    assert {case["guard"] for case in cases} == {False, True}


def test_every_gate_mutation_is_armed_and_the_mutant_still_parses():
    """The three ways a planted defect silently disarms — a rename, a
    re-indent, a syntax break — all measured against the SHIPPED source.

    A patch that matches nothing would report NEEDLE-MISSED for a needle that
    was never planted; the certified family's own ``_TERM_NEEDLE`` is cut at
    four-space indentation and does NOT match this kernel, which is exactly the
    drift this test exists to catch.

    THE SITE COUNT IS ASSERTED, NOT MERELY ``> 0``. Several patchers here
    increment once per INDEPENDENT needle (m2 seven, m3 nine, f9 four, f6 two):
    if some drifted and one still matched, ``hits > 0`` would read as armed
    while the mutant carried only PART of the declared defect, and a device leg
    would certify a partial redirect as though it were the whole one."""
    gate = _gate_module()
    import gate_triton_offdiag as certified  # noqa: PLC0415

    inherited = {name: getattr(certified, attribute)
                 for name, attribute in gate.INHERITED_PATCHES.items()}
    source = pathlib.Path(folded.__file__).read_text()
    for name, entry in gate.MUTATIONS.items():
        patch = entry["patch"] or inherited[name]
        mutated, hits = patch(source)
        assert hits > 0, f"{name} is DISARMED: it matched nothing"
        assert hits == entry["expected_hits"], (
            f"{name} patched {hits} sites, declared {entry['expected_hits']}: "
            f"either the kernel moved or the mutant is partial")
        assert mutated != source, f"{name} changed nothing"
        compile(mutated, f"<mutant {name}>", "exec")


def test_the_certified_patchers_this_file_respells_still_do_not_match():
    """WHY m1 and m4 are re-spelled here instead of imported.

    The certified gate's ``_TERM_NEEDLE`` is cut at FOUR-space indentation and
    this kernel's term helper is nested one level deeper, so the inherited
    patchers match ZERO sites — the DISARMED failure mode, which is silent
    unless someone measures it. This test measures it. The patchers this file
    DOES import (m2, m3, m9, the commuted-row-sum null) are indentation-
    independent and are asserted to still hit, at their declared counts, by
    ``test_every_gate_mutation_is_armed_and_the_mutant_still_parses``.

    If this test ever fails, the certified needles have been re-cut and the two
    local re-spellings should be re-examined rather than kept out of habit."""
    _gate_module()
    import gate_triton_offdiag as certified  # noqa: PLC0415

    source = pathlib.Path(folded.__file__).read_text()
    for attribute in ("mutate_m1_hoisted_coefficient",
                      "mutate_m4_distribute_quarter"):
        _mutated, hits = getattr(certified, attribute)(source)
        assert hits == 0, (
            f"{attribute} now matches this kernel; the local re-spelling may "
            f"be redundant — re-measure before keeping both")


def test_the_gate_mutation_table_still_carries_its_declared_nulls():
    """The leg's description names three nulls; carrying two of them is how a
    control quietly stops being measured. m4 (0.25 distributed) had no entry at
    all, and its own caveat — 'away from underflow' — is why it is carried on a
    SUBNORMAL needle as well as the normal one."""
    gate = _gate_module()
    nulls = {name for name, entry in gate.MUTATIONS.items()
             if entry["expectation"] is False}
    assert "null_commuted_row_sum" in nulls
    assert "m4_distributed_quarter" in nulls
    assert gate.MUTATIONS["m4_distributed_quarter_subnormal"][
        "needle"] == "subnormal"
    assert gate.MUTATION_NEEDLES["subnormal"]["amplitude"] == "subnormal"


def test_the_fold_abstention_mutant_zeroes_exactly_the_fold_lane():
    """f9 must be the MINIMAL spelling of "the abstention at stepping.py:1282
    dropped": on the declared fold-X needle it may zero ``at_x`` for Ey and Ez
    and nothing else, which is exactly what its ``mask_fold`` host analogue
    does. The coarser both-guards-dropped spelling also zeroes planes on two
    PERIODIC axes, so a 'caught' verdict from it evidences "some wall masking
    changed bytes" — that defect is still carried, as f11, under its own claim.
    """
    gate = _gate_module()
    source = pathlib.Path(folded.__file__).read_text()
    minimal, hits = gate.mutate_f9_wall_mask_stops_abstaining_on_the_fold(source)
    assert hits == 4 and "WM_X," not in minimal
    assert "WM_Y, WM_Z)" in minimal, "Ex's lanes must be untouched"
    coarse, coarse_hits = gate.mutate_f11_wall_mask_unconditional(source)
    assert coarse_hits == 1 and "if WMA:" not in coarse
    # The two host analogues differ: mask_fold touches only MIRROR axes.
    needle = gate._needle_state("fold_partner")
    outs = {}
    for label, kwargs in (("shipped", {}), ("fold", {"mask_fold": True}),
                          ("all", {"mask_all": True})):
        state = {n: needle["state"][n].copy() for n in gate.ALL_NAMES}
        gate.reference_update_e(state, needle["coefficients"], needle["rows"],
                                needle["inv_eps"], needle["kinds"],
                                needle["phases"], needle["reflect"],
                                needle["walls"], **kwargs)
        outs[label] = b"".join(state[n].tobytes() for n in gate.STATE_NAMES)
    assert outs["fold"] != outs["shipped"]
    assert outs["all"] != outs["shipped"]
    assert outs["all"] != outs["fold"], (
        "if the two analogues agreed, f9 and f11 would be one defect")


def test_the_gate_reference_census_is_a_negative_zero_count():
    """The gate records two different numbers under two different names. A bare
    sign-bit tally counts ordinary negative numbers — about half the words of a
    random seed — and reporting it as a 'census' would say the reference grids
    see the signed-zero class in quantity. They do not."""
    gate = _gate_module()
    state = {"Ex": np.array([-0.0, -1.0, 0.0, 2.0], np.float32)}
    assert gate.negative_zero_census(state, ("Ex",)) == 1
    assert gate.negative_word_tally(state, ("Ex",)) == 2


def test_the_gates_variant_transcription_agrees_with_the_pinned_one():
    """The mutation leg's analogues run through a SECOND transcription. If it
    drifted from the one the reference leg pinned against ``stepping.update_E``,
    every analogue would measure the drift instead of the defect."""
    gate = _gate_module()
    needle = gate._needle_state("fold_partner")
    left = {n: needle["state"][n].copy() for n in gate.ALL_NAMES}
    right = {n: needle["state"][n].copy() for n in gate.ALL_NAMES}
    gate.reference_update_e(left, needle["coefficients"], needle["rows"],
                            needle["inv_eps"], needle["kinds"],
                            needle["phases"], needle["reflect"],
                            needle["walls"])
    gate.reference_variant_update_e(right, needle["coefficients"],
                                    needle["rows"], needle["inv_eps"],
                                    needle["kinds"], needle["phases"],
                                    needle["reflect"], needle["walls"],
                                    "shipped")
    # Bytes compared here rather than through the gate's own ``state_bytes``:
    # these are NumPy arrays either way, so ``tobytes`` asks nothing of a host
    # bridge and this test cannot be re-coupled to whichever backend module some
    # other file's import happened to bind. (A sibling family's test used to
    # install ``sys.modules['cupy'] = numpy``, which made exactly that bridge
    # reach for a non-existent ``numpy.asnumpy``; the stub is gone and
    # ``meep_gpu/conftest.py`` now fails anything that reinstates it.)
    assert all(left[n].tobytes() == right[n].tobytes()
               for n in gate.STATE_NAMES)


def test_the_unfolded_arm_is_the_certified_kernel_body_verbatim():
    """Identity claim (i)'s STRUCTURAL half: with ``MG`` false the term helper
    must be ``offdiag_update_e._offdiag_term``'s body VERBATIM, so an unfolded
    axis reduces to the certified instruction sequence by construction rather
    than by trusting a compiler to fold a multiply by 1.0 away."""
    gate = _gate_module()
    verdict = gate.verbatim_else_arm()
    assert verdict["else_arm_is_verbatim"], verdict["folded_else_statements"]
    assert verdict["return_is_verbatim"]


def test_the_gate_reference_refuses_an_unknown_ghost_name():
    """A typo in a control or an analogue must RAISE, never fall through to the
    shipped arm: a silent fallthrough reports 'needle missed' for a needle that
    was never planted."""
    gate = _gate_module()
    field = np.zeros((6, 5, 4), np.float32)
    with pytest.raises(ValueError):
        gate.reference_shift_down(field, 0, stepping.MIRROR, 1, "typo")
    with pytest.raises(ValueError):
        gate.reference_shift_up(field, 0, stepping.MIRROR, 1, None, "typo")


def _stub_device_leg(compared, identical, certified):
    def leg(results, out_path, device_ok, *args, **kwargs):
        return {"compared": compared, "identical": identical,
                "certified": certified, "pass": certified,
                "host_half_pass": True, "cases": [], "errors": 0}
    return leg


@pytest.mark.parametrize("certified,identical,expect_pass,expect_code", [
    (False, 0, False, 1),      # every uint32 compare FAILED
    (True, 156, True, 0),      # every uint32 compare identical
])
def test_the_gate_verdict_reads_the_byte_outcome_not_the_compare_count(
        tmp_path, monkeypatch, certified, identical, expect_pass, expect_code):
    """``compared`` is a COUNT and ``certified`` is the OUTCOME.

    Reading only the count is how a device round in which every comparison
    failed still stamped ``passed: true``, printed ``[gate] passed`` and exited
    0. This test drives the real ``main`` with the five device host halves
    stubbed to a fully FAILING device leg and requires exit 1 — and with the
    same stub certified, exit 0, so it cannot pass by refusing everything."""
    gate = _gate_module()
    for name in gate.DEVICE_LEGS:
        monkeypatch.setitem(gate.DEVICE_LEG_HOST_HALVES, name,
                            _stub_device_leg(156, identical, certified))
        monkeypatch.setitem(gate.DEVICE_HALF_IMPLEMENTED, name, True)
    monkeypatch.setattr(gate, "device_available", lambda: (True, "stubbed"))
    # With a ``cp`` that is not None — real CuPy on a device host, and once a
    # sibling family's ``sys.modules['cupy'] = numpy`` stub anywhere — main()
    # takes the CuPy subnormal-policy branch and refuses to certify for the RIGHT
    # reason (no strip exercised), masking the verdict logic this test is about.
    # Pin the laptop path so the only thing under test is how the verdict reads
    # the legs, on whichever host runs it.
    monkeypatch.setattr(gate, "cp", None)
    out = tmp_path / "gate.json"
    code = gate.main(["--legs", ",".join(gate.DEVICE_LEGS), "--out", str(out)])
    data = json.loads(out.read_text())
    assert code == expect_code
    assert data["passed"] is expect_pass
    assert (data["status"] == "passed") is expect_pass
    if not expect_pass:
        assert data["byte_failures"], "a failing compare must be NAMED"


def _sweep_summary(gate, guarded_ran, guarded_identical, fusion_ran,
                   fusion_identical):
    cases = ([{"compared": True, "guard": False,
               "verdict": {"bit_identical": index < guarded_identical}}
              for index in range(guarded_ran)]
             + [{"compared": True, "guard": True,
                 "verdict": {"bit_identical": index < fusion_identical}}
                for index in range(fusion_ran)])
    return gate.summarize_sweep(cases)


def test_the_sweep_certifies_the_fusion_off_rows_and_controls_the_other_axis():
    """The certified configuration is ENABLE_FP_FUSION=False.

    Merging the fusion-on rows into ``certified`` makes a byte-PERFECT kernel
    report ``certified: false`` on this tranche's own measured expectation
    (offdiag 0/28 fusion-on rows identical). Dropping the fusion axis from the
    verdict entirely loses the certified family's non-vacuity control. Both
    outcomes are driven here."""
    gate = _gate_module()

    perfect = gate.certify_sweep(_sweep_summary(gate, 28, 28, 28, 0))
    assert perfect["certified"] is True and perfect["pass"] is True
    assert perfect["identical"] == 28 and perfect["compared"] == 56, (
        "the MERGED count is exactly what must not drive the verdict")

    one_bad = gate.certify_sweep(_sweep_summary(gate, 28, 27, 28, 0))
    assert one_bad["certified"] is False and one_bad["pass"] is False

    never_diverged = gate.certify_sweep(_sweep_summary(gate, 28, 28, 28, 28))
    assert never_diverged["certified"] is True
    assert never_diverged["pass"] is False, "the fusion control must bite"
    assert "fusion-control-never-diverged" in never_diverged[
        "fusion_control_note"]

    no_device = gate.certify_sweep(_sweep_summary(gate, 0, 0, 0, 0))
    assert no_device["certified"] is False and no_device["pass"] is False


def test_mirror_codes_null_is_measured_with_a_live_control():
    """The host half of the two-mirror-code null must not be a tautology. It
    reads the COMPILED SURFACE (which never names either mirror code) and the
    PLAN, and f10's mutant is the control proving the scan can see such a name.
    """
    gate = _gate_module()
    verdict = gate.mirror_codes_share_one_arm()
    assert verdict["compiled_surface_mentions_a_mirror_code"] == []
    assert verdict["f10_mutant_mentions_one"], "the scan is not armed"
    assert verdict["check_is_armed"] is True
    assert verdict["plan_fields_differing_apart_from_the_code"] == []
    assert verdict["identical"] is True


def test_the_gate_carries_three_simultaneous_mirror_planes():
    """The predicate ADMITS three folded axes, so a later plan_step round would
    route one. Nothing in the family carried a case with more than two."""
    gate = _gate_module()
    folds = [sum(1 for code in case["codes"] if code in folded.MIRROR_CODES)
             for case in gate.sweep_cases()]
    assert max(folds) == 3, "the sweep carries no three-plane case"
    assert any(len(spec["fold"]) == 3 for spec in gate.REFERENCE_GRIDS), (
        "no reference grid folds three axes")


def test_the_identity_leg_carries_a_slot_level_counterexample():
    """Without a case whose live row is not E_a and whose bytes still agree, a
    ROW-level reachability predicate passes the leg unchanged."""
    gate = _gate_module()
    counterexamples = [
        case for case in gate.IDENTITY_CASES
        if case["rows"].startswith("single_")
        and case["expect_identical"]
        and not case["rows"].endswith("_E" + AXES["XYZ".index(case["fold"])])]
    assert len(counterexamples) == 3, [c["name"] for c in gate.IDENTITY_CASES]


def test_the_thin_fold_refusal_case_builds_a_grid_that_is_actually_thin():
    """A folded PERIODIC axis bottoms out at 3 stored cells, so the clause
    ``stored_cells <= MIRROR_SOURCE_INDEX`` is unreachable that way and the case
    recorded the backend clause alone — indistinguishable from a pristine grid.
    A folded METALLIC axis reaches it."""
    gate = _gate_module()
    for cell_x in (0.05, 0.1, 0.15, 0.2):
        grid = Grid(resolution=10.0, cell_size=(cell_x, 1.2, 1.2), courant=0.35,
                    symmetry=(Mirror("X", 1),), xp=np)
        assert grid.stored_cells(0) > folded.MIRROR_SOURCE_INDEX
    fields, pml = gate.build_reference_fields(np, dict(
        name="thin", fold="X", phase=1, cell=(0.15, 1.2, 1.2),
        boundaries="metallic", courant=0.35, rows="varying_full"))
    assert fields.grid.stored_cells(0) <= folded.MIRROR_SOURCE_INDEX
    reasons = folded.folded_offdiag_constitutive_coverage(fields, pml).reasons
    assert any("2 stored cells" in reason for reason in reasons)


def test_the_bfast_refusal_case_can_be_built_at_all():
    """``bfast_active`` is a read-only property over ``bfast_scaled_k``, so
    ``object.__setattr__`` RAISES and the case that used it was silently dropped
    by a bare ``except: continue`` while the leg still reported every case
    passing. It must come from the CONSTRUCTOR."""
    grid = Grid(resolution=10.0, cell_size=(2.0, 1.2, 1.2), courant=0.35,
                symmetry=(Mirror("X", 1),), xp=np)
    with pytest.raises(AttributeError):
        object.__setattr__(grid, "bfast_active", True)
    gate = _gate_module()
    fields, pml = gate.build_reference_fields(np, dict(
        name="bfast", fold="X", phase=1, cell=(2.0, 1.2, 1.2), boundaries=None,
        courant=0.35, rows="varying_full",
        grid=dict(bfast_scaled_k=(0.3, 0.0, 0.0))))
    assert fields.grid.bfast_active is True
    reasons = folded.folded_offdiag_constitutive_coverage(fields, pml).reasons
    assert any("BFAST" in reason for reason in reasons)


def test_mirror_ghost_axes_is_the_same_fact_as_the_boundary_codes():
    """``MG_*`` and ``BC*`` are one fact derived in one place; a disagreement
    reads stored row 2 without the parity, or applies the parity to an ordinary
    neighbour."""
    for codes in ((0, 1, 2), (3, 3, 3), (0, 0, 0), (2, 1, 3)):
        flags = folded.mirror_ghost_axes(codes)
        assert flags == tuple(int(c in folded.MIRROR_CODES) for c in codes)
    plan = _plan(codes=(2, 0, 3), weights=(-1.0, 1.0, 1.0))
    assert plan.ghost_axes == (1, 0, 1)


# ---------------------------------------------------------------------------
# The DEVICE-HALF machinery, at the merge bar
# ---------------------------------------------------------------------------
#
# None of the launches below happen on a laptop, but the DECISIONS the device
# legs make are pure functions and every one of them has a way of turning a
# harness fault into a certification. Those are pinned here, so a classifier
# that started calling a never-launched mutant "caught" fails in the ordinary
# test run rather than in an artifact nobody re-reads.

def _distinct():
    return {"cache_keys_differ": True, "ptx_available": True,
            "ptx_differs_from_every_shipped": True}


@pytest.mark.parametrize("expectation,moved,launches,hits,distinct,status", [
    # HARNESS FAULTS come first and are never reported as measurements.
    (True, 0, 1, 0, _distinct(), "DISARMED"),
    (True, 0, 0, 3, _distinct(), "NO-LAUNCH"),
    (True, 99, 1, 3, {"cache_keys_differ": False}, "STALE-BINARY"),
    (True, 99, 1, 3, {"cache_keys_differ": True, "ptx_available": True,
                      "ptx_differs_from_every_shipped": False}, "STALE-BINARY"),
    # A launched, distinct mutant: now the verdict is about the kernel.
    (True, 512, 1, 3, _distinct(), "CAUGHT"),
    (True, 0, 1, 3, _distinct(), "NEEDLE-MISSED"),
    (False, 0, 1, 1, _distinct(), "NULL-AS-PREDICTED"),
    (False, 7, 1, 1, _distinct(), "UNEXPECTEDLY-CAUGHT"),
    (None, 5, 1, 2, _distinct(), "RECORDED-CAUGHT"),
    (None, 0, 1, 2, _distinct(), "RECORDED-NOT-CAUGHT"),
])
def test_the_mutation_classifier_separates_harness_faults_from_findings(
        expectation, moved, launches, hits, distinct, status):
    """A mutant that never matched, never launched, or compiled to the shipped
    bytes says nothing about the kernel — and each has its OWN name, because
    'NEEDLE-MISSED' would blame the needle for a harness fault.

    The STALE-BINARY rows are platform fact (c) made a test: a renamed mutant
    can be served the shipped binary, and a leg that read the resulting equality
    as 'not caught' would certify a defect it never introduced.
    """
    gate = _gate_module()
    assert gate.classify_mutation(expectation, moved, launches, distinct,
                                  hits) == status


def test_every_mutation_failure_status_fails_the_leg():
    """The classifier's names are only useful if the verdict reads them: every
    status that means 'this row measured nothing, or measured the wrong thing'
    must be in the failure set, and the three PASSING outcomes must not."""
    gate = _gate_module()
    for status in ("DISARMED", "NO-LAUNCH", "STALE-BINARY", "NEEDLE-MISSED",
                   "UNEXPECTEDLY-CAUGHT", "ERROR"):
        assert status in gate.MUTATION_FAILURE_STATUSES
    for status in ("CAUGHT", "NULL-AS-PREDICTED", "RECORDED-CAUGHT",
                   "RECORDED-NOT-CAUGHT"):
        assert status not in gate.MUTATION_FAILURE_STATUSES


def _license_row(name, predicted, metallic_same, periodic_same,
                 new_same=True, launches=1):
    return {
        "name": name,
        "predicted_certified_diverges": predicted,
        "device_agrees_with_prediction": (not metallic_same) == predicted,
        "new_kernel": {"bit_identical": new_same, "launches": launches},
        "certified_metallic": {"bit_identical": metallic_same,
                               "launches": launches},
        "certified_periodic": {"bit_identical": periodic_same,
                               "launches": launches},
    }


def test_the_licensing_verdict_is_two_sided():
    """The leg that licenses a new kernel must be able to FAIL in both
    directions, and each conjunct closes a different way of being wrong.

    A certified body that diverged EVERYWHERE — including where the fold cannot
    be seen — is evidence of a broken harness, not of an insufficient kernel;
    one that diverged NOWHERE licenses nothing; and a divergence under only one
    reading of the fold leaves 'you handed it the wrong arm' available as an
    answer. All three must be refused.
    """
    gate = _gate_module()

    healthy = {"cases": [_license_row("live", True, False, False),
                         _license_row("unreachable", False, True, True)],
               "new_kernel_identical": 2}
    assert gate.certify_license(dict(healthy))["certified"] is True

    diverges_everywhere = {
        "cases": [_license_row("live", True, False, False),
                  _license_row("unreachable", False, False, False)],
        "new_kernel_identical": 2}
    verdict = gate.certify_license(dict(diverges_everywhere))
    assert verdict["certified"] is False
    assert verdict["prediction_disagreements"] == ["unreachable"]

    diverges_nowhere = {
        "cases": [_license_row("live", False, True, True),
                  _license_row("unreachable", False, True, True)],
        "new_kernel_identical": 2}
    assert gate.certify_license(dict(diverges_nowhere))["certified"] is False

    one_reading_only = {
        "cases": [_license_row("live", True, False, True),
                  _license_row("unreachable", False, True, True)],
        "new_kernel_identical": 2}
    assert gate.certify_license(dict(one_reading_only))["certified"] is False

    new_kernel_wrong = {
        "cases": [_license_row("live", True, False, False, new_same=False),
                  _license_row("unreachable", False, True, True)],
        "new_kernel_identical": 1}
    assert gate.certify_license(dict(new_kernel_wrong))["certified"] is False

    never_launched = {
        "cases": [_license_row("live", True, False, False, launches=0),
                  _license_row("unreachable", False, True, True)],
        "new_kernel_identical": 2}
    assert gate.certify_license(dict(never_launched))["certified"] is False


def test_every_device_half_is_declared_written():
    """The gate refuses to certify a leg whose device half does not exist. This
    pins that no leg is silently re-flagged as unwritten while the artifact
    still reports a verdict for it."""
    gate = _gate_module()
    assert set(gate.DEVICE_HALF_IMPLEMENTED) == set(gate.DEVICE_LEGS)
    assert all(gate.DEVICE_HALF_IMPLEMENTED.values())
    assert gate.DEVICE_HALF_TODO == {}
    assert "license" in gate.DEVICE_LEGS


def test_the_licensing_leg_runs_on_every_reference_configuration():
    """The licensing question is asked of the SAME real configurations the
    reference leg pins against ``stepping.update_E`` — not of a friendlier
    hand-picked subset — and its enumeration includes the two grids where the
    fold is deliberately unreachable, which are the only rows that can supply
    the 'it agrees somewhere' half of the verdict."""
    gate = _gate_module()
    names = [spec["name"] for spec in gate.REFERENCE_GRIDS]
    assert "foldX_only_row_x_unreachable" in names
    assert "foldX_single_Ey_Ez_no_x_partner" in names
    assert len(names) >= 16


def test_the_composition_probe_carries_the_signed_zero_lattice_seeding():
    """A plain zero-init is a FIXED POINT of this sub-step, so the composition
    probe's needle class must be the +-0 LATTICE, and the case that carries it
    must run with NO source (a source would flood the signed-zero background
    the class exists to watch)."""
    if str(PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(PARITY_DIR))
    probe = importlib.import_module(
        "probe_triton_folded_offdiag_composition")
    seeded = [case for case in probe.CASES
              if case["seeding"] == "signed_zero_lattice"]
    assert len(seeded) == 1
    assert seeded[0]["sources"] == "none"
    assert {case["seeding"] for case in probe.CASES} == {
        "random", "signed_zero_lattice"}


def test_the_composition_probe_spans_the_fold_case_discipline():
    """Both terminations, both plane phases, one AND two folded axes, and a
    reduced run — asserted from the enumeration, which is data."""
    if str(PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(PARITY_DIR))
    probe = importlib.import_module(
        "probe_triton_folded_offdiag_composition")
    phases = {phase for case in probe.CASES
              for _axis, phase in case["mirrors"]}
    assert phases == {1, -1}
    assert any("metallic" in case["boundaries"] for case in probe.CASES)
    assert any(all(b == "periodic" for b in case["boundaries"])
               for case in probe.CASES)
    assert {len(case["mirrors"]) for case in probe.CASES} == {1, 2}
    assert any(case["dimensions"] == 2 for case in probe.CASES)
    assert any(not float(case["courant"]).is_integer()
               and abs(case["courant"] * 256 - round(case["courant"] * 256)) < 1e-9
               and (round(case["courant"] * 256)
                    & (round(case["courant"] * 256) - 1)) != 0
               for case in probe.CASES)
    assert set(probe.SLOTS) == {"step_B", "fill_B", "update_H", "step_D",
                                "fill_D", "update_E"}
    assert len(probe.MUTATIONS) == 2


def test_the_ghost_fill_install_predicate_names_the_measured_conflict():
    """The composition probe installs the COMBINED near+far ghost fill at the
    driver's NEAR slot, and ``zero_metal_*`` runs BETWEEN the two fill calls
    (driver.py:3208-3210). That reordering is byte-visible exactly when a far
    reflect row and a metallic axis are live at once — measured: the one case in
    the probe's list with a folded PERIODIC EVEN axis beside a metallic axis
    diverged at step 1 in 48 words starting at ``Bx``, three sub-steps upstream
    of this family's ``update_E``, while every other case was exact.

    The predicate is pinned against the ENUMERATION rather than against a
    re-derivation of itself: it must refuse exactly one of the probe's cases,
    and that case must be the one with both conditions.
    """
    if str(PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(PARITY_DIR))
    probe = importlib.import_module("probe_triton_folded_offdiag_composition")
    refused = []
    for case in probe.CASES:
        driver = probe.build_driver(case, np, prefer_gpu=False)
        try:
            sound, why = probe.ghost_fill_install_is_sound(driver)
            reflect = stepping._far_reflect_rows(driver.grid)
            far = any(row is not None for row in reflect)
            metal = any(driver.grid.is_metallic(a) for a in range(3))
            assert sound is not (far and metal)
            if not sound:
                refused.append(case["name"])
                assert "zero_metal" in why and "ARRAY PATH" in why
        finally:
            driver.close()
    assert refused == ["foldY_partial_row"]
    # And the four-slot fallback still carries the slot this family exists for.
    assert "update_E" in probe.CORE_SLOTS
    assert set(probe.CORE_SLOTS) < set(probe.SLOTS)


def test_the_attribution_leg_can_exonerate_and_can_convict():
    """The attribution leg's four installs are only evidence if one of them
    removes THIS family's kernel from the run entirely: without that row, "the
    six-slot install diverged" is compatible with the kernel being at fault.
    """
    if str(PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(PARITY_DIR))
    probe = importlib.import_module("probe_triton_folded_offdiag_composition")
    subsets = {label: (slots, expect)
               for label, slots, expect in probe.ATTRIBUTION_SUBSETS}
    assert "update_E" not in subsets[
        "certified_five_slots_array_path_update_E"][0]
    assert subsets["certified_five_slots_array_path_update_E"][1] is False
    assert subsets["all_six_slots"][1] is False
    assert subsets["four_core_slots_array_path_fills"][1] is True
    assert subsets["update_E_only"][1] is True
    # Both directions are represented, so the leg can fail either way.
    assert {expect for _slots, expect in subsets.values()} == {True, False}
