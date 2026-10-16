"""The CUDA COMPLEX cylindrical (Dcyl, |m| >= 1) curl slice: predicate, text, constants.

THE BIT-IDENTITY GATE LIVES ELSEWHERE -- it needs a device and a sweep, so it is
``parity/meep_gpu/gate_cuda_cylindrical_complex.py``. What is pinned HERE is the rest
of the slice's silent-failure surface, and every failure below is a plausible, smooth,
WRONG field rather than a crash:

* THE PREDICATE. Every configuration it refuses would be mis-stepped if it leaked, so
  the refusals are mutated and every mutation must be caught. A predicate no mutation
  exercises is indistinguishable from one that returns True.
* THE THREE-WAY DISJOINTNESS. There are now THREE Dcyl-or-complex curl pairs and each
  of the other two must STILL refuse what this one admits -- the real Dcyl pair by
  storage, the Cartesian complex pair by coordinates -- and this one must refuse what
  they admit. If adding this pair widened either, a run would reach a kernel with the
  wrong stride or the wrong arithmetic. Asserted in both directions on real objects.
* THE DEVICE TEXT. Five things separate this pair from the certified complex Cartesian
  one, four of them ARITHMETIC (the radial difference the scan supplies, the i*m/r
  pass and its ordering against the recurrence and the mask, the |m| = 1 curl-row
  replacement, the per-|m| axis rules). The gate can see a wrong answer; only a text
  pin can see a grouping change that is exact at one Courant and wrong at another, or
  an ordering that happens to be invisible on every case the sweep reaches.
* THE PRELUDE IS IMPORTED, NOT COPIED. The certified complex strings are this family's
  whole non-cylindrical body; a second copy would drift and would silently un-arm the
  shared mutation battery, whose needles are those exact characters.
* THE HOST-SIDE CONSTANTS. ``imr_rinv_row`` and ``axis_courant_words`` re-derive the
  array path's i*m/r ``rinv`` row and its axis rules' double factors, whose words are
  load-bearing down to the sign of a zero. They are checked against ``stepping``'s
  own values by CALLING both, not by reading either.
* THE ARGUMENT ORDER. The kernel signature and the launcher's tuple are two lists that
  must agree; ``RawKernel`` does not check, so a swapped pair is a wrong answer at a
  wrong pointer.

THE SOURCE IS IMPORTED, NOT READ WITH ``ast``, and that is why
``cylindrical_complex_kernels`` imports CuPy LAZILY: its device text is built from the
imported prelude, so there is no module-level string for ``ast`` to read, and the
merge bar is a machine with no GPU. The module's own docstring gives the argument.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import types

import numpy
import pytest

from ..fields import IYEE_SHIFTS, Fields
from ..grid import Grid
from ..pml import PML
from .. import stepping
from . import complex_emitter, coverage, cylindrical_coverage
from . import cylindrical_complex_kernels as family

SUB_STEPS = ("step_B", "step_D")


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__``.

    The predicate refuses any backend whose module is not named "cupy", which is the
    one thing about the real device library that cannot be reproduced off device --
    and the only thing this stands in for. Everything else the fixture exercises is a
    real ``Grid``/``Fields``/``PML``.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


def build(shape=(16, 1, 20), m=1, z_kind="metallic", complex_storage=True,
          courant=0.35, cylindrical=True, **grid_kwargs):
    """A frozen ``(fields, layer, grid)`` triple.

    ``Grid`` owns the r axis's boundary pair on a cylindrical cell (grid.py:498-508),
    so only z is passed, and the PML is asked for the HIGH r face only: cell 0 is the
    axis, a boundary condition rather than a wall.
    """
    xp = _NumpyWearingCupysName()
    if cylindrical:
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), 0.0, float(shape[2])),
                    cylindrical=True, m=m, boundaries={"z": z_kind},
                    courant=courant, xp=xp, **grid_kwargs)
        thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    else:
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
                    boundaries=("periodic", "periodic", z_kind),
                    courant=courant, xp=xp, **grid_kwargs)
        thickness = tuple((0, 0) if grid.shape[a] < 6 else (2, 2) for a in range(3))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = PML(grid=grid, thickness=thickness)
    return fields, layer, grid


def covered(fields, layer, grid, sub_step="step_B"):
    return cylindrical_coverage.covers_pml_cylindrical_complex_curl(
        fields, layer, grid, sub_step)


def source(sub_step="step_B"):
    return family.cylindrical_complex_source(sub_step)


# --------------------------------------------------------------------------
# The predicate: it admits what it must and refuses everything else BY NAME
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("m", [0, 1, -1, 2, 3, 5])
@pytest.mark.parametrize("z_kind", ["metallic", "periodic"])
def test_it_admits_every_m_class_and_both_z_terminations(sub_step, m, z_kind):
    """The whole domain this pair claims, on real objects rather than a stub.

    m = 0 is in the list since 2026-09-04: a complex-storage m = 0 run is the third
    ``m_class`` arm, and the corpus row ``dipole_in_vacuum_cyl_off_axis.py`` is one.
    """
    fields, layer, grid = build(m=m, z_kind=z_kind)
    assert covered(fields, layer, grid, sub_step) == (True, "covered")


def test_the_accurate_near_axis_branch_is_admitted():
    """``accurate_fields_near_cylorigin`` selects ZERO_ROWS and must not be a refusal.

    ``Grid`` refuses a Courant above ~1/(|m| + 0.5) on that branch, so the case is
    built at a Courant it accepts; the point is that the FLAG is read and admitted,
    not that any Courant is.
    """
    fields, layer, grid = build(m=3, courant=0.2,
                                accurate_fields_near_cylorigin=True)
    assert grid.accurate_fields_near_cylorigin is True
    assert covered(fields, layer, grid) == (True, "covered")
    assert family.zero_rows(3, grid.accurate_fields_near_cylorigin) == 1


@pytest.mark.parametrize("m", [1, -1, 3])
def test_a_run_with_no_absorber_is_refused(m):
    fields, layer, grid = build(m=m)
    ok, why = cylindrical_coverage.covers_pml_cylindrical_complex_curl(
        fields, None, grid, "step_B")
    assert not ok and "no active PML layer" in why


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_m_zero_partitions_on_storage_alone(sub_step):
    """m = 0 with complex64 storage is ADMITTED; m = 0 with real storage is not.

    Until 2026-09-04 this predicate refused m = 0 by name and the real pair refused
    complex storage by name, so the complex m = 0 row (a constructible configuration
    the corpus carries) fell between the two. The split is now the storage clause
    alone, in both directions -- and neither pair admits the other's triple.
    """
    fields, layer, grid = build(m=0, complex_storage=True)
    assert covered(fields, layer, grid, sub_step) == (True, "covered")
    real_ok, real_why = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert real_ok is False and "complex64 storage" in real_why

    fields, layer, grid = build(m=0, complex_storage=False)
    ok, why = covered(fields, layer, grid, sub_step)
    assert ok is False and "real float32 storage" in why
    assert cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)[0] is True


def test_real_storage_is_refused_by_name():
    fields, layer, grid = build(m=1, complex_storage=False)
    ok, why = covered(fields, layer, grid)
    assert not ok and "real float32 storage" in why


def test_a_cartesian_grid_is_refused():
    fields, layer, grid = build(shape=(12, 12, 12), cylindrical=False)
    ok, why = covered(fields, layer, grid)
    assert not ok and "not a cylindrical (Dcyl) grid" in why


def test_a_bloch_phase_is_refused_by_name():
    """No Dcyl corpus row carries one and the kernel passes every phase flag 0."""
    fields, layer, grid = build(m=1, z_kind="periodic", k_point=(0.0, 0.0, 0.25))
    ok, why = covered(fields, layer, grid)
    assert not ok and "Bloch phase on a cylindrical grid" in why


def test_a_backend_that_is_not_cupy_is_refused():
    fields, layer, grid = build(m=1)
    grid.xp = numpy
    ok, why = covered(fields, layer, grid)
    assert not ok and why == "backend is not CuPy"


def test_a_conductivity_disqualifies_only_the_sub_step_that_carries_it():
    """``_apply_curl`` reads the conductivity per TERM (stepping.py:508)."""
    fields, layer, grid = build(m=1)
    real = fields.condfac_for
    fields.condfac_for = lambda name: object() if name in ("Dx", "Dy", "Dz") else real(name)
    try:
        ok_b, _ = covered(fields, layer, grid, "step_B")
        ok_d, why_d = covered(fields, layer, grid, "step_D")
    finally:
        fields.condfac_for = real
    assert ok_b is True
    assert ok_d is False and "carries a conductivity" in why_d


def test_an_unreadable_conductivity_reader_is_refused_outright():
    """Inferring "no conductivity" from an absent reader is admission by absence."""
    fields, layer, grid = build(m=1)
    fields.condfac_for = None
    ok, why = covered(fields, layer, grid)
    assert not ok and "does not expose condfac_for" in why


def test_a_single_radial_row_is_refused_and_the_array_path_agrees():
    """nr = 1 is the |m| = 1 axis increment reading past the end of Ez.

    The refusal is checked against what the ARRAY PATH does on the same grid rather
    than against the clause's own prose: NumPy raises ``IndexError`` from
    ``stepping.py:671``'s ``xp.take(Ez, 1, axis=0)``, which is the loud half of the
    same fact. (On CuPy ``take`` does not bounds-check, which is why the clause is
    the only thing between a GPU run and a silently wrong axis row -- that half
    cannot be measured here and is not asserted here.)
    """
    xp = _NumpyWearingCupysName()
    grid = Grid(resolution=1.0, cell_size=(1.0, 0.0, 20.0), cylindrical=True, m=1,
                boundaries={"z": "metallic"}, courant=0.35, xp=xp)
    assert tuple(grid.shape)[0] == 1
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = PML(grid=grid, thickness={"z": 4})
    ok, why = covered(fields, layer, grid)
    assert not ok and "radial extent is 1 cell" in why

    with pytest.raises(IndexError):
        stepping.step_B(fields, layer)


@pytest.mark.parametrize("attribute", ["Bx", "fu_Dz", "Ey", "Hz"])
def test_a_volume_with_the_wrong_dtype_is_refused(attribute):
    """Real storage read as complex is a scrambled volume, not a launch failure."""
    fields, layer, grid = build(m=1)
    setattr(fields, attribute, numpy.zeros(tuple(grid.shape), dtype=numpy.float32))
    ok, why = covered(fields, layer, grid)
    assert not ok and attribute in why and "not complex64" in why


def test_a_non_contiguous_volume_is_refused():
    fields, layer, grid = build(shape=(16, 1, 20), m=1)
    padded = numpy.zeros((16, 1, 40), dtype=numpy.complex64)
    fields.Bz = padded[:, :, ::2]
    ok, why = covered(fields, layer, grid)
    assert not ok and "Bz is not C-contiguous" in why


def test_the_shared_helpers_this_module_borrows_all_still_exist():
    """A rename in ``coverage.py`` must fail HERE, not at the first Dcyl run."""
    for name in cylindrical_coverage.SHARED_HELPERS:
        assert hasattr(coverage, name), name
    assert "_complex_volume_problem" in cylindrical_coverage.SHARED_HELPERS


# --------------------------------------------------------------------------
# THE THREE-WAY DISJOINTNESS -- in both directions, on real objects
# --------------------------------------------------------------------------

@pytest.mark.parametrize("m", [0, 1, -1, 3])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_neither_existing_pair_admits_what_this_one_admits(m, sub_step):
    """Adding a third pair must not widen either of the two it sits beside."""
    fields, layer, grid = build(m=m)
    assert covered(fields, layer, grid, sub_step)[0] is True

    real_ok, real_why = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert real_ok is False and "complex64 storage" in real_why

    cart_ok, cart_why = coverage.covers_real_pml_complex_curl(
        fields, layer, grid, sub_step)
    assert cart_ok is False and "cylindrical (Dcyl)" in cart_why

    plain_ok, plain_why = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert plain_ok is False


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_this_pair_refuses_what_the_real_dcyl_pair_admits(sub_step):
    """The m = 0 real-storage triple belongs to the other pair and only to it."""
    fields, layer, grid = build(m=0, complex_storage=False)
    real_ok, _ = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert real_ok is True
    ok, why = covered(fields, layer, grid, sub_step)
    assert ok is False and ("grid.m = 0" in why or "real float32 storage" in why)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_this_pair_refuses_what_the_cartesian_complex_pair_admits(sub_step):
    fields, layer, grid = build(shape=(12, 12, 12), cylindrical=False)
    cart_ok, _ = coverage.covers_real_pml_complex_curl(
        fields, layer, grid, sub_step)
    assert cart_ok is True
    ok, why = covered(fields, layer, grid, sub_step)
    assert ok is False and "not a cylindrical (Dcyl) grid" in why


def test_the_boundary_code_mapping_is_the_one_the_real_pair_uses():
    """One defensible answer for the r axis, and ONE spelling of it."""
    fields, layer, grid = build(m=1)
    codes = family.cylindrical_complex_boundary_codes(grid)
    kinds = tuple(coverage.real_pml_boundary_kinds(grid))
    assert kinds[0] == cylindrical_coverage.CYL_AXIS
    assert codes == cylindrical_coverage.cylindrical_boundary_codes(kinds)
    assert codes[0] == coverage.BC_CODES[cylindrical_coverage.METALLIC]
    assert codes[1] == coverage.BC_CODES[cylindrical_coverage.PERIODIC]


# --------------------------------------------------------------------------
# The host-side constants, checked by CALLING the array path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("m", [0, 1, -1, 2, 3, 5, -4])
def test_m_class_and_zero_rows_agree_with_stepping(m):
    """``zero_rows`` must be ``_cylindrical_axis_rows``' slice length, both branches.

    The m class is ``stepping``'s own three-way branch (``m == 0`` / ``abs(m) == 1`` /
    else, ``_cylindrical_axis_zero_B`` :657-671), including the m = 0 arm.
    """
    assert family.m_class(m) == (family.M_ZERO if m == 0 else
                                 family.M_ONE if abs(m) == 1 else family.M_MANY)
    for accurate in (False, True):
        rows = stepping._cylindrical_axis_rows(
            types.SimpleNamespace(m=m, accurate_fields_near_cylorigin=accurate))
        expected = 0 if abs(m) < 2 else (rows.stop - rows.start)
        assert family.zero_rows(m, accurate) == expected, (m, accurate)


def test_m_class_zero_is_the_third_arm_and_the_three_codes_are_distinct():
    """The device text branches on the integer; three arms need three values."""
    assert family.m_class(0) == family.M_ZERO == 0
    assert len({family.M_ZERO, family.M_ONE, family.M_MANY}) == 3
    assert family.zero_rows(0, False) == 0 and family.zero_rows(0, True) == 0


@pytest.mark.parametrize("courant", [0.5, 0.35, 0.3, 0.123456789])
def test_the_axis_courant_words_are_the_double_factors_stepping_multiplies_by(courant):
    """The axis rules' factors travel as DOUBLES: ``-Courant`` on B, ``+Courant`` and
    ``4*Courant`` on D, each exact in double (``stepping._realnum_double_product``).

    The product each feeds is the array path's, measured on random complex64 operands
    as uint32 words; and at a Courant that is not a float32 word, a float32 factor is a
    different word somewhere, which is why the factor is not rounded on the host.
    """
    _fields, _layer, grid = build(shape=(8, 1, 6), m=1, courant=courant)
    (minus,) = family.axis_courant_words("step_B", grid)
    plus, four = family.axis_courant_words("step_D", grid)
    for word in (minus, plus, four):
        assert isinstance(word, numpy.float64)
    assert minus == -float(grid.courant) and plus == float(grid.courant)
    assert four == 4.0 * float(grid.courant)
    rng = numpy.random.default_rng(int(courant * 1e6))
    values = (rng.uniform(-1, 1, 64) + 1j * rng.uniform(-1, 1, 64)).astype(numpy.complex64)
    differs = False
    for word in (minus, plus, four):
        theirs = stepping._realnum_double_product(float(word), values)
        mine = numpy.empty_like(values)
        mine.real = (values.real.astype(numpy.float64) * word).astype(numpy.float32)
        mine.imag = (values.imag.astype(numpy.float64) * word).astype(numpy.float32)
        assert numpy.array_equal(mine.view(numpy.uint32), theirs.view(numpy.uint32))
        rounded = values * numpy.float32(word)
        differs |= not numpy.array_equal(rounded.view(numpy.uint32), theirs.view(numpy.uint32))
    assert differs == (float(numpy.float32(grid.courant)) != float(grid.courant)), courant
    with pytest.raises(ValueError):
        family.axis_courant_words("update_H", grid)


@pytest.mark.parametrize("m", [1, -1, 2, 3, -5])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_imr_increment_is_word_for_word_the_array_paths(m, sub_step):
    """Built by CALLING ``stepping._cylindrical_imr_increment``, never by re-reading it.

    The kernel forms ``((rinv*g.im) * -1, rinv*g.re)`` from the bound real row
    (``imr_increment``); the array path's increment is the same quarter turn of the
    partner by its own cached row. Compared on random complex64 partners as uint32
    words, for both of the sub-step's terms, whose rows are indexed by the TARGET's own
    radial Yee shift -- one 0 and one 1 per side, so a row built with one shift for both
    targets is a defect the gate arms.
    """
    fields, _layer, grid = build(shape=(9, 1, 4), m=m)
    rows = grid.shape[0]
    rng = numpy.random.default_rng(97 + 31 * (m + 8))
    theirs_terms = {"step_B": stepping._CYLINDRICAL_IMR_B,
                    "step_D": stepping._CYLINDRICAL_IMR_D}[sub_step]
    bound = family.imr_rows_for(sub_step, grid, rows)
    assert len(bound) == 2
    for (index, target, sign), (term, _partner, their_sign), row in zip(
            family.IMR_TERMS[sub_step], theirs_terms, bound):
        assert (term.target, their_sign) == (target, sign)
        assert index in (0, 2)
        shape = tuple(grid.shape)
        partner = (rng.uniform(-1, 1, shape)
                   + 1j * rng.uniform(-1, 1, shape)).astype(numpy.complex64)
        theirs = stepping._cylindrical_imr_increment(fields, target, partner, sign)
        rinv = family.imr_rinv_row(grid, target, sign, rows)
        assert rinv.dtype == numpy.float32 and rinv.shape == (rows,)
        assert numpy.array_equal(rinv.view(numpy.uint32), row.view(numpy.uint32))
        column = rinv.reshape(-1, 1, 1)
        mine = numpy.empty_like(partner)
        mine.real = (column * partner.imag) * numpy.float32(-1.0)
        mine.imag = column * partner.real
        assert numpy.array_equal(mine.view(numpy.uint32), theirs.view(numpy.uint32)), target
    assert {IYEE_SHIFTS[target][0] for _i, target, _s in family.IMR_TERMS[sub_step]} == {0, 1}


def _seeded_axis_grid(m, courant):
    fields, layer, grid = build(shape=(8, 1, 6), m=m, courant=courant)
    rng = numpy.random.default_rng(4 + 31 * (m + 8))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        volume = getattr(fields, name)
        volume.real = rng.uniform(-1, 1, volume.shape).astype(numpy.float32)
        volume.imag = rng.uniform(-1, 1, volume.shape).astype(numpy.float32)
    return fields, layer, grid


def _double_product(word, values):
    out = numpy.empty_like(values)
    out.real = (values.real.astype(numpy.float64) * word).astype(numpy.float32)
    out.imag = (values.imag.astype(numpy.float64) * word).astype(numpy.float32)
    return out


def _words(values):
    return numpy.ascontiguousarray(values).view(numpy.float32).view(numpy.uint32)


@pytest.mark.parametrize("m", [1, -1])
@pytest.mark.parametrize("courant", [0.35, 0.5, 0.3])
def test_the_b_axis_increment_row_matches_steppings_own(m, courant):
    """The whole |m| = 1 B-side increment, from ``stepping``, on a seeded grid.

    ``_cylindrical_axis_increment_B`` is CALLED; the kernel's spelling -- the turn
    ``(m*Ez1.im, (m*Ez1.re) * -1)`` by the bound ``m_word``, the bracket in float32 left
    to right, one double product with the bound ``-Courant`` -- is reassembled on the host
    and compared word for word.
    """
    fields, layer, grid = _seeded_axis_grid(m, courant)
    electric = {n: getattr(fields, n) for n in ("Ex", "Ey", "Ez")}
    boundaries = stepping._boundary_kinds(grid, layer)
    target, theirs = stepping._cylindrical_axis_increment_B(
        fields, electric, boundaries, (None, None, None), float(grid.dt / grid.dx))
    assert target == "Bx"
    assert family.AXIS_INCREMENT_TARGET["step_B"] == 0
    (minus,) = family.axis_courant_words("step_B", grid)
    m_word = numpy.float32(float(m))
    ep = electric["Ey"]
    ep_above = stepping._shift_up(numpy, ep, 2, boundaries[2])
    e1 = electric["Ez"][1]
    turn = numpy.empty_like(e1)
    turn.real = m_word * e1.imag
    turn.imag = (m_word * e1.real) * numpy.float32(-1.0)
    mine = _double_product(minus, (ep[0] - ep_above[0]) - turn)
    assert numpy.array_equal(_words(mine), _words(theirs))


@pytest.mark.parametrize("m", [1, -1])
@pytest.mark.parametrize("courant", [0.35, 0.5, 0.3])
def test_the_d_axis_increment_row_matches_steppings_own(m, courant):
    """The |m| = 1 D-side increment: ``(Hr - Hr_zdn) - 2*Hz`` then the ``+Courant``
    double product, against ``_cylindrical_axis_increment_D`` CALLED on a seeded grid."""
    fields, layer, grid = _seeded_axis_grid(m, courant)
    magnetic = {n: getattr(fields, n) for n in ("Hx", "Hy", "Hz")}
    boundaries = stepping._boundary_kinds(grid, layer)
    target, theirs = stepping._cylindrical_axis_increment_D(
        fields, magnetic, boundaries, (None, None, None), float(grid.dt / grid.dx))
    assert target == "Dy"
    assert family.AXIS_INCREMENT_TARGET["step_D"] == 1
    plus, _four = family.axis_courant_words("step_D", grid)
    hr = magnetic["Hx"]
    hr_below = stepping._shift_down(numpy, hr, 2, boundaries[2], "Hx")
    hz = magnetic["Hz"][0]
    two_hz = numpy.empty_like(hz)
    two_hz.real = numpy.float32(2.0) * hz.real
    two_hz.imag = numpy.float32(2.0) * hz.imag
    mine = _double_product(plus, (hr[0] - hr_below[0]) - two_hz)
    assert numpy.array_equal(_words(mine), _words(theirs))


# --------------------------------------------------------------------------
# The device text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_emitted_source_is_pure_ascii_and_declares_one_kernel(sub_step):
    """ASCII is a COMPILE requirement: NVRTC writes the source through a locale open."""
    text = source(sub_step)
    text.encode("ascii")
    assert family.shipped_kernel_names(text) == {family.KERNELS[sub_step][0]}


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_emitted_body_carries_no_fused_multiply_add(sub_step):
    """The certified prelude emits one part-wise body, and this family adds none of its
    own."""
    text = source(sub_step)
    assert "__fmaf_rn" not in text
    assert complex_emitter._HEAD in text
    assert complex_emitter._TAIL in text
    assert family._TEMPLATES[sub_step].replace(
        "__BOUNDS__", complex_emitter.curl_bounds_block(sub_step)) in text


def test_the_certified_strings_are_imported_not_copied():
    """The non-cylindrical body must be the certified characters, not a second copy."""
    for sub_step in SUB_STEPS:
        assert complex_emitter._PART_PRODUCTS in source(sub_step)
    # And the module must not carry its own copy of any of them.
    module_text = open(family.__file__, encoding="utf-8").read()
    for helper in ("cf_load", "cshift_up", "cshift_dn", "pml_apply", "curl_stencil",
                   "mul_field_left", "mul_coefficient_left", "rotate_field_left"):
        assert f"__device__ __forceinline__ cf {helper}" not in module_text, helper
        assert f"__device__ __forceinline__ void {helper}" not in module_text, helper


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_no_floating_point_division_anywhere(sub_step):
    """The i*m/r ``rinv`` row is bound as an array PRECISELY so the kernel performs none."""
    text = source(sub_step)
    body = text[text.index('extern "C"'):]
    stripped = re.sub(r"//[^\n]*", "", body)
    assert "/" not in stripped.replace("/ (ny * nz)", "").replace(
        "idx / nz", "").replace("idx / (ny * nz)", ""), stripped


def _target_block(text, target):
    """One target's ``{ ... }`` block of the emitted kernel, from its comment line."""
    block = text[text.index(f"    // {target}:"):]
    return block[:block.index("\n    }\n")]


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_radial_difference_is_the_whole_curl_of_one_target(sub_step):
    """Bz's (Dz's) curl is ``dtdx`` times the scan's radial difference -- ONE product,
    read at the target's own index -- and no other target reads it."""
    text = source(sub_step)
    assert text.count("cf_load(rad,") == 1
    target = family.CURL_ARRAYS[sub_step][0][2]
    block = _target_block(text, target)
    assert "cf curl = mul_coefficient_left(dtdx, cf_load(rad, idx));" in block
    # Not the stencil: the target's curl is not formed from the partner volumes.
    assert "curl_stencil" not in block
    if sub_step == "step_D":
        # Dx's Hy operand stays RAW -- the radial sum reaches Dz and nothing else.
        assert "cf f_2 = cf_load(Hy, idx);" in _target_block(text, "Dx")


_CURL_MASK = r"if \((bc_[xyz]) == BC_METALLIC && (\w) == 0\) {} = cf_zero\(\);"


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_imr_pass_follows_its_targets_recurrence_and_masks_its_own_increment(sub_step):
    """The array path's order: the curl's recurrence first, then the pass's increment,
    masked by the target's own ownership rule, through the folded ladder; target 1 takes
    none."""
    text = source(sub_step)
    targets, _sources = family.CURL_ARRAYS[sub_step]
    for index, target, _sign in family.IMR_TERMS[sub_step]:
        call = f"imr_increment(imr{index}[i],"
        assert text.count(call) == 1
        block = _target_block(text, target)
        at = block.index(call)
        recurrence = block.index(f"pml_apply({target}, fu_{target}, idx, curl,")
        fold = block.index(f"fold_apply({target}, fu_{target}, idx, inc,")
        assert recurrence < at < fold, target
        curl_masks = re.findall(_CURL_MASK.format("curl"), block[:recurrence])
        inc_masks = re.findall(_CURL_MASK.format("inc"), block[at:fold])
        assert inc_masks and inc_masks == curl_masks, target
    assert "fold_apply(" not in _target_block(text, targets[1])


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_axis_increment_replaces_the_masked_row_before_the_recurrence(sub_step):
    """REPLACE, never accumulate, after the mask and before the recurrence: the
    caller's ``curl[row 0] = -dfcnd`` (``stepping.step_B`` / ``step_D``)."""
    text = source(sub_step)
    target = family.CURL_ARRAYS[sub_step][0][family.AXIS_INCREMENT_TARGET[sub_step]]
    word = "minus_courant" if sub_step == "step_B" else "plus_courant"
    block = _target_block(text, target)
    replace = block.index("if (m_class == 1 && i == 0) {")
    assert block.index("BC_METALLIC") < replace < block.index(f"pml_apply({target},")
    assert f"curl = cf_neg(double_product({word}, " in block
    # Accumulation is the defect; it must not be the spelling.
    assert "curl = cf_add(curl" not in text and "curl.re = curl.re +" not in text
    if sub_step == "step_B":
        assert "cf e1 = cf_load(Ez, idx + sx);" in block
        assert "turn.re = m_word * e1.im;" in block
        assert "turn.im = (m_word * e1.re) * -1.0f;" in block
        assert "cf_sub(cf_sub(f_2, ss), turn)" in block
    else:
        assert ("cf bracket = cf_sub(cf_sub(f_1, sf), mul_coefficient_left(2.0f, f_2));"
                in block)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_per_m_axis_rules_are_the_array_paths(sub_step):
    """|m| = 1 zeroes Dz ONLY and its FIELD only; |m| >= 2 zeroes all six volumes."""
    text = source(sub_step)
    prefix = "B" if sub_step == "step_B" else "D"
    zeroing = text[text.index("if (m_class == 2 && i < zero_rows) {"):]
    for component in ("x", "y", "z"):
        assert f"cf_store({prefix}{component}, idx, cf_zero());" in zeroing
        assert f"cf_store(fu_{prefix}{component}, idx, cf_zero());" in zeroing
    if sub_step == "step_D":
        assert "if (m_class == 1 && i == 0) {\n        cf_store(Dz, idx, cf_zero());" in text
        # The FIELD only: fu_Dz must NOT be zeroed by the |m| = 1 rule.
        one = text.index("if (m_class == 1 && i == 0) {\n        cf_store(Dz")
        assert "fu_Dz" not in text[one:text.index("}", one)]
    else:
        # The B side does NOTHING at |m| = 1 -- read off _cylindrical_axis_zero_B,
        # which branches on m == 0 and abs(m) > 1 only.
        assert "if (m_class == 1" in text  # the curl-row increment, and only that
        assert text.count("if (m_class == 1") == 1


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_m_zero_axis_rules_are_the_array_paths_and_the_imr_terms_are_guarded(sub_step):
    """The third arm (2026-09-04), read off ``stepping`` rather than off the real pair.

    B: ``Bx[r=0] = 0``, the FIELD only (:657-659). D: the on-axis ``Dz`` post-add of
    the host-rounded ``4*Courant`` times the RAW stored Hp with the coefficient on the
    LEFT and Dz on the left of the add, then ``Dy[r=0] = 0`` (:583-587). Both i*m/r
    call sites are GUARDED on ``m_class != 0`` rather than fed a zero row, because
    ``curl - (0 * f)`` is not the identity on a ``-0.0`` curl; the |m| = 1 and
    |m| >= 2 blocks are untouched by the arm.
    """
    text = source(sub_step)
    assert text.count("if (m_class != 0) {") == 2
    for index, _target, _sign in family.IMR_TERMS[sub_step]:
        guard = text.index(f"imr_increment(imr{index}[i],")
        assert text.rindex("if (m_class != 0) {", 0, guard) > text.rindex("    {", 0, guard)
    assert text.count("if (m_class == 0 && i == 0) {") == 1
    tail = text[text.index("if (m_class == 0 && i == 0) {"):]
    tail = tail[:tail.index("\n    }")]
    if sub_step == "step_B":
        assert "cf_store(Bx, idx, cf_zero());" in tail
        assert "fu_Bx" not in tail and "By" not in tail and "Bz" not in tail
        assert "four_courant" not in text
    else:
        assert ("cf_store(Dz, idx, cf_add(cf_load(Dz, idx), "
                "double_product(four_courant, cf_load(Hy, idx))));") in tail
        assert tail.index("cf_store(Dz") < tail.index("cf_store(Dy, idx, cf_zero());")
        assert "fu_D" not in tail and "rad" not in tail
        # The tail runs AFTER every recurrence and after both |m| >= 1 blocks.
        assert text.index("if (m_class == 0 && i == 0) {") > text.rindex("pml_apply(")
        assert text.index("if (m_class == 0 && i == 0) {") > text.index(
            "if (m_class == 2 && i < zero_rows) {")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_ownership_mask_is_the_certified_bodys(sub_step):
    """Per target, on every axis whose Yee shift is 0 and which is metallic."""
    text = source(sub_step)
    expected = {"step_B": [("bc_x", "i"), ("bc_y", "j"), ("bc_z", "k")],
                "step_D": [("bc_y", "j"), ("bc_z", "k"), ("bc_x", "i"),
                           ("bc_z", "k"), ("bc_x", "i"), ("bc_y", "j")]}[sub_step]
    found = re.findall(r"if \((bc_[xyz]) == BC_METALLIC && (\w) == 0\) curl = cf_zero\(\);",
                       text)
    assert found == expected


def test_the_kernel_names_and_the_uncertified_set_partition_exactly():
    """A kernel added here without a gate verdict must be visible as uncertified."""
    shipped = set()
    for sub_step in SUB_STEPS:
        shipped |= family.shipped_kernel_names(source(sub_step))
    # THE PARTITION, not "everything is uncertified". Until 2026-08-27 this family had
    # no gate verdict and the two spellings were the same set; the gate has since
    # RELEASED both kernels under both policies (certification.json
    # cuda_cylindrical_complex_2026-08-27), so the invariant that actually matters is
    # that every shipped kernel is described by exactly one of the two sets -- a kernel
    # in neither would be shipped, wireable and described by nothing.
    assert shipped == set(family.CERTIFIED_KERNELS) | set(family.UNCERTIFIED_KERNELS)
    assert not set(family.CERTIFIED_KERNELS) & set(family.UNCERTIFIED_KERNELS)
    record = json.loads(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "certification.json"), encoding="utf-8").read())
    text = json.dumps(record)
    for name in family.UNCERTIFIED_KERNELS:
        assert f'"{name}"' not in text, (
            f"certification.json names {name}, which has no gate verdict")


def test_the_corpus_digest_moves_when_the_text_moves(monkeypatch):
    before = family.corpus_digest()
    monkeypatch.setattr(family, "_CYLINDRICAL_HELPERS",
                        family._CYLINDRICAL_HELPERS + "\n// moved\n")
    assert family.corpus_digest() != before


def test_a_sub_step_this_family_does_not_emit_is_refused():
    with pytest.raises(ValueError):
        family.cylindrical_complex_source("update_H")


# --------------------------------------------------------------------------
# The signature and the launcher's argument list
# --------------------------------------------------------------------------

_SIGNATURE = re.compile(r'extern "C" __global__ void \w+\(([^)]*)\)', re.S)


def parameters(sub_step):
    match = _SIGNATURE.search(source(sub_step))
    out = []
    for raw in match.group(1).split(","):
        cleaned = " ".join(raw.replace("__restrict__", "").split())
        kind, identifier = cleaned.rsplit(" ", 1)
        out.append((kind.strip(), identifier.lstrip("*")))
    return out


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_signature_is_the_order_the_launcher_builds(sub_step):
    """``RawKernel`` does not check; a swapped pair is a wrong answer at a wrong pointer."""
    targets, sources = family.CURL_ARRAYS[sub_step]
    # The B side binds the turn's m and the |m| = 1 factor; the D side the |m| = 1 factor
    # and the m = 0 on-axis Dz factor. Both travel as doubles.
    scalars = (["m_word", "minus_courant"] if sub_step == "step_B"
               else ["plus_courant", "four_courant"])
    expected = (list(targets) + ["fu_" + n for n in targets] + list(sources)
                + ["rad", "imr0", "imr2", "nx", "ny", "nz", "dtdx"] + scalars
                + [f"{stem}_{axis}" for axis in "xyz" for stem in ("kms", "sinv")]
                + ["bc_x", "bc_y", "bc_z", "m_class", "zero_rows", "cb"])
    assert [name for _kind, name in parameters(sub_step)] == expected


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_every_pointer_parameter_is_a_float_word_pointer(sub_step):
    """complex64 reaches the kernel as float32 WORDS; a complex type here is a lie. The
    chunk record is the one integer vector."""
    for kind, name in parameters(sub_step):
        if "*" in kind:
            expected = "int" if name == "cb" else "float"
            assert kind.replace("const", "").strip().startswith(expected), (kind, name)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_branch_axes_are_runtime_ints_and_the_courant_words_doubles(sub_step):
    """The boundary codes, the m class and ZERO_ROWS are branches, not compiled in; the
    axis factors are doubles, because a float32 factor is another word."""
    kinds = dict((name, kind) for kind, name in parameters(sub_step))
    for name in ("bc_x", "bc_y", "bc_z", "m_class", "zero_rows", "nx", "ny", "nz"):
        assert kinds[name] == "int", name
    doubles = ("minus_courant",) if sub_step == "step_B" else ("plus_courant",
                                                                "four_courant")
    for name in doubles:
        assert kinds[name] == "double", name
    assert kinds["dtdx"] == "float"


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_launcher_tuple_matches_the_signature_kind_for_kind(sub_step, monkeypatch):
    """``curl_arguments`` against the parsed signature: same length, and each slot the
    host type ``RawKernel`` marshals for that C type."""
    monkeypatch.setattr(family, "word_view", lambda array: array)
    fields, _layer, grid = build(m=1)
    targets, _sources = family.CURL_ARRAYS[sub_step]
    shape = tuple(getattr(fields, targets[0]).shape)
    radial = numpy.zeros(shape, numpy.complex64)
    rows = family.imr_rows_for(sub_step, grid, shape[0])
    tables = {f"{stem}_{axis}": numpy.zeros(4, numpy.float32)
              for axis in "xyz" for stem in ("kms", "sinv")}
    record = numpy.zeros(25, numpy.int32)
    arguments = family.curl_arguments(
        sub_step, fields, radial, rows, 0.35, 1, family.axis_courant_words(sub_step, grid),
        tables, (1, 0, 1), family.M_ONE, 0, record)
    kinds = parameters(sub_step)
    assert len(arguments) == len(kinds)
    host_type = {"int": numpy.int32, "float": numpy.float32, "double": numpy.float64}
    for (kind, name), value in zip(kinds, arguments):
        if "*" in kind:
            assert isinstance(value, numpy.ndarray), name
        else:
            assert type(value) is host_type[kind], (name, type(value))
    named = dict(zip([name for _kind, name in kinds], arguments))
    for name in targets:
        assert named[name] is getattr(fields, name)
    assert named["rad"] is radial and named["cb"] is record
    with pytest.raises(ValueError, match="radial difference"):
        family.curl_arguments(
            sub_step, fields, radial[:-1], rows, 0.35, 1,
            family.axis_courant_words(sub_step, grid), tables, (1, 0, 1), family.M_ONE, 0,
            record)


def test_the_launcher_refuses_a_call_without_its_grid_layer_or_chunk_record():
    """A defaulted grid, layer or division would be a silently different sub-step."""
    fields, layer, grid = build(m=1)
    record = numpy.zeros(25, numpy.int32)
    with pytest.raises(ValueError, match="pass the grid and the layer"):
        family.step_cylindrical_complex("step_B", fields, grid=grid, pml=None,
                                        chunk_bounds=record)
    with pytest.raises(ValueError, match="pass the grid and the layer"):
        family.step_cylindrical_complex("step_B", fields, pml=layer,
                                        chunk_bounds=record)
    with pytest.raises(TypeError):
        family.step_cylindrical_complex("step_B", fields, grid, layer)
    with pytest.raises(ValueError, match="sub_step"):
        family.step_cylindrical_complex("update_H", fields, grid, layer,
                                        chunk_bounds=record)
    with pytest.raises(ValueError, match="handed none"):
        family.chunk_record_argument(None)
    for wrong in (numpy.zeros(24, numpy.int32), numpy.zeros(25, numpy.int64)):
        with pytest.raises(ValueError, match="int32"):
            family.chunk_record_argument(wrong)


def test_the_sub_lattice_table_is_the_one_stepping_reads():
    """B reads the HALF-INTEGER positions and D the INTEGER ones; swapped it is silent."""
    assert family.HALF_INTEGER == {"step_B": True, "step_D": False}
    assert family.HALF_INTEGER["step_B"] is complex_emitter.HALF_INTEGER["step_B"]
    assert family.HALF_INTEGER["step_D"] is complex_emitter.HALF_INTEGER["step_D"]


def test_the_imr_term_table_is_the_one_stepping_applies():
    """Four call sites and only four, with the array path's own targets, partners and
    signs (``stepping._CYLINDRICAL_IMR_B`` / ``_D``), and the kernel reads that partner."""
    assert family.IMR_TERMS == {
        "step_B": ((0, "Bx", +1.0), (2, "Bz", -1.0)),
        "step_D": ((0, "Dx", -1.0), (2, "Dz", +1.0))}
    for sub_step, theirs, terms in (
            ("step_B", stepping._CYLINDRICAL_IMR_B, stepping.B_CURL_TERMS),
            ("step_D", stepping._CYLINDRICAL_IMR_D, stepping.D_CURL_TERMS)):
        mine = family.IMR_TERMS[sub_step]
        assert [(terms.index(term), term.target, sign) for term, _p, sign in theirs] == \
            [tuple(entry) for entry in mine]
        text = source(sub_step)
        for (index, target, _sign), (_term, partner, _s) in zip(mine, theirs):
            block = _target_block(text, target)
            if index == 0:
                assert f"cf f_1 = cf_load({partner}, idx);" in block
                assert "imr_increment(imr0[i], f_1)" in block
            else:
                assert f"imr_increment(imr2[i], cf_load({partner}, idx))" in block


def _executable_text(path) -> str:
    """A module's source with comments and docstrings removed.

    ONE HOME FOR A RULE THIS PACKAGE ALREADY WROTE DOWN
    (``test_triton_kernels.code_of``): the prose in these modules NAMES the things
    they must not touch — that is how a reader learns which track is wired and
    which is not — so an ownership check that greps the RAW file fires on its own
    documentation. Measured 2026-09-02: correcting the four kernel-package
    docstrings that still claimed ``plan_fast_path`` returns None on every branch,
    and naming a gate ARTIFACT PATH in a refusal reason, turned four of these
    checks red without a single executable reference moving. Stripping to
    executable text is what makes the check about behaviour, and it is exactly
    what the companion test below this one has always done.
    """
    import ast as _ast

    from meep_gpu.code_identity import strip_docstrings

    return _ast.unparse(strip_docstrings(_ast.parse(
        pathlib.Path(path).read_text(encoding="utf-8"))))


def test_nothing_in_the_engine_imports_this_package():
    """This FAMILY is not reachable from the seam, which is narrower than it was.

    ``fastpath`` composes the hand-CUDA table as its second kernel table since Phase
    2, so "the engine never names ``cuda_kernels``" stopped being true — and the
    clause that replaced it is about THIS module rather than about the package: the
    seam reaches the table through ``cuda_kernels.arms``, which selects a family by
    predicate, and no file in the engine may name a family module directly. A
    dispatcher that named one would be choosing arithmetic rather than asking the
    table for it.

    ``driver.py`` and ``stepping.py`` keep the stricter rule unchanged: the driver
    consults a plan and never a composer, and ``stepping.py`` is the ORACLE every
    gate compares against.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("fastpath.py", "driver.py", "stepping.py"):
        code = _executable_text(os.path.join(root, name))
        assert "cylindrical_complex_kernels" not in code, name
    for name in ("driver.py", "stepping.py"):
        code = _executable_text(os.path.join(root, name))
        assert "cuda_kernels" not in code, name
