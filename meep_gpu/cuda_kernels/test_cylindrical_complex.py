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
  one, four of them ARITHMETIC (the prefix substitution, the i*m/r term and its
  ordering against the mask, the |m| = 1 curl-row replacement, the per-|m| axis
  rules). The gate can see a wrong answer; only a text pin can see a grouping change
  that is exact at one Courant and wrong at another, or an ordering that happens to be
  invisible on every case the sweep reaches.
* THE PRELUDE IS IMPORTED, NOT COPIED. The certified complex strings are this family's
  whole non-cylindrical body; a second copy would drift and would silently un-arm the
  shared mutation battery, whose needles are those exact characters.
* THE HOST-SIDE CONSTANTS. ``imr_coefficient_row`` and ``axis_increment_scalars`` are
  re-derivations of two array-path expressions whose float32 words are load-bearing
  down to the sign of a zero. They are checked against ``stepping``'s own values by
  CALLING both, not by reading either.
* THE ARGUMENT ORDER. The kernel signature and the launcher's tuple are two lists that
  must agree; ``RawKernel`` does not check, so a swapped pair is a wrong answer at a
  wrong pointer.

THE SOURCE IS IMPORTED, NOT READ WITH ``ast``, and that is why
``cylindrical_complex_kernels`` imports CuPy LAZILY: its device text is a FUNCTION of
the expansion arm, so there is no module-level string for ``ast`` to read, and the
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
ARMS = ("FMA_V1", "NAIVE")

#: A licence verdict shaped exactly as ``complex_fields.expansion_license`` returns
#: one, with the fields ``coverage.complex_expansion_refusal`` reads. Built here
#: rather than loaded from a probe artifact so this file needs no results directory;
#: the SHAPE is pinned against the real arbiter by
#: :func:`test_the_licence_stub_is_the_shape_the_arbiter_returns`.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "refusals": [], "basis": "measured",
           "policy": "ieee_keep_ftz_stripped", "policy_resolved": "keep"}
POLICY = "keep"


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
        fields, layer, grid, sub_step, license=LICENCE, subnormal_policy=POLICY)


def source(sub_step="step_B", arm="FMA_V1"):
    return family.cylindrical_complex_source(sub_step, arm)


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
        fields, None, grid, "step_B", license=LICENCE, subnormal_policy=POLICY)
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


def test_a_licence_cut_under_the_other_policy_is_refused():
    """The arm is compiled in at this seam and is POLICY-CONDITIONAL."""
    fields, layer, grid = build(m=1)
    ok, why = cylindrical_coverage.covers_pml_cylindrical_complex_curl(
        fields, layer, grid, "step_B", license=LICENCE, subnormal_policy="flush")
    assert not ok and "does not transfer across that boundary" in why


def test_no_licence_at_all_is_refused_before_anything_else():
    fields, layer, grid = build(m=1)
    ok, why = cylindrical_coverage.covers_pml_cylindrical_complex_curl(
        fields, layer, grid, "step_B")
    assert not ok and "no expansion licence" in why


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
    assert "complex_expansion_refusal" in cylindrical_coverage.SHARED_HELPERS


def test_the_licence_stub_is_the_shape_the_arbiter_returns():
    """The stub above must carry every key the refusal function reads.

    Pinned against the REAL arbiter's key set rather than against a list written
    here, so a licence that grows a required field fails at the merge bar.
    """
    from ..triton_kernels import complex_fields  # noqa: PLC0415

    verdict = complex_fields.expansion_license({
        "measurements": [], "policy": "ieee_keep_ftz_stripped"})
    for key in ("arm", "expansion", "refusals", "basis"):
        assert key in verdict, key
        assert key in LICENCE, key
    assert coverage.complex_expansion_refusal(LICENCE, POLICY) is None


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
        fields, layer, grid, sub_step, license=LICENCE, subnormal_policy=POLICY)
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
        fields, layer, grid, sub_step, license=LICENCE, subnormal_policy=POLICY)
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


@pytest.mark.parametrize("dtdx", [0.5, 0.35, 0.2, 0.123456789])
def test_the_axis_coefficient_is_nep50s_one_rounding_of_the_python_float(dtdx):
    """``Dz[axis] += (4.0*(dt/dx)) * Hy[axis]`` casts the weak scalar ONCE.

    The multiplicand the array path forms is a float64 ``4.0 * dtdx`` that NEP-50
    converts to the array's complex64 -- one rounding of the real word, +0.0 for the
    imaginary -- before the full complex product. The host helper returns exactly
    that word, and the product it feeds is the array path's, measured on random
    complex64 operands as uint32 words.
    """
    rng = numpy.random.default_rng(int(dtdx * 1e6))
    hy = (rng.uniform(-1, 1, 64) + 1j * rng.uniform(-1, 1, 64)).astype(numpy.complex64)
    coefficient = family.axis_coefficient(dtdx)
    assert coefficient.dtype == numpy.float32
    assert coefficient == numpy.float32(4.0 * float(dtdx))
    array_path = (4.0 * float(dtdx)) * hy
    rounded_first = numpy.complex64(complex(float(coefficient), 0.0)) * hy
    assert array_path.dtype == numpy.complex64
    assert (array_path.view(numpy.uint32) == rounded_first.view(numpy.uint32)).all()


@pytest.mark.parametrize("m", [1, -1, 2, 3, 5])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_imr_rows_are_word_for_word_the_array_paths(m, sub_step):
    """Built by CALLING ``stepping._cylindrical_imr_term``, never by re-reading it.

    The array path forms the row, multiplies it by a partner volume and NEGATES the
    product; the row itself is recovered from that by dividing out a partner of
    exactly one, which is exact in float32 for these words.
    """
    fields, _layer, grid = build(shape=(9, 1, 4), m=m)
    rows = grid.shape[0]
    dtdx = float(grid.dt / grid.dx)
    ones = numpy.ones(tuple(grid.shape), dtype=numpy.complex64)
    for index, target, sign in family.IMR_TERMS[sub_step]:
        mine = family.imr_coefficient_row(numpy, target, sign, m, dtdx, rows,
                                          numpy.complex64)
        theirs = -stepping._cylindrical_imr_term(fields, target, ones, sign)
        assert theirs.shape == tuple(grid.shape)
        column = numpy.ascontiguousarray(theirs[:, 0, 0])
        assert numpy.array_equal(
            mine.view(numpy.float32).view(numpy.uint32),
            column.view(numpy.float32).view(numpy.uint32)), (target, sign)
        # The Yee shift is the TARGET's own; using 0 for every target is a defect
        # the gate arms, and this is the fact that makes it one.
        assert index in (0, 2)
        assert IYEE_SHIFTS[target][0] in (0, 1)


@pytest.mark.parametrize("m", [1, -1])
@pytest.mark.parametrize("dtdx", [0.35, 0.5, 0.37])
def test_the_imr_row_real_word_is_exactly_positive_zero(m, dtdx):
    """``(-1j) * X`` launders the real word to +0.0 for BOTH signs of X.

    Measured rather than asserted from the spelling: the arm choice at this call site
    is degenerate BECAUSE of it, and the module docstring says so.
    """
    for target, sign in (("Bx", +1.0), ("Bz", -1.0), ("Dx", -1.0), ("Dz", +1.0)):
        row = family.imr_coefficient_row(numpy, target, sign, m, dtdx, 6,
                                         numpy.complex64)
        words = numpy.ascontiguousarray(row.real).view(numpy.uint32)
        assert set(words.tolist()) == {0}, (target, sign, m, dtdx)


@pytest.mark.parametrize("m", [1, -1])
@pytest.mark.parametrize("dtdx", [0.35, 0.5])
def test_the_axis_increment_scalars_are_the_array_paths(m, dtdx):
    """And the second one's real word is a SIGNED ZERO for m < 0 -- measured."""
    minus_dtdx, (re, im) = family.axis_increment_scalars(m, dtdx)
    assert numpy.float32(minus_dtdx) == numpy.float32(-dtdx)
    expected = numpy.complex64(1j * (m * dtdx))
    assert numpy.float32(re).view(numpy.uint32) == \
        numpy.float32(expected.real).view(numpy.uint32)
    assert numpy.float32(im).view(numpy.uint32) == \
        numpy.float32(expected.imag).view(numpy.uint32)
    sign_bit = int(numpy.float32(re).view(numpy.uint32)) >> 31
    assert sign_bit == (1 if m < 0 else 0)


def test_the_axis_increment_row_matches_steppings_own(monkeypatch):
    """The whole |m| = 1 B-side increment, from ``stepping``, on a seeded grid.

    Not a re-derivation: ``_cylindrical_axis_increment_B`` is CALLED and the two
    scalars this module binds are used to reassemble the same row in float32.
    """
    fields, layer, grid = build(shape=(8, 1, 6), m=-1)
    rng = numpy.random.default_rng(4)
    for name in ("Ex", "Ey", "Ez"):
        volume = getattr(fields, name)
        volume.real = rng.uniform(-1, 1, volume.shape).astype(numpy.float32)
        volume.imag = rng.uniform(-1, 1, volume.shape).astype(numpy.float32)
    electric = {n: getattr(fields, n) for n in ("Ex", "Ey", "Ez")}
    boundaries = stepping._boundary_kinds(grid, layer)
    phases = (None, None, None)
    dtdx = float(grid.dt / grid.dx)
    target, theirs = stepping._cylindrical_axis_increment_B(
        fields, electric, boundaries, phases, dtdx)
    assert target == "Bx"
    assert family.AXIS_INCREMENT_TARGET["step_B"] == 0

    minus_dtdx, (inc_re, inc_im) = family.axis_increment_scalars(-1, dtdx)
    ep = electric["Ey"]
    ep_above = stepping._shift_up(numpy, ep, 2, boundaries[2], None)
    first = numpy.complex64(minus_dtdx) * (ep[0] - ep_above[0])
    second = numpy.complex64(complex(inc_re, inc_im)) * numpy.take(electric["Ez"], 1, 0)
    mine = (first - second).astype(numpy.complex64)
    assert numpy.array_equal(
        numpy.ascontiguousarray(mine).view(numpy.float32).view(numpy.uint32),
        numpy.ascontiguousarray(theirs).view(numpy.float32).view(numpy.uint32))


# --------------------------------------------------------------------------
# The device text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_emitted_source_is_pure_ascii_and_declares_one_kernel(sub_step, arm):
    """ASCII is a COMPILE requirement: NVRTC writes the source through a locale open."""
    text = source(sub_step, arm)
    text.encode("ascii")
    assert family.shipped_kernel_names(text) == {family.KERNELS[sub_step][0]}


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_two_arms_differ_and_only_in_the_arm_block(sub_step):
    """A wrong arm is a wrong answer, so the arms must be different strings."""
    fma, naive = source(sub_step, "FMA_V1"), source(sub_step, "NAIVE")
    assert fma != naive
    assert "__fmaf_rn" in fma and "__fmaf_rn" not in naive
    # Everything outside the arm block is identical, which is what "the arm is this
    # emitter's ONLY parameter" means.
    for text in (fma, naive):
        assert complex_emitter._HEAD in text
        assert complex_emitter._TAIL in text
        assert family._TEMPLATES[sub_step] in text


@pytest.mark.parametrize("arm", ARMS)
def test_the_certified_strings_are_imported_not_copied(arm):
    """The non-cylindrical body must be the certified characters, not a second copy."""
    for sub_step in SUB_STEPS:
        text = source(sub_step, arm)
        assert complex_emitter._ARM_SOURCE[complex_emitter.EXPANSIONS[arm]] in text
    # And the module must not carry its own copy of any of the three.
    module_text = open(family.__file__, encoding="utf-8").read()
    for helper in ("cf_load", "cshift_up", "cshift_dn", "pml_apply",
                   "mul_field_left", "mul_coefficient_left", "rotate_field_left"):
        assert f"__device__ __forceinline__ cf {helper}" not in module_text, helper
        assert f"__device__ __forceinline__ void {helper}" not in module_text, helper


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_no_floating_point_division_anywhere(sub_step):
    """The i*m/r row is bound as an array PRECISELY so the kernel performs none."""
    text = source(sub_step)
    body = text[text.index('extern "C"'):]
    stripped = re.sub(r"//[^\n]*", "", body)
    assert "/" not in stripped.replace("/ (ny * nz)", "").replace(
        "idx / nz", "").replace("idx / (ny * nz)", ""), stripped


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_prefix_substitution_is_the_shipped_grouping(sub_step):
    """Bz is ONE subtract and ONE multiply; Dz swaps only its ``first`` source."""
    text = source(sub_step)
    if sub_step == "step_B":
        assert "cf pfx_up = cf_load(pfx, idx + sx);" in text
        assert ("cf curl = mul_coefficient_left(dtdx, cf_sub(pfx_up, pfx_here));"
                in text)
        # NOT the distributed form, which is a different float32 number per plane.
        assert "mul_coefficient_left(dtdx, pfx_up)" not in text
    else:
        assert "cf f_1 = cf_load(pfx, idx);" in text
        assert "cf sf = cshift_dn(pfx, idx, i, nx, sx, bc_x, ph_x, noph);" in text
        # Dx's Hy operands stay RAW -- the prefix reaches Dz and nothing else.
        assert text.count("cf_load(pfx,") == 1
        assert "cf f_2 = cf_load(Hy, idx);" in text


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_imr_term_is_on_targets_0_and_2_only_and_before_the_mask(sub_step):
    """Ordering is the array path's: after the dtdx curl, BEFORE the ownership mask."""
    text = source(sub_step)
    assert text.count("mul_complex_left(cf_load(imr0, i)") == 1
    assert text.count("mul_complex_left(cf_load(imr2, i)") == 1
    for row in ("imr0", "imr2"):
        add = text.index(f"curl = cf_sub(curl, mul_complex_left(cf_load({row}, i)")
        after = text[add:]
        mask = after.index("== BC_METALLIC")
        assert mask > 0, row
        # No mask may precede the add inside the same target block.
        block_start = text.rindex("    {", 0, add)
        assert "BC_METALLIC" not in text[block_start:add], row


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_axis_increment_replaces_the_row_and_negates_per_word(sub_step):
    """REPLACE, never accumulate: the two differ only on a stored signed zero."""
    text = source(sub_step)
    assert "if (m_class == 1 && i == 0) {" in text
    assert "curl.re = inc.re * -1.0f;" in text
    assert "curl.im = inc.im * -1.0f;" in text
    # Accumulation is the defect; it must not be the spelling.
    assert "curl.re = curl.re +" not in text
    if sub_step == "step_B":
        assert "cf e1 = cf_load(Ez, sx + j * sy + k);" in text
        assert "mul_coefficient_left(minus_dtdx, cf_sub(f_2, ss))" in text
        assert "mul_complex_left(q, e1)" in text
    else:
        assert "cf two_c = mul_coefficient_left(2.0f, f_2);" in text
        assert "cf s = cf_sub(cf_sub(f_1, sf), two_c);" in text


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
    for row in ("imr0", "imr2"):
        guard = text.index(f"curl = cf_sub(curl, mul_complex_left(cf_load({row}, i)")
        assert text.rindex("if (m_class != 0) {", 0, guard) > text.rindex("    {", 0, guard)
    assert text.count("if (m_class == 0 && i == 0) {") == 1
    tail = text[text.index("if (m_class == 0 && i == 0) {"):]
    tail = tail[:tail.index("\n    }")]
    if sub_step == "step_B":
        assert "cf_store(Bx, idx, cf_zero());" in tail
        assert "fu_Bx" not in tail and "By" not in tail and "Bz" not in tail
        assert "axis_coef" not in text
    else:
        assert ("cf_store(Dz, idx, cf_add(cf_load(Dz, idx),\n"
                "                                 mul_coefficient_left(axis_coef, "
                "cf_load(Hy, idx))));") in tail
        assert tail.index("cf_store(Dz") < tail.index("cf_store(Dy, idx, cf_zero());")
        assert "fu_D" not in tail and "pfx" not in tail
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


def test_the_arm_is_required_and_never_defaulted():
    with pytest.raises(TypeError):
        family.cylindrical_complex_source("step_B")  # noqa: PLC0304 - the point
    with pytest.raises(ValueError):
        family.cylindrical_complex_source("step_B", "SOMETHING_ELSE")
    with pytest.raises(ValueError):
        family.cylindrical_complex_source("update_H", "FMA_V1")


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
    # THE m = 0 ON-AXIS Dz MULTIPLICAND rides on the D side only (2026-09-04): the
    # B side's m = 0 rule is a zero store and binds no scalar.
    expected = (list(targets) + ["fu_" + n for n in targets] + list(sources)
                + ["pfx", "imr0", "imr2",
                   "nx", "ny", "nz", "dtdx", "minus_dtdx", "inc_re", "inc_im"]
                + (["axis_coef"] if sub_step == "step_D" else [])
                + [f"{stem}_{axis}" for axis in "xyz" for stem in ("kms", "sinv")]
                + ["bc_x", "bc_y", "bc_z", "ph_x", "ph_y", "ph_z",
                   "m_class", "zero_rows"])
    assert [name for _kind, name in parameters(sub_step)] == expected


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_every_pointer_parameter_is_a_float_word_pointer(sub_step):
    """complex64 reaches the kernel as float32 WORDS; a complex type here is a lie."""
    for kind, name in parameters(sub_step):
        if "*" in kind:
            assert kind.replace("const", "").strip().startswith("float"), (kind, name)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_branch_axes_are_runtime_ints_not_compiled_in(sub_step):
    """The boundary codes, the phase flags, the m class and ZERO_ROWS are branches."""
    kinds = dict((name, kind) for kind, name in parameters(sub_step))
    for name in ("bc_x", "bc_y", "bc_z", "ph_x", "ph_y", "ph_z",
                 "m_class", "zero_rows", "nx", "ny", "nz"):
        assert kinds[name] == "int", name


def test_the_launcher_refuses_a_gridless_call_that_omits_a_derived_quantity():
    """A defaulted m class or increment would be a silently different sub-step."""
    fields, _layer, grid = build(m=1)
    with pytest.raises(ValueError, match="exactly one of grid"):
        family.step_cylindrical_complex(
            "step_B", fields, "FMA_V1", grid=grid, pml=None, tables={},
            boundary_codes=(1, 0, 1))
    with pytest.raises(ValueError, match="without a grid the caller must supply"):
        family.step_cylindrical_complex(
            "step_B", fields, "FMA_V1", tables={"kms_x": None},
            boundary_codes=(1, 0, 1))


def test_the_sub_lattice_table_is_the_one_stepping_reads():
    """B reads the HALF-INTEGER positions and D the INTEGER ones; swapped it is silent."""
    assert family.HALF_INTEGER == {"step_B": True, "step_D": False}
    assert family.HALF_INTEGER["step_B"] is complex_emitter.HALF_INTEGER["step_B"]
    assert family.HALF_INTEGER["step_D"] is complex_emitter.HALF_INTEGER["step_D"]


def test_the_imr_term_table_is_the_one_stepping_applies():
    """Four call sites and only four, with the array path's own signs."""
    assert family.IMR_TERMS == {
        "step_B": ((0, "Bx", +1.0), (2, "Bz", -1.0)),
        "step_D": ((0, "Dx", -1.0), (2, "Dz", +1.0))}
    source_text = open(stepping.__file__, encoding="utf-8").read()
    for sub_step, terms in family.IMR_TERMS.items():
        for _index, target, sign in terms:
            spelling = f'_cylindrical_imr_term(fields, "{target}"'
            assert spelling in source_text, target
            call = source_text[source_text.index(spelling):]
            assert call[:200].split(")")[0].endswith(f"{sign:+.1f}"), (target, sign)


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
