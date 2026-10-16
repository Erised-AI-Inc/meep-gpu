"""Behaviour pins for the off-diagonal constitutive row as MEEP's C evaluates it.

MEEP's ``step_update_EDHB`` (step_generic.cpp:580-581, :611 with PML, :719 without)::

    OFFDIAG(u, g, sx) = 0.25 * ((g[i] + g[i - sx]) * u[i] + (g[i + s] + g[(i + s) - sx]) * u[i + s])
    f[i] = gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)

The pair sums, the products and their sum are ``realnum`` (float32) operations; ``0.25``
is a double literal, so each OFFDIAG is a double, the row is summed left to right in
double and the store rounds it once. A term whose chunk deleted its ``chi1inv`` array
(anisotropic_averaging.cpp:245-292) is absent there, not ``+0``. What is pinned, and
against what:

* the row, word for word, against a scalar transcription of that C (one cell at a time,
  NumPy float32 scalars for the ``realnum`` operations, Python floats for the doubles),
  on real and complex storage — and the float32 form the engine used before, which must
  differ on these operands (the discrimination control);
* per-chunk presence: on a two-chunk division whose coefficient is zero over every sample
  of one chunk, a ``-0`` row stays ``-0`` in that chunk's cells and becomes ``+0`` where
  the array is kept; one chunk keeps it everywhere;
* :meth:`~.chunk_map.ChunkLayout.chunk_samples`, :meth:`~.chunk_map.ChunkLayout.offdiag_kept`
  and :meth:`~.chunk_map.ChunkLayout.offdiag_presence` on hand-built divisions, including
  a wrapping axis's image row and ``-0.0`` counting as zero;
* the cylindrical axis row takes no off-diagonal term (update_eh.cpp:197-208);
* ``chi1inv`` is the reciprocal of the caller's float64 permittivity rounded once
  (``FdtdDriver.set_epsilon`` and ``set_epsilon_components``, grid shape and MEEP's
  padded shape), not ``f32(1/f32(eps))``;
* a nonlinear component multiplies the double row by its Pade factor in double and rounds
  once (MEEP's MOST GENERAL CASE, step_generic.cpp:590-601). The factor itself is not
  pinned here.

The word-level MEEP comparison is the AP6e acceptance (8 lifted cases x 3 legs against
contraction-off MEEP 1.33.0 single precision), which needs the reference build; these
pins need nothing but NumPy.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from . import stepping
from .chunk_map import ChunkLayout, grid_frames
from .driver import FdtdDriver
from .fields import Fields
from .grid import Grid

NAMES = ("Ex", "Ey", "Ez")
AXES = ("x", "y", "z")
NEG_ZERO = np.uint32(0x80000000)


def _words(values):
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _driver(complex_fields=False, cell=(0.6, 0.5, 0.4)):
    return FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=complex_fields)


def _random_tensor(driver, rng, partners_per_row=2):
    """Per-row diagonal permittivity (float32-exact), random off-diagonal rows, random D."""
    shape = driver.shape
    diagonal = {name: np.full(shape, value, dtype=np.float32)
                for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
    rows = {}
    for row in NAMES:
        partners = [p for p in NAMES if p != row][:partners_per_row]
        rows[row] = {p: rng.uniform(-0.3, 0.3, size=shape).astype(np.float32) for p in partners}
    driver.set_epsilon_components(diagonal, chi1inv_offdiagonal=rows)
    return rows


def _fill_d(driver, rng, complex_fields):
    for axis in AXES:
        # A wide dynamic range, so a float32 running sum and a double one part company.
        magnitude = 10.0 ** rng.uniform(-3, 3, size=driver.shape)
        pattern = rng.standard_normal(driver.shape) * magnitude
        if complex_fields:
            pattern = pattern + 1j * rng.standard_normal(driver.shape) * magnitude
            getattr(driver.fields, "D" + axis)[...] = pattern.astype(np.complex64)
        else:
            getattr(driver.fields, "D" + axis)[...] = pattern.astype(np.float32)


def _transcribed_row(g, u_diag, rows, component, double_row=True):
    """MEEP's row for one real part, one cell at a time (periodic k = 0, plain modulo wraps).

    ``g`` maps axis name -> float32 volume. ``double_row=False`` is the float32 form the
    engine used before (each term rounded to float32 and added in float32): the control.
    """
    own = AXES.index(component[1])
    shape = g["x"].shape
    out = np.empty(shape, dtype=np.float32)
    f32 = np.float32
    for i, j, k in np.ndindex(shape):
        def at(array, d):
            return array[(i + d[0]) % shape[0], (j + d[1]) % shape[1], (k + d[2]) % shape[2]]
        gs_us = f32(g[component[1]][i, j, k]) * f32(u_diag[i, j, k])  # realnum * realnum
        accumulator = float(gs_us) if double_row else gs_us
        for offset in (1, 2):  # cycle_direction order
            partner_axis = (own + offset) % 3
            u_row = rows.get("E" + AXES[partner_axis])
            if u_row is None:
                continue
            gp = g[AXES[partner_axis]]
            down = [0, 0, 0]
            down[partner_axis] = -1
            up = [0, 0, 0]
            up[own] = 1
            both = [up[n] + down[n] for n in range(3)]
            low = f32(at(gp, (0, 0, 0)) + at(gp, down))       # g[i] + g[i - sx]
            high = f32(at(gp, up) + at(gp, both))             # g[i + s] + g[(i + s) - sx]
            summed = f32(f32(low * at(u_row, (0, 0, 0))) + f32(high * at(u_row, up)))
            if double_row:
                accumulator = accumulator + 0.25 * float(summed)  # the double literal
            else:
                accumulator = f32(accumulator + f32(0.25 * summed))
        out[i, j, k] = f32(accumulator)  # the store rounds once
    return out


@pytest.mark.parametrize("complex_fields", [False, True], ids=["real", "complex"])
def test_the_row_is_summed_in_double_and_rounded_once(complex_fields):
    """Every stored E word is MEEP's: float32 pairs and products, a double row, one rounding."""
    rng = np.random.default_rng(20261007)
    driver = _driver(complex_fields)
    rows = _random_tensor(driver, rng)
    _fill_d(driver, rng, complex_fields)
    fields = driver.fields
    d = {axis: np.array(getattr(fields, "D" + axis), copy=True) for axis in AXES}
    stepping.update_E(fields, None)
    parts = (("real", np.real), ("imag", np.imag)) if complex_fields else (("real", lambda a: a),)
    differ_from_float32 = 0
    for component in NAMES:
        stored = np.asarray(fields.get_E(component))
        u = np.asarray(fields.inverse_epsilon_for(component))
        for label, take in parts:
            g = {axis: np.ascontiguousarray(take(d[axis]), dtype=np.float32) for axis in AXES}
            expected = _transcribed_row(g, u, rows[component], component)
            got = np.ascontiguousarray(take(stored), dtype=np.float32)
            mismatched = int(np.count_nonzero(_words(got) != _words(expected)))
            assert mismatched == 0, (
                f"{component} {label}: {mismatched} of {expected.size} words differ from the "
                f"transcribed double row")
            control = _transcribed_row(g, u, rows[component], component, double_row=False)
            differ_from_float32 += int(np.count_nonzero(_words(control) != _words(expected)))
    assert differ_from_float32 > 0, "the operands do not separate the double row from the float32 one"
    driver.close()


def test_a_lone_partner_takes_the_first_slot_and_the_row_is_still_double():
    """One off-diagonal entry per row (MEEP's 2x2 branch after its SWAP, step_generic.cpp:572-576)."""
    rng = np.random.default_rng(11)
    driver = _driver()
    rows = _random_tensor(driver, rng, partners_per_row=1)
    _fill_d(driver, rng, False)
    fields = driver.fields
    d = {axis: np.array(getattr(fields, "D" + axis), copy=True) for axis in AXES}
    stepping.update_E(fields, None)
    for component in NAMES:
        expected = _transcribed_row(d, np.asarray(fields.inverse_epsilon_for(component)),
                                    rows[component], component)
        assert np.array_equal(_words(fields.get_E(component)), _words(expected)), component
    driver.close()


def test_the_terms_are_added_in_cycle_direction_order():
    """``gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)``: the y partner of Ex first, then z.

    Uniform operands make each term exact: Dx = 1 with chi1inv_xx = 1, the y term
    ``Dy * u_xy = 2^-30 * 2^-30 = 2^-60`` and the z term ``Dz * u_xz = -1``. In MEEP's
    order the double row is ``(1 + 2^-60) - 1 = 0`` (the tiny term is absorbed first);
    the other order keeps ``2^-60``. Random operands almost never separate the two.
    """
    driver = _driver()
    shape = driver.shape
    tiny = np.float32(2.0 ** -30)
    driver.set_epsilon_components(
        {n: np.ones(shape, dtype=np.float32) for n in NAMES},
        chi1inv_offdiagonal={"Ex": {"Ey": np.full(shape, tiny), "Ez": np.full(shape, -1.0, dtype=np.float32)}})
    fields = driver.fields
    fields.Dx[...] = np.float32(1.0)
    fields.Dy[...] = tiny
    fields.Dz[...] = np.float32(1.0)
    stepping.update_E(fields, None)
    assert bool(np.all(_words(fields.get_E("Ex")) == 0)), "the z term was added before the y term"
    driver.close()


# --- per-chunk presence -----------------------------------------------------------------


def _two_chunks_along_x(grid):
    """Two chunks splitting x at the window's middle, y and z whole; no PML flags."""
    frames = grid_frames(grid)
    lo = [frame.window[0] if not frame.invariant else 0 for frame in frames]
    hi = [frame.window[1] if not frame.invariant else 0 for frame in frames]
    mid = lo[0] + 2 * (frames[0].owned // 2)
    boxes = [((lo[0], lo[1], lo[2]), (mid, hi[1], hi[2])),
             ((mid, lo[1], lo[2]), (hi[0], hi[1], hi[2]))]
    return ChunkLayout(boxes, [(False, False, False)] * 2, frames), mid


def test_a_term_whose_chunk_dropped_its_array_is_absent_not_plus_zero():
    """A ``-0`` row stays ``-0`` where MEEP's chunk deleted the array and is ``+0`` where it kept it.

    Ex's only off-diagonal entry (partner Ey) is nonzero on x rows strictly inside chunk 1,
    so chunk 0 — which allocates x rows 0 .. mid/2 and, on this wrapping axis, the image
    row 0 — deletes it. Dx = -0 everywhere and Dy = +0, so ``gs*us = -0`` and the term is
    ``0.25 * ((0 + 0) * u + (0 + 0) * u) = +0``. Adding it gives ``+0``; MEEP adds nothing in
    chunk 0. The default one-chunk division keeps the array everywhere: all ``+0``.
    """
    def run(split):
        driver = _driver()
        shape = driver.shape
        layout, mid = _two_chunks_along_x(driver.grid)
        frame = layout.frames[0]
        split_row = (mid - frame.origin) // 2
        coefficient = np.zeros(shape, dtype=np.float32)
        coefficient[split_row + 1:, :, :] = 0.125
        driver.set_epsilon_components({n: np.full(shape, 2.0, dtype=np.float32) for n in NAMES},
                                      chi1inv_offdiagonal={"Ex": {"Ey": coefficient}})
        if split:
            driver.set_chunk_layout(layout)
            assert layout.offdiag_kept(coefficient) == (False, True)
        fields = driver.fields
        fields.Dx[...] = np.float32(-0.0)
        fields.Dy[...] = np.float32(0.0)
        fields.Dz[...] = np.float32(0.0)
        stepping.update_E(fields, None)
        words = _words(fields.get_E("Ex")).copy()
        owner = layout.chunk_of([frame.doubled(True).reshape(-1, 1, 1),
                                 np.zeros((1, 1, 1), dtype=int), np.zeros((1, 1, 1), dtype=int)])
        driver.close()
        return words, owner.reshape(-1)

    words, owner = run(split=True)
    in_chunk0 = np.broadcast_to((owner == 0).reshape(-1, 1, 1), words.shape)
    assert in_chunk0.any() and (~in_chunk0).any()
    assert bool(np.all(words[in_chunk0] == NEG_ZERO)), "the dropped term was added in chunk 0"
    assert bool(np.all(words[~in_chunk0] == 0)), "the kept term was not added in chunk 1"
    one_chunk, _ = run(split=False)
    assert bool(np.all(one_chunk == 0)), "one chunk keeps the array, so every cell adds the +0 term"


def test_chunk_samples_and_kept_flags_follow_the_chunk_boxes():
    """A chunk allocates its owned rows plus the one past them; ``-0.0`` is zero; None when all keep."""
    driver = _driver()
    layout, mid = _two_chunks_along_x(driver.grid)
    frame = layout.frames[0]
    n = frame.stored
    split_row = (mid - frame.origin) // 2
    low, high = layout.chunk_samples(0)
    # Integer-lattice doubled coordinate origin + 2i: chunk 0 holds [lo, mid], chunk 1 [mid, hi],
    # and on this wrapping axis row 0 (the low wall's image of the high one) belongs to both.
    assert np.flatnonzero(low).tolist() == list(range(0, split_row + 1))
    assert np.flatnonzero(high).tolist() == [0] + list(range(split_row, n))
    for axis in (1, 2):
        assert all(bool(line.all()) for line in layout.chunk_samples(axis))
    shape = driver.shape
    coefficient = np.zeros(shape, dtype=np.float32)
    coefficient[split_row, 0, 0] = -0.0
    assert layout.offdiag_kept(coefficient) == (False, False), "-0.0 is MEEP's trivial value"
    coefficient[split_row, 0, 0] = 1.0  # The shared boundary row: both chunks allocate it.
    assert layout.offdiag_kept(coefficient) == (True, True)
    assert layout.offdiag_presence(coefficient, (1, 0, 0)) is None
    coefficient[...] = 0.0
    coefficient[0, 1, 1] = 1.0  # The wrapped image row: both chunks allocate it too.
    assert layout.offdiag_kept(coefficient) == (True, True)
    coefficient[...] = 0.0
    coefficient[1, 2, 2] = 1.0  # Strictly inside chunk 0.
    assert layout.offdiag_kept(coefficient) == (True, False)
    presence = layout.offdiag_presence(coefficient, (1, 0, 0))
    owner = layout.chunk_of([frame.doubled(True).reshape(-1, 1, 1),
                             np.zeros((1, 1, 1), dtype=int), np.zeros((1, 1, 1), dtype=int)]).reshape(-1)
    assert np.array_equal(presence, np.broadcast_to((owner == 0).reshape(-1, 1, 1), shape))
    driver.close()


def test_the_cylindrical_axis_row_takes_no_offdiagonal_term():
    """``update_eh`` re-steps the r = 0 row with every off-diagonal NULL (update_eh.cpp:197-208)."""
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.0, 0.4), cylindrical=True)
    fields = Fields(grid)
    shape = grid.shape
    ones = np.ones(shape, dtype=np.float32)
    fields.set_epsilon_volumes({n: ones for n in NAMES}, {n: ones for n in NAMES},
                               chi1inv_offdiagonal={"Ez": {"Ex": 0.1 * ones}, "Ex": {"Ez": 0.1 * ones}})
    assert int(grid.origin_doubled(0)) == 0
    on_axis = fields.offdiag_presence("Ez", "Ex", None)  # Ez: r-shift 0, row 0 is r = 0.
    assert on_axis is not None
    assert not bool(np.asarray(on_axis)[0].any()) and bool(np.asarray(on_axis)[1:].all())
    assert fields.offdiag_presence("Ex", "Ez", None) is None  # Er sits at r = dr/2.


# --- chi1inv from the caller's permittivity -----------------------------------------------


def _inexact_reciprocals(rng, shape):
    """float32 chi1inv words whose float64 reciprocal does not survive a float32 round trip."""
    words = rng.uniform(0.05, 1.0, size=shape).astype(np.float32)
    epsilon = 1.0 / words.astype(np.float64)
    old = (1.0 / epsilon.astype(np.float32)).astype(np.float32)
    return words, epsilon, int(np.count_nonzero(_words(old) != _words(words)))


@pytest.mark.parametrize("padded", [False, True], ids=["grid_shape", "meep_padded_shape"])
def test_chi1inv_is_the_callers_reciprocal_rounded_once(padded):
    rng = np.random.default_rng(5)
    driver = _driver()
    shape = tuple(n + 1 for n in driver.shape) if padded else driver.shape
    words, epsilon, moved_by_round_trip = _inexact_reciprocals(rng, shape)
    assert moved_by_round_trip > 0, "the operands do not separate the two forms"
    trimmed = words[:-1, :-1, :-1] if padded else words
    driver.set_epsilon_components({n: epsilon for n in NAMES})
    for name in NAMES:
        assert np.array_equal(_words(driver.fields.inverse_epsilon_for(name)), _words(trimmed)), name
        assert np.asarray(driver.fields.inverse_epsilon_for(name)).flags["C_CONTIGUOUS"]
    driver.set_epsilon(epsilon)
    for name in NAMES:
        assert np.array_equal(_words(driver.fields.inverse_epsilon_for(name)), _words(trimmed)), name
    assert np.array_equal(np.asarray(driver.fields.eps), (epsilon[:-1, :-1, :-1] if padded else epsilon)
                          .astype(np.float32))
    driver.close()


def test_a_float32_permittivity_keeps_its_float32_reciprocal():
    """Double rounding of a float32 quotient is exact (53 >= 2*24 + 2), so float32 input is unchanged."""
    rng = np.random.default_rng(6)
    driver = _driver()
    epsilon = rng.uniform(1.0, 13.0, size=driver.shape).astype(np.float32)
    driver.set_epsilon(epsilon)
    assert np.array_equal(_words(driver.fields.inverse_epsilon_for("Ez")),
                          _words((1.0 / epsilon).astype(np.float32)))
    driver.close()


# --- nonlinear ----------------------------------------------------------------------------


def test_a_nonlinear_component_scales_the_double_row_and_rounds_once():
    """``fw = (gs*us + OFFDIAG + OFFDIAG) * u``: the double row times the factor, one rounding.

    The Pade factor ``u`` is the engine's own (:func:`stepping.calc_nonlinear_u`); this
    pins only that the row is not rounded before the multiply.
    """
    rng = np.random.default_rng(3)
    driver = _driver()
    _random_tensor(driver, rng)
    driver.set_chi3(1e-3)
    _fill_d(driver, rng, False)
    for axis in AXES:
        getattr(driver.fields, "D" + axis)[...] *= np.float32(1e-3)
    fields = driver.fields
    displacement = stepping._nonlinear_displacement(fields, None)
    for component in NAMES:
        row = stepping._offdiagonal_terms(fields, component, displacement)
        gs = displacement["volumes"][component]
        us = fields.inverse_epsilon_for(component)
        transverse = stepping._nonlinear_transverse_sums(fields, component, displacement)
        factor = stepping.calc_nonlinear_u(stepping._nonlinear_dsqr(gs, transverse, stepping._WHOLE),
                                           gs, us, fields.chi2_for(component), fields.chi3_for(component))
        expected = (row[0] * factor.astype(np.float64)).astype(np.float32)
        produced = stepping._nonlinear_constitutive(fields, component, displacement)
        assert np.array_equal(_words(produced), _words(expected)), component
        rounded_first = (row[0].astype(np.float32) * factor).astype(np.float32)
        assert int(np.count_nonzero(_words(rounded_first) != _words(expected))) > 0, (
            f"{component}: the operands do not separate rounding before and after the factor")
    driver.close()
