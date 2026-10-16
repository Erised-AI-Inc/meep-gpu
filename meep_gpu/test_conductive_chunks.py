"""Behaviour pins for a conductivity under a PML: MEEP's per-chunk conductive forms.

MEEP keeps a conductivity array per structure chunk and deletes it when every sample the
chunk takes is zero (structure.cpp:883-900); a chunk without one steps the lossless PML
forms even where the component is lossy elsewhere. Its conductive forms add the
``f_cond`` increment ``(fcnd' - fcnd)`` as one term (step_generic.cpp:167, :209), and its
``condinv`` is formed in double and rounded once (structure.cpp:699). What is pinned
here, and against what:

* ``condinv`` and ``condfac`` word for word against MEEP's two expressions, with sigma
  values whose float32 spelling of ``condinv`` rounds differently (the test bites);
* the samples that decide a chunk's conductivity: ``LOOP_OVER_VOL``'s owned samples plus
  one (below on the integer lattice, above on the half-integer one), and a chunk whose
  only lossy sample is that one keeping its conductivity, on both lattices;
* the per-cell forms against a scalar transcription of ``step_generic.cpp``'s eight
  branches, with signed zeros planted, on a division where some chunks keep the
  conductivity and some do not;
* the two mutants the identity legs found -- the flattened grouping and conductive forms
  run in a chunk that keeps none -- each move a word, and only on dsig cells for the
  second;
* the auxiliary contract: ``f_cond`` changes only on the dsig cells of the chunks that
  keep the conductivity, ``fu`` only on the dsigu cells.

Host NumPy only; no MEEP. The identity against MEEP itself is the bit_parity_0_10 AP6d
acceptance (coff MEEP 1.33.0, cases k1-k5).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any

import numpy
import pytest

from . import stepping
from .chunk_map import ChunkLayout, RegionPlan
from .driver import FdtdDriver
from .fields import IYEE_SHIFTS
from .grid import Grid
from .pml import PML

F32 = numpy.float32


def _words(array: Any) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint32)


def _small():
    grid = Grid(resolution=10.0, cell_size=(4.0, 4.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})
    layer = PML(grid, thickness={"x": 4, "y": 4})
    return grid, layer, ChunkLayout.one_rank(grid, layer)


# --- condinv and condfac -------------------------------------------------------------


SIGMAS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7, 1.0, 2.0)


@pytest.mark.parametrize("side", ["D", "B"])
def test_condinv_is_formed_in_double_and_rounded_once(side):
    """``1 / (1 + conductivity[i] * dt * 0.5)`` in double, one rounding (structure.cpp:699).

    The float32 spelling of the same expression rounds twice and gives another word at
    sigma = 0.1, 0.25, 0.3, 0.5 and 2.0 with dt = 0.05; the check below requires that at
    least three of the sampled sigmas tell the two apart, so it cannot pass on either
    spelling.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 0.0), resolution=10.0, dimensions=2)
    try:
        shape = driver.grid.shape
        values = numpy.zeros(shape, dtype=F32)
        values.reshape(-1)[:len(SIGMAS)] = SIGMAS
        components = tuple(side + axis for axis in "xyz")
        mapping = {c: values for c in components}
        if side == "D":
            driver.set_conductivity(mapping)
        else:
            driver.set_b_conductivity(mapping)
        dt = float(driver.grid.dt)
        assert dt == 0.05
        want_inv = (1.0 / (1.0 + (values.astype(numpy.float64) * dt) * 0.5)).astype(F32)
        single_inv = (F32(1.0) / (F32(1.0) + values * F32(dt / 2.0))).astype(F32)
        dt2 = F32(F32(dt) * F32(0.5))
        want_fac = (F32(1.0) - values * dt2).astype(F32)
        assert numpy.count_nonzero(_words(want_inv) != _words(single_inv)) >= 3
        for component in components:
            got_inv = numpy.asarray(driver.fields.condinv_for(component))
            got_fac = numpy.asarray(driver.fields.condfac_for(component))
            assert numpy.array_equal(_words(got_inv), _words(want_inv)), component
            assert numpy.array_equal(_words(got_fac), _words(want_fac)), component
    finally:
        driver.close()


# --- the samples that decide a chunk's conductivity ----------------------------------


def _owned_rows(layout, axis, half, chunk):
    frame = layout.frames[axis]
    lo, hi = layout.boxes[chunk][0][axis], layout.boxes[chunk][1][axis]
    points = frame.doubled(half)
    return set(int(r) for r in numpy.flatnonzero((points > lo) & (points <= hi)))


def test_loop_over_vol_adds_one_sample_below_on_the_integer_lattice_and_above_on_the_half():
    """``LOOP_OVER_VOL`` is ``[lo + s, hi + s]`` doubled (vec.hpp:170-172).

    Owned samples are ``lo < p <= hi``; so the integer lattice adds the sample AT ``lo``
    (one row below the owned run) and the half-integer lattice the one at ``hi + 1`` (one
    row above). Past the stored range -- the top wall plane of this metallic axis and
    the half-integer sample beyond it -- the last stored row stands in.
    """
    grid, _layer, layout = _small()
    frame = layout.frames[0]
    checked = 0
    for chunk in range(len(layout.boxes)):
        lo, hi = layout.boxes[chunk][0][0], layout.boxes[chunk][1][0]
        for half in (False, True):
            owned = _owned_rows(layout, 0, half, chunk)
            extra_coordinate = hi + 1 if half else lo
            extra = (extra_coordinate - frame.origin - (1 if half else 0)) // 2
            extra = min(max(extra, 0), frame.stored - 1)
            assert set(layout.loop_rows(0, half, chunk).tolist()) == owned | {extra}, (chunk, half)
            if half:
                assert extra == max(owned) + 1 or extra == frame.stored - 1
            else:
                assert extra == min(owned) - 1 or extra == 0
            checked += 1
    assert checked == 2 * len(layout.boxes)


def test_on_a_wrapping_axis_the_top_chunks_extra_sample_is_the_periodic_image():
    """A periodic x cut into two chunks: past the window's top the sample is row 0.

    The top chunk's integer loop ends on the ``big_corner`` plane, which is stored row 0
    (its periodic image), and its half-integer loop ends half a cell above it, which is
    half-integer row 0. The low chunk's loops stay below the cut.
    """
    from .chunk_map import grid_frames  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=(2.0, 1.0, 0.0), dimensions=2,
                boundaries={"x": "periodic", "y": "metallic"})
    frames = grid_frames(grid)
    assert frames[0].wraps and frames[0].stored == frames[0].owned
    (x0, x1), (y0, y1) = frames[0].window, frames[1].window
    cut = x0 + 2 * (frames[0].owned // 2)
    boxes = [((x0, y0, 0), (cut, y1, 0)), ((cut, y0, 0), (x1, y1, 0))]
    layout = ChunkLayout(boxes, [(False, False, False)] * 2, frames, source="record")
    n = frames[0].owned
    top_integer = layout.loop_rows(0, False, 1).tolist()
    top_half = layout.loop_rows(0, True, 1).tolist()
    assert 0 in top_integer and n - 1 in top_integer and len(top_integer) == n - n // 2 + 1
    assert 0 in top_half and len(top_half) == n - n // 2 + 1
    low_half = layout.loop_rows(0, True, 0).tolist()
    assert low_half == list(range(0, n // 2 + 1))
    sigma = numpy.zeros(grid.shape, dtype=F32)
    sigma[0, :, :] = 0.5  # only row 0, which the low chunk owns on the half lattice
    assert IYEE_SHIFTS["Dx"][0] == 1
    assert layout.conductive_chunks(sigma, IYEE_SHIFTS["Dx"]) == (True, True)
    sigma[0, :, :] = 0.0
    sigma[n - 1, :, :] = 0.5  # only the top chunk's own last half-integer row
    assert layout.conductive_chunks(sigma, IYEE_SHIFTS["Dx"]) == (False, True)


@pytest.mark.parametrize("component", ["Dz", "Dx"])
def test_a_chunk_whose_only_lossy_sample_is_the_extra_one_keeps_its_conductivity(component):
    """The k4 mechanism, on both lattices: the extra sample alone keeps an array.

    Dz sits on the integer x lattice, so the x-high chunks take one sample below their
    owned rows; Dx on the half-integer one, so the x-low chunks take one above. A
    conductivity on that row only (owned by the interior chunks) keeps the conductivity
    of the chunks beside it in MEEP; the owned samples alone would delete it.
    """
    grid, layer, layout = _small()
    iyee = IYEE_SHIFTS[component]
    half = bool(iyee[0])
    lows = sorted({lo[0] for lo, _ in layout.boxes})
    target = [k for k, (lo, _hi) in enumerate(layout.boxes) if lo[0] == (lows[0] if half else lows[-1])]
    row = (max(_owned_rows(layout, 0, half, target[0])) + 1 if half
           else min(_owned_rows(layout, 0, half, target[0])) - 1)
    sigma = numpy.zeros(grid.shape, dtype=F32)
    sigma[row, :, :] = 0.5
    present = layout.conductive_chunks(sigma, iyee)
    owned_only = []
    for k in range(len(layout.boxes)):
        rows = [sorted(_owned_rows(layout, a, bool(iyee[a]), k)) if not layout.frames[a].invariant
                else [0] for a in range(3)]
        owned_only.append(bool(numpy.any(sigma[numpy.ix_(*rows)] != 0)))
    for k in target:
        assert present[k] and not owned_only[k], (k, present, owned_only)
    # The chunks that own the row keep it under both rules.
    assert sum(present) == sum(owned_only) + len(target)
    mask = layout.conductive_mask(sigma, iyee)
    assert numpy.array_equal(mask, numpy.asarray(present)[layout.owner_volume(iyee)])


def test_the_owner_volume_is_meeps_owns_on_every_owned_cell():
    """``owner_volume`` agrees with ``chunk_of`` (``owns``, vec.cpp:445-462) wherever MEEP owns a cell."""
    grid, _layer, layout = _small()
    for component, iyee in IYEE_SHIFTS.items():
        owner = layout.owner_volume(iyee)
        coordinates = [layout.frames[a].doubled(bool(iyee[a])).reshape([-1 if b == a else 1 for b in range(3)])
                       for a in range(3)]
        direct = layout.chunk_of(coordinates)
        owned = numpy.ones(owner.shape, dtype=bool)
        for a in range(3):
            line = layout.owned(a, bool(iyee[a]))
            owned &= line.reshape([-1 if b == a else 1 for b in range(3)])
        assert numpy.array_equal(owner[owned], numpy.broadcast_to(direct, owner.shape)[owned]), component


# --- the forms -----------------------------------------------------------------------


def _vector(layer, kind, axis, half):
    return numpy.asarray(getattr(layer, f"{kind}_{'xyz'[axis]}{'_h' if half else ''}")).reshape(-1)


def _meep_cell(form, x, u, c, X, kms, sinv, kmu, sinu, cf, ci):
    """One cell of step_generic.cpp's step_curl, in float32, for the named branch.

    Returns (f, fu, fcnd); an auxiliary the branch does not touch is returned as given.
    """
    if form == "cnd_both":       # :200-210
        c_new = ((cf * c) - X) * ci
        u_new = ((kms * u) + (c_new - c)) * sinv
        return ((kmu * x) + u_new - u) * sinu, u_new, c_new
    if form == "cnd_dsig":       # :160-168
        c_new = ((cf * c) - X) * ci
        return ((kms * x) + (c_new - c)) * sinv, u, c_new
    if form == "cnd_dsigu":      # :112-119
        u_new = ((cf * u) - X) * ci
        return ((kmu * x) + u_new - u) * sinu, u_new, c
    if form == "cnd_neither":    # :84-90
        return ((cf * x) - X) * ci, u, c
    if form == "both":           # :226-234
        u_new = ((kms * u) - X) * sinv
        return ((kmu * x) + u_new - u) * sinu, u_new, c
    if form == "dsig":           # :182-183
        return ((kms * x) - X) * sinv, u, c
    if form == "dsigu":          # :138-141
        u_new = u - X
        return ((kmu * x) + u_new - u) * sinu, u_new, c
    return x - X, u, c           # :104, neither


def _form_of(dsig, dsigu, lossy):
    base = "both" if (dsig and dsigu) else "dsig" if dsig else "dsigu" if dsigu else "neither"
    return ("cnd_" + base) if lossy else base


def _setup_forms(component, seed=11):
    grid, layer, layout = _small()
    plan = RegionPlan(layout, grid, layer)
    term = {t.target: t for t in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)}[component]
    iyee = IYEE_SHIFTS[component]
    shape = grid.shape
    rng = numpy.random.default_rng(seed)
    sigma = (0.1 + rng.random(shape) * 0.6).astype(F32)
    sigma[shape[0] // 2:] = 0.0            # the high-x chunks keep no conductivity
    sigma[rng.random(shape) < 0.2] = 0.0   # zero cells inside the lossy chunks too
    dt2 = F32(F32(grid.dt) * F32(0.5))
    cf = (F32(1.0) - sigma * dt2).astype(F32)
    ci = (1.0 / (1.0 + (sigma.astype(numpy.float64) * float(grid.dt)) * 0.5)).astype(F32)
    arrays = [(rng.standard_normal(shape) * 0.5).astype(F32) for _ in range(4)]
    field, fu, f_cond, curl = arrays
    # Signed zeros planted in every operand: the output selects must keep them.
    for array, value in ((field, -0.0), (fu, 0.0), (f_cond, -0.0), (curl, -0.0)):
        array.reshape(-1)[::7] = value
    curl.reshape(-1)[3::11] = 0.0
    return grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond, curl


def _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl, *, conductive="plan"):
    half = term.target[0] == "B"
    kms, sinv = stepping._curl_coefficients(plan.pml, term.dsig, half)
    kmu, sinu = stepping._curl_coefficients(plan.pml, term.dsigu, half)
    if conductive == "plan":
        chunks = plan.conductive_chunks(term.target, sigma)
        conductive = None if all(chunks) else plan.conductive_mask(term.target, sigma)
    got = [field.copy(), fu.copy(), f_cond.copy()]
    stepping._apply_conductive_pml_update(
        numpy, got[0], curl, cf, ci, kms, sinv, kmu, sinu, got[1], got[2],
        dsig_active=plan.axis_mask("xyz".index(term.dsig), term.target),
        dsigu_active=plan.mask("curl", term.target), conductive=conductive)
    return got


@pytest.mark.parametrize("component", ["Dx", "Dy", "Dz", "Bx", "By", "Bz"])
def test_every_cell_steps_meeps_branch_for_its_chunk(component):
    """Cell for cell against ``step_generic.cpp``'s eight branches, chosen per chunk, on owned cells."""
    (grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond,
     curl) = _setup_forms(component)
    got = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl)
    a_sig, a_sigu = "xyz".index(term.dsig), "xyz".index(term.dsigu)
    dsig = layout.membership_mask(a_sig, iyee)
    dsigu = layout.membership_mask(a_sigu, iyee)
    lossy = layout.conductive_mask(sigma, iyee)
    kms, sinv = _vector(layer, "kms", a_sig, bool(iyee[a_sig])), _vector(layer, "sinv", a_sig, bool(iyee[a_sig]))
    kmu, sinu = _vector(layer, "kms", a_sigu, bool(iyee[a_sigu])), _vector(layer, "sinv", a_sigu, bool(iyee[a_sigu]))
    want = [field.copy(), fu.copy(), f_cond.copy()]
    seen = set()
    for cell in numpy.ndindex(field.shape):
        k, ku = cell[a_sig], cell[a_sigu]
        form = _form_of(bool(dsig[cell]), bool(dsigu[cell]), bool(lossy[cell]))
        seen.add(form)
        out = _meep_cell(form, field[cell], fu[cell], f_cond[cell], curl[cell], kms[k], sinv[k],
                         kmu[ku], sinu[ku], cf[cell], ci[cell])
        for array, value in zip(want, out):
            array[cell] = value
    # MEEP steps owned cells only; an unowned wall row (its curl is masked to zero and
    # the wall pass overwrites it) may carry a PML coefficient that is not 1.
    owned = numpy.ones(field.shape, dtype=bool)
    for a in range(3):
        owned &= layout.owned(a, bool(iyee[a])).reshape([-1 if b == a else 1 for b in range(3)])
    owned = owned.reshape(-1)
    for name, a, b in zip(("f", "fu", "f_cond"), got, want):
        differing = int(numpy.count_nonzero((_words(a) != _words(b)) & owned))
        assert differing == 0, (component, name, differing, sorted(seen))
        if name != "f":  # the auxiliaries are touched on their own cells only, owned or not
            assert numpy.array_equal(_words(a), _words(b)), (component, name)
    # The division reaches a lossy and a lossless chunk on dsig cells for every component
    # whose dsig axis is resolved (z is invariant here).
    if term.dsig != "z":
        assert {"cnd_dsig", "dsig"} & seen == {"cnd_dsig", "dsig"} or \
               {"cnd_both", "both"} & seen == {"cnd_both", "both"}, sorted(seen)


def test_the_flattened_grouping_moves_a_word():
    """``((kms * x) + fcnd') - fcnd`` -- the grouping before this change -- is another number."""
    (grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond,
     curl) = _setup_forms("Dz")
    got = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl, conductive=None)
    half = False
    kms, sinv = stepping._curl_coefficients(layer, term.dsig, half)
    c_new = ((cf * f_cond) - curl) * ci
    flattened = (((field * kms) + c_new) - f_cond) * sinv
    dsig = numpy.broadcast_to(plan.axis_mask("xyz".index(term.dsig), term.target), field.shape)
    dsigu = numpy.broadcast_to(plan.mask("curl", term.target), field.shape)
    only = dsig & ~dsigu
    assert numpy.count_nonzero(_words(got[0][only]) != _words(flattened[only])) > 0


@pytest.mark.parametrize("component", ["Dz", "Bz"])
def test_a_chunk_without_conductivity_differs_from_the_conductive_forms_only_on_dsig(component):
    """Stepping every chunk as conductive moves words on the dsig cells of the lossless chunks, and nowhere else.

    Off dsig every sample of such a chunk is zero, so ``cf = ci = 1`` and the conductive
    forms store the lossless words; on dsig they run the ``f_cond`` stage, and
    ``(fcnd - X) - fcnd`` is not ``-X``.
    """
    (grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond,
     curl) = _setup_forms(component)
    chosen = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl)
    everywhere = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl, conductive=None)
    dsig = numpy.broadcast_to(plan.axis_mask("xyz".index(term.dsig), term.target), field.shape)
    lossless = ~layout.conductive_mask(sigma, iyee)
    moved = _words(chosen[0]) != _words(everywhere[0])
    assert numpy.count_nonzero(moved & (dsig & lossless).reshape(-1)) > 0
    assert numpy.count_nonzero(moved & ~(dsig & lossless).reshape(-1)) == 0


def test_a_lossless_chunk_keeps_meeps_signed_zero_where_a_zero_f_cond_would_not():
    """The lossless dsig form is not the conductive one with ``f_cond = 0``, ``cf = ci = 1``.

    With ``kms * f = -0`` and ``X = +0`` MEEP's lossless form stores
    ``((kms * f) - X) * sinv = -0`` (step_generic.cpp:182-183); the conductive form fed a
    zero auxiliary stores ``((kms * f) + ((0 - X) - 0)) * sinv = +0``. Equal in value, not
    in word: the lossless chunks need their own forms.
    """
    (grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond,
     curl) = _setup_forms("Dz")
    dsig = numpy.broadcast_to(plan.axis_mask("xyz".index(term.dsig), term.target), field.shape)
    dsigu = numpy.broadcast_to(plan.mask("curl", term.target), field.shape)
    lossless = ~layout.conductive_mask(sigma, iyee)
    owned = numpy.ones(field.shape, dtype=bool)
    for a in range(3):
        owned &= layout.owned(a, bool(iyee[a])).reshape([-1 if b == a else 1 for b in range(3)])
    cells = dsig & ~dsigu & lossless & owned
    assert numpy.count_nonzero(cells) > 0
    field[cells] = F32(-0.0)
    curl[cells] = F32(0.0)
    f_cond[cells] = F32(0.0)
    got = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl)
    assert numpy.all(_words(got[0][cells]) == numpy.uint32(0x80000000))
    zero_cf = numpy.where(lossless, F32(1.0), cf).astype(F32)
    zero_ci = numpy.where(lossless, F32(1.0), ci).astype(F32)
    as_conductive = _run(plan, term, sigma, zero_cf, zero_ci, field, fu, f_cond, curl,
                         conductive=None)
    assert numpy.all(_words(as_conductive[0][cells]) == numpy.uint32(0x00000000))


@pytest.mark.parametrize("component", ["Dx", "Dy", "Dz", "Bx", "By", "Bz"])
def test_the_auxiliaries_change_only_on_their_forms_cells(component):
    """``f_cond`` on dsig cells of the lossy chunks, ``fu`` on dsigu cells; a NaN elsewhere survives."""
    (grid, layer, layout, plan, term, iyee, sigma, cf, ci, field, fu, f_cond,
     curl) = _setup_forms(component)
    sentinel = numpy.uint32(0x7FC5A5A5)
    dsig = numpy.broadcast_to(plan.axis_mask("xyz".index(term.dsig), term.target), field.shape)
    dsigu = numpy.broadcast_to(plan.mask("curl", term.target), field.shape)
    cond_cells = (dsig & layout.conductive_mask(sigma, iyee)).reshape(-1)
    fu_cells = dsigu.reshape(-1)
    fu_planted, f_cond_planted = fu.copy(), f_cond.copy()
    _words(fu_planted)  # contiguous
    fu_planted.reshape(-1).view(numpy.uint32)[~fu_cells] = sentinel
    f_cond_planted.reshape(-1).view(numpy.uint32)[~cond_cells] = sentinel
    clean = _run(plan, term, sigma, cf, ci, field, fu, f_cond, curl)
    planted = _run(plan, term, sigma, cf, ci, field, fu_planted, f_cond_planted, curl)
    assert numpy.all(_words(planted[1])[~fu_cells] == sentinel)
    assert numpy.all(_words(planted[2])[~cond_cells] == sentinel)
    assert numpy.array_equal(_words(planted[0]), _words(clean[0]))
    assert numpy.array_equal(_words(planted[1])[fu_cells], _words(clean[1])[fu_cells])
    assert numpy.array_equal(_words(planted[2])[cond_cells], _words(clean[2])[cond_cells])
