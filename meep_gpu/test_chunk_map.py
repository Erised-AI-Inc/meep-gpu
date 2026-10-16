"""Behaviour pins for :mod:`chunk_map`: MEEP's chunk division as data.

What is pinned, and against what:

* the one-rank division's membership is the rule it replaced, on every face, wrap and
  sub-lattice (a frozen copy of that rule is the oracle here), and the places where that
  rule departs from MEEP's own one-rank boxes are pinned row by row rather than changed;
* a recorded division (boxes and flags as MEEP's ``get_chunk_volumes`` and
  ``structure_chunk::use_pml`` give them) is read row by row from each row's owner, a
  split inside the PML effort volume moves exactly the rows MEEP moves, and a local
  split is not separable and takes the mask route;
* the kernel record reproduces every line and refuses what it cannot hold;
* the region plan's runs and modes, and the unit-coefficient precondition;
* the driver's layout seam (default, lifted, refusals, the region plan rebuilt from it);
* with MEEP installed (a child process, as ``test_from_meep`` runs its cases): the lifted
  division equals the one-rank one on default lifts, is read from MEEP otherwise, and a
  ``BinaryPartition`` ``chunk_layout`` with mirror symmetry is refused by name when MEEP
  builds the structure from it, at the gate and on MEEP's own state (a mirrored axis
  whose grid MEEP did not halve), while the partition MEEP writes back after a halving
  build lifts folded, on the default lift's grid.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from .chunk_map import (
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayout,
    ChunkLayoutRefusal,
    RegionPlan,
    check_unit_coefficients,
    chunk_bounds_record,
    grid_frames,
    membership_from_record,
    record_index,
)
from .driver import FdtdDriver
from .grid import Grid, Mirror
from .pml import PML

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


def _rule_before_the_move(n_cells, faces, half_integer, wraps):
    """The one-rank rule as ``PML`` carried it before it moved to ``chunk_map``, frozen."""
    low, high = faces
    inside = numpy.zeros(n_cells, dtype=bool)
    if low > 0:
        span = min(n_cells, int(low + 1.5))
        if half_integer:
            inside[0:span] = True
        else:
            inside[1: min(n_cells, span + 1)] = True
    if high > 0:
        span = min(n_cells, int(high + 1.5))
        if half_integer:
            inside[n_cells - span:] = True
        else:
            inside[max(0, n_cells - span + 1):] = True
            if wraps:
                inside[0] = True
    return inside


def _12x12():  # The absorber-placement evidence packet's cell.
    grid = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    return grid, PML(grid, thickness={"x": 40, "y": 40})


# --- the one-rank division ------------------------------------------------------------


def test_one_rank_owns_one_integer_plane_past_the_inner_edge():
    """The chunk rule is structure.cpp:509-524 ownership, NOT the sigma support.

    MEEP's per-side effort volume is `int(cells + 1.5)` array cells from the wall and
    chunk selection reads sigsize per CHUNK (step_db.cpp:56-59), so the boundary
    cells whose own sigma is exactly zero still step the both-active branch.
    Measured discriminators on the 12x12 evidence-packet cell (CW Ez at x = -5,
    against CPU MEEP): deposits at y = -4.0 (cell 40) and y = -3.975 (cells 40+41)
    are both-active — stepping either with the dsig-only form reads 2.3e-01 … 4.6e-01
    where the recurrence reads ~7e-07 — while y = -3.9 (cell 42) runs dsig-only
    (4.9583e-01 under the recurrence, 5.6135e-07 dsig-only; the absorber-placement
    evidence packet §1.5-1.6). Integer samples: a chunk spanning doubled
    [c0, c1] owns c0+2..c1, so the low chunk owns cells 1..span and the boundary
    plane belongs to it, not to the interior neighbour; half-integer samples are
    c0+1..c1-1, cells 0..span-1.
    """
    grid, layer = _12x12()
    layout = ChunkLayout.one_rank(grid, layer)
    integer = layout.membership(1, False)
    assert bool(integer[40]) is True and bool(integer[41]) is True  # the measured pair
    assert bool(integer[42]) is False  # the measured dsig-only cell
    assert bool(integer[grid.ny - 41]) is False and bool(integer[grid.ny - 40]) is True
    assert bool(integer[0]) is True, "cell 0 is the high wall's own plane on a wrapping axis"
    half = layout.membership(1, True)
    assert bool(half[0]) is True and bool(half[40]) is True
    assert bool(half[41]) is False
    assert bool(half[grid.ny - 41]) is True and bool(half[grid.ny - 42]) is False
    assert not layout.membership(2, False).any() and not layout.membership(2, True).any()
    assert layout.source == "one_rank" and layout.is_separable()


def _sweep():
    """Grids and per-face tables: every count parity, wrap, fold and Yee lattice."""
    faces_list = ((0, 0), (1, 0), (0, 1), (2, 2), (2.5, 1), (1, 3.3), (0.3, 0.7), (5.5, 0),
                  (0, 5.5), (5.75, 0), (0, 6.25))
    for dims, n, boundary, fold in itertools.product((3, 2), (7, 8, 13, 41),
                                                     ("periodic", "metallic", "mixed"),
                                                     (None, "X", "Y")):
        resolved = "xyz" if dims == 3 else "xy"
        boundaries = (None if boundary == "periodic" else
                      {a: "metallic" for a in resolved if boundary == "metallic" or a != "x"})
        size = n / 10.0
        cell = (size, size + 0.1, size + 0.2) if dims == 3 else (size, size + 0.1, 0.0)
        grid = Grid(resolution=10.0, cell_size=cell, dimensions=dims, boundaries=boundaries,
                    symmetry=(Mirror(fold, 1),) if fold else ())
        for faces in faces_list:
            spec = {a: ((0 if grid.is_mirrored("xyz".index(a)) else faces[0]), faces[1])
                    for a in resolved}
            yield grid, spec
    for n, boundary in itertools.product((7, 12, 13), ("periodic", "metallic")):
        grid = Grid(resolution=10.0, cell_size=(n / 10.0, 0.0, n / 10.0 + 0.3), cylindrical=True, m=0,
                    boundaries={"z": boundary})
        for faces in faces_list:
            yield grid, {"x": (0, faces[1]), "z": faces}


def test_one_rank_is_the_rule_it_replaced_on_every_face_wrap_and_lattice():
    """A driver built without MEEP must step exactly as before: the rule moved, it did not change."""
    lines = 0
    for grid, spec in _sweep():
        try:
            layer = PML(grid, thickness=spec)
        except ValueError:
            continue  # The PML refuses this table outright; there is nothing to compare.
        layout = ChunkLayout.one_rank(grid, layer)
        for axis, half in itertools.product(range(3), (False, True)):
            frame = layout.frames[axis]
            expected = (numpy.zeros(frame.stored, dtype=bool) if frame.invariant else
                        _rule_before_the_move(frame.stored, layer.thickness_by_face[axis], half,
                                              grid.axis_wraps(axis)))
            assert numpy.array_equal(layout.membership(axis, half), expected), (grid, spec, axis, half)
            lines += 1
    assert lines > 3000


def test_one_rank_boxes_depart_from_the_rule_in_three_named_places():
    """Where the one-rank rule and MEEP's own one-rank boxes disagree — kept, and pinned.

    The membership a driver built without MEEP steps with is the rule, verbatim (word
    for word what this package has always stepped). The layout's boxes are MEEP's
    one-rank division: a face's effort volume is ``int(cells + 1.5)`` cells from MEEP's
    wall. Resolving those boxes through MEEP's ``owns`` differs from the rule in three
    places, each a single row; a LIFT steps MEEP's division read from MEEP and is not
    affected. Measured on a lift through the first one (Mirror(Y) with k_point = 0 and
    PML(1) on an 8x8 cell, CW Ez line on y = 3.0, the stored row the rule misses):
    relative L2 against CPU MEEP 8.91e-01 with the rule, 7.43e-07 with MEEP's division.
    """
    def boxes_only(layout):
        return ChunkLayout(layout.boxes, layout.pml_flags, layout.frames, faces=layout.faces)

    # 1. A folded PERIODIC axis with a high-face layer: the stored array runs one row
    #    past MEEP's window, and the rule counts from the stored end.
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.9, 1.0), symmetry=(Mirror("X", 1),))
    layout = ChunkLayout.one_rank(grid, PML(grid, thickness={"x": (0, 1)}))
    frame = layout.frames[0]
    assert (frame.stored, frame.owned, frame.far_ghost_row) == (6, 5, 4)
    boxed = boxes_only(layout)
    assert numpy.flatnonzero(layout.membership(0, False)).tolist() == [5]
    assert numpy.flatnonzero(boxed.membership(0, False)).tolist() == [4, 5]
    assert numpy.flatnonzero(layout.membership(0, True)).tolist() == [4, 5]
    assert numpy.flatnonzero(boxed.membership(0, True)).tolist() == [3, 4, 5]  # 5 is the far ghost
    # 2. A wrapping axis whose low run is clamped to the whole axis: MEEP's low chunk
    #    owns the wrapped image of cell 0; the rule never sets cell 0 from the low face.
    grid = Grid(resolution=10.0, cell_size=(0.7, 0.8, 0.9))
    layout = ChunkLayout.one_rank(grid, PML(grid, thickness={"x": (5.5, 0)}))
    assert not bool(layout.membership(0, False)[0])
    assert bool(boxes_only(layout).membership(0, False)[0])
    # 3. A folded metallic axis so short that the high run covers stored row 2: the near
    #    ghost reads its image row there; the rule reads False.
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.9, 1.0), symmetry=(Mirror("X", 1),),
                boundaries="metallic")
    layout = ChunkLayout.one_rank(grid, PML(grid, thickness={"x": (0, 3.3)}))
    assert layout.frames[0].near_ghost_row == 2
    assert not bool(layout.membership(0, False)[0]) and bool(layout.membership(0, False)[2])
    assert bool(boxes_only(layout).membership(0, False)[0])
    # Everywhere else the two agree: the 12x12 cell on every axis and lattice.
    grid, layer = _12x12()
    layout = ChunkLayout.one_rank(grid, layer)
    for axis, half in itertools.product(range(3), (False, True)):
        assert numpy.array_equal(boxes_only(layout).membership(axis, half), layout.membership(axis, half))


# --- a recorded division ----------------------------------------------------------------

#: Probe case (a)'s division, read off MEEP 1.33.0 after init_sim (10 x 10 um, res 20,
#: PML(1.0), metallic walls): a 3 x 3 product split at doubled -158 / 158.
_CASE_A_EDGES = ((-200, -158), (-158, 158), (158, 200))


def _case_a_grid():
    return Grid(resolution=20.0, cell_size=(10.0, 10.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})


def _case_a_division(split_y=None, split_x_range=None):
    """Case (a)'s boxes; ``split_y`` cuts the low-y PML row at that doubled y.

    With ``split_x_range`` the cut is LOCAL (only the chunk over that x range is cut),
    the mechanism behind the forced-``num_chunks`` divisions that are not separable.
    """
    boxes = []
    for (x0, x1), (y0, y1) in itertools.product(_CASE_A_EDGES, _CASE_A_EDGES):
        cut = split_y is not None and (y0, y1) == (-200, -158) and (
            split_x_range is None or (x0, x1) == split_x_range)
        if cut:
            boxes.append(((x0, y0, 0), (x1, split_y, 0)))
            boxes.append(((x0, split_y, 0), (x1, y1, 0)))
        else:
            boxes.append(((x0, y0, 0), (x1, y1, 0)))
    flags = []
    for lo, hi in boxes:
        x_flag = (lo[0], hi[0]) != (-158, 158)
        # structure_chunk::use_pml on y: the sliver [-160, -158] reaches no sigma
        # (distance 40 half-cells from the wall at -200 equals the extent).
        y_flag = (lo[1], hi[1]) in ((-200, -158), (-200, -160), (158, 200))
        flags.append((x_flag, y_flag, False))
    return boxes, flags


def test_a_recorded_division_reads_every_row_from_its_owner():
    """MEEP's default division equals the one-rank rule; a split inside the effort volume moves its rows.

    The split is phase 0's ``layouty`` (an explicit y cut at doubled -160 across the
    whole cell): MEEP's own words change with it (54,514 value-differing words against
    the default from step 100), and a map derived from the division restored 0. The
    rows it moves are the sliver's: half-integer row 20 (doubled -159) and integer row
    21 (doubled -158), on the y line only.
    """
    grid = _case_a_grid()
    layer = PML(grid, thickness={"x": 20, "y": 20})
    one_rank = ChunkLayout.one_rank(grid, layer)
    boxes, flags = _case_a_division()
    default = ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face, source="meep")
    for axis, half in itertools.product(range(3), (False, True)):
        assert numpy.array_equal(default.membership(axis, half), one_rank.membership(axis, half))
    assert default.boundaries(0) == default.boundaries(1) == (-200, -158, 158, 200)
    boxes, flags = _case_a_division(split_y=-160)
    split = ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face, source="meep")
    assert split.is_separable()
    assert numpy.flatnonzero(split.membership(1, True) != one_rank.membership(1, True)).tolist() == [20]
    assert numpy.flatnonzero(split.membership(1, False) != one_rank.membership(1, False)).tolist() == [21]
    assert numpy.array_equal(split.membership(0, False), one_rank.membership(0, False))
    assert split.boundaries(1) == (-200, -160, -158, 158, 200)
    # Unowned rows: the metallic wall row of the integer lattice reads False and is not owned.
    assert not bool(split.owned(0, False)[0]) and bool(split.owned(0, True)[0])


def test_chunk_of_is_meeps_owns_and_the_axis_row_belongs_to_the_chunk_at_r0():
    """``grid_volume::owns`` (vec.cpp:445-462): ``lo < p <= hi`` per axis, one owner per point.

    The readback's B/H alias rule asks which chunk owns a centred point and whether it
    owns each of its four words. On a cylindrical grid a point ON the axis (doubled
    ``r = 0``) is owned by a chunk whose box starts there; nothing owns ``r < 0``.
    """
    grid = _case_a_grid()
    layer = PML(grid, thickness={"x": 20, "y": 20})
    boxes, flags = _case_a_division()
    layout = ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face, source="meep")
    x = numpy.arange(-201, 202).reshape(-1, 1, 1)
    y = numpy.array([-157, 0, 158]).reshape(1, -1, 1)
    owner = layout.chunk_of([x, y, numpy.zeros((1, 1, 1), dtype=int)])
    for (i, j, _), k in numpy.ndenumerate(owner):
        px, py = int(x[i, 0, 0]), int(y[0, j, 0])
        hits = [n for n, (lo, hi) in enumerate(layout.boxes)
                if lo[0] < px <= hi[0] and lo[1] < py <= hi[1]]
        assert k == (hits[0] if hits else -1), (px, py, k, hits)
    assert int(owner[0, 1, 0]) == -1 and int(owner[1, 1, 0]) == -1, "the low wall at doubled -200 is owned by none"
    assert bool(numpy.all(layout.chunk_owns(owner, [x, y, 0]) == (owner >= 0)))
    cyl = Grid(resolution=10.0, cell_size=(1.0, 0.0, 1.2), cylindrical=True)
    frames = grid_frames(cyl)
    r_split = ChunkLayout([((0, 0, -12), (6, 0, 12)), ((6, 0, -12), (20, 0, 12))], [(False,) * 3] * 2, frames)
    r = numpy.array([-1, 0, 1, 6, 7]).reshape(-1, 1, 1)
    z = numpy.array([[[3]]])
    assert _layout_owners(r_split, r, z) == [-1, 0, 0, 0, 1]
    assert _layout_owners(r_split, r, z, cylindrical=False) == [-1, -1, 0, 0, 1]


def _layout_owners(layout, r, z, cylindrical=True):
    return [int(v) for v in layout.chunk_of([r, numpy.zeros((1, 1, 1), dtype=int), z], cylindrical).reshape(-1)]


def test_a_local_split_in_the_effort_volume_is_not_separable_and_takes_the_mask_route():
    """The forced-``num_chunks`` mechanism: a sliver over PART of the cross-section.

    Cutting only the interior-x chunk of the low-y PML row leaves doubled y in (-160,
    -158] covered by corner chunks that carry the y layer and by the sliver that does
    not, so no y line exists: the division is refused by the record and served by the
    mask, which is True on the corner columns and False on the sliver.
    """
    grid = _case_a_grid()
    layer = PML(grid, thickness={"x": 20, "y": 20})
    boxes, flags = _case_a_division(split_y=-160, split_x_range=(-158, 158))
    layout = ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face, source="meep")
    assert layout.membership(1, True) is None and layout.membership(1, False) is None
    assert sorted(layout.non_separable()) == [(1, False), (1, True)]
    assert layout.boundaries(1) is None  # The cut at -160 is local.
    with pytest.raises(ChunkLayoutRefusal, match="not separable along y"):
        chunk_bounds_record(layout)
    mask = layout.membership_mask(1, (1, 1, 0))  # Bz: half-integer on x and y.
    assert mask.shape == (200, 200, 1)
    assert bool(mask[5, 20, 0]) is True     # corner column (x low slab), sliver row
    assert bool(mask[100, 20, 0]) is False  # interior column, sliver row
    assert bool(mask[100, 10, 0]) is True   # interior column, inside the y layer's chunk
    assert numpy.array_equal(mask[:, 30:, 0], numpy.broadcast_to(
        ChunkLayout.one_rank(grid, layer).membership(1, True)[30:], (200, 170)))
    plan = RegionPlan(layout, grid, layer)
    assert plan.spec("curl", "Bz").mode == "mask"      # dsigu = y
    assert plan.spec("curl", "By").mode != "mask"      # dsigu = x, separable
    # The separable axis keeps its line, and its mask is that line broadcast.
    x_mask = layout.membership_mask(0, (0, 1, 0))
    assert numpy.array_equal(x_mask[:, :, 0], numpy.broadcast_to(
        layout.membership(0, False)[:, None], (200, 200)))


def test_a_division_for_another_grid_is_refused_by_name():
    """A box set that does not span this grid's window describes a different run."""
    grid = _case_a_grid()
    boxes, flags = _case_a_division()
    boxes[0] = ((-202, -200, 0), (-158, -158, 0))
    with pytest.raises(ChunkLayoutRefusal, match="describes another grid"):
        ChunkLayout(boxes, flags, grid_frames(grid))
    boxes, flags = _case_a_division()
    del boxes[4], flags[4]  # The interior chunk: a hole in the window.
    with pytest.raises(ChunkLayoutRefusal, match="no chunk owns"):
        ChunkLayout(boxes, flags, grid_frames(grid))


# --- the kernel record ------------------------------------------------------------------


def test_the_record_reproduces_every_line_and_refuses_what_it_cannot_hold():
    grid, layer = _12x12()
    layout = ChunkLayout.one_rank(grid, layer)
    record = chunk_bounds_record(layout)
    assert record.dtype == numpy.int32 and record.shape == (RECORD_WORDS,)
    # Integer lattice, x: the wrapped cell 0 joins the low run [1, 42) -> [0, 42); then [200, 240).
    assert [int(record[record_index(False, 0, w)]) for w in range(4)] == [0, 42, 200, 240]
    assert [int(record[record_index(True, 1, w)]) for w in range(4)] == [0, 41, 199, 240]
    assert [int(record[record_index(False, 2, w)]) for w in range(4)] == [0, 0, 0, 0]
    assert int(record[RECORD_PRESENCE_WORD]) == 0b011  # z is invariant on a 2-D grid
    for axis, half in itertools.product(range(3), (False, True)):
        n = layout.frames[axis].stored
        assert numpy.array_equal(membership_from_record(record, half, axis, n), layout.membership(axis, half))
    # Three runs on one line: x slabs carrying, not carrying, carrying, not, carrying.
    grid = _case_a_grid()
    edges = ((-200, -120), (-120, -40), (-40, 40), (40, 120), (120, 200))
    boxes = [((x0, -200, 0), (x1, 200, 0)) for x0, x1 in edges]
    flags = [(k % 2 == 0, False, False) for k in range(5)]
    striped = ChunkLayout(boxes, flags, grid_frames(grid))
    with pytest.raises(ChunkLayoutRefusal, match="has 3 runs"):
        chunk_bounds_record(striped)


def test_the_capsule_record_rebuilds_the_layout_exactly():
    grid, layer = _12x12()
    boxes, flags = _case_a_division(split_y=-160)
    case_a = _case_a_grid()
    for layout, on in ((ChunkLayout.one_rank(grid, layer), grid),
                       (ChunkLayout(boxes, flags, grid_frames(case_a), source="meep"), case_a)):
        rebuilt = ChunkLayout.from_record(json.loads(json.dumps(layout.to_record())))
        assert (rebuilt.boxes, rebuilt.pml_flags, rebuilt.frames, rebuilt.source) == (
            layout.boxes, layout.pml_flags, layout.frames, layout.source)
        for axis, half in itertools.product(range(3), (False, True)):
            assert numpy.array_equal(rebuilt.membership(axis, half), layout.membership(axis, half))
        assert numpy.array_equal(chunk_bounds_record(rebuilt), chunk_bounds_record(layout))


# --- the region plan --------------------------------------------------------------------


def test_the_region_plan_takes_each_split_axis_runs_and_a_mode_per_backend():
    grid, layer = _12x12()
    layout = ChunkLayout.one_rank(grid, layer)
    numpy_plan = RegionPlan(layout, grid, layer, backend="numpy")
    cupy_plan = RegionPlan(layout, grid, layer, backend="cupy")
    bz = numpy_plan.spec("curl", "Bz")  # dsigu = y, half-integer: y is the innermost resolved axis in 2-D
    assert (bz.axis, bz.half_integer, bz.inside, bz.view, bz.mode) == (
        1, True, ((0, 41), (199, 240)), "pair", "select")
    assert bz.outside == ((41, 199),)
    assert cupy_plan.spec("curl", "Bz").mode == "split"
    hx = numpy_plan.spec("constitutive", "Hx")  # own axis x, integer
    assert (hx.axis, hx.half_integer, hx.inside, hx.view, hx.mode) == (
        0, False, ((0, 42), (200, 240)), "slices", "split")
    for key in (("curl", "Bx"), ("curl", "Dx"), ("constitutive", "Hz"), ("constitutive", "Ez")):
        assert numpy_plan.spec(*key).mode == "outside_only"  # their split axis is the invariant z
    assert len(numpy_plan.specs) == 12
    # 3-D: z is innermost, so a component split on z selects on NumPy.
    grid3 = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0))
    layer3 = PML(grid3, thickness=4)
    plan3 = RegionPlan(ChunkLayout.one_rank(grid3, layer3), grid3, layer3)
    assert plan3.spec("curl", "Bx").axis == 2 and plan3.spec("curl", "Bx").mode == "select"
    assert plan3.spec("curl", "By").axis == 0 and plan3.spec("curl", "By").mode == "split"


def test_the_unit_coefficient_precondition_refuses_by_name():
    """Every coefficient is exactly 1.0 on every owned row outside its membership run.

    Holds on the one-rank default; a coefficient planted off 1.0 outside the run is
    refused naming the vector and row. On a folded periodic axis with a fractional
    high face the one-rank rule leaves a graded row outside its run (the first
    departure above), and the precondition is what catches it; MEEP's own boxes for the
    same layer pass.
    """
    grid, layer = _12x12()
    layout = ChunkLayout.one_rank(grid, layer)
    check_unit_coefficients(layout, layer)
    layer.sinv_x = layer.sinv_x.copy()
    layer.sinv_x[100, 0, 0] = numpy.float32(0.5)
    with pytest.raises(ChunkLayoutRefusal, match=r"sinv_x\[100\]"):
        check_unit_coefficients(layout, layer)
    grid = Grid(resolution=10.0, cell_size=(4.0, 1.0, 1.0), symmetry=(Mirror("X", 1),))
    layer = PML(grid, thickness={"x": (0, 2.4)})
    layout = ChunkLayout.one_rank(grid, layer)
    with pytest.raises(ChunkLayoutRefusal, match=r"_x\[19\]"):
        check_unit_coefficients(layout, layer)
    check_unit_coefficients(ChunkLayout(layout.boxes, layout.pml_flags, layout.frames,
                                        faces=layout.faces), layer)


# --- the driver's seam ------------------------------------------------------------------


def _case_a_driver():
    driver = FdtdDriver(cell_size=(10.0, 10.0, 0.0), resolution=20.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    driver.setup_pml({"x": 20, "y": 20})
    return driver


def test_the_driver_steps_the_one_rank_division_until_a_lift_sets_its_own():
    driver = _case_a_driver()
    try:
        default = driver.chunk_layout
        assert default.source == "one_rank" and driver.chunk_layout is default
        assert numpy.array_equal(driver.chunk_bounds_record(), chunk_bounds_record(default))
        # The region plan the PML sub-steps dispatch on is built from that division.
        plan = driver.fields.region_plan
        assert plan is not None and plan.layout is default and plan.pml is driver.pml
        # Integer y row 21 (doubled -158) is inside the y layer's one-rank chunk, so Dz
        # (dsigu = y, integer lattice) steps the recurrence there.
        assert bool(numpy.asarray(plan.mask("curl", "Dz"))[0, 21, 0]) is True
        boxes, flags = _case_a_division(split_y=-160)
        lifted = ChunkLayout(boxes, flags, grid_frames(driver.grid),
                             faces=driver.pml.thickness_by_face, source="meep")
        driver.set_chunk_layout(lifted)
        assert driver.chunk_layout is lifted
        assert numpy.array_equal(driver.chunk_bounds_record(), chunk_bounds_record(lifted))
        # Row 21 is the sliver's: outside every chunk carrying the y layer, so the
        # rebuilt plan steps it with the dsig-only form.
        plan = driver.fields.region_plan
        assert plan.layout is lifted
        assert bool(numpy.asarray(plan.mask("curl", "Dz"))[0, 21, 0]) is False
        with pytest.raises(ValueError, match="built for the absorber"):
            driver.setup_pml({"x": 20, "y": 10})
        driver.step()
        with pytest.raises(RuntimeError, match="before stepping"):
            driver.set_chunk_layout(lifted)
    finally:
        driver.close()


def test_set_chunk_layout_refuses_another_grid_and_another_absorber():
    driver = _case_a_driver()
    try:
        grid, layer = _12x12()
        with pytest.raises(ValueError, match="built for another grid"):
            driver.set_chunk_layout(ChunkLayout.one_rank(grid, layer))
        boxes, flags = _case_a_division()
        with pytest.raises(ValueError, match="built for the absorber"):
            driver.set_chunk_layout(ChunkLayout(boxes, flags, grid_frames(driver.grid),
                                                faces=((10, 10), (20, 20), (0, 0))))
        with pytest.raises(TypeError, match="ChunkLayout"):
            driver.set_chunk_layout(boxes)
        first = driver.chunk_layout
        driver.setup_pml({"x": 10, "y": 10})
        assert driver.chunk_layout is not first  # A new layer gives a new one-rank division.
        assert int(driver.chunk_bounds_record()[record_index(False, 0, 1)]) == 12  # [1, 12)
    finally:
        driver.close()


# --- with MEEP: the lifted division ---------------------------------------------------------

_CHILD_SCRIPT = r'''
import io, json, sys, contextlib
import numpy
import meep as mp
from meep_gpu.from_meep import gpu_compatibility, lift_simulation, MeepSimulationNotLiftable
from meep_gpu.chunk_map import ChunkLayout, ChunkLayoutRefusal, chunk_bounds_record

def cw(component, center, size=mp.Vector3()):
    return [mp.Source(mp.ContinuousSource(frequency=0.5), component=component, center=center, size=size)]

CASES = {
    "pml_2d_metallic": lambda: mp.Simulation(resolution=20, cell_size=mp.Vector3(10, 10),
        boundary_layers=[mp.PML(1.0)], sources=cw(mp.Ez, mp.Vector3())),
    "pml_3d_two_mirrors": lambda: mp.Simulation(resolution=15, cell_size=mp.Vector3(2.6, 2.2, 2.3),
        boundary_layers=[mp.PML(0.5)], symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=-1)],
        sources=cw(mp.Hx, mp.Vector3())),
    "pml_cylindrical_m1": lambda: mp.Simulation(resolution=10, cell_size=mp.Vector3(3, 0, 6),
        dimensions=mp.CYLINDRICAL, m=1, boundary_layers=[mp.PML(1.0)], sources=cw(mp.Ez, mp.Vector3(2.5, 0, 0))),
    "one_sided_periodic": lambda: mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 2, 4),
        k_point=mp.Vector3(), boundary_layers=[mp.PML(0.5, direction=mp.Z, side=mp.High)],
        sources=cw(mp.Ez, mp.Vector3())),
    "asymmetric_faces": lambda: mp.Simulation(resolution=10, cell_size=mp.Vector3(6, 5),
        boundary_layers=[mp.PML(1.0, direction=mp.X, side=mp.Low), mp.PML(0.55, direction=mp.X, side=mp.High),
                         mp.PML(0.75, direction=mp.Y)], sources=cw(mp.Ez, mp.Vector3())),
    "folded_periodic_high_face": lambda: mp.Simulation(resolution=10, cell_size=mp.Vector3(8, 8),
        boundary_layers=[mp.PML(1.0)], k_point=mp.Vector3(), symmetries=[mp.Mirror(mp.Y, phase=1)],
        sources=cw(mp.Ez, mp.Vector3(0, 3.0), mp.Vector3(8, 0))),
}

out = {}
for name, build in CASES.items():
    driver = lift_simulation(build(), prefer_gpu=False)
    lifted = driver.chunk_layout
    one_rank = ChunkLayout.one_rank(driver.grid, driver.pml)
    rows = {}
    for axis in range(3):
        for half in (False, True):
            a, b = lifted.membership(axis, half), one_rank.membership(axis, half)
            rows[f"{axis}{int(half)}"] = numpy.flatnonzero(a != b).tolist()
    out[name] = {"source": lifted.source, "rows": rows, "separable": lifted.is_separable(),
                 "record": chunk_bounds_record(lifted).tolist()}
    driver.close()

# A forced division that is not separable: lifted, with the notice; the record refuses.
sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(10, 10), boundary_layers=[mp.PML(1.0)],
                    sources=cw(mp.Ez, mp.Vector3()))
sim.num_chunks = 53
stderr = io.StringIO()
with contextlib.redirect_stderr(stderr):
    driver = lift_simulation(sim, prefer_gpu=False)
try:
    chunk_bounds_record(driver.chunk_layout)
    refused = None
except ChunkLayoutRefusal as error:
    refused = str(error)
out["forced_53"] = {"non_separable": [list(map(int, p)) for p in driver.chunk_layout.non_separable()],
                    "notice": [l for l in stderr.getvalue().splitlines() if "not separable" in l],
                    "record_refusal": refused, "chunks": len(driver.chunk_layout.boxes)}
driver.close()

# A BinaryPartition chunk_layout with mirror symmetry: MEEP does not halve its grid, so
# the gate and the lift refuse it. A Simulation chunk_layout is loaded after the halving.
def mirrored_c(**extra):
    return mp.Simulation(resolution=15, cell_size=mp.Vector3(2.6, 2.2, 2.3), boundary_layers=[mp.PML(0.5)],
                         symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=-1)],
                         sources=cw(mp.Hx, mp.Vector3()), **extra)
sim = mirrored_c(chunk_layout=mp.BinaryPartition(data=[(mp.X, -0.7333), 0, 1]))
out["layout_with_mirrors_gate"] = list(gpu_compatibility(sim).reasons)
try:
    lift_simulation(sim, prefer_gpu=False).close()
    out["layout_with_mirrors"] = None
except MeepSimulationNotLiftable as error:
    out["layout_with_mirrors"] = list(error.reasons)
# The same refusal keyed on what MEEP built: after init_sim the grid is not halved on
# the mirrored axes, which the lift names before its registration pin can.
from meep_gpu.from_meep import _require_folded_grid_halved
unhalved = mirrored_c(chunk_layout=mp.BinaryPartition(data=[(mp.X, -0.7333), 0, 1]))
unhalved.init_sim()
try:
    _require_folded_grid_halved(mp, unhalved, False)
    out["layout_with_mirrors_state"] = None
except MeepSimulationNotLiftable as error:
    out["layout_with_mirrors_state"] = list(error.reasons)
halved = mirrored_c()
halved.init_sim()
_require_folded_grid_halved(mp, halved, False)  # A default division is halved: no refusal.
# A partition MEEP WROTE BACK: the structure is built first with no layout, so MEEP halves
# it, then sets sim.chunk_layout to that structure's BinaryPartition (simulation.py:2117-2121),
# as test_simulation.py's test_infer_dimensions does. The gate reads the built structure,
# halved on both mirrored axes, and the lift folds it on the default lift's grid.
written_back = mirrored_c()
written_back._init_structure()
out["written_back_is_partition"] = isinstance(written_back.chunk_layout, mp.BinaryPartition)
out["written_back_gate"] = list(gpu_compatibility(written_back).reasons)
try:
    driver = lift_simulation(written_back, prefer_gpu=False)
    out["written_back_lift_refused"] = None
except MeepSimulationNotLiftable as error:
    out["written_back_lift_refused"] = list(error.reasons)
    driver = None
if driver is not None:
    default_driver = lift_simulation(mirrored_c(), prefer_gpu=False)
    out["written_back_shape"] = [list(driver.shape), list(default_driver.shape)]
    out["written_back_mirrored"] = [bool(driver.grid.is_mirrored(axis)) for axis in range(3)]
    one_rank = ChunkLayout.one_rank(driver.grid, driver.pml)
    out["written_back_one_rank"] = all(
        numpy.array_equal(driver.chunk_layout.membership(a, h), one_rank.membership(a, h))
        for a in range(3) for h in (False, True))
    driver.close()
    default_driver.close()
# That partition copied into a NEW mirrored simulation: no structure yet, so MEEP would build
# from it without halving, and the gate refuses it.
out["copied_partition_gate"] = list(gpu_compatibility(
    mirrored_c(chunk_layout=written_back.chunk_layout)).reasons)
# A structure built FROM a given partition before the lift exists and is not halved: the
# gate's read of it refuses, naming the axes.
prebuilt = mirrored_c(chunk_layout=mp.BinaryPartition(data=[(mp.X, -0.7333), 0, 1]))
prebuilt._init_structure()
out["prebuilt_unhalved_gate"] = list(gpu_compatibility(prebuilt).reasons)
source_sim = mirrored_c()
source_sim.init_sim()
driver = lift_simulation(mirrored_c(chunk_layout=source_sim), prefer_gpu=False)
one_rank = ChunkLayout.one_rank(driver.grid, driver.pml)
out["simulation_layout_with_mirrors"] = all(
    numpy.array_equal(driver.chunk_layout.membership(a, h), one_rank.membership(a, h))
    for a in range(3) for h in (False, True))
driver.close()

with open(sys.argv[1], "w") as handle:
    json.dump(out, handle)
'''


@requires_meep
@skip_without_meep
def test_the_lifted_division_is_read_from_meep_and_equals_one_rank_on_default_lifts(tmp_path):
    """The lift steps MEEP's own division; on a default division that is the one-rank rule.

    Five default lifts (2-D metallic, 3-D with two mirrors, cylindrical m = 1, a periodic
    cell with a one-sided z layer, asymmetric per-side faces): every membership line of
    the lifted layout equals the one-rank rule's. The sixth — a folded PERIODIC axis
    with a high-face layer — departs on exactly the two rows the one-rank rule misses
    (``test_one_rank_boxes_depart_from_the_rule_in_three_named_places``), and the lift
    takes MEEP's. A forced ``num_chunks = 53`` division is not separable along y: it is
    lifted with the named notice and the kernel record refuses it. A ``BinaryPartition``
    ``chunk_layout`` with mirror symmetry, given before MEEP builds the structure, is
    refused by name at the gate, at the lift, and by the lift's check of MEEP's own grid
    (not halved on the mirrored axes); so is the same partition copied into a new
    simulation, and a structure built from it before the lift (the gate reads it and names
    the unhalved axes). The partition MEEP writes back after a halving build (a structure
    built first with no layout) passes the gate and lifts folded on the default lift's
    grid and division; a ``chunk_layout`` given as a Simulation, which MEEP loads after
    halving, lifts.
    """
    script = tmp_path / "chunk_map_case.py"
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    output = tmp_path / "chunk_map_case.json"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = os.pathsep.join(
        [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    completed = subprocess.run([sys.executable, str(script), str(output)], capture_output=True,
                               text=True, env=environment, timeout=600)
    assert completed.returncode == 0 and output.exists(), (
        f"chunk_map child failed (exit {completed.returncode}).\nstdout tail:\n"
        f"{completed.stdout[-3000:]}\nstderr tail:\n{completed.stderr[-3000:]}")
    data = json.loads(output.read_text(encoding="utf-8"))
    for name in ("pml_2d_metallic", "pml_3d_two_mirrors", "pml_cylindrical_m1", "one_sided_periodic",
                 "asymmetric_faces"):
        entry = data[name]
        assert entry["source"] == "meep" and entry["separable"], name
        assert all(rows == [] for rows in entry["rows"].values()), (name, entry["rows"])
    folded = data["folded_periodic_high_face"]["rows"]
    assert folded["10"] == [31] and folded["11"] == [30], folded  # y, integer and half-integer
    assert all(rows == [] for key, rows in folded.items() if key not in ("10", "11")), folded
    forced = data["forced_53"]
    assert forced["non_separable"] and all(axis == 1 for axis, _half in forced["non_separable"]), forced
    assert forced["notice"] and forced["record_refusal"] and "not separable" in forced["record_refusal"]
    for key in ("layout_with_mirrors_gate", "layout_with_mirrors", "layout_with_mirrors_state",
                "copied_partition_gate", "prebuilt_unhalved_gate"):
        reasons = data[key]
        assert reasons and any("BinaryPartition chunk_layout together with mirror symmetry" in r
                               for r in reasons), (key, reasons)
    assert not any("not halved" in r for r in data["copied_partition_gate"]), data["copied_partition_gate"]
    assert any("MEEP's grid was not halved on y, z" in r for r in data["prebuilt_unhalved_gate"]), (
        data["prebuilt_unhalved_gate"])
    assert data["written_back_is_partition"] is True
    assert data["written_back_gate"] == [], data["written_back_gate"]
    assert data["written_back_lift_refused"] is None, data["written_back_lift_refused"]
    assert data["written_back_shape"][0] == data["written_back_shape"][1], data["written_back_shape"]
    assert data["written_back_mirrored"] == [False, True, True], data["written_back_mirrored"]
    assert data["written_back_one_rank"] is True
    assert data["simulation_layout_with_mirrors"] is True
