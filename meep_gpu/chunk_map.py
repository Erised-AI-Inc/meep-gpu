"""
MEEP's chunk division as data: which stored cells each PML form owns.

MEEP selects the PML branch of every curl and constitutive update PER STRUCTURE
CHUNK, from ``sigsize``: ``dsig`` / ``dsigu`` (step_db.cpp:56-59) and ``dsigw``
(update_eh.cpp:162-163) are set for a chunk exactly when that chunk allocated
sigma on the axis, which ``structure_chunk::use_pml`` decides from the chunk's own
extent along the axis (structure.cpp:625-652). Which chunk a cell belongs to is
MEEP's ``grid_volume::owns``: ``lo < p <= hi`` on the doubled lattice
(vec.cpp:445-462). So the form a cell steps with is a property of the chunk
division, not of the cell's own sigma: a chunk that carries a layer owns cells
whose sigma is exactly zero.

:class:`ChunkLayout` holds that division: per chunk, its box in MEEP's doubled
coordinates and its per-axis ``sigsize > 1`` flags, plus each axis's
:class:`AxisFrame` (where the stored indices sit on the doubled lattice). Two
constructors make one:

* :meth:`ChunkLayout.from_meep` reads the lifted simulation's own division
  (``structure.get_chunk_volumes()`` after its ``init_sim``) and transcribes
  ``structure_chunk::use_pml`` for the flags. This is what a lift steps with.
* :meth:`ChunkLayout.one_rank` is the default for a driver built without MEEP:
  the one-rank division, a ``{low, interior, high}`` product per axis, whose
  membership is the package's one-rank rule (:func:`one_rank_line`), moved here
  from the PML verbatim.

Everything else is derived from the layout and cached on it:

* :meth:`ChunkLayout.membership` — per (axis, Yee sub-lattice), a bool line over
  the stored index: is the cell's owning chunk one that carries this axis's
  layer? ``None`` when the division is not separable along the axis (the
  owner's flag depends on the other coordinates). Ghost rows take the value of
  the row they image (``stepping.MIRROR_SOURCE_INDEX`` for the near ghost of a
  mirrored axis, ``stepping._far_reflect_rows`` for the far ghost of a folded
  periodic one), so every consumer keys one line.
* :meth:`ChunkLayout.owned` — which stored rows MEEP owns on an axis (after the
  periodic translation of a wrapping axis); walls, the cylindrical axis row and
  ghosts are not owned.
* :meth:`ChunkLayout.membership_mask` — the full bool volume, for divisions that
  are not separable.
* :meth:`ChunkLayout.owner_volume` — per stored cell of a component lattice, the
  chunk that owns it; every per-chunk fact a cell steps with is read through it.
* :meth:`ChunkLayout.conductive_chunks` / :meth:`ChunkLayout.conductive_mask` —
  per chunk, whether MEEP keeps a conductivity array for a component there: it
  deletes the array when every sample ``LOOP_OVER_VOL`` visits is zero
  (:meth:`ChunkLayout.loop_rows`, structure.cpp:883-900), and a chunk without one
  steps the non-conductive PML forms.
* :meth:`ChunkLayout.boundaries` — the chunk cuts along an axis.
* :meth:`ChunkLayout.offdiag_kept` / :meth:`ChunkLayout.offdiag_presence` — per
  chunk, whether MEEP keeps an off-diagonal ``chi1inv`` array there (it deletes one
  that is zero over every sample the chunk allocates, :meth:`ChunkLayout.chunk_samples`),
  and that verdict spread over each chunk's owned cells, for ``update_E``'s row sum.
* :func:`chunk_bounds_record` — the kernels' ``int32`` record, ``[a0, a1) U [b0,
  b1)`` per (sub-lattice, axis) plus a partner-presence word.
* :class:`RegionPlan` — per curl and constitutive component, the inside and
  outside index runs of its split axis and the route a backend evaluates them
  by (views, a whole-grid select on the line, or a select on the full mask),
  with the unit-coefficient precondition checked: every PML coefficient is
  exactly ``1.0`` on every owned cell outside its membership. The stepping code
  dispatches on it (``stepping._apply_curl``, ``update_H``, ``update_E``), and
  ``Fields.drive_field`` and the deposit repair read the constitutive regions
  from it.

Inputs: a Grid and a PML (one-rank) or a lifted ``mp.Simulation`` (MEEP).
Outputs: host NumPy bool lines, masks and ``int32`` records, and per-backend
broadcast masks the region plan caches for the stepping code.

MEEP source references:
- structure.cpp:100-140 (chunk boxes: partition intersected with effort volumes),
  :509-524 (``structure::use_pml``: the per-side effort volume),
  :208-234 (``boundary_region::apply``: which faces exist), :625-652
  (``pml_x`` and ``structure_chunk::use_pml``: ``found_pml``)
- vec.cpp:445-462 (``owns``), :465-473 (``has_boundary``), :692-719
  (``boundary_location``); vec.hpp:1100-1104 (``little_owned_corner0``)
- step_db.cpp:56-59, update_eh.cpp:162-163 (the per-chunk branch selection)
- structure.cpp:883-900 (a chunk's conductivity deleted when trivial), vec.hpp:170-172
  (``LOOP_OVER_VOL``: the samples that decide it)
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy

AXIS_NAMES = ("x", "y", "z")

#: Doubled-coordinate box of one chunk: ``((lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z))``.
Box = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
#: Per-face thickness in cells, ``((x_low, x_high), (y_low, y_high), (z_low, z_high))``.
Faces = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

#: The kernel record: 2 sub-lattices x 3 axes x ``[a0, a1, b0, b1]``, then the
#: partner-presence word. Word ``RECORD_PRESENCE_WORD`` holds bit ``a`` set when axis
#: ``a`` is not translationally invariant (``Grid.is_invariant``).
RECORD_BOUNDS_WORDS = 24
RECORD_PRESENCE_WORD = 24
RECORD_WORDS = 25

NO_FACES: Faces = ((0, 0), (0, 0), (0, 0))

NON_SEPARABLE_NOTICE = (
    "meep_gpu: the lifted chunk division is not separable along {where}: which PML form "
    "a cell steps depends on more than its index along that axis, so no per-axis chunk "
    "record describes it."
)


class ChunkLayoutRefusal(ValueError):
    """A chunk layout, record or region plan that cannot be built, refused by name."""


def record_index(half_integer: bool, axis: int, word: int) -> int:
    """Position of ``[a0, a1, b0, b1][word]`` for (sub-lattice, axis) in the record."""
    if axis not in (0, 1, 2) or word not in (0, 1, 2, 3):
        raise ValueError(f"record_index needs axis in 0..2 and word in 0..3, got {axis}, {word}")
    return ((1 if half_integer else 0) * 3 + axis) * 4 + word


@dataclass(frozen=True)
class AxisFrame:
    """Where one axis's stored indices sit on MEEP's doubled lattice.

    Stored index ``i`` of a component with Yee shift ``s`` on this axis is the
    doubled coordinate ``origin + 2*i + s`` (``Grid.origin_doubled``). MEEP's window
    on the axis is ``[origin, origin + 2*owned]`` (``Grid.big_corner_doubled``); a
    wrapping axis translates a coordinate outside it by the period ``2*owned``.

    ``near_ghost_row`` is set on a mirrored axis: the stored row the shift-0 cell 0
    images (``stepping.MIRROR_SOURCE_INDEX``). ``far_ghost_row`` is set on a folded
    periodic axis: the stored row the shift-1 last slot images
    (``stepping._far_reflect_rows``).
    """

    origin: int
    stored: int
    owned: int
    wraps: bool
    invariant: bool
    near_ghost_row: Optional[int] = None
    far_ghost_row: Optional[int] = None

    @property
    def window(self) -> Tuple[int, int]:  # MEEP's (little_corner, big_corner) on this axis.
        return self.origin, self.origin + 2 * self.owned

    def doubled(self, half_integer: bool) -> numpy.ndarray:  # Doubled coordinate of every stored row.
        return self.origin + 2 * numpy.arange(self.stored, dtype=numpy.int64) + (1 if half_integer else 0)

    @classmethod
    def of_grid(cls, grid: Any, axis: int) -> "AxisFrame":
        """The frame the grid's own registration rules give one axis."""
        from .stepping import MIRROR_SOURCE_INDEX, _far_reflect_rows, _stored_past_owned  # noqa: PLC0415

        invariant = bool(grid.is_invariant(axis))
        mirrored = bool(grid.is_mirrored(axis))
        far = _far_reflect_rows(grid)[axis] if _stored_past_owned(grid, axis) else None
        return cls(
            origin=int(grid.origin_doubled(axis)),
            stored=int(grid.stored_cells(axis)),
            owned=int(grid.owned_cells(axis)),
            wraps=bool(grid.axis_wraps(axis)),
            invariant=invariant,
            near_ghost_row=MIRROR_SOURCE_INDEX if mirrored and not invariant else None,
            far_ghost_row=None if far is None else int(far),
        )


def grid_frames(grid: Any) -> Tuple[AxisFrame, AxisFrame, AxisFrame]:
    """The three axis frames of one grid."""
    return tuple(AxisFrame.of_grid(grid, axis) for axis in range(3))  # type: ignore[return-value]


def one_rank_line(n_cells: int, faces: Tuple[float, float], half_integer: bool,
                  wraps: bool) -> numpy.ndarray:
    """Per stored cell: is its sample OWNED by a one-rank chunk that CARRIES this axis's layer?

    MEEP selects the PML recurrence per structure chunk from ``sigsize``
    (step_db.cpp:56-59), and ``sigsize[d] > 1`` for every cell of a chunk that
    overlaps the layer — including cells whose own sigma is exactly zero. On one
    rank the chunk is the per-side effort volume ``structure::use_pml`` registers
    (structure.cpp:509-524)::

        pml_volume.set_num_direction(d, int(dx * user_volume.a + 1 + 0.5));

    ``span = int(cells + 1.5)`` array cells measured from the wall, where
    ``choose_chunkdivision`` splits. Ownership then follows MEEP's owned-corner
    arithmetic (vec.hpp:1100-1104, ``little_owned_corner0 = little_corner + 2 -
    iyee_shift``): a chunk spanning doubled indices ``[c0, c1]`` owns the integer
    samples ``c0 + 2 .. c1`` and the half-integer samples ``c0 + 1 .. c1 - 1``, so
    the INTEGER lattice plane on the chunk's own upper boundary belongs to it, not
    to its interior neighbour. Per Yee offset on this axis:

    * integer (shift 0): low face owns cells ``1 .. span``; high face owns the top
      ``span`` planes — stored cells ``n - span + 1 .. n - 1`` plus, on a wrapping
      axis, stored cell 0 (the image of the wall plane it owns).
    * half-integer (shift 1): low face owns cells ``0 .. span - 1``; high face
      ``n - span .. n - 1``.

    ``n_cells`` is the STORED count, as this rule has always been written. That
    departs from MEEP's own division in two places, both pinned by
    ``test_chunk_map.py`` rather than changed here (this rule is the one-rank default
    and changing it changes words): on a folded PERIODIC axis with a high-face layer
    the stored array runs one cell past MEEP's window, so the high run starts one
    row late on each lattice; and on a wrapping axis whose low run is clamped to the
    whole axis, the wrapped cell 0 reads False where MEEP's low chunk owns its image.

    Measured discriminators (12x12 um cell, res 20, PML(2) all faces, CW Ez at
    x = -5, whole-volume complex relative L2 against CPU MEEP): a deposit at
    y = -4.0 (cell 40, sigma exactly 0) and one at y = -3.975 (cells 40 + 41)
    both step the BOTH-ACTIVE recurrence, while y = -3.9 (cell 42) steps the
    dsig-only one (the absorber-placement evidence packet, §1.5-1.6). That is a
    boundary no reading of the cells' own sigma can see: it is zero at all three.
    """
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
                inside[0] = True  # The stored image of the high wall plane.
    return inside


def chunk_pml_flags(boxes: Sequence[Box], frames: Sequence[AxisFrame],
                    user_volume: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
                    registered_faces: Faces, resolution: float) -> List[Tuple[bool, bool, bool]]:
    """Per chunk and axis: does ``structure_chunk::use_pml`` allocate sigma (``sigsize > 1``)?

    Transcribed from structure.cpp:625-652: chunk ``[lo, hi]`` carries axis ``d`` iff
    some ``i`` in ``[lo_d, hi_d + 1]`` has ``pml_x(i, dx, bloc, a) > 0`` for a face MEEP
    installed on ``d``, ``bloc = user_volume.boundary_location(side, d)`` (the
    user-volume corner on that side times ``0.5 * (1/a)``, vec.cpp:692-719 and
    vec.hpp:1055). ``registered_faces`` is MEEP's own face table in cells; ``dx * (2 *
    a)`` is ``2 * cells`` exactly (scaling by a power of two), so the snapped extent is
    ``pml.half_cell_extent(cells)``, and the distance term is evaluated in IEEE double
    as MEEP writes it, ``(int)(fabs(bloc - here) * (2 * a) + 0.5)`` with ``here = i *
    0.5 / a``. An invariant axis carries nothing.
    """
    from .pml import half_cell_extent  # noqa: PLC0415

    a = float(resolution)
    scale = 0.5 * (1.0 / a)
    user_lo, user_hi = user_volume
    flags: List[Tuple[bool, bool, bool]] = []
    for lo, hi in boxes:
        found = []
        for axis in range(3):
            carries = False
            if not frames[axis].invariant:
                here = numpy.arange(lo[axis], hi[axis] + 2, dtype=numpy.float64) * 0.5 / a
                for side, wall in ((0, user_lo[axis]), (1, user_hi[axis])):
                    cells = registered_faces[axis][side]
                    if cells <= 0:
                        continue
                    extent = half_cell_extent(cells)  # (int)(dx * (2 * a) + 0.5)
                    bloc = float(wall) * scale
                    distance = (numpy.abs(bloc - here) * (2 * a) + 0.5).astype(numpy.int64)
                    if numpy.any(extent - distance > 0):
                        carries = True
                        break
            found.append(carries)
        flags.append((found[0], found[1], found[2]))
    return flags


def _true_runs(line: numpy.ndarray) -> List[Tuple[int, int]]:  # [start, stop) runs of True.
    padded = numpy.concatenate(([False], numpy.asarray(line, dtype=bool), [False]))
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]


def faces_equal(a: Faces, b: Faces) -> bool:  # Numeric comparison of two per-face tables.
    return all(float(a[axis][side]) == float(b[axis][side]) for axis in range(3) for side in range(2))


def _mirror_low_dropped(faces: Faces, frames: Sequence[AxisFrame]) -> Faces:
    """The package's face table: a mirrored axis keeps only its high face."""
    out = []
    for axis in range(3):
        low, high = faces[axis]
        out.append((0 if frames[axis].near_ghost_row is not None else low, high))
    return (out[0], out[1], out[2])


class ChunkLayout:
    """One chunk division: boxes, per-chunk per-axis PML flags, and the axis frames.

    ``boxes[k]`` is chunk ``k``'s ``(lo, hi)`` in MEEP's doubled coordinates (an
    invariant axis carries any constant), ``pml_flags[k][d]`` is
    ``sigsize[d] > 1`` for that chunk, ``frames`` maps stored indices onto the
    doubled lattice, and ``faces`` is the package's per-face thickness table the
    flags were derived for (a driver refuses a layout whose faces differ from its
    PML). ``source`` names the constructor (``"one_rank"``, ``"meep"`` or
    ``"record"``).

    Construction checks registration: on every resolved axis the boxes' union is
    exactly the frame's window ``[origin, origin + 2*owned]``, and the boxes
    partition the window (pairwise disjoint under ``owns``' half-open ``(lo, hi]``,
    volumes summing to the window's), as MEEP's chunks do. A layout whose boxes
    describe another grid, or leave a hole, is refused by name rather than mapped.
    """

    def __init__(self, boxes: Sequence[Box], pml_flags: Sequence[Sequence[bool]],
                 frames: Sequence[AxisFrame], faces: Faces = NO_FACES, source: str = "record",
                 lines: Optional[Dict[Tuple[int, bool], numpy.ndarray]] = None) -> None:
        if len(boxes) == 0 or len(boxes) != len(pml_flags):
            raise ChunkLayoutRefusal(
                f"a chunk layout needs one flag triple per box, got {len(boxes)} boxes and "
                f"{len(pml_flags)} flag triples")
        if len(frames) != 3:
            raise ChunkLayoutRefusal(f"a chunk layout needs three axis frames, got {len(frames)}")
        self.boxes: Tuple[Box, ...] = tuple(
            (tuple(int(v) for v in lo), tuple(int(v) for v in hi)) for lo, hi in boxes)  # type: ignore[misc]
        self.pml_flags: Tuple[Tuple[bool, bool, bool], ...] = tuple(
            tuple(bool(v) for v in flags) for flags in pml_flags)  # type: ignore[misc]
        self.frames: Tuple[AxisFrame, AxisFrame, AxisFrame] = tuple(frames)  # type: ignore[assignment]
        self.faces: Faces = tuple((faces[a][0], faces[a][1]) for a in range(3))  # type: ignore[assignment]
        self.source = source
        self._lines = dict(lines or {})
        self._cache: Dict[Any, Any] = {}
        for k, (lo, hi) in enumerate(self.boxes):
            if any(lo[a] > hi[a] for a in range(3)) or any(len(v) != 3 for v in (lo, hi)):
                raise ChunkLayoutRefusal(f"chunk {k} has an inverted or malformed box {lo}..{hi}")
            if any(self.frames[a].invariant and self.pml_flags[k][a] for a in range(3)):
                raise ChunkLayoutRefusal(
                    f"chunk {k} carries a PML flag on an invariant axis; MEEP installs no face "
                    f"there (boundary_region::apply, structure.cpp:208-218)")
        for axis, frame in enumerate(self.frames):
            if frame.invariant:
                continue
            lows = [lo[axis] for lo, _ in self.boxes]
            highs = [hi[axis] for _, hi in self.boxes]
            if (min(lows), max(highs)) != frame.window:
                raise ChunkLayoutRefusal(
                    f"the chunk boxes span doubled {AXIS_NAMES[axis]} [{min(lows)}, {max(highs)}] "
                    f"where this grid's window is {list(frame.window)}: the division describes "
                    f"another grid (a folded axis MEEP did not halve, or a registration slip), "
                    f"so no cell of it can be mapped onto this one")
        self._require_partition()

    def _require_partition(self) -> None:  # The boxes tile the window: disjoint, no hole.
        resolved = [axis for axis in range(3) if not self.frames[axis].invariant]
        lows = numpy.asarray([[lo[a] for a in resolved] for lo, _ in self.boxes], dtype=numpy.int64)
        highs = numpy.asarray([[hi[a] for a in resolved] for _, hi in self.boxes], dtype=numpy.int64)
        if not resolved:
            if len(self.boxes) != 1:
                raise ChunkLayoutRefusal(f"a grid with no resolved axis has one chunk, got {len(self.boxes)}")
            return
        overlap = numpy.ones((len(self.boxes), len(self.boxes)), dtype=bool)
        for k in range(len(resolved)):
            overlap &= (lows[:, None, k] < highs[None, :, k]) & (lows[None, :, k] < highs[:, None, k])
        numpy.fill_diagonal(overlap, False)
        if overlap.any():
            first, second = (int(v) for v in numpy.argwhere(overlap)[0])
            raise ChunkLayoutRefusal(
                f"chunks {first} and {second} overlap ({self.boxes[first]} and {self.boxes[second]}); "
                f"MEEP's chunks partition the grid")
        volume = int(numpy.prod(highs - lows, axis=1).sum())
        window = int(numpy.prod([self.frames[a].window[1] - self.frames[a].window[0] for a in resolved]))
        if volume != window:
            raise ChunkLayoutRefusal(
                f"no chunk owns part of the window: the boxes cover {volume} of its {window} doubled "
                f"volume units, and a division must cover its window")

    # --- constructors ---------------------------------------------------------------

    @classmethod
    def one_rank(cls, grid: Any, pml: Any = None) -> "ChunkLayout":
        """The one-rank division a driver built without MEEP steps with.

        The boxes are MEEP's one-rank ``{low, interior, high}`` product per axis: a
        side's effort volume spans ``int(cells + 1.5)`` cells from MEEP's wall
        (structure.cpp:509-524), so its box is ``[origin, origin + 2*span]`` or
        ``[big - 2*span, big]`` doubled. They serve :meth:`boundaries` and the
        per-chunk views. The membership lines are :func:`one_rank_line` — the
        package's one-rank rule, verbatim — so the default steps exactly as before;
        ``test_chunk_map.py`` pins where that rule and these boxes disagree.
        """
        frames = grid_frames(grid)
        faces = NO_FACES if pml is None else pml.thickness_by_face
        intervals: List[List[Tuple[int, int, bool]]] = []
        for axis, frame in enumerate(frames):
            if frame.invariant:
                intervals.append([(frame.origin, frame.origin, False)])
                continue
            lo_wall, hi_wall = frame.window
            low, high = faces[axis]
            low_hi = lo_wall + 2 * min(frame.owned, int(low + 1.5)) if low > 0 else lo_wall
            high_lo = hi_wall - 2 * min(frame.owned, int(high + 1.5)) if high > 0 else hi_wall
            if low_hi <= high_lo:
                pieces = [(lo_wall, low_hi, True), (low_hi, high_lo, False), (high_lo, hi_wall, True)]
            else:
                # The two effort volumes overlap; MEEP's add_to_effort_volumes cuts the
                # overlap into a chunk of its own, and every piece carries the layer.
                pieces = [(lo_wall, high_lo, True), (high_lo, low_hi, True), (low_hi, hi_wall, True)]
            intervals.append([piece for piece in pieces if piece[0] < piece[1]])
        boxes: List[Box] = []
        flags: List[Tuple[bool, bool, bool]] = []
        for px in intervals[0]:
            for py in intervals[1]:
                for pz in intervals[2]:
                    boxes.append(((px[0], py[0], pz[0]), (px[1], py[1], pz[1])))
                    flags.append((px[2], py[2], pz[2]))
        lines = {}
        for axis, frame in enumerate(frames):
            for half in (False, True):
                if frame.invariant:
                    lines[(axis, half)] = numpy.zeros(frame.stored, dtype=bool)
                else:
                    lines[(axis, half)] = one_rank_line(frame.stored, faces[axis], half, frame.wraps)
        return cls(boxes, flags, frames, faces=faces, source="one_rank", lines=lines)

    @classmethod
    def from_meep(cls, sim: Any, grid: Any, registered_faces: Optional[Faces]) -> "ChunkLayout":
        """The lifted simulation's own division, read after its ``init_sim``.

        Boxes are ``sim.structure.get_chunk_volumes()``'s corners (``r``/``z`` on a
        cylindrical grid, under the x = R, y = P, z = Z mapping). The flags transcribe
        ``structure_chunk::use_pml`` (structure.cpp:625-652): chunk ``k`` carries axis
        ``d`` iff some ``i`` in ``[lo_d, hi_d + 1]`` has ``pml_x(i, dx, bloc, a) > 0``
        for a face MEEP installed on ``d``, with ``bloc =
        user_volume.boundary_location(side, d)`` — the FULL cell's wall, which on a
        folded axis lies outside the halved grid. ``registered_faces`` is MEEP's own
        per-face table in cells (both faces of a folded axis; nothing on an invariant
        or one-cell axis — ``from_meep._resolve_pml_faces`` with no fold); ``None``
        means no face absorbs. ``dx * (2 * a)`` is ``2 * cells`` exactly (scaling by a
        power of two), so the snapped extent is ``pml.half_cell_extent(cells)``; the
        distance term is evaluated in IEEE double as MEEP writes it.
        """
        frames = grid_frames(grid)
        cylindrical = bool(getattr(grid, "cylindrical", False))
        structure = sim.structure

        def corner(vec: Any) -> Tuple[int, int, int]:
            if cylindrical:
                return (int(vec.r()), 0, int(vec.z()))
            return (int(vec.x()), int(vec.y()), int(vec.z()))

        def resolved(point: Tuple[int, int, int]) -> Tuple[int, int, int]:
            return tuple(0 if frames[axis].invariant else point[axis] for axis in range(3))  # type: ignore[return-value]

        boxes: List[Box] = [
            (resolved(corner(volume.little_corner())), resolved(corner(volume.big_corner())))
            for volume in structure.get_chunk_volumes()]
        user_volume = (corner(structure.user_volume.little_corner()),
                       corner(structure.user_volume.big_corner()))
        faces = registered_faces if registered_faces is not None else NO_FACES
        flags = chunk_pml_flags(boxes, frames, user_volume, faces, float(sim.resolution))
        return cls(boxes, flags, frames, faces=_mirror_low_dropped(faces, frames), source="meep")

    # --- capsule record -------------------------------------------------------------

    def to_record(self) -> Dict[str, Any]:
        """A JSON-ready description that :meth:`from_record` rebuilds exactly."""
        return {
            "format": "meep_gpu/chunk_layout/1",
            "source": self.source,
            "faces": [[float(v) for v in face] for face in self.faces],
            "frames": [dict(vars(frame)) for frame in self.frames],
            "boxes": [[list(lo), list(hi)] for lo, hi in self.boxes],
            "pml_flags": [list(flags) for flags in self.pml_flags],
            "lines": {f"{axis}:{int(half)}": [int(v) for v in line]
                      for (axis, half), line in sorted(self._lines.items())},
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "ChunkLayout":
        if record.get("format") != "meep_gpu/chunk_layout/1":
            raise ChunkLayoutRefusal(f"not a chunk layout record: format {record.get('format')!r}")
        lines = {}
        for key, values in record.get("lines", {}).items():
            axis, half = key.split(":")
            lines[(int(axis), bool(int(half)))] = numpy.asarray(values, dtype=bool)
        faces = tuple(tuple(face) for face in record["faces"])
        return cls([(tuple(lo), tuple(hi)) for lo, hi in record["boxes"]], record["pml_flags"],
                   [AxisFrame(**frame) for frame in record["frames"]], faces=faces,  # type: ignore[arg-type]
                   source=record["source"], lines=lines)

    # --- derived views --------------------------------------------------------------

    def _resolve(self, axis: int, half_integer: bool) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """(doubled coordinate after the periodic translation, owned) for every stored row."""
        frame = self.frames[axis]
        points = frame.doubled(half_integer)
        lo, hi = frame.window
        if frame.wraps:
            period = hi - lo
            points = numpy.where(points <= lo, points + period, points)
            points = numpy.where(points > hi, points - period, points)
        owned = (points > lo) & (points <= hi)
        return points, owned

    def owned(self, axis: int, half_integer: bool) -> numpy.ndarray:
        """Which stored rows MEEP owns on this axis and sub-lattice (``owns``, vec.cpp:445-462).

        A wrapping axis's cell 0 is owned through its periodic image; a metallic or
        mirror wall row, the cylindrical axis row and a folded periodic axis's far
        slot are not. An invariant axis's one row counts as owned.
        """
        key = ("owned", axis, bool(half_integer))
        if key not in self._cache:
            frame = self.frames[axis]
            if frame.invariant:
                self._cache[key] = numpy.ones(frame.stored, dtype=bool)
            else:
                self._cache[key] = self._resolve(axis, half_integer)[1]
        return self._cache[key]

    def _ghost_rows(self, axis: int, half_integer: bool) -> Dict[int, int]:  # ghost row -> image row
        frame = self.frames[axis]
        rows: Dict[int, int] = {}
        if frame.near_ghost_row is not None and not half_integer:
            rows[0] = frame.near_ghost_row
        if frame.far_ghost_row is not None and half_integer:
            rows[frame.stored - 1] = frame.far_ghost_row
        return rows

    def membership(self, axis: int, half_integer: bool) -> Optional[numpy.ndarray]:
        """Per stored row: does the chunk owning it carry this axis's layer?

        ``half_integer`` is the component's Yee sub-lattice ON THIS AXIS. Rows MEEP
        does not own read False, except the ghost rows, which read the row they
        image. ``None`` when the division is not separable along the axis: some
        owned doubled coordinate is covered by chunks that disagree on the flag, so
        no line can describe it (:meth:`membership_mask` can).
        """
        axis = int(axis)
        key = ("membership", axis, bool(half_integer))
        if key in self._cache:
            return self._cache[key]
        frame = self.frames[axis]
        if (axis, bool(half_integer)) in self._lines:
            line: Optional[numpy.ndarray] = numpy.array(self._lines[(axis, bool(half_integer))], dtype=bool)
        elif frame.invariant:
            line = numpy.zeros(frame.stored, dtype=bool)
        else:
            points, owned = self._resolve(axis, half_integer)
            any_true = numpy.zeros(points.size, dtype=bool)
            any_false = numpy.zeros(points.size, dtype=bool)
            for (lo, hi), flags in zip(self.boxes, self.pml_flags):
                covered = (points > lo[axis]) & (points <= hi[axis])
                if flags[axis]:
                    any_true |= covered
                else:
                    any_false |= covered
            if numpy.any(any_true & any_false & owned):
                line = None
            else:
                line = any_true & owned
                for row, image in self._ghost_rows(axis, half_integer).items():
                    if not owned[row]:
                        line[row] = line[image]
        self._cache[key] = line
        return line

    def is_separable(self) -> bool:
        """Every (axis, sub-lattice) has a membership line."""
        return not self.non_separable()

    def non_separable(self) -> List[Tuple[int, bool]]:
        """The (axis, half_integer) pairs with no membership line."""
        return [(axis, half) for axis in range(3) for half in (False, True)
                if self.membership(axis, half) is None]

    def membership_mask(self, axis: int, iyee: Sequence[int]) -> numpy.ndarray:
        """The full bool volume for one component lattice: does each cell's owner carry ``axis``?

        The general form, for divisions :meth:`membership` cannot describe. Each
        cell's doubled coordinate is resolved per axis (periodic translation; a ghost
        row takes its image row's coordinate; an unowned wall row on another axis
        takes the nearest owned row's) and looked up among the boxes with MEEP's
        ``owns``. A cell on an unowned wall row of ``axis`` itself reads False, as
        the line does. On a separable division this equals the line broadcast.
        """
        axis = int(axis)
        key = ("mask", axis, tuple(int(v) for v in iyee))
        if key in self._cache:
            return self._cache[key]
        if (axis, bool(iyee[axis])) in self._lines:
            line = self.membership(axis, bool(iyee[axis]))
            shape = [1, 1, 1]
            shape[axis] = line.size  # type: ignore[union-attr]
            full = tuple(frame.stored for frame in self.frames)
            mask = numpy.broadcast_to(line.reshape(shape), full).copy()  # type: ignore[union-attr]
            self._cache[key] = mask
            return mask
        flags = numpy.asarray([chunk_flags[axis] for chunk_flags in self.pml_flags], dtype=bool)
        mask = flags[self.owner_volume(iyee)]
        wall_rows = self._resolved_rows(iyee)[1][axis]
        if wall_rows is not None and wall_rows.any():
            shape = [1, 1, 1]
            shape[axis] = wall_rows.size
            mask &= ~numpy.broadcast_to(wall_rows.reshape(shape), mask.shape)
        self._cache[key] = mask
        return mask

    def _resolved_rows(self, iyee: Sequence[int]) -> Tuple[List[numpy.ndarray], List[Optional[numpy.ndarray]]]:
        """Per axis, the doubled coordinate each stored row is looked up at, and its wall rows.

        Periodic translation on a wrapping axis; a ghost row takes its image row's
        coordinate; any other row MEEP does not own (a wall row, the cylindrical axis
        row) takes the nearest owned row's. The second list holds, per resolved axis,
        the rows that are unowned walls (ghosts excluded); ``None`` on an invariant axis.
        """
        key = ("resolved", tuple(int(v) for v in iyee))
        if key in self._cache:
            return self._cache[key]
        coordinates: List[numpy.ndarray] = []
        walls: List[Optional[numpy.ndarray]] = []
        for other, frame in enumerate(self.frames):
            if frame.invariant:
                coordinates.append(numpy.zeros(frame.stored, dtype=numpy.int64))
                walls.append(None)
                continue
            points, owned = self._resolve(other, bool(iyee[other]))
            points = points.copy()
            for row, image in self._ghost_rows(other, bool(iyee[other])).items():
                if not owned[row]:
                    points[row] = points[image]
                    owned = owned.copy()
                    owned[row] = True
            walls.append(~owned)
            unowned = numpy.flatnonzero(~owned)
            if unowned.size:
                owned_at = numpy.flatnonzero(owned)
                for row in unowned:
                    nearest = owned_at[numpy.argmin(numpy.abs(owned_at - row))]
                    points[row] = points[nearest]
            coordinates.append(points)
        self._cache[key] = (coordinates, walls)
        return coordinates, walls

    def owner_volume(self, iyee: Sequence[int]) -> numpy.ndarray:
        """Per stored cell of one component lattice: the index of the chunk that owns it.

        Each cell's doubled coordinate is resolved per axis as :meth:`membership_mask`
        resolves it (:meth:`_resolved_rows`) and looked up among the boxes with MEEP's
        ``owns`` (``lo < p <= hi``, vec.cpp:445-462). Every per-chunk fact a cell steps
        with (its PML flags, whether its chunk keeps a conductivity) is read through
        this one volume. Refused by name when a resolved coordinate is owned by no chunk.
        """
        key = ("owner", tuple(int(v) for v in iyee))
        if key in self._cache:
            return self._cache[key]
        coordinates, _walls = self._resolved_rows(iyee)
        grids = numpy.meshgrid(*coordinates, indexing="ij")
        owner = numpy.full(grids[0].shape, -1, dtype=numpy.int64)
        for k, (lo, hi) in enumerate(self.boxes):
            hit = owner < 0
            for other, frame in enumerate(self.frames):
                if not frame.invariant:
                    hit &= (grids[other] > lo[other]) & (grids[other] <= hi[other])
            owner[hit] = k
        if numpy.any(owner < 0):
            raise ChunkLayoutRefusal("a cell's resolved coordinate is owned by no chunk")
        self._cache[key] = owner
        return owner

    def loop_rows(self, axis: int, half_integer: bool, chunk: int) -> numpy.ndarray:
        """The stored rows of the samples MEEP's ``LOOP_OVER_VOL`` visits along one axis of a chunk.

        ``LOOP_OVER_VOL(gv, c, i)`` runs from ``little_corner + iyee_shift`` to
        ``big_corner + iyee_shift`` (vec.hpp:170-172): for a component at shift ``s`` on
        this axis, the doubled coordinates ``lo + s, lo + s + 2, ..., hi + s`` of the
        chunk's box ``[lo, hi]``. That is the chunk's owned samples plus ONE more: below
        them on the integer lattice (``lo``, owned by the chunk beneath it or a wall),
        above them on the half-integer one (``hi + 1``, owned by the chunk above it).
        ``structure_chunk::set_conductivity`` keeps or deletes a chunk's conductivity
        array from exactly these samples (structure.cpp:883-900).

        Each sample is read at the stored row of its doubled coordinate
        (:meth:`AxisFrame.doubled`). On a wrapping axis a sample at or past the window's
        top is read at its periodic image, one period ``2 * owned`` lower: the integer
        ``big_corner`` plane, which the top chunk owns, is stored row 0, and so is the
        half-integer sample half a cell above it (INFERRED from the periodic geometry
        MEEP evaluates there; no MEEP leg divides a wrapping axis with a conductivity).
        On an axis that does not wrap, a sample past the last stored row -- the top wall
        plane, or the half-integer sample half a cell beyond the window's top -- has no
        stored row and is read at the last one. That is an approximation (OPEN): MEEP
        evaluates the material at that point, so a conductivity that begins within the
        last half cell below the top wall and extends outward is not seen. An invariant
        axis has one row.
        """
        frame = self.frames[axis]
        if frame.invariant:
            return numpy.arange(frame.stored, dtype=numpy.int64)
        shift = 1 if half_integer else 0
        lo, hi = self.boxes[chunk][0][axis], self.boxes[chunk][1][axis]
        coordinates = numpy.arange(lo + shift, hi + shift + 1, 2, dtype=numpy.int64)
        offsets = coordinates - frame.origin - shift
        if numpy.any(offsets % 2):
            raise ChunkLayoutRefusal(
                f"chunk {chunk}'s box {lo}..{hi} on {AXIS_NAMES[axis]} is off the doubled lattice "
                f"of origin {frame.origin}: its corners are not integer-lattice points")
        rows = offsets // 2
        if frame.wraps:
            if frame.stored != frame.owned:
                raise ChunkLayoutRefusal(
                    f"axis {AXIS_NAMES[axis]} wraps with {frame.stored} stored rows and "
                    f"{frame.owned} owned: no periodic image row for its top samples")
            return rows % frame.owned
        return numpy.clip(rows, 0, frame.stored - 1)

    def conductive_chunks(self, sigma: Any, iyee: Sequence[int]) -> Tuple[bool, ...]:
        """Per chunk: does MEEP keep a conductivity array for this component there?

        ``structure_chunk::set_conductivity`` samples the conductivity over
        ``LOOP_OVER_VOL(gv, c, i)`` -- the chunk's owned samples plus one
        (:meth:`loop_rows`) -- and deletes the chunk's array, leaving
        ``s->conductivity[c][d]`` NULL, when every sample is exactly zero
        (structure.cpp:883-900). A chunk without one steps the non-conductive PML forms
        (step_generic.cpp:104, :138-141, :182-183, :231-234) even where the component is
        lossy elsewhere. ``sigma`` is the component's conductivity volume on this
        layout's stored shape (host or device array); ``iyee`` its Yee shifts.
        """
        from .backends import to_numpy  # noqa: PLC0415

        values = numpy.asarray(to_numpy(sigma))
        stored = tuple(frame.stored for frame in self.frames)
        if values.shape != stored:
            raise ChunkLayoutRefusal(
                f"a conductivity of shape {values.shape} does not lie on this layout's stored "
                f"shape {stored}")
        present = []
        for k in range(len(self.boxes)):
            rows = [self.loop_rows(axis, bool(iyee[axis]), k) for axis in range(3)]
            present.append(bool(numpy.any(values[numpy.ix_(*rows)] != 0)))
        return tuple(present)

    def conductive_mask(self, sigma: Any, iyee: Sequence[int]) -> numpy.ndarray:
        """Per stored cell: does the chunk owning it keep a conductivity for this component?

        :meth:`conductive_chunks` read through :meth:`owner_volume`: a full bool volume,
        because the presence is a per-chunk fact and not a per-axis one.
        """
        present = numpy.asarray(self.conductive_chunks(sigma, iyee), dtype=bool)
        return present[self.owner_volume(iyee)]

    # --- off-diagonal chi1inv per chunk ----------------------------------------------

    def chunk_samples(self, axis: int) -> Tuple[numpy.ndarray, ...]:
        """Per chunk (indexed like :attr:`boxes`), which stored rows of ``axis`` it ALLOCATES.

        A structure chunk evaluates and stores a material array at every sample of its
        own grid volume, ``LOOP_OVER_VOL`` (meep/vec.hpp:170-172): doubled coordinates
        ``little_corner + iyee_shift(c)`` to ``big_corner + iyee_shift(c)`` inclusive,
        one sample past the rows it owns. Stored row ``i`` of a component with Yee shift
        ``s`` sits at doubled ``origin + 2*i + s``, so the shift cancels: chunk ``k``
        allocates row ``i`` exactly when ``lo_k <= origin + 2*i <= hi_k``, on either
        sub-lattice. A wrapping axis compares the row's translated coordinate
        (:meth:`_resolve`); the row standing for the high wall there is also the image
        of the low wall, which a chunk starting at the low wall allocates. Rows outside
        the window (a far slot) are allocated by no chunk. An invariant axis has one row,
        allocated by every chunk.
        """
        axis = int(axis)
        key = ("samples", axis)
        if key in self._cache:
            return self._cache[key]
        frame = self.frames[axis]
        if frame.invariant:
            lines = tuple(numpy.ones(frame.stored, dtype=bool) for _ in self.boxes)
        else:
            points = self._resolve(axis, False)[0]
            lo_wall, hi_wall = frame.window
            out = []
            for lo, hi in self.boxes:
                line = (points >= lo[axis]) & (points <= hi[axis])
                if frame.wraps and lo[axis] == lo_wall:
                    line = line | (points == hi_wall)
                out.append(line)
            lines = tuple(out)
        self._cache[key] = lines
        return lines

    def offdiag_kept(self, coefficient: Any) -> Tuple[bool, ...]:
        """Per chunk (indexed like :attr:`boxes`): does MEEP keep this off-diagonal chi1inv array?

        ``structure_chunk::set_chi1inv`` deletes an off-diagonal ``chi1inv[c][d]`` in a
        chunk when every sample that chunk allocates is exactly zero
        (anisotropic_averaging.cpp:245-292; ``-0.0 == 0``), and ``update_eh`` then passes
        NULL for it (update_eh.cpp:191-196), so ``step_update_EDHB`` drops that OFFDIAG
        term in that chunk rather than adding a zero (step_generic.cpp:568-583).
        ``coefficient`` is the row component's stored array (host or device; read on
        the host). A sample no stored row holds is not read: on a non-wrapping axis
        whose stored rows end below the high wall plane, a chunk ending at the wall
        allocates that plane and this reads it as zero. Only a coefficient that is
        nonzero on that plane and zero on every stored sample of the same chunk is
        then read as dropped where MEEP keeps it.
        """
        from .backends import to_numpy  # noqa: PLC0415

        nonzero = numpy.asarray(to_numpy(coefficient)) != 0
        full = tuple(frame.stored for frame in self.frames)
        if nonzero.shape != full:
            raise ChunkLayoutRefusal(
                f"an off-diagonal coefficient of shape {nonzero.shape} is not this layout's stored "
                f"grid {full}")
        samples = [self.chunk_samples(axis) for axis in range(3)]
        kept = []
        for k in range(len(self.boxes)):
            rows = [numpy.flatnonzero(samples[axis][k]) for axis in range(3)]
            kept.append(bool(nonzero[numpy.ix_(*rows)].any()) if all(r.size for r in rows) else False)
        return tuple(kept)

    def offdiag_presence(self, coefficient: Any, iyee: Sequence[int]) -> Optional[numpy.ndarray]:
        """Per cell of a component lattice: does its owning chunk keep this off-diagonal array?

        :meth:`offdiag_kept` per chunk, read through :meth:`owner_volume` (the same
        owner :meth:`membership_mask` reads; ghost and unowned rows as there).
        ``None`` when every chunk keeps it, so the caller adds the term everywhere. A
        host bool volume of the stored shape otherwise. A chunk that drops the array
        holds an identically zero coefficient over every sample its owned cells read
        (``u[i]`` and ``u[i + s]``), so presence decides only whether a zero term is
        ADDED, which moves a ``-0`` accumulator to ``+0`` and nothing else.
        """
        kept = self.offdiag_kept(coefficient)
        if all(kept):
            return None
        return numpy.asarray(kept, dtype=bool)[self.owner_volume(iyee)]

    def chunk_owns(self, chunk: Any, coordinates: Sequence[Any], cylindrical: bool = False) -> numpy.ndarray:
        """Does chunk ``chunk`` own each point? MEEP's ``grid_volume::owns`` (vec.cpp:445-462).

        ``chunk`` is a chunk index per point (an integer array broadcastable against the
        points; ``-1`` owns nothing). ``coordinates`` holds one broadcastable integer
        array per axis, doubled coordinates on this layout's frames; an invariant axis is
        not tested. Ownership is ``lo < p <= hi`` on every resolved axis; on a
        cylindrical grid (package x is r) a point on the axis, doubled ``r = 0``, is also
        owned by a chunk whose box starts there (vec.cpp:448-450).
        """
        index = numpy.asarray(chunk, dtype=numpy.int64)
        lows = numpy.asarray([lo for lo, _ in self.boxes], dtype=numpy.int64)
        highs = numpy.asarray([hi for _, hi in self.boxes], dtype=numpy.int64)
        valid = index >= 0
        safe = numpy.where(valid, index, 0)
        owned = valid.copy()
        for axis, frame in enumerate(self.frames):
            if frame.invariant:
                continue
            points = numpy.asarray(coordinates[axis], dtype=numpy.int64)
            lo, hi = lows[safe, axis], highs[safe, axis]
            inside = (points > lo) & (points <= hi)
            if cylindrical and axis == 0:
                inside = inside | ((points == lo) & (lo == 0))
            owned = owned & inside
        return owned

    def chunk_of(self, coordinates: Sequence[Any], cylindrical: bool = False) -> numpy.ndarray:
        """The chunk that owns each point (:meth:`chunk_owns`), or ``-1`` where none does.

        The boxes partition the window (checked at construction), so at most one chunk
        owns a point. The result has the broadcast shape of the per-axis coordinates.
        """
        arrays = [numpy.asarray(coordinates[axis], dtype=numpy.int64)
                  for axis in range(3) if not self.frames[axis].invariant]
        shape = numpy.broadcast_shapes(*[a.shape for a in arrays]) if arrays else ()
        out = numpy.full(shape, -1, dtype=numpy.int64)
        for k in range(len(self.boxes)):
            hit = self.chunk_owns(numpy.full(shape, k, dtype=numpy.int64), coordinates, cylindrical)
            if numpy.any(hit & (out >= 0)):
                raise ChunkLayoutRefusal(f"a point is owned by chunk {k} and by another; chunks partition the grid")
            out = numpy.where(hit, k, out)
        return out

    def boundaries(self, axis: int) -> Optional[Tuple[int, ...]]:
        """The doubled coordinates where chunks are cut along ``axis``, window ends included.

        ``None`` when some cut is local — a box face along ``axis`` that does not span
        the whole cross-section — so the axis has no set of global cuts (read
        :attr:`boxes`).
        """
        cuts = sorted({lo[axis] for lo, _ in self.boxes} | {hi[axis] for _, hi in self.boxes})
        for value in cuts:
            for lo, hi in self.boxes:
                if lo[axis] < value < hi[axis]:
                    return None
        return tuple(int(v) for v in cuts)


def chunk_bounds_record(layout: ChunkLayout) -> numpy.ndarray:
    """The kernels' record: ``int32[25]``, ``[sub-lattice][axis][a0, a1, b0, b1]`` + presence.

    Membership of stored row ``i`` is ``(a0 <= i < a1) or (b0 <= i < b1)``; an empty run
    is ``[0, 0)``. Word 24 has bit ``a`` set when axis ``a`` is not invariant (a curl
    partner along it is present, ``fields.cpp:428-453``). Refused by name: a division
    that is not separable on some (axis, sub-lattice), a line with more than two true
    runs after the ghost adjustment, and any line the recomputed formula does not
    reproduce exactly.
    """
    record = numpy.zeros(RECORD_WORDS, dtype=numpy.int32)
    presence = 0
    for axis, frame in enumerate(layout.frames):
        if not frame.invariant:
            presence |= 1 << axis
        for half in (False, True):
            line = layout.membership(axis, half)
            if line is None:
                raise ChunkLayoutRefusal(
                    f"the chunk division is not separable along {AXIS_NAMES[axis]} on the "
                    f"{'half-integer' if half else 'integer'} lattice, so no [a0,a1) U [b0,b1) record "
                    f"describes it")
            runs = _true_runs(line)
            if len(runs) > 2:
                raise ChunkLayoutRefusal(
                    f"the {AXIS_NAMES[axis]} membership on the {'half-integer' if half else 'integer'} "
                    f"lattice has {len(runs)} runs {runs}; the record holds two")
            words = [0, 0, 0, 0]
            for slot, (start, stop) in enumerate(runs):
                words[2 * slot], words[2 * slot + 1] = start, stop
            index = numpy.arange(line.size)
            recomputed = (((index >= words[0]) & (index < words[1]))
                          | ((index >= words[2]) & (index < words[3])))
            if not numpy.array_equal(recomputed, line):
                raise ChunkLayoutRefusal(
                    f"the {AXIS_NAMES[axis]} record {words} does not reproduce its membership line")
            for word in range(4):
                record[record_index(half, axis, word)] = words[word]
    record[RECORD_PRESENCE_WORD] = presence
    return record


def membership_from_record(record: Sequence[int], half_integer: bool, axis: int,
                           n_cells: int) -> numpy.ndarray:
    """Decode one line of :func:`chunk_bounds_record` (the kernels' formula, on the host)."""
    a0, a1, b0, b1 = (int(record[record_index(half_integer, axis, word)]) for word in range(4))
    index = numpy.arange(n_cells)
    return ((index >= a0) & (index < a1)) | ((index >= b0) & (index < b1))


# --- region plan ----------------------------------------------------------------------

#: How a backend evaluates one component's two forms.
#: ``outside_only``: the membership line is all False — one form, whole grid.
#: ``inside_only``: all True — the other form, whole grid.
#: ``split``: each form on its own views of the split axis.
#: ``select``: both forms whole-grid, then a select on the line (NumPy, innermost axis).
#: ``mask``: both forms whole-grid, then a select on the full membership volume (a
#: division that is not separable, or a plan built with ``route="mask"``).
REGION_MODES = ("outside_only", "inside_only", "split", "select", "mask")

#: The routes a plan can be forced onto. Every form is pointwise and every route
#: stores the same word in every array, so the route is a cost choice; forcing one
#: is how the equality of the three is measured (``test_region_dispatch.py``).
REGION_ROUTES = ("split", "select", "mask")


class RegionSpec(NamedTuple):
    """One component's form regions along its split axis.

    ``sub_step`` is ``"curl"`` (split axis: the component's dsigu axis) or
    ``"constitutive"`` (its own axis, dsigw). ``inside`` / ``outside`` are the
    ``[start, stop)`` runs of the membership line and of its complement; ``view`` is
    ``"none"``, ``"slice"`` (one inside run), ``"pair"`` (two equal-width inside runs
    at both ends) or ``"slices"``. A ``"pair"`` is evaluated as two slices; the
    single ``(2, s)`` strided view is a launch-count detail for the CuPy timing work,
    and stores the same words.
    """

    sub_step: str
    component: str
    axis: int
    half_integer: bool
    inside: Tuple[Tuple[int, int], ...]
    outside: Tuple[Tuple[int, int], ...]
    view: str
    mode: str


def _innermost_resolved_axis(frames: Sequence[AxisFrame]) -> Optional[int]:
    resolved = [axis for axis, frame in enumerate(frames) if not frame.invariant and frame.stored > 1]
    return resolved[-1] if resolved else None


def _complement(runs: Sequence[Tuple[int, int]], n: int) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    cursor = 0
    for start, stop in runs:
        if start > cursor:
            out.append((cursor, start))
        cursor = stop
    if cursor < n:
        out.append((cursor, n))
    return tuple(out)


def _view(runs: Sequence[Tuple[int, int]], n: int) -> str:
    if not runs:
        return "none"
    if len(runs) == 1:
        return "slice"
    if (len(runs) == 2 and runs[0][0] == 0 and runs[1][1] == n
            and runs[0][1] - runs[0][0] == runs[1][1] - runs[1][0]):
        return "pair"
    return "slices"


def _coefficient_vector(pml: Any, kind: str, axis: int, half_integer: bool) -> numpy.ndarray:
    from .backends import to_numpy  # noqa: PLC0415

    suffix = "_h" if half_integer else ""
    return numpy.asarray(to_numpy(getattr(pml, f"{kind}_{AXIS_NAMES[axis]}{suffix}"))).reshape(-1)


def _refuse_coefficient(kind: str, axis: int, half_integer: bool, rows: numpy.ndarray,
                        vector: numpy.ndarray, where: str) -> None:
    name = f"{kind}_{AXIS_NAMES[axis]}{'_h' if half_integer else ''}"
    row = int(rows[0])
    raise ChunkLayoutRefusal(
        f"{name}[{row}] = {float(vector[row])!r} on {where} outside every chunk that carries "
        f"the {AXIS_NAMES[axis]} layer ({rows.size} such rows): the form outside the run "
        f"would not be MEEP's plain form there. The layer and the chunk division disagree.")


def check_unit_coefficients(layout: ChunkLayout, pml: Any) -> None:
    """Every PML coefficient is exactly ``1.0`` on every owned row outside its membership run.

    The precondition under which the dsig-only curl form equals MEEP's plain form bit
    for bit (``kms = sinv = 1.0f`` makes ``((kms*f) - curl)*sinv`` the word ``f - curl``,
    ±0 and subnormals included) and under which the constitutive keys on dsigw alone.
    Checked on ``kms``, ``sinv`` and ``kps`` of each (axis, sub-lattice); rows MEEP
    does not own (walls, the cylindrical axis, ghosts) are not stepped by either form
    and are not checked. A line the division cannot describe (not separable) is
    checked per cell by :func:`check_unit_coefficients_masked` when the plan builds
    its mask route. Refused by name with the first offending row.
    """
    if pml is None:
        return
    for axis, frame in enumerate(layout.frames):
        if frame.invariant:
            continue
        for half in (False, True):
            line = layout.membership(axis, half)
            if line is None:
                continue  # The mask route checks per cell when it is built.
            outside = layout.owned(axis, half) & ~line
            for kind in ("kms", "sinv", "kps"):
                vector = _coefficient_vector(pml, kind, axis, half)
                bad = numpy.flatnonzero(outside & (vector != numpy.float32(1.0)))
                if bad.size:
                    _refuse_coefficient(kind, axis, half, bad, vector, "a row")


def check_unit_coefficients_masked(layout: ChunkLayout, pml: Any, axis: int,
                                   iyee: Sequence[int]) -> None:
    """The unit-coefficient precondition for one component lattice, cell by cell.

    The general form of :func:`check_unit_coefficients`, for a division whose
    membership along ``axis`` is not a line: a coefficient row of ``axis`` must read
    exactly ``1.0`` wherever ANY owned cell of that row, on the component lattice
    ``iyee``, lies outside the full membership mask.
    """
    if pml is None or layout.frames[axis].invariant:
        return
    mask = layout.membership_mask(axis, iyee)
    owned = numpy.ones(mask.shape, dtype=bool)
    for other, frame in enumerate(layout.frames):
        line = layout.owned(other, bool(iyee[other]))
        shape = [1, 1, 1]
        shape[other] = line.size
        owned = owned & line.reshape(shape)
    others = tuple(a for a in range(3) if a != axis)
    rows = numpy.any(owned & ~mask, axis=others)
    half = bool(iyee[axis])
    for kind in ("kms", "sinv", "kps"):
        vector = _coefficient_vector(pml, kind, axis, half)
        bad = numpy.flatnonzero(rows & (vector != numpy.float32(1.0)))
        if bad.size:
            _refuse_coefficient(kind, axis, half, bad, vector, "a cell of a row")


def region_backend(xp: Any) -> str:
    """``"cupy"`` for the CuPy module, ``"numpy"`` for everything else (host arrays)."""
    return "cupy" if getattr(xp, "__name__", "").split(".")[0] == "cupy" else "numpy"


class RegionPlan:
    """Per curl and constitutive component: its split axis, its runs, and the route a backend uses.

    Built from a layout, a grid and a PML; :func:`check_unit_coefficients` runs first,
    and :func:`check_unit_coefficients_masked` for every component that takes the
    mask route. ``backend`` is ``"numpy"`` or ``"cupy"``: CuPy splits every
    separable axis (a ufunc on a view is one launch); NumPy selects on the innermost
    resolved axis, where views would make inner rows shorter than the grid's own,
    and splits elsewhere. ``route`` forces every component onto one route
    (:data:`REGION_ROUTES`) — the measurement that the routes agree — and is
    ``None`` in every run, which takes the backend's choice. Every form is pointwise,
    so the route is a cost choice only: the three store the same word in every
    array.

    The plan is the one place the stepping code reads membership from. Its masks
    are cached per array module: :meth:`mask` for a component's own split,
    :meth:`axis_mask` for another axis on the component's lattice (the conductive
    curl tests dsig as well as dsigu), and :meth:`conductive_mask` for the chunks that
    keep a component's conductivity (the conductive curl's third test).
    """

    def __init__(self, layout: ChunkLayout, grid: Any, pml: Any, backend: str = "numpy",
                 route: Optional[str] = None) -> None:
        if backend not in ("numpy", "cupy"):
            raise ChunkLayoutRefusal(f"backend must be 'numpy' or 'cupy', got {backend!r}")
        if route is not None and route not in REGION_ROUTES:
            raise ChunkLayoutRefusal(f"route must be one of {REGION_ROUTES} or None, got {route!r}")
        from .fields import IYEE_SHIFTS  # noqa: PLC0415
        from .stepping import (B_CURL_TERMS, D_CURL_TERMS, E_CONSTITUTIVE_TERMS,  # noqa: PLC0415
                               H_CONSTITUTIVE_TERMS)

        if grid_frames(grid) != layout.frames:
            raise ChunkLayoutRefusal("the region plan's grid is not the grid the layout was built for")
        check_unit_coefficients(layout, pml)
        self.layout = layout
        self.grid = grid
        self.pml = pml
        self.backend = backend
        self.route = route
        self._masks: Dict[Any, Any] = {}
        self._conductive: Dict[str, Tuple[Any, Tuple[bool, ...], Dict[Any, Any]]] = {}
        innermost = _innermost_resolved_axis(layout.frames)
        specs: Dict[Tuple[str, str], RegionSpec] = {}
        entries: List[Tuple[str, str, int, Tuple[int, ...]]] = []
        for term in (*B_CURL_TERMS, *D_CURL_TERMS):
            others = (AXIS_NAMES.index(term.dsig),)
            entries.append(("curl", term.target, AXIS_NAMES.index(term.dsigu), others))
        for name, _source, own in (*H_CONSTITUTIVE_TERMS, *E_CONSTITUTIVE_TERMS):
            entries.append(("constitutive", name, AXIS_NAMES.index(own), ()))
        for sub_step, component, axis, others in entries:
            iyee = IYEE_SHIFTS[component]
            half = iyee[axis] == 1
            n = layout.frames[axis].stored
            line = layout.membership(axis, half)
            for checked in (axis, *others):
                if layout.membership(checked, iyee[checked] == 1) is None:
                    check_unit_coefficients_masked(layout, pml, checked, iyee)
            if line is None or route == "mask":
                inside = () if line is None else tuple(_true_runs(line))
                outside = () if line is None else _complement(inside, n)
                specs[(sub_step, component)] = RegionSpec(
                    sub_step, component, axis, half, inside, outside,
                    "none" if line is None else _view(inside, n), "mask")
                continue
            inside = tuple(_true_runs(line))
            outside = _complement(inside, n)
            if not inside:
                mode = "outside_only"
            elif not outside:
                mode = "inside_only"
            elif route is not None:
                mode = route
            elif backend == "numpy" and axis == innermost:
                mode = "select"
            else:
                mode = "split"
            specs[(sub_step, component)] = RegionSpec(sub_step, component, axis, half, inside, outside,
                                                      _view(inside, n), mode)
        self.specs = specs

    def spec(self, sub_step: str, component: str) -> RegionSpec:
        return self.specs[(sub_step, component)]

    def mask(self, sub_step: str, component: str, xp: Any = numpy, outside: bool = False) -> Any:
        """The inside (or ``outside``) cells of one component's split, broadcastable to the field.

        A line reshaped onto its axis where the spec selects on the line, the full
        membership volume where it takes the mask route.
        """
        from .fields import IYEE_SHIFTS  # noqa: PLC0415

        spec = self.spec(sub_step, component)
        return self._mask(spec.axis, tuple(IYEE_SHIFTS[component]), spec.mode == "mask", xp, outside)

    def axis_mask(self, axis: int, component: str, xp: Any = numpy) -> Any:
        """Membership of ``axis``'s layer on ``component``'s lattice, broadcastable to the field."""
        from .fields import IYEE_SHIFTS  # noqa: PLC0415

        iyee = tuple(IYEE_SHIFTS[component])
        full = self.route == "mask" or self.layout.membership(axis, iyee[axis] == 1) is None
        return self._mask(int(axis), iyee, full, xp, False)

    def _mask(self, axis: int, iyee: Tuple[int, ...], full: bool, xp: Any, outside: bool) -> Any:
        lattice = iyee if full else bool(iyee[axis])
        key = (axis, lattice, full, outside, getattr(xp, "__name__", repr(xp)))
        hit = self._masks.get(key)
        if hit is None:
            if full:
                host = self.layout.membership_mask(axis, iyee)
            else:
                line = self.layout.membership(axis, bool(iyee[axis]))
                shape = [1, 1, 1]
                shape[axis] = line.size  # type: ignore[union-attr]
                host = line.reshape(shape)  # type: ignore[union-attr]
            if outside:
                host = ~host
            hit = xp.asarray(numpy.ascontiguousarray(host))
            self._masks[key] = hit
        return hit

    def inside_at(self, sub_step: str, component: str, shape: Tuple[int, ...], cells: Any,
                  xp: Any = numpy) -> Any:
        """Per listed cell (a tuple of index arrays into a volume of ``shape``): inside the split?"""
        return xp.broadcast_to(self.mask(sub_step, component, xp), shape)[cells]

    def conductive_chunks(self, component: str, sigma: Any) -> Tuple[bool, ...]:
        """Per chunk: does it keep a conductivity for ``component``? (:meth:`ChunkLayout.conductive_chunks`)

        Cached per component against the identity of ``sigma``: installing a new
        conductivity volume (``Fields._set_conductivity_side`` builds new arrays) is a
        new entry, never a stale one.
        """
        from .fields import IYEE_SHIFTS  # noqa: PLC0415

        hit = self._conductive.get(component)
        if hit is None or hit[0] is not sigma:
            hit = (sigma, self.layout.conductive_chunks(sigma, IYEE_SHIFTS[component]), {})
            self._conductive[component] = hit
        return hit[1]

    def conductive_mask(self, component: str, sigma: Any, xp: Any = numpy, outside: bool = False) -> Any:
        """The cells whose chunk keeps a conductivity for ``component`` (or, ``outside``, does not).

        A full volume in ``xp``'s arrays (:meth:`ChunkLayout.conductive_mask`), cached with
        :meth:`conductive_chunks`.
        """
        from .fields import IYEE_SHIFTS  # noqa: PLC0415

        self.conductive_chunks(component, sigma)
        masks = self._conductive[component][2]
        key = (outside, getattr(xp, "__name__", repr(xp)))
        mask = masks.get(key)
        if mask is None:
            host = self.layout.conductive_mask(sigma, IYEE_SHIFTS[component])
            mask = xp.asarray(numpy.ascontiguousarray(~host if outside else host))
            masks[key] = mask
        return mask


def announce_non_separable(layout: ChunkLayout, stream: Any = None) -> Optional[str]:
    """Print the named notice when a lifted division is not separable; return it, or None."""
    missing = layout.non_separable()
    if not missing:
        return None
    where = ", ".join(f"{AXIS_NAMES[axis]} ({'half-integer' if half else 'integer'})"
                      for axis, half in missing)
    text = NON_SEPARABLE_NOTICE.format(where=where)
    print(text, file=stream if stream is not None else sys.stderr, flush=True)
    return text


__all__ = [
    "AxisFrame", "ChunkLayout", "ChunkLayoutRefusal", "RegionPlan", "RegionSpec",
    "REGION_MODES", "REGION_ROUTES", "RECORD_WORDS", "RECORD_BOUNDS_WORDS", "RECORD_PRESENCE_WORD",
    "NO_FACES", "announce_non_separable", "check_unit_coefficients",
    "check_unit_coefficients_masked", "chunk_bounds_record",
    "chunk_pml_flags", "faces_equal", "grid_frames",
    "membership_from_record", "one_rank_line", "record_index", "region_backend",
]
