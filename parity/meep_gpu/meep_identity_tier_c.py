#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tier C of MEEP identity: the lifted driver against the reference MEEP, word for word.

    python parity/meep_gpu/meep_identity_tier_c.py compare --out DIR [--mode keep|flush]
        [--cases a,b,c,c_prime] [--cases-file PATH] [--steps N] [--rows FILE]
    python parity/meep_gpu/meep_identity_tier_c.py replay --source DIR [--mode keep|flush]
        [--cases ...] [--rows FILE]

WHAT IT ANSWERS. Does the package's array path store MEEP's words? ``compare`` is the
only tier that steps MEEP (``meep_gpu/meep_identity.py`` states the three tiers). It runs
where an identity-reference MEEP is installed (``tools/check_install.py
--require-identity-reference``) and writes what Tier A replays; ``replay`` is Tier A on
such a directory, in a process that cannot import MEEP.

COMPARE, per case, each in a process of its own:

1. NumPy, then MEEP, is imported; the thread is driven to the mode
   (``subnormal_policy._drive_host_fpu``: MEEP's own ``set_zero_subnormals`` first, and
   where that has no effect, on arm64, the floating-point environment
   ``FE_DFL_DISABLE_DENORMS_ENV``, FPCR.FZ) and the FPU is measured; a mode the host
   cannot reach is refused by name. ``keep`` is the reference of both hosts; ``flush``
   is MEEP's x86 flush-on-import at ``OMP_NUM_THREADS=1`` (refused at any other count),
   and on arm64 MEEP stepped under FZ, which stores the x86 flush reference's words
   (R13; phase0/P0-d1m, 0 of 72,801,288 differing words).
2. The case is built and lifted on the NumPy reference under
   ``capsule.recording_construction``; the capsule is written at step 0 with the
   simulation, so it carries the lifted chunk division. The cases are the four of
   ``meep_identity_smoke.py``, or those of ``--cases-file``: a module with
   ``build(mp, label)`` (or ``BUILDERS[label](mp)``) and ``CASES``, and optionally
   ``steps_planned(mp, sim)`` in place of Table 2's step rule and
   ``after_lift(mp, sim, driver)``, whose dict is recorded as ``case_facts`` (a raise is
   the case's error).
3. PAIR ORDER: at every compared step (``meep_identity.compared_steps``; N is Table 2's
   step rule unless ``--steps``) the driver steps to it, with every source current it
   evaluates recorded, then MEEP steps the same simulation to it with
   ``sim.run(until=(m - 1/2) dt)``, and ``fields.t`` is asserted.
4. WORDS. MEEP's stored words are read EXACTLY on the package's lattice
   (``little_corner + iyee_shift + 2 i`` per axis; checked against the lift's own
   ``_lattice_indices`` for E and D; on a Dcyl grid ``iveccyl(r, z)`` and MEEP's
   ``Er``/``Ep``/... for the driver's ``Ex``/``Ey``/...): each point's word is copied from
   ``fields_chunk::f[c][0]`` (and ``f[c][1]`` on complex storage) of the chunk that owns
   it, at that chunk's ``grid_volume::index(c, ivec)``, with no arithmetic (``MeepReader``).
   ``fields.get_field(component, ivec)``, MEEP's point read, returns ``(1, +0) * word`` in
   ``complex<double>`` and so cannot carry a signed zero of a complex word; it is still
   read at every point as the reader's check: MEEP's point read applied to the exact words
   (``meep_point_read``) must reproduce it bit for bit, and its float32 cast must be exact.
   A compared point no chunk owns but a chunk's array holds (a metallic wall) is read from
   the first chunk holding it, and every chunk holding it must store the same word; a
   compared point no chunk holds is refused by name. A held point of a B component in a
   chunk whose B array is its H array (MEEP gives H an array of its own only where it needs
   one) holds the word H's connection wrote: where that word is not B's point read, MEEP's
   B there is its point read, which is read, and the shared word must be H's point read
   (refused on complex storage; ``MeepReader``). A held point of a B component in a chunk
   that took an H array of its own in the step just run holds the word it held before that
   step (MEEP built that step's connections while the chunk's H was its B): where that word
   is not its lattice image's, it must be the word read one step earlier, and the image's
   exact word is read (``MeepReader``). Under ``flush`` the stored-word reads and their
   compare run with subnormals kept (DAZ/FZ would read a stored subnormal as zero), and
   the mode is restored and measured before either side steps again; MEEP's and the
   package's readers run in the mode. They are compared with the driver's stored words as bit patterns on
   the compared cells (owned and wall, ``meep_identity.cell_classes``): value-differing,
   word-differing and sign-of-zero-only counts. Ghost cells are compared too, with MEEP's
   point read, and reported, never judged. MEEP's ``sim.get_array`` of every component is compared with
   ``Fields.to_meep_array`` the same way. The FPU mode is re-measured at every compared
   step.
5. DIGESTS of MEEP's words: the stored and reader digest of every component at every
   compared step, and the plane digests up to step 20 (``meep_identity``'s schema).
6. A PLANTED FLIP at step min(20, N): one word of the driver's most populated component
   is flipped by one ulp in a copy. The word compare must count exactly one word, the
   digest must differ from MEEP's, and exactly one plane digest, the word's own, must
   differ.
7. THE SMOKE RECORD. For a case ``meep_identity_smoke.json`` records, at its steps,
   MEEP's point read on the whole lattice (ghosts included, as the smoke reads them) must
   digest to the smoke record's: the MEEP stepped here is the identity reference that
   minted it, and the lift beside it changed nothing MEEP stores.

It writes ``<out>/<mode>/<case>/`` (``capsule.capsule``, ``currents.json``,
``digests.json``, ``planes.npy``, ``result.json``) and, after every case, ``<out>/
meep_identity.json``, a ledger in the fixtures' schema that ``replay`` and
``record_meep_identity.py`` read. Progress: one flushed line per case and compared step;
``--rows`` appends one JSON line per case.

Exit status: 0 when every compared word is equal and every planted flip was caught
(``replay``: every digest equal and every flip caught and placed); 1 when a word or a
digest differs; 3 when it could not run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MARK = "TIER-C-JSON "
#: A case's wall-clock ceiling; ``MEEP_IDENTITY_CASE_TIMEOUT`` (seconds) raises it for a
#: corpus row whose MEEP word reads alone outrun the default.
CASE_TIMEOUT_S = int(os.environ.get("MEEP_IDENTITY_CASE_TIMEOUT", "1800"))
#: MEEP's Dcyl spelling of each package component. The driver names a Dcyl run's
#: components in its Cartesian letters (``from_meep``: ``mp.Er`` -> ``"Ex"``, ``mp.Ep`` ->
#: ``"Ey"``), so ``getattr(mp, name)`` would ask MEEP for a component its grid lacks.
CYLINDRICAL_NAMES = {"Ex": "Er", "Ey": "Ep", "Ez": "Ez", "Dx": "Dr", "Dy": "Dp", "Dz": "Dz",
                     "Hx": "Hr", "Hy": "Hp", "Hz": "Hz", "Bx": "Br", "By": "Bp", "Bz": "Bz"}
CASE_FILES = {"capsule": "capsule.capsule", "currents": "currents.json",
              "digests": "digests.json", "planes": "planes.npy"}


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path: Optional[str]) -> Any:
    """The case registry: ``meep_identity_smoke`` or a ``--cases-file`` module."""
    if not path:
        from parity.meep_gpu import meep_identity_smoke  # noqa: PLC0415

        return meep_identity_smoke
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("meep_identity_tier_c_cases", os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the cases file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_case(cases: Any, mp: Any, label: str) -> Any:
    builder = getattr(cases, "build", None) or getattr(cases, "build_case", None)
    if builder is not None:
        return builder(mp, label)
    return cases.BUILDERS[label](mp)


# --------------------------------------------------------------------------
# Words (MEEP side)
# --------------------------------------------------------------------------

def meep_component(mp: Any, driver: Any, name: str) -> Any:
    """MEEP's component for one package name: ``mp.Er`` for ``"Ex"`` on a Dcyl grid."""
    if getattr(driver.grid, "cylindrical", False):
        return getattr(mp, CYLINDRICAL_NAMES[name])
    return getattr(mp, name)


def lattice(mp: Any, sim: Any, driver: Any, name: str) -> Tuple[List[int], ...]:
    """MEEP ivec coordinates of every stored cell of one component, per axis.

    ``little_corner + iyee_shift + 2 i`` per axis; on a Dcyl grid the axes are (r, phi,
    z) with the phi axis invariant (a Dcyl ivec carries r and z only), read with
    ``.r()`` and ``.z()``. E and D are checked against the lift's own
    ``_lattice_indices`` (which reads materials, where MEEP relocates a not-owned point
    itself). Then the STORED WINDOW (``Grid.stored_window_low``): on a Bloch-periodic
    axis a Yee-shift-0 component's stored cell 0 is MEEP's OWNED top point
    ``little_corner + 2 N``, not ``little_corner``, which MEEP does not own and whose
    ``get_field`` is the owned word times the Bloch phase in double, not a stored word.
    On a folded periodic axis the far ghost slot (one shift-1 slot past ``big_corner``)
    is read at its image about the second mirror, as the lift samples it.
    """
    from meep_gpu.from_meep import _lattice_axis_stores_far_ghost, _lattice_indices  # noqa: PLC0415

    grid = driver.grid
    gv = sim.fields.gv
    component = meep_component(mp, driver, name)
    shift = gv.iyee_shift(component)
    corner = gv.little_corner()
    cylindrical = bool(getattr(grid, "cylindrical", False))
    if cylindrical:
        shifts = (shift.r(), 0, shift.z())
    else:
        shifts = (shift.x(), shift.y(), shift.z())
    firsts = tuple((corner.r(), 0, corner.z())[axis] + shifts[axis] if cylindrical else
                   (corner.x(), corner.y(), corner.z())[axis] + shifts[axis] for axis in range(3))
    counts = (int(grid.nx), int(grid.ny), int(grid.nz))
    axes = []
    for axis in range(3):
        first = firsts[axis]
        values = ([int(first)] if grid.is_invariant(axis) else
                  [int(first + 2 * i) for i in range(counts[axis])])
        if not grid.is_invariant(axis) and _lattice_axis_stores_far_ghost(grid, axis):
            # The far ghost slot of a folded periodic axis: one shift-1 slot past MEEP's
            # big_corner, a point no MEEP chunk holds. The lift samples its image about
            # the second mirror (corner + 2 + n_full; ``from_meep._lattice_indices``), and
            # so does this reader: the slot is a ghost cell, reported and never judged,
            # and the image is an owned point whose word reads exactly.
            big = gv.big_corner()
            last = ((big.r(), 0, big.z()) if cylindrical else (big.x(), big.y(), big.z()))[axis]
            if values[-1] - last == 1:
                full = (int(grid.nx_full), int(grid.ny_full), int(grid.nz_full))[axis]
                mirror = int(firsts[axis] - shifts[axis]) + 2 + full
                values[-1] = 2 * mirror - values[-1]
        axes.append(values)
    if name[0] in "ED":
        pinned = _lattice_indices(gv, grid, "E" + name[1], component, 2.0 * float(sim.resolution))
        if [list(int(v) for v in axis) for axis in pinned] != axes:
            raise RuntimeError(f"{name}: the lattice rule disagrees with the lift's "
                               "_lattice_indices")
    for axis in range(3):
        if not grid.is_invariant(axis) and grid.stored_window_low(axis, int(shifts[axis]) % 2) == 1:
            axes[axis][0] = int(firsts[axis] + 2 * counts[axis])
    return tuple(axes)


#: MEEP's NUM_FIELD_COMPONENTS (src/meep.hpp): Ex .. Bz, the field components a
#: ``fields_chunk`` holds an array pair for.
NUM_FIELD_COMPONENTS = 20


def _chunk_parts(chunk_address: int, component: int) -> Tuple[int, int]:
    """``fields_chunk::f[component][0]`` and ``[1]`` of the chunk at ``chunk_address``.

    ``realnum *f[NUM_FIELD_COMPONENTS][2]`` is the first member of MEEP 1.33.0's
    ``fields_chunk`` (src/meep.hpp), a class with no virtual function, so the pointer pair
    of component c sits at the chunk's address plus ``2 c`` pointers. A NULL pointer reads 0.
    """
    import ctypes  # noqa: PLC0415

    pair = (ctypes.c_void_p * (2 * NUM_FIELD_COMPONENTS)).from_address(chunk_address)
    return int(pair[2 * component] or 0), int(pair[2 * component + 1] or 0)


def _float_words(address: int, count: int) -> Any:
    """A NumPy view of ``count`` stored ``realnum`` (float32) words at ``address``."""
    import ctypes  # noqa: PLC0415

    import numpy  # noqa: PLC0415

    return numpy.ctypeslib.as_array((ctypes.c_float * count).from_address(address))


class MeepReader:
    """MEEP's stored words of one component on the package's lattice, read exactly.

    Each point's word is MEEP's own: ``f[c][0][i]`` (and ``f[c][1][i]`` on complex
    storage) of one chunk, ``i = grid_volume::index(c, ivec)``, copied with no arithmetic.
    An OWNED point is read from the chunk ``fields::get_field(c, ivec)`` reads
    (monitor.cpp:145-157): the first, in chunk order, whose ``grid_volume::owns`` the point,
    with the identity symmetry (sn = 0) and no wrap. A point no chunk owns but some chunk's
    array holds (0 <= ivec - little_corner <= 2 n on every axis: the low plane of a
    metallic wall, which MEEP allocates as zero and never writes, since ``find_metals``
    lists owned points only and ``connect_the_chunks`` skips it through
    ``on_metal_boundary``, boundaries.cpp:315-344 and 417; the not-owned low plane of a
    Bloch axis, which the chunks' connections fill; or a mirror ghost) is HELD: it is read from the
    first chunk, in chunk order, that holds it, and every other chunk holding it must store
    the same word (``held_copies_differing`` counts the points where one does not). The
    chunks' grid volumes are ``structure::get_chunk_volumes()``, in the fields' chunk order
    (each ``fields_chunk``'s ``gv`` is its structure chunk's). Owners, holders and indices
    are resolved once; the array pointers are read again at every read, because MEEP
    allocates some lazily (H in a PML, at the first step: fields_dump.cpp).

    ``fields.get_field``, MEEP's point read, returns ``(1, +0) * word`` in
    ``complex<double>``: a complex word's imaginary -0 beside a real part >= +0, or its
    real -0 beside an imaginary part <= -0, reads +0. It is read too, at every point, as
    this reader's check (``read`` returns its counts): at every owned point
    ``meep_point_read`` of the exact word must equal it word for word, and its float32 cast
    must be exact. At a held point ``get_field`` reads no stored word (0.0 where no chunk
    owns the point, or a symmetry image times its phase); ``held_point_read_differing``
    counts the held words that differ from it, reported only.

    A B component's held point in a chunk that keeps H and B in ONE array has no B word of
    its own. ``fields_chunk::alloc_f`` sets ``f[H] = f[B]`` (fields.cpp:480-494) and
    ``update_eh`` gives H an array of its own only where it needs one: mu, a polarization
    or a PML conductivity along the component (update_eh.cpp:165-171). Such a chunk's
    not-owned point receives B's connection and then H's into the same word
    (``fields::step``, step.cpp:72 then 90), so the word left there is H's: the owned
    image's H, where MEEP's B at the point (``get_field``, which brings the point into the
    user volume, boundaries.cpp:207-230) is the image's B; MEEP never reads that B
    ("we don't use the non-owned B for anything", boundaries.cpp:369-377). Where the
    shared word differs from B's point read, the word read is B's point read, and the
    shared word must be H's point read there (``held_aliased_unexplained`` counts the points
    where it is not). On complex storage the point read cannot carry a signed zero, so such
    a point is counted in ``held_aliased_substituted`` and a caller refuses it.
    ``held_aliased`` marks every held B point of a shared chunk (the arrays are compared at
    every read: H gets its own array lazily, at the first step).

    A B component's held point in a chunk that took an H array of its own in the step just
    run was not written by that step. ``connect_the_chunks`` sends B between two chunks only
    when one of them keeps H in an array of its own (``B_redundant``, boundaries.cpp:359-365,
    402-403 and 491-492), judged when the connections are built; ``update_eh`` allocates H
    in the first step that needs it (update_eh.cpp:165-170) and only then marks the
    connections stale (update_eh.cpp:53-56), so they are rebuilt at the next
    ``step_boundaries`` (step.cpp:226; step.cpp:81), after that step's B boundary
    (step.cpp:72). In that step no B reaches the chunk's held points (``step_db`` writes from
    ``little_owned_corner0`` up, owned points only: step_db.cpp:124, meep/vec.hpp:1102-1104,
    vec.cpp:445-460; the new H array is a copy, so B's array keeps its word); from the next
    step on, B's connection writes the image's B there. So where such a word is not the
    image's, it must be the word the point held at the previous read, one step earlier,
    with the image's owner keeping H in B's array then too (``held_unconnected_unexplained``
    counts the points where any of this fails), and the word read is the image's exact
    stored word: the point brought into the user volume one lattice vector up on each axis
    where it sits on the low plane (``locate_point_in_user_volume``, boundaries.cpp:207-230),
    on a Cartesian grid with no symmetry, where a zero Bloch wavevector along that axis makes
    ``eikna`` exactly 1 and the connection a copy (boundaries.cpp:93-98, 404-405 and
    494-495). ``held_unconnected`` marks every held B point of such a chunk at that read,
    ``held_unconnected_substituted`` the points whose word was read from the image.

    A point no chunk holds has no MEEP word: its word is the point read and ``stored`` marks
    it False, so a caller refuses it on the cells it judges. A component MEEP does not
    allocate (``f[c][0]`` NULL) reads (+0, +0), as ``get_field`` returns; ``unallocated``
    counts its points. On real storage MEEP holds no imaginary array (``f[c][1]`` NULL) and
    on complex storage it holds one beside every real array; a disagreement with the
    package's storage is refused.
    """

    def __init__(self, mp: Any, sim: Any, component: Any, axes: Tuple[List[int], ...],
                 complex_storage: bool, cylindrical: bool = False):
        import numpy  # noqa: PLC0415

        self.fields = sim.fields
        self.get_field = sim.fields.get_field
        self.component = component
        self.component_index = int(component)
        # MEEP's H counterpart of a B component (Bx .. Bz are Hx .. Hz + 10, meep/vec.hpp).
        self.h_component = (self.component_index - 10
                            if int(mp.Bx) <= self.component_index <= int(mp.Bz) else None)
        self.shape = tuple(len(axis) for axis in axes)
        if cylindrical:
            self.points = [mp.iveccyl(r, z) for r in axes[0] for _ in axes[1] for z in axes[2]]
            coordinates = [(r, z) for r in axes[0] for _ in axes[1] for z in axes[2]]
        else:
            self.points = [mp.ivec(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
            coordinates = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
        self.complex_storage = complex_storage
        volumes = sim.structure.get_chunk_volumes()
        self.num_chunks = int(sim.fields.num_chunks)
        if len(volumes) != self.num_chunks:
            raise RuntimeError(f"{len(volumes)} structure chunk volumes for "
                               f"{self.num_chunks} fields chunks")
        self.ntot = [int(volume.ntot()) for volume in volumes]
        # Only the grid's own directions: a D1 grid is z, a D2 grid x and y, Dcyl r and z.
        active = ([0, 1] if cylindrical else
                  {int(mp.D1): [2], int(mp.D2): [0, 1]}.get(int(sim.fields.gv.dim), [0, 1, 2]))
        point_array = numpy.asarray(coordinates, dtype=numpy.int64).reshape(len(coordinates), -1)
        self._coordinates = point_array
        point_array = point_array[:, active]
        self._mp = mp
        self._volumes = volumes
        self._active = active
        # The plain lattice image of a held point (see the class docstring): a Cartesian grid
        # with Bloch-periodic axes (``Simulation`` calls ``fields::use_bloch`` with its
        # ``k_point``) and no symmetry, so the user volume is the grid volume.
        k_point = getattr(sim, "k_point", None)
        self._plain_lattice = bool(not cylindrical and k_point and not getattr(sim, "symmetries", None))
        if self._plain_lattice:
            gv = sim.fields.gv
            low = gv.little_corner()
            self._lattice_low = [int(low.x()), int(low.y()), int(low.z())]
            self._lattice_span = [2 * int(gv.nx()), 2 * int(gv.ny()), 2 * int(gv.nz())]
            self._zero_k = [complex(k_point.x) == 0, complex(k_point.y) == 0, complex(k_point.z) == 0]

        def counts(volume: Any) -> List[int]:
            every = ([volume.nr(), volume.nz()] if cylindrical else
                     [volume.nx(), volume.ny(), volume.nz()])
            return [int(every[axis]) for axis in active]

        def corner(vector: Any) -> List[int]:
            every = [vector.r(), vector.z()] if cylindrical else [vector.x(), vector.y(), vector.z()]
            return [int(every[axis]) for axis in active]

        size = len(coordinates)
        owner = numpy.full(size, -1, dtype=numpy.int64)
        index = numpy.zeros(size, dtype=numpy.int64)
        holders = []
        for chunk, volume in enumerate(volumes):
            low = numpy.asarray(corner(volume.little_corner()), dtype=numpy.int64)
            high = low + 2 * numpy.asarray(counts(volume), dtype=numpy.int64)
            # grid_volume::owns needs 0 < p - little_corner <= 2 n (and, on a Dcyl grid's
            # axis chunk, p.r() == little_corner.r()), so only held points are tried.
            held = numpy.flatnonzero(numpy.all((point_array >= low) & (point_array <= high), axis=1))
            holders.append(held)
            for k in held[owner[held] < 0]:
                if volume.owns(self.points[k]):
                    owner[k] = chunk
                    index[k] = int(volume.index(component, self.points[k]))
        holder = numpy.full(size, -1, dtype=numpy.int64)
        holder_index = numpy.zeros(size, dtype=numpy.int64)
        self.copies = []
        for chunk, held in enumerate(holders):
            loose = held[owner[held] < 0]
            if not loose.size:
                continue
            where = numpy.asarray([int(volumes[chunk].index(component, self.points[k]))
                                   for k in loose], dtype=numpy.int64)
            first = holder[loose] < 0
            holder[loose[first]] = chunk
            holder_index[loose[first]] = where[first]
            if (~first).any():
                self.copies.append((chunk, loose[~first], where[~first]))
        bound = numpy.asarray(self.ntot, dtype=numpy.int64)
        for chunks_of, indices_of in ((owner, index), (holder, holder_index)):
            used = chunks_of >= 0
            if numpy.any((indices_of[used] < 0) | (indices_of[used] >= bound[chunks_of[used]])):
                raise RuntimeError("a grid_volume index falls outside its chunk's array")
        for chunk, positions, indices in self.copies:
            if numpy.any((indices < 0) | (indices >= bound[chunk])):
                raise RuntimeError("a grid_volume index falls outside its chunk's array")
        self.owned = (owner >= 0).reshape(self.shape)
        self.stored = ((owner >= 0) | (holder >= 0)).reshape(self.shape)
        source = numpy.where(owner >= 0, owner, holder)
        source_index = numpy.where(owner >= 0, index, holder_index)
        self.groups = []
        for chunk in range(self.num_chunks):
            positions = numpy.flatnonzero(source == chunk)
            if positions.size:
                self.groups.append((chunk, positions, source_index[positions]))
        self.unallocated = 0
        self.held_copies_differing = numpy.zeros(self.shape, dtype=bool)
        self.held_point_read_differing = 0
        self.held_aliased = numpy.zeros(self.shape, dtype=bool)
        self.held_aliased_substituted = numpy.zeros(self.shape, dtype=bool)
        self.held_aliased_unexplained = numpy.zeros(self.shape, dtype=bool)
        self.held_unconnected = numpy.zeros(self.shape, dtype=bool)
        self.held_unconnected_substituted = numpy.zeros(self.shape, dtype=bool)
        self.held_unconnected_unexplained = numpy.zeros(self.shape, dtype=bool)
        # (step, per-chunk f[H] == f[B], exact real words, exact imaginary words) at the
        # latest read (here: construction), and at the read of the step before it.
        self._state = self._snapshot() if self.h_component is not None else None
        self._before = None

    def _chunks(self) -> Any:
        import ctypes  # noqa: PLC0415

        return (ctypes.c_void_p * self.num_chunks).from_address(int(self.fields.chunks))

    def _exact(self, chunks: Any) -> Tuple[Any, Any, int]:
        """The exact stored words at every point, ``(real, imag, unallocated)``."""
        import numpy  # noqa: PLC0415

        count = len(self.points)
        real = numpy.zeros(count, dtype=numpy.float32)
        imag = numpy.zeros(count, dtype=numpy.float32)
        unallocated = 0
        for chunk, positions, indices in self.groups:
            words = self._gather(chunks, chunk, indices)
            if words is None:
                unallocated += int(positions.size)
                continue
            real[positions], imag[positions] = words
        return real, imag, unallocated

    def _all_shares_h(self, chunks: Any) -> List[bool]:
        return [self._shares_h(chunks, chunk) for chunk in range(self.num_chunks)]

    def _snapshot(self) -> Tuple[int, List[bool], Any, Any]:
        chunks = self._chunks()
        real, imag, _unallocated = self._exact(chunks)
        return int(self.fields.t), self._all_shares_h(chunks), real, imag

    def _image_words(self, chunks: Any, where: Any) -> Tuple[Any, Any, Any]:
        """The exact words of the plain lattice images of points ``where``.

        ``(real, imag, owner)``: the image is the point one lattice vector up
        (``ilattice_vector``, 2 n in ivec units) on every axis where it sits on the grid's
        low plane, each such axis with a zero Bloch wavevector; it is read from the first
        chunk, in chunk order, that owns it, as ``get_field`` reads. ``owner`` is -1 where
        there is no such image.
        """
        import numpy  # noqa: PLC0415

        real = numpy.zeros(len(where), dtype=numpy.float32)
        imag = numpy.zeros(len(where), dtype=numpy.float32)
        owner = numpy.full(len(where), -1, dtype=numpy.int64)
        if not self._plain_lattice:
            return real, imag, owner
        for n, k in enumerate(where):
            point = [int(v) for v in self._coordinates[k]]
            moved, plain = False, True
            for axis in self._active:
                if point[axis] == self._lattice_low[axis]:
                    plain = plain and self._zero_k[axis]
                    point[axis] += self._lattice_span[axis]
                    moved = True
            if not (moved and plain):
                continue
            image = self._mp.ivec(*point)
            for chunk, volume in enumerate(self._volumes):
                if volume.owns(image):
                    owner[n] = chunk
                    words = self._gather(chunks, chunk,
                                         numpy.asarray([int(volume.index(self.component, image))]))
                    if words is not None:
                        real[n], imag[n] = words[0][0], words[1][0]
                    break
        return real, imag, owner

    def _shares_h(self, chunks: Any, chunk: int) -> bool:
        """Whether a chunk keeps this B component and its H in one array (``f[H] == f[B]``)."""
        address = int(chunks[chunk] or 0)
        if not address:
            raise RuntimeError(f"MEEP's fields chunk {chunk} is NULL")
        first, _ = _chunk_parts(address, self.component_index)
        h_first, _ = _chunk_parts(address, self.h_component)
        return bool(first) and first == h_first

    def point_read(self) -> Any:
        """``fields.get_field`` at every point, complex128 (MEEP's point read)."""
        import numpy  # noqa: PLC0415

        count = len(self.points)
        return numpy.fromiter(map(self.get_field, itertools.repeat(self.component, count),
                                  self.points), dtype=numpy.complex128, count=count)

    def _gather(self, chunks: Any, chunk: int, indices: Any) -> Any:
        """``(real, imag)`` words of ``indices`` in one chunk; ``None`` when unallocated."""
        import numpy  # noqa: PLC0415

        address = int(chunks[chunk] or 0)
        if not address:
            raise RuntimeError(f"MEEP's fields chunk {chunk} is NULL")
        first, second = _chunk_parts(address, self.component_index)
        if not first:
            return None
        if bool(second) != self.complex_storage:
            raise RuntimeError(
                f"MEEP's chunk {chunk} holds {'an' if second else 'no'} imaginary array for "
                f"component {self.component_index}, the package's storage is "
                f"{'complex' if self.complex_storage else 'real'}")
        real = _float_words(first, self.ntot[chunk])[indices]
        imag = (_float_words(second, self.ntot[chunk])[indices] if second else
                numpy.zeros(len(indices), dtype=numpy.float32))
        return real, imag

    def read(self):
        """``(words, inexact, imaginary, point_read_differing)``.

        ``words``: the exact stored words, float32 or complex64 in the lattice's shape.
        ``inexact``: point reads whose float32 cast is not exact. ``imaginary``: on real
        storage, point reads with an imaginary part. ``point_read_differing``: owned
        words whose ``meep_point_read`` differs from MEEP's point read (0 when the exact
        read and ``get_field`` read the same stored word). ``unallocated``,
        ``held_copies_differing``, ``held_aliased``, ``held_aliased_substituted``,
        ``held_aliased_unexplained``, ``held_unconnected``, ``held_unconnected_substituted``
        and ``held_unconnected_unexplained`` (masks), ``held_point_read_differing`` and
        ``point_words`` (MEEP's point read cast to float32, in the lattice's shape) are left
        on the reader.
        """
        import numpy  # noqa: PLC0415

        if int(self.fields.num_chunks) != self.num_chunks:
            raise RuntimeError("MEEP's fields chunks changed after the reader was built")
        count = len(self.points)
        chunks = self._chunks()
        real, imag, unallocated = self._exact(chunks)
        step = int(self.fields.t)
        state = None
        if self.h_component is not None:
            state = (step, self._all_shares_h(chunks), real.copy(), imag.copy())
            if self._state is not None and step != self._state[0]:
                self._before = self._state
            self._state = state
        differing = numpy.zeros(count, dtype=bool)
        for chunk, positions, indices in self.copies:
            words = self._gather(chunks, chunk, indices)
            if words is None:
                words = (numpy.zeros(len(indices), numpy.float32),) * 2
            differing[positions] |= ((words[0].view(numpy.uint32) != real[positions].view(numpy.uint32))
                                     | (words[1].view(numpy.uint32) != imag[positions].view(numpy.uint32)))
        self.held_copies_differing = differing.reshape(self.shape)
        self.unallocated = unallocated
        values = self.point_read()
        parts = [values.real.copy(), values.imag.copy()]
        casts = [part.astype(numpy.float32) for part in parts]
        inexact = 0
        with numpy.errstate(invalid="ignore"):
            for part, cast in zip(parts if self.complex_storage else parts[:1],
                                  casts if self.complex_storage else casts[:1]):
                inexact += int(numpy.count_nonzero((cast.astype(numpy.float64) != part)
                                                   & ~numpy.isnan(part)))
        imaginary = 0 if self.complex_storage else int(numpy.count_nonzero(parts[1]))
        owned = self.owned.reshape(-1)
        aliased = numpy.zeros(count, dtype=bool)
        substituted = numpy.zeros(count, dtype=bool)
        unexplained = numpy.zeros(count, dtype=bool)
        unconnected = numpy.zeros(count, dtype=bool)
        if state is not None:
            before = self._before
            for chunk, positions, _indices in self.groups:
                held = positions[~owned[positions]]
                if not held.size:
                    continue
                if state[1][chunk]:
                    aliased[held] = True
                elif before is not None and before[1][chunk]:
                    unconnected[held] = True
        unconnected_substituted = numpy.zeros(count, dtype=bool)
        unconnected_unexplained = numpy.zeros(count, dtype=bool)
        if unconnected.any():
            # A held B point of a chunk that took an H array of its own since the previous
            # read: in that step the connections still treated the chunk as H == B, so no B
            # reached the point. Where its word is not the image's, it must be the word it
            # held one step earlier, the image's owner then keeping H in B's array too, and
            # the image's exact word is read.
            where = numpy.flatnonzero(unconnected)
            image_real, image_imag, image_owner = self._image_words(chunks, where)
            differs = (image_owner >= 0) & (
                (real[where].view(numpy.uint32) != image_real.view(numpy.uint32))
                | (imag[where].view(numpy.uint32) != image_imag.view(numpy.uint32)))
            stale = ((real[where].view(numpy.uint32) == before[2][where].view(numpy.uint32))
                     & (imag[where].view(numpy.uint32) == before[3][where].view(numpy.uint32)))
            sender = numpy.asarray([bool(before[1][int(o)]) if o >= 0 else False for o in image_owner],
                                   dtype=bool)
            explained = differs & stale & sender & (step == before[0] + 1)
            unconnected_unexplained[where[differs & ~explained]] = True
            moved = where[explained]
            unconnected_substituted[moved] = True
            real[moved] = image_real[explained]
            imag[moved] = image_imag[explained]
        self.held_unconnected = unconnected.reshape(self.shape)
        self.held_unconnected_substituted = unconnected_substituted.reshape(self.shape)
        self.held_unconnected_unexplained = unconnected_unexplained.reshape(self.shape)
        if aliased.any():
            # A held B point of a chunk whose B array is its H array: the word there is the
            # one H's connection wrote last. Where it is not B's point read, B's point read
            # is MEEP's B, and the shared word must be H's point read.
            where = numpy.flatnonzero(aliased)
            stored_parts = [real[where], imag[where]][:2 if self.complex_storage else 1]
            b_parts = [cast[where] for cast in casts][:len(stored_parts)]
            differs = numpy.zeros(where.size, dtype=bool)
            for mine, theirs in zip(stored_parts, b_parts):
                differs |= mine.view(numpy.uint32) != theirs.view(numpy.uint32)
            if differs.any():
                moved = where[differs]
                h_read = numpy.fromiter(
                    (self.get_field(self.h_component, self.points[k]) for k in moved),
                    dtype=numpy.complex128, count=moved.size)
                h_parts = [h_read.real.astype(numpy.float32), h_read.imag.astype(numpy.float32)]
                not_h = numpy.zeros(moved.size, dtype=bool)
                for mine, theirs in zip(stored_parts, h_parts):
                    not_h |= mine[differs].view(numpy.uint32) != theirs.view(numpy.uint32)
                unexplained[moved[not_h]] = True
                substituted[moved] = True
                real[moved] = casts[0][moved]
                imag[moved] = casts[1][moved]
        self.held_aliased = aliased.reshape(self.shape)
        self.held_aliased_substituted = substituted.reshape(self.shape)
        self.held_aliased_unexplained = unexplained.reshape(self.shape)
        # A point no chunk holds has no MEEP word: it carries the point read.
        loose = ~self.stored.reshape(-1)
        real[loose] = casts[0][loose]
        imag[loose] = casts[1][loose]
        if self.complex_storage:
            array = numpy.empty(count, dtype=numpy.complex64)
            array.real, array.imag = real, imag
            read = numpy.empty(count, dtype=numpy.complex64)
            read.real, read.imag = casts[0], casts[1]
            check = (meep_point_read(array).view(numpy.uint32).reshape(count, 2)
                     != read.view(numpy.uint32).reshape(count, 2)).any(axis=1)
        else:
            array = real
            check = real.view(numpy.uint32) != casts[0].view(numpy.uint32)
        self.held_point_read_differing = int(numpy.count_nonzero(check & ~owned & ~loose))
        self.point_words = (read if self.complex_storage else casts[0]).reshape(self.shape)
        return array.reshape(self.shape), inexact, imaginary, int(numpy.count_nonzero(check & owned))


def subnormal_words(array: Any) -> int:
    """How many stored float32 words of ``array`` are subnormal (bit patterns, no arithmetic)."""
    import numpy  # noqa: PLC0415

    words = numpy.ascontiguousarray(array).view(numpy.uint32).reshape(-1)
    return int(numpy.count_nonzero(((words & numpy.uint32(0x7F800000)) == 0)
                                   & ((words & numpy.uint32(0x007FFFFF)) != 0)))


def meep_point_read(array: Any) -> Any:
    """The words MEEP's point reader returns for these stored words (monitor.cpp:145-157).

    fields::get_field(c, ivec) returns S.phase_shift(c, sn) * kphase * word in
    complex<double>; on an owned lattice point, sn = 0 and no wrap, the factor is
    (1, 0) * (1, 0) = (1, +0), so the read is (1*a - 0*b, 1*b + 0*a). A real
    component is read as (x, +0) and keeps its word. A complex one loses a signed
    zero: an imaginary -0 beside a real part >= +0 reads +0, and a real -0
    beside an imaginary part <= -0 reads +0. This applies that read to the
    package's stored words, so a compare of the result with MEEP's read is blind to
    exactly what MEEP's read cannot carry, and to nothing else.
    """
    import numpy  # noqa: PLC0415

    values = numpy.asarray(array)
    if values.dtype != numpy.complex64:
        return values
    a = values.real.astype(numpy.float64)
    b = values.imag.astype(numpy.float64)
    out = numpy.empty(values.shape, dtype=numpy.complex64)
    out.real = (1.0 * a - 0.0 * b).astype(numpy.float32)
    out.imag = (1.0 * b + 0.0 * a).astype(numpy.float32)
    return out


def word_counts(ours: Any, theirs: Any, mask: Any) -> Dict[str, int]:
    """``{words, word_differing, value_differing, sign_of_zero_only}`` on ``mask``'s cells."""
    import numpy  # noqa: PLC0415

    from meep_gpu.meep_identity import compared_words  # noqa: PLC0415

    a = compared_words(ours, mask)
    b = compared_words(theirs, mask)
    fa, fb = a.view(numpy.float32), b.view(numpy.float32)
    same_bits = a == b
    with numpy.errstate(invalid="ignore"):
        same_value = (fa == fb) | (numpy.isnan(fa) & numpy.isnan(fb))
    return {"words": int(a.size), "word_differing": int((~same_bits).sum()),
            "value_differing": int((~same_value).sum()),
            "sign_of_zero_only": int((same_value & ~same_bits).sum())}


def _add(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)


# --------------------------------------------------------------------------
# The compare child: one case
# --------------------------------------------------------------------------

def child_compare(spec: Dict[str, Any]) -> Dict[str, Any]:
    import numpy  # noqa: PLC0415, F401 - NumPy before MEEP, as every reference run

    import meep as mp  # noqa: PLC0415

    from parity.meep_gpu import meep_identity_smoke as smoke  # noqa: PLC0415
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    from meep_gpu import subnormal_policy  # noqa: PLC0415

    case, mode, out = spec["case"], spec["mode"], Path(spec["out"])
    mp.verbosity(0)
    # MEEP's own switch first (on x86 it is the mechanism, and the flush reference is
    # MEEP's own flush-on-import); where it has no effect (arm64) the package's lever
    # installs the thread's floating-point environment, FE_DFL_DISABLE_DENORMS_ENV for
    # flush (FPCR.FZ), which steps MEEP to the x86 flush reference's words (R13).
    _module, mechanism = subnormal_policy._drive_host_fpu(mode == "flush")  # noqa: SLF001
    state = identity.subnormal_state()
    result: Dict[str, Any] = {
        "case": case, "mode": mode, "subnormals": state, "subnormal_mechanism": mechanism,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "meep": {"version": str(mp.__version__), "single_precision": bool(mp.is_single_precision()),
                 "file": os.path.realpath(mp.__file__)},
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "python": platform.python_version(), "numpy": numpy.__version__}}
    if state["mode"] != mode:
        result["error"] = (f"the FPU is {state['mode']} after {mechanism}, not {mode}: no "
                           f"{mode} reference on this host")
        return result
    if mode == "flush" and os.environ.get("OMP_NUM_THREADS") != "1":
        result["error"] = ("flush is a reference only at OMP_NUM_THREADS=1, and it is "
                           f"{os.environ.get('OMP_NUM_THREADS') or 'unset'}")
        return result
    if not mp.is_single_precision():
        result["error"] = "this MEEP is not a single-precision build"
        return result

    check = smoke.load_tool("check_install")
    result["meep"]["images"] = {os.path.basename(path): check.sha256_of(path)
                                for path in check.loaded_image_paths()
                                if os.path.basename(path).startswith(("libmeep", "_meep"))}
    result["libm"] = check.libm_identity()

    import meep_gpu  # noqa: PLC0415
    from meep_gpu import capsule  # noqa: PLC0415

    from parity.meep_gpu.sweep_corpus_lift_parity import _timing  # noqa: PLC0415

    started = time.time()
    cases = load_cases(spec.get("cases_file"))
    sim = build_case(cases, mp, case)
    with capsule.recording_construction() as recorder:
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
    if driver.gpu is not None:
        result["error"] = f"the lift resolved driver.gpu = {driver.gpu!r}, not the NumPy reference"
        return result
    cylindrical = bool(getattr(driver.grid, "cylindrical", False))
    result["grid_shape"] = [int(value) for value in driver.grid.shape]
    result["cylindrical"] = cylindrical
    if hasattr(cases, "after_lift"):
        # The cases file's own checks on the lifted object (a corpus row asserts the grid
        # it was measured on); a raise is this case's error, never a result.
        result["case_facts"] = cases.after_lift(mp, sim, driver)
    if spec.get("steps") is not None:
        n_steps, rule = int(spec["steps"]), "--steps"
    elif hasattr(cases, "steps_planned"):
        n_steps, rule = int(cases.steps_planned(mp, sim)), "the cases file's steps_planned(mp, sim)"
    else:
        n_steps = int(_timing(mp, sim, float(sim.resolution), 200, 12000)["steps_planned"])
        rule = "sweep_corpus_lift_parity._timing(mp, sim, resolution, 200, 12000)['steps_planned']"
    steps = identity.compared_steps(n_steps)
    plane_steps = [step for step in steps if step <= identity.PLANE_STEPS]
    flip_step = plane_steps[-1]
    out.mkdir(parents=True, exist_ok=True)
    try:
        info = capsule.write_capsule(out / CASE_FILES["capsule"], driver, recorder, sim=sim)
        capsule_text = f"capsule {os.path.getsize(info.path)} bytes"
    except capsule.CapsuleRefusal as exc:
        # A driver the capsule schema cannot carry (DFT monitors, schema 1) is still word-
        # compared: the words do not depend on the capsule. The refusal is recorded, and
        # the case's directory then lacks the capsule, so it cannot be minted or replayed.
        info = None
        result["capsule_refused"] = str(exc)
        capsule_text = f"capsule REFUSED ({exc})"
    classes = {name: identity.cell_classes(driver.grid, name) for name in identity.COMPONENTS}
    masks = {name: classes[name]["compared"] for name in identity.COMPONENTS}
    ghosts = {name: classes[name]["ghost"] for name in identity.COMPONENTS}
    complex_storage = driver.get_field("Ez", cell_centered=False).dtype.kind == "c"
    readers = {name: MeepReader(mp, sim, meep_component(mp, driver, name),
                                lattice(mp, sim, driver, name), complex_storage, cylindrical)
               for name in identity.COMPONENTS}

    @contextlib.contextmanager
    def exact_reads():
        """Stored words are read and compared with subnormals kept, whatever the mode.

        Under flush, ``get_field``'s float-to-double promotion, the cast back and the
        value compare would read a stored subnormal word as zero (DAZ on x86, FZ on
        arm64), so a subnormal on one side against a zero on the other could compare
        equal. MEEP and the driver are stepped in the mode; only the reads leave it, and
        the mode is put back and measured before either steps again.
        """
        if mode != "flush":
            yield
            return
        subnormal_policy._drive_host_fpu(False)  # noqa: SLF001
        if identity.subnormal_state()["mode"] != "keep":
            raise RuntimeError("the stored-word reads could not leave flush")
        try:
            yield
        finally:
            subnormal_policy._drive_host_fpu(True)  # noqa: SLF001
            if identity.subnormal_state()["mode"] != "flush":
                raise RuntimeError("the FPU did not return to flush after the reads")

    smoke_record = smoke.load_record(smoke.RECORD)
    smoke_case = (smoke_record["cases"].get(case)
                  if smoke_record.get("mode") == mode and not spec.get("cases_file") else None)
    smoke_check = {"record": os.path.basename(smoke.RECORD), "steps": [], "equal": 0, "of": 0,
                   "lattice_equal": None}
    if smoke_case is not None:
        smoke_check["lattice_equal"] = all(
            [list(axis) for axis in smoke.expand_lattice(smoke_case["lattice"][name])]
            == [list(axis) for axis in lattice(mp, sim, driver, name)] for name in identity.COMPONENTS)
    say(f"  {case} {mode}: lifted, stored {list(driver.grid.shape)}, {len(steps)} compared steps "
        f"to N = {n_steps}; {capsule_text}")
    totals = {"stored": {}, "ghost": {}, "reader": {}, "stored_read": {}, "point_read_loss": {}}
    first_read_difference = None
    stored_digests: Dict[str, Dict[str, str]] = {}
    reader_digests: Dict[str, Dict[str, str]] = {}
    plane_rows = []
    reads = {"inexact": 0, "imaginary": 0, "point_read_differing": 0, "unallocated": 0,
             "not_stored_compared": 0, "not_stored_ghost": 0, "held_compared": 0,
             "held_copies_differing": 0, "held_point_read_differing": 0, "held_aliased": 0,
             "held_aliased_substituted": 0, "held_aliased_unexplained": 0,
             "held_aliased_complex": 0, "held_unconnected": 0, "held_unconnected_substituted": 0,
             "held_unconnected_unexplained": 0}
    subnormal = {"meep_stored": 0, "package_stored": 0}
    fpu_checks = 0
    flip: Dict[str, Any] = {}
    first_difference = None
    differing_steps: List[int] = []
    with capsule.recording_currents(driver) as currents:
        for number, target in enumerate(steps, 1):
            while driver.step_count < target:
                driver.step()
            todo = target - int(sim.fields.t)
            if todo < 0:
                raise RuntimeError(f"MEEP is at t={int(sim.fields.t)}, cannot step to {target}")
            if todo:
                sim.run(until=(todo - 0.5) * sim.fields.dt)
            if int(sim.fields.t) != target or driver.step_count != target:
                raise RuntimeError(f"step {target}: MEEP at {int(sim.fields.t)}, driver at "
                                   f"{driver.step_count}")
            now = identity.subnormal_state()
            fpu_checks += 1
            if now["mode"] != mode:
                raise RuntimeError(f"the FPU is {now['mode']} at step {target}, not {mode}")
            alias = driver.h_alias_rule()
            # The readers do arithmetic (MEEP's interpolation sums), so both run in the mode.
            reader_pairs = {}
            for name in identity.COMPONENTS:
                # A 0-D cell reads as a 0-d array on both sides; one word is one word.
                meep_array = numpy.atleast_1d(numpy.asarray(
                    sim.get_array(component=meep_component(mp, driver, name))))
                package_array = numpy.atleast_1d(numpy.asarray(driver.fields.to_meep_array(name, alias)))
                if meep_array.dtype not in (numpy.float32, numpy.complex64):
                    raise RuntimeError(f"MEEP's get_array of {name} is {meep_array.dtype}: not a "
                                       "single-precision read")
                reader_pairs[name] = (meep_array, package_array)
            stored_digests[str(target)] = {}
            reader_digests[str(target)] = {}
            planes = []
            step_counts = {"stored": {}, "reader": {}}
            per_component = {}
            with exact_reads():
                for name in identity.COMPONENTS:
                    theirs, inexact, imaginary, point_read_differing = readers[name].read()
                    reads["inexact"] += inexact
                    reads["imaginary"] += imaginary
                    reads["point_read_differing"] += point_read_differing
                    reads["unallocated"] += readers[name].unallocated
                    reads["not_stored_compared"] += int(numpy.count_nonzero(
                        ~readers[name].stored & masks[name]))
                    reads["not_stored_ghost"] += int(numpy.count_nonzero(
                        ~readers[name].stored & ghosts[name]))
                    reads["held_compared"] += int(numpy.count_nonzero(
                        readers[name].stored & ~readers[name].owned & masks[name]))
                    reads["held_copies_differing"] += int(numpy.count_nonzero(
                        readers[name].held_copies_differing & masks[name]))
                    reads["held_point_read_differing"] += readers[name].held_point_read_differing
                    for key in ("held_aliased", "held_aliased_substituted",
                                "held_aliased_unexplained", "held_unconnected",
                                "held_unconnected_substituted", "held_unconnected_unexplained"):
                        reads[key] += int(numpy.count_nonzero(getattr(readers[name], key)
                                                              & masks[name]))
                    if complex_storage:
                        reads["held_aliased_complex"] += int(numpy.count_nonzero(
                            readers[name].held_aliased_substituted & masks[name]))
                    ours = driver.get_field(name, cell_centered=False)
                    subnormal["meep_stored"] += subnormal_words(theirs)
                    subnormal["package_stored"] += subnormal_words(ours)
                    counts = word_counts(ours, theirs, masks[name])
                    _add(totals["stored"], counts)
                    _add(step_counts["stored"], counts)
                    # Reported, never judged: the compare a point-read Tier C made, MEEP's
                    # point read (meep_point_read) applied to the package's words against
                    # MEEP's point read; and MEEP's own words against that read, the words
                    # MEEP stores that its point read does not return.
                    read_counts = word_counts(meep_point_read(ours), readers[name].point_words,
                                              masks[name])
                    _add(totals["stored_read"], read_counts)
                    _add(totals["point_read_loss"], word_counts(theirs, readers[name].point_words,
                                                                masks[name]))
                    if first_read_difference is None and read_counts["word_differing"]:
                        first_read_difference = {"step": target, "component": name,
                                                 "word_differing": read_counts["word_differing"]}
                    if ghosts[name].any():
                        # Ghosts are reported against MEEP's point read, as they always were.
                        _add(totals["ghost"], word_counts(ours, readers[name].point_words,
                                                          ghosts[name]))
                    meep_array, package_array = reader_pairs[name]
                    if meep_array.shape != package_array.shape or meep_array.dtype != package_array.dtype:
                        counts_reader = {"words": int(meep_array.size), "word_differing": int(meep_array.size),
                                         "value_differing": int(meep_array.size), "sign_of_zero_only": 0}
                    else:
                        every = numpy.ones(meep_array.shape, dtype=bool)
                        counts_reader = word_counts(package_array, meep_array, every)
                    _add(totals["reader"], counts_reader)
                    _add(step_counts["reader"], counts_reader)
                    stored_digests[str(target)][name] = identity.digest_words(
                        identity.compared_words(theirs, masks[name]))
                    reader_digests[str(target)][name] = identity.digest_words(identity.reader_words(meep_array))
                    if target in plane_steps:
                        planes.append(identity.plane_digests(theirs, masks[name]))
                    if counts["word_differing"] or counts_reader["word_differing"]:
                        per_component[name] = {"stored_word_differing": counts["word_differing"],
                                               "stored_value_differing": counts["value_differing"],
                                               "reader_word_differing": counts_reader["word_differing"]}
                    if smoke_case is not None and not complex_storage and str(target) in smoke_case["digests"]:
                        smoke_check["of"] += 1
                        # The smoke record digests MEEP's point read on the whole lattice.
                        smoke_check["equal"] += (smoke.digest(readers[name].point_words)
                                                 == smoke_case["digests"][str(target)][name])
                        if name == identity.COMPONENTS[-1]:
                            smoke_check["steps"].append(target)
                if planes:
                    plane_rows.append(numpy.stack(planes))
                if target == flip_step:
                    flip = plant_flip(identity, driver, masks, stored_digests[str(target)],
                                      plane_rows[-1], readers, target)
            if per_component:
                differing_steps.append(target)
                if first_difference is None:
                    name = next(iter(per_component))
                    first_difference = {"step": target, "component": name,
                                        "stored_word_differing": per_component[name]["stored_word_differing"],
                                        "reader_word_differing": per_component[name]["reader_word_differing"],
                                        "components": per_component}
            say(f"  {case} {number}/{len(steps)} step {target}: stored "
                f"{step_counts['stored'].get('word_differing', 0)} of "
                f"{step_counts['stored'].get('words', 0)} words differ, reader "
                f"{step_counts['reader'].get('word_differing', 0)} of "
                f"{step_counts['reader'].get('words', 0)} ({time.time() - started:.1f} s)")
    table = currents.table()
    currents_sha256 = table.write(out / CASE_FILES["currents"])
    plane_array = numpy.stack(plane_rows)
    document = {
        "schema": identity.DIGESTS_SCHEMA, "version": identity.DIGESTS_VERSION,
        "case": case, "mode": mode, "n_steps": n_steps, "steps_rule": rule, "steps": steps,
        "components": list(identity.COMPONENTS),
        "storage": "complex64" if complex_storage else "float32",
        "compared_cells": {name: int(masks[name].sum()) for name in identity.COMPONENTS},
        "ghost_cells": {name: int(ghosts[name].sum()) for name in identity.COMPONENTS},
        "stored": stored_digests, "reader": reader_digests,
        "planes": {"file": CASE_FILES["planes"], "steps": plane_steps, "axis": "x",
                   "digest_bytes": identity.PLANE_DIGEST_BYTES},
        "planted_flip": {key: flip[key] for key in ("step", "component", "word", "plane")},
        "words_of": ("MEEP 1.33.0, fields_chunk::f read exactly on the package's lattice; "
                     "reader: sim.get_array"),
    }
    files = identity.write_digests(out, document, plane_array)
    driver.close()
    result.update({
        "n_steps": n_steps, "steps": steps,
        "capsule": None if info is None else {"sha256": info.file_sha256,
                                              "bytes": os.path.getsize(info.path),
                                              "load_calls": info.construction_calls},
        "currents_sha256": currents_sha256, "files": files,
        "stored": totals["stored"], "ghost": totals["ghost"], "reader": totals["reader"],
        "stored_read_equivalent": totals["stored_read"], "first_read_difference": first_read_difference,
        "meep_point_read_loss": totals["point_read_loss"],
        "reads": reads, "fpu_checks": fpu_checks, "planted_flip": flip,
        "first_difference": first_difference, "differing_steps": differing_steps,
        "subnormal_words": subnormal, "smoke_record": smoke_check,
        "chunk_division": {"chunks": (len(sim.structure.get_chunk_volumes()) if info is None else
                                      len(capsule.read_capsule(out / CASE_FILES["capsule"])
                                          .chunk_division["boxes"]))},
        "seconds": round(time.time() - started, 3)})
    if reads["inexact"] or reads["imaginary"]:
        result["error"] = (f"{reads['inexact']} MEEP read(s) were not exactly a float32 and "
                           f"{reads['imaginary']} had an imaginary part: not stored words")
    elif reads["point_read_differing"]:
        result["error"] = (f"{reads['point_read_differing']} exact stored word(s) are not what "
                           "MEEP's point read returns for them: the exact read is not MEEP's word")
    elif reads["not_stored_compared"]:
        result["error"] = (f"{reads['not_stored_compared']} compared point(s) are held by no MEEP "
                           "chunk: MEEP stores no word there")
    elif reads["held_copies_differing"]:
        result["error"] = (f"{reads['held_copies_differing']} compared point(s) no chunk owns are "
                           "held with different words by two MEEP chunks: no single stored word")
    elif reads["held_aliased_unexplained"]:
        result["error"] = (f"{reads['held_aliased_unexplained']} compared B point(s) of a chunk "
                           "whose B array is its H array hold neither B's nor H's point read")
    elif reads["held_aliased_complex"]:
        result["error"] = (f"{reads['held_aliased_complex']} compared B point(s) of a chunk whose "
                           "B array is its H array are read through MEEP's point read on complex "
                           "storage, which cannot carry a signed zero: refused")
    elif reads["held_unconnected_unexplained"]:
        result["error"] = (f"{reads['held_unconnected_unexplained']} compared B point(s) of a chunk "
                           "that took an H array of its own since the previous read hold neither "
                           "their image's word nor, one step on, the word they held before")
    return result


def plant_flip(identity: Any, driver: Any, masks: Dict[str, Any], digests: Dict[str, str],
               planes: Any, readers: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Flip one word of the most populated component in a copy; the three checks."""
    import numpy  # noqa: PLC0415

    best, best_count = None, -1
    for name in identity.COMPONENTS:
        words = identity.compared_words(driver.get_field(name, cell_centered=False), masks[name])
        count = int(numpy.count_nonzero(words & numpy.uint32(0x7FFFFFFF)))
        if count > best_count:
            best, best_count = name, count
    array = driver.get_field(best, cell_centered=False)
    words = identity.compared_words(array, masks[best])
    nonzero = numpy.flatnonzero(words & numpy.uint32(0x7FFFFFFF))
    index = int(nonzero[0]) if nonzero.size else 0
    parts = 2 if array.dtype.kind == "c" else 1
    cell, _part = identity.word_cell(masks[best], parts, index)
    flipped = identity.flip_word(words, index)
    theirs = identity.compared_words(readers[best].read()[0], masks[best])
    differing = int((flipped != theirs).sum())
    planes_differing = identity.compare_planes(
        planes[identity.COMPONENTS.index(best)], identity.flipped_planes(array, masks[best], index))
    return {"step": step, "component": best, "word": index, "plane": cell[0],
            "words_differing": differing,
            "digest_caught": identity.digest_words(flipped) != digests[best],
            "planes_differing": planes_differing,
            "caught": differing == 1 and identity.digest_words(flipped) != digests[best]
                      and planes_differing == [cell[0]]}


# --------------------------------------------------------------------------
# The replay child: Tier A on a directory, MEEP blocked
# --------------------------------------------------------------------------

class _MeepBlocker:
    """A meta-path finder that makes ``import meep`` fail in this process."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "meep" or name.startswith("meep."):
            raise ImportError("MEEP is blocked in a Tier A replay")
        return None


def child_replay(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "meep" in sys.modules:
        return {"case": spec["case"], "error": "meep is imported in the replay process"}
    sys.meta_path.insert(0, _MeepBlocker())
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    result = identity.replay(spec["case"], mode=spec["mode"], root=Path(spec["source"]),
                             watch=bool(spec.get("watch")),
                             progress=lambda text: say("  " + text)).as_dict()
    result["meep_imported"] = "meep" in sys.modules
    result["holds"] = (result["stored_equal"] == result["digests"]
                       and result["reader_equal"] == result["digests"]
                       and result["planted_flip"]["caught"] and result["planted_flip"]["placed"]
                       and not result["meep_imported"])
    return result


def child_main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    try:
        result = (child_replay if spec["kind"] == "replay" else child_compare)(spec)
    except Exception as exc:  # noqa: BLE001 - reported by the parent
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        result = {"case": spec.get("case"), "error": f"{type(exc).__name__}: {exc}"}
    print(MARK + json.dumps(result, sort_keys=True, default=str), flush=True)
    return 0


def run_child(spec: Dict[str, Any]) -> Dict[str, Any]:
    """One case in a process of its own; the child's result, or an error."""
    with tempfile.TemporaryDirectory(prefix="meep-identity-tier-c-") as scratch:
        spec_path = os.path.join(scratch, "case.json")
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
        command = [sys.executable, "-u", os.path.abspath(__file__), "--child-input", spec_path]
        errors_path = os.path.join(scratch, "stderr.txt")
        result, other = None, []
        with open(errors_path, "w", encoding="utf-8") as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True,
                                       cwd=scratch)
            timer = threading.Timer(CASE_TIMEOUT_S, process.kill)
            timer.start()
            try:
                for line in process.stdout:
                    if line.startswith(MARK):
                        result = json.loads(line[len(MARK):])
                    elif line.startswith("["):
                        print(line.rstrip("\n"), flush=True)
                    else:
                        other.append(line.rstrip("\n"))
                process.wait()
            finally:
                timer.cancel()
        with open(errors_path, encoding="utf-8", errors="replace") as handle:
            tail = (handle.read().strip().splitlines() or other)[-6:]
    if result is not None:
        if result.get("error"):
            result["stderr_tail"] = tail
        return result
    return {"case": spec.get("case"), "error": f"exited {process.returncode} before reporting "
                                              f"(a kill after {CASE_TIMEOUT_S} s reads -9): "
                                              f"{' | '.join(tail)}"}


def append_row(path: Optional[str], row: Dict[str, Any]) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()


# --------------------------------------------------------------------------
# The parent: compare
# --------------------------------------------------------------------------

def compare_holds(result: Dict[str, Any]) -> bool:
    """Every compared stored and reader word equal, the flip caught, the smoke record equal."""
    if result.get("error"):
        return False
    smoke = result.get("smoke_record") or {}
    return (result["stored"].get("word_differing", 1) == 0
            and result["stored"].get("value_differing", 1) == 0
            and result["reader"].get("word_differing", 1) == 0
            and result["stored"].get("words", 0) > 0
            and bool(result["planted_flip"].get("caught"))
            and smoke.get("equal") == smoke.get("of")
            and smoke.get("lattice_equal") is not False
            and result.get("fpu_checks") == len(result.get("steps") or ()))


def write_stage_ledger(out: Path, mode: str, results: Dict[str, Dict[str, Any]],
                       oracle: Dict[str, Any]) -> Dict[str, Any]:
    """``<out>/meep_identity.json``: the cases written so far, in the fixtures' schema."""
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    cases, total = {}, 0
    for case, result in sorted(results.items()):
        files = {}
        for role, name in CASE_FILES.items():
            path = out / mode / case / name
            if not path.exists():  # a refused capsule: the case cannot be minted
                continue
            files[role] = {"file": f"{mode}/{case}/{name}", "sha256": sha256_file(str(path)),
                           "bytes": path.stat().st_size}
            total += path.stat().st_size
        cases[case] = {"files": files, "tier_c": tier_c_summary(result)}
        if result.get("capsule_refused"):
            cases[case]["capsule_refused"] = result["capsule_refused"]
    ledger = {"schema": identity.LEDGER_SCHEMA, "version": identity.LEDGER_VERSION,
              "about": ("A Tier C output directory (parity/meep_gpu/meep_identity_tier_c.py "
                        "compare), in the fixtures' schema; not minted"),
              "minted": None, "fixture_bytes": total,
              "sets": {mode: {"cases": cases, "array_path_oracle": oracle}}}
    text = json.dumps(ledger, indent=1, sort_keys=True) + "\n"
    (out / identity.LEDGER_FILE).write_text(text, encoding="utf-8")
    return ledger


def tier_c_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("n_steps", "steps", "stored", "ghost", "reader", "reads", "fpu_checks",
            "planted_flip", "smoke_record", "subnormals", "subnormal_mechanism", "omp_num_threads",
            "meep", "libm", "host", "chunk_division", "seconds")
    return {key: result.get(key) for key in keys}


def command_compare(arguments: argparse.Namespace) -> int:
    from meep_gpu import array_path_oracle  # noqa: PLC0415
    from parity.meep_gpu import meep_identity_smoke as smoke  # noqa: PLC0415

    out = Path(arguments.out).resolve()
    registry = load_cases(arguments.cases_file) if arguments.cases_file else smoke
    cases = [c for c in arguments.cases.split(",") if c] if arguments.cases else list(registry.CASES)
    out.mkdir(parents=True, exist_ok=True)
    say(f"Tier C compare, mode {arguments.mode}, cases {cases}, into {out}")
    run = array_path_oracle.live()
    oracle = {"fixture_set": run.fixture_set, "fingerprint_sha256": run.fingerprint_sha256}
    say(f"array-path oracle {run.fingerprint_sha256[:16]} ({run.fixture_set})")
    results: Dict[str, Dict[str, Any]] = {}
    status = 0
    for number, case in enumerate(cases, 1):
        target = out / arguments.mode / case
        if target.exists():
            print(f"COMPARE NOT RUN: {target} exists; give a new --out", flush=True)
            return 3
        say(f"case {number}/{len(cases)} {case}")
        result = run_child({"kind": "compare", "case": case, "mode": arguments.mode,
                            "out": str(target), "steps": arguments.steps,
                            "cases_file": (os.path.abspath(arguments.cases_file)
                                           if arguments.cases_file else None)})
        if result.get("error"):
            say(f"case {case}: NOT RUN: {result['error']}")
            append_row(arguments.rows, {"case": case, "mode": arguments.mode, "error": result["error"]})
            # A case refused before the child made its directory (a build or lift refusal)
            # still records its refusal by name.
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target.parent / f"{case}.error.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=1, sort_keys=True, default=str)
            return 3
        with open(target / "result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, sort_keys=True, default=str)
        results[case] = result
        write_stage_ledger(out, arguments.mode, results, oracle)
        holds = compare_holds(result)
        stored, reader, ghost = result["stored"], result["reader"], result["ghost"]
        say(f"case {case}: stored {stored['word_differing']} word-differing and "
            f"{stored['value_differing']} value-differing of {stored['words']} compared words; "
            f"reader {reader['word_differing']} of {reader['words']}; ghost "
            f"{ghost.get('word_differing', 0)} of {ghost.get('words', 0)} word-differing "
            f"({ghost.get('sign_of_zero_only', 0)} sign of zero only, reported); planted flip "
            f"{'caught' if result['planted_flip'].get('caught') else 'NOT caught'}; smoke record "
            f"{result['smoke_record']['equal']} of {result['smoke_record']['of']}; "
            f"{'HOLDS' if holds else 'DOES NOT HOLD'} ({result['seconds']} s)")
        append_row(arguments.rows, {"case": case, "mode": arguments.mode, "holds": holds,
                                    **tier_c_summary(result)})
        if not holds:
            status = 1
    print(f"Tier C compare: {sum(compare_holds(r) for r in results.values())} of {len(cases)} "
          f"case(s) hold in mode {arguments.mode}: " + ("IDENTICAL." if status == 0 else "NOT IDENTICAL."),
          flush=True)
    return status


def command_replay(arguments: argparse.Namespace) -> int:
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    source = Path(arguments.source).resolve()
    try:
        ledger = identity.read_ledger(source)
    except identity.IdentityRefusal as exc:
        print(f"REPLAY NOT RUN: {exc}", flush=True)
        return 3
    cases_here = sorted(((ledger["sets"].get(arguments.mode) or {}).get("cases") or {}))
    cases = [c for c in arguments.cases.split(",") if c] if arguments.cases else cases_here
    status = 0
    held = 0
    for number, case in enumerate(cases, 1):
        say(f"replay {number}/{len(cases)} {case} ({arguments.mode}) from {source}")
        result = run_child({"kind": "replay", "case": case, "mode": arguments.mode,
                            "source": str(source), "watch": bool(arguments.watch)})
        if result.get("error"):
            say(f"replay {case}: NOT RUN: {result['error']}")
            append_row(arguments.rows, {"case": case, "error": result["error"]})
            return 3
        flip = result["planted_flip"]
        say(f"replay {case}: stored {result['stored_equal']} and reader {result['reader_equal']} of "
            f"{result['digests']} digests equal; first difference {result['first_difference']}; "
            f"planted flip caught {flip['caught']}, placed {flip['placed']} (planes "
            f"{flip.get('planes_differing')}); meep imported {result['meep_imported']}; "
            f"{'HOLDS' if result['holds'] else 'DOES NOT HOLD'} ({result['seconds']} s)")
        append_row(arguments.rows, result)
        held += bool(result["holds"])
        if not result["holds"]:
            status = 1
    print(f"Tier A replay: {held} of {len(cases)} case(s) hold in mode {arguments.mode}: "
          + ("REPRODUCED." if status == 0 else "NOT REPRODUCED."), flush=True)
    return status


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--child-input", dest="child_input", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")
    compare = sub.add_parser("compare", help="lift, pair-order step and word-compare against MEEP")
    compare.add_argument("--out", required=True)
    compare.add_argument("--mode", choices=("keep", "flush"), default="keep")
    compare.add_argument("--cases", help="a comma-separated subset of the cases")
    compare.add_argument("--cases-file", dest="cases_file",
                         help="a module with build(mp, label) and CASES (default: the four probe cases)")
    compare.add_argument("--steps", type=int, help="N instead of Table 2's step rule")
    compare.add_argument("--rows", help="append one JSON line per case here")
    replay_parser = sub.add_parser("replay", help="Tier A on a directory, MEEP blocked")
    replay_parser.add_argument("--source", required=True,
                               help="a compare --out directory or a fixtures directory")
    replay_parser.add_argument("--mode", choices=("keep", "flush"), default="keep")
    replay_parser.add_argument("--cases", help="a comma-separated subset of the cases")
    replay_parser.add_argument("--watch", action="store_true",
                               help="also count complex multiplies (meep_identity's interceptor)")
    replay_parser.add_argument("--rows", help="append one JSON line per case here")
    arguments = parser.parse_args(argv)
    if arguments.child_input:
        return child_main(arguments.child_input)
    if arguments.command == "compare":
        return command_compare(arguments)
    if arguments.command == "replay":
        return command_replay(arguments)
    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
