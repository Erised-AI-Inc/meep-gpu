"""The off-diagonal presence words the CUDA off-diagonal kernels read, built and checked on the host.

MEEP deletes an off-diagonal ``chi1inv`` array in every structure chunk where it is
identically zero over the samples that chunk allocates, and ``update_eh`` passes NULL
for it there, so ``step_update_EDHB`` DROPS that OFFDIAG term in that chunk rather than
adding a zero (``chunk_map.ChunkLayout.offdiag_kept``; anisotropic_averaging.cpp,
update_eh.cpp:191-196). The array path reads the verdict per cell
(``Fields.offdiag_presence``, ``stepping._offdiagonal_terms``). Dropping and adding a
zero differ only when the row so far is ``-0``, which is reachable.

THE DEVICE RULE (``offdiag_emitter.PRELUDE``: ``chunk_row``, ``chunk_label``,
``chunk_keeps``). A kernel learns which chunk a cell belongs to from the run's chunk
record ``cb``: along each axis, on the component's own Yee sub-lattice there, the label
``chunk_coverage.chunk_owner_label`` reads (0 in the low run, 2 in the high run, 1
otherwise), with the unowned metallic wall row 0 of the integer lattice read at row 1
(``ChunkLayout.owner_volume`` reads a wall row at the nearest owned row). The cell's code
is ``lx + 3*ly + 9*lz``, and the term is added where bit ``code`` of the row's 27-bit
PRESENCE WORD is set. One word per (row, partner) slot, in ``coverage.OFFDIAG_ROW_SLOTS``
order: :func:`presence_words`.

WHY THE WORD IS BUILT FROM THE ORACLE'S OWN VOLUME AND CHECKED, rather than derived from
the per-chunk verdicts. The array path's presence is a per-cell volume read through its
own chunk division (``stepping._chunk_layout``); the kernel's is a per-chunk bit read
through the record the composition binds. The builder reads the first, groups its cells
by the code the device rule gives them, and REFUSES BY NAME a code whose cells disagree
-- so a launch either reproduces the array path's presence on every stored cell, ghost
and wall rows included, or does not happen. A division whose record does not name its
chunks (not a ``{low, interior, high}`` product along an axis) is refused there, and so
is a layout the oracle reads that is not the record's.

NUMPY ONLY; no CuPy. The launchers call it once per frozen configuration.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy

try:
    from ..chunk_map import record_index
    from ..fields import IYEE_SHIFTS
    from .coverage import BC_CODES, OFFDIAG_ROW_SLOTS
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    from meep_gpu.chunk_map import record_index
    from meep_gpu.fields import IYEE_SHIFTS
    from meep_gpu.cuda_kernels.coverage import BC_CODES, OFFDIAG_ROW_SLOTS

__all__ = ["ALL_CHUNKS_KEPT", "PresenceRefusal", "cell_chunk_codes",
           "expand_presence_word", "launch_chunk_arguments", "presence_words"]

#: The word of a row whose array no chunk deletes: all 27 chunk codes set.
ALL_CHUNKS_KEPT = (1 << 27) - 1

#: The metallic boundary code the kernels read (``coverage.BC_CODES``); the device rule
#: reads the wall row of the integer lattice at row 1 on such an axis.
_METALLIC = BC_CODES["metallic"]


class PresenceRefusal(ValueError):
    """A presence word the device rule cannot carry, refused by name."""


def _labels(record: Sequence[int], half_integer: bool, axis: int, rows: numpy.ndarray) -> numpy.ndarray:
    """``chunk_label`` of the device text over an array of rows (the host twin)."""
    a0, a1, b0, b1 = (int(record[record_index(half_integer, axis, word)]) for word in range(4))
    low = (rows >= a0) & (rows < a1)
    high = (rows >= b0) & (rows < b1)
    return numpy.where(low, 0, numpy.where(high, 2, 1))


def cell_chunk_codes(record: Sequence[int], iyee: Sequence[int], shape: Sequence[int],
                     codes: Sequence[int]) -> numpy.ndarray:
    """Per stored cell of one component lattice: the chunk code its kernel reads.

    The host twin of the device rule: per axis, the row (``chunk_row``: an
    integer-lattice row 0 on a metallic axis reads row 1), its label off the record on
    the component's sub-lattice ``iyee[axis]`` there (``chunk_label``), and the code
    ``lx + 3*ly + 9*lz``. ``codes`` are the kernel's three ``bc_*`` arguments.
    """
    labels = []
    for axis in range(3):
        n = int(shape[axis])
        rows = numpy.arange(n, dtype=numpy.int64)
        half = bool(iyee[axis])
        if not half and int(codes[axis]) == _METALLIC and n > 1:
            rows = numpy.where(rows == 0, 1, rows)
        line = _labels(record, half, axis, rows)
        view = [1, 1, 1]
        view[axis] = n
        labels.append(line.reshape(view))
    return (labels[0] + 3 * labels[1] + 9 * labels[2]).astype(numpy.int64) * numpy.ones(
        tuple(int(n) for n in shape), dtype=numpy.int64)


def expand_presence_word(word: int, chunk_codes: numpy.ndarray) -> numpy.ndarray:
    """The per-cell mask the device reads from one presence word (``chunk_keeps``)."""
    return ((int(word) >> chunk_codes) & 1).astype(bool)


def presence_word(present: Optional[Any], chunk_codes: numpy.ndarray, label: str) -> int:
    """One row's 27-bit word from the array path's per-cell presence, or a named refusal.

    ``present`` is ``Fields.offdiag_presence``'s answer (``None``: every cell adds the
    term). Each bit is the one value the cells of that chunk code hold; a code no cell
    reads is clear. A code whose cells disagree is a division the record cannot describe
    to this kernel, and is refused.
    """
    if present is None:
        return ALL_CHUNKS_KEPT
    try:
        from ..backends import to_numpy  # noqa: PLC0415
    except ImportError:  # loaded by path, outside the package
        from meep_gpu.backends import to_numpy  # noqa: PLC0415

    mask = numpy.asarray(to_numpy(present), dtype=bool)
    mask = numpy.broadcast_to(mask, chunk_codes.shape)
    word = 0
    for code in range(27):
        cells = mask[chunk_codes == code]
        if cells.size == 0:
            continue
        if cells.all():
            word |= 1 << code
        elif cells.any():
            lz, rest = divmod(code, 9)
            ly, lx = divmod(rest, 3)
            raise PresenceRefusal(
                f"{label}: the cells that read chunk labels (x {lx}, y {ly}, z {lz}) off the "
                f"chunk record disagree on whether their chunk keeps the array "
                f"({int(cells.sum())} of {cells.size} do), so no per-chunk presence word "
                f"reproduces the array path's presence on this division")
    if not numpy.array_equal(expand_presence_word(word, chunk_codes), mask):
        raise PresenceRefusal(f"{label}: the presence word does not expand back to the "
                              f"array path's presence")
    return word


def presence_words(fields: Any, record: Sequence[int], codes: Sequence[int],
                   layout: Any) -> Tuple[int, ...]:
    """The six presence words, in ``OFFDIAG_ROW_SLOTS`` order; 0 for a dead slot.

    ``record`` is the chunk record the kernel is handed (host ``int32[25]``), ``codes``
    its three boundary codes, and ``layout`` the chunk division the array path reads its
    presence through (``stepping._chunk_layout(fields, pml)``). Every live word is
    checked to expand to the array path's presence on every stored cell, or refused by
    name (:class:`PresenceRefusal`).
    """
    record = [int(value) for value in record]
    shape = tuple(int(n) for n in fields.grid.shape)
    words = []
    for row, partner in OFFDIAG_ROW_SLOTS:
        if fields.chi1inv_offdiagonal_for(row).get(partner) is None:
            words.append(0)
            continue
        chunk_codes = cell_chunk_codes(record, IYEE_SHIFTS[row], shape, codes)
        present = fields.offdiag_presence(row, partner, layout)
        words.append(presence_word(present, chunk_codes, f"chi1inv[{row!r}][{partner!r}]"))
    return tuple(words)


def launch_chunk_arguments(fields: Any, pml: Any, codes: Sequence[int],
                           record: Optional[Sequence[int]] = None) -> Tuple[list, Tuple[int, ...]]:
    """``(record, kept)`` for one off-diagonal launch: the record and the presence words.

    ``record`` is the host chunk record the launch binds; ``None`` builds it from the
    division the array path steps this ``Fields`` with (``stepping._chunk_layout``: the
    installed division, else the region plan's one-rank one). ``codes`` are the
    constitutive half's three boundary codes, the ones the device rule reads its wall
    rows by. The words are built from the array path's own presence through that same
    division and refused by name where the record does not reproduce it, so a record
    whose chunks are not the array path's cannot launch.
    """
    try:
        from .. import stepping  # noqa: PLC0415 - NumPy/CuPy array path
        from ..chunk_map import chunk_bounds_record  # noqa: PLC0415
    except ImportError:  # loaded by path, outside the package
        from meep_gpu import stepping  # noqa: PLC0415
        from meep_gpu.chunk_map import chunk_bounds_record  # noqa: PLC0415
    layout = stepping._chunk_layout(fields, pml)  # noqa: SLF001
    if record is None:
        record = chunk_bounds_record(layout)
    record = [int(value) for value in record]
    return record, presence_words(fields, record, codes, layout)
