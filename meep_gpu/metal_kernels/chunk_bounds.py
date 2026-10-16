"""The chunk record as the Metal kernels bind it: one struct, its host packing, its decode.

MEEP selects each cell's PML form per structure chunk (step_db.cpp:56-59,
update_eh.cpp:162-163), and :mod:`meep_gpu.chunk_map` reduces the division a run
steps with to one ``int32[25]`` record (:func:`~meep_gpu.chunk_map.chunk_bounds_record`):
for each Yee sub-lattice and axis, two index runs ``[a0, a1) U [b0, b1)`` of stored
rows whose owning chunk carries that axis's layer, then a partner-presence word with
bit ``a`` set when axis ``a`` is not invariant. This module fixes how a Metal kernel
reads that record. It emits no kernel; the kernel templates and their plans take
it up when they read membership, and a plan checks the record it binds through
:func:`plan_words`.

THE STRUCT IS THE RECORD, WORD FOR WORD. :data:`MEMBERS` lists the 25 members in the
record's own order (:func:`~meep_gpu.chunk_map.record_index`), every member a 4-byte
``int`` except the trailing ``uint`` presence word, so the struct has no padding, its
size is 100 bytes and its alignment 4. A flat ``int32[25]`` buffer therefore binds as
``constant ChunkBounds&`` with no repacking, and a family that carries its scalars in a
``Params`` struct embeds it as one member (:func:`params_member`) at any 4-aligned
offset. The test file checks the member order against ``record_index`` for all 25
positions and the numpy dtype's field offsets against ``4 * position``.

NAMES ARE TAGGED BY SUB-LATTICE: ``cbi_*`` for the integer lattice and ``cbh_*`` for
the half-integer one, so a fused kernel that reads both lattices on one axis never
declares the same local twice. The membership locals are ``in_i_<axis>`` and
``in_h_<axis>``; a kernel that evaluates membership at a second coordinate (a
neighbour-straddle recompute, a carried ghost) passes a ``suffix``.

ONE EMITTER, ANCHORED. :func:`prologue` writes the bounds block between
:data:`BLOCK_BEGIN` and :data:`BLOCK_END`, so a lifter that splices a certified body
finds the block by those two lines rather than by its contents. The membership formula
is the record's: ``(i >= a0 && i < a1) || (i >= b0 && i < b1)``, an empty run being
``[0, 0)``. :func:`decode` evaluates the same formula on the host from the packed
struct's named fields, which is what the tests compare with
:func:`~meep_gpu.chunk_map.membership_from_record` and with the layout's own lines.

NO TORCH AT MODULE LEVEL: only :func:`tensor` imports it, inside the function.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy

from ..chunk_map import (
    AXIS_NAMES,
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayoutRefusal,
    chunk_bounds_record,
    record_index,
)

__all__ = [
    "BLOCK_BEGIN", "BLOCK_END", "MEMBERS", "PRESENCE_MEMBER", "STRUCT",
    "binding", "bounds_locals", "decode", "describe", "for_layout", "member_name",
    "membership_lines", "membership_name", "pack", "params_alias", "params_member",
    "plan_words", "presence_lines", "prologue", "record_dtype", "record_dtype_fields",
    "struct_source", "tensor", "words",
]

#: The Metal struct name, shared by the base singles' buffer binding and every
#: ``Params`` that embeds the record.
STRUCT = "ChunkBounds"

#: Sub-lattice tags: integer (shift 0) and half-integer (shift 1).
LATTICE_TAGS: Dict[bool, str] = {False: "i", True: "h"}

#: The four bound words of one (sub-lattice, axis) entry, in record order.
WORD_NAMES: Tuple[str, str, str, str] = ("a0", "a1", "b0", "b1")

#: The partner-presence member (record word 24).
PRESENCE_MEMBER = "cb_presence"

#: The bounds block's anchor lines.
BLOCK_BEGIN = "// chunk bounds: begin (metal_kernels/chunk_bounds.py)"
BLOCK_END = "// chunk bounds: end"

#: The default stored-index expression per axis: the ``int i, j, k`` every Cartesian
#: template decodes from the linear index (``shaders.py``).
DEFAULT_COORDINATES: Tuple[str, str, str] = ("i", "j", "k")

#: One (sub-lattice, axis) entry: ``(half_integer, axis)``.
Entry = Tuple[bool, int]

#: Every entry, in record order.
ALL_ENTRIES: Tuple[Entry, ...] = tuple(
    (half, axis) for half in (False, True) for axis in range(3))


def member_name(half_integer: bool, axis: int, word: int) -> str:
    """The struct member holding record word ``[a0, a1, b0, b1][word]`` of one entry."""
    if axis not in (0, 1, 2) or word not in (0, 1, 2, 3):
        raise ValueError(f"member_name needs axis in 0..2 and word in 0..3, got {axis}, {word}")
    return f"cb{LATTICE_TAGS[bool(half_integer)]}_{AXIS_NAMES[axis]}_{WORD_NAMES[word]}"


def membership_name(half_integer: bool, axis: int, suffix: str = "") -> str:
    """The membership local of one entry: ``in_h_y`` for the half-integer y line."""
    return f"in_{LATTICE_TAGS[bool(half_integer)]}_{AXIS_NAMES[int(axis)]}{suffix}"


def _members() -> Tuple[str, ...]:
    names: List[Optional[str]] = [None] * RECORD_WORDS
    for half, axis in ALL_ENTRIES:
        for word in range(4):
            position = record_index(half, axis, word)
            if names[position] is not None:
                raise AssertionError(f"record word {position} is assigned twice")
            names[position] = member_name(half, axis, word)
    if names[RECORD_PRESENCE_WORD] is not None:
        raise AssertionError("the presence word collides with a bound word")
    names[RECORD_PRESENCE_WORD] = PRESENCE_MEMBER
    missing = [position for position, name in enumerate(names) if name is None]
    if missing:
        raise AssertionError(f"record words {missing} have no struct member")
    return tuple(name for name in names if name is not None)


#: The struct members in record order: position ``p`` holds record word ``p``.
MEMBERS: Tuple[str, ...] = _members()


def record_dtype() -> numpy.dtype:
    """The numpy structured dtype of one packed record: 24 ``<i4`` bounds, one ``<u4``."""
    return numpy.dtype([(name, "<u4" if name == PRESENCE_MEMBER else "<i4")
                        for name in MEMBERS])


def record_dtype_fields(name: str = "cb") -> List[Tuple[str, numpy.dtype]]:
    """The field a family appends to its ``Params`` dtype to embed the record."""
    return [(str(name), record_dtype())]


def struct_source() -> str:
    """The Metal declaration of :data:`STRUCT`, members in record order."""
    lines = [f"struct {STRUCT} {{"]
    for name in MEMBERS:
        lines.append(f"    {'uint' if name == PRESENCE_MEMBER else 'int'} {name};")
    lines.append("};")
    return "\n".join(lines)


def params_member(name: str = "cb", indent: str = "    ") -> str:
    """The ``Params`` member line that embeds the record: ``ChunkBounds cb;``."""
    return f"{indent}{STRUCT} {name};"


def params_alias(struct: str = "prm", name: str = "cb", indent: str = "    ") -> str:
    """The line a ``Params`` kernel adds before its body, so the block reads ``cb.``."""
    return f"{indent}constant {STRUCT}& {name} = {struct}.{name};"


def binding(index: int, name: str = "cb") -> str:
    """The kernel parameter that binds the record as its own buffer.

    Refused at or past ``device.MAX_BUFFER_BINDINGS``: attribute index 31 does not
    compile on this toolchain.
    """
    from .device import MAX_BUFFER_BINDINGS  # noqa: PLC0415

    index = int(index)
    if not 0 <= index < MAX_BUFFER_BINDINGS:
        raise ValueError(
            f"buffer index {index} is outside 0..{MAX_BUFFER_BINDINGS - 1}; a Metal "
            f"kernel binds at most {MAX_BUFFER_BINDINGS} buffers")
    return f"constant {STRUCT}& {name} [[buffer({index})]]"


def _entries(entries: Optional[Sequence[Entry]]) -> Tuple[Entry, ...]:
    chosen = ALL_ENTRIES if entries is None else tuple(
        (bool(half), int(axis)) for half, axis in entries)
    for half, axis in chosen:
        if axis not in (0, 1, 2):
            raise ValueError(f"an entry's axis must be 0, 1 or 2, got {axis}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"an entry is listed twice: {chosen}")
    return chosen


def bounds_locals(entries: Optional[Sequence[Entry]] = None, struct: str = "cb",
                  indent: str = "    ") -> str:
    """``const int cbh_y_a0 = cb.cbh_y_a0;`` for every word of the listed entries."""
    lines = []
    for half, axis in _entries(entries):
        for word in range(4):
            member = member_name(half, axis, word)
            lines.append(f"{indent}const int {member} = {struct}.{member};")
    return "\n".join(lines)


def membership_lines(entries: Optional[Sequence[Entry]] = None,
                     coordinates: Sequence[str] = DEFAULT_COORDINATES,
                     suffix: str = "", indent: str = "    ") -> str:
    """One ``const bool in_<lattice>_<axis><suffix>`` per entry, evaluated at ``coordinates``.

    Reads the locals :func:`bounds_locals` declares. ``coordinates`` are the stored
    index expressions per axis, each of type ``int``.
    """
    if len(coordinates) != 3:
        raise ValueError(f"coordinates must name one index expression per axis, got {coordinates}")
    lines = []
    for half, axis in _entries(entries):
        at = f"({coordinates[axis]})"
        a0, a1, b0, b1 = (member_name(half, axis, word) for word in range(4))
        lines.append(
            f"{indent}const bool {membership_name(half, axis, suffix)} = "
            f"({at} >= {a0} && {at} < {a1}) || ({at} >= {b0} && {at} < {b1});")
    return "\n".join(lines)


def presence_lines(axes: Sequence[int] = (0, 1, 2), struct: str = "cb",
                   indent: str = "    ") -> str:
    """``const bool has_<axis>`` per listed axis: the record's partner-presence bit."""
    lines = []
    for axis in axes:
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError(f"a presence axis must be 0, 1 or 2, got {axis}")
        lines.append(f"{indent}const bool has_{AXIS_NAMES[axis]} = "
                     f"({struct}.{PRESENCE_MEMBER} & {1 << axis}u) != 0u;")
    return "\n".join(lines)


def prologue(entries: Optional[Sequence[Entry]] = None,
             coordinates: Sequence[str] = DEFAULT_COORDINATES,
             struct: str = "cb", presence: Sequence[int] = (),
             indent: str = "    ") -> str:
    """The anchored bounds block: the entries' locals, their membership, and presence bits."""
    parts = [f"{indent}{BLOCK_BEGIN}", bounds_locals(entries, struct, indent),
             membership_lines(entries, coordinates, "", indent)]
    if presence:
        parts.append(presence_lines(presence, struct, indent))
    parts.append(f"{indent}{BLOCK_END}")
    return "\n".join(part for part in parts if part)


def words(record: Any, shape: Optional[Sequence[int]] = None) -> numpy.ndarray:
    """A validated contiguous ``int32[25]`` copy of a chunk record: the bytes that bind.

    Refused by name (:class:`~meep_gpu.chunk_map.ChunkLayoutRefusal`): a record of the
    wrong length or a non-integer dtype, a value outside ``int32``, a run that is
    reversed or negative, two runs that overlap or are out of order, presence bits
    outside the three axes, and, when ``shape`` is given, a run past that axis's
    stored extent.
    """
    array = numpy.asarray(record)
    if array.shape != (RECORD_WORDS,):
        raise ChunkLayoutRefusal(
            f"a chunk record has {RECORD_WORDS} words, got shape {array.shape}")
    if array.dtype.kind not in "iu":
        raise ChunkLayoutRefusal(f"a chunk record holds integers, got dtype {array.dtype}")
    wide = array.astype(numpy.int64)
    info = numpy.iinfo(numpy.int32)
    if wide.min() < info.min or wide.max() > info.max:
        raise ChunkLayoutRefusal("a chunk record word does not fit in int32")
    for half, axis in ALL_ENTRIES:
        a0, a1, b0, b1 = (int(wide[record_index(half, axis, word)]) for word in range(4))
        where = f"{AXIS_NAMES[axis]} on the {'half-integer' if half else 'integer'} lattice"
        if a0 < 0 or b0 < 0 or a1 < a0 or b1 < b0:
            raise ChunkLayoutRefusal(
                f"the {where} runs [{a0}, {a1}) U [{b0}, {b1}) are reversed or negative")
        if a1 > a0 and b1 > b0 and b0 <= a1:
            raise ChunkLayoutRefusal(
                f"the {where} runs [{a0}, {a1}) U [{b0}, {b1}) overlap, touch or are out of "
                f"order; the record holds two separate runs, the lower first")
        if a1 == a0 and b1 > b0:
            raise ChunkLayoutRefusal(
                f"the {where} record has an empty first run and a second one "
                f"[{b0}, {b1}); a single run is the first")
        if shape is not None and max(a1, b1) > int(shape[axis]):
            raise ChunkLayoutRefusal(
                f"the {where} runs reach row {max(a1, b1)} of an axis with "
                f"{int(shape[axis])} stored rows")
    presence = int(wide[RECORD_PRESENCE_WORD])
    if presence & ~0b111:
        raise ChunkLayoutRefusal(
            f"the presence word {presence:#x} sets a bit outside the three axes")
    return numpy.ascontiguousarray(wide.astype(numpy.int32))


def plan_words(record: Any, shape: Sequence[int], owner: str) -> numpy.ndarray:
    """:func:`words` of the record a plan binds, checked against the plan's shape.

    Every plan that launches a kernel reading the record states which division it
    binds: there is no default, because a default would decide on the caller's
    behalf which cells take which form. Refused by name
    (:class:`~meep_gpu.chunk_map.ChunkLayoutRefusal`), with ``owner`` naming the
    plan or builder: no record (``None``); everything :func:`words` refuses against
    ``shape``; and a presence word that disagrees with the shape — an axis that
    stores more than one cell with its bit cleared, or an absent axis whose runs
    are not empty. An absent partner is an invariant axis, which stores one cell
    (``Grid.is_invariant``), and the curl's stencil form follows the bit.
    """
    if record is None:
        raise ChunkLayoutRefusal(
            f"{owner} was given no chunk record (chunk_map.chunk_bounds_record): the "
            f"kernel selects every PML form per chunk, and a plan without the record "
            f"would bind no division at all; pass chunk_bounds explicitly")
    extent = tuple(int(n) for n in shape)
    if len(extent) != 3:
        raise ChunkLayoutRefusal(f"{owner} binds a chunk record to a {len(extent)}-axis shape")
    try:
        host = words(record, extent)
    except ChunkLayoutRefusal as exc:
        raise ChunkLayoutRefusal(f"{owner}: {exc}") from None
    presence = int(host[RECORD_PRESENCE_WORD])
    for axis, n in enumerate(extent):
        if presence >> axis & 1:
            continue
        if n != 1:
            raise ChunkLayoutRefusal(
                f"{owner}: the chunk record marks {AXIS_NAMES[axis]} invariant (no curl "
                f"partner along it), but the plan stores {n} cells there")
        for half in (False, True):
            if any(int(host[record_index(half, axis, word)]) for word in range(4)):
                raise ChunkLayoutRefusal(
                    f"{owner}: the chunk record marks {AXIS_NAMES[axis]} invariant but "
                    f"holds a run on its {'half-integer' if half else 'integer'} lattice")
    return host


def for_layout(layout: Any, shape: Optional[Sequence[int]] = None) -> numpy.ndarray:
    """:func:`words` of the layout's own record (``chunk_map.chunk_bounds_record``)."""
    return words(chunk_bounds_record(layout), shape)


def pack(record: Any) -> numpy.ndarray:
    """One packed struct, shape ``(1,)`` of :func:`record_dtype`, from a chunk record.

    Assign it into a ``Params`` record's ``cb`` field, or bind its bytes directly.
    """
    return numpy.frombuffer(words(record).tobytes(), dtype=record_dtype()).copy()


def decode(packed: Any, half_integer: bool, axis: int, n_cells: int) -> numpy.ndarray:
    """The membership line of one entry, read from the packed struct's named fields.

    The host evaluation of the formula :func:`membership_lines` emits.
    """
    if isinstance(packed, numpy.ndarray) and packed.dtype == record_dtype():
        row = packed.reshape(-1)[0]
    else:
        row = pack(packed)[0]
    a0, a1, b0, b1 = (int(row[member_name(half_integer, axis, word)]) for word in range(4))
    index = numpy.arange(int(n_cells))
    return ((index >= a0) & (index < a1)) | ((index >= b0) & (index < b1))


def tensor(record: Any, device: str = "mps") -> Any:
    """The record as a plan-owned ``torch.int32`` tensor of 25 words on ``device``.

    Outside the residency registry, as every ``Params`` tensor is: nothing on the
    device writes it and nothing on the host reads it back, and ``Residency.mirror``
    binds only float32 and complex64 volumes.
    """
    import torch  # noqa: PLC0415

    host = words(record)
    return torch.from_numpy(host.copy()).to(torch.device(device))


def describe(record: Any) -> Mapping[str, Any]:
    """The record as a JSON-ready block: runs per entry and the presence bits."""
    host = words(record)
    runs = {}
    for half, axis in ALL_ENTRIES:
        a0, a1, b0, b1 = (int(host[record_index(half, axis, word)]) for word in range(4))
        runs[f"{LATTICE_TAGS[half]}_{AXIS_NAMES[axis]}"] = [[a0, a1], [b0, b1]]
    return {"runs": runs, "presence": int(host[RECORD_PRESENCE_WORD])}
