"""The chunk-record decode block every hand-CUDA kernel reads its PML forms from.

MEEP chooses the PML form of each curl and constitutive update per structure chunk
(``dsig``/``dsigu``, step_db.cpp:56-59; ``dsigw``, update_eh.cpp:162-163). A CUDA kernel
learns the run's chunk division from one ``int32[25]`` device vector, the kernels' ``cb``
argument: ``chunk_map.chunk_bounds_record``, laid out ``[sub-lattice][axis][a0, a1, b0,
b1]`` and then the partner-presence word. Stored row ``r`` of a component whose Yee
sub-lattice along ``axis`` is ``half`` belongs to a chunk that carries ``axis``'s layer
exactly when ``(a0 <= r < a1) || (b0 <= r < b1)`` (``chunk_map.membership_from_record``,
the host twin of the decode below).

THIS MODULE IS THE ONE SPELLING OF THAT DECODE ON THIS TABLE (bit_parity_0_10 DESIGN.md
D6, D10). A kernel's text carries the block as a literal, directly after its index decode
(``int i = idx / (ny * nz);``), and the kernel's own tests require that literal to equal
:func:`bounds_block` for the kernel's entries byte for byte. A fused product that welds
two kernels emits ONE block over the union of their entries, so the names below are
tagged by sub-lattice and never collide:

* ``cbi_<axis>_<word>`` / ``cbh_<axis>_<word>``: the four record words of the integer /
  half-integer lattice on ``axis``;
* ``in_i_<axis>`` / ``in_h_<axis>``: the membership flag of the cell's own stored index
  on that axis and lattice;
* ``cb_presence`` and ``pr_<axis>``: the presence word and the bit of ``axis`` (set when
  the axis is not invariant, so a curl partner differenced along it is present:
  fields.cpp:428-453, ``stepping.curl_presence``).

WHICH ENTRIES A KERNEL READS is the region plan's rule (``chunk_map.RegionPlan``): a curl
component reads its dsigu axis and a constitutive component its own axis, each on the
component's own Yee sub-lattice along that axis (``fields.IYEE_SHIFTS``). So the B curl
reads the half-integer record, the D curl and ``update_H`` the integer one, and
``update_E`` the half-integer one (:func:`curl_entries`, :func:`constitutive_entries`).

The record offsets are ``chunk_map.record_index``'s, imported rather than restated.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ..chunk_map import AXIS_NAMES, RECORD_PRESENCE_WORD, record_index

__all__ = ["DECODE_END", "COORDINATES", "WORDS", "Entry", "bounds_name", "flag_name",
           "presence_name", "membership_test", "bounds_block", "curl_entries",
           "constitutive_entries", "insert_bounds_block", "strip_bounds_block"]

#: The index decode's last line in every flat-indexed kernel of this table. The block
#: follows it directly.
DECODE_END = "    int i = idx / (ny * nz);\n"

#: The stored index along each axis, as the kernels name it.
COORDINATES = ("i", "j", "k")

#: The four record words of one (sub-lattice, axis), in record order.
WORDS = ("a0", "a1", "b0", "b1")

#: One entry of a block: ``(half_integer, axis)``.
Entry = Tuple[bool, int]

#: The block's opening comment. Comment lines only: no anchor reads them.
_COMMENT = (
    "// THE CHUNK RECORD (chunk_map.chunk_bounds_record): row r of this lattice and axis",
    "// lies in a chunk that carries the axis's layer iff (a0 <= r < a1) || (b0 <= r < b1).",
)


def _lattice(half_integer: bool) -> str:
    return "h" if half_integer else "i"


def bounds_name(half_integer: bool, axis: int, word: str) -> str:
    """The local holding record word ``word`` of (sub-lattice, axis): ``cbh_y_a0``."""
    if word not in WORDS:
        raise ValueError(f"word must be one of {WORDS}, got {word!r}")
    return f"cb{_lattice(half_integer)}_{AXIS_NAMES[axis]}_{word}"


def flag_name(half_integer: bool, axis: int) -> str:
    """The membership flag of the cell's own row on (sub-lattice, axis): ``in_h_y``."""
    return f"in_{_lattice(half_integer)}_{AXIS_NAMES[axis]}"


def presence_name(axis: int) -> str:
    """The partner-presence bit of ``axis``: ``pr_x``."""
    return f"pr_{AXIS_NAMES[axis]}"


def membership_test(half_integer: bool, axis: int, row: str) -> str:
    """The membership test of stored row ``row`` (a C expression) on (sub-lattice, axis).

    The one spelling of the test: the block's flag lines use it at the cell's own
    coordinate, and a kernel that steps a constitutive statement at a cell other than its
    own (a fused pair's fill carry, at a mirror ghost) uses it at that cell's index, keyed
    on the same row its coefficient entry is read at. The block must already declare the
    four record words of (sub-lattice, axis).
    """
    names = [bounds_name(half_integer, axis, word) for word in WORDS]
    return (f"({row} >= {names[0]} && {row} < {names[1]}) || "
            f"({row} >= {names[2]} && {row} < {names[3]})")


def _canonical(entries: Iterable[Entry]) -> List[Entry]:
    seen = sorted({(bool(half), int(axis)) for half, axis in entries})
    for _half, axis in seen:
        if axis not in (0, 1, 2):
            raise ValueError(f"an entry's axis must be 0, 1 or 2, got {axis}")
    return seen


def bounds_block(entries: Iterable[Entry], presence: Sequence[int] = (),
                 indent: str = "    ") -> str:
    """The decode lines for ``entries`` (and the presence bits of ``presence``), with a
    trailing blank line.

    Entries are emitted in canonical order (integer lattice first, then x, y, z), so the
    block of a set of entries has one spelling whatever order a caller names them in. Each
    entry is two statements: its four record words, then its flag.
    """
    lines = [f"{indent}{line}\n" for line in _COMMENT]
    for half, axis in _canonical(entries):
        names = [bounds_name(half, axis, word) for word in WORDS]
        loads = ", ".join(f"{name} = cb[{record_index(half, axis, slot)}]"
                          for slot, name in enumerate(names))
        lines.append(f"{indent}const int {loads};\n")
        lines.append(f"{indent}const int {flag_name(half, axis)} = "
                     f"{membership_test(half, axis, COORDINATES[axis])};\n")
    axes = sorted({int(axis) for axis in presence})
    if axes:
        lines.append(f"{indent}const int cb_presence = cb[{RECORD_PRESENCE_WORD}];\n")
        for axis in axes:
            lines.append(f"{indent}const int {presence_name(axis)} = "
                         f"(cb_presence >> {axis}) & 1;\n")
    lines.append("\n")
    return "".join(lines)


def curl_entries(components: Sequence[str]) -> List[Entry]:
    """The entries a curl over ``components`` reads: each one's dsigu axis, on its lattice.

    ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` name the dsigu axis per component, and
    ``fields.IYEE_SHIFTS`` the component's sub-lattice along it; this is the region
    plan's own rule (``chunk_map.RegionPlan``, sub-step ``"curl"``).
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415 - NumPy only
    from ..stepping import B_CURL_TERMS, D_CURL_TERMS  # noqa: PLC0415 - NumPy only

    terms = {term.target: term for term in (*B_CURL_TERMS, *D_CURL_TERMS)}
    out = []
    for component in components:
        axis = AXIS_NAMES.index(terms[component].dsigu)
        out.append((IYEE_SHIFTS[component][axis] == 1, axis))
    return out


def constitutive_entries(components: Sequence[str]) -> List[Entry]:
    """The entries a constitutive update over ``components`` reads: each one's own axis.

    The region plan's sub-step ``"constitutive"`` (dsigw is the component's own axis,
    ``stepping.H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``), on the component's own
    sub-lattice along it.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415 - NumPy only

    out = []
    for component in components:
        axis = AXIS_NAMES.index(component[1])
        out.append((IYEE_SHIFTS[component][axis] == 1, axis))
    return out


def insert_bounds_block(code: str, kernel: str, entries: Iterable[Entry],
                        presence: Sequence[int] = ()) -> str:
    """``code`` with the block for ``entries`` inserted after ``kernel``'s index decode.

    For a text emitted rather than spelt as a literal. The decode is followed by one blank
    line in every kernel of this table; the block goes after it and brings its own
    trailing blank line, so the text reads decode, blank, block, blank, body -- the
    layout the literal kernels carry. Exactly once and refusing by name: ``kernel`` must
    be declared once, decode ``i`` once with a blank line after it, and must not carry a
    block already.
    """
    marker = f'extern "C" __global__ void {kernel}('
    if code.count(marker) != 1:
        raise ValueError(f"the text does not declare {kernel} exactly once")
    head, tail = code.split(marker, 1)
    if tail.count(DECODE_END) != 1:
        raise ValueError(f"{kernel} decodes i {tail.count(DECODE_END)} times, not once; "
                         f"the chunk-record block has no anchor")
    if "cb[" in tail:
        raise ValueError(f"{kernel} already reads the chunk record")
    if tail.count(DECODE_END + "\n") != 1:
        raise ValueError(f"{kernel}'s decode is not followed by a blank line; the "
                         f"chunk-record block has no anchor")
    block = bounds_block(entries, presence)
    return head + marker + tail.replace(DECODE_END + "\n", DECODE_END + "\n" + block, 1)


def strip_bounds_block(code: str, kernel_or_body: str, entries: Iterable[Entry],
                       presence: Sequence[int] = ()) -> str:
    """``code`` with the block for ``entries`` removed from after its index decode.

    The inverse of :func:`insert_bounds_block`, for a fused product that lifts two
    kernels' bodies and emits ONE block over the union of their entries (DESIGN.md D10):
    each lifted body's own block is removed here, and refused by name unless it is
    exactly the block :func:`bounds_block` emits for that body's entries, directly after
    its decode and its blank line. ``kernel_or_body`` names the text in the refusal.
    """
    block = bounds_block(entries, presence)
    anchor = DECODE_END + "\n" + block
    if code.count(anchor) != 1:
        raise AssertionError(
            f"{kernel_or_body} does not carry the chunk-record block of {sorted(set(entries))} "
            f"(presence {sorted(set(presence))}) exactly once directly after its decode; "
            f"the fused product would decode a record its lifted body no longer reads")
    return code.replace(anchor, DECODE_END + "\n", 1)
