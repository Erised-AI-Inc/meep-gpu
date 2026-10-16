"""MEEP's folded passes on Metal: the beta and BFAST increments after the curl.

MEEP does not fold the special-kz beta coupling or the BFAST correction into the curl.
``step_db`` runs, per component, ``STEP_CURL`` and then ``STEP_BFAST`` (step_db.cpp:
124-142), and after every component the beta pass over the x and y components
(:161-175). Each pass adds its increment through its own additive ladder
(step_generic.cpp:255-333 beta, :335-533 BFAST), keyed on the same dsig and dsigu
layers the curl keys on. The array path transcribes that as
:func:`meep_gpu.stepping._apply_folded_increment`, called after ``_apply_curl``; this
module is the one Metal spelling of the same ladder, for every family that carries a
folded pass (the beta curls, the BFAST curl, and every product that lifts their
bodies).

IN REGISTER, AFTER THE FORMS, BEFORE THE STORES. A certified curl body ends its
per-chunk forms with ``n{t}`` (the ``fu`` word the guarded store writes) and ``v{t}``
(the field word the one-line store writes) in registers. The pass runs between the
two: it updates ``n{t}`` and ``v{t}`` by name, so the certified store lines are kept
byte for byte, and a float32 stored and reloaded is the identity on its bits, so the
pass's own roundings are the array path's without a second launch. Two passes on one
target do not commute in rounding; a family that carries both inserts them in
MEEP's order, the BFAST pass first.

THE LADDER (``stepping._apply_folded_increment``), per target ``t`` with its increment
``inc`` already negated (curl sign convention) and masked to ``+0`` on cells MEEP's
loop does not own::

    df = sinv_dsig * inc
    outside the dsigu chunks:  f  = f - df
    inside them:               fu = fu - df;  f = f - (sinv_dsigu * df)

``sinv`` of an axis is exactly 1 outside the chunks that carry its layer (the region
plan's precondition), so the outside form is MEEP's ``f += raw`` there and
``f += raw * siginv`` in a dsig-only chunk. Every stage SUBTRACTS the negated
increment, which is the word of adding ``raw`` (``x - (-y)`` is ``x + y``, signed zeros
included), and ``x - (+0)`` is ``x``, so a masked cell keeps its word, ``-0`` included.
``fu`` is touched on its own cells only: ``n{t}`` is stored only under ``du{t}``.
Complex storage scales each part by the real coefficient (``part_products.scale``).

WHAT IS CHECKED, NOT ASSUMED. :func:`with_folded_pass` and :func:`insert_folded_pass`
require every anchor they edit to occur exactly once, and :func:`check_forms` requires
the locals the ladder names (``du{t}``, ``n{t}``, ``v{t}``, the three ``si_*``) to be
declared by the certified text the pass is inserted into, so an emitter that renamed a
local is a named failure and never a kernel that reads an undeclared or stale name.
The mask rows are read off the certified source (:func:`ownership_rows`), so the
increment is masked by exactly the predicate the curl is masked by
(``stepping._mask_non_owned_cells`` applies one predicate to both).

No conductivity: every family that carries a folded pass refuses a conductivity on
its curl targets by name, so the conductive arms of the ladder (``raw * cndinv`` and
the ``f_cond`` update) have no Metal spelling.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .shaders import CURL_CYCLE

__all__ = [
    "FOLDED_PASS_ANCHOR", "FOLDED_PASS_BEGIN", "FOLDED_PASS_END",
    "check_forms", "extend_params_dtype", "extend_params_words", "folded_pass_block",
    "fold_ladder", "insert_folded_pass", "insert_split_pass", "masked_rows",
    "ownership_rows", "scaled", "split_pass", "split_pass_registers",
    "with_folded_pass",
    "with_params_members",
]

#: The first guarded auxiliary store of a certified curl body. The per-chunk forms end
#: immediately above it, so a folded pass is inserted IN FRONT of it.
FOLDED_PASS_ANCHOR = "    if (du0) { u0[ii] = n0; }\n"

#: The two comment lines a folded pass opens and closes with, so a product that lifts
#: a body carrying a pass can find it as one block.
FOLDED_PASS_BEGIN = "    // --- folded pass: begin (stepping._apply_folded_increment)"
FOLDED_PASS_END = "    // --- folded pass: end"

#: One ownership-mask row of a certified curl source, as the certified emitters write
#: it (``shaders.ownership_mask``, ``symmetry.folded_top_plane_mask``).
_MASK_ROW = re.compile(r"^    curl([012]) = (\w+) \? (.+) : curl\1;$", re.MULTILINE)


def ownership_rows(source: str) -> Tuple[Tuple[int, str, str], ...]:
    """The ``(target, flag, zero)`` ownership-mask rows of an emitted curl source.

    Read off the certified text rather than recomputed: the cell-0 rows of every
    non-periodic axis and, on a folded PERIODIC axis, the top-plane rows.
    """
    return tuple((int(target), flag, zero)
                 for target, flag, zero in _MASK_ROW.findall(source))


def masked_rows(rows: Iterable[Tuple[int, str, str]], name: str,
                targets: Sequence[int], zero: Optional[str] = None) -> str:
    """The rows re-emitted on another variable: ``{name}{t} = flag ? zero : {name}{t};``.

    ``zero`` overrides the literal the rows carry (a boolean flag takes ``false``).
    Rows of a target outside ``targets`` are dropped: that target runs no pass.
    """
    lines = [f"    {name}{target} = {flag} ? {zero if zero is not None else literal} "
             f": {name}{target};"
             for target, flag, literal in rows if target in targets]
    return "\n".join(lines) or f"    // no unowned cell here: {name} is not masked"


def scaled(coefficient: str, value: str, complex_storage: bool) -> str:
    """``coefficient * value``: one product, or one product per part (``part_products.scale``)."""
    if complex_storage:
        return f"float2({coefficient} * {value}.x, {coefficient} * {value}.y)"
    return f"({coefficient} * {value})"


def fold_ladder(targets: Sequence[int], increment: str, scratch: str,
                complex_storage: bool) -> str:
    """MEEP's additive ladder for one pass, in register, on the targets that run it.

    ``{increment}{t}`` is the negated, masked increment; ``{scratch}{t}`` names the
    ``df`` local. ``n{t}`` and ``v{t}`` are reassigned by name (module docstring).
    """
    kind = "float2" if complex_storage else "float"
    lines = []
    for target in targets:
        dsig, dsigu = CURL_CYCLE[target]
        inc, df = f"{increment}{target}", f"{scratch}{target}"
        si, si_u = f"si_{'xyz'[dsig]}", f"si_{'xyz'[dsigu]}"
        lines += [
            f"    {kind} {df} = {scaled(si, inc, complex_storage)};",
            f"    n{target} = du{target} ? (n{target} - {df}) : n{target};",
            f"    v{target} = du{target} ? (v{target} - {scaled(si_u, df, complex_storage)})"
            f" : (v{target} - {df});",
        ]
    return "\n".join(lines)


def folded_pass_block(title: str, statements: str) -> str:
    """One pass's text between :data:`FOLDED_PASS_BEGIN` and :data:`FOLDED_PASS_END`."""
    return f"{FOLDED_PASS_BEGIN}\n    // {title}\n{statements}\n{FOLDED_PASS_END}\n"


#: The locals the ladder reads or reassigns, each declared by a certified body's forms.
_FORM_DECLARATIONS = (
    re.compile(r"const bool du0 = \w+, du1 = \w+, du2 = \w+;"),
    re.compile(r"float km_x = kmx\[i\], si_x = sinvx\[i\];"),
    re.compile(r"float km_y = kmy\[j\], si_y = sinvy\[j\];"),
    re.compile(r"float km_z = kmz\[k\], si_z = sinvz\[k\];"),
) + tuple(re.compile(rf"\bfloat2? n{t} = ") for t in range(3)) + tuple(
    re.compile(rf"\bfloat2? v{t} = ") for t in range(3))


def check_forms(text: str, owner: str) -> None:
    """Raise unless ``text`` declares every local the ladder names, each exactly once."""
    for pattern in _FORM_DECLARATIONS:
        count = len(pattern.findall(text))
        if count != 1:
            raise AssertionError(
                f"{owner}: the certified curl text declares {pattern.pattern!r} {count} "
                f"times, not once; the folded pass cannot name its locals")


def insert_folded_pass(text: str, block: str, owner: str) -> str:
    """``block`` inserted in front of :data:`FOLDED_PASS_ANCHOR`, which must occur once."""
    count = text.count(FOLDED_PASS_ANCHOR)
    if count != 1:
        raise AssertionError(
            f"{owner}: the certified curl text carries its first guarded auxiliary store "
            f"({FOLDED_PASS_ANCHOR.strip()!r}) {count} times, not once; the folded pass "
            f"cannot be placed")
    check_forms(text, owner)
    return text.replace(FOLDED_PASS_ANCHOR, block + "\n" + FOLDED_PASS_ANCHOR, 1)


def with_folded_pass(source: str, kernels: Tuple[str, str], parameters: str,
                     records: Tuple[int, int], block: str, owner: str) -> str:
    """A certified curl SOURCE with a folded pass, by three checked edits.

    ``kernels`` renames the entry point; ``parameters`` (whole lines) are bound in
    front of the chunk record, which moves from buffer ``records[0]`` to
    ``records[1]``; ``block`` is inserted by :func:`insert_folded_pass`. Each edit must
    land exactly once: a certified emitter that changed under the family is a named
    failure, never a kernel spliced around a missing line.
    """
    old_kernel, new_kernel = kernels
    old_record = f"    constant ChunkBounds& cb [[buffer({records[0]})]],\n"
    new_record = f"    constant ChunkBounds& cb [[buffer({records[1]})]],\n"
    for needle, what in ((f"kernel void {old_kernel}(", "its entry point"),
                         (old_record, "its chunk-record binding")):
        count = source.count(needle)
        if count != 1:
            raise AssertionError(
                f"{owner}: the certified curl source carries {what} ({needle.strip()!r}) "
                f"{count} times, not once; the folded pass cannot be placed")
    source = source.replace(f"kernel void {old_kernel}(", f"kernel void {new_kernel}(", 1)
    source = source.replace(old_record, parameters + new_record, 1)
    return insert_folded_pass(source, block, owner)


# ---------------------------------------------------------------------------
# A pass carried into a certified fused product
# ---------------------------------------------------------------------------
#
# A fused product whose certified body re-emits the curl forms around an ownership
# carve-out (the fold-carry pairs) forms ``n{t}`` and stores ``fu`` for every thread,
# then forms ``v{t}`` only inside each component's owned block. The pass is the same
# statements there, RELOCATED rather than re-spelled: everything but the ``v`` updates
# goes in front of the first guarded ``fu`` store, and each target's ``v`` update
# directly after the one statement that declares that target's ``v``. Each statement
# keeps its operands and its order relative to the registers it reads and writes, so
# the words are the unsplit pass's.

_V_UPDATE = re.compile(r"^    v([012]) = du\1 \? ")


def split_pass(block: str) -> Tuple[str, Dict[int, str]]:
    """``(shared block, {target: v-update statement})`` of one pass block."""
    shared: List[str] = []
    updates: Dict[int, str] = {}
    for line in block.splitlines():
        match = _V_UPDATE.match(line)
        if match:
            target = int(match.group(1))
            if target in updates:
                raise AssertionError(f"the pass updates v{target} twice")
            updates[target] = line.strip()
        else:
            shared.append(line)
    return "\n".join(shared) + "\n", updates


#: One ``n`` update of the ladder, as :func:`fold_ladder` writes it.
_N_UPDATE = re.compile(r"^    n([012]) = du\1 \? \(n\1 - (\w+)\) : n\1;$")


def split_pass_registers(source: str, block: str
                         ) -> Tuple[str, Dict[int, str], Tuple[int, ...]]:
    """``(shared block, {target: v update}, targets whose fu word is nb{t})``.

    :func:`split_pass` with the ladder's ``n`` updates written to their own ``nb{t}``
    locals, typed as ``source`` declares ``n0`` (see :func:`insert_split_pass`).
    """
    shared, updates = split_pass(block)
    kind = "float2" if re.search(r"^\s*float2 n0 = ", source, re.MULTILINE) else "float"
    lines: List[str] = []
    stored: List[int] = []
    for line in shared.splitlines():
        match = _N_UPDATE.match(line)
        if match:
            target = int(match.group(1))
            stored.append(target)
            lines.append(f"    {kind} nb{target} = du{target} ? "
                         f"(n{target} - {match.group(2)}) : n{target};")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n", updates, tuple(stored)


def insert_split_pass(source: str, block: str, owner: str) -> str:
    """``block`` carried into a fused source whose forms are split around ownership.

    The shared statements go in front of :data:`FOLDED_PASS_ANCHOR`; each ``v`` update
    after the unique statement declaring that ``v`` (``float v{t} = `` or
    ``float2 v{t} = ``), at its indentation. Every anchor must occur exactly once.

    THE ``fu`` WORD AND THE CURL'S ``n`` ARE TWO REGISTERS HERE. In a split source the
    displacement ``v{t}`` is formed AFTER the guarded ``fu`` store, from ``n{t}`` -- the
    curl step's word, which MEEP's curl stores before its pass runs. So the ladder's
    ``fu`` update is written to its own local ``nb{t}`` and the store of target ``t``
    takes it (``u{t}[ii] = nb{t}``); ``n{t}`` keeps the curl step's word for the
    displacement, and the ``v`` update then applies the pass to the displacement once.
    """
    shared, updates, stored = split_pass_registers(source, block)
    count = source.count(FOLDED_PASS_ANCHOR)
    if count != 1:
        raise AssertionError(
            f"{owner}: the fused source carries its first guarded auxiliary store "
            f"{count} times, not once; the pass cannot be placed")
    head, tail = source.split(FOLDED_PASS_ANCHOR, 1)
    tail = FOLDED_PASS_ANCHOR + tail
    for target in stored:
        store = f"    if (du{target}) {{ u{target}[ii] = n{target}; }}\n"
        if (head + tail).count(store) != 1 or store not in tail:
            raise AssertionError(
                f"{owner}: the fused source stores fu of target {target} "
                f"{(head + tail).count(store)} times, or above the pass's anchor; the "
                f"pass's fu word cannot be stored")
        tail = tail.replace(
            store, f"    if (du{target}) {{ u{target}[ii] = nb{target}; }}\n", 1)
    source = head + shared + "\n" + tail
    lines = source.split("\n")
    for target, update in sorted(updates.items()):
        declared = re.compile(rf"^(\s*)float2? v{target} = ")
        hits = [index for index, line in enumerate(lines) if declared.match(line)]
        if len(hits) != 1:
            raise AssertionError(
                f"{owner}: the fused source declares v{target} {len(hits)} times, not "
                f"once; the pass's v update cannot be placed")
        indent = declared.match(lines[hits[0]]).group(1)
        lines.insert(hits[0] + 1, indent + update)
    return "\n".join(lines)


def with_params_members(source: str, kernels: Tuple[str, str], members: str,
                        unpack: str, owner: str) -> str:
    """A certified ``Params`` source renamed and carrying more scalars.

    ``kernels`` renames the entry point; ``members`` (one line) joins ``Params`` in
    front of the embedded chunk record, and ``unpack`` (one line) in front of the
    record's alias, so the record keeps its place last in both. Each anchor must occur
    exactly once.
    """
    old_kernel, new_kernel = kernels
    record_member = "    ChunkBounds cb;\n"
    alias = "    constant ChunkBounds& cb = prm.cb;\n"
    for needle, what in ((f"kernel void {old_kernel}(", "its entry point"),
                         (record_member, "its embedded chunk record"),
                         (alias, "its chunk-record alias")):
        count = source.count(needle)
        if count != 1:
            raise AssertionError(
                f"{owner}: the certified fused source carries {what} ({needle.strip()!r}) "
                f"{count} times, not once")
    source = source.replace(f"kernel void {old_kernel}(", f"kernel void {new_kernel}(", 1)
    source = source.replace(record_member, members + "\n" + record_member, 1)
    return source.replace(alias, unpack + "\n" + alias, 1)


def extend_params_dtype(parent: Any, extra: Sequence[Tuple[str, str]]) -> Any:
    """A ``Params`` host dtype with ``extra`` 4-byte members in front of ``cb``.

    The members before ``cb`` keep their offsets, the extra ones take ``cb``'s old
    offset onward, and ``cb`` follows them; the itemsize is rounded up to the struct's
    alignment (8 where a ``float2`` member exists, otherwise 4), as Metal lays the
    struct out.
    """
    import numpy  # noqa: PLC0415

    names = list(parent.names)
    if names[-1] != "cb":
        raise AssertionError(f"the parent Params record does not end with cb: {names}")
    fields = parent.fields
    start = fields["cb"][1]
    out_names = names[:-1]
    formats = [fields[name][0] for name in out_names]
    offsets = [fields[name][1] for name in out_names]
    for index, (name, kind) in enumerate(extra):
        out_names.append(name)
        formats.append(numpy.dtype(kind))
        offsets.append(start + 4 * index)
    out_names.append("cb")
    formats.append(fields["cb"][0])
    offsets.append(start + 4 * len(extra))
    alignment = 8 if any(numpy.dtype(kind).shape == (2,) for kind in formats) else 4
    size = offsets[-1] + fields["cb"][0].itemsize
    size += (-size) % alignment
    return numpy.dtype({"names": out_names, "formats": formats, "offsets": offsets,
                        "itemsize": size})


def extend_params_words(parent: Any, words: Any, extra: Sequence[Tuple[str, str]],
                        values: Sequence[Any]) -> Any:
    """The parent's host ``int32`` words re-laid in :func:`extend_params_dtype`.

    Every parent member is copied by name, so the parent's own computation of each
    field is the one that binds; the extra members take ``values``.
    """
    import numpy  # noqa: PLC0415

    record = numpy.frombuffer(numpy.ascontiguousarray(words, dtype=numpy.int32).tobytes(),
                              dtype=parent)[:1]
    out = numpy.zeros(1, dtype=extend_params_dtype(parent, extra))
    for name in parent.names:
        out[name] = record[name]
    for (name, kind), value in zip(extra, values):
        out[name] = numpy.dtype(kind).type(value)
    return numpy.frombuffer(out.tobytes(), dtype=numpy.int32).copy()
