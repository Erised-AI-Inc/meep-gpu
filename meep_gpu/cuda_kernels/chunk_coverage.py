"""Which chunk divisions the hand-CUDA kernels may step — with no CuPy in sight.

MEEP selects the PML form of every curl and constitutive update per structure chunk
(``dsig``/``dsigu``, step_db.cpp:56-59; ``dsigw``, update_eh.cpp:162-163), so the form
a cell steps with is a property of the run's chunk division. A CUDA kernel reads that
division as data: the ``int32[25]`` record of ``chunk_map.chunk_bounds_record``, one
device vector per composition (``arms.StepContext.chunk_bounds_device``, the kernels'
``cb`` argument). This module decides, over plain facts the composer derives
(``arms.StepContext.chunk_facts``), whether a kernel may be handed that record:

* every arm refuses a division no record describes — not separable along an axis, or
  a line with more than two runs (owner ruling R12: the array path's mask route steps
  it);
* the families that index MEEP's chunks beyond the membership runs — the off-diagonal
  presence masks and the cylindrical r prefix, each row declaring ``chunk_axes`` —
  also refuse a division whose record's run labels do not name its chunk intervals
  one to one along those axes (:func:`chunk_owner_label`,
  :func:`chunk_product_problem`).

A SIBLING OF ``coverage.py``, UNDER THE SAME DISCIPLINE, AND NOT A SECTION OF IT.
Nothing here imports beyond ``__future__`` and ``typing`` (the same AST check
``coverage.py`` is held to pins it, ``test_chunk_binding.py``), so the clause is
evaluable on a host with no GPU. It is not in ``coverage.py`` because that file is
pinned by the CUDA ledger's certification blocks (``subject_sha256`` /
``revision_sha256``) and this clause reads none of its contents: adding it there
would have drifted every block that pins it for no change in what those blocks
certify.

THE ORDER IN AN ARM. ``arms.coverage_adapter`` asks this clause after the family's own
predicate admits, so a configuration a family refuses keeps the family's reason and
the coverage census's attribution moves only where the division is the question.
``fused_pairs.install_fused_pairs`` asks it of every admitted fused product, with the
product's own ``chunk_axes``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# The chunk division (one clause for every arm, and one for the chunk-indexed few)
# ---------------------------------------------------------------------------
#
# THE RECORD IS WHAT A KERNEL READS, so it is what this clause judges. A CUDA
# kernel learns which PML form a cell steps from one int32 vector, laid out by
# ``chunk_map.chunk_bounds_record``: per (Yee sub-lattice, axis) the membership
# runs ``[a0, a1) U [b0, b1)`` of the chunks that carry that axis's layer, then a
# partner-presence word. A division that no such record describes -- one that is
# not separable along an axis (the owner's flag depends on another coordinate), or
# a line with more than two runs -- is stepped on the array path's mask route and
# refused by every kernel (owner ruling R12). The record builder's own refusal is
# the reason, verbatim, so the array path and this table name the same defect.
#
# TWO FAMILIES READ MORE THAN MEMBERSHIP. MEEP deletes an off-diagonal chi1inv
# array in every chunk where it is identically zero and drops the term there
# (anisotropic_averaging.cpp:316-320, update_eh.cpp:191-196), so the off-diagonal
# kernels index a per-chunk presence mask; and MEEP's cylindrical ``f_rderiv_int``
# is a prefix sum restarted at every chunk along r (step_db.cpp:104-118). Both
# index the chunk a cell belongs to, and both read that index off the same record:
# :func:`chunk_owner_label`. That is sound only where the record's labels name the
# chunk intervals along the axis one to one -- a {low, interior, high} product
# along it, with no chunk cut the runs do not show. :func:`chunk_product_problem`
# measures exactly that, row by row, so a wrapping axis whose stored cell 0 images
# the high wall plane (it joins the low run while its chunk is the high one) is
# refused by name rather than read as the low chunk.

#: The axes along which a family's kernels index MEEP's chunks beyond the
#: membership runs, as a registry row's (and a fused product's) ``chunk_axes``.
#: Off-diagonal presence masks are indexed on every axis; the cylindrical prefix
#: restarts at chunk cuts along r, which is axis 0 under the x = R, y = P, z = Z
#: mapping. Every other family reads membership only and declares ``()``.
CHUNK_AXES_OFFDIAG: Tuple[int, ...] = (0, 1, 2)
CHUNK_AXES_CYLINDRICAL: Tuple[int, ...] = (0,)

#: How many words a chunk record carries: 2 sub-lattices x 3 axes x
#: ``[a0, a1, b0, b1]``, then the presence word. Transcribed from
#: ``chunk_map.RECORD_WORDS`` because this module imports nothing;
#: ``test_chunk_binding.py`` pins the two equal.
CHUNK_RECORD_WORDS = 25

_AXIS_NAMES = ("x", "y", "z")


def chunk_record_words(record: Sequence[int], half_integer: bool,
                       axis: int) -> Tuple[int, int, int, int]:
    """``(a0, a1, b0, b1)`` for one (sub-lattice, axis) of a chunk record.

    The index arithmetic is ``chunk_map.record_index``'s, transcribed: word ``w`` of
    (sub-lattice ``s``, axis ``a``) sits at ``(s * 3 + a) * 4 + w`` with ``s = 1`` for
    the half-integer lattice. ``test_chunk_binding.py`` pins the two equal.
    """
    base = ((1 if half_integer else 0) * 3 + int(axis)) * 4
    return (int(record[base]), int(record[base + 1]), int(record[base + 2]),
            int(record[base + 3]))


def chunk_owner_label(words: Sequence[int], row: int) -> int:
    """The chunk label a kernel reads off the record for stored row ``row``.

    ``0`` inside ``[a0, a1)``, ``2`` inside ``[b0, b1)``, ``1`` otherwise -- the low,
    high and interior chunk of a {low, interior, high} product when the record's runs
    are those chunks. It is the ONE spelling of that index: the off-diagonal presence
    masks and the cylindrical prefix segments are built and read through it, and
    :func:`chunk_product_problem` admits a division only where it names the chunk
    intervals one to one.
    """
    a0, a1, b0, b1 = (int(value) for value in words)
    if a0 <= row < a1:
        return 0
    if b0 <= row < b1:
        return 2
    return 1


def chunk_product_problem(words: Sequence[int],
                          owners: Sequence[Optional[int]]) -> Optional[str]:
    """Does :func:`chunk_owner_label` name this line's chunk intervals one to one?

    ``owners[row]`` is the index of the chunk interval that owns stored row ``row``
    along the axis (ghost rows: their image's), or ``None`` where no form is stepped
    and nothing is required (a wall row). Returns ``None`` when every interval reads
    one label and no two intervals share one, else the first disagreement by name.
    """
    labels: Dict[int, Set[int]] = {}
    first_row: Dict[Tuple[int, int], int] = {}
    for row, owner in enumerate(owners):
        if owner is None:
            continue
        label = chunk_owner_label(words, row)
        labels.setdefault(int(owner), set()).add(label)
        first_row.setdefault((int(owner), label), row)
    for owner in sorted(labels):
        if len(labels[owner]) > 1:
            rows = ", ".join(f"row {first_row[(owner, label)]} reads {label}"
                             for label in sorted(labels[owner]))
            return (f"chunk interval {owner} reads more than one record label "
                    f"({rows}), so the record does not say which chunk its cells "
                    f"belong to")
    seen: Dict[int, int] = {}
    for owner in sorted(labels):
        label = next(iter(labels[owner]))
        if label in seen:
            return (f"chunk intervals {seen[label]} and {owner} both read record label "
                    f"{label}, so the record cannot tell them apart")
        seen[label] = owner
    return None


def chunk_division_refusal(facts: Mapping[str, Any],
                           chunk_axes: Sequence[int] = ()) -> Optional[str]:
    """Why no CUDA kernel may step this chunk division, or ``None``.

    ``facts`` is ``arms.StepContext.chunk_facts()``: plain data, so this decision runs
    without importing the layout's module. Three refusals, in order:

    * ``problem``: the composer could not derive a division at all;
    * ``record_refusal``: no ``int32[25]`` record describes it (not separable along
      an axis, or more than two runs on a line) -- every arm (owner ruling R12);
    * ``product_problems[axis]`` for each axis in ``chunk_axes``: the record's run
      labels do not name the chunk intervals along it -- only the families that
      index chunks (the off-diagonal and cylindrical ones) ask.
    """
    problem = facts.get("problem")
    if problem:
        return f"the chunk division could not be read: {problem}"
    record_refusal = facts.get("record_refusal")
    if record_refusal:
        return (f"{record_refusal}; the CUDA kernels read the division only through "
                f"that record, so the array path steps it")
    product_problems = facts.get("product_problems") or {}
    for axis in chunk_axes:
        why = product_problems.get(int(axis))
        if why:
            return (f"the chunk division is not a {{low, interior, high}} product along "
                    f"{_AXIS_NAMES[int(axis)]} that the record names: {why}; this "
                    f"family indexes MEEP's chunks along that axis, so the array path "
                    f"steps it")
    return None
