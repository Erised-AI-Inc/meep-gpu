"""The chunk records the harness hands to a plan built from bare arrays.

Every Triton and Metal PML plan binds a chunk record (``chunk_map.chunk_bounds_record``: the
``[a0, a1) U [b0, b1)`` membership runs per Yee sub-lattice and axis, then the
partner-presence word), and the harness route has no default:
``launch.plan_from_arrays`` and its siblings refuse a plan built without one by
name. So a gate or probe states which division its reference steps with, through
one of the constructors here:

* :func:`layer_record` — the one-rank division of an engine ``Grid`` and ``PML``
  (``chunk_map.ChunkLayout.one_rank``), the division a driver built without MEEP
  steps with. For a reference that is the package's own array path on that layer.
* :func:`every_cell_record` — every cell of every resolved axis inside a layer
  chunk, on both lattices. For a hand reference that steps the split-field
  recurrence on every cell, which is what every hand reference of these gates
  does today, and for random coefficient tables, which hold a non-unit coefficient
  on every cell and so meet the unit-coefficient precondition (every coefficient is
  exactly 1 outside its membership run) only when no cell is outside it.
* :func:`two_run_record` — two runs per resolved axis, the integer and half-integer
  lattices' boundaries one row apart. For a hand reference that keys its two forms
  on the record itself, so every boundary plane, a two-run line and a lattice swap
  are exercised on any shape.
* :func:`region_plan_record` — the division of the region plan installed on a
  ``Fields``. For a gate whose oracle is ``stepping`` itself: the oracle's first
  PML sub-step installs that plan (``stepping._region_plan``, the one-rank division
  of the grid and layer when no driver installed one), and the kernel leg that runs
  after it must choose its forms by the same division.

None is a fallback: a caller picks one, by name, at the call site.

Host-only NumPy; the plans upload the record themselves.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from meep_gpu.chunk_map import (
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayout,
    chunk_bounds_record,
    record_index,
)


def layer_record(grid: Any, pml: Any) -> np.ndarray:
    """The one-rank chunk record of ``grid`` with the absorber ``pml`` (``None``: none)."""
    return chunk_bounds_record(ChunkLayout.one_rank(grid, pml))


def every_cell_record(shape: Sequence[int]) -> np.ndarray:
    """The record of a division in which every stored cell carries every axis's layer.

    Along each axis that stores more than one cell, both sub-lattices hold the one
    run ``[0, n)`` and the partner-presence bit is set; an axis that stores one cell
    is invariant, with no run and no bit.
    """
    extent = tuple(int(n) for n in shape)
    if len(extent) != 3:
        raise ValueError(f"a chunk record describes three axes, got shape {extent}")
    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    presence = 0
    for axis, n in enumerate(extent):
        if n <= 1:
            continue
        presence |= 1 << axis
        for half in (False, True):
            record[record_index(half, axis, 0)] = 0
            record[record_index(half, axis, 1)] = n
    record[RECORD_PRESENCE_WORD] = presence
    return record


def two_run_record(shape: Sequence[int]) -> np.ndarray:
    """A record with two membership runs on every axis that stores four or more cells.

    Along an axis storing ``n >= 4`` cells, with ``q = max(1, n // 4)``: the integer
    lattice holds ``[0, q) U [n - q, n)`` and the half-integer lattice
    ``[0, q + 1) U [n - q, n)``, so the two lattices' low boundaries are one row apart
    and swapping them moves a plane. An axis storing two or three cells holds the one
    run ``[0, 1)`` on the integer lattice and ``[1, 2)`` on the half-integer one; an
    axis storing one cell is invariant, with no run and no presence bit. Every
    resolved axis sets its presence bit, so the curl takes both partners there.
    """
    extent = tuple(int(n) for n in shape)
    if len(extent) != 3:
        raise ValueError(f"a chunk record describes three axes, got shape {extent}")
    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    presence = 0
    for axis, n in enumerate(extent):
        if n <= 1:
            continue
        presence |= 1 << axis
        if n >= 4:
            q = max(1, n // 4)
            runs = {False: (0, q, n - q, n), True: (0, q + 1, n - q, n)}
        else:
            runs = {False: (0, 1, 0, 0), True: (1, 2, 0, 0)}
        for half, words in runs.items():
            for word, value in enumerate(words):
                record[record_index(half, axis, word)] = value
    record[RECORD_PRESENCE_WORD] = presence
    return record


def region_plan_record(fields: Any) -> np.ndarray:
    """The record of the region plan installed on ``fields`` (``Fields.region_plan``).

    Refused by name when none is installed: the oracle leg has not stepped yet, and a
    record of some other division would choose other forms.
    """
    plan = getattr(fields, "region_plan", None)
    if plan is None:
        raise RuntimeError("no region plan is installed on these fields: the oracle leg "
                           "must step before the kernel leg reads its division")
    return np.ascontiguousarray(chunk_bounds_record(plan.layout), dtype=np.int32)


__all__ = ["every_cell_record", "layer_record", "region_plan_record", "two_run_record"]
