"""Measured cost as a slot arbiter: the timing record's format, and the consult that reads it.

Plan of record: ``the design notes (meep-gpu-dispatch-preference-plan)`` (Design 1-7).
Today a fused product wins a slot on SHAPE, never on speed: a longer span displaces
the products it contains wherever it is admitted, with no size gate. This module is
the instrument that lets a MEASUREMENT veto that -- and only veto it.

UNWIRED. Nothing in the dispatch path imports this module; the wiring is a small
edit in its own re-certification round, because every route gate's ``selected`` map
can move with it. Until then this file changes nothing about any run.

=============================================================================
WHAT IS DECIDED HERE, AND WHAT NEVER IS
=============================================================================

ARBITRATION ONLY. The record may change WHO holds a slot. It can never change what a
kernel computes: a vetoed candidate leaves the products it would have displaced in
their slots, and those are certified products running their own certified arithmetic.

A VETO FIRST, NOT A RANKING. One direction, failing closed (plan section 3):

    A candidate may not displace the products it supersedes when the record says it
    is SLOWER than them on this shape class, by more than the recorded spreads. With
    no record for the class, behaviour is today's -- the longer span wins.

So every path that is not "measured, and slower beyond spread" answers ``no_record``
or ``not_vetoed``, and both mean the same thing to the caller: install as today.

=============================================================================
THE RECORD
=============================================================================

One JSON document per kernel table, cut by ``parity/meep_gpu/cut_timing_record.py``
from timing rows and never written by hand (it carries a digest of its own body, and
a record whose digest does not match REFUSES). Its ``keys`` map is::

    key   = product | storage | dimensions | susceptibilities | bracketed/clean | lo-hi
    value = measured fused ms/step, the same-table baseline ms/step, their ratio, the
            spread of each leg, the rows that fed it, and what the product displaced

* ``product`` is the fused label exactly as the release tables spell it.
* ``storage``, ``dimensions`` and ``susceptibilities`` are read from the same run-shape
  dict the release tables are evaluated on (:func:`shape_class`), so the record and the
  release cannot spell an axis two ways.
* ``bracketed`` says whether the product's seam carries the deposit-repair bracket on
  this run. A two-pair plan's fixed cost follows the SOURCE'S seam, not the product, so
  a key without this axis cannot express it (plan addendum, 2026-09-17).
* ``lo-hi`` is an INCLUSIVE cell-count band whose edges are sizes at least
  ``min_rows`` rows measured. Bands are data in the record. Nothing is extrapolated:
  below the smallest such size, above the largest, and inside a gap where the verdict
  changes between two measured sizes, the answer is ``no_record``.
  A BAND'S INTERIOR IS INTERPOLATED. Its edges are measured; a size between them need
  not have been run at all, and the key's verdict is applied to it. That is why an
  edge must be corroborated: a single session sets no band boundary, because the
  window spreads beside a row price only that session's own noise and not the spread
  BETWEEN sessions, which the corpus shows at a few parts in a thousand to two parts
  in a hundred -- the size of the thinnest margins measured.
* A key VETOES only when every row that fed it is slower than that row's own two
  window spreads allow (:func:`least_margin`); the pooled medians and extremes beside
  it are for reading. A key with too few rows is written ``unmeasured`` with the reason.

The record also names the deposit-repair ROUTE its rows ran (``route``) -- the digest
of ``deposit_repair.py`` AND what the bracket did per step: the index, the repairs per
bracketed step, whether any source was restricted, whether any fell back. ONE DIGEST
CARRIES MORE THAN ONE ROUTE (the corpus has ``e4f88c1e`` running 3-per-step full and
1-per-step restricted), and the bracket's price is most of what a bracketed key
measures, so comparing the digest alone would let a record cut on one route veto on a
tree running the other, a factor-of-three error in the very quantity being vetoed on.
The caller passes the LIVE route facts and any mismatch is ``no_record``. That is the
expected steady state between a bracket change and the re-cut after it.

A clean key is asked only for the digest. It is priced only by rows in which its
product was the plan's sole fused product, so no bracket ran inside the number and
what the bracket does per step cannot move it; the module is still in the step, so
the digest is still compared.

The record also names the fused-pair EMITTERS its rows timed (``subject_barrier``),
and the caller passes the live digests the same way: a record must not veto a kernel
that was repaired after it was cut. A source some admitted rows recorded and others
did not cannot be compared -- it is listed in ``allowed_unpinned`` and named in the
verdict as not compared, never pooled as though "not recorded" were a digest.

=============================================================================
THE ENVIRONMENT-VARIABLE CONTRACT
=============================================================================

This module reads NO environment variable (``the development notes``: only ``fastpath.py``
and ``subnormal_policy.py`` do). The mode and the record path are ARGUMENTS. In mode
``span`` the consult returns before touching its record or any file, so it reproduces
today's behaviour exactly and cannot fail.

=============================================================================
WHAT A WIRING STEP MUST ASSUME, STATED SO IT CAN BE ARGUED WITH
=============================================================================

A key prices a whole PLAN -- (B_singles + D_singles) / (B_fused + D_fused), attributed
to the one product that carries the bracket -- and applying it to a slot decision takes
four assumptions that no row of the corpus establishes:

1. ADDITIVITY: a plan's per-step cost is its slots' costs plus a bracket attached to the
   SOURCE's seam.
2. THE BRACKET VANISHES with the veto: the post-veto plan (the partner still fused, the
   vetoed product back on single arms) was never timed and no row of that shape exists.
3. THE VERDICT TRANSFERS ACROSS COMPOSITIONS: a key has no co-fused axis, so a number
   measured beside a fused partner is applied to a run where that partner is not
   admitted. The measurements list their ``co_fused`` partners so the residue is at
   least visible: a partner that WINS dilutes the loss (an under-veto, safe), one that
   LOSES inflates it (a false veto, unsafe), and near a crossover the partner's few
   percent is the decision.
4. ``bracketed`` IS COMPUTED AT ARBITRATION exactly as the bench computed it. That one
   is feasible today: the call site holds ``context.sources`` / ``context.grid`` and
   ``deposit_repair.seam_source_reasons(..., 'B'/'D')`` answers the same question.

The veto direction survives 1-3 (they move the ratio toward a weaker veto or admit a
partner's few percent); a RANKING would not, which is why plan section 3 defers it.

=============================================================================
WHAT THE WIRING SITE MUST DO WITH A RECORD IT CANNOT READ
=============================================================================

DECIDED, SO THAT THE WIRING DOES NOT HAVE TO: a record that is present and invalid is
a fault, and :func:`consult_path` RAISES :class:`RecordRefused` on it rather than
papering it over as an absence. That raise is for TOOLS -- a gate, a cutter, a report.
The dispatch path must not take a plan-time exception from a file: it loads the record
ONCE, at the top of a run, inside a ``try`` that catches :class:`RecordRefused`, prints
the refusal beside the ``selected`` map and carries ``None`` from there on, which every
consult answers ``no_record`` (today's behaviour). One read, one place to fail, and a
loud one. ``the development notes`` states the same rule beside the flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

#: The record format this module reads. A record stamped otherwise refuses by name.
#: ``/2`` (2026-09-20) carries a whole route identity and a subject barrier; a ``/1``
#: record compared the repair digest alone and named no emitter, so it is refused
#: rather than read with its guards missing.
SCHEMA = "meep_gpu.dispatch_preference.timing/2"

#: ``measured`` consults the record; ``span`` is today's rule and consults nothing.
MODES = ("measured", "span")

#: What a consult can answer. Only the first displaces nothing.
OUTCOMES = ("vetoed", "not_vetoed", "no_record")

#: The kernel tables a record may be cut for.
TABLES = ("triton", "cuda", "metal")

#: The two spellings of the bracket axis inside a key.
BRACKETED, CLEAN = "bracketed", "clean"

STORAGES = ("real", "complex")
STATUSES = ("measured", "unmeasured")

#: The facts that identify ONE deposit-repair program. The digest alone does not: the
#: same bytes run a full three-target bracket or a restricted one-target bracket, and
#: the bench's own counters say which a row took.
ROUTE_FACTS = ("deposit_repair_sha256", "index", "repairs_per_bracketed_step",
               "restricted", "fell_back")

#: How the bracket addressed the grid. ``None`` is the fifth spelling and means no row
#: was bracketed at all, in which case there is no per-step repair count either.
ROUTE_INDEXES = ("linear", "3-tuple", "mixed", "never-ran")

#: The emitter sources whose digest is a POOLING BARRIER, per kernel table: they are
#: the code a row's fused kernels came out of. Scoped to the table because
#: ``fused_polarization_pair`` is a CUDA module (``cuda_kernels/fused_pairs.py``
#: imports it inside the polarization builders and nothing under ``triton_kernels/``
#: names it), so comparing it on a Triton record would compare a file those rows never
#: ran. ``fused_pairs`` is the shared fusion block -- the seam loop, the two-slot
#: install and the span arbitration this record is to be wired into -- and is a
#: barrier on every table.
#:
#: WHAT THIS CANNOT COVER, AND WHY IT IS NOT SPELLED HERE: a product's OWN device text
#: comes from a per-product module (``cuda_kernels/fused_electric_pair.py``, the
#: ``triton_kernels/*.py`` emitters, the Metal kernel sources), and no timing row pins
#: one. A record therefore cannot say which per-product kernel text it timed; adding
#: those pins is a change to ``parity/meep_gpu/bench_fused_products.PINNED_SOURCES``
#: and its rows, not to this table.
SUBJECT_SOURCES_BY_TABLE: Dict[str, Tuple[str, ...]] = {
    "triton": ("fused_pairs",),
    "cuda": ("fused_pairs", "fused_polarization_pair"),
    "metal": ("fused_pairs", "metal_dispatch", "metal_launch", "metal_device"),
}

#: Where each barrier source lives, relative to this file: the ONE spelling the live
#: digest is read from, so the cutter and the consult cannot name two files.
SUBJECT_FILES = {
    "fused_pairs": "cuda_kernels/fused_pairs.py",
    "fused_polarization_pair": "cuda_kernels/fused_polarization_pair.py",
    "metal_dispatch": "metal_dispatch.py",
    "metal_launch": "metal_kernels/launch.py",
    "metal_device": "metal_kernels/device.py",
}

#: What a row that did not pin a source records. It is a VALUE, never a digest: rows
#: that state it do not pool with rows that state one.
NOT_RECORDED = "not recorded"

_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: The field that holds the digest of every other field.
DIGEST_FIELD = "record_sha256"


class RecordRefused(ValueError):
    """A timing record that may not be used, refused by name (``code``)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ConsultRefused(ValueError):
    """A consult asked a question it cannot be asked, refused by name (``code``)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# The shape class: the axes of a key that a RUN supplies
# ---------------------------------------------------------------------------

class ShapeClass(NamedTuple):
    """What a run contributes to a key: every axis but the product and the band."""

    storage: str
    dimensions: int
    susceptibilities: int
    bracketed: bool
    cells: int


def cells_of(grid_shape: Sequence[int]) -> int:
    """The cell count of a grid shape: the product of its extents.

    THE ONE PLACE IT IS COMPUTED. The cutter bands rows on this number and the consult
    looks a run up by it, so both call this function rather than multiplying twice.
    """
    try:
        extents = [int(value) for value in grid_shape]
    except (TypeError, ValueError) as exc:
        raise ConsultRefused("unreadable_run_shape",
                             f"grid_shape {grid_shape!r} is not a sequence of "
                             f"integers") from exc
    if not extents or any(value < 1 for value in extents):
        raise ConsultRefused("unreadable_run_shape",
                             f"grid_shape {grid_shape!r} names no positive extents")
    total = 1
    for value in extents:
        total *= value
    return total


def shape_class(run_shape: Mapping[str, Any], *, bracketed: bool,
                cells: Optional[int] = None) -> ShapeClass:
    """The key axes of one run, from the run-shape dict the release tables read.

    ``run_shape`` is what ``fastpath._run_shape`` reports (and what every timing row
    records under ``plans.fused.run_shape``): ``dimensions``, ``grid_shape``,
    ``complex_storage`` and ``susceptibilities`` are read from it verbatim. The
    susceptibility CLASS is the count itself, the value the release tables pin.
    ``bracketed`` is not a property of the shape -- it is whether the candidate's seam
    carries the deposit-repair bracket on this run -- so the caller states it.
    """
    if not isinstance(run_shape, Mapping):
        raise ConsultRefused("unreadable_run_shape",
                             f"run_shape is a {type(run_shape).__name__}, not a mapping")
    for axis in ("dimensions", "grid_shape"):
        if run_shape.get(axis) is None:
            raise ConsultRefused("unreadable_run_shape",
                                 f"run_shape carries no {axis!r}")
    try:
        dimensions = int(run_shape["dimensions"])
        susceptibilities = int(run_shape.get("susceptibilities") or 0)
    except (TypeError, ValueError) as exc:
        raise ConsultRefused("unreadable_run_shape", str(exc)) from exc
    if not isinstance(bracketed, bool):
        raise ConsultRefused("unreadable_run_shape",
                             f"bracketed is {bracketed!r}, not a bool")
    return ShapeClass(
        storage="complex" if run_shape.get("complex_storage") else "real",
        dimensions=dimensions,
        susceptibilities=susceptibilities,
        bracketed=bracketed,
        cells=int(cells) if cells is not None else cells_of(run_shape["grid_shape"]),
    )


def class_string(product: str, storage: str, dimensions: int, susceptibilities: int,
                 bracketed: bool) -> str:
    """A key without its band: everything a lookup matches exactly."""
    return "|".join((str(product), str(storage), str(int(dimensions)),
                     str(int(susceptibilities)), BRACKETED if bracketed else CLEAN))


def key_string(product: str, storage: str, dimensions: int, susceptibilities: int,
               bracketed: bool, lo_cells: int, hi_cells: int) -> str:
    """The one spelling of a record key. The cutter writes it; the consult reads it."""
    return (class_string(product, storage, dimensions, susceptibilities, bracketed)
            + f"|{int(lo_cells)}-{int(hi_cells)}")


def parse_key(key: str) -> Dict[str, Any]:
    """A key string back into its axes. A product label never contains ``|``."""
    parts = str(key).split("|")
    band = re.match(r"^(\d+)-(\d+)$", parts[-1]) if len(parts) == 6 else None
    if (band is None or parts[1] not in STORAGES or parts[4] not in (BRACKETED, CLEAN)
            or not parts[0] or not parts[2].isdigit() or not parts[3].isdigit()):
        raise RecordRefused("malformed_record", f"key {key!r} is not "
                            "product|storage|dimensions|susceptibilities|"
                            "bracketed-or-clean|lo-hi")
    return {"product": parts[0], "storage": parts[1], "dimensions": int(parts[2]),
            "susceptibilities": int(parts[3]), "bracketed": parts[4] == BRACKETED,
            "band": {"lo_cells": int(band.group(1)), "hi_cells": int(band.group(2))}}


# ---------------------------------------------------------------------------
# The veto rule
# ---------------------------------------------------------------------------

#: Decimal places a margin is rounded to before its sign is read, so a row sitting on
#: its threshold is on it on every platform.
MARGIN_DECIMALS = 6


def veto_margin(ratio: float, fused_spread: float, baseline_spread: float) -> float:
    """By how much ONE row is slower than its spreads could explain. Positive is slower.

    ``ratio`` is baseline ms/step over fused ms/step, so above 1 the candidate is
    faster. Each spread is that leg's own window spread, (max - min) / median, on that
    row. The margin is ``1 - (fused_spread + baseline_spread) - ratio``: how far below
    1 the ratio sits once both legs are allowed to have moved by their full spread.
    """
    return round(1.0 - (float(fused_spread) + float(baseline_spread)) - float(ratio),
                 MARGIN_DECIMALS)


def slower_beyond_spread(ratio: float, fused_spread: float,
                         baseline_spread: float) -> bool:
    """Whether one row says the candidate is slower than what it displaces."""
    return veto_margin(ratio, fused_spread, baseline_spread) > 0.0


def least_margin(measurements: Sequence[Mapping[str, Any]]) -> float:
    """A key's margin: the SMALLEST any of its rows shows, each against its own spreads.

    A KEY VETOES ONLY WHEN EVERY ROW IN IT DOES. Pooled extremes are the wrong
    instrument: the most favourable ratio of one row set against the noisiest spread
    of another describes no measurement that was taken, and on a band that pools a
    3x loss at 24,000 cells with a near-tie at 2,400,000 it would clear the whole band.
    Row by row, a band is slower exactly when each size in it is.
    """
    return min(veto_margin(row["ratio"], row["fused_spread"], row["baseline_spread"])
               for row in measurements)


# ---------------------------------------------------------------------------
# Bytes and digests
# ---------------------------------------------------------------------------

def _canonical(document: Mapping[str, Any]) -> bytes:
    body = {key: value for key, value in document.items() if key != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def record_digest(document: Mapping[str, Any]) -> str:
    """sha256 of every field but the digest itself, in canonical JSON."""
    return hashlib.sha256(_canonical(document)).hexdigest()


def seal(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The document with its own digest stamped. Idempotent."""
    sealed = {key: value for key, value in document.items() if key != DIGEST_FIELD}
    sealed[DIGEST_FIELD] = record_digest(sealed)
    return sealed


def dumps(document: Mapping[str, Any]) -> str:
    """The record's bytes on disk: sorted keys, one-space indent, a final newline."""
    return json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True,
                      allow_nan=False) + "\n"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number_fact(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def route_facts_missing(route: Any) -> List[str]:
    """The route facts this mapping does not state, sorted. Empty means a full identity.

    A fact is unstated when its name is absent, its value has the wrong type, or it is
    ``None`` outside the one spelling that allows it: ``index`` and
    ``repairs_per_bracketed_step`` are both ``None`` together when nothing was
    bracketed, and neither is ``None`` alone.
    """
    if not isinstance(route, Mapping):
        return list(ROUTE_FACTS)
    missing = set()
    digest = route.get("deposit_repair_sha256")
    if not isinstance(digest, str) or not _SHA256.match(digest):
        missing.add("deposit_repair_sha256")
    for name in ("restricted", "fell_back"):
        if not isinstance(route.get(name), bool):
            missing.add(name)
    index, per_step = route.get("index"), route.get("repairs_per_bracketed_step")
    unbracketed = index is None and "index" in route
    if not unbracketed and index not in ROUTE_INDEXES:
        missing.add("index")
    if unbracketed:
        # both facts are stated, and both are None: a route that names no index states
        # no repair count either, and one that omits the key states nothing at all
        if per_step is not None or "repairs_per_bracketed_step" not in route:
            missing.add("repairs_per_bracketed_step")
    elif not _number_fact(per_step):
        missing.add("repairs_per_bracketed_step")
    return sorted(missing)


def route_id(route: Mapping[str, Any]) -> str:
    """One deposit-repair program, as a refusal and a record both spell it.

    THE ONE SPELLING. The cutter names a route with it, the record carries it, and the
    consult compares it, so a program cannot be written one way and asked another.
    """
    missing = route_facts_missing(route)
    if missing:
        raise ConsultRefused("unreadable_repair_route",
                             f"the route states no {', '.join(missing)}")
    digest = str(route["deposit_repair_sha256"])[:12]
    if route["index"] is None:
        return f"{digest}:no-bracketed-row"
    return (f"{digest}:{route['index']}:"
            f"{route['repairs_per_bracketed_step']:g}-per-step:"
            f"{'restricted' if route['restricted'] else 'full'}"
            f"{':fell-back' if route['fell_back'] else ''}")


def route_differences(record_route: Mapping[str, Any],
                      live_route: Mapping[str, Any]) -> List[str]:
    """The facts on which two routes disagree, sorted."""
    return sorted(name for name in ROUTE_FACTS
                  if record_route.get(name) != live_route.get(name))


def compared_emitters(record: Mapping[str, Any]) -> Dict[str, str]:
    """The emitter digests a record CAN be compared on: one recorded value each.

    A source whose rows did not agree on one recorded digest -- because some rows did
    not record it at all -- is left out here and named in the verdict instead. Pooling
    ``not recorded`` with a digest would make the barrier answer "the same" about two
    things it cannot compare.
    """
    barrier = record.get("subject_barrier") or {}
    digests = barrier.get("digests") or {}
    out = {}
    for source in barrier.get("sources") or ():
        values = list(digests.get(source) or ())
        # comparable exactly when the rows agree on ONE digest and none of them left the
        # question open; ``allowed_unpinned`` is the cutter's decision to admit such a
        # source, and this reads the consequence off the values themselves
        if len(values) == 1 and values[0] != NOT_RECORDED:
            out[source] = values[0]
    return out


def live_subject_sha256(table: str, root: Optional[str] = None) -> Dict[str, str]:
    """The barrier sources of ``table`` as they stand in the tree, by file digest.

    A FILE READ, NOT AN IMPORT, like :func:`repair_route_sha256`: asking which record
    applies must not pull a kernel table into the process.
    """
    if table not in TABLES:
        raise ConsultRefused("unknown_table", f"{table!r} is not one of {TABLES}")
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    return {source: file_sha256(os.path.join(root, SUBJECT_FILES[source]))
            for source in SUBJECT_SOURCES_BY_TABLE[table]}


def repair_route_sha256(path: Optional[str] = None) -> str:
    """The live deposit-repair route: the digest of ``deposit_repair.py`` as it stands.

    A FILE READ, NOT AN IMPORT, so asking does not pull the dispatch path into a
    process that only wants to know which record applies. The default is the module
    beside this one, which is the file every timing row digests under
    ``provenance.sha256.deposit_repair``.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "deposit_repair.py")
    return file_sha256(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _malformed(detail: str) -> RecordRefused:
    return RecordRefused("malformed_record", detail)


def _number(entry: Mapping[str, Any], name: str, key: str, *, positive: bool = False,
            optional: bool = False) -> Optional[float]:
    value = entry.get(name)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(f"key {key!r}: {name} is {value!r}, not a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise _malformed(f"key {key!r}: {name} is not finite")
    if value < 0 or (positive and value <= 0):
        raise _malformed(f"key {key!r}: {name} is {value!r}")
    return float(value)


def _validate_entry(key: str, entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise _malformed(f"key {key!r}: its value is not an object")
    axes = parse_key(key)
    status = entry.get("status")
    if status not in STATUSES:
        raise _malformed(f"key {key!r}: status is {status!r}, not one of {STATUSES}")
    if not isinstance(entry.get("bracketed"), bool):
        raise _malformed(f"key {key!r}: bracketed is {entry.get('bracketed')!r}")
    band = entry.get("band")
    if (not isinstance(band, Mapping)
            or not all(isinstance(band.get(edge), int) and not isinstance(band[edge], bool)
                       for edge in ("lo_cells", "hi_cells"))
            or band["lo_cells"] < 1 or band["lo_cells"] > band["hi_cells"]):
        raise _malformed(f"key {key!r}: band is {band!r}")
    for axis, value in axes.items():
        held = dict(entry[axis]) if axis == "band" else entry.get(axis)
        if held != value:
            raise _malformed(f"key {key!r}: the entry's {axis} is {held!r}, which is "
                             f"not what its key spells ({value!r})")
    displaces = entry.get("displaces")
    if (not isinstance(displaces, list)
            or not all(isinstance(arm, str) and arm for arm in displaces)):
        raise _malformed(f"key {key!r}: displaces is {displaces!r}, not a list of "
                         "arm labels")
    measurements = entry.get("measurements", [])
    if not isinstance(measurements, list):
        raise _malformed(f"key {key!r}: measurements is not a list")
    if status == "unmeasured":
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise _malformed(f"key {key!r}: an unmeasured key states no reason")
        if entry.get("ratio_max") is None:
            return
    for name in ("fused_ms_per_step", "baseline_ms_per_step", "ratio", "ratio_max"):
        _number(entry, name, key, positive=True)
    for name in ("fused_spread", "baseline_spread"):
        _number(entry, name, key)
    if not measurements:
        raise _malformed(f"key {key!r}: it states numbers and names no measurement "
                         "they came from")
    for index, row in enumerate(measurements):
        if not isinstance(row, Mapping):
            raise _malformed(f"key {key!r}: measurement {index} is not an object")
        _number(row, "ratio", f"{key} measurement {index}", positive=True)
        for name in ("fused_spread", "baseline_spread"):
            _number(row, name, f"{key} measurement {index}")
    margin = entry.get("veto_margin")
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or abs(margin - least_margin(measurements)) > 10.0 ** -MARGIN_DECIMALS):
        raise _malformed(f"key {key!r}: veto_margin is {margin!r} and its own "
                         f"measurements give {least_margin(measurements)}")
    stated = entry.get("slower_beyond_spread")
    if stated is not (least_margin(measurements) > 0.0):
        raise _malformed(f"key {key!r}: slower_beyond_spread is {stated!r}, which its "
                         f"own measurements do not give (least margin "
                         f"{least_margin(measurements)})")


def validate(record: Any, expect_table: Optional[str] = None) -> None:
    """Refuse, by name, a record that may not be consulted. Returns nothing."""
    if not isinstance(record, Mapping):
        raise _malformed(f"the record is a {type(record).__name__}, not an object")
    if record.get("schema") != SCHEMA:
        raise RecordRefused("unsupported_schema",
                            f"the record is stamped {record.get('schema')!r}; this "
                            f"reader accepts {SCHEMA!r} and nothing else")
    if record.get(DIGEST_FIELD) != record_digest(record):
        raise RecordRefused("record_digest_mismatch",
                            f"{DIGEST_FIELD} is {record.get(DIGEST_FIELD)!r} and the "
                            f"body digests to {record_digest(record)}: the record was "
                            "edited after it was cut, or never sealed")
    if record.get("table") not in TABLES:
        raise _malformed(f"table is {record.get('table')!r}, not one of {TABLES}")
    if not isinstance(record.get("recorded_utc"), str) or not _UTC.match(
            record["recorded_utc"]):
        raise _malformed(f"recorded_utc is {record.get('recorded_utc')!r}")
    route = record.get("route")
    missing = route_facts_missing(route)
    if missing:
        raise _malformed("route states no " + ", ".join(missing) + ": a record names "
                         "the whole program its rows ran, not its digest alone")
    if route.get("id") != route_id(route):
        raise _malformed(f"route id is {route.get('id')!r} and its own facts spell "
                         f"{route_id(route)}")
    barrier = record.get("subject_barrier")
    if (not isinstance(barrier, Mapping)
            or not isinstance(barrier.get("sources"), list)
            or not isinstance(barrier.get("digests"), Mapping)
            or not isinstance(barrier.get("allowed_unpinned"), list)):
        raise _malformed("subject_barrier names no sources, digests and "
                         "allowed_unpinned")
    for source in barrier["sources"]:
        values = barrier["digests"].get(source)
        if (not isinstance(values, list) or not values
                or not all(value == NOT_RECORDED
                           or (isinstance(value, str) and _SHA256.match(value))
                           for value in values)):
            raise _malformed(f"subject_barrier records {values!r} for {source!r}")
    keys = record.get("keys")
    if not isinstance(keys, Mapping):
        raise _malformed("keys is not an object")
    bands: Dict[str, List[Tuple[int, int, str]]] = {}
    for key, entry in keys.items():
        _validate_entry(key, entry)
        band = entry["band"]
        bands.setdefault(key.rsplit("|", 1)[0], []).append(
            (band["lo_cells"], band["hi_cells"], key))
    for held in bands.values():
        held.sort()
        for (_lo, hi, first), (lo, _hi, second) in zip(held, held[1:]):
            if lo <= hi:
                raise _malformed(f"bands overlap: {first!r} and {second!r}")
    if expect_table is not None and record["table"] != expect_table:
        raise RecordRefused("wrong_table",
                            f"the record was cut for the {record['table']!r} table and "
                            f"was asked for as the {expect_table!r} table's")


def load_record(path: str, expect_table: Optional[str] = None) -> Dict[str, Any]:
    """Read and validate one record. Every failure is a named refusal."""
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RecordRefused("unreadable_record", f"{path}: {exc}") from exc
    validate(record, expect_table=expect_table)
    return record


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """What one consult answered, with the evidence it answered on.

    ``vetoed`` is the only outcome that changes a plan: the candidate does not take
    the slots and what it would have displaced stays. ``not_vetoed`` and ``no_record``
    both leave today's rule in force; they differ in whether a measurement was read.
    """

    outcome: str
    reason: str
    mode: str
    table: str
    candidate: str
    displaces: Tuple[str, ...]
    key: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def vetoed(self) -> bool:
        return self.outcome == "vetoed"

    def as_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome, "reason": self.reason, "mode": self.mode,
                "table": self.table, "candidate": self.candidate,
                "displaces": list(self.displaces), "key": self.key,
                "evidence": json.loads(json.dumps(self.evidence))}

    def describe(self) -> str:
        """One line a route gate can print beside its ``selected`` map."""
        head = (f"{self.candidate} over {'+'.join(self.displaces) or 'nothing'} "
                f"[{self.table}, {self.mode}]: {self.outcome} ({self.reason})")
        evidence = self.evidence
        if evidence.get("ratio_max") is None:
            return head
        return (f"{head} -- key {self.key}: fused {evidence['fused_ms_per_step']:.3f} "
                f"ms/step against {evidence['baseline_ms_per_step']:.3f}, best row "
                f"{evidence['ratio_max']:.3f}x, least margin "
                f"{evidence['veto_margin']:+.3f}, {evidence['rows']} row(s), "
                f"{evidence['cells']:,} cells")


def _entry_evidence(record: Mapping[str, Any], entry: Mapping[str, Any],
                    cells: int) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "cells": cells,
        "band": dict(entry["band"]),
        "rows": entry.get("rows"),
        "displaces_measured": list(entry["displaces"]),
        "record_sha256": record.get(DIGEST_FIELD),
        "recorded_utc": record.get("recorded_utc"),
        "record_route": record["route"]["deposit_repair_sha256"],
        "record_route_id": record["route"]["id"],
        "emitters_compared": sorted(compared_emitters(record)),
        # named, because these are the sources the record could NOT compare
        "emitters_not_compared": sorted(
            set((record.get("subject_barrier") or {}).get("sources") or ())
            - set(compared_emitters(record))),
    }
    for name in ("fused_ms_per_step", "baseline_ms_per_step", "ratio", "ratio_min",
                 "ratio_max", "fused_spread", "baseline_spread", "veto_margin"):
        evidence[name] = entry.get(name)
    measurements = entry.get("measurements") or []
    evidence["cases"] = sorted({str(row.get("case")) for row in measurements})
    evidence["artifacts"] = sorted({str(row.get("artifact")) for row in measurements})
    return evidence


def consult(record: Optional[Mapping[str, Any]], *, table: str, candidate: str,
            displaces: Sequence[str], run_shape: Mapping[str, Any], bracketed: bool,
            mode: str, repair_route: Optional[Mapping[str, Any]],
            subject_sha256: Optional[Mapping[str, str]],
            cells: Optional[int] = None, validated: bool = False) -> Verdict:
    """May ``candidate`` take the slots of the products in ``displaces`` on this run?

    THE ARGUMENTS ARE WHAT THE TWO ARBITRATION SITES ALREADY HOLD. A fused pair about
    to absorb its seam knows its own label and the two arms the arm table selected
    (``_pair_may_absorb``'s ``arms``); a longer span about to supersede a shorter
    product knows both labels (``_superseded_by_a_longer_span``'s candidates). The
    run shape is ``fastpath._run_shape``'s dict and ``bracketed`` is whether the
    candidate's seam carries the deposit-repair bracket on this run.

    ``repair_route`` is the LIVE route the bracket will take, as
    :data:`ROUTE_FACTS` spells it, and ``subject_sha256`` the live emitter digests
    (:func:`live_subject_sha256`). Both are compared against what the record was cut
    on, and neither may be guessed: a route or emitter the caller cannot state is
    ``no_record``, because a record whose program cannot be shown to be this one is a
    record about another run.

    ``mode`` and the record are ARGUMENTS; this module reads no environment variable.
    ``span`` returns first, before the record is looked at: today's behaviour, exactly.

    The record is validated on every call unless ``validated`` says the caller holds it
    straight from :func:`load_record` -- a plan asks once per candidate, and re-digesting
    the whole record each time buys nothing the load did not already establish.
    """
    candidate = str(candidate)
    displaced = tuple(str(arm) for arm in displaces)

    def answer(outcome: str, reason: str, key: Optional[str] = None,
               evidence: Optional[Dict[str, Any]] = None) -> Verdict:
        return Verdict(outcome=outcome, reason=reason, mode=mode, table=str(table),
                       candidate=candidate, displaces=displaced, key=key,
                       evidence=evidence or {})

    if mode not in MODES:
        raise ConsultRefused("unknown_mode", f"mode is {mode!r}, not one of {MODES}")
    if mode == "span":
        return answer("no_record", "span_mode_record_not_consulted")
    if record is None:
        return answer("no_record", "no_record_supplied")
    if not validated:
        validate(record)
    shape = shape_class(run_shape, bracketed=bracketed, cells=cells)
    if record["table"] != table:
        return answer("no_record", "record_is_for_another_table",
                      evidence={"record_table": record["table"]})
    recorded = record["route"]
    # A BRACKETED KEY IS ASKED FOR THE WHOLE PROGRAM, A CLEAN ONE FOR THE DIGEST.
    # Most of a bracketed number IS the bracket, and one digest runs more than one
    # bracket; a clean key's rows carried no bracket at all (the cutter prices a clean
    # product only from plans in which it is the sole fused product), so what the
    # bracket does per step cannot move it.
    wanted_facts = ROUTE_FACTS if shape.bracketed else ("deposit_repair_sha256",)
    unstated = [name for name in wanted_facts
                if name in route_facts_missing(repair_route)]
    if unstated:
        return answer("no_record", "live_repair_route_not_stated",
                      evidence={"record_route_id": recorded["id"],
                                "not_stated": unstated,
                                "record_sha256": record.get(DIGEST_FIELD)})
    differs = [name for name in route_differences(recorded, repair_route)
               if name in wanted_facts]
    if differs:
        return answer("no_record", "record_is_for_another_repair_route",
                      evidence={"record_route": recorded["deposit_repair_sha256"],
                                "live_route": repair_route["deposit_repair_sha256"],
                                "record_route_id": recorded["id"],
                                "live_route_id": route_id(dict(
                                    recorded, **{name: repair_route[name]
                                                 for name in wanted_facts})),
                                "differs_on": differs,
                                "record_sha256": record.get(DIGEST_FIELD)})
    compared = compared_emitters(record)
    not_stated = sorted(source for source in compared
                        if not (subject_sha256 or {}).get(source))
    if not_stated:
        return answer("no_record", "live_emitters_not_stated",
                      evidence={"not_stated": not_stated,
                                "record_emitters": dict(compared),
                                "record_sha256": record.get(DIGEST_FIELD)})
    moved = sorted(source for source, digest in compared.items()
                   if subject_sha256[source] != digest)
    if moved:
        return answer("no_record", "record_is_for_another_emitter",
                      evidence={"differs_on": moved,
                                "record_emitters": dict(compared),
                                "live_emitters": {source: subject_sha256[source]
                                                  for source in compared},
                                "record_sha256": record.get(DIGEST_FIELD)})
    wanted = class_string(candidate, shape.storage, shape.dimensions,
                          shape.susceptibilities, shape.bracketed)
    held = sorted((entry["band"]["lo_cells"], entry["band"]["hi_cells"], key)
                  for key, entry in record["keys"].items()
                  if key.rsplit("|", 1)[0] == wanted)
    if not held:
        return answer("no_record", "class_not_in_record",
                      evidence={"class": wanted, "cells": shape.cells})
    inside = [key for lo, hi, key in held if lo <= shape.cells <= hi]
    if not inside:
        return answer("no_record", "outside_the_measured_bands",
                      evidence={"class": wanted, "cells": shape.cells,
                                "measured_bands": [[lo, hi] for lo, hi, _key in held]})
    key = inside[0]
    entry = record["keys"][key]
    evidence = _entry_evidence(record, entry, shape.cells)
    if entry["status"] != "measured":
        evidence["unmeasured_because"] = entry.get("reason")
        return answer("no_record", "key_unmeasured", key, evidence)
    if sorted(entry["displaces"]) != sorted(displaced):
        return answer("no_record", "baseline_is_not_what_the_candidate_displaces",
                      key, evidence)
    if entry["veto_margin"] > 0.0:
        return answer("vetoed", "slower_beyond_spread", key, evidence)
    return answer("not_vetoed", "not_slower_beyond_spread", key, evidence)


def consult_path(path: Optional[str], **question: Any) -> Verdict:
    """:func:`consult` against the record at ``path``.

    In mode ``span`` the path is never opened. In mode ``measured`` a path that does
    not exist is ``no_record`` -- a table with no record yet is today's behaviour --
    while a file that exists and is not a valid record REFUSES, because a record that
    cannot be read is a fault to surface rather than an absence to paper over.
    """
    if question.get("mode") == "span" or path is None or not os.path.exists(path):
        return consult(None, **question)
    return consult(load_record(path), validated=True, **question)
