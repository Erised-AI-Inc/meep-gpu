"""Measured cost as a slot arbiter: the timing record's format, and the consult that reads it.

In one line: a recorded measurement may withhold a slower fused product, and nothing else.
Today a fused product wins a slot on SHAPE, never on speed: a longer span displaces
the products it contains wherever it is admitted, with no size gate. This module is
the instrument that lets a MEASUREMENT veto that -- and only veto it.

WIRED in 1.0.1 (the measured-cost preference), and priced by size where a record
carries a cost model (:func:`consult_cost`). ``fastpath``'s composer rung (5) and
``metal_dispatch``'s (5M) consult it once per fused candidate the shape rule
installed, through ``fastpath._apply_preference``: a candidate the record measured
slower than its certified singles is withheld from that table's offer, the slot
composes again with the product left out, and the veto stands only if the singles
the record priced are what took the slots. Every route gate's ``selected`` map can
move with it, so the record's digest is written into the dispatch record beside the
``selected`` map (``preference.records``), and ``MEEP_GPU_PREFER=span`` reproduces
the shape rule byte for byte.

=============================================================================
What is decided here, and what never is
=============================================================================

ARBITRATION ONLY. The record may change WHO holds a slot. It can never change what a
kernel computes: a vetoed candidate leaves the products it would have displaced in
their slots, and those are certified products running their own certified arithmetic.

A veto first, not a ranking. One direction, failing closed:

    A candidate may not displace the products it supersedes when the record says it
    is SLOWER than them on this shape class, by more than the recorded spreads. With
    no record for the class, behaviour is today's -- the longer span wins.

So every path that is not "measured, and slower beyond spread" answers ``no_record``
or ``not_vetoed``, and both mean the same thing to the caller: install as today.

=============================================================================
THE RECORD
=============================================================================

One JSON document per kernel table per card, cut by
``parity/meep_gpu/cut_timing_record.py`` from timing rows and never written by hand
(it carries a digest of its own body, and a record whose digest does not match
REFUSES). :func:`record_path` spells where it lives --
``<table>_kernels/timing_cc86.json``, with ``timing_cc86.report.txt`` beside it, and
for the Metal table ``metal_kernels/timing_applegpu_g13s.json``, named for the Apple
GPU family -- because a kernel's cost is a property of the ARCHITECTURE that ran it:
the same bytes on another architecture have another ratio, and that ratio is the whole
veto. So the architecture is in the file name (a reader sees from a directory listing
which architectures have been timed and which have not), a new architecture's record
is ADDED beside the others and never replaces one, :func:`validate` refuses a record
whose rows do not name exactly one of them, and :func:`consult` answers ``no_record``
when the device it is asked about is not that one.

The rows ran on one CARD, and the card is in the name too where it is not the
architecture's representative (the section "One record per card" below): the bare name
above is the architecture's REPRESENTATIVE, read by every card of the architecture that
has no record of its own, and ``timing_cc86-nvidia_geforce_rtx_3090.json`` (the
architecture, ``-``, the card's slug, :func:`card_slug`) is that one card's own record,
read by that card alone. :func:`load_record` refuses a card-named file whose rows name
another card (``record_named_for_another_card``). Its ``keys`` map is::

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
  A band's interior is interpolated. Its edges are measured; a size between them need
  not have been run at all, and the key's verdict is applied to it. That is why an
  edge must be corroborated: a single session sets no band boundary, because the
  window spreads beside a row price only that session's own noise and not the spread
  BETWEEN sessions, which the corpus shows at a few parts in a thousand to two parts
  in a hundred -- the size of the thinnest margins measured.
* A key VETOES only when every row that fed it is slower than that row's own two
  window spreads allow (:func:`least_margin`); the pooled medians and extremes beside
  it are for reading. A key with too few rows is written ``unmeasured`` with the reason.

A ``/3`` record (:data:`SCHEMA_COSTS`) carries all of that and a COST MODEL as well:
per product class, the fused and the baseline ms/step at each size it measured, which
:func:`consult_cost` reads at the run's own cell count (the last section of this
module). Its points are recomputed from the record's own measurements when it is read.
A ``/2`` record is read exactly as it always was.

The record also names the deposit-repair ROUTE its rows ran (``route``) -- the digest
of ``deposit_repair.py`` AND what the bracket did per step: the index, the repairs per
bracketed step, whether any source was restricted, whether any fell back. One digest
carries more than one route (the corpus has ``e4f88c1e`` running 3-per-step full and
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
The environment-variable contract
=============================================================================

This module reads NO environment variable (only ``fastpath.py``
and ``subnormal_policy.py`` do). The mode and the record path are ARGUMENTS. In mode
``span`` the consult returns before touching its record or any file, so it reproduces
today's behaviour exactly and cannot fail.

=============================================================================
What a wiring step must assume, stated so it can be argued with
=============================================================================

A key prices a whole PLAN -- (B_singles + D_singles) / (B_fused + D_fused), attributed
to the one product that carries the bracket -- and applying it to a slot decision takes
four assumptions that no row of the corpus establishes:

1. ADDITIVITY: a plan's per-step cost is its slots' costs plus a bracket attached to the
   SOURCE's seam.
2. The bracket vanishes with the veto: the post-veto plan (the partner still fused, the
   vetoed product back on single arms) was never timed and no row of that shape exists.
3. The verdict transfers across compositions: a key has no co-fused axis, so a number
   measured beside a fused partner is applied to a run where that partner is not
   admitted. The measurements list their ``co_fused`` partners so the residue is at
   least visible: a partner that WINS dilutes the loss (an under-veto, safe), one that
   LOSES inflates it (a false veto, unsafe), and near a crossover the partner's few
   percent is the decision.
4. ``bracketed`` is computed at arbitration exactly as the bench computed it. That one
   is feasible today: the call site holds ``context.sources`` / ``context.grid`` and
   ``deposit_repair.seam_source_reasons(..., 'B'/'D')`` answers the same question.

The veto direction survives 1-3 (they move the ratio toward a weaker veto or admit a
partner's few percent); a RANKING would not, which is why a ranking is deferred.

=============================================================================
What the wiring site must do with a record it cannot read
=============================================================================

Decided, so that the wiring does not have to: a record that is present and invalid is
a fault, and :func:`consult_path` RAISES :class:`RecordRefused` on it rather than
papering it over as an absence. That raise is for TOOLS -- a gate, a cutter, a report.
The dispatch path must not take a plan-time exception from a file: it loads the record
ONCE, at the top of a run, inside a ``try`` that catches :class:`RecordRefused`, prints
the refusal beside the ``selected`` map and carries ``None`` from there on, which every
consult answers ``no_record`` (today's behaviour). One read, one place to fail, and a
loud one.

WHICH file it loads is decided the same way, off the LIVE device: :func:`record_path`
for its architecture and its card (the name the run's ``environment.device.name``
records) when that card has a record of its own, else the architecture's representative
at the bare name. A card's own record that answers ``no_record`` for a barrier (another
route, another emitter) is the answer: the run does not fall through to the
representative, which was cut on another card. A host whose architecture has never
been timed reads no record at all -- it is never priced off another architecture's
ratios, key by key.

=============================================================================
The portable prior: an architecture with no record (fast by default, measured cost)
=============================================================================

"No record, so the shape rule" would make speed depend on whether a timing run has
reached this architecture: an uncertified GPU, a new compute capability, a new Apple
GPU family, would keep a fused product that each timed architecture of its vendor
measured slower than its singles. That must never happen. So a table with no record of
its own for the live architecture is arbitrated by :func:`portable_prior` instead,
consulted through :func:`consult_prior`, and the dispatch record and the run's NOTE
line say so. So is a product the live architecture's record is SILENT on -- no key
for its class, a key with too few rows, a size outside the bands -- because there the
architecture is in the same position (``fastpath.PRIOR_FILLS_RECORD_SILENCE``); a
measured verdict, VETO or keep, always wins over the prior.

THE RULE (:data:`PRIOR_RULE`). A fused product whose seam carries the deposit-repair
bracket, on a grid of at most ``bracketed_veto_up_to_cells`` cells, takes its certified
singles; every other candidate keeps the shape rule. The ceiling is DATA, derived per
architecture from every shipped record of that architecture (every table of its vendor,
every card) that holds on the live tree (the deposit-repair digest and that record's own
emitters) and measures at least one key: the largest upper edge of a measured bracketed
VETO band that lies below every measured bracketed keep band of that architecture. A
keep never transfers -- a keep IS the shape rule -- and a clean key never transfers,
because a clean product's verdict is the kernels' and the kernels are the
architecture's.

PER VENDOR, NEVER ACROSS VENDORS (:data:`VENDOR_BY_TABLE`). The Triton and the
hand-written CUDA tables read a prior derived from NVIDIA records only, and the Metal
table one derived from Apple records only. A crossover is a ratio of host work to kernel
work on one vendor's stack -- its driver, its launch path, its memory -- so an NVIDIA
crossover is no measurement of an Apple GPU, nor the reverse. A vendor none of whose
records holds on the live tree has no prior: its tables keep the shape rule, and
``why_none`` says why. The prior is derived for every vendor and architecture at once
(``by_vendor[vendor]["by_architecture"]``), and each verdict it gives names its vendor,
the live architecture, the architecture its ceiling came from and the records of that
architecture, so a dispatch record says whose measurements arbitrated it.

FROM THE NEAREST TIMED ARCHITECTURE (:func:`nearest_architecture`). A table whose live
architecture has no record of its own reads the prior of its vendor's nearest TIMED
architecture -- one with a record that holds on the tree and measures a key -- never the
vendor's records pooled: for an NVIDIA card the nearest compute capability, read as the
number ``major * 10 + minor`` (8.6 is 86), the lower one winning a tie; for an Apple GPU
the closest chip generation, read off the GPU family's spelling ``applegpu_g<generation>
<variant>`` (``applegpu_g13s`` is generation 13), the architecture itself first, then the
smallest difference in generation, then the lower generation, then the first spelling in
order. A live architecture or a timed one whose spelling the rule cannot read is not
guessed at: the table answers ``no_record`` naming it (``live_architecture_unparsed``,
``prior_architecture_unparsed``) and keeps the shape rule. The nearest timed
architecture's prior is the answer even when it holds no ceiling (its records measured
no bracketed veto below their keeps): a farther architecture is not asked instead. A
product the live architecture's own record is silent on reads the same rule, which
finds that architecture itself when its records measure a key. Only the bracketed
veto crosses architectures, never a whole record, and never across vendors.

A RELEASE SHIPS A TIMED RECORD OF EACH VENDOR (:func:`timed_vendors`). Every release
ships at least one timed NVIDIA record and one timed Apple record: a record that loads,
holds on the tree it ships in (:func:`barrier_refusal`, the very test the prior uses a
record by) and measures at least one key. A record that prices nothing, or that would
answer ``no_record`` on its own architecture, arbitrates no GPU of its vendor, so it does
not count. :func:`timed_vendors` is the one home of that rule: the package test and the
public export both ask it, the export of the exported tree's own copy.

Why a bracketed veto transfers and nothing else does. What a bracketed key prices is
mostly the bracket: ``LeadingRepairPlan.save`` before the fused launch and
``TrailingRepairPlan.apply`` after it, host-issued gathers and scatters around the
source's seam. Measured on the recorded architecture it is flat: the two-pair fused
step reads 0.74-0.88 ms/step from 24,000 to 2,400,000 cells on both NVIDIA tables while
the singles it replaced grow with the grid, so the fused product loses by a FIXED
per-step host cost until the kernels' own saving outgrows it. That cost is the host's,
not the GPU's, which is why it transfers between architectures that one vendor's stack
drives, and only between those. Below the smallest measured size the fixed cost only weighs more: in the
2026-10-16 records every bracketed row at or below 2,400,000 cells reads below 1, 70 of
70 on the two NVIDIA tables from 250 cells up (0.42-0.81x), and every bracketed row at
5,400,000 cells reads above 1 (8 of 8, 1.02-1.04x).

What it costs when it is wrong, which is why it is a prior and not a measurement. A
GPU slower than the recorded one has its crossover lower, so between that crossover
and the ceiling the prior keeps singles that would have lost by at most the fused
product's asymptotic gain (1.02-1.04x at 5,400,000 cells, the measured keeps). A
GPU faster than the recorded one has its crossover higher, and above the ceiling the
shape rule stands where singles may still win. Against that, the shape rule it
replaces loses 1.4-2.4x on every bracketed two-pair plan below the band. The rule is
asymmetric on purpose, in the direction the measurements give.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

#: The record format this module reads. A record stamped otherwise refuses by name.
#: ``/2`` (2026-09-20) carries a whole route identity and a subject barrier; a ``/1``
#: record compared the repair digest alone and named no emitter, so it is refused
#: rather than read with its guards missing.
SCHEMA = "meep_gpu.dispatch_preference.timing/2"

#: The record format that also carries a size-aware COST MODEL (:func:`consult_cost`):
#: everything ``/2`` carries, plus per-size cost points for every product class it
#: measured (``costs``), recomputed from the record's own measurements when it is read.
#: A ``/2`` record is read exactly as before; only a ``/3`` record is priced by size.
SCHEMA_COSTS = "meep_gpu.dispatch_preference.timing/3"

#: Every stamp this reader accepts.
READABLE_SCHEMAS = (SCHEMA, SCHEMA_COSTS)

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
#: What this cannot cover, and why it is not spelled here: a product's OWN device text
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

    The one place it is computed. The cutter bands rows on this number and the consult
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

    A key vetoes only when every row in it does. Pooled extremes are the wrong
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

    The one spelling. The cutter names a route with it, the record carries it, and the
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

    A file read, not an import, like :func:`repair_route_sha256`: asking which record
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

    A file read, not an import, so asking does not pull the dispatch path into a
    process that only wants to know which record applies. The default is the module
    beside this one, which is the file every timing row digests under
    ``provenance.sha256.deposit_repair``.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "deposit_repair.py")
    return file_sha256(path)


# ---------------------------------------------------------------------------
# One record per table per compute capability
# ---------------------------------------------------------------------------

#: The one spelling of a compute capability, the one every ledger entry and every run
#: record writes: ``8.6``. CuPy hands the same number out three ways (``"86"``, a
#: major/minor pair, the string), and ``fastpath`` is the single place those are
#: normalised. This module reads FILES and imports no kernel table (the
#: environment-variable contract above), so it accepts the normal spelling and refuses
#: the others BY NAME rather than keeping a second normaliser that could disagree with
#: that one and decide a comparison by which reader answered first.
CAPABILITY = re.compile(r"^[1-9][0-9]*\.[0-9]$")

#: The Apple GPU family, the Metal table's architecture: ``applegpu_g13s``. A Metal row
#: stamps it where an NVIDIA row stamps its compute capability
#: (``environment.device.compute_capability``), and it is the spelling the Metal run
#: records key on (``metal_runs.ARCHITECTURE_KEY``; ``test_dispatch_preference`` holds
#: the two patterns equal, because this module imports no package module).
APPLE_FAMILY = re.compile(r"^applegpu_[a-z0-9_]+$")

#: Which spelling each table's architecture takes. An NVIDIA table's record is named for
#: a compute capability and the Metal table's for an Apple GPU family; a value of the
#: other spelling is refused by name, so neither table's record can be cut or looked up
#: under the other's architecture.
ARCHITECTURE_BY_TABLE = {"triton": CAPABILITY, "cuda": CAPABILITY, "metal": APPLE_FAMILY}

#: Whose GPUs run each table: the vendor a table's portable prior is derived for, from that
#: vendor's records alone (the module docstring, "The portable prior"). It follows the
#: architecture spelling above: a compute capability is NVIDIA's, an Apple GPU family Apple's.
VENDOR_BY_TABLE = {"triton": "nvidia", "cuda": "nvidia", "metal": "apple"}

#: Every vendor, in the order the prior lists them.
VENDORS = tuple(sorted(set(VENDOR_BY_TABLE.values())))


def architecture_refusal(table: str, architecture: Any) -> Optional[str]:
    """Why ``architecture`` cannot name a record of ``table``, or None when it can."""
    pattern = ARCHITECTURE_BY_TABLE.get(table)
    if pattern is None:
        return f"{table!r} is not one of {TABLES}"
    if pattern.match(str(architecture)):
        return None
    if table == "metal":
        return (f"{architecture!r} is not an Apple GPU family spelled applegpu_<name>, "
                f"which is how a Metal row names the architecture it ran on")
    return (f"{architecture!r} is not a compute capability spelled major.minor, which "
            f"is how every record names the device its rows ran on")


def record_capabilities(record: Any) -> Tuple[str, ...]:
    """The compute capabilities a record's admitted rows were timed on, sorted.

    Read off ``host``, which is what the ROWS stamped from the device they ran on, so
    the answer is the card's and never the cutter's. Values come back AS RECORDED: a
    malformed one is reported rather than dropped, because dropping it would let a
    record carrying one good capability beside one unreadable one read as a record
    about a single architecture. :func:`validate` is where that is refused.
    """
    hosts = record.get("host") if isinstance(record, Mapping) else None
    if not isinstance(hosts, (list, tuple)):
        return ()
    # An unreadable row is kept, not dropped. Dropping it would let a record carrying
    # one good capability beside one unstamped host read as a single-architecture
    # record, which is exactly the case :func:`validate` exists to refuse: it counts
    # these, so the sentinel is what makes the count wrong and the refusal fire.
    seen = set()
    for host in hosts:
        value = host.get("compute_capability") if isinstance(host, Mapping) else None
        text = "" if value is None else str(value).strip()
        seen.add(text or "<no compute capability stamped>")
    return tuple(sorted(seen))


def record_path(table: str, architecture: str, card: Any = None,
                root: Optional[str] = None) -> str:
    """Where ``table``'s record for ONE architecture lives, and for one card. The one spelling.

    The cutter writes this path and the consult looks it up, so a record cannot be
    cut under a name nothing reads -- or, worse, under a name that is read for another
    architecture or another card. The report lands beside it, ``.json`` replaced by
    ``.report.txt``.

    ``root`` defaults to the package directory, beside the kernel tables, which is
    where the shipped records are. An NVIDIA table's record is named for its compute
    capability (``timing_cc86.json``) and the Metal table's for its Apple GPU family
    (``timing_applegpu_g13s.json``); a value in the other table's spelling, or in
    neither, is refused by name (:func:`architecture_refusal`). That bare name is the
    architecture's REPRESENTATIVE. With ``card`` (a device name as a run records it, or
    its slug), the name is that card's own record beside it:
    ``timing_cc86-nvidia_geforce_rtx_3090.json`` (:func:`card_slug`; a name that leaves
    no slug is refused, ``unnamed_card``). A second architecture's record lands beside
    the first, never in it, and so does a second card's.
    """
    if table not in TABLES:
        raise ConsultRefused("unknown_table", f"{table!r} is not one of {TABLES}")
    refusal = architecture_refusal(table, architecture)
    if refusal is not None:
        raise ConsultRefused("unreadable_capability", refusal)
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    name = (str(architecture) if table == "metal"
            else f"cc{str(architecture).replace('.', '')}")
    if card is not None:
        name += CARD_SEPARATOR + card_slug(card)
    return os.path.join(root, f"{table}_kernels", f"timing_{name}.json")


# ---------------------------------------------------------------------------
# One record per card: the names a record is read and written under
# ---------------------------------------------------------------------------
#
# A record's rows ran on one card, and its ratios are that card's. The card is the
# exact device name the rows stamp (``host[].device``), the name a run's dispatch record
# carries as ``environment.device.name``. Its SLUG is that name in lower case with every
# run of characters outside ``[a-z0-9]`` one ``_`` and none at either end:
# ``NVIDIA GeForce RTX 3090`` is ``nvidia_geforce_rtx_3090``. Cards match by slug, never
# by prefix or family: an RTX 3090 Ti is not an RTX 3090. A card's own record is named
# ``timing_<architecture>-<slug>.json``; the bare ``timing_<architecture>.json`` is the
# architecture's representative, cut on whichever card its ``host[]`` names. Neither a
# slug nor an architecture's spelling contains ``-``, so a name reads back one way
# (:func:`record_name_parts`), and every ``timing_*.json`` glob finds both.

#: What separates a card record's slug from its architecture in a file name.
CARD_SEPARATOR = "-"
#: The shape of a slug.
CARD_SLUG = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
#: The runs of characters a slug does not keep.
_NOT_IN_A_SLUG = re.compile(r"[^a-z0-9]+")
#: A record's file name: an architecture, and a card's slug when it is a card's own.
_RECORD_NAME = re.compile(r"^timing_(?P<architecture>[a-z0-9_]+)(?:" + CARD_SEPARATOR
                          + r"(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*))?\.json$")
#: An NVIDIA architecture as a file name spells it: ``cc86`` is compute capability 8.6.
_NVIDIA_SPELLING = re.compile(r"^cc(?P<major>[1-9][0-9]*)(?P<minor>[0-9])$")


def card_slug(name: Any) -> str:
    """A card's part of a file name: ``NVIDIA GeForce RTX 3090`` -> ``nvidia_geforce_rtx_3090``.

    Lower case; every maximal run of characters outside ``[a-z0-9]`` one ``_``; none at
    either end. Deterministic and idempotent, so the rows' stamp (``host[].device``) and
    a run's ``environment.device.name`` of one card give one spelling. A name that leaves
    nothing names no card and is refused (``unnamed_card``).
    """
    text = "" if name is None else str(name)
    slug = _NOT_IN_A_SLUG.sub("_", text.lower()).strip("_")
    if not slug:
        raise RecordRefused("unnamed_card",
                            f"the device name {name!r} names no card, and a record is the "
                            "evidence of one named card")
    return slug


def record_devices(record: Any) -> Tuple[str, ...]:
    """The exact device names a record's admitted rows were timed on, sorted.

    Read off ``host``, as :func:`record_capabilities` reads the architecture; a row that
    stamps no name is kept as ``""`` rather than dropped, so a record with one named and
    one unnamed device does not read as one card's.
    """
    hosts = record.get("host") if isinstance(record, Mapping) else None
    if not isinstance(hosts, (list, tuple)):
        return ()
    return tuple(sorted({str((host.get("device") if isinstance(host, Mapping) else None)
                             or "") for host in hosts}))


def record_name_parts(path: str) -> Optional[Tuple[str, Optional[str]]]:
    """``(architecture, slug)`` a record's file name says, or None for no record's name.

    The architecture comes back as the rows stamp it (``8.6``, ``applegpu_g13s``); the
    slug is None for an architecture's representative. A name is read back only when
    :func:`record_path` spells that architecture the same way, so this is the inverse of
    the one spelling and never a second one.
    """
    match = _RECORD_NAME.match(os.path.basename(path))
    if match is None:
        return None
    spelled = match.group("architecture")
    nvidia = _NVIDIA_SPELLING.match(spelled)
    table, architecture = (("triton", f"{nvidia.group('major')}.{nvidia.group('minor')}")
                           if nvidia else ("metal", spelled))
    try:
        if os.path.basename(record_path(table, architecture, root="")) != (
                f"timing_{spelled}.json"):
            return None
    except ConsultRefused:
        return None
    return architecture, match.group("slug")


def _name_refusal(path: str, record: Mapping[str, Any]) -> Optional[str]:
    """Why the card in ``path``'s name is not the card ``record`` was cut on, or None.

    Only a card's own name says whose rows are in the file; the representative's bare
    name holds any one card's record, and a name of no record (a candidate in a staging
    directory, say) says nothing to disagree with.
    """
    parts = record_name_parts(path)
    if parts is None or parts[1] is None:
        return None
    devices = record_devices(record)
    slugs = []
    for device in devices:
        try:
            slugs.append(card_slug(device))
        except RecordRefused:
            slugs.append("<no card>")
    if slugs == [parts[1]]:
        return None
    return (f"{os.path.basename(path)} is named for the card {parts[1]} and its rows were "
            f"timed on {list(devices)}: a card's own record is read by that card alone, "
            "so its name must be the card its rows ran on")


# ---------------------------------------------------------------------------
# How near two architectures of one vendor are
# ---------------------------------------------------------------------------

#: The Apple GPU family as the distance rule reads it: ``applegpu_g13s`` is chip
#: generation 13, variant ``s``. The spelling Metal reports for a device
#: (``MTLDevice.architecture.name``) and every Metal row and record stamps; a family
#: :data:`APPLE_FAMILY` admits in another shape names a record, but no generation, so
#: no distance is read off it.
APPLE_GENERATION = re.compile(r"^applegpu_g(?P<generation>[1-9][0-9]*)(?P<variant>[a-z])$")

#: What the nearest timed architecture is nearest BY, per vendor, as a verdict names it.
NEAREST_BASIS = {"nvidia": "nearest compute capability", "apple": "closest chip generation"}

#: The basis a verdict names when the prior came from the live architecture itself.
OWN_ARCHITECTURE = "own architecture"


def architecture_number(vendor: str, architecture: Any) -> Optional[int]:
    """The number the nearest rule measures ``architecture``'s distance in, or None.

    NVIDIA: the compute capability ``major * 10 + minor`` (``8.6`` is 86, ``9.0`` is 90),
    from the one spelling :data:`CAPABILITY` admits. Apple: the chip generation
    (``applegpu_g13s`` is 13, :data:`APPLE_GENERATION`). None for any other spelling, for
    no architecture at all and for a vendor of neither: a distance the rule cannot read
    is never guessed.
    """
    text = "" if architecture is None else str(architecture)
    if vendor == "nvidia":
        if not CAPABILITY.match(text):
            return None
        major, minor = text.split(".")
        return int(major) * 10 + int(minor)
    if vendor == "apple":
        match = APPLE_GENERATION.match(text)
        return None if match is None else int(match.group("generation"))
    return None


def nearest_architecture(vendor: str, live: Any, timed: Sequence[str]
                         ) -> Optional[Tuple[str, str, int]]:
    """``(architecture, basis, distance)``: the timed architecture nearest ``live``.

    ``timed`` are the vendor's timed architectures, as their records stamp them. The
    order: the live architecture itself; then the smallest distance in
    :func:`architecture_number`; then the lower number (on NVIDIA the lower compute
    capability wins a tie, on Apple the lower generation); then the first spelling in
    order (two Apple families of one generation). None when ``timed`` is empty. A live
    spelling, or a timed one, whose distance cannot be read is refused by name
    (:class:`ConsultRefused`, ``live_architecture_unparsed`` or
    ``prior_architecture_unparsed``): which architecture is nearest is never guessed.
    """
    if vendor not in NEAREST_BASIS:
        raise ConsultRefused("unknown_vendor", f"{vendor!r} is not one of {VENDORS}")
    if not timed:
        return None
    here = architecture_number(vendor, live)
    if here is None:
        raise ConsultRefused("live_architecture_unparsed", (
            f"the device's architecture {live!r} is not a {vendor} architecture the "
            f"nearest rule can read a distance from, so which timed architecture is "
            f"nearest is not known"))
    unread = sorted(str(item) for item in timed if architecture_number(vendor, item) is None)
    if unread:
        raise ConsultRefused("prior_architecture_unparsed", (
            f"the timed {vendor} architecture(s) {unread} cannot be read for a distance, "
            f"so which timed architecture is nearest {live} is not known"))

    def order(item: str) -> Tuple[bool, int, int, str]:
        number = architecture_number(vendor, item)
        return (str(item) != str(live), abs(number - here), number, str(item))

    best = min((str(item) for item in timed), key=order)
    basis = OWN_ARCHITECTURE if best == str(live) else NEAREST_BASIS[vendor]
    return best, basis, abs(architecture_number(vendor, best) - here)


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
    if record.get("schema") not in READABLE_SCHEMAS:
        raise RecordRefused("unsupported_schema",
                            f"the record is stamped {record.get('schema')!r}; this "
                            f"reader accepts {' and '.join(map(repr, READABLE_SCHEMAS))} "
                            "and nothing else")
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
    capabilities = record_capabilities(record)
    if (len(capabilities) != 1
            or architecture_refusal(record["table"], capabilities[0]) is not None):
        raise RecordRefused(
            "host_names_no_single_capability",
            f"the admitted rows name {list(capabilities)} as the compute capability "
            f"they ran on, and one record prices one architecture: a record naming "
            f"none cannot say which architecture its ratios describe, and one naming "
            f"two pooled two architectures' kernels into a single number -- the number "
            f"a veto is read off")
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
    # A record cut from part of a campaign names what it lacks (``missing``, optional):
    # each released (product, case) that no admitted row timed, with the reason. The
    # consult never reads it -- an absent key already answers ``no_record`` -- so it
    # is checked only for shape, and a record that names nothing missing omits it.
    missing_block = record.get("missing")
    if missing_block is not None:
        if not isinstance(missing_block, list) or not missing_block:
            raise _malformed("missing is present and is not a non-empty list")
        for item in missing_block:
            if (not isinstance(item, Mapping)
                    or not all(isinstance(item.get(name), str) and item.get(name)
                               for name in ("product", "case", "reason"))):
                raise _malformed(f"missing names {item!r}, not a product, a case and "
                                 "a reason")
    _validate_costs(record)
    if expect_table is not None and record["table"] != expect_table:
        raise RecordRefused("wrong_table",
                            f"the record was cut for the {record['table']!r} table and "
                            f"was asked for as the {expect_table!r} table's")


def load_record(path: str, expect_table: Optional[str] = None) -> Dict[str, Any]:
    """Read and validate one record. Every failure is a named refusal.

    A file named as one card's own record (``timing_<architecture>-<slug>.json``) whose
    rows were timed on another card, or on more than one, is refused
    (``record_named_for_another_card``): that card alone reads the file, so a name that
    disagrees with its content would price one card off another's rows.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RecordRefused("unreadable_record", f"{path}: {exc}") from exc
    validate(record, expect_table=expect_table)
    why = _name_refusal(path, record)
    if why is not None:
        raise RecordRefused("record_named_for_another_card", why)
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


def consult(record: Optional[Mapping[str, Any]], *, table: str,
            capability: Optional[str], candidate: str,
            displaces: Optional[Sequence[str]], run_shape: Mapping[str, Any],
            bracketed: bool,
            mode: str, repair_route: Optional[Mapping[str, Any]],
            subject_sha256: Optional[Mapping[str, str]],
            cells: Optional[int] = None, validated: bool = False) -> Verdict:
    """May ``candidate`` take the slots of the products in ``displaces`` on this run?

    The arguments are what the two arbitration sites already hold. A fused pair about
    to absorb its seam knows its own label and the two arms the arm table selected
    (``_pair_may_absorb``'s ``arms``); a longer span about to supersede a shorter
    product knows both labels (``_superseded_by_a_longer_span``'s candidates). The
    run shape is ``fastpath._run_shape``'s dict and ``bracketed`` is whether the
    candidate's seam carries the deposit-repair bracket on this run.

    ``capability`` is the compute capability of the device this plan will run on, in
    the one spelling a record names (``8.6``). It is REQUIRED and it is COMPARED, not
    normalised: a record prices the architecture its rows ran on and no other, so a
    capability that is not the record's answers ``no_record``, and so does a device
    whose capability could not be read (``None``) -- a run that cannot say which
    architecture it is on may not be priced from a card it may not be.

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

    ``displaces`` is ``None`` only on the composer's first ask, before the slots have
    been composed without the candidate, when what would take them is not yet known:
    the baseline comparison is then skipped and the evidence says so
    (``displaces_checked: False``). Such a verdict is a CANDIDATE veto and never a
    decision -- the composer withholds the product, composes again, and asks once more
    with the arms that took the slots, which is the only answer it acts on.
    """
    candidate = str(candidate)
    displaced = (() if displaces is None
                 else tuple(str(arm) for arm in displaces))

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
    # The architecture is a barrier, like the repair route and the emitters below it:
    # the ratio a record carries was measured on one card's instruction set and
    # occupancy, and applying it to another is a false veto waiting to happen. An
    # unread device is treated as the mismatch it might be.
    recorded_capabilities = record_capabilities(record)
    if capability is None:
        return answer("no_record", "capability_unread",
                      evidence={"record_capability": list(recorded_capabilities),
                                "record_sha256": record.get(DIGEST_FIELD)})
    if list(recorded_capabilities) != [str(capability)]:
        return answer("no_record", "record_is_for_another_capability",
                      evidence={"record_capability": list(recorded_capabilities),
                                "live_capability": str(capability),
                                "record_sha256": record.get(DIGEST_FIELD)})
    recorded = record["route"]
    # A bracketed key is asked for the whole program, a clean one for the digest.
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
    evidence["displaces_checked"] = displaces is not None
    if displaces is not None and sorted(entry["displaces"]) != sorted(displaced):
        return answer("no_record", "baseline_is_not_what_the_candidate_displaces",
                      key, evidence)
    if entry["veto_margin"] > 0.0:
        return answer("vetoed", "slower_beyond_spread", key, evidence)
    return answer("not_vetoed", "not_slower_beyond_spread", key, evidence)


def consult_path(path: Optional[str], **question: Any) -> Verdict:
    """:func:`consult` against the record at ``path``, which is :func:`record_path`'s.

    The question (``capability`` among it) travels through untouched, so the file this
    opens and the device the record is compared against are stated by the same caller.

    In mode ``span`` the path is never opened. In mode ``measured`` a path that does
    not exist is ``no_record`` -- a table with no record yet is today's behaviour --
    while a file that exists and is not a valid record REFUSES, because a record that
    cannot be read is a fault to surface rather than an absence to paper over.
    """
    if question.get("mode") == "span" or path is None or not os.path.exists(path):
        return consult(None, **question)
    return consult(load_record(path), validated=True, **question)


# ---------------------------------------------------------------------------
# The portable prior: an architecture with no record of its own
# ---------------------------------------------------------------------------

#: The prior's one rule, as the dispatch record and the NOTE line name it.
PRIOR_RULE = "bracketed_seam_at_or_below_the_recorded_crossover"


def record_files(root: Optional[str] = None) -> List[str]:
    """Every shipped record under ``root``: ``<table>_kernels/timing_*.json``, sorted.

    Representatives and cards' own records alike. The prior is derived from all of them,
    per architecture: an architecture's prior pools every table and card of that
    architecture, because what it carries across is the bracket, which is neither a
    table's nor a card's; a table reads the prior of its vendor's nearest timed
    architecture, never a pool of several architectures and never another vendor's.
    """
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    found = []
    for table in TABLES:
        folder = os.path.join(root, f"{table}_kernels")
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        found.extend(os.path.join(folder, name) for name in names
                     if name.startswith("timing_") and name.endswith(".json"))
    return sorted(found)


def _record_bands(record: Mapping[str, Any]) -> Tuple[List[List[int]], List[List[int]]]:
    """``(veto bands, keep bands)`` of a record's measured bracketed keys, sorted."""
    vetoes: List[List[int]] = []
    keeps: List[List[int]] = []
    for entry in (record.get("keys") or {}).values():
        if entry.get("status") != "measured" or entry.get("bracketed") is not True:
            continue
        band = [int(entry["band"]["lo_cells"]), int(entry["band"]["hi_cells"])]
        (vetoes if float(entry["veto_margin"]) > 0.0 else keeps).append(band)
    return sorted(vetoes), sorted(keeps)


def barrier_refusal(record: Mapping[str, Any], *, repair_route_sha256: Optional[str],
                    subject_sha256_by_table: Mapping[str, Mapping[str, str]]
                    ) -> Optional[str]:
    """Why ``record`` does not hold on the live tree, or None when it does.

    A record holds where its own barriers do: the deposit-repair digest it was cut on is
    the live one (``repair_route_sha256``; ``None``, a digest that could not be read,
    holds nothing), and every emitter digest its table compares (:func:`compared_emitters`)
    is the live one in ``subject_sha256_by_table[table]``. A record that does not hold
    would answer ``no_record`` on its own architecture. :func:`portable_prior` uses a
    record and :func:`timed_vendors` counts one only where this answers None and the
    record measures a key (:func:`timing_refusal`).
    """
    if repair_route_sha256 is None or (
            record["route"]["deposit_repair_sha256"] != repair_route_sha256):
        return (f"cut on deposit-repair route {record['route']['id']}; the live "
                f"deposit_repair.py digests to {str(repair_route_sha256)[:12]}")
    live = subject_sha256_by_table.get(str(record.get("table"))) or {}
    moved = sorted(source for source, digest in compared_emitters(record).items()
                   if live.get(source) != digest)
    if moved:
        return f"its emitters {', '.join(moved)} moved since it was cut"
    return None


def timing_refusal(record: Mapping[str, Any], *, repair_route_sha256: Optional[str],
                   subject_sha256_by_table: Mapping[str, Mapping[str, str]]
                   ) -> Optional[str]:
    """Why ``record`` does not TIME its architecture on the live tree, or None when it does.

    A record times its architecture when it holds there (:func:`barrier_refusal`) and
    measures at least one key: one that prices nothing arbitrates no GPU. The one test
    both :func:`portable_prior` (which architectures are timed, and from which records)
    and :func:`timed_vendors` (what a release ships) read.
    """
    why = barrier_refusal(record, repair_route_sha256=repair_route_sha256,
                          subject_sha256_by_table=subject_sha256_by_table)
    if why is None and not any(entry.get("status") == "measured"
                               for entry in (record.get("keys") or {}).values()):
        why = "it measures no key"
    return why


def portable_prior(records: Sequence[Tuple[str, Mapping[str, Any]]], *,
                   repair_route_sha256: Optional[str],
                   subject_sha256_by_table: Mapping[str, Mapping[str, str]],
                   refused: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """The priors an architecture with no record of its own reads, one per timed architecture.

    ``records`` are ``(path, record)`` pairs already through :func:`load_record`;
    ``refused`` names the files that were not, with why. A record is USED only where it
    times its architecture on the live tree (:func:`timing_refusal`: the deposit-repair
    digest ``repair_route_sha256`` and the emitter digests its own table compares,
    ``subject_sha256_by_table[table]``, hold, and it measures a key), because a record
    that would answer ``no_record`` on its own architecture must not price another one.

    ``by_vendor[vendor]["by_architecture"][architecture]`` is the prior of one timed
    architecture of :data:`VENDORS`, over that architecture's records used (every table of
    the vendor, every card): ``bracketed_veto_up_to_cells`` is the largest upper edge of a
    measured bracketed VETO band that lies below the lowest edge of every measured
    bracketed keep band; ``None`` when none has such a band, and that prior then vetoes
    nothing (``why_none`` says so). ``by_vendor[vendor]["why_none"]`` says why a vendor has
    no timed architecture at all. No number is stated across architectures or vendors:
    which architecture's prior a table reads is :func:`consult_prior`'s, by
    :func:`nearest_architecture`. The top-level ``derived_from`` lists every record used,
    each naming its ``vendor`` and ``architecture``, and ``not_used`` every record that
    was not, with why.
    """
    used: List[Dict[str, Any]] = []
    not_used: List[Dict[str, Any]] = [{"path": path, "why": why}
                                      for path, why in refused]
    for path, record in records:
        table = record.get("table")
        why = timing_refusal(record, repair_route_sha256=repair_route_sha256,
                             subject_sha256_by_table=subject_sha256_by_table)
        if why is not None:
            not_used.append({"path": path, "why": why})
            continue
        vetoes, keeps = _record_bands(record)
        used.append({"path": path, "table": table,
                     "vendor": VENDOR_BY_TABLE[str(table)],
                     "architecture": list(record_capabilities(record)),
                     "record_sha256": record.get(DIGEST_FIELD),
                     "veto_bands": vetoes, "keep_bands": keeps})
    by_vendor: Dict[str, Dict[str, Any]] = {}
    for vendor in VENDORS:
        own = [item for item in used if item["vendor"] == vendor]
        by_architecture: Dict[str, Dict[str, Any]] = {}
        for architecture in sorted({item["architecture"][0] for item in own}):
            sources = [item for item in own if item["architecture"][0] == architecture]
            keep_los = [lo for item in sources for lo, _hi in item["keep_bands"]]
            veto_his = [hi for item in sources for _lo, hi in item["veto_bands"]]
            floor = min(keep_los) if keep_los else None
            below = [hi for hi in veto_his if floor is None or hi < floor]
            ceiling = max(below) if below else None
            by_architecture[architecture] = {
                "bracketed_veto_up_to_cells": ceiling,
                "lowest_bracketed_keep_cells": floor, "derived_from": sources,
                "why_none": None if ceiling is not None else (
                    f"no {vendor} record of {architecture} used measures a bracketed "
                    f"veto below its keeps")}
        by_vendor[vendor] = {
            "tables": sorted(name for name, owner in VENDOR_BY_TABLE.items()
                             if owner == vendor),
            "by_architecture": by_architecture,
            "why_none": None if by_architecture else (
                f"no shipped {vendor} record holds on this tree and measures a key")}
    return {"rule": PRIOR_RULE, "by_vendor": by_vendor, "derived_from": used,
            "not_used": not_used}


def timed_vendors(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Which vendors the package under ``root`` ships a TIMED record for, and why not.

    A record is timed for its table's vendor (:data:`VENDOR_BY_TABLE`) when it loads as
    its own table's record (:func:`load_record`) and times its architecture on the tree
    under ``root`` (:func:`timing_refusal` against that tree's ``deposit_repair.py`` and
    emitters: it holds there and carries at least one measured key, the test
    :func:`portable_prior` uses a record by). A release ships at least one timed record of
    every vendor of :data:`VENDORS` (the module docstring, "A release ships a timed record
    of each vendor").

    ``root`` is a package directory laid out as this one (``deposit_repair.py`` and the
    ``<table>_kernels/`` folders); it defaults to this module's own. Every digest is a
    file read under ``root``: nothing is imported. Returns, for every vendor,
    ``{"records": [the timed files, relative to root], "not_timed": [{"path", "why"}],
    "why_none": None, or why the vendor has no timed record}``.
    """
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    try:
        live_route: Optional[str] = repair_route_sha256(
            os.path.join(root, "deposit_repair.py"))
    except OSError:
        live_route = None
    subjects: Dict[str, Dict[str, str]] = {}
    for table in TABLES:
        try:
            subjects[table] = live_subject_sha256(table, root)
        except OSError:
            subjects[table] = {}
    out: Dict[str, Dict[str, Any]] = {vendor: {"records": [], "not_timed": [],
                                               "why_none": None} for vendor in VENDORS}
    for path in record_files(root):
        name = os.path.relpath(path, root).replace(os.sep, "/")
        table = os.path.basename(os.path.dirname(path))[:-len("_kernels")]
        block = out[VENDOR_BY_TABLE[table]]
        try:
            record = load_record(path, expect_table=table)
        except RecordRefused as exc:
            block["not_timed"].append({"path": name, "why": f"refused: {exc}"})
            continue
        why = timing_refusal(record, repair_route_sha256=live_route,
                             subject_sha256_by_table=subjects)
        if why is not None:
            block["not_timed"].append({"path": name, "why": why})
            continue
        block["records"].append(name)
    for vendor, block in out.items():
        if block["records"]:
            continue
        block["why_none"] = (
            f"no {vendor} record ships" if not block["not_timed"] else
            f"no {vendor} record is timed: " + "; ".join(
                f"{item['path']}: {item['why']}" for item in block["not_timed"]))
    return out


def consult_prior(prior: Optional[Mapping[str, Any]], *, table: str, architecture: Any,
                  candidate: str, bracketed: bool, cells: int, mode: str) -> Verdict:
    """May ``candidate`` keep its slots on an architecture with no record of its own?

    The same one-directional answer :func:`consult` gives -- ``vetoed`` is the only
    outcome that changes a plan -- and the reason names the prior, so a dispatch record
    never reads a prior's veto as a measurement of this card. ``displaces`` is not asked:
    the prior is product-agnostic, and the composer confirms on the second composition
    that the slots went to certified singles, never to the array path or to another
    fused product.

    The prior asked is that of the timed architecture of ``table``'s vendor
    (:data:`VENDOR_BY_TABLE`) NEAREST the live ``architecture``
    (:func:`nearest_architecture`), never a pool of several architectures and never
    another vendor's: the evidence names the
    ``vendor``, the ``live_architecture``, the ``prior_architecture`` its ceiling came
    from, the ``basis`` (``own architecture``, the nearest compute capability, the
    closest chip generation), the ``distance`` and that architecture's records
    (``derived_from``). A vendor with no timed architecture answers ``no_prior``, and so
    does a nearest architecture whose records measure no bracketed veto below their keeps
    (a farther one is not asked instead). A live or timed spelling whose distance cannot
    be read answers ``live_architecture_unparsed`` or ``prior_architecture_unparsed``.
    Every one of these keeps the shape rule. A prior that carries no per-vendor block was
    not derived by :func:`portable_prior` and is refused by name.
    """
    if mode not in MODES:
        raise ConsultRefused("unknown_mode", f"mode is {mode!r}, not one of {MODES}")

    def answer(outcome: str, reason: str, evidence: Optional[Dict[str, Any]] = None
               ) -> Verdict:
        return Verdict(outcome=outcome, reason=reason, mode=mode, table=str(table),
                       candidate=str(candidate), displaces=(), key=None,
                       evidence=dict(evidence or {}, rule=PRIOR_RULE))

    if mode == "span":
        return answer("no_record", "span_mode_record_not_consulted")
    if not isinstance(bracketed, bool):
        raise ConsultRefused("unreadable_run_shape",
                             f"bracketed is {bracketed!r}, not a bool")
    vendor = VENDOR_BY_TABLE.get(str(table))
    if vendor is None:
        raise ConsultRefused("unknown_table", f"{table!r} is not one of {TABLES}")
    own: Optional[Mapping[str, Any]] = None
    if prior is not None:
        by_vendor = prior.get("by_vendor")
        if not isinstance(by_vendor, Mapping):
            raise ConsultRefused("unreadable_prior", (
                "the prior carries no per-vendor block (by_vendor), so it was not "
                "derived by portable_prior and names no vendor's records"))
        own = by_vendor.get(vendor)
        if own is not None and not isinstance(own.get("by_architecture"), Mapping):
            raise ConsultRefused("unreadable_prior", (
                f"the prior's {vendor} block carries no per-architecture block "
                "(by_architecture), so it was not derived by portable_prior"))
    timed = sorted((own or {}).get("by_architecture") or {})
    evidence: Dict[str, Any] = {
        "cells": int(cells), "bracketed": bracketed, "vendor": vendor,
        "live_architecture": architecture, "prior_architecture": None, "basis": None,
        "distance": None, "bracketed_veto_up_to_cells": None, "derived_from": []}
    if not timed:
        evidence["why_none"] = ((own or {}).get("why_none")
                                or ("no prior supplied" if prior is None
                                    else f"the prior holds no {vendor} block"))
        return answer("no_record", "no_prior", evidence)
    try:
        nearest = nearest_architecture(vendor, architecture, timed)
    except ConsultRefused as exc:
        evidence["why_none"] = exc.detail
        return answer("no_record", exc.code, evidence)
    chosen, basis, distance = nearest
    block = own["by_architecture"][chosen]
    ceiling = block.get("bracketed_veto_up_to_cells")
    evidence.update(
        prior_architecture=chosen, basis=basis, distance=distance,
        bracketed_veto_up_to_cells=ceiling,
        derived_from=[{name: item[name] for name in
                       ("path", "table", "architecture", "record_sha256")}
                      for item in block.get("derived_from") or ()])
    if ceiling is None:
        evidence["why_none"] = block.get("why_none")
        return answer("no_record", "no_prior", evidence)
    if not bracketed:
        return answer("no_record", "prior_says_nothing_about_a_clean_seam", evidence)
    if int(cells) > int(ceiling):
        return answer("no_record", "prior_above_the_recorded_crossover", evidence)
    return answer("vetoed", "prior_bracketed_seam_at_or_below_the_recorded_crossover",
                  evidence)


# ---------------------------------------------------------------------------
# The cost model: a product priced at the run's own size (schema /3)
# ---------------------------------------------------------------------------
#
# A band key answers for a run of cells by the verdict its measured sizes share, and a
# size between two bands whose verdicts differ is silence. A ``/3`` record also carries,
# per product class, the cost of the fused product and of the certified singles it
# displaces AT EACH MEASURED SIZE (``costs``), and the consult reads them at the run's
# own cell count: the product gives way to its singles exactly when, at that size, it
# is predicted slower than they are by more than the two legs' spreads allow -- the
# rule a key's rows are held to, applied at the size the run has. Every product class
# the record measured is priced this way, clean seams as well as bracketed ones.
#
# What it predicts with, stated so it can be argued with:
#
# * the curve is the measured points and straight lines between them in the cell
#   count (per-step time is a floor up to a knee and a bandwidth line after it, and a
#   line between two measured sizes follows both); exact at a measured size, and
#   nothing outside the smallest and largest measured size, where the band keys and
#   the portable prior answer as for a ``/2`` record;
# * the verdict is read off the MARGIN's line, each point carrying the least margin of
#   its own rows, so the verdict is continuous at every measured size and two points
#   that agree give their verdict everywhere between them (the medians on the same
#   line are evidence);
# * a point is a size at least ``min_rows`` rows measured (the band edge rule);
# * the four assumptions the band keys already make (the module docstring), the
#   first of them additivity: a plan's cost is its slots' costs, so withholding each
#   product predicted slower than its singles is the fastest predicted combination
#   wherever the products hold disjoint slots. Where withholding one hands its slots
#   to another FUSED product, the composer asks that product by its own table at this
#   size and, when it is predicted slower than its own singles too, withholds both,
#   which needs no price of one fused product against another. Where that product is
#   kept, no record prices one against the other, and the composer keeps the shape
#   rule's plan, as it does for a band veto.
#
# The points are DATA, recomputed from the record's own measurements every time it is
# read, so a record cannot state a curve its rows do not give.

#: How the points are read between measured sizes. A record names the rule it was cut
#: for; another name is refused, so a later rule is a new name, never a silent change.
COST_RULE = "margin_piecewise_linear_in_cells_between_measured_sizes"

#: The cost consult's answers that mean the record has no price for this product at
#: this size. The band keys are asked then, exactly as for a ``/2`` record.
COST_SILENCE = ("no_cost_curve_for_the_class", "outside_the_cost_points")

#: The :func:`consult` answers that mean the record cannot be applied to this run at
#: all. The cost consult answers each of them exactly as :func:`consult` does.
BARRIER_REASONS = ("span_mode_record_not_consulted", "no_record_supplied",
                   "record_is_for_another_table", "capability_unread",
                   "record_is_for_another_capability", "live_repair_route_not_stated",
                   "record_is_for_another_repair_route", "live_emitters_not_stated",
                   "record_is_for_another_emitter")

#: The fields of one cost point.
POINT_FIELDS = ("cells", "rows", "fused_ms_per_step", "baseline_ms_per_step",
                "fused_spread", "baseline_spread", "veto_margin",
                "slower_beyond_spread")

#: What ``rules.costs`` says in a record :func:`add_costs` stamped from a cut.
COST_RULE_TEXT = (
    "per product class with a measured key, one cost point per size at least min_rows "
    "rows measured (median fused and baseline ms/step, largest window spread per leg, "
    "and the least margin of that size's rows, the veto rule of a key); the consult "
    "reads a point's own margin at a measured size and, strictly between two measured "
    "sizes, the margin on the straight line between those two points' margins; never "
    "outside them")


def carries_costs(record: Any) -> bool:
    """Whether ``record`` is stamped with the cost model (``/3``)."""
    return isinstance(record, Mapping) and record.get("schema") == SCHEMA_COSTS


def _round6(value: float) -> float:
    return round(float(value), MARGIN_DECIMALS)


def cost_points(measurements: Sequence[Mapping[str, Any]],
                min_rows: int) -> List[Dict[str, Any]]:
    """One point per measured size with at least ``min_rows`` rows, sorted by cells.

    A point is the median fused and baseline ms/step of that size's rows, the largest
    window spread any of them showed on each leg, and the LEAST margin of its rows
    (:func:`least_margin`): the numbers a key states, per size, so at a measured size
    the curve and a key over that size give one verdict. A size fewer rows measured
    sets no point, for the reason it sets no band edge: one session prices only its
    own noise.
    """
    by_size: Dict[int, List[Mapping[str, Any]]] = {}
    for row in measurements:
        by_size.setdefault(int(row["cells"]), []).append(row)
    points = []
    for cells in sorted(by_size):
        rows = by_size[cells]
        if len(rows) < int(min_rows):
            continue
        points.append({
            "cells": cells, "rows": len(rows),
            "fused_ms_per_step": _round6(statistics.median(
                float(row["fused_ms_per_step"]) for row in rows)),
            "baseline_ms_per_step": _round6(statistics.median(
                float(row["baseline_ms_per_step"]) for row in rows)),
            "fused_spread": _round6(max(float(row["fused_spread"]) for row in rows)),
            "baseline_spread": _round6(max(float(row["baseline_spread"])
                                           for row in rows)),
            "veto_margin": least_margin(rows),
            "slower_beyond_spread": least_margin(rows) > 0.0,
        })
    return points


def build_costs(keys: Mapping[str, Mapping[str, Any]],
                min_rows: int) -> Dict[str, Dict[str, Any]]:
    """The ``costs`` block of a ``/3`` record, from its own keys. The one spelling.

    One curve per product class (a key without its band), pooled over the class's
    keys. A class is priced only when at least one of its keys is ``measured`` and all
    of them name the same slots and the same displaced singles (the cutter writes a
    class whose rows disagree on what the product displaces as one unmeasured key), and
    only at its sizes with ``min_rows`` rows.
    """
    classes: Dict[str, List[Mapping[str, Any]]] = {}
    for key in sorted(keys):
        classes.setdefault(key.rsplit("|", 1)[0], []).append(keys[key])
    costs: Dict[str, Dict[str, Any]] = {}
    for name in sorted(classes):
        entries = classes[name]
        if not any(entry.get("status") == "measured" for entry in entries):
            continue
        displaced = {tuple(entry.get("displaces") or ()) for entry in entries}
        slots = {tuple(entry.get("slots") or ()) for entry in entries}
        if len(displaced) != 1 or len(slots) != 1:
            continue
        points = cost_points([row for entry in entries
                              for row in entry.get("measurements") or ()], min_rows)
        if not points:
            continue
        first = entries[0]
        costs[name] = {
            "product": first["product"], "storage": first["storage"],
            "dimensions": first["dimensions"],
            "susceptibilities": first["susceptibilities"],
            "bracketed": first["bracketed"], "slots": list(next(iter(slots))),
            "displaces": list(next(iter(displaced))), "points": points}
    return costs


def _validate_costs(record: Mapping[str, Any]) -> None:
    """A ``/3`` record's cost model, recomputed; a ``/2`` record carries none."""
    if not carries_costs(record):
        stray = [name for name in ("costs", "cost_model") if name in record]
        if stray:
            raise _malformed(f"the record is stamped {record.get('schema')!r} and carries "
                             f"{stray}: a cost model is read only from a "
                             f"{SCHEMA_COSTS!r} record")
        return
    min_rows = (record.get("rules") or {}).get("min_rows_per_key")
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise _malformed(f"rules.min_rows_per_key is {min_rows!r}; a cost point is a "
                         "size at least that many rows measured")
    wanted = {"rule": COST_RULE, "min_rows_per_point": min_rows}
    if record.get("cost_model") != wanted:
        raise RecordRefused("unsupported_cost_model",
                            f"cost_model is {record.get('cost_model')!r}; this reader "
                            f"prices {wanted!r} and nothing else")
    costs = record.get("costs")
    if not isinstance(costs, Mapping):
        raise _malformed("costs is not an object")
    for name, entry in costs.items():
        points = entry.get("points") if isinstance(entry, Mapping) else None
        if not isinstance(points, list) or not points:
            raise _malformed(f"costs {name!r} names no points")
        for point in points:
            for field_name in ("fused_ms_per_step", "baseline_ms_per_step"):
                value = point.get(field_name) if isinstance(point, Mapping) else None
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not value > 0 or value == float("inf")):
                    raise _malformed(f"costs {name!r}: a point's {field_name} is "
                                     f"{value!r}")
    try:
        again = build_costs(record["keys"], min_rows)
    except (KeyError, TypeError, ValueError) as exc:
        # a /2 key validates without the per-row sizes and times a curve reads
        raise _malformed(f"a measurement cannot price a cost point: {exc!r}") from exc
    if _canonical({"costs": costs}) != _canonical({"costs": again}):
        differs = sorted(name for name in set(costs) | set(again)
                         if costs.get(name) != again.get(name))
        raise _malformed(f"costs is not what the record's own measurements give "
                         f"(classes {differs}): a record cannot state a curve its rows "
                         "do not give")


def add_costs(record: Mapping[str, Any]) -> Dict[str, Any]:
    """``record`` stamped ``/3`` with the cost model its own measurements give, sealed.

    The one way a ``/3`` record is made: ``parity/meep_gpu/cut_timing_record.py`` calls
    it on what it cut. A record that names its cut (``cut``) is stamped as the cutter
    stamps one (``cut.costs`` and ``rules.costs``), so a ``/2`` record re-stamped here is
    byte for byte the record its rows cut with ``--costs``, and ``--check`` recomputes it.
    """
    body = {name: value for name, value in record.items()
            if name not in (DIGEST_FIELD, "costs", "cost_model")}
    if isinstance(body.get("cut"), Mapping):
        body["cut"] = dict(body["cut"], costs=True)
        body["rules"] = dict(body.get("rules") or {}, costs=COST_RULE_TEXT)
    min_rows = (body.get("rules") or {}).get("min_rows_per_key")
    body["schema"] = SCHEMA_COSTS
    body["cost_model"] = {"rule": COST_RULE, "min_rows_per_point": min_rows}
    body["costs"] = build_costs(body.get("keys") or {}, int(min_rows or 0) or 1)
    sealed = seal(body)
    validate(sealed)
    return sealed


def predict_cost(entry: Mapping[str, Any], cells: int) -> Optional[Dict[str, Any]]:
    """A class's cost at ``cells``, read off its points by :data:`COST_RULE`, or None.

    None below the smallest and above the largest measured size. At a measured size
    the point's own margin answers (the least margin of its rows, as a key's does).
    Strictly between two, the MARGIN is the straight line between the two points'
    margins, so the curve is continuous at every measured size: two neighbours that
    agree give their verdict everywhere between them, as a band holding both does,
    and two that disagree give the crossover. The fused and baseline ms/step, their
    ratio and the spreads on the same line are evidence for reading; the verdict is
    the margin's. (A margin recomputed from the interpolated medians would be another
    statistic than the least row margin a point carries, and could veto one cell away
    from a size whose rows keep.)
    """
    points = list(entry.get("points") or ())
    cells = int(cells)
    if not points or cells < points[0]["cells"] or cells > points[-1]["cells"]:
        return None
    exact = [point for point in points if point["cells"] == cells]
    if exact:
        point = exact[0]
        return {"cells": cells, "between": [cells, cells], "weight": 0.0,
                "fused_ms_per_step": point["fused_ms_per_step"],
                "baseline_ms_per_step": point["baseline_ms_per_step"],
                "ratio": _round6(point["baseline_ms_per_step"]
                                 / point["fused_ms_per_step"]),
                "fused_spread": point["fused_spread"],
                "baseline_spread": point["baseline_spread"],
                "veto_margin": point["veto_margin"]}
    low = max((point for point in points if point["cells"] < cells),
              key=lambda point: point["cells"])
    high = min((point for point in points if point["cells"] > cells),
               key=lambda point: point["cells"])
    weight = (cells - low["cells"]) / float(high["cells"] - low["cells"])

    def at(name: str) -> float:
        return float(low[name]) + weight * (float(high[name]) - float(low[name]))

    fused, baseline = at("fused_ms_per_step"), at("baseline_ms_per_step")
    return {"cells": cells, "between": [low["cells"], high["cells"]],
            "weight": _round6(weight), "fused_ms_per_step": _round6(fused),
            "baseline_ms_per_step": _round6(baseline), "ratio": _round6(baseline / fused),
            "fused_spread": _round6(at("fused_spread")),
            "baseline_spread": _round6(at("baseline_spread")),
            "veto_margin": _round6(at("veto_margin"))}


def consult_cost(record: Optional[Mapping[str, Any]], *, table: str,
                 capability: Optional[str], candidate: str,
                 displaces: Optional[Sequence[str]], run_shape: Mapping[str, Any],
                 bracketed: bool, mode: str, repair_route: Optional[Mapping[str, Any]],
                 subject_sha256: Optional[Mapping[str, str]],
                 cells: Optional[int] = None, validated: bool = False) -> Verdict:
    """May ``candidate`` take the slots of ``displaces`` at THIS run's size?

    The question :func:`consult` is asked, answered from the record's cost points
    instead of its bands. The record's barriers (the table, the architecture, the
    deposit-repair route, the emitters) are :func:`consult`'s own: it is asked first and
    any of :data:`BARRIER_REASONS` is returned as it answered. Past them, a record that
    is not ``/3``, a class with no curve and a size outside the curve are
    :data:`COST_SILENCE` (``no_record``); otherwise ``vetoed`` when the margin
    :func:`predict_cost` reads at this size is positive (the product is predicted slower
    than its singles beyond the spreads), else ``not_vetoed``. ``displaces`` is checked as :func:`consult` checks it, against what
    the curve priced.
    """
    gate = consult(record, table=table, capability=capability, candidate=candidate,
                   displaces=displaces, run_shape=run_shape, bracketed=bracketed,
                   mode=mode, repair_route=repair_route, subject_sha256=subject_sha256,
                   cells=cells, validated=validated)
    if gate.outcome == "no_record" and gate.reason in BARRIER_REASONS:
        return gate
    shape = shape_class(run_shape, bracketed=bracketed, cells=cells)
    displaced = (() if displaces is None else tuple(str(arm) for arm in displaces))
    wanted = class_string(candidate, shape.storage, shape.dimensions,
                          shape.susceptibilities, shape.bracketed)

    def answer(outcome: str, reason: str, evidence: Dict[str, Any]) -> Verdict:
        return Verdict(outcome=outcome, reason=reason, mode=mode, table=str(table),
                       candidate=str(candidate), displaces=displaced, key=wanted,
                       evidence=dict(evidence, cost_model=COST_RULE,
                                     record_sha256=(record or {}).get(DIGEST_FIELD)))

    if not carries_costs(record):
        return answer("no_record", "no_cost_curve_for_the_class",
                      {"class": wanted, "cells": shape.cells,
                       "why": f"the record is stamped {(record or {}).get('schema')!r}"})
    entry = (record.get("costs") or {}).get(wanted)
    if entry is None:
        return answer("no_record", "no_cost_curve_for_the_class",
                      {"class": wanted, "cells": shape.cells})
    predicted = predict_cost(entry, shape.cells)
    evidence: Dict[str, Any] = {
        "class": wanted, "cells": shape.cells,
        "measured_sizes": [point["cells"] for point in entry["points"]],
        "displaces_measured": list(entry["displaces"])}
    if predicted is None:
        return answer("no_record", "outside_the_cost_points", evidence)
    evidence["predicted"] = predicted
    evidence["displaces_checked"] = displaces is not None
    if displaces is not None and sorted(entry["displaces"]) != sorted(displaced):
        return answer("no_record", "baseline_is_not_what_the_candidate_displaces",
                      evidence)
    if predicted["veto_margin"] > 0.0:
        return answer("vetoed", "predicted_slower_beyond_spread", evidence)
    return answer("not_vetoed", "predicted_not_slower_beyond_spread", evidence)
