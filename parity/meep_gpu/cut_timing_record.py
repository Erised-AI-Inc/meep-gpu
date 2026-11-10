#!/usr/bin/env python
"""Cut a kernel table's dispatch-preference timing record from timing rows on disk.

The record (``meep_gpu/<backend>_kernels/timing_cc<capability>.json``, as
``dispatch_preference.record_path`` spells it) is what
``meep_gpu/dispatch_preference.py`` consults; its format and the veto rule live there,
and this file imports both so the two cannot spell an axis two ways. How records are
cut and shipped: ``docs/development/timing.md`` (The cuts). The record is CUT, never
written by hand, the way ``fingerprints.json`` is cut by a device gate::

    python cut_timing_record.py --table triton --rows results/<tree> [--rows ...] \\
        --select-route <id> --out ../../meep_gpu/triton_kernels/timing_cc86.json
    python cut_timing_record.py --check ../../meep_gpu/triton_kernels/timing_cc86.json
    python cut_timing_record.py --list [--root DIR]

=============================================================================
What a row must be to feed a key
=============================================================================

Rows come in two spellings, both read here: ``bench_fused_products.py``'s (one row per
case, the fused / unfused / array legs inside it) and the deposit-repair timing harness's
(one row per LEG; a timed leg is paired with the ``singles`` leg of its own point). A row
is ADMITTED only if

1. its verdict is ``TIMED`` and it passed EVERY floor the bench records, the same-table
   control among them -- a control served by another kernel table compares two tables,
   not a product with what it displaces. A floor that was never recorded did not pass.
   The within-pass WINDOW spread is no floor (since the between-pass gate, 2026-10-16): a row
   written before then carries ``spread_within_gate`` among its floors and is read as if
   it did not, and the gate is the BETWEEN-PASS spread (below);
2. its plan was SERVED by the record's table alone (``substitution.tables_dispatched``,
   what actually dispatched, beside ``drive_table``, which is what was asked for). A
   plan whose slots two tables served is not a measurement of either one's plan: a
   term common to both legs moves the ratio toward 1, so such a row cannot manufacture
   a veto, but its ms/step is not this table's and the rule here claims a same-table
   comparison. A row that does not record the field cannot say, and is left out;
3. it was lifted on the device (``prefer_gpu``): a host smoke row is a harness check;
4. no foreign process held its device, before or after. An NVIDIA row is read from its
   device probe (``box_before``, ``box_after``), joined by device UUID
   (``CUDA_VISIBLE_DEVICES`` index -> ``gpus`` -> ``compute_apps``), NOT the row's own
   ``foreign_compute_apps_on_pinned_gpu``, which compares an index with a UUID and so
   names every neighbour on any device. A Metal row (``drive_table`` metal) carries no
   such probe: a Mac has one Metal device, and the bench's own host record
   (``host_before``, ``host_after``) lists every other process found driving it
   (``foreign_processes``, the bench's process, its ancestors and its children excluded),
   which is what is read. A host record whose neighbours were not read (no list, the read
   named under ``unreadable``, or a record taken off macOS, where no read runs) is not
   read, never clean: the row is left out. A row with no after-probe (``box_after``, or
   ``host_after`` for a Metal row) is admitted, COUNTED and named in its key
   (``--require-box-after`` refuses it instead);
5. it records which ``deposit_repair.py`` it timed.

Everything else is left out BY NAME, in the record (``excluded``) and the report.

The between-pass gate (2026-10-16). The admitted rows of one case and grid shape, on one
deposit-repair program and timed by one instrument (window count and target), are that
point's passes: a sweep point's p1 and p2, never a DRIVE unit beside them. When there are two or more, the spread of their fused and of their
baseline rates, (max - min) / median, must be at most 5 % on each leg, or every row of
the point is left out by name: a figure is a median of pass medians, and medians that
do not reproduce are not one. A point timed once (a DRIVE row) has no between-pass
spread and is not struck by it; the record's own ``min_rows`` still keeps such a size
from setting a band edge. Each row's window spreads stay in the record
(``fused_spread`` / ``baseline_spread``) and in the veto margin, which is
``dispatch_preference``'s rule and no admission gate. A record cut by this rule stamps
``cut.gate`` (``between_pass``); a record with no stamp was cut by the window floor,
and ``--check`` recomputes each by the rule it was cut by, so the 2026-09-21 records
recompute byte for byte.

=============================================================================
One record, one program
=============================================================================

The deposit-repair bracket is most of what a bracketed key prices, so rows timed
before and after a bracket change must never pool. The route a bracketed row ran is
read off its own counters -- digest, linear or 3-tuple index, repairs per bracketed
step (3 for the full bracket, 1 once restricted to the written component), whether any
source was restricted, whether any fell back -- and rows on two routes REFUSE
(``mixed_repair_route``) unless ``--select-route`` names one, in which case the others
are left out by name. One digest is not one route: the corpus has ``e4f88c1e`` running
both ``3-per-step:full`` and ``1-per-step:restricted``, so the record carries the whole
identity (``dispatch_preference.route_id``) and the consult compares all of it.

The fused-pair EMITTERS are a barrier too, scoped to the table that ran
(``dispatch_preference.SUBJECT_SOURCES_BY_TABLE``): comparing a CUDA-only module on a
Triton record compares a file those rows never ran. ``not recorded`` is a VALUE, not a
digest -- a row that did not pin an emitter is not evidence that it ran its neighbour's
-- so a source some rows pinned and others did not REFUSES (``unpinned_subject_source``)
until ``--allow-unpinned-subject`` names it, and the record then carries the straddle
and the consult does not compare that source at all. The bench, ``fastpath`` and
``driver`` digests are instruments and are recorded, not compared.

=============================================================================
One record, one card
=============================================================================

One record, one card, for the same reason one deeper: the ratio a key carries was
measured on one card's instruction set, clocks and occupancy, so rows from two devices
REFUSE (``mixed_device``) and the capability the surviving rows stamp -- read off the
rows, never typed -- names the architecture in the file name. The card is the exact
device name the rows stamp (``host[].device``, which a run's dispatch record carries as
``environment.device.name``), and its SLUG (``dispatch_preference.card_slug``, the
reader's, so the two cannot spell a card two ways) is that name in lower case with
every run of characters outside ``[a-z0-9]`` one ``_`` and none at either end:
``NVIDIA GeForce RTX 3090`` is ``nvidia_geforce_rtx_3090``. Cards match by slug, never
by prefix or family: an RTX 3090 Ti is not an RTX 3090. Two exact names with one
slug are two cards, and an architecture's records keep one exact name per slug (the
second is refused, ``existing_record_names_another_device``), so a slug never stands for
two cards.

A card's record has two names it may take (:func:`destination`):

* ``<table>_kernels/timing_<arch>.json``, the name ``dispatch_preference.record_path``
  gives the architecture: its REPRESENTATIVE, the record the dispatch path reads for
  every card of that architecture that has no record of its own. Which card it is, is
  data: the record names it in ``host[]``, and the report beside it says so on its
  ``card:`` line;
* ``<table>_kernels/timing_<arch>-<slug>.json``, a CARD RECORD beside it
  (``dispatch_preference.record_path`` with the card), which the dispatch path reads
  for that card alone. Neither a slug nor an architecture's spelling contains ``-``, so
  the name parses one way (``dispatch_preference.record_name_parts``), and every
  ``timing_*.json`` glob in the tree finds it. The reader refuses a card record whose
  rows name another card (``record_named_for_another_card``).

The first card cut for an architecture lands at the bare name and represents it. A
re-cut of a card rewrites that card's own file, wherever it sits; another card's cut
lands beside, under its own card name, and replaces nothing. A cut never retires,
renames, moves or designates: changing an architecture's representative is a change to
the tracked records. Retire the representative's record and report, and the chosen
card's own record and report when it has them (a re-cut goes back to a card's own file
while it exists), then cut the chosen card's rows, which land at the bare name left
empty. ``--out`` must be exactly the destination, and anything else is refused by name
before either file is written: ``out_names_another_capability`` (a name of another
architecture, or of no record), ``existing_record_names_another_device`` (a file
holding another card's record; another exact device's record under this card's slug,
at either name; a file whose rows name more than one device; there is no override),
``existing_record_unreadable`` (a file at ``--out``, at the bare name or at the card's
own name that cannot be read to say whose it is), ``out_names_another_card`` (any other
name; the refusal states the rule), ``card_has_two_records`` (the card is already at
both names, so neither can be chosen) and ``unnamed_card`` (rows that stamp no device
name). Records are added, and a cut replaces only its own card's record.

The dispatch path reads a card's own record for that card and the representative for
every other card of the architecture; an architecture with neither reads the prior of
its vendor's nearest timed architecture. A package may ship any number of cards'
records beside a representative, each at the name its content agrees with, one file per
card per table and architecture (the layout test in
``meep_gpu/test_dispatch_preference.py``). ``--list`` prints every record under a root
with its role, card, digest and whether its barriers hold on this tree. The Metal
table's records are named for the Apple GPU family its rows stamp
(``timing_applegpu_g13s.json``), by the same rule.

=============================================================================
A record cut from part of a campaign names what it lacks
=============================================================================

With ``--account-for-release`` (the command line's default) the cut reads the table's
release (``fastpath.RELEASED_FUSED_ARMS``, ``fastpath_cuda.CUDA_RELEASED_FUSED_ARMS``,
``metal_dispatch.METAL_RELEASED_FUSED_ARMS``: each released product and the route
cases that drive it) and writes ``missing``: every released (product, case) that no
row it read accounts for, with the reason -- the rows of that case were not timings
(an ``ERROR`` row, as eleven DRIVE rows of the first 1.0 timing were), or no
row of the case was cut at all. A partial set therefore cuts, and says what it owes,
where it used to leave a record the release check refused. ``--check`` recomputes ``missing`` against
the release in the tree it runs in, so a release that moved since the cut DIFFERS.

=============================================================================
Which product a row prices
=============================================================================

A row times a PLAN: every fused product in it against the single arms in the same
slots. One number cannot price two products, so:

* a plan with one fused product prices that product;
* a plan with several prices only the one that carries the deposit-repair bracket,
  and only when exactly one does -- the whole-step cost follows the source's seam
  (``fused_ablation_2026-09-17_seamswap``), and the clean partner's own delta is the
  residue inside that key, visible under ``co_fused``;
* a plan whose products are ALL clean prices none of them. Writing the joint ratio
  into a key for each would let a partner that loses veto a product nothing measured
  to be slower than: a false veto, the one error a fail-closed veto may not make.

=============================================================================
BANDS
=============================================================================

Per class (product, storage, dimensions, susceptibilities, bracketed), the measured
cell counts are sorted and cut into maximal runs sharing one verdict. A run is then
TRIMMED to sizes at least ``--min-rows`` rows measured, and that trimmed span is the
band ``[lo, hi]``, inclusive. The remainder at either end is written as its own
``unmeasured`` key, with its rows and numbers, so the next timing campaign sees what it owes.
A band's interior is interpolated -- its edges are measured, the sizes between them
need not be -- which is why an edge must be corroborated: ``fused_spread`` and
``baseline_spread`` price the noise inside a window, never the spread BETWEEN
sessions, and at neighbouring sizes the corpus shows that at the size of the thinnest
margins measured. Where the verdict changes between two adjacent sizes the gap is
written as an explicit ``unmeasured`` key, and outside the measured sizes there is no
key at all. Nothing is extrapolated.

The output is deterministic: sorted keys, no wall clock (``recorded_utc`` is the
newest admitted row's), paths relative to ``--base``, and a digest of every file read.
``--check`` recomputes a record from the artifacts it names and refuses on any
difference.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meep_gpu import dispatch_preference as dp  # noqa: E402

#: This file, as the record names it.
CUT_BY = "parity/meep_gpu/cut_timing_record.py"

#: Fewest admitted rows a band needs to be ``measured``. One row is one session's
#: median; two is the least that can disagree.
MIN_ROWS_PER_KEY = 2

#: The floors ``bench_fused_products.run_case`` recorded until the between-pass gate. All
#: seven, by name: a row written before one of them existed did not pass it. What a
#: record cut by the window rule (:data:`WINDOW_GATE`, every record cut before the between-pass gate)
#: requires.
REQUIRED_FLOORS = ("substitution_proved", "bit_identical", "moved_words_positive",
                   "non_finite_zero", "no_compiles_inside_timed_region",
                   "spread_within_gate", "singles_are_the_same_table")
#: The bench floor that read a leg's within-pass window spread: no floor under the between-pass gate.
WINDOW_FLOORS = ("spread_within_gate",)
#: The floors a record cut by the between-pass rule requires: the six that gate. The
#: ``_R33`` suffix here and on :data:`RULES_R33`, and the ``PHASE0_`` prefix of
#: :data:`PHASE0_BASELINE_LEG`, are historical names for the 2026-10-16 between-pass
#: rule and the deposit-repair timing harness; code and tests read them by name.
REQUIRED_FLOORS_R33 = tuple(name for name in REQUIRED_FLOORS if name not in WINDOW_FLOORS)
#: The two rules a record may be cut by, as ``cut.gate`` names them. The window rule is
#: never stamped: every record cut before the between-pass gate carries no ``gate`` and recomputes by it.
WINDOW_GATE, BETWEEN_PASS_GATE = "window", "between_pass"
GATES = (WINDOW_GATE, BETWEEN_PASS_GATE)
#: The between-pass spread a point's passes may show: the 5 % gate.
BETWEEN_PASS_MAX = 0.05

#: Sources whose digest is a POOLING BARRIER (with ``deposit_repair``, which the route
#: carries): they emit the fused kernels a row times. Scoped per table and spelled in
#: the module the consult reads, so a record cannot be cut on one barrier and read
#: against another. Every source any table names is still RECORDED for every table.
SUBJECT_SOURCES = tuple(sorted({source for sources
                                in dp.SUBJECT_SOURCES_BY_TABLE.values()
                                for source in sources}))

#: What the emitter barrier cannot cover, stated in every record rather than left for
#: a reader to assume. A product's own device text comes from a per-product module and
#: no timing row pins one, which is a change to the bench's ``PINNED_SOURCES``.
CANNOT_COVER = ("a product's own device text comes from a per-product module "
                "(cuda_kernels/fused_electric_pair.py, the triton_kernels/ emitters, "
                "the Metal kernel sources) and no timing row pins one: this record "
                "cannot say which per-product kernel text it timed")

#: Sources recorded and listed, never compared: the harness and the dispatch path.
INSTRUMENT_SOURCES = ("bench", "fastpath", "driver", "phase0_harness")

NOT_RECORDED = dp.NOT_RECORDED

#: The repair-harness leg every other leg of a point is measured against.
PHASE0_BASELINE_LEG = "singles"

EXIT_OK, EXIT_DIFFERS, EXIT_REFUSED, EXIT_ARTIFACTS_ABSENT = 0, 1, 2, 3

DECIMALS = 6


class CutRefused(ValueError):
    """The cutter will not write a record from these rows, refused by name."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def say(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------

def _relative(path: str, base: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(base)).replace(
        os.sep, "/")


def row_files(roots: Sequence[str]) -> List[str]:
    """Every ``rows.jsonl`` under the roots (a root may be the file itself)."""
    found = set()
    for root in roots:
        if not os.path.exists(root):
            raise CutRefused("artifacts_absent", f"{root} does not exist")
        if os.path.isfile(root):
            found.add(os.path.abspath(root))
            continue
        for path in glob.glob(os.path.join(root, "**", "rows.jsonl"), recursive=True):
            found.add(os.path.abspath(path))
    return sorted(found)


def read_files(files: Sequence[str], base: str) -> Tuple[List[Dict[str, Any]],
                                                          List[Dict[str, Any]]]:
    """``(lines, file_table)``: every parsed line with where it came from."""
    lines: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for path in sorted(files, key=lambda item: _relative(item, base)):
        relative = _relative(path, base)
        count = 0
        with open(path, "rb") as handle:
            payload = handle.read()
        for number, raw in enumerate(payload.splitlines(), 1):
            if not raw.strip():
                continue
            count += 1
            try:
                row = json.loads(raw)
            except ValueError as exc:
                raise CutRefused("unreadable_row", f"{relative}:{number}: {exc}") from exc
            lines.append({"artifact": relative, "line": number, "row": row,
                          "sha256": hashlib.sha256(raw.strip()).hexdigest()})
        table.append({"path": relative, "rows": count,
                      "sha256": hashlib.sha256(payload).hexdigest()})
    return lines, table


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

def _box_foreign(box: Mapping[str, Any]) -> Optional[List[str]]:
    """Foreign compute processes on the row's own device, or None if unreadable.

    A row's tables name its device by card id and its own process as ``own`` since they
    leave the machine (``timing_host.recorded_compute_apps``), by raw UUID and pid before:
    both tables of one row share one spelling, and either is read."""
    visible = str(box.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    gpus = box.get("gpus")
    if not visible or not isinstance(gpus, str) or "compute_apps" not in box:
        return None
    uuid = None
    for entry in gpus.splitlines():
        parts = [part.strip() for part in entry.split(",")]
        if len(parts) >= 2 and visible in (parts[0], parts[1]):
            uuid = parts[1]
    if uuid is None:
        return None
    foreign = []
    for entry in str(box.get("compute_apps") or "").splitlines():
        parts = [part.strip() for part in entry.split(",")]
        if len(parts) >= 2 and parts[0] == uuid and parts[1] not in (
                "own", str(box.get("pid"))):
            foreign.append(entry.strip())
    return foreign


#: The read of the bench's host record that lists the processes driving the Metal device
#: (``bench_fused_products.host_state``): when it fails it is named under ``unreadable``
#: and ``foreign_processes`` stays empty, so an empty list alone is no reading.
HOST_NEIGHBOURS_READ = "neighbours"
#: The platform on which that read runs; off it the record lists no process and read none.
HOST_NEIGHBOURS_PLATFORM = "darwin"


def _after_probe(row: Mapping[str, Any]) -> str:
    """The field a row's after-probe of its device is in: ``host_after`` for a Metal row
    (its host record), ``box_after`` for every other."""
    return "host_after" if row.get("drive_table") == "metal" else "box_after"


def _host_foreign(host: Mapping[str, Any]) -> Optional[List[str]]:
    """The other processes driving the Metal device, as a Metal row's host record lists
    them, or None when the record did not read them.

    The bench's own record (``bench_fused_products.host_state``) lists every process
    found driving the one Metal device, its own process, its ancestors and its children
    excluded, each by its command. A read that failed is named under ``unreadable`` and
    leaves the list empty, and off macOS no read runs at all, so either is no reading and
    never a clean one."""
    foreign = host.get("foreign_processes")
    unreadable = host.get("unreadable")
    if (not isinstance(foreign, list) or not isinstance(unreadable, Mapping)
            or HOST_NEIGHBOURS_READ in unreadable
            or host.get("platform") != HOST_NEIGHBOURS_PLATFORM):
        return None
    return [str(item) for item in foreign]


def _metal_contended(row: Mapping[str, Any], require_after: bool) -> Optional[str]:
    """:func:`contended` for a Metal row: its neighbours, read from its host record."""
    marks = []
    for when in ("host_before", "host_after"):
        host = row.get(when)
        if not isinstance(host, Mapping):
            if when == "host_before":
                return "contention cannot be read: the row carries no host_before"
            if require_after:
                return "the row carries no after-probe of its device"
            continue
        foreign = _host_foreign(host)
        if foreign is None:
            return (f"contention cannot be read: {when} does not record the processes "
                    "driving the Metal device")
        if foreign:
            marks.append(f"{when.split('_')[1]}: {len(foreign)} foreign process(es) "
                         "driving the Metal device")
    return "contended -- " + "; ".join(marks) if marks else None


def contended(row: Mapping[str, Any], require_after: bool = False) -> Optional[str]:
    """Why this row is not a quiet-device measurement, or None if it is one.

    A Metal row is read from its host record (:func:`_metal_contended`), every other
    from its device probe, joined by device UUID."""
    if row.get("drive_table") == "metal":
        return _metal_contended(row, require_after)
    marks = []
    for when in ("box_before", "box_after"):
        box = row.get(when)
        if not isinstance(box, Mapping):
            if when == "box_before":
                return "contention cannot be read: the row carries no box_before"
            if require_after:
                return "the row carries no after-probe of its device"
            continue
        foreign = _box_foreign(box)
        if foreign is None:
            return f"contention cannot be read: {when} does not identify the device"
        if foreign:
            marks.append(f"{when.split('_')[1]}: {len(foreign)} foreign process(es) "
                         "on the row's device")
    return "contended -- " + "; ".join(marks) if marks else None


#: THE PREFERENCE RULE: the dispatch preference a fused row's process must have run
#: under. A row prices the SHAPE RULE'S product against its singles -- that is what a
#: key is -- and the package default arbitrates those products by the records cut
#: here, so a fused leg timed under the default may have composed the singles a record
#: already chose and timed nothing a key can price; a record cut from such rows loses
#: its own vetoes, and the freeze after it vetoes nothing. So a row is admitted only
#: when its ``process_environment`` names this value of the package's switch
#: (``fastpath.PREFERENCE_SWITCH``), which ``bench_fused_products.resolve_preference``
#: sets and records. A record cut by this rule stamps ``cut.fused_leg_preference``; a
#: record with no stamp was cut before it and ``--check`` recomputes it without.
FUSED_LEG_PREFERENCE = "span"
#: Where a row records the switches its process ran under (the bench's
#: ``ENVIRONMENT_KEY``).
ENVIRONMENT_KEY = "process_environment"
#: The prefix of the refusal the preference rule writes, for a reader matching reasons.
NOT_THE_SHAPE_RULE = "fused leg not timed under the shape rule"


def preference_refusal(row: Mapping[str, Any], required: Optional[str]) -> Optional[str]:
    """Why this row's fused leg was not timed under ``required``, or None.

    ``required`` None is a record cut before :data:`FUSED_LEG_PREFERENCE` existed, and
    asks nothing. The switch is the package's, read late like the release. Asked by
    :func:`gather` after :func:`admission_refusal`, of every row that one admits.
    """
    if required is None:
        return None
    from meep_gpu import fastpath  # noqa: PLC0415

    switch = fastpath.PREFERENCE_SWITCH
    if not isinstance(row.get(ENVIRONMENT_KEY), Mapping):
        return (f"{NOT_THE_SHAPE_RULE}: the row records no {ENVIRONMENT_KEY}, so it "
                f"cannot say its fused leg ran under {switch}={required}")
    value = row[ENVIRONMENT_KEY].get(switch)
    if value != required:
        return (f"{NOT_THE_SHAPE_RULE}: {ENVIRONMENT_KEY} records {switch}={value!r}, "
                f"not {required!r}, so the leg may have timed the default's composition "
                "rather than the product a key prices")
    return None


def _floor_refusal(row: Mapping[str, Any], gate: str = BETWEEN_PASS_GATE) -> Optional[str]:
    floors = row.get("floors") or {}
    window = gate == WINDOW_GATE
    required = REQUIRED_FLOORS if window else REQUIRED_FLOORS_R33
    missing = [name for name in required if name not in floors]
    if missing:
        return "floor not recorded: " + ", ".join(missing)
    # Under the between-pass gate the window floor gates nothing: a row written before
    # the bench moved it out of its floors is read as one written after.
    failed = sorted(name for name, passed in floors.items() if passed is not True
                    and (window or name not in WINDOW_FLOORS))
    if failed:
        return "floor failed: " + ", ".join(failed)
    if (row.get("substitution") or {}).get(
            "fused_and_singles_used_the_same_table") is not True:
        return ("the control did not run the same kernel table as the fused leg "
                "(substitution.fused_and_singles_used_the_same_table)")
    # The bench's ``reportable`` is every floor it recorded; under the between-pass gate
    # it is re-derived from the floors that gate, read above, so the window floor of a row
    # written before that gate is not read back through it.
    if window and row.get("reportable") is not True:
        return "the row is not reportable"
    return None


def admission_refusal(row: Mapping[str, Any], table: str,
                      require_after: bool, gate: str = BETWEEN_PASS_GATE) -> Optional[str]:
    """The first reason this row may not feed a record, or None."""
    if row.get("drive_table") != table:
        return f"a row of the {row.get('drive_table')!r} table"
    if row.get("legs_mode") == "gpu-only":
        # One leg, no singles: such a row says what dispatched and how fast, never what
        # fusing saved against the singles, which is all a preference record keys on.
        return "a gpu-only row (the fused leg alone): it times no product against its singles"
    if row.get("verdict") != "TIMED" and not (
            row.get("verdict") == "TIMED-NOT-REPORTABLE" and row.get("floors")):
        return f"verdict {row.get('verdict')}"
    if row.get("prefer_gpu") is not True:
        return "a host smoke row (prefer_gpu is not true), never a device timing"
    refusal = _floor_refusal(row, gate)
    if refusal:
        return refusal
    # WHAT SERVED, beside the floor's "the control ran the same table as the fused
    # leg": a plan whose slots two tables served is not this table's plan at all.
    served = (row.get("substitution") or {}).get("tables_dispatched")
    if served is None:
        return "the row does not record which kernel tables its plans dispatched"
    if sorted(served) != [table]:
        return (f"the plan's slots were served by {sorted(served)}, not by the "
                f"{table!r} table alone")
    refusal = contended(row, require_after)
    if refusal:
        return refusal
    if not ((row.get("provenance") or {}).get("sha256") or {}).get("deposit_repair"):
        return "no deposit_repair digest recorded: the row cannot say which bracket it timed"
    return None


# ---------------------------------------------------------------------------
# One measurement: a (fused leg, baseline leg) pair with its plan
# ---------------------------------------------------------------------------

def _leg_numbers(leg: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    seconds, spread = leg.get("median_seconds_per_step"), leg.get("spread")
    if not isinstance(seconds, (int, float)) or not isinstance(spread, (int, float)):
        return None
    if seconds <= 0 or spread < 0:
        return None
    return 1e3 * float(seconds), float(spread)


def _route_of(leg: Mapping[str, Any], sha: str,
              patch: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The deposit-repair route a fused leg ran, or None when nothing is bracketed."""
    block = leg.get("deposit_repair_route") or {}
    if not block.get("bracketed_slots"):
        return None
    linear = int(block.get("linear_saves", 0)) + int(block.get("linear_repairs", 0))
    cells = int(block.get("cells_saves", 0)) + int(block.get("cells_repairs", 0))
    index = ("linear" if linear and not cells else "3-tuple" if cells and not linear
             else "mixed" if linear and cells else "never-ran")
    steps = int(leg.get("steps_per_window") or 0) * int(leg.get("windows") or 0)
    repairs = int(block.get("linear_repairs", 0)) + int(block.get("cells_repairs", 0))
    per_step = round(repairs / steps, 3) if steps else 0.0
    if patch is not None:            # the repair harness keeps its own two counters
        recorded = True
        restricted = int(patch.get("restriction_applied", 0)) > 0
        fell_back = int(patch.get("restriction_fell_back", 0)) > 0
    else:
        recorded = "restricted_sources" in block or "restriction_fallbacks" in block
        restricted = int(block.get("restricted_sources", 0)) > 0
        fell_back = int(block.get("restriction_fallbacks", 0)) > 0
    route = {"deposit_repair_sha256": sha, "index": index,
             "repairs_per_bracketed_step": per_step, "restricted": restricted,
             "fell_back": fell_back, "counters_recorded": recorded,
             "plan_changed_inside_window": bool(block.get("plan_changed_inside_window"))}
    # ONE SPELLING, in the module the consult reads: a route named here and a route
    # asked for there cannot drift apart.
    route["id"] = dp.route_id(route)
    return route


def _host_of(plan: Mapping[str, Any]) -> Dict[str, Any]:
    environment = plan.get("environment") or {}
    device = environment.get("device") or {}
    return {"device": device.get("name"),
            "compute_capability": device.get("compute_capability"),
            "cuda_driver": device.get("cuda_driver"),
            "cuda_runtime": device.get("cuda_runtime"),
            "backend": environment.get("backend"),
            "backend_version": environment.get("backend_version"),
            "triton": environment.get("triton")}


def _policy_of(plan: Mapping[str, Any]) -> Optional[str]:
    return ((plan.get("subnormal") or {}).get("gate") or {}).get("policy_in_force")


def instrument_of(row: Mapping[str, Any], fused_leg: Mapping[str, Any]) -> str:
    """The timing instrument a row's fused leg was measured with, as the row records it:
    its window count and the probe floor its window target set (``bench_fused_products.
    size_chunk``: 0.1 of the target, at least 0.05 s). A DRIVE unit and a sweep point of
    one case and grid shape are two instruments -- and two rows of the campaign, each with
    its own passes -- so the between-pass gate never pools them."""
    probe = (((row.get("sizing") or {}).get("fused") or {}).get("probe") or {})
    floor = probe.get("min_seconds")
    return (f"windows={fused_leg.get('windows')};probe_min_s="
            f"{round(float(floor), 6) if isinstance(floor, (int, float)) else None}")


def build_measurement(ref: Mapping[str, Any], fused_leg: Mapping[str, Any],
                      baseline_leg: Mapping[str, Any], source: str,
                      baseline_line: Optional[int] = None
                      ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(measurement, refusal)`` for one admitted (fused, baseline) pair."""
    row = ref["row"]
    fused, baseline = _leg_numbers(fused_leg), _leg_numbers(baseline_leg)
    if fused is None or baseline is None:
        return None, "no per-leg timing (median seconds per step and spread)"
    plan = (row.get("plans") or {}).get("fused") or {}
    shape = plan.get("run_shape")
    if not isinstance(shape, Mapping):
        return None, "no run shape recorded for the fused plan"
    substitution = row.get("substitution") or {}
    welded = substitution.get("welded_arms") or {}
    novel = set((substitution.get("novel_slots") or {}).values()) or set(welded)
    products = {label: list(slots) for label, slots in welded.items() if label in novel}
    if not products:
        return None, "the fused plan names no fused product"
    sha = row["provenance"]["sha256"]["deposit_repair"]
    route = _route_of(fused_leg, sha, row.get("patch_counters"))
    if route is not None and route["plan_changed_inside_window"]:
        return None, "the plan re-froze inside a timed window"
    bracketed_slots = set((fused_leg.get("deposit_repair_route") or {})
                          .get("bracketed_slots") or ())
    unfused = substitution.get("unfused_arms") or {}
    try:
        grid_shape = [int(value) for value in (row.get("grid_shape")
                                               or shape.get("grid_shape"))]
        cells = dp.cells_of(grid_shape)
        dp.shape_class(shape, bracketed=False, cells=cells)
    except (dp.ConsultRefused, TypeError) as exc:
        return None, f"unreadable run shape: {exc}"
    digests = dict(row["provenance"]["sha256"])
    measurement = {
        "artifact": ref["artifact"], "line": ref["line"], "source": source,
        "instrument": instrument_of(row, fused_leg),
        "case": row.get("case"), "utc": row.get("utc"), "grid_shape": grid_shape,
        "cells": cells, "run_shape": dict(shape),
        "fused_ms_per_step": fused[0], "fused_spread": fused[1],
        "baseline_ms_per_step": baseline[0], "baseline_spread": baseline[1],
        "ratio": baseline[0] / fused[0],
        "products": {label: {"slots": slots,
                             "bracketed": bool(bracketed_slots & set(slots)),
                             "displaces": [unfused.get(slot) for slot in slots]}
                     for label, slots in products.items()},
        "route": route, "sha": sha, "digests": digests,
        "host": _host_of(plan), "policy": _policy_of(plan),
        "after_probe": isinstance(row.get(_after_probe(row)), Mapping),
    }
    if baseline_line is not None:
        measurement["baseline_line"] = baseline_line
    if any(arm is None for product in measurement["products"].values()
           for arm in product["displaces"]):
        return None, "the control leg names no arm for a slot the fused product holds"
    return measurement, None


def _rate_spread(milliseconds: Sequence[float]) -> float:
    """(max - min) / median of the RATES one leg's passes read (the gate is spelled on
    pass rates; a rate is the reciprocal of ms/step)."""
    rates = [1.0 / value for value in milliseconds]
    return (max(rates) - min(rates)) / statistics.median(rates)


def between_pass(measurements: List[Dict[str, Any]], excluded: List[Dict[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """The measurements whose point's passes agree, the others left out by name.

    A POINT is a case and grid shape of one source (bench or repair-harness leg) on ONE
    deposit-repair program -- the route identity :func:`one_program` compares, else the
    digest of an unbracketed row -- so rows of two routes never pool here, and a
    ``--select-route`` leaves the other route out by its own name, not by this one's --
    timed by ONE instrument (:func:`instrument_of`), so a DRIVE unit (six 2 s windows) and
    a sweep point at the same grid (four 1.5 s) stay two rows, as their keys are
    (``timing_rows.fused_key``). Its admitted rows are its passes. Two or more must
    agree within :data:`BETWEEN_PASS_MAX` on the fused leg's rate and on the baseline's,
    or every row of the point is left out. One row has no between-pass spread and is
    not struck by it."""
    points: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for measurement in measurements:
        route = measurement.get("route")
        program = route["id"] if route is not None else f"digest:{measurement['sha']}"
        points.setdefault((str(measurement["source"]), str(measurement["case"]),
                           json.dumps(measurement["grid_shape"]), program,
                           str(measurement.get("instrument"))), []).append(measurement)
    struck: Dict[int, str] = {}
    for _point, members in sorted(points.items()):
        if len(members) < 2:
            continue
        wide = []
        for leg in ("fused", "baseline"):
            value = _rate_spread([m[f"{leg}_ms_per_step"] for m in members])
            if value > BETWEEN_PASS_MAX:
                wide.append(f"{leg} {value:.4f}")
        if wide:
            reason = (f"between-pass spread over the {len(members)} passes of this case, "
                      f"grid shape and instrument: {', '.join(wide)} > {BETWEEN_PASS_MAX} "
                      "(between-pass gate)")
            for member in members:
                struck[id(member)] = reason
    for measurement in measurements:
        if id(measurement) in struck:
            excluded.append({"artifact": measurement["artifact"],
                             "line": measurement["line"], "case": measurement["case"],
                             "reason": struck[id(measurement)]})
    return [m for m in measurements if id(m) not in struck]


def gather(lines: Sequence[Mapping[str, Any]], table: str, require_after: bool,
           gate: str = BETWEEN_PASS_GATE,
           preference: Optional[str] = FUSED_LEG_PREFERENCE
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """``(measurements, excluded, counts)`` over every line read, duplicates dropped,
    under ``gate`` (:data:`GATES`): the between-pass rule applies :func:`between_pass`;
    ``preference`` is the preference rule (:func:`preference_refusal`)."""
    if gate not in GATES:
        raise CutRefused("unknown_gate", f"{gate!r} is not one of {GATES}")
    seen: Dict[str, Mapping[str, Any]] = {}
    unique: List[Mapping[str, Any]] = []
    for ref in lines:
        if ref["sha256"] in seen:
            continue
        seen[ref["sha256"]] = ref
        unique.append(ref)
    measurements: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    baselines = 0

    def leave_out(ref: Mapping[str, Any], reason: str) -> None:
        excluded.append({"artifact": ref["artifact"], "line": ref["line"],
                         "case": ref["row"].get("case"), "reason": reason})

    phase0: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for ref in unique:
        row = ref["row"]
        if "leg" in row and "all_legs_ms_per_step" in row:
            phase0.setdefault((ref["artifact"], str(row.get("case")),
                               json.dumps(row.get("grid_shape"))), []).append(ref)
            continue
        refusal = (admission_refusal(row, table, require_after, gate)
                   or preference_refusal(row, preference))
        if refusal is None:
            legs = row.get("per_leg") or {}
            measurement, refusal = build_measurement(
                ref, legs.get("fused") or {}, legs.get("unfused") or {}, "bench")
            if measurement is not None:
                measurements.append(measurement)
        if refusal is not None:
            leave_out(ref, refusal)

    for _point, refs in sorted(phase0.items()):
        base_refs = [r for r in refs if r["row"].get("leg") == PHASE0_BASELINE_LEG]
        base = base_refs[0] if len(base_refs) == 1 else None
        base_refusal = ("the point carries no single baseline leg" if base is None else
                        admission_refusal(base["row"], table, require_after, gate)
                        or preference_refusal(base["row"], preference))
        if base is not None and base["row"].get("drive_table") == table:
            baselines += 1
        for ref in refs:
            row = ref["row"]
            if row.get("leg") == PHASE0_BASELINE_LEG:
                continue
            refusal = (admission_refusal(row, table, require_after, gate)
                       or preference_refusal(row, preference))
            if refusal is None and base_refusal is not None:
                refusal = f"baseline leg: {base_refusal}"
            if refusal is None:
                measurement, refusal = build_measurement(
                    ref, row.get("per_leg") or {}, base["row"].get("per_leg") or {},
                    f"phase0:{row.get('leg')}", baseline_line=base["line"])
                if measurement is not None:
                    measurements.append(measurement)
            if refusal is not None:
                leave_out(ref, refusal)
    counts = {"read": len(lines), "duplicates": len(lines) - len(unique),
              "baseline_legs": baselines}
    if gate == BETWEEN_PASS_GATE:
        measurements = between_pass(measurements, excluded)
    return measurements, excluded, counts


# ---------------------------------------------------------------------------
# One record, one program
# ---------------------------------------------------------------------------

def _tally(values: Iterable[str]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return "; ".join(f"{value} ({count} row(s))" for value, count in sorted(counts.items()))


def subject_barrier(table: str, measurements: Sequence[Mapping[str, Any]],
                    allow_unpinned: Sequence[str]) -> Dict[str, Any]:
    """The emitter barrier of ``table``, or a named refusal.

    ``not recorded`` is a VALUE: a row that did not pin a source is not evidence that
    it ran the source its neighbour pinned, so the two do not pool. Two RECORDED
    digests refuse outright; a straddle between recorded and unrecorded refuses until
    it is allowed by name, and what is allowed is written here and left out of what
    the consult compares (:func:`dispatch_preference.compared_emitters`).
    """
    allowed = sorted(set(allow_unpinned))
    unknown = [source for source in allowed if source not in SUBJECT_SOURCES]
    if unknown:
        raise CutRefused("unknown_subject_source",
                         f"--allow-unpinned-subject names {unknown}, and the barrier "
                         f"sources are {list(SUBJECT_SOURCES)}")
    digests: Dict[str, List[str]] = {}
    unpinned: Dict[str, int] = {}
    for source in dp.SUBJECT_SOURCES_BY_TABLE[table]:
        values = sorted({m["digests"].get(source) or NOT_RECORDED
                         for m in measurements})
        missing = sum(1 for m in measurements if not m["digests"].get(source))
        recorded = [value for value in values if value != NOT_RECORDED]
        if len(recorded) > 1:
            raise CutRefused(
                "mixed_subject_digest",
                f"the admitted rows timed {len(recorded)} different {source} sources ("
                + ", ".join(value[:12] for value in recorded) + "); rows timed on the "
                "older one were superseded and must not be passed beside the newer")
        if missing and recorded and source not in allowed:
            raise CutRefused(
                "unpinned_subject_source",
                f"{missing} of {len(measurements)} admitted rows do not pin {source} "
                f"and {len(measurements) - missing} pin {recorded[0][:12]}: 'not "
                "recorded' is not a digest, and a row that cannot say which emitter it "
                "timed is not evidence that it timed that one. Pass "
                f"--allow-unpinned-subject {source} to admit the straddle -- the record "
                "will carry it and the consult will not compare that source -- or cut "
                "the two populations apart")
        digests[source] = values
        unpinned[source] = missing
    return {"sources": list(dp.SUBJECT_SOURCES_BY_TABLE[table]), "digests": digests,
            "rows_not_recording": unpinned,
            "allowed_unpinned": [source for source in allowed if source in digests],
            "cannot_cover": CANNOT_COVER}


def one_program(measurements: List[Dict[str, Any]], excluded: List[Dict[str, Any]],
                select_route: Optional[str]) -> Tuple[List[Dict[str, Any]],
                                                       Dict[str, Any]]:
    """The measurements of ONE program and its route block, or a named refusal."""
    def leave_out(measurement: Mapping[str, Any], reason: str) -> None:
        excluded.append({"artifact": measurement["artifact"], "line": measurement["line"],
                         "case": measurement["case"], "reason": reason})

    if select_route is not None:
        kept = []
        for measurement in measurements:
            route = measurement["route"]
            if route is not None and route["id"] != select_route:
                leave_out(measurement, f"another repair route: {route['id']}")
            elif route is None and measurement["sha"][:12] != select_route.split(":")[0]:
                leave_out(measurement, "another deposit_repair digest: "
                                       f"{measurement['sha'][:12]}")
            else:
                kept.append(measurement)
        measurements = kept
    if not measurements:
        raise CutRefused("nothing_admitted",
                         "no row passed every floor on the selected route; a record "
                         "with no measurement in it is not written")
    routes = sorted({m["route"]["id"] for m in measurements if m["route"] is not None})
    digests = sorted({m["sha"] for m in measurements})
    if len(routes) > 1 or len(digests) > 1:
        raise CutRefused(
            "mixed_repair_route",
            "the admitted rows ran more than one deposit-repair program and one record "
            "is one program. Bracketed routes: "
            + (_tally(m["route"]["id"] for m in measurements if m["route"] is not None)
               or "none")
            + ". deposit_repair digests: " + _tally(m["sha"][:12] for m in measurements)
            + ". Name one with --select-route <id>; the others are then left out by name")
    policies = sorted({str(m["policy"]) for m in measurements})
    if len(policies) > 1:
        raise CutRefused("mixed_policy", "the admitted rows ran under more than one "
                         f"subnormal policy: {policies}")
    devices = sorted({(str(m["host"]["device"]), str(m["host"]["compute_capability"]))
                      for m in measurements})
    if len(devices) > 1:
        raise CutRefused("mixed_device", "the admitted rows were timed on more than one "
                         f"device: {devices}")
    bracketed = [m for m in measurements if m["route"] is not None]
    recorded_on = sum(1 for m in bracketed if m["route"]["counters_recorded"])
    if bracketed:
        first = bracketed[0]["route"]
        route = {name: first[name] for name in (
            "deposit_repair_sha256", "index", "repairs_per_bracketed_step",
            "restricted", "fell_back", "id")}
    else:
        route = {"deposit_repair_sha256": digests[0], "index": None,
                 "repairs_per_bracketed_step": None, "restricted": False,
                 "fell_back": False}
        route["id"] = dp.route_id(route)
    route["restriction_counters_recorded_on"] = (
        f"{recorded_on} of {len(bracketed)} bracketed rows")
    return measurements, route


# ---------------------------------------------------------------------------
# Keys and bands
# ---------------------------------------------------------------------------

def _rounded(value: float) -> float:
    return round(float(value), DECIMALS)


def _stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ratios = [row["ratio"] for row in rows]
    stats = {
        "fused_ms_per_step": _rounded(statistics.median(
            row["fused_ms_per_step"] for row in rows)),
        "baseline_ms_per_step": _rounded(statistics.median(
            row["baseline_ms_per_step"] for row in rows)),
        "ratio": _rounded(statistics.median(ratios)),
        "ratio_min": _rounded(min(ratios)),
        "ratio_max": _rounded(max(ratios)),
        "fused_spread": _rounded(max(row["fused_spread"] for row in rows)),
        "baseline_spread": _rounded(max(row["baseline_spread"] for row in rows)),
    }
    # Row by row, each against its own two spreads (dispatch_preference.least_margin):
    # the pooled extremes above are for reading, and describe no row that was timed.
    stats["veto_margin"] = dp.least_margin(rows)
    stats["slower_beyond_spread"] = stats["veto_margin"] > 0.0
    return stats


def attribute(measurements: Sequence[Mapping[str, Any]]
              ) -> Tuple[Dict[Tuple[Any, ...], List[Dict[str, Any]]], int]:
    """``({class: [attributed row]}, rows-times-products not attributed)``.

    One number, one product. A row prices its plan's sole fused product, or -- when
    the plan holds several -- the one and only one of them that carries the bracket.
    A plan of two clean products prices NEITHER: one joint ratio written into a key
    for each would let a partner that loses veto an innocent product.
    """
    classes: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    skipped = 0
    for measurement in measurements:
        shape = dp.shape_class(measurement["run_shape"], bracketed=False,
                               cells=measurement["cells"])
        products = measurement["products"]
        bracketed = sorted(label for label, held in products.items()
                           if held["bracketed"])
        priced = (sorted(products) if len(products) == 1
                  else bracketed if len(bracketed) == 1 else [])
        for label, product in sorted(products.items()):
            others = [(other, held["bracketed"]) for other, held
                      in sorted(products.items()) if other != label]
            if label not in priced:
                skipped += 1
                continue
            row = {name: measurement[name] for name in (
                "artifact", "line", "source", "case", "utc", "grid_shape", "cells",
                "after_probe")}
            if "baseline_line" in measurement:
                row["baseline_line"] = measurement["baseline_line"]
            for name in ("fused_ms_per_step", "baseline_ms_per_step", "ratio",
                         "fused_spread", "baseline_spread"):
                row[name] = _rounded(measurement[name])
            row["co_fused"] = [{"product": other, "bracketed": bracketed}
                               for other, bracketed in others]
            row["_slots"] = product["slots"]
            row["_displaces"] = product["displaces"]
            classes.setdefault((label, shape.storage, shape.dimensions,
                                shape.susceptibilities, product["bracketed"]),
                               []).append(row)
    return classes, skipped


def _public(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: value for name, value in row.items() if not name.startswith("_")}


def band_keys(identity: Tuple[Any, ...], rows: List[Dict[str, Any]],
              min_rows: int) -> Dict[str, Dict[str, Any]]:
    """Every key of one class: its bands, and the gaps where the verdict changes."""
    label, storage, dimensions, susceptibilities, bracketed = identity
    rows = sorted(rows, key=lambda r: (r["cells"], str(r["case"]), str(r["utc"]),
                                       r["artifact"], r["line"]))
    displaced = sorted({tuple(row["_displaces"]) for row in rows})
    axes = {"product": label, "storage": storage, "dimensions": dimensions,
            "susceptibilities": susceptibilities, "bracketed": bracketed,
            "slots": list(rows[0]["_slots"]), "displaces": list(displaced[0])}

    def entry(lo: int, hi: int, members: List[Dict[str, Any]], status: str,
              reason: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        value = dict(axes, status=status, band={"lo_cells": lo, "hi_cells": hi},
                     rows=len(members), measurements=[_public(r) for r in members])
        if members:
            value.update(_stats(members))
        if reason is not None:
            value["reason"] = reason
        return dp.key_string(label, storage, dimensions, susceptibilities, bracketed,
                             lo, hi), value

    sizes = sorted({row["cells"] for row in rows})
    if len(displaced) > 1:
        reason = ("the rows disagree on what the product displaces ("
                  + " / ".join("+".join(arms) for arms in displaced)
                  + "), so they are not one comparison")
        return dict([entry(sizes[0], sizes[-1], rows, "unmeasured", reason)])
    runs: List[List[int]] = []
    verdicts: List[bool] = []
    for size in sizes:
        verdict = _stats([r for r in rows if r["cells"] == size])["slower_beyond_spread"]
        if runs and verdicts[-1] is verdict:
            runs[-1].append(size)
        else:
            runs.append([size])
            verdicts.append(verdict)
    counts = {size: sum(1 for row in rows if row["cells"] == size) for size in sizes}

    def uncorroborated(span: Sequence[int]) -> str:
        return (f"no size in it carries {min_rows} rows, so none can set a band edge: "
                + "; ".join(f"{size:,} ({counts[size]} row"
                            f"{'' if counts[size] == 1 else 's'})" for size in span))

    keys: Dict[str, Dict[str, Any]] = {}
    for index, run in enumerate(runs):
        # A band edge is its reach, and one session may not set it. The run is trimmed
        # to sizes at least min_rows rows measured; the remainder keeps its rows and
        # its numbers under its own unmeasured key, so the next timing campaign sees what it owes.
        corroborated = [size for size in run if counts[size] >= min_rows]
        lo = corroborated[0] if corroborated else run[0]
        hi = corroborated[-1] if corroborated else run[-1]
        if corroborated and run[0] < lo:
            below = [size for size in run if size < lo]
            keys.update([entry(run[0], lo - 1,
                               [r for r in rows if r["cells"] < lo
                                and r["cells"] >= run[0]],
                               "unmeasured", uncorroborated(below))])
        members = [r for r in rows if lo <= r["cells"] <= hi]
        key, value = entry(lo, hi, members,
                           "measured" if corroborated else "unmeasured",
                           None if corroborated else uncorroborated(run))
        keys[key] = value
        if corroborated and hi < run[-1]:
            above = [size for size in run if size > hi]
            keys.update([entry(hi + 1, run[-1],
                               [r for r in rows if hi < r["cells"] <= run[-1]],
                               "unmeasured", uncorroborated(above))])
        if index + 1 < len(runs) and runs[index + 1][0] - run[-1] > 1:
            key, value = entry(
                run[-1] + 1, runs[index + 1][0] - 1, [], "unmeasured",
                f"the verdict changes between {run[-1]:,} and {runs[index + 1][0]:,} "
                "cells and no row lies between them")
            keys[key] = value
    return keys


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------

RULES = {
    "ratio": "baseline ms/step over fused ms/step, per row; above 1 the candidate is "
             "faster than what it displaces",
    "numbers": "fused and baseline ms/step and ratio are medians over the key's rows; "
               "fused_spread and baseline_spread are the largest window spread "
               "((max - min) / median) any feeding row showed on that leg",
    "veto": "a row's margin is 1 - (its fused_spread + its baseline_spread) - its ratio; "
            "veto_margin is the least margin of the key's rows and slower_beyond_spread "
            "is veto_margin > 0: EVERY row is slower by more than its own two spreads",
    "band": "per class, the sorted measured cell counts are cut into maximal runs with "
            "one verdict and each run is trimmed to sizes with at least min_rows rows; "
            "a band is [lo_cells, hi_cells] inclusive whose EDGES are those sizes, its "
            "INTERIOR is interpolated (a size between the edges need not have been run "
            "and the verdict is applied to it), the trimmed remainder and a verdict "
            "change between two adjacent sizes are explicit unmeasured keys, and "
            "outside the measured sizes there is no key: nothing is extrapolated",
    "attribution": "a row times a plan and prices its SOLE fused product, or -- when "
                   "the plan holds several -- the one and only one of them carrying the "
                   "deposit-repair bracket, listing the others under co_fused; a plan "
                   "whose products are all clean prices none of them",
    "storage": "complex when the run shape's complex_storage is true, else real",
    "susceptibilities": "the run shape's susceptibilities count, verbatim, as the "
                        "release tables pin it",
    "bracketed": "whether any slot the product holds is in the fused leg's "
                 "deposit_repair_route.bracketed_slots",
    "admission": "verdict TIMED, every recorded floor passed and all seven recorded "
                 "(the same-table control among them), the plan's slots served by this "
                 "table alone (substitution.tables_dispatched), lifted on the device, "
                 "no foreign process on the row's device by UUID before or after, and a "
                 "recorded deposit_repair digest",
    "one_program": "rows on more than one deposit-repair route (the whole identity: "
                   "digest, index, repairs per bracketed step, restricted, fell back), "
                   "fused-pair emitter, subnormal policy or device refuse to pool; a "
                   "source some rows pinned and others did not refuses until it is "
                   "allowed by name, and is then not compared at all",
}

#: The rules a record cut by the between-pass gate states: the window rule's, with
#: admission restated and the gate itself named. A record cut by the window rule states
#: :data:`RULES` byte for byte. A record holds the rule set it was cut by (``rules``), and
#: ``--check`` recomputes the whole record and compares it byte for byte, so editing
#: either set breaks the check of every record already cut by it until that record is
#: re-cut.
RULES_R33 = dict(
    RULES,
    admission="verdict TIMED or TIMED-NOT-REPORTABLE with every gating floor passed and "
              "all six recorded (the same-table control among them; the within-pass "
              "window floor of a row written before the between-pass gate is not read), the plan's slots "
              "served by this table alone (substitution.tables_dispatched), lifted on "
              "the device, no foreign process on the row's device by UUID before or "
              "after, a recorded deposit_repair digest, and the between-pass gate",
    between_pass="the between-pass gate (2026-10-16): the admitted rows of one case, grid "
                 "shape, repair program and timing instrument (window count and the probe "
                 "floor its target sets) are its passes; two or more enter only when the "
                 "spread of their "
                 "fused and of their baseline rates, (max - min) / median, is at most "
                 "5 % on each leg, else every row of the point is left out by name; a "
                 "point timed once has no between-pass spread and is not struck by it",
    window_spread="each row's within-pass window spreads are recorded (fused_spread, "
                  "baseline_spread) and gate no admission; the veto margin reads them as "
                  "dispatch_preference spells it",
)

#: The rules a Metal record states: the between-pass rule's, with admission restated for
#: the Metal table, whose rows read their neighbours from the bench's host record and
#: record the Metal launch-rate floor beside the six, and the between-pass sentence
#: stated without the date the gate was introduced on (a shipped record's sealed text
#: names no date). Stamped into a Metal record alone (:func:`rules_for`), so
#: :data:`RULES` and :data:`RULES_R33`, and every NVIDIA record cut by them, recompute
#: byte for byte; editing this set re-cuts every Metal record.
RULES_METAL = dict(
    RULES_R33,
    admission="verdict TIMED or TIMED-NOT-REPORTABLE with every gating floor passed (the "
              "Metal launch-rate floor, windows_launched_at_the_proved_rate, among the "
              "recorded floors) and all six recorded (the same-table control among them; "
              "the within-pass window floor of a row written before the between-pass gate "
              "is not read), the plan's slots served by this table alone "
              "(substitution.tables_dispatched), lifted on the device, no foreign process "
              "driving the Metal device before or after, read from the row's host record "
              "(host_before and host_after, foreign_processes; a record that did not read "
              "them is not read), a recorded deposit_repair digest, and the between-pass "
              "gate",
    between_pass="the between-pass gate: the admitted rows of one case, grid shape, repair "
                 "program and timing instrument (window count and the probe floor its "
                 "target sets) are its passes; two or more enter only when the spread of "
                 "their fused and of their baseline rates, (max - min) / median, is at most "
                 "5 % on each leg, else every row of the point is left out by name; a point "
                 "timed once has no between-pass spread and is not struck by it",
)


def rules_for(table: str, gate: str) -> Dict[str, str]:
    """The rule set a record of ``table`` cut under ``gate`` states: :data:`RULES` for
    the window rule, else :data:`RULES_METAL` for the Metal table and :data:`RULES_R33`
    for an NVIDIA one. ``--check`` recomputes a record through :func:`cut`, so by the
    same selection."""
    if gate == WINDOW_GATE:
        return RULES
    return RULES_METAL if table == "metal" else RULES_R33


#: Where each table's release lives: (module, name) of ``{product: route cases}``, the
#: map the dispatch path itself reads. Read only when the release is accounted for.
RELEASE_TABLES = {"triton": ("meep_gpu.fastpath", "RELEASED_FUSED_ARMS"),
                  "cuda": ("meep_gpu.fastpath_cuda", "CUDA_RELEASED_FUSED_ARMS"),
                  "metal": ("meep_gpu.metal_dispatch", "METAL_RELEASED_FUSED_ARMS")}

#: The admission refusals (:func:`admission_refusal`) that say no timing of this table
#: was made at all: the bench row is not a timing, is another table's row, or is a host
#: smoke row. An excluded row of this kind accounts for its case without timing it.
#: A row whose fused leg did not run under the shape rule (:data:`NOT_THE_SHAPE_RULE`)
#: timed no product a key prices, so it is one of them.
NOT_A_TIMING = ("verdict ", "a row of the ", "a host smoke row", NOT_THE_SHAPE_RULE)


def released_cases(table: str) -> Dict[str, List[str]]:
    """``{product: [route case]}`` of the table's shipped release."""
    import importlib  # noqa: PLC0415 - the dispatch layer, read only when asked for

    module, name = RELEASE_TABLES[table]
    release = getattr(importlib.import_module(module), name)
    return {str(label): sorted(str(case) for case in cases)
            for label, cases in sorted(release.items())}


def missing_from_release(keys: Mapping[str, Mapping[str, Any]],
                         excluded: Sequence[Mapping[str, Any]],
                         released: Mapping[str, Sequence[str]]) -> List[Dict[str, str]]:
    """Every released (product, case) the record does not account for, with why.

    The release check's own rule (``test_cut_timing_record``): a case is accounted for
    when a key prices a row of it, or when a row of it was timed on this table and
    excluded by name (a floor failed, two tables served the plan); a product is
    accounted for when it is a key, the unpriced partner of a priced product, or every
    one of its cases was timed and excluded. What fails that rule is named here rather
    than refused, so a partial set cuts and says what it owes.
    """
    measured = [row for entry in keys.values() for row in entry["measurements"]]
    priced = {row["case"] for row in measured}
    timed_out = {item["case"] for item in excluded
                 if not str(item["reason"]).startswith(NOT_A_TIMING)}
    keyed = {entry["product"] for entry in keys.values()}
    partnered = {partner["product"] for row in measured for partner in row["co_fused"]}
    reasons: Dict[str, List[str]] = {}
    for item in excluded:
        reasons.setdefault(str(item["case"]), [])
        if item["reason"] not in reasons[str(item["case"])]:
            reasons[str(item["case"])].append(str(item["reason"]))
    out: List[Dict[str, str]] = []
    for product, cases in sorted(released.items()):
        silent = (product not in keyed and product not in partnered
                  and not set(cases) <= timed_out)
        for case in cases:
            if case in timed_out:
                continue
            if case not in priced:
                why = ("no timing of this table: " + "; ".join(sorted(reasons[case]))
                       if case in reasons else "no row of this case among the rows cut")
            elif silent:
                why = ("rows of this case were priced, and none prices this product or "
                       "names it as an unpriced partner")
            else:
                continue
            out.append({"product": product, "case": case, "reason": why})
    return out


def cut(table: str, roots: Sequence[str], *, base: str = HERE,
        min_rows: int = MIN_ROWS_PER_KEY, select_route: Optional[str] = None,
        require_box_after: bool = False,
        allow_unpinned_subject: Sequence[str] = (),
        released: Optional[Mapping[str, Sequence[str]]] = None,
        gate: str = BETWEEN_PASS_GATE,
        preference: Optional[str] = FUSED_LEG_PREFERENCE,
        costs: bool = False) -> Dict[str, Any]:
    """The sealed record for ``table`` from every ``rows.jsonl`` under ``roots``.

    ``gate`` is the rule the rows are admitted by (:data:`GATES`): the between-pass gate
    of 2026-10-16, stamped in ``cut.gate``, or the window floor every record cut before it was
    cut by, which is not stamped, so such a record recomputes byte for byte.

    ``released`` (``{product: [route case]}``, :func:`released_cases`) accounts for the
    release: the record names what it lacks under ``missing``. Absent, nothing is
    accounted and the record carries neither the block nor the parameter, which is how
    every record cut before the block existed recomputes byte for byte.

    ``preference`` is the dispatch preference a fused row must have run under
    (:data:`FUSED_LEG_PREFERENCE`, stamped in ``cut.fused_leg_preference``); ``None`` is
    the rule every record cut before it was cut by, which is not stamped.

    ``costs`` stamps the record ``/3`` (``dispatch_preference.SCHEMA_COSTS``): the same
    keys, plus the size-aware cost points ``dispatch_preference.add_costs`` derives from
    them, stamped in ``cut.costs``. False writes ``/2``, which is how every record cut
    before the cost model recomputes byte for byte.
    """
    if table not in dp.TABLES:
        raise CutRefused("unknown_table", f"{table!r} is not one of {dp.TABLES}")
    if int(min_rows) < 1:
        raise CutRefused("bad_minimum", "--min-rows is at least 1")
    files = row_files(roots)
    lines, file_table = read_files(files, base)
    if not lines:
        raise CutRefused("no_rows", f"no rows.jsonl line under {list(roots)}")
    measurements, excluded, counts = gather(lines, table, require_box_after, gate,
                                            preference)
    measurements, route = one_program(measurements, excluded, select_route)
    barrier = subject_barrier(table, measurements, allow_unpinned_subject)
    classes, skipped = attribute(measurements)
    keys: Dict[str, Dict[str, Any]] = {}
    for identity, rows in sorted(classes.items(), key=lambda item: repr(item[0])):
        keys.update(band_keys(identity, rows, int(min_rows)))
    pinned: Dict[str, List[str]] = {}
    for source in ("deposit_repair",) + SUBJECT_SOURCES + INSTRUMENT_SOURCES:
        pinned[source] = sorted({m["digests"].get(source) or NOT_RECORDED
                                 for m in measurements})
    hosts = sorted({json.dumps(m["host"], sort_keys=True) for m in measurements})
    excluded.sort(key=lambda item: (item["artifact"], item["line"], item["reason"]))
    inputs_digest = hashlib.sha256("\n".join(
        f"{item['path']} {item['sha256']}" for item in file_table).encode()).hexdigest()
    record = {
        "schema": dp.SCHEMA,
        "table": table,
        "recorded_utc": max(str(m["utc"]) for m in measurements),
        "cut": {"by": CUT_BY, "table": table, "min_rows": int(min_rows),
                "select_route": select_route,
                "require_box_after": bool(require_box_after),
                "allow_unpinned_subject": sorted(set(allow_unpinned_subject))},
        "rules": dict(rules_for(table, gate), min_rows_per_key=int(min_rows)),
        "route": route,
        "subject_barrier": barrier,
        "host": [json.loads(host) for host in hosts],
        "policy": sorted({str(m["policy"]) for m in measurements}),
        "pinned_sources": pinned,
        "inputs": {"roots": sorted(_relative(root, base) for root in roots),
                   "files": file_table, "sha256": inputs_digest},
        "rows": dict(counts, admitted=len(measurements), excluded=len(excluded)),
        "contention": {"admitted_without_an_after_probe": sum(
            1 for m in measurements if not m["after_probe"])},
        "attribution": {"not_attributed": skipped,
                        "why": "rows-times-products left unpriced: a product is priced "
                               "only as its plan's sole fused product, or as the plan's "
                               "one bracketed product"},
        "excluded": excluded,
        "keys": keys,
    }
    if gate != WINDOW_GATE:
        record["cut"]["gate"] = gate
    if preference is not None:
        record["cut"]["fused_leg_preference"] = preference
        record["rules"]["fused_leg_preference"] = (
            f"a fused row is admitted only when its {ENVIRONMENT_KEY} records the "
            f"package's dispatch-preference switch set to {preference!r}: the shape "
            "rule composes its fused leg alone, so the row prices the product a key "
            "names and not the composition the default arbitrated")
    if released is not None:
        record["cut"]["account_for_release"] = True
        lacking = missing_from_release(keys, excluded, released)
        if lacking:
            record["missing"] = lacking
    # /3: the same record, stamped and priced by dispatch_preference.add_costs, so a /2
    # record re-stamped there is byte for byte this cut (cut.costs, rules.costs, costs)
    sealed = dp.add_costs(record) if costs else dp.seal(record)
    dp.validate(sealed, expect_table=table)
    return sealed


# ---------------------------------------------------------------------------
# The preview of served_in_dispatch_preferred
# ---------------------------------------------------------------------------

PREVIEW_LABEL = ("PREVIEW -- served_in_dispatch_preferred as it WOULD read if this "
                 "record were in force on its own route. Nothing consults the record "
                 "yet, no board reports this number, and the join below is this tool's, "
                 "not a board's")

#: The field letters a SOURCE can inject, as ``configuration.source_field_types``
#: spells them and as a slot name ends. The census path reads a product's bracket off
#: its leading slot's letter, and a letter outside this set cannot answer the question.
CENSUS_SOURCE_FIELD_LETTERS = ("B", "D")


def _census_rows(census: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name in ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl"):
        path = os.path.join(census, name)
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("measured") is False:
                    continue
                rows[f"{row.get('leg')}:{row.get('row')}"] = row
    return rows


def _census_configuration(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A census row's configuration as the boards hand it on: with its lift facts."""
    return dict(row.get("configuration") or {}, facts=row.get("facts") or {})


def _census_shape(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The run shape of one census row, by the boards' own reader."""
    import dispatch_reachability  # noqa: PLC0415 - imports the dispatch path; preview only
    return dispatch_reachability.run_shape_from_census(_census_configuration(row))


def _released_here(label: str, row: Mapping[str, Any], table: str) -> Optional[bool]:
    """Whether the shipped release admits ``label`` on this row; None if unaskable."""
    try:
        import dispatch_reachability  # noqa: PLC0415
        return bool(dispatch_reachability.released_for_row(
            label, _census_configuration(row), table)["dispatches"])
    except Exception:  # noqa: BLE001 - a preview never fails the cut
        return None


def _instances(board: Mapping[str, Any], census: Mapping[str, Mapping[str, Any]],
               labels: Mapping[str, str]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """``(instances, how)``: one per seam-instance a fused product's predicate admits."""
    listed = board.get("seam_instances")
    if isinstance(listed, list):
        out = []
        for item in listed:
            if item.get("predicate_admits") and item.get("product") in labels:
                out.append({"row": item["row"], "label": labels[item["product"]],
                            "bracketed": bool(item.get("in_seam_source")),
                            "seam_instances": 1,
                            "displaces": [item.get("curl_arm"),
                                          item.get("constitutive_arm")]})
        return out, "the board's own seam_instances (predicate_admits, by product)"
    out = []
    for name, row in sorted(census.items()):
        sources = set((row.get("configuration") or {}).get("source_field_types") or ())
        for product, label in sorted(labels.items()):
            block = row.get(product)
            if not isinstance(block, Mapping) or not block.get("covered_modulo_backend"):
                continue
            spans = block.get("spans") or ()
            # The bracket rule, checked rather than assumed. A product carries the
            # bracket when the field its LEADING slot writes is one a source injects,
            # and configuration.source_field_types spells those fields with the same
            # letters the slots end in. A leading slot that names no such field
            # (update_H, update_E) cannot answer the question, and answering "clean"
            # for it would ask the record for the wrong key.
            first = str(spans[0]) if spans else ""
            letter = first.rsplit("_", 1)[-1]
            known = letter in CENSUS_SOURCE_FIELD_LETTERS
            # A product spanning N slots serves N - 1 seam-instances of the 597: a
            # three-slot weld is one product and two of them, and a veto removes both.
            out.append({"row": name, "label": label,
                        "bracketed": letter in sources if known else None,
                        "unreadable_bracket": None if known else
                        ("the census cannot say whether this span carries the bracket: "
                         f"spans[0] is {first!r}, whose field letter is not one a "
                         "source can inject"),
                        "seam_instances": max(1, len(spans) - 1), "displaces": None})
    if not out:
        return None, ("the board carries no per-instance rows and the census rows carry "
                      "no per-product predicate blocks")
    return out, ("the census's per-product predicate blocks (covered_modulo_backend); "
                 "the board lists no per-instance rows, so slot arbitration between "
                 "products is NOT applied and the displaced arms are taken from the "
                 "record")


def preview(record: Mapping[str, Any], board_path: str,
            census: Optional[str] = None,
            live_route: Optional[str] = None) -> Dict[str, Any]:
    """How many dispatching seam-instances fall in a vetoed band. A preview, labelled.

    ``live_route`` is the deposit-repair digest of the tree the preview is cut on. It
    travels in the returned dict as well as in the report, so a reader of the JSON
    cannot quote the numbers without the fact that the record describes another tree.
    """
    with open(board_path) as handle:
        board = json.load(handle)
    block = ((board.get("aggregate") or {}).get("served_in_dispatch_measurement")
             or board.get("served_in_dispatch_measurement") or {})
    arms = dict(block.get("arms") or {})
    released = set(block.get("released_arms") or arms)
    labels = {product: label for product, label in
              ((block.get("label_map") or {}).get("products") or {}).items()
              if label in released}
    denominator = ((board.get("aggregate") or {}).get("denominator")
                   or board.get("denominator"))
    out: Dict[str, Any] = {
        "label": PREVIEW_LABEL, "table": record["table"], "board": board_path,
        "denominator": denominator,
        "record_route": record["route"]["deposit_repair_sha256"],
        "record_route_id": record["route"]["id"],
        "live_repair_route": live_route,
        "record_is_for_the_live_route": (
            None if live_route is None
            else live_route == record["route"]["deposit_repair_sha256"]),
        "served_in_dispatch": (block.get("served_in_dispatch")
                               if block.get("served_in_dispatch") is not None
                               else (board.get("aggregate") or {}).get(
                                   "served_in_dispatch")),
        "products": {}, "vetoed": None, "served_in_dispatch_preferred_preview": None,
    }
    census = census or board.get("census")
    if not census or not os.path.isdir(census):
        out["not_joined_because"] = f"the census directory {census!r} is not on this host"
        return out
    out["census"] = census
    rows = _census_rows(census)
    instances, how = _instances(board, rows, labels)
    if instances is None:
        out["not_joined_because"] = how
        return out
    out["joined_from"] = how
    # The record's own route and architecture, because the preview asks what it would
    # read IF it were in force on the runs its rows timed; the live route travels
    # beside the answer. The reader pins a record to exactly one capability
    # (dispatch_preference.validate), and cut() validates what it sealed.
    route = record["route"]
    capability, = dp.record_capabilities(record)
    displaced_by_product: Dict[str, List[List[str]]] = {}
    for entry in record["keys"].values():
        held = displaced_by_product.setdefault(entry["product"], [])
        if entry["displaces"] not in held:
            held.append(entry["displaces"])
    products: Dict[str, Dict[str, Any]] = {}
    unjoined = envelope_unasked = 0

    def tally_of(label: str) -> Dict[str, Any]:
        return products.setdefault(label, {
            "dispatching_on_the_board": arms.get(label, 0), "joined": 0,
            "outside_the_release_envelope": 0, "bracketed": 0, "vetoed": 0,
            "not_vetoed": 0, "no_record": {}, "vetoed_rows": []})

    for instance in instances:
        tally = tally_of(instance["label"])
        row = rows.get(instance["row"])
        if row is None:
            unjoined += 1
            continue
        admitted = _released_here(instance["label"], row, record["table"])
        if admitted is None:
            envelope_unasked += 1
        elif not admitted:
            tally["outside_the_release_envelope"] += instance["seam_instances"]
            continue
        weight = instance["seam_instances"]
        tally["joined"] += weight
        if instance["bracketed"] is None:
            # the bracket axis decides WHICH key is asked for, so an unreadable
            # bracket is no record rather than a guess at the clean one
            reason = f"unknown: {instance['unreadable_bracket']}"
            tally["no_record"][reason] = tally["no_record"].get(reason, 0) + weight
            continue
        tally["bracketed"] += int(instance["bracketed"])
        displaces = instance["displaces"]
        if displaces is None:
            known = displaced_by_product.get(instance["label"]) or []
            displaces = known[0] if len(known) == 1 else []
        verdict = dp.consult(
            record, table=record["table"], capability=capability,
            candidate=instance["label"],
            displaces=displaces, run_shape=_census_shape(row),
            bracketed=instance["bracketed"], mode="measured",
            repair_route=route, subject_sha256=dp.compared_emitters(record),
            cells=int(row["grid_cells"]), validated=True)
        if verdict.outcome == "no_record":
            reason = (f"{'bracketed' if instance['bracketed'] else 'clean'}: "
                      f"{verdict.reason}")
            tally["no_record"][reason] = tally["no_record"].get(reason, 0) + weight
        else:
            tally[verdict.outcome] += weight
            if verdict.vetoed:
                tally["vetoed_rows"].append(instance["row"])
    for label in arms:
        tally_of(label)
    for tally in products.values():
        tally["no_record"] = dict(sorted(tally["no_record"].items()))
        tally["vetoed_rows"].sort()
        tally["join_agrees_with_the_board"] = (
            tally["joined"] == tally["dispatching_on_the_board"])
    out["products"] = dict(sorted(products.items()))
    out["instances_without_a_census_row"] = unjoined
    out["instances_whose_release_envelope_could_not_be_asked"] = envelope_unasked
    out["joined"] = sum(tally["joined"] for tally in products.values())
    out["bracketed"] = sum(tally["bracketed"] for tally in products.values())
    out["vetoed"] = sum(tally["vetoed"] for tally in products.values())
    out["not_vetoed"] = sum(tally["not_vetoed"] for tally in products.values())
    reasons: Dict[str, int] = {}
    for tally in products.values():
        for reason, count in tally["no_record"].items():
            reasons[reason] = reasons.get(reason, 0) + count
    out["no_record"] = dict(sorted(reasons.items()))
    out["join_agrees_with_the_board"] = all(
        tally["join_agrees_with_the_board"] for tally in products.values())
    if isinstance(out["served_in_dispatch"], int):
        out["served_in_dispatch_preferred_preview"] = (out["served_in_dispatch"]
                                                       - out["vetoed"])
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def render_report(record: Mapping[str, Any], preview: Optional[Mapping[str, Any]] = None,
                  live_route: Optional[str] = None) -> str:
    keys = record["keys"]
    measured = {key: entry for key, entry in keys.items() if entry["status"] == "measured"}
    unmeasured = {key: entry for key, entry in keys.items()
                  if entry["status"] != "measured"}
    rows = record["rows"]
    out = [
        f"Dispatch-preference timing record -- {record['table']} table",
        f"record_sha256 {record[dp.DIGEST_FIELD]}",
        f"recorded_utc {record['recorded_utc']} (the newest admitted row)",
        f"cut by {record['cut']['by']} from " + ", ".join(record["inputs"]["roots"]),
        f"card: {', '.join(_device_names(record))} (architecture "
        f"{', '.join(dp.record_capabilities(record))})",
        f"route {record['route']['id']} -- deposit_repair "
        f"{record['route']['deposit_repair_sha256']}; restriction counters recorded on "
        f"{record['route']['restriction_counters_recorded_on']}",
        "emitters compared: " + (", ".join(
            f"{source} {digest[:12]}"
            for source, digest in sorted(dp.compared_emitters(record).items()))
            or "none"),
    ]
    if record["cut"].get("gate") == BETWEEN_PASS_GATE:
        # A window-rule record carries no stamp and its report reads as it always did.
        out.append("gate: " + record["rules"]["between_pass"])
    barrier = record["subject_barrier"]
    for source in barrier["allowed_unpinned"]:
        out.append(
            f"NOT COMPARED: {source} -- {barrier['rows_not_recording'][source]} of "
            f"{record['rows']['admitted']} admitted rows do not pin it, so the record "
            "cannot say which one they timed and the consult compares no digest for it "
            "(--allow-unpinned-subject)")
    out.append("the barrier cannot cover: " + barrier["cannot_cover"])
    if live_route is not None and live_route != record["route"]["deposit_repair_sha256"]:
        out += [
            "",
            "This record describes a repair route that is not the one in the tree. "
            f"meep_gpu/deposit_repair.py digests to {live_route}; every row here timed "
            f"{record['route']['deposit_repair_sha256']}. A consult handed the live "
            "digest answers no_record for every key until rows timed on the live bytes "
            "are cut.",
        ]
    out += [
        "",
        f"rows: read {rows['read']}, duplicate lines {rows['duplicates']}, baseline legs "
        f"{rows['baseline_legs']}, admitted {rows['admitted']}, left out "
        f"{rows['excluded']}; admitted without an after-probe "
        f"{record['contention']['admitted_without_an_after_probe']}; "
        f"rows-times-products not attributed {record['attribution']['not_attributed']}",
        f"keys: measured {len(measured)}, unmeasured {len(unmeasured)} "
        f"(minimum {record['rules']['min_rows_per_key']} rows a key)",
        "",
        "MEASURED",
    ]
    for key, entry in sorted(measured.items()):
        out.append(
            f"  {'VETO' if entry['slower_beyond_spread'] else 'keep'}  {key}  "
            f"{entry['ratio']:.3f}x (best row {entry['ratio_max']:.3f}, least margin "
            f"{entry['veto_margin']:+.3f}), "
            f"fused {entry['fused_ms_per_step']:.3f} against "
            f"{entry['baseline_ms_per_step']:.3f} ms/step over "
            f"{'+'.join(entry['displaces'])}, {entry['rows']} rows")
    out += ["", "UNMEASURED"]
    for key, entry in sorted(unmeasured.items()):
        best = (f"; best row {entry['ratio_max']:.3f}x"
                if entry.get("ratio_max") is not None else "")
        out.append(f"  {key}  {entry['reason']}{best}")
    reasons: Dict[str, int] = {}
    for item in record["excluded"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    out += ["", "LEFT OUT, BY REASON"]
    out += [f"  {count:4d}  {reason}" for reason, count in sorted(reasons.items())]
    if record["cut"].get("account_for_release"):
        lacking = record.get("missing") or []
        out += ["", f"MISSING FROM THE RELEASE: {len(lacking)} released (product, case) "
                    "pair(s) this record does not account for"]
        out += [f"  {item['product']} / {item['case']}: {item['reason']}"
                for item in lacking]
    if preview is not None:
        out += ["", preview["label"]]
        if preview.get("not_joined_because"):
            out.append(f"  not joined: {preview['not_joined_because']}")
        else:
            denominator = preview["denominator"]
            out += [
                f"  board {preview['board']}",
                f"  joined from {preview['joined_from']}",
                f"  served_in_dispatch {preview['served_in_dispatch']} / {denominator}; "
                f"instances in a vetoed band {preview['vetoed']}; "
                f"served_in_dispatch_preferred (PREVIEW) "
                f"{preview['served_in_dispatch_preferred_preview']} / {denominator}",
                f"  of {preview['joined']} joined seam-instances {preview['bracketed']} "
                f"carry the bracket; the record vetoes {preview['vetoed']}, clears "
                f"{preview['not_vetoed']}, and has no record for the rest:",
            ]
            out += [f"    {count:4d}  {reason}"
                    for reason, count in preview["no_record"].items()]
            if not preview["join_agrees_with_the_board"]:
                out.append("  The join and the board disagree on the products marked "
                           "below, so the preview total is not the board's arithmetic")
            out.append("  product: dispatching on the board / joined / bracketed / "
                       "vetoed / not vetoed / no record")
            for label, tally in preview["products"].items():
                none = sum(tally["no_record"].values())
                flag = ("" if tally["join_agrees_with_the_board"]
                        else "   <-- the join and the board disagree")
                out.append(f"  {label}: {tally['dispatching_on_the_board']} / "
                           f"{tally['joined']} / {tally['bracketed']} / "
                           f"{tally['vetoed']} / {tally['not_vetoed']} / {none}{flag}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# One record, one card: the names a record may be written under
# ---------------------------------------------------------------------------
#
# The names are the reader's (``dispatch_preference.record_path``, with its ``card``
# argument, ``card_slug`` and ``record_name_parts``): the dispatch path reads a card's
# own record under exactly the name this tool writes it, and nothing here spells one.

def _device_names(record: Mapping[str, Any]) -> List[str]:
    """Every device name a record's ``host[]`` rows stamp, as recorded, sorted."""
    return sorted({str((host or {}).get("device")) for host in record["host"]})


def record_card(record: Mapping[str, Any]) -> str:
    """The one card a record's rows ran on: the exact name its ``host[]`` rows stamp.

    A card on two drivers is two host rows and one card; two names are two cards, which
    the cut already refuses (``mixed_device``); no name is ``unnamed_card``.
    """
    names = sorted({str((host or {}).get("device") or "") for host in record["host"]})
    if len(names) > 1:
        raise CutRefused("mixed_device", "the admitted rows were timed on more than one "
                                         f"device: {names}")
    name = names[0] if names else ""
    try:
        dp.card_slug(name)
    except dp.RecordRefused as refusal:
        raise CutRefused(refusal.code, f"{refusal.detail}, so the record cannot be "
                                       "placed") from refusal
    return name


def _devices_named(path: str) -> Optional[List[str]]:
    """The devices the record at ``path`` says it timed, or ``None`` if it cannot say.

    A lenient read on purpose: the only question is whose evidence is at that name, and
    a record that fails the reader's other rules still answers it. One that cannot be
    parsed at all (or is not there) answers nothing.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return _device_names(json.load(handle))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _card_at(path: str, card: str) -> Optional[str]:
    """The one card whose record is at ``path``, or None when nothing is there.

    A file there that cannot say it is ONE card's record is refused, whatever ``--out``
    names: one that does not read as a record (``existing_record_unreadable``), and one
    whose rows name more than one device (``existing_record_names_another_device``).
    Either could be any card's, this one's included, so a cut neither writes over it
    nor places this card's record beside it.
    """
    if not os.path.exists(path):
        return None
    held = _devices_named(path)
    if not held:
        raise CutRefused(
            "existing_record_unreadable",
            f"{path} is already there and does not read as a record, so it cannot say "
            f"which card it was cut on, and this cut of {card} can be placed neither over "
            f"it nor beside it. Restore or retire it first")
    if len(held) > 1:
        raise CutRefused(
            "existing_record_names_another_device",
            f"{path} was cut on {held}, more than one device, so it is not one card's "
            f"record, and this cut of {card} can be placed neither over it nor beside it")
    return held[0]


def _slug_or_none(name: str) -> Optional[str]:
    """``name``'s slug, or None for a name that leaves none (it then matches no card)."""
    try:
        return dp.card_slug(name)
    except dp.RecordRefused:
        return None


def destination(table: str, record: Mapping[str, Any], directory: str) -> str:
    """Where this cut's record of ``table`` is written in ``directory``.

    The bare name (the architecture's representative) when it already holds this card's
    record; else this card's own name when that file exists (a re-cut rewrites a card's
    file wherever it sits); else the bare name when nothing is there (the first card cut
    for an architecture represents it); else this card's own name, beside the
    representative. Refused first, whatever ``--out`` names: a file at either name that
    cannot say it is one card's record (:func:`_card_at`); another exact device's record
    under this card's slug, at this card's own name or as the representative
    (``existing_record_names_another_device``: two cards, and one architecture's records
    keep one exact name per slug, so a slug never stands for two cards); and a card
    already at both names (``card_has_two_records``), since neither could be chosen
    without hiding the other.
    """
    capability, = dp.record_capabilities(record)
    card = record_card(record)
    slug = dp.card_slug(card)
    bare = os.path.join(directory, os.path.basename(dp.record_path(table, capability)))
    own = os.path.join(directory, os.path.basename(
        dp.record_path(table, capability, card=card)))
    at_bare = _card_at(bare, card)
    at_own = _card_at(own, card)
    if at_own not in (None, card):
        raise CutRefused(
            "existing_record_names_another_device",
            f"{own}, this card's own name, holds a record cut on {at_own!r}, and these "
            f"rows were timed on {card!r}: two exact names with the one slug {slug} are "
            "two cards, and an architecture's records keep one exact name per slug. There "
            "is no override")
    if at_bare not in (None, card) and _slug_or_none(at_bare) == slug:
        raise CutRefused(
            "existing_record_names_another_device",
            f"{bare} was cut on {at_bare!r} and these rows were timed on {card!r}: two "
            f"exact names with the one slug {slug} are two cards, and an architecture's "
            "records keep one exact name per slug. There is no override")
    if at_bare == card and at_own == card:
        raise CutRefused(
            "card_has_two_records",
            f"{bare} and {own} both hold a record of {card}: one card has one record, so "
            "retire one of them before cutting again")
    if at_bare == card:
        return bare
    if at_own is not None:
        return own
    if at_bare is None:
        return bare
    return own


def place(table: str, out: str, record: Mapping[str, Any]) -> str:
    """The capability the admitted rows stamp, once ``--out`` is this record's destination.

    The name is part of the record. One record prices one card of one architecture --
    rows from two devices already refuse (``mixed_device``) -- and the consult looks the
    file up by :func:`dispatch_preference.record_path`. So the architecture is read off
    the rows, the card off the rows too, and ``--out`` has to be exactly where
    :func:`destination` puts that card's record in ``--out``'s directory (a staging
    tree, a release copy): any other name is refused, and the refusal names the right
    one. Nothing is written by a refusal.
    """
    # cut() seals and then validates, and the reader refuses a record whose rows do
    # not name exactly ONE compute capability, so there is one to name the file for.
    capability, = dp.record_capabilities(record)
    card = record_card(record)
    wanted = destination(table, record, os.path.dirname(out) or ".")
    named = os.path.basename(wanted)
    parts = dp.record_name_parts(out)
    if parts is None or parts[0] != capability:
        raise CutRefused(
            "out_names_another_capability",
            f"the admitted rows were timed on architecture {capability}, on {card}, "
            f"whose record is named {named} here; --out names "
            f"{os.path.basename(out)!r}. The architecture is in the name so that a "
            f"consult on another architecture cannot find this record at all")
    if os.path.exists(out):
        held = _devices_named(out)
        if held is None:
            raise CutRefused(
                "existing_record_unreadable",
                f"{out} is already there and does not read as a record, so it cannot "
                f"say which device it was cut on; this cut timed {card}, whose record "
                f"is named {named} here. A cut never writes over a record that cannot "
                f"say whose it is")
        if held != [card]:
            raise CutRefused(
                "existing_record_names_another_device",
                f"{out} was cut on {held} and these rows were timed on {[card]}: a cut "
                f"replaces only its own card's record, and this one's is named {named} "
                f"here")
    if os.path.abspath(out) != os.path.abspath(wanted):
        raise CutRefused(
            "out_names_another_card",
            f"--out names {os.path.basename(out)!r}, and the record of {card} on "
            f"architecture {capability} is named {named} here: the bare name when it "
            f"already holds this card's record, else the card's own name when that file "
            f"is there, else the bare name when it is empty, else the card's own name "
            f"beside it. A cut never moves a card's record: to make this card the "
            f"representative, retire the representative's record and report, and this "
            f"card's own if it has them, then cut again")
    return capability


# ---------------------------------------------------------------------------
# The records under a root
# ---------------------------------------------------------------------------

def list_records(root: Optional[str] = None) -> int:
    """One line per record under ``root`` (default: the package): read only.

    Each line names the file's table and architecture, its role (the architecture's
    ``representative``, the name the dispatch path reads for every card of the
    architecture with no record of its own; a ``card record`` beside it, which that card
    alone reads; or ``named for neither``), its exact card and
    slug, ``record_sha256`` (first 12), ``recorded_utc``, schema, and whether its
    deposit-repair route and its compared emitters hold on this tree, as the consult
    compares them. A file the reader refuses is named with its refusal.
    """
    if root is None:
        root = os.path.dirname(os.path.abspath(dp.__file__))
    files = dp.record_files(root)
    if not files:
        say(f"no timing record under {root}")
        return EXIT_OK
    live_route = dp.repair_route_sha256()
    code = EXIT_OK
    for path in files:
        name = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            record = dp.load_record(path)
            card = record_card(record)
        except (dp.RecordRefused, CutRefused) as refusal:
            say(f"{name}: REFUSED {refusal}")
            code = EXIT_REFUSED
            continue
        table = record["table"]
        capability, = dp.record_capabilities(record)
        if os.path.abspath(path) == os.path.abspath(dp.record_path(table, capability,
                                                                   root=root)):
            role = "representative"
        elif os.path.abspath(path) == os.path.abspath(
                dp.record_path(table, capability, card=card, root=root)):
            role = "card record"
        else:
            role = "named for neither"
        recorded = record["route"]["deposit_repair_sha256"]
        live = dp.live_subject_sha256(table)
        moved = sorted(source for source, digest in dp.compared_emitters(record).items()
                       if live.get(source) != digest)
        route = ("route holds: yes" if recorded == live_route else
                 f"route holds: no ({recorded[:12]} against the live {live_route[:12]})")
        emitters = ("emitters hold: yes" if not moved else
                    f"emitters hold: no ({', '.join(moved)} moved)")
        say(f"{name}: table {table}, architecture {capability}, {role}, card {card} "
            f"({dp.card_slug(card)}), record_sha256 {record[dp.DIGEST_FIELD][:12]}, "
            f"recorded_utc {record['recorded_utc']}, schema {record['schema']}, "
            f"{route}, {emitters}")
    return code


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def check(path: str, base: str) -> int:
    """Recompute a record from the artifacts it names; refuse on any difference."""
    try:
        record = dp.load_record(path)
    except dp.RecordRefused as refusal:
        say(f"REFUSED {refusal}")
        return EXIT_REFUSED
    parameters = record["cut"]
    roots = [os.path.join(base, root) for root in record["inputs"]["roots"]]
    absent = [root for root in roots if not os.path.exists(root)]
    if absent:
        say(f"cannot recompute {path}: artifacts absent on this host: {absent}")
        return EXIT_ARTIFACTS_ABSENT
    try:
        released = (released_cases(parameters["table"])
                    if parameters.get("account_for_release") else None)
        again = cut(parameters["table"], roots, base=base,
                    min_rows=parameters["min_rows"],
                    select_route=parameters["select_route"],
                    # By the preference rule it was cut by: no stamp predates the rule
                    preference=parameters.get("fused_leg_preference"),
                    require_box_after=parameters["require_box_after"],
                    allow_unpinned_subject=parameters["allow_unpinned_subject"],
                    released=released,
                    # By the rule it was cut by: no stamp is the window floor
                    gate=parameters.get("gate", WINDOW_GATE),
                    costs=bool(parameters.get("costs")))
    except (CutRefused, dp.RecordRefused) as refusal:
        say(f"DIFFERS {path}: the rows on disk no longer cut at all -- {refusal}")
        return EXIT_DIFFERS
    if dp.dumps(again) == dp.dumps(record):
        say(f"OK {path} recomputes byte for byte from {len(record['inputs']['files'])} "
            f"file(s), {record['rows']['read']} rows ({record[dp.DIGEST_FIELD][:12]})")
        return EXIT_OK
    fields = sorted(name for name in set(record) | set(again)
                    if record.get(name) != again.get(name))
    keys = sorted(key for key in set(record["keys"]) | set(again["keys"])
                  if record["keys"].get(key) != again["keys"].get(key))
    say(f"DIFFERS {path}: fields {fields}" + (f"; keys {keys}" if keys else ""))
    return EXIT_DIFFERS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--table", choices=dp.TABLES)
    parser.add_argument("--rows", action="append", default=[],
                        help="a rows.jsonl, or a directory searched for them; repeatable")
    parser.add_argument("--out", help="the record to write; the report lands beside it. "
                                      "It is where the rows' card's record goes in its "
                                      "directory: the name dispatch_preference."
                                      "record_path gives the capability the rows stamp, "
                                      "or that name's card record beside it")
    parser.add_argument("--no-account-for-release", dest="account_for_release",
                        action="store_false",
                        help="do not read the table's release; the record then names "
                             "nothing it lacks (the default accounts for every released "
                             "(product, case) under missing)")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS_PER_KEY)
    parser.add_argument("--select-route",
                        help="the one repair route to cut, as a refusal names it")
    parser.add_argument("--require-box-after", action="store_true",
                        help="refuse rows that carry no after-probe of their device")
    parser.add_argument("--allow-unpinned-subject", action="append", default=[],
                        metavar="SOURCE",
                        help="admit rows that do not pin SOURCE beside rows that do; "
                             "the record carries the straddle and the consult then "
                             "compares no digest for it. Repeatable")
    parser.add_argument("--base", default=HERE,
                        help="what recorded paths are relative to (default: this directory)")
    parser.add_argument("--board", help="a fusion board JSON, for the labelled preview")
    parser.add_argument("--census", help="the census directory, if not the board's own")
    parser.add_argument("--costs", action="store_true",
                        help="cut a /3 record: the keys plus per-size cost points, which "
                             "the dispatch preference reads at the run's own size")
    parser.add_argument("--check", metavar="RECORD",
                        help="recompute RECORD from its artifacts and refuse on a difference")
    parser.add_argument("--list", action="store_true",
                        help="print every record under --root with its role, card, digest "
                             "and whether it holds on this tree; writes nothing")
    parser.add_argument("--root", help="with --list: the directory laid out as the package "
                                       "lays its records out (default: the package)")
    args = parser.parse_args(argv)
    if args.list:
        return list_records(args.root)
    if args.check:
        return check(args.check, args.base)
    if not (args.table and args.rows and args.out):
        parser.error("--table, --rows and --out are required to cut a record")
    try:
        record = cut(args.table, args.rows, base=args.base, min_rows=args.min_rows,
                     select_route=args.select_route,
                     require_box_after=args.require_box_after,
                     allow_unpinned_subject=args.allow_unpinned_subject,
                     released=(released_cases(args.table) if args.account_for_release
                               else None),
                     costs=args.costs)
        # Before anything is written: the name has to be where this card's record goes,
        # and whatever is already under it has to be this cut's own card's.
        capability = place(args.table, args.out, record)
    # dp.RecordRefused is the READER refusing what this tool built (cut seals and then
    # validates). It leaves by the same door as every other refusal, named.
    except (CutRefused, dp.RecordRefused) as refusal:
        say(f"REFUSED {refusal}")
        return EXIT_REFUSED
    say(f"read {record['rows']['read']} rows from {len(record['inputs']['files'])} "
        f"file(s); admitted {record['rows']['admitted']}")
    try:
        live = dp.repair_route_sha256()
    except OSError:
        live = None
    shown = (preview(record, args.board, args.census, live_route=live)
             if args.board else None)
    report = render_report(record, preview=shown, live_route=live)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(dp.dumps(record))
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    with open(stem + ".report.txt", "w") as handle:
        handle.write(report)
    say(report)
    say(f"wrote {args.out} and {stem}.report.txt -- the record of compute capability "
        f"{capability}, which is the architecture its rows ran on and the only one it "
        f"prices, cut on {record_card(record)}")
    if dp.record_name_parts(args.out)[1] is not None:
        representative = os.path.basename(dp.record_path(args.table, capability))
        absent = ("" if os.path.exists(os.path.join(os.path.dirname(args.out),
                                                     representative))
                  else ", which is not in this directory: here the architecture has no "
                       "representative")
        say(f"{os.path.basename(args.out)} is a card record: the dispatch path reads it "
            f"for {record_card(record)} alone, and every other card of architecture "
            f"{capability} reads {representative}, the architecture's "
            f"representative{absent}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
