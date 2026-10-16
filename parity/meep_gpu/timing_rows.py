#!/usr/bin/env python
"""Timing rows: the unit the timing round times, keeps, validates and cuts from.

WHY THIS EXISTS. Certification is modular: any subset of gates re-runs alone and each
record updates on its own (``rounds/1.0/chunks.py``, ``artifact_drift.py``). Timing is
made the same way here. The unit is a ROW, and every table, ladder and dispatch-preference
record is CUT by tool from the newest valid row of each key, never assembled by hand, so
any subset of rows can be re-timed at any time and the cuts re-run.

=============================================================================
THE ROW, AND ITS KEY
=============================================================================

Three kinds, one record each (schema :data:`SCHEMA`):

* ``ladder`` -- one (case, size, tier) of the identical-case ladder WITH THE LEGS IT IS
  COMPARED AGAINST: every GPU route's row, the array-path control and every MEEP leg,
  all from one ladder session on one idle box (R5: a ratio pairs legs timed in one
  session; R18: nothing else on the box). Key ``ladder|<architecture>|<case>|<cells>|
  <tier>``.
* ``fused`` -- one ``bench_fused_products.py`` unit: a (table, case, size) and its pass,
  the rows a dispatch-preference record is cut from. Key ``fused|<architecture>|
  <table>|<case>|<size>|p<pass>``, ``<size>`` ``res<r>`` or ``drive`` (the route gate's
  own size). The pass is in the key because a band edge needs two rows
  (``cut_timing_record.MIN_ROWS_PER_KEY``): the sweep times each point twice, and both
  are keys.
* ``check`` -- a DISPATCH CHECK of another key (``check|<that key>``): the same GPU
  legs at the same case and size on short windows, timed for nothing, kept for the
  launch counters (below).

A record is self-contained: the session it ran in, the unit's command, every process's
in-harness IMPORT RECORD (``gate_provenance.IMPORTED_KEY``, written by the benches
themselves since 1.0, replacing the external import-record hook), the LAUNCH COUNTERS of
every GPU leg (per slot: arm, kernel table, dispatches, and the family and gate the plan
cites), its ENVIRONMENT (the reference environment's build record and the MEEP build
records), the HOST GATES it passed (the quiet window read twice, the variance
precondition, the R18 watcher's readings and intrusions, power recorded and never
gated), its legs' medians WITH THEIR GATE STATUS (R20), and a digest of its own body.

=============================================================================
VALID, STALE, OR OWED A DISPATCH CHECK
=============================================================================

A row is assessed against a RELEASE TREE (:func:`assess`). The files its import records
name that the tree holds at other bytes, or not at all, have MOVED (the rule of
``artifact_drift.py``, which every rebind tool applies). No moved file: VALID. Otherwise
each moved file is one of three kinds, and the rule that narrows a re-time to the KEY is
in the second:

1. A STEP-PATH file (:meth:`Attribution.step_path`) changes every row that loaded it:
   every file outside a kernel table directory (the driver, the planners
   ``fastpath.py`` / ``fastpath_cuda.py`` / ``metal_dispatch.py``, the deposit repair,
   the host-write doors, the lift, the harness), and inside one the fused-pair
   emitters and the shared fusion block (``dispatch_preference.SUBJECT_FILES``: the
   seam loop, the two-slot install and the span arbitration run at plan time for
   every plan), the Metal residency stack, each table's package ``__init__`` and arm
   table (``cuda_kernels/arms.py`` and ``registry.py``, ``metal_kernels/arms.py`` and
   ``registry.py``, ``triton_kernels/launch.py``'s ``STEP_ORDER``), and every file a
   table's ``driver_dispatch`` ledger entry pins that no arm's entry pins (the Metal
   subnormal module; a fused pair's own module, which the CUDA route record pins
   beside its weld, is that ARM's kernel text and is read by rule 2, its effect on
   what dispatches by rule 3). In the 1.0 round's 150 NVIDIA
   import records the planners, ``deposit_repair.py``, ``host_writes.py``,
   ``fused_pairs.py`` and ``fused_polarization_pair.py`` were in EVERY row of both
   tables: here the re-time stays per table, because pooling a row timed before such
   a change with one timed after it would compare two programs.
2. A KERNEL file of an ARM: stale exactly when the row's launch counters show an arm
   whose kernel text the file belongs to DISPATCHED (``dispatches`` above zero). The
   arm's ledger entry (``families[arm].gate`` in the row's plan, looked up in the four
   ledgers) pins the arm's own modules, and every kernel-directory file those modules
   can import, at module level or inside a function, belongs to the arm too
   (``import_closure.Resolver.edges``, transitively): an upper bound on the text its
   launch builds, which catches a shared helper its curated pins leave out (the CUDA
   own-cell hoist module is pinned by one entry and imported by every hoisted kernel).
   The GATE's run closure is not used: a multi-product probe reaches nearly every
   kernel module, which would re-time every row for any kernel. An arm whose gate key
   no ledger holds (the 1.0 Triton single arms cite the sweep
   ``family_recert_2026-08-14``), or whose entry pins no kernel file, resolves by its
   BACKEND SLICE: every kernel file of its table belongs to it. Which rule each arm
   resolved by is written into the assessment.
3. A kernel file no arm the row launched belongs to is EXEMPT, but the row is then
   OWED A DISPATCH CHECK: a predicate in that file decides whether another arm is
   admitted, so the row is valid only once a ``check`` record of its key, cut wholly
   on the release tree, shows the same arms dispatched on the same slots. A dispatch
   check is a short GPU run; a re-time is the whole row, MEEP legs included.

4. A LEDGER (:data:`DATA_FILES`: the files the dispatch path reads as data to decide
   which arms it admits) is no arm's kernel text: a change to one owes every row that
   read it a dispatch check, as rule 3. The bench records their digests beside its
   import record (``data_sha256``).

A row that ran package code and records no launched slot (an ERROR row) cannot be
narrowed: every moved file re-times it. A row is REFUSED by name and never cut when:
a process that ran package code has no import record, names no package module in it,
or records no ledger digests; the unit was not opened on a quiet window of two clear
readings or was not watched while it ran (R18: a row collected by hand has neither);
the watcher saw an intrusion; a ladder entry of the row was written outside the
session that records it (R5: a ladder resumed across sessions); the ladder unit
exited non-zero (its passes are not the plan's); it has no reference environment build
record (R27); or its body does not match its digest.

=============================================================================
THE CUTS
=============================================================================

* :func:`select` -- per key, the newest valid row that carries a measurement (sessions
  ordered by their end), else the newest valid row at all, so a key whose only rows
  failed reaches the cut and is named there. THE STORE lives under
  ``parity/meep_gpu/results/<name>/`` (the ladder's snapshot leaves ``results`` out, and a
  record cut from it names its rows ``results/...``, as the shipped records do); the
  units' run root stays outside every tree the ladder reads.
* :func:`cut_ladder_table` -- the identical-case table of an architecture from the
  selected ladder rows: every leg's median over its passes with its gate status. The
  GPU and array-control legs carry the 5 % gate (within the bench's own floors and
  between passes); MEEP legs carry their between-pass spread, RECORDED AND NEVER GATED
  (owner ruling), and are struck only for what is not spread (digest, precision,
  threads, monitor). A ratio is the route's median over MEEP's best unstruck
  reference-configuration median and says whether it enters: it enters when the
  route's leg passed the gate AND the session's variance precondition was GO (R18). The
  table is accounted against the CATALOG: every catalog key with no valid measured row,
  never timed included, is named under ``missing``.
* :func:`cut_records` -- the dispatch-preference records, per table PER ARCHITECTURE,
  through ``cut_timing_record.cut`` on the selected fused rows' own bench lines, with
  the release accounted for (a partial set names its missing keys). Each architecture's
  record is written to the name ``dispatch_preference.record_path`` gives it; a record
  of an architecture whose rows are absent is never touched, and a record cut on
  another device is never replaced. Every catalog fused key of a table and
  architecture with no valid measured row is named in the result and in the report
  beside the record (the record's own ``missing`` names the released (product, case)
  pairs it lacks); a table and architecture the catalog names and no row reaches is
  named as not cut.

usage:
  timing_rows.py status   --store S --release-root R [--catalog C] [--json OUT]
                          [--environment-record F]
  timing_rows.py bill     --store S FILE [FILE ...]
  timing_rows.py table    --store S --release-root R --catalog C --architecture A
                          --out-md M --out-json J [--environment-record F]
  timing_rows.py records  --store S --release-root R --catalog C --out-root DIR [--base B]
                          [--environment-record F]
  timing_rows.py report   --store S --release-root R --architecture A --out-dir DIR
                          [--catalog C] [--environment-record F]

``--environment-record`` names the round's reference environment build record (R27):
every row that ran in another environment is then stale and owed a re-time. Without it,
a cut whose selected rows ran in two environments is refused by name.

Stdlib only (with the harness's ``artifact_drift``, ``import_closure`` and
``cut_timing_record``), and imported by no gate: a change here voids no artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for _path in (HERE, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import artifact_drift  # noqa: E402

SCHEMA = "meep_gpu.timing_row/1"
KINDS = ("ladder", "fused", "check")
ROLES = ("release", "ab")
#: The field holding the digest of every other field.
DIGEST_FIELD = "record_sha256"
#: The import record a bench writes on every row (``gate_provenance.IMPORTED_KEY``).
IMPORTED_KEY = "imported_source_sha256"
#: Beside it, the digests of the ledgers the process's kernel tables read
#: (:data:`DATA_FILES`), the versions of the libraries it loaded, and the package,
#: device and compiler switches of its environment (``bench_fused_products``).
DATA_KEY = "data_sha256"
LIBRARIES_KEY = "imported_libraries"
ENVIRONMENT_KEY = "process_environment"
#: The catalog the release round's driver reads (``rounds/1.0/cert8.py``).
CATALOG_SCHEMA = "meep_gpu.cert8_catalog/1"
#: The 5 % gate: a GPU or array leg's within-window floor (the bench's) and its
#: between-pass spread.
GATE = 0.05
#: How a MEEP leg's pass spread is read (owner ruling, the 2026-09-25 Mac night and the
#: 1.0 round): recorded beside its median, never a reason to strike it.
MEEP_SPREAD_RULE = ("MEEP's own pass spread is recorded and never strikes a leg "
                    "(owner ruling)")
STATUSES = ("valid", "check_owed", "stale", "refused")
#: The kernel table directories: a file under one may be ONE ARM's kernel text.
KERNEL_DIRS = {"triton": "meep_gpu/triton_kernels/", "cuda": "meep_gpu/cuda_kernels/",
               "metal": "meep_gpu/metal_kernels/"}
#: Kernel-directory files every row of a table runs whatever it dispatched: the table's
#: package init and its arm table (composed at plan time for every plan), and
#: ``triton_kernels/launch.py``, whose ``STEP_ORDER`` every composer re-imports. The
#: emitters and the Metal residency stack join from ``dispatch_preference.SUBJECT_FILES``
#: and every ``driver_dispatch`` ledger pin from the ledgers (:class:`Attribution`).
KERNEL_STEP_PATH = (
    "meep_gpu/triton_kernels/__init__.py", "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/cuda_kernels/__init__.py", "meep_gpu/cuda_kernels/arms.py",
    "meep_gpu/cuda_kernels/registry.py",
    "meep_gpu/metal_kernels/__init__.py", "meep_gpu/metal_kernels/arms.py",
    "meep_gpu/metal_kernels/registry.py",
)
#: The four ledgers, per table (``tools/ci/check_hygiene.LEDGERS``).
LEDGERS = {"triton": ("meep_gpu/triton_kernels/fingerprints.json",),
           "cuda": ("meep_gpu/cuda_kernels/certification.json",
                    "meep_gpu/cuda_kernels/fingerprints.json"),
           "metal": ("meep_gpu/metal_kernels/fingerprints.json",)}
#: The ledgers as data: what the dispatch path reads to admit an arm. No arm's kernel
#: text, so a change to one owes a dispatch check, never an arm's re-time.
DATA_FILES = frozenset(path for paths in LEDGERS.values() for path in paths)
#: The ledger entry that binds each table's route record (DESIGN section 8.2 step 3).
ROUTE_ENTRY = "driver_dispatch"
_SLUG = re.compile(r"[^A-Za-z0-9._=-]+")


class RowRefused(ValueError):
    """A row record that may not be read or written, refused by name."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def ladder_key(architecture: str, case: str, cells: int, tier: str) -> str:
    return f"ladder|{architecture}|{case}|{int(cells)}|{tier}"


def fused_key(architecture: str, table: str, case: str, size: str, pass_index: int) -> str:
    if not (size == "drive" or re.match(r"^res\d+$", str(size))):
        raise RowRefused("bad_size", f"{size!r} is neither 'drive' nor res<r>")
    return f"fused|{architecture}|{table}|{case}|{size}|p{int(pass_index)}"


def check_key(target: str) -> str:
    return f"check|{target}"


def parse_key(key: str) -> Dict[str, Any]:
    parts = str(key).split("|")
    if parts[0] == "check":
        return {"kind": "check", "target": "|".join(parts[1:])}
    if parts[0] == "ladder" and len(parts) == 5:
        return {"kind": "ladder", "architecture": parts[1], "case": parts[2],
                "cells": int(parts[3]), "tier": parts[4]}
    if parts[0] == "fused" and len(parts) == 6:
        return {"kind": "fused", "architecture": parts[1], "table": parts[2],
                "case": parts[3], "size": parts[4], "pass": int(parts[5][1:])}
    raise RowRefused("bad_key", f"{key!r} is not a timing row key")


def slug(key: str) -> str:
    """A directory name for ``key``: readable, and unique by a digest of the key."""
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return f"{_SLUG.sub('_', key.replace('|', '__'))}__{digest}"


# ---------------------------------------------------------------------------
# The catalog: which keys an architecture's cuts account for
# ---------------------------------------------------------------------------

def load_catalog(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, Mapping) or catalog.get("schema") != CATALOG_SCHEMA:
        raise RowRefused("unsupported_catalog", f"{path}: schema "
                         f"{(catalog or {}).get('schema')!r}, not {CATALOG_SCHEMA!r}")
    if not catalog.get("architecture"):
        raise RowRefused("unsupported_catalog", f"{path}: names no architecture")
    return dict(catalog)


def catalog_keys(catalog: Mapping[str, Any],
                 drive: Optional[Any] = None, skip_release_drive: bool = False
                 ) -> List[str]:
    """Every row key the catalog names, ladder rows first. A table whose ``drive`` is
    ``"release"`` takes the release's own DRIVE cases from ``drive(table)``; with
    ``skip_release_drive`` they are left out (a cut accounts for them through the
    record's own release accounting, ``cut_timing_record.missing_from_release``)."""
    architecture = str(catalog["architecture"])
    keys = []
    ladder = catalog.get("ladder") or {}
    for tier, sizes in sorted((ladder.get("sizes") or {}).items()):
        for case in ladder.get("cases") or ():
            for cells in sizes:
                keys.append(ladder_key(architecture, case, int(cells), tier))
    for table, spec in sorted(((catalog.get("fused") or {}).get("tables") or {}).items()):
        cases = spec.get("drive") or ()
        if cases == "release":
            if drive is not None:
                cases = drive(table)
            elif skip_release_drive:
                cases = ()
            else:
                raise RowRefused("release_drive", f"the catalog takes the {table} DRIVE "
                                 "cases from the release, and no reader of them was given")
        for case in cases:
            keys.append(fused_key(architecture, table, case, "drive", 1))
        sweep = spec.get("sweep") or {}
        for case in sweep.get("cases") or ():
            for res in sweep.get("res") or ():
                for index in range(1, int(sweep.get("passes") or 1) + 1):
                    keys.append(fused_key(architecture, table, case, f"res{res}", index))
    return keys


def catalog_tables(catalog: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The (table, architecture) pairs whose dispatch-preference record the catalog
    asks for, whether or not it names a key of them (a release-DRIVE table with no
    sweep names none)."""
    architecture = str(catalog["architecture"])
    return [(str(table), architecture) for table in
            sorted(((catalog.get("fused") or {}).get("tables") or {}))]


# ---------------------------------------------------------------------------
# The record: seal, store, load
# ---------------------------------------------------------------------------

def _canonical(document: Mapping[str, Any]) -> bytes:
    body = {k: v for k, v in document.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def seal(document: Mapping[str, Any]) -> Dict[str, Any]:
    sealed = dict(document)
    sealed[DIGEST_FIELD] = hashlib.sha256(_canonical(document)).hexdigest()
    return sealed


def dumps(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def validate(record: Any) -> None:
    """Refuse, by name, a record that is not a sealed timing row."""
    if not isinstance(record, Mapping):
        raise RowRefused("malformed", "a row record is a JSON object")
    if record.get("schema") != SCHEMA:
        raise RowRefused("unsupported_schema", f"stamped {record.get('schema')!r}, this "
                                               f"reader accepts {SCHEMA!r}")
    if record.get(DIGEST_FIELD) != hashlib.sha256(_canonical(record)).hexdigest():
        raise RowRefused("record_digest_mismatch", "the body does not match its digest: "
                                                   "edited after it was written")
    if record.get("kind") not in KINDS or record.get("role") not in ROLES:
        raise RowRefused("malformed", f"kind {record.get('kind')!r} role "
                                      f"{record.get('role')!r}")
    parsed = parse_key(record.get("key", ""))
    if parsed["kind"] != record["kind"]:
        raise RowRefused("malformed", f"key {record['key']!r} is not a {record['kind']} key")
    session = record.get("session") or {}
    if not all(isinstance(session.get(name), str) and session.get(name)
               for name in ("id", "started_utc", "ended_utc")):
        raise RowRefused("malformed", "the session names no id, start and end")


def record_path(store: str, record: Mapping[str, Any]) -> str:
    """``<store>/<kind>/<key slug>/<session id>__<session end>.json``: the end is in the
    name because a session the same command resumes re-runs a unit that did not finish,
    and that attempt's record is written beside the first, never over it."""
    session = record["session"]
    name = _SLUG.sub("_", f"{session['id']}__{session['ended_utc']}")
    return os.path.join(store, record["kind"], slug(record["key"]), f"{name}.json")


def write_record(store: str, record: Mapping[str, Any],
                 bench_line: Optional[bytes] = None) -> str:
    """Seal and write ``record`` (and, for a fused row, its bench line verbatim).

    APPEND-ONLY: a key's directory gains one file per attempt, and an attempt's file is
    never written twice, so every record a cut named is still there to ``--check``.
    """
    sealed = record if record.get(DIGEST_FIELD) else seal(record)
    validate(sealed)
    path = record_path(store, sealed)
    if os.path.exists(path):
        raise RowRefused("exists", f"{path} is already written: a session's row is "
                                   "written once")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if bench_line is not None:
        with open(path[:-5] + ".rows.jsonl", "wb") as handle:
            handle.write(bench_line.rstrip(b"\n") + b"\n")
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(dumps(sealed))
    os.replace(temporary, path)
    return path


def load_store(store: str) -> List[Dict[str, Any]]:
    """Every record under ``store``, each with its path under ``_path`` (not sealed)."""
    out: List[Dict[str, Any]] = []
    for kind in KINDS:
        base = os.path.join(store, kind)
        if not os.path.isdir(base):
            continue
        for directory in sorted(os.listdir(base)):
            folder = os.path.join(base, directory)
            for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
                if not name.endswith(".json"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as handle:
                    record = json.load(handle)
                try:
                    validate(record)
                except RowRefused as refusal:
                    record = {"_path": path, "_unreadable": str(refusal),
                              "key": record.get("key") if isinstance(record, dict)
                              else None}
                    out.append(record)
                    continue
                record["_path"] = path
                out.append(record)
    return out


def public(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Launch counters
# ---------------------------------------------------------------------------

def launch_counters_of(row: Mapping[str, Any], process: str) -> List[Dict[str, Any]]:
    """Per GPU leg of a bench row: its slots' counters, with the family and gate the plan
    cites for each slot's arm. Read off ``plans[<leg>]`` as the row recorded it."""
    out = []
    for leg, plan in sorted((row.get("plans") or {}).items()):
        if not isinstance(plan, Mapping):
            continue
        families = plan.get("families") or {}
        slots = {}
        for slot, counter in sorted((plan.get("launch_counters") or {}).items()):
            if not isinstance(counter, Mapping):
                continue
            arm = counter.get("arm")
            family = families.get(arm) if isinstance(families, Mapping) else None
            slots[slot] = {"arm": arm, "backend": counter.get("backend"),
                           "dispatches": int(counter.get("dispatches") or 0),
                           "family": (family or {}).get("family"),
                           "gate": (family or {}).get("gate")}
        out.append({"process": process, "leg": leg, "slots": slots})
    return out


def dispatched(record: Mapping[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    """``(backend, arm, gate)`` of every slot any GPU leg of the row launched."""
    seen = set()
    for block in record.get("launch_counters") or ():
        for counter in (block.get("slots") or {}).values():
            if counter.get("dispatches", 0) > 0 and counter.get("arm"):
                seen.add((str(counter.get("backend")), str(counter["arm"]),
                          counter.get("gate")))
    return sorted(seen, key=lambda item: (item[0], item[1], str(item[2])))


def dispatch_map(record: Mapping[str, Any]) -> Set[Tuple[str, str, str, str, str]]:
    """``(process, leg, slot, arm, backend)`` for every launched slot: what a dispatch
    check compares."""
    out = set()
    for block in record.get("launch_counters") or ():
        for slot, counter in (block.get("slots") or {}).items():
            if counter.get("dispatches", 0) > 0:
                out.add((str(block.get("process")), str(block.get("leg")), str(slot),
                         str(counter.get("arm")), str(counter.get("backend"))))
    return out


# ---------------------------------------------------------------------------
# Which arm a file belongs to: the ledgers and the gates' run closures
# ---------------------------------------------------------------------------

class Attribution:
    """The release tree's own answer to "whose certification does this file carry?"."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.entries: Dict[str, Dict[str, Set[str]]] = {}
        self.route_pins: Set[str] = set()
        for table, paths in LEDGERS.items():
            held = self.entries.setdefault(table, {})
            for relative in paths:
                path = os.path.join(self.root, relative)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as handle:
                    ledger = json.load(handle)
                for key, entry in ledger.items():
                    pins = entry.get("source_sha256") if isinstance(entry, Mapping) else None
                    if not isinstance(pins, Mapping):
                        continue
                    files = {self._normal(table, name) for name in pins}
                    if key == ROUTE_ENTRY:
                        self.route_pins |= files
                        continue
                    held.setdefault(key, set()).update(files)
        self._deps: Dict[str, Tuple[str, ...]] = {}
        self._resolver = None
        self.subject_files = {f"meep_gpu/{path}"
                              for path in release_subject_files(self.root).values()}

    @staticmethod
    def _normal(table: str, name: str) -> str:
        """Ledger pins are repo-relative; the oldest Triton route pins are bare names."""
        name = str(name)
        if name.startswith(("meep_gpu/", "parity/")):
            return name
        return f"meep_gpu/{name}"

    def step_path(self, name: str) -> Optional[str]:
        """Why ``name`` changes every row that loaded it, or None for a kernel file."""
        if not any(name.startswith(prefix) for prefix in KERNEL_DIRS.values()):
            return "outside every kernel table directory: every row that loaded it ran it"
        if name in KERNEL_STEP_PATH:
            return "the table's package init or arm table, composed for every plan"
        if name in self.subject_files:
            return ("a fused-pair emitter or the shared fusion block "
                    "(dispatch_preference.SUBJECT_FILES)")
        if name in self.route_pins and not self.pinned_by_an_arm(name):
            return f"pinned by a table's {ROUTE_ENTRY} record and by no arm's entry"
        return None

    def pinned_by_an_arm(self, name: str) -> bool:
        return any(name in files for entries in self.entries.values()
                   for files in entries.values())

    def kernel_deps(self, files: Iterable[str]) -> Set[str]:
        """Every kernel-directory file ``files`` can import, at module level or inside a
        function, transitively through kernel-directory files: an upper bound on the
        kernel text an arm's own modules build (``import_closure.Resolver.edges``)."""
        if self._resolver is None:
            import import_closure  # noqa: PLC0415 - the static reader, on demand

            self._resolver = import_closure.Resolver(Path(self.root))
        seen: Set[str] = set()
        todo = [name for name in files if _kernel_file(name)]
        while todo:
            name = todo.pop()
            if name in seen or not _kernel_file(name):
                continue
            seen.add(name)
            if name not in self._deps:
                try:
                    self._deps[name] = (tuple(self._resolver.edges(name, module_level=False))
                                        + tuple(self._resolver.parents(name)))
                except Exception:  # noqa: BLE001 - an unreadable module adds nothing more
                    self._deps[name] = ()
            todo.extend(self._deps[name])
        return seen

    def arm_files(self, backend: str, gate: Optional[str]) -> Tuple[Optional[Set[str]], str]:
        """``(files, rule)``: the kernel files the arm's certification carries; ``None``
        means its table's whole kernel slice. The arm's ledger entry (the gate key the
        plan cites) pins its own modules; what they can import within the kernel
        directories is theirs too. No entry, or an entry pinning no kernel file, cannot
        say: the arm takes its table's slice."""
        found = None
        for table in ([backend] if backend in self.entries else []) + sorted(self.entries):
            if gate and gate in self.entries.get(table, {}):
                found = (table, self.entries[table][gate])
                break
        if found is None:
            return None, f"backend-slice:{backend} (no ledger entry {gate!r})"
        own = {name for name in found[1] if _kernel_file(name)}
        if not own:
            return None, f"backend-slice:{backend} (entry {gate!r} pins no kernel file)"
        return self.kernel_deps(own), f"ledger:{found[0]}:{gate}"


def _kernel_file(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in KERNEL_DIRS.values())


def release_subject_files(root: str) -> Dict[str, str]:
    """``dispatch_preference.SUBJECT_FILES`` as the RELEASE TREE at ``root`` spells it,
    read as a literal (nothing is imported: the harness's own package may be another
    tree). Refused by name when the tree does not state it, because without it the
    emitters would read as one arm's kernel text and be narrowed."""
    path = os.path.join(root, "meep_gpu", "dispatch_preference.py")
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "SUBJECT_FILES"):
                value = ast.literal_eval(node.value)
                if isinstance(value, dict) and value:
                    return {str(k): str(v) for k, v in value.items()}
    except (OSError, SyntaxError, ValueError) as error:
        raise RowRefused("no_subject_files", f"{path}: {type(error).__name__}: {error}")
    raise RowRefused("no_subject_files", f"{path} states no SUBJECT_FILES literal: the "
                                         "emitters cannot be told from an arm's kernel text")


# ---------------------------------------------------------------------------
# The environment a row ran in (R27)
# ---------------------------------------------------------------------------

def file_sha256(path: Optional[str]) -> Optional[str]:
    """The sha256 of a reference environment's build record, as a row records it
    (``cert8.environment_block``), or None for no file."""
    if not path:
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def environment_id(record: Mapping[str, Any]) -> Optional[str]:
    """The sha256 of the reference environment's build record the row ran in (R27)."""
    reference = (record.get("environment") or {}).get("reference")
    return reference.get("sha256") if isinstance(reference, Mapping) else None


def one_environment(records: Iterable[Mapping[str, Any]], what: str) -> None:
    """Refuse, by name, one cut of rows from more than one environment.

    THE IMPORT RECORD DOES NOT SEE AN ENVIRONMENT. Rows timed in two reference
    environments (R27: another compiler, another Triton, another CuPy, another MEEP
    build), or whose processes loaded two versions of one library, ran two programs
    although every import record matches the tree, and one record or table of them would
    compare the two. "None recorded" is a value: it does not pool with a recorded one."""
    environments: Dict[str, List[str]] = {}
    libraries: Dict[str, Dict[str, List[str]]] = {}
    for record in records:
        key = str(record.get("key"))
        environments.setdefault(environment_id(record) or "none recorded", []).append(key)
        for process in record.get("processes") or ():
            for name, version in sorted((process.get(LIBRARIES_KEY) or {}).items()):
                held = libraries.setdefault(str(name), {}).setdefault(str(version), [])
                if key not in held:
                    held.append(key)
    if len(environments) > 1:
        raise RowRefused(
            "mixed_environment",
            f"{what}: the rows ran in {len(environments)} reference environments (R27): "
            + "; ".join(f"{digest[:12]}: {len(keys)} key(s), e.g. {keys[0]}"
                        for digest, keys in sorted(environments.items()))
            + ". Pass --environment-record with the round's build record: every row of "
              "another environment is then stale and owed a re-time")
    mixed = {name: versions for name, versions in sorted(libraries.items())
             if len(versions) > 1}
    if mixed:
        raise RowRefused(
            "mixed_libraries",
            f"{what}: the rows' processes loaded more than one version of "
            + "; ".join(f"{name} ({', '.join(f'{v}: {len(k)} key(s), e.g. {k[0]}' for v, k in sorted(versions.items()))})"
                        for name, versions in mixed.items())
            + ": two programs, which one cut does not pool")


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def moved_files(record: Mapping[str, Any], root: str,
                cache: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, List[str]]:
    """``{file: [process roles]}`` for every recorded file the tree holds at other bytes:
    the modules of the import record and the ledgers recorded beside it."""
    cache = {} if cache is None else cache
    out: Dict[str, List[str]] = {}
    for process in record.get("processes") or ():
        recorded = dict(process.get(IMPORTED_KEY) or {})
        recorded.update(process.get(DATA_KEY) or {})
        for name in artifact_drift.moved_files(recorded, Path(root), cache):
            out.setdefault(name, []).append(str(process.get("role")))
    return out


def _clear(reading: Any) -> bool:
    return isinstance(reading, Mapping) and not reading.get("intrusions")


def refusals(record: Mapping[str, Any], allow_unrecorded_environment: bool = False
             ) -> List[str]:
    """Why ``record`` may never be cut, whatever tree it is read against."""
    if record.get("_unreadable"):
        return [f"unreadable: {record['_unreadable']}"]
    why = []
    for process in record.get("processes") or ():
        if not process.get("ran_package"):
            continue
        imported = process.get(IMPORTED_KEY)
        if not imported:
            why.append(f"no import record: process {process.get('role')} ran package code "
                       "and recorded no imported_source_sha256")
        elif not any(str(name).startswith("meep_gpu/") for name in imported):
            why.append(f"no package module in the import record of process "
                       f"{process.get('role')}: the package it ran was not the tree the "
                       "record is keyed to")
        if process.get(DATA_KEY) is None:
            why.append(f"no ledger digests: process {process.get('role')} ran package code "
                       f"and recorded no {DATA_KEY}, so a ledger change could not be seen")
    if not record.get("processes"):
        why.append("no process recorded")
    gates = record.get("gates") or {}
    # R18, AS EVIDENCE: a unit opened on a quiet window of two clear readings and watched
    # while it ran. A row collected by hand carries neither and is never cut.
    if gates.get("collected_by_hand"):
        why.append("collected by hand: no quiet window or watcher was recorded around the "
                   "unit (R18)")
    else:
        quiet = gates.get("quiet")
        if not (isinstance(quiet, list) and len(quiet) >= 2 and all(map(_clear, quiet))):
            why.append("no quiet window of two clear readings before the unit (R18)")
        watcher = gates.get("watcher")
        if not isinstance(watcher, Mapping) or not watcher.get("readings"):
            why.append("no watcher readings while the unit ran (R18)")
    watcher = gates.get("watcher") or {}
    if watcher.get("intrusions"):
        why.append(f"intruded: the watcher saw {len(watcher['intrusions'])} intrusion(s) "
                   "during the unit (R18)")
    outside = gates.get("session_mismatch")
    if outside:
        why.append(f"another session: {len(outside)} ledger entr(y/ies) of this row were "
                   f"written outside the session that records it, e.g. {outside[0]} (R5)")
    if record.get("kind") == "ladder" or str(record.get("key", "")).startswith(
            "check|ladder|"):
        rc = (record.get("unit") or {}).get("rc")
        if rc != 0:
            why.append(f"the ladder unit exited rc {rc}: its passes are not the plan's")
    mismatch = gates.get("architecture_mismatch")
    if mismatch:
        why.append(f"architecture: the rows stamp {mismatch} and the key names "
                   f"{record.get('architecture')}")
    environment = (record.get("environment") or {}).get("reference")
    if not environment and not allow_unrecorded_environment:
        why.append("no reference environment build record (R27)")
    return why


def assess(record: Mapping[str, Any], root: str, attribution: "Attribution",
           checks: Sequence[Mapping[str, Any]] = (),
           allow_unrecorded_environment: bool = False,
           cache: Optional[Dict[str, Optional[str]]] = None,
           environment: Optional[str] = None) -> Dict[str, Any]:
    """The row's status against the release tree at ``root``, with every reason.
    ``environment`` is the sha256 of the round's reference environment build record:
    a row that ran in another is stale, whatever its import record says (R27)."""
    out: Dict[str, Any] = {"key": record.get("key"), "path": record.get("_path"),
                           "session": (record.get("session") or {}).get("id"),
                           "moved": {}, "stale": [], "exempt": {}, "arms": [],
                           "refused": refusals(record, allow_unrecorded_environment)}
    if out["refused"]:
        out["status"] = "refused"
        return out
    ran_in = environment_id(record)
    if environment is not None and ran_in != environment:
        out["stale"].append(
            f"the reference environment moved (R27): the row ran in "
            f"{(ran_in or 'none recorded')[:12]}, the round names {environment[:12]}; an "
            "environment is no file of the import record, and another one ran another "
            "program")
        out["status"] = "stale"
        return out
    moved = moved_files(record, root, cache)
    out["moved"] = moved
    if not moved:
        out["status"] = "valid"
        return out
    arms = []
    for backend, arm, gate in dispatched(record):
        files, rule = attribution.arm_files(backend, gate)
        arms.append({"backend": backend, "arm": arm, "gate": gate, "rule": rule,
                     "files": files})
    out["arms"] = [{k: v for k, v in arm.items() if k != "files"} for arm in arms]
    # NO LAUNCH COUNTERS, NO NARROWING: a row that ran package code and records no slot
    # it launched (an ERROR row, a row file without its plans) cannot say which arm a
    # kernel file belongs to, so every moved file re-times it.
    unnarrowed = not arms and any(p.get("ran_package") for p in record["processes"])
    for name in sorted(moved):
        if name in DATA_FILES and not unnarrowed:
            out["exempt"][name] = ("a ledger the dispatch path reads to admit arms: no "
                                   "arm's kernel text, and it may change what dispatches, "
                                   "so a dispatch check is owed")
            continue
        program = attribution.step_path(name)
        if program is not None:
            out["stale"].append(f"{name}: {program}")
            continue
        if unnarrowed:
            out["stale"].append(f"{name}: the row records no launched slot, so its "
                                "kernel files cannot be narrowed to an arm")
            continue
        table = next(t for t, prefix in KERNEL_DIRS.items() if name.startswith(prefix))
        owners = [arm for arm in arms
                  if (arm["files"] is None and arm["backend"] == table)
                  or (arm["files"] is not None and name in arm["files"])]
        if owners:
            out["stale"].append(f"{name}: kernel text of an arm this row launched ("
                                + ", ".join(f"{a['backend']}:{a['arm']} by {a['rule']}"
                                            for a in owners) + ")")
        else:
            out["exempt"][name] = ("no arm this row launched carries it; a predicate in "
                                   "it may admit another arm, so a dispatch check is owed")
    if out["stale"]:
        out["status"] = "stale"
        return out
    witness = dispatch_witness(record, checks, root, cache, environment)
    out["dispatch_check"] = witness
    out["status"] = "valid" if witness["ok"] else "check_owed"
    return out


def dispatch_witness(record: Mapping[str, Any], checks: Sequence[Mapping[str, Any]],
                     root: str, cache: Optional[Dict[str, Optional[str]]] = None,
                     environment: Optional[str] = None) -> Dict[str, Any]:
    """The newest check of ``record``'s key cut wholly on the tree (and, given one, in
    the round's environment), and its verdict."""
    want = dispatch_map(record)
    if not want:
        return {"ok": False, "check": None,
                "why": "the row records no launched slot: no check can witness it"}
    # THE LEGS THE ROW TIMED: a check lifts every leg, and a row timed with the fused
    # leg alone is compared on that leg.
    timed = {(str(block.get("process")), str(block.get("leg")))
             for block in record.get("launch_counters") or ()}
    candidates = sorted((c for c in checks if c.get("key") == check_key(record["key"])
                         and not refusals(c, True)),
                        key=lambda c: c["session"]["ended_utc"], reverse=True)
    for candidate in candidates:
        if moved_files(candidate, root, cache):
            continue
        if environment is not None and environment_id(candidate) != environment:
            continue
        lifted = {(str(block.get("process")), str(block.get("leg")))
                  for block in candidate.get("launch_counters") or ()}
        got = {item for item in dispatch_map(candidate) if item[:2] in timed}
        absent = sorted(timed - lifted)
        return {"ok": got == want and not absent, "check": candidate.get("_path"),
                "session": candidate["session"]["id"],
                "differs": sorted(map(list, got ^ want)),
                "legs_the_check_did_not_lift": [list(pair) for pair in absent]}
    return {"ok": False, "check": None,
            "why": "no dispatch check of this key was cut on the release tree"
                   + ("" if environment is None else " in the round's environment")}


# ---------------------------------------------------------------------------
# Selection: the newest valid row per key
# ---------------------------------------------------------------------------

def measured(record: Mapping[str, Any]) -> bool:
    """Whether a row carries a measurement: a fused row a timing verdict, a ladder row a
    GPU leg's median. An ERROR row is valid evidence of a failure, not a timing."""
    legs = record.get("legs") or {}
    if record.get("kind") == "fused":
        return str(legs.get("verdict") or "").startswith("TIMED")
    return any(leg.get("median") is not None for leg in (legs.get("gpu") or {}).values())


def select(records: Sequence[Mapping[str, Any]], root: str,
           allow_unrecorded_environment: bool = False,
           role: str = "release", environment: Optional[str] = None) -> Dict[str, Any]:
    """Per key: the newest valid row, every assessment, and what the key is owed.
    ``environment`` (:func:`file_sha256` of the round's build record) makes a row of
    any other reference environment stale."""
    attribution = Attribution(root)
    cache: Dict[str, Optional[str]] = {}
    checks = [r for r in records if r.get("kind") == "check"]
    keys: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("kind") == "check" or record.get("role", role) != role:
            continue
        keys.setdefault(str(record.get("key")), []).append(record)
    out: Dict[str, Any] = {}
    for key, held in sorted(keys.items()):
        held = sorted(held, key=lambda r: ((r.get("session") or {}).get("ended_utc", ""),
                                           (r.get("session") or {}).get("id", "")),
                      reverse=True)
        assessed = [assess(r, root, attribution, checks, allow_unrecorded_environment,
                           cache, environment) for r in held]
        # THE NEWEST VALID ROW THAT CARRIES A MEASUREMENT, else the newest valid row at
        # all: a later attempt that ERRORed does not shadow an earlier timing, and a key
        # whose only valid rows failed still reaches the cut, which names it missing.
        valid = [r for r, a in zip(held, assessed) if a["status"] == "valid"]
        chosen = next((r for r in valid if measured(r)), valid[0] if valid else None)
        newest = assessed[0]
        # A KEY WITH A VALID MEASURED ROW OWES NOTHING, even when a later attempt was
        # refused (an intrusion, another tree).
        owed = ("nothing" if chosen is not None and measured(chosen) else
                "a dispatch check" if newest["status"] == "check_owed" else "a re-time")
        out[key] = {"selected": chosen, "assessments": assessed, "owed": owed}
    return out


def bill(records: Sequence[Mapping[str, Any]], files: Sequence[str], root: str
         ) -> Dict[str, List[str]]:
    """Which keys a change to ``files`` re-times, which it owes a dispatch check, and
    which it leaves alone: the timing half of ``import_closure.py bill``."""
    attribution = Attribution(root)
    changed = set(files)
    out: Dict[str, List[str]] = {"retime": [], "check": [], "untouched": []}
    newest: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("kind") == "check" or record.get("_unreadable"):
            continue
        key = str(record["key"])
        if key not in newest or (record["session"]["ended_utc"]
                                 > newest[key]["session"]["ended_utc"]):
            newest[key] = record
    for key, record in sorted(newest.items()):
        loaded = {name for process in record.get("processes") or ()
                  for name in list(process.get(IMPORTED_KEY) or {})
                  + list(process.get(DATA_KEY) or {})}
        hit = sorted(changed & loaded)
        if not hit:
            out["untouched"].append(key)
            continue
        arms = [(backend, attribution.arm_files(backend, gate)[0])
                for backend, _arm, gate in dispatched(record)]
        unnarrowed = not arms and any(p.get("ran_package")
                                      for p in record.get("processes") or ())
        stale = False
        for name in hit:
            if name in DATA_FILES and not unnarrowed:
                continue
            if attribution.step_path(name) is not None or unnarrowed:
                stale = True
                break
            table = next(t for t, prefix in KERNEL_DIRS.items() if name.startswith(prefix))
            if any((files is None and backend == table) or (files and name in files)
                   for backend, files in arms):
                stale = True
                break
        out["retime" if stale else "check"].append(key)
    return out


# ---------------------------------------------------------------------------
# Medians and their gate status (R20)
# ---------------------------------------------------------------------------

def spread(values: Iterable[Optional[float]]) -> Optional[float]:
    found = [float(v) for v in values if isinstance(v, (int, float)) and v]
    if len(found) < 2:
        return None
    return (max(found) - min(found)) / statistics.median(found)


def leg_median(passes: Sequence[Mapping[str, Any]], kind: str) -> Dict[str, Any]:
    """One leg over its passes: its median, its between-pass spread, its gate status.

    ``kind`` is ``gpu`` (a route or the array control: the 5 % gate, every pass TIMED
    and unstruck, between-pass spread within it) or ``meep`` (struck only for what is
    not spread; the pass spread recorded, never gated)."""
    values = [p.get("rate") for p in passes]
    struck = [f"pass {p.get('pass')}: {p['struck']}" for p in passes if p.get("struck")]
    # A MEEP leg's spread is over the passes that enter its median; a GPU leg's over all.
    between = spread([p.get("rate") for p in passes if not p.get("struck")]
                     if kind == "meep" else values)
    if kind == "meep":
        kept = [p.get("rate") for p in passes if not p.get("struck")
                and isinstance(p.get("rate"), (int, float))]
        gate = ("struck: " + "; ".join(struck) if struck and not kept else
                "ungated: " + MEEP_SPREAD_RULE + (
                    f"; between-pass spread {between:.4f}" if between is not None else "")
                + (f"; {len(struck)} pass(es) struck and left out: " + "; ".join(struck)
                   if struck else ""))
        median = statistics.median(kept) if kept else None
        status = "struck" if median is None else "ungated"
    else:
        reasons = list(struck)
        if between is not None and between > GATE:
            reasons.append(f"between-pass spread {between:.4f} > {GATE}")
        numbers = [v for v in values if isinstance(v, (int, float))]
        median = statistics.median(numbers) if numbers else None
        status = "struck" if reasons or median is None else "passed"
        gate = ("passed the 5 % gate" if status == "passed"
                else "struck by the 5 % gate: " + ("; ".join(reasons) or "no measurement"))
    return {"median": None if median is None else round(float(median), 4),
            "passes": [{"pass": p.get("pass"), "rate": p.get("rate"),
                        "spread": p.get("spread"), "struck": p.get("struck")}
                       for p in passes],
            "between_pass_spread": None if between is None else round(between, 6),
            "status": status, "gate": gate}


# ---------------------------------------------------------------------------
# Building records from what a unit wrote
# ---------------------------------------------------------------------------

#: The ladder tiers whose rows are figures of record (``timing_ladder.TIMED_TIERS``).
TIMED_TIERS = ("priority", "rest", "largest")


def _jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def _last(path: str) -> Optional[Dict[str, Any]]:
    rows = _jsonl(path)
    return rows[-1] if rows else None


def _relative(path: str, root: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(
            os.sep, "/")
    except ValueError:
        return os.path.basename(path)


def meep_label(entry: Mapping[str, Any]) -> str:
    """``meep@<ranks>`` and the configuration a control changed: the leg's name."""
    configuration = entry.get("configuration") or {}
    parts = [f"meep@{entry.get('ranks')}"]
    if configuration.get("build") not in (None, "reference"):
        parts.append(str(configuration["build"]))
    if int(configuration.get("threads_per_rank") or 1) > 1:
        parts.append(f"t{configuration['threads_per_rank']}")
    if entry.get("binding_label") == "bound-hwthread" or (
            entry.get("measured") or {}).get("binding_label") == "bound-hwthread":
        parts.append("bound-hwthread")
    if configuration.get("split_chunks_evenly") is False:
        parts.append("splitcost")
    return "-".join(parts)


def base_record(kind: str, key: str, *, session: Mapping[str, Any], role: str,
                architecture: Optional[str], unit: Mapping[str, Any],
                environment: Mapping[str, Any], gates: Mapping[str, Any]
                ) -> Dict[str, Any]:
    return {"schema": SCHEMA, "kind": kind, "key": key, "role": role,
            "architecture": architecture, "session": dict(session), "unit": dict(unit),
            "environment": dict(environment), "gates": dict(gates),
            "processes": [], "launch_counters": [], "legs": {}, "notes": []}


def row_architecture(row: Mapping[str, Any]) -> Optional[str]:
    """The architecture a bench row stamped: an NVIDIA compute capability, or the Apple
    GPU family a Metal row stamps in the same field."""
    for plan in (row.get("plans") or {}).values():
        device = ((plan or {}).get("environment") or {}).get("device") or {}
        if device.get("compute_capability"):
            return str(device["compute_capability"])
    return None


def ladder_drivers(ladder_dir: str) -> Dict[str, str]:
    """The sha256 of the ladder's own drivers as the run held them, keyed repo-relative:
    its copies (``--drivers-mode copy``), else its snapshot of the tree. The ladder is the
    instrument -- gates, windows, interleaving -- so a change to it re-times its rows."""
    import timing_ladder  # noqa: PLC0415 - its DRIVER_FILES, one spelling

    out: Dict[str, str] = {}
    for name in timing_ladder.DRIVER_FILES:
        for candidate in (os.path.join(ladder_dir, "drivers", name),
                          os.path.join(ladder_dir, "tree", "parity", "meep_gpu", name)):
            if os.path.isfile(candidate):
                with open(candidate, "rb") as handle:
                    out[f"parity/meep_gpu/{name}"] = hashlib.sha256(
                        handle.read()).hexdigest()
                break
    return out


def bench_process(row: Mapping[str, Any], role: str, artifact: str) -> Dict[str, Any]:
    """A bench process as its row recorded it: the import record and, beside it, the
    ledgers its tables read, the libraries it loaded and its switches."""
    return {"role": role, "artifact": artifact, "ran_package": True,
            IMPORTED_KEY: row.get(IMPORTED_KEY), DATA_KEY: row.get(DATA_KEY),
            LIBRARIES_KEY: row.get(LIBRARIES_KEY), ENVIRONMENT_KEY: row.get(ENVIRONMENT_KEY)}


def outside_session(entries: Iterable[Mapping[str, Any]], session: Mapping[str, Any]
                    ) -> List[str]:
    """The ledger entries not written between the session's start and end (an entry
    with no time is outside: it cannot say when it ran)."""
    started, ended = session.get("started_utc"), session.get("ended_utc")
    out = []
    for entry in entries:
        stamp = entry.get("utc")
        if not (isinstance(stamp, str) and isinstance(started, str)
                and isinstance(ended, str) and started <= stamp <= ended):
            out.append(f"{entry.get('ledger')} pass {entry.get('pass')} at {stamp} "
                       f"(session {started} to {ended})")
    return out


def ladder_records(ladder_dir: str, *, session: Mapping[str, Any], architecture: str,
                   environment: Mapping[str, Any], gates: Mapping[str, Any],
                   unit: Mapping[str, Any], role: str = "release",
                   tiers: Sequence[str] = TIMED_TIERS) -> List[Dict[str, Any]]:
    """One record per (case, size, tier) a ladder run timed, from its ledger and the row
    files the ledger names. Nothing is transcribed; every number is read where the
    ladder wrote it."""
    entries = _jsonl(os.path.join(ladder_dir, "ladder_rows.jsonl"))
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("tier") not in tiers or entry.get("cells_planned") is None:
            continue
        groups.setdefault((str(entry.get("case")), int(entry["cells_planned"]),
                           str(entry["tier"])), []).append(entry)
    out = []
    for (case, cells, tier), group in sorted(groups.items()):
        key = ladder_key(architecture, case, cells, tier)
        record = base_record("ladder", key, session=session, role=role,
                             architecture=architecture, unit=unit,
                             environment=environment, gates=gates)
        drivers = ladder_drivers(ladder_dir)
        if drivers:
            record["processes"].append({"role": "ladder", "artifact": "drivers",
                                        "ran_package": False, IMPORTED_KEY: drivers})
        gpu: Dict[str, List[Dict[str, Any]]] = {}
        array: List[Dict[str, Any]] = []
        meep: Dict[str, Dict[str, Any]] = {}
        stamped = set()
        for entry in sorted(group, key=lambda e: int(e.get("index") or 0)):
            ledger = entry.get("ledger")
            measured = entry.get("measured") or {}
            if ledger == "gpu":
                route = str(entry.get("route"))
                fused = (measured.get("legs") or {}).get("fused") or {}
                gpu.setdefault(route, []).append({
                    "pass": entry.get("pass"), "rate": fused.get("mcell_steps_per_s"),
                    "spread": fused.get("spread"), "struck": entry.get("struck")})
                row = _last(str(entry.get("rows_file") or "")) or {}
                stamped.add(row_architecture(row))
                record["processes"].append(bench_process(
                    row, f"gpu:{route}:p{entry.get('pass')}",
                    _relative(str(entry.get("rows_file")), ladder_dir)))
                record["launch_counters"] += launch_counters_of(row, f"gpu:{route}")
            elif ledger == "array_control":
                array.append({"pass": entry.get("pass"),
                              "rate": entry.get("mcell_steps_per_s_mean"),
                              "spread": max([s.get("spread") or 0.0 for s in
                                             entry.get("sources") or ()] or [0.0]),
                              "struck": entry.get("struck")})
                # A STANDALONE CONTROL (``--array-control row``) is a process of its own
                # that runs the package: it is a process of the row, with the import
                # record its row carries. ``gpu_array_control.py`` writes none, so such
                # a row is refused by name rather than admitted on the GPU processes'
                # records alone (the 1.0 round's exit hook covered that process too).
                # The default control is read off the GPU rows' own array legs, whose
                # processes are recorded above.
                if entry.get("mode") == "standalone":
                    for source in entry.get("sources") or ():
                        own = os.path.join(str(source.get("row_dir") or ""), "rows.jsonl")
                        row = _last(own) or {}
                        record["processes"].append({
                            "role": f"array_control:p{entry.get('pass')}",
                            "artifact": _relative(own, ladder_dir), "ran_package": True,
                            IMPORTED_KEY: row.get(IMPORTED_KEY),
                            DATA_KEY: row.get(DATA_KEY)})
            elif ledger == "meep":
                label = meep_label(entry)
                leg = meep.setdefault(label, {"ranks": entry.get("ranks"),
                                              "control": entry.get("control"),
                                              "passes": []})
                leg["passes"].append({"pass": entry.get("pass"),
                                      "rate": measured.get("mcell_steps_per_s"),
                                      "spread": measured.get("spread"),
                                      "struck": entry.get("struck")})
                found = [r for r in _jsonl(str(entry.get("rows_file") or ""))
                         if r.get("row_id") == entry.get("row_id")
                         and r.get("row") == "measurement"]
                imported = (found[-1] if found else {}).get(IMPORTED_KEY)
                record["processes"].append({
                    "role": f"meep:{label}:p{entry.get('pass')}",
                    "artifact": _relative(str(entry.get("rows_file")), ladder_dir),
                    "ran_package": bool(imported and any(
                        name.startswith("meep_gpu/") for name in imported)),
                    IMPORTED_KEY: imported})
            else:
                record["notes"].append({"ledger": ledger, "pass": entry.get("pass"),
                                        "reason": entry.get("reason")
                                        or entry.get("struck")})
        # R5, READ OFF THE LEDGER: every entry of the row was written inside the session
        # that records it. A ladder resumed across sessions, or a directory collected
        # under bounds it did not run in, carries legs of another session.
        outside = outside_session(group, session)
        if outside:
            record["gates"] = dict(record["gates"], session_mismatch=outside)
        stamped.discard(None)
        if stamped and stamped != {architecture}:
            record["notes"].append({"ledger": "architecture",
                                    "reason": f"the GPU rows stamp {sorted(stamped)}, "
                                              f"the key names {architecture}"})
            record["gates"] = dict(record["gates"], architecture_mismatch=sorted(stamped))
        record["legs"] = {
            "gpu": {route: leg_median(passes, "gpu") for route, passes in sorted(gpu.items())},
            "array": leg_median(array, "gpu") if array else None,
            "meep": {label: dict(leg_median(leg["passes"], "meep"), ranks=leg["ranks"],
                                 control=leg["control"])
                     for label, leg in sorted(meep.items())},
        }
        out.append(record)
    return out


def fused_record(rows_path: str, *, key: str, session: Mapping[str, Any],
                 environment: Mapping[str, Any], gates: Mapping[str, Any],
                 unit: Mapping[str, Any], role: str = "release",
                 kind: str = "fused") -> Tuple[Dict[str, Any], bytes]:
    """The record of one bench unit and its row's line, kept verbatim for the cutter."""
    with open(rows_path, "rb") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        raise RowRefused("no_row", f"{rows_path} holds no row")
    line = lines[-1]
    row = json.loads(line)
    parsed = parse_key(key if kind != "check" else key[len("check|"):])
    record = base_record(kind, key, session=session, role=role,
                         architecture=parsed.get("architecture"), unit=unit,
                         environment=environment, gates=gates)
    stamped = row_architecture(row)
    if stamped is not None and stamped != parsed.get("architecture"):
        record["gates"] = dict(record["gates"], architecture_mismatch=[stamped])
        record["notes"].append({"ledger": "architecture",
                                "reason": f"the row stamps {stamped}, the key names "
                                          f"{parsed.get('architecture')}"})
    record["processes"].append(bench_process(row, f"bench:{row.get('drive_table')}",
                                             os.path.basename(rows_path)))
    record["launch_counters"] = launch_counters_of(row, f"bench:{row.get('drive_table')}")
    legs = {}
    for label, leg in sorted((row.get("per_leg") or {}).items()):
        per_step = leg.get("median_seconds_per_step")
        legs[label] = {"ms_per_step": None if per_step is None else 1e3 * per_step,
                       "spread": leg.get("spread"),
                       "gate": ("passed the bench's floors" if row.get("reportable")
                                else "struck by the bench's floors: " + ", ".join(
                                    k for k, v in (row.get("floors") or {}).items()
                                    if not v))}
    record["legs"] = {"verdict": row.get("verdict"), "per_leg": legs,
                      "ratios": row.get("ratios")}
    return record, line


# ---------------------------------------------------------------------------
# The identical-case table of one architecture
# ---------------------------------------------------------------------------

def cut_ladder_table(selection: Mapping[str, Any], architecture: str,
                     expected: Sequence[str] = ()) -> Dict[str, Any]:
    """The identical-case table from the selected ladder rows of ``architecture``.

    ``expected`` is the catalog's keys: one with no row in the store at all is named
    under ``missing`` beside the keys whose rows are none of them valid and measured,
    so a cut from part of the round says what it lacks rather than dropping it. Rows of
    two environments are refused by name (:func:`one_environment`)."""
    rows = []
    missing = []
    chosen_records = []
    for key in sorted(set(expected) - set(selection)):
        parsed = parse_key(key)
        if parsed["kind"] == "ladder" and parsed["architecture"] == architecture:
            missing.append({"key": key, "owed": "a re-time",
                            "why": ["no row of this key in the store"]})
    for key, chosen in sorted(selection.items()):
        parsed = parse_key(key)
        if parsed["kind"] != "ladder" or parsed["architecture"] != architecture:
            continue
        record = chosen["selected"]
        if record is None or not measured(record):
            missing.append({"key": key, "owed": chosen["owed"],
                            "why": [a["status"] for a in chosen["assessments"]]
                            + ([] if record is None else
                               ["the newest valid row carries no GPU median"])})
            continue
        chosen_records.append(record)
        legs = record.get("legs") or {}
        routes = {name: leg for name, leg in (legs.get("gpu") or {}).items()}
        meep = legs.get("meep") or {}
        reference = {name: leg for name, leg in meep.items() if not leg.get("control")}
        best_name, best = None, None
        for name, leg in sorted(reference.items()):
            if leg["status"] != "struck" and leg["median"] is not None and (
                    best is None or leg["median"] > best["median"]):
                best_name, best = name, leg
        ratios = {}
        # THE VARIANCE PRECONDITION IS IN FORCE (R18): a ratio of a session whose
        # precondition was not run, or did not read GO, is printed and does not enter.
        variance = _variance_status(record)
        for name, leg in sorted(routes.items()):
            if best is None or leg["median"] is None:
                ratios[name] = {"ratio": None, "enters": False,
                                "why": "no unstruck MEEP reference leg" if best is None
                                else "no GPU median"}
                continue
            ratios[name] = {"ratio": round(leg["median"] / best["median"], 4),
                            "over": best_name,
                            "enters": leg["status"] == "passed" and variance == "GO",
                            "why": f"{name}: {leg['gate']}; MEEP {best_name}: "
                                   f"{best['gate']}; variance precondition: {variance}"}
        rows.append({"key": key, "case": parsed["case"], "cells": parsed["cells"],
                     "tier": parsed["tier"], "session": record["session"]["id"],
                     "row": "/".join(Path(record["_path"]).parts[-3:])
                     if record.get("_path") else None,
                     "record_sha256": record.get(DIGEST_FIELD),
                     "variance": variance,
                     "gpu": routes, "array": legs.get("array"), "meep": meep,
                     "meep_best": best_name, "ratios": ratios})
    one_environment(chosen_records, f"the {architecture} identical-case table")
    return {"schema": "meep_gpu.timing_table/1", "architecture": architecture,
            "rules": {"gate": GATE, "meep": MEEP_SPREAD_RULE,
                      "selection": "per key the newest row VALID on the release tree "
                                   "that carries a measurement",
                      "ratio": "route median over the best unstruck MEEP reference "
                               "median of the same row (one session, R5); it enters when "
                               "the route passed the gate and the session's variance "
                               "precondition read GO (R18)"},
            "rows": rows, "missing": missing}


def _variance_status(record: Mapping[str, Any]) -> str:
    """The session's variance precondition as a row of the table states it."""
    verdict = (record.get("gates") or {}).get("variance")
    if not verdict:
        return "not recorded"
    if verdict.get("not_run"):
        return f"not run: {verdict['not_run']}"
    return str(verdict.get("verdict") or ("GO" if verdict.get("ok") else "NO-GO"))


def render_table(table: Mapping[str, Any]) -> str:
    out = [f"# Identical-case table, {table['architecture']}", "",
           "Every figure is a median over a row's passes, read from the row records; "
           "each carries its gate status. " + table["rules"]["meep"] + ".", "",
           "| case | cells | tier | route | median | gate | MEEP best | median | ratio | enters "
           "| variance |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in table["rows"]:
        best = row["meep"].get(row["meep_best"]) if row["meep_best"] else None
        for name, leg in sorted(row["gpu"].items()):
            ratio = row["ratios"].get(name) or {}
            out.append(
                f"| {row['case']} | {row['cells']:,} | {row['tier']} | {name} | "
                f"{leg['median']} | {leg['gate']} | {row['meep_best']} | "
                f"{best['median'] if best else None} | {ratio.get('ratio')} | "
                f"{'yes' if ratio.get('enters') else 'no'} | {row['variance']} |")
    if table["missing"]:
        out += ["", f"Keys without a valid row: {len(table['missing'])}"]
        out += [f"- {item['key']}: owed {item['owed']}" for item in table["missing"]]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The dispatch-preference records, per table per architecture
# ---------------------------------------------------------------------------

def cut_records(selection: Mapping[str, Any], out_root: str, base: str = HERE,
                tables: Sequence[str] = ("triton", "cuda", "metal"),
                released: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
                write: bool = True, expected: Sequence[str] = (),
                named: Sequence[Tuple[str, str]] = ()) -> List[Dict[str, Any]]:
    """Cut one record per (table, architecture) present among the selected fused rows.

    ADDED, NEVER REPLACED: each lands at ``dispatch_preference.record_path`` under
    ``out_root``; an architecture with no selected row is not touched, and a file cut on
    another device is refused (``cut_timing_record.refuse_to_replace_another_device``).

    NOTHING IS DROPPED IN SILENCE: ``expected`` is the catalog's keys. Every fused key of
    a (table, architecture) with no valid measured row -- never timed included -- is in
    the result's ``missing_keys`` and in the report beside the record, and a (table,
    architecture) the catalog or the store names that no selected row reaches is in the
    result as not cut. ``named`` is the catalog's (table, architecture) pairs
    (:func:`catalog_tables`): a table whose DRIVE cases are the release's and that has no
    sweep names no key of its own, and with no row it would otherwise vanish.

    ONE ENVIRONMENT PER RECORD: the selected rows of a (table, architecture) that ran in
    two reference environments, or loaded two versions of one library, are refused by
    name (:func:`one_environment`), as ``cut_timing_record`` refuses two subjects.
    """
    import cut_timing_record as cut  # noqa: PLC0415
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    groups: Dict[Tuple[str, str], List[str]] = {}
    chosen_records: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    lacking: Dict[Tuple[str, str], List[str]] = {}
    for table, architecture in named:
        if table in tables:
            groups.setdefault((str(table), str(architecture)), [])
    for key in sorted(set(selection) | set(expected)):
        parsed = parse_key(key)
        if parsed["kind"] != "fused" or parsed["table"] not in tables:
            continue
        group = (parsed["table"], parsed["architecture"])
        chosen = selection.get(key)
        record = chosen["selected"] if chosen else None
        if record is None or not measured(record):
            lacking.setdefault(group, []).append(
                f"{key}: " + ("no row of this key in the store" if chosen is None
                              else f"owed {chosen['owed']}"))
        if record is None:
            groups.setdefault(group, [])
            continue
        lines = record["_path"][:-5] + ".rows.jsonl"
        groups.setdefault(group, []).append(lines)
        chosen_records.setdefault(group, []).append(record)
    written = []
    for (table, architecture), files in sorted(groups.items()):
        target = dp.record_path(table, architecture, root=out_root)
        missing_keys = lacking.get((table, architecture), [])
        result = {"table": table, "architecture": architecture, "path": target,
                  "rows": len(files), "missing_keys": missing_keys}
        if not files:
            result["refused"] = (f"not cut: no valid row of {table} on {architecture}; "
                                 f"{len(missing_keys)} key(s) without one")
            written.append(result)
            continue
        try:
            one_environment(chosen_records.get((table, architecture), ()),
                            f"the {table} record on {architecture}")
            record = cut.cut(table, sorted(files), base=base,
                             released=(released or {}).get(table)
                             if released is not None else cut.released_cases(table))
            cut.out_capability(table, target, record)
            cut.refuse_to_replace_another_device(target, record)
        except (cut.CutRefused, dp.RecordRefused, RowRefused) as refusal:
            result["refused"] = str(refusal)
            written.append(result)
            continue
        result["missing"] = len(record.get("missing") or ())
        result["record_sha256"] = record[dp.DIGEST_FIELD]
        if write:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as handle:
                handle.write(dp.dumps(record))
            report = cut.render_report(record)
            report += (f"\nROW KEYS WITHOUT A VALID MEASURED ROW: {len(missing_keys)} of "
                       f"the catalog's and the store's {table} keys on {architecture}\n"
                       + "".join(f"  {item}\n" for item in missing_keys))
            with open(target[:-5] + ".report.txt", "w") as handle:
                handle.write(report)
        written.append(result)
    return written


def stage_fused_report(selection: Mapping[str, Any], architecture: str, out_dir: str
                       ) -> Dict[str, Any]:
    """Stage the selected fused rows of ``architecture`` as ``build_fused_timing_report``
    reads them: the DRIVE rows in one file, the sweep as ``sweep/<table>/<case>_res<R>/
    rows.jsonl`` (every pass of a point in its file). Copies of the rows' own lines, in
    key order, so the report is a view of exactly the rows the records were cut from.
    Rows of two environments are refused by name (:func:`one_environment`)."""
    drive: List[bytes] = []
    sweep: Dict[Tuple[str, str, str], List[bytes]] = {}
    one_environment([chosen["selected"] for key, chosen in sorted(selection.items())
                     if chosen["selected"] is not None
                     and parse_key(key)["kind"] == "fused"
                     and parse_key(key)["architecture"] == architecture],
                    f"the {architecture} fused-product report")
    for key, chosen in sorted(selection.items()):
        parsed = parse_key(key)
        record = chosen["selected"]
        if (parsed["kind"] != "fused" or record is None
                or parsed["architecture"] != architecture):
            continue
        with open(record["_path"][:-5] + ".rows.jsonl", "rb") as handle:
            line = handle.read().rstrip(b"\n") + b"\n"
        if parsed["size"] == "drive":
            drive.append(line)
        else:
            sweep.setdefault((parsed["table"], parsed["case"], parsed["size"]),
                             []).append(line)
    os.makedirs(out_dir, exist_ok=True)
    drive_path = os.path.join(out_dir, "drive_rows.jsonl")
    with open(drive_path, "wb") as handle:
        handle.write(b"".join(drive))
    sweep_root = os.path.join(out_dir, "sweep")
    for (table, case, size), lines in sorted(sweep.items()):
        folder = os.path.join(sweep_root, table, f"{case}_{size}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "rows.jsonl"), "wb") as handle:
            handle.write(b"".join(lines))
    return {"drive": drive_path, "drive_rows": len(drive), "sweep": sweep_root,
            "sweep_points": len(sweep)}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _status_lines(selection: Mapping[str, Any]) -> List[str]:
    lines = []
    counts: Dict[str, int] = {}
    for key, chosen in sorted(selection.items()):
        newest = chosen["assessments"][0]
        counts[chosen["owed"]] = counts.get(chosen["owed"], 0) + 1
        detail = (newest["refused"] or newest["stale"]
                  or list(newest.get("exempt") or ()) or ["no file moved"])
        lines.append(f"{key}: newest {newest['status']} (session {newest['session']}); "
                     f"owed {chosen['owed']}; " + "; ".join(detail[:3]))
    total = len(selection)
    lines.append("keys: " + ", ".join(f"owed {name} {count} of {total}"
                                      for name, count in sorted(counts.items())))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="each key: its newest row's status and what "
                                           "it is owed")
    status.add_argument("--store", required=True)
    status.add_argument("--release-root", default=REPO_ROOT)
    status.add_argument("--allow-unrecorded-environment", action="store_true")
    status.add_argument("--catalog", help="name the catalog's keys with no row as well")
    status.add_argument("--json")
    billing = sub.add_parser("bill", help="the keys a change to FILE re-times or owes a "
                                          "dispatch check")
    billing.add_argument("--store", required=True)
    billing.add_argument("--release-root", default=REPO_ROOT)
    billing.add_argument("files", nargs="+")
    table = sub.add_parser("table", help="cut the identical-case table of an architecture")
    table.add_argument("--store", required=True)
    table.add_argument("--release-root", default=REPO_ROOT)
    table.add_argument("--catalog", required=True,
                       help="the round's catalog: every key it names is accounted for")
    table.add_argument("--architecture", required=True)
    table.add_argument("--out-md", required=True)
    table.add_argument("--out-json", required=True)
    table.add_argument("--allow-unrecorded-environment", action="store_true")
    records = sub.add_parser("records", help="cut the dispatch-preference records, one "
                                             "per table per architecture")
    records.add_argument("--store", required=True)
    records.add_argument("--release-root", default=REPO_ROOT)
    records.add_argument("--out-root", required=True,
                         help="the package directory the records land in "
                              "(<table>_kernels/timing_<architecture>.json)")
    records.add_argument("--catalog", required=True,
                         help="the round's catalog: every fused key it names is accounted "
                              "for")
    records.add_argument("--base", default=HERE)
    records.add_argument("--allow-unrecorded-environment", action="store_true")
    report = sub.add_parser("report", help="the fused-product report of an architecture, "
                                           "from the selected fused rows")
    report.add_argument("--store", required=True)
    report.add_argument("--release-root", default=REPO_ROOT)
    report.add_argument("--architecture", required=True)
    report.add_argument("--catalog", help="name the catalog's fused keys of this "
                                          "architecture that no valid row reaches")
    report.add_argument("--out-dir", required=True)
    report.add_argument("--allow-unrecorded-environment", action="store_true")
    for selecting in (status, table, records, report):
        selecting.add_argument(
            "--environment-record", metavar="FILE",
            help="the round's reference environment build record (R27): a row that "
                 "ran in another environment is stale and owed a re-time")
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except RowRefused as refusal:
        say(f"REFUSED {refusal}")
        return 2


def _main(args: argparse.Namespace) -> int:
    held = load_store(args.store)
    if args.command == "bill":
        found = bill(held, args.files, args.release_root)
        total = sum(len(v) for v in found.values())
        for name in ("retime", "check", "untouched"):
            say(f"{name}: {len(found[name])} of {total} keys")
            for key in found[name]:
                say(f"  {key}")
        return 0
    selection = select(held, args.release_root, args.allow_unrecorded_environment,
                       environment=file_sha256(args.environment_record))
    catalog_file = getattr(args, "catalog", None)
    catalog = load_catalog(catalog_file) if catalog_file else None
    expected = (catalog_keys(catalog, skip_release_drive=True) if catalog else [])
    named = catalog_tables(catalog) if catalog else []
    if args.command == "status":
        for line in _status_lines(selection):
            say(line)
        absent = sorted(set(expected) - set(selection))
        say(f"catalog keys with no row in the store: {len(absent)} of {len(expected)}")
        for key in absent:
            say(f"  {key}: owed a re-time")
        if args.json:
            with open(args.json, "w") as handle:
                json.dump({k: {"owed": v["owed"], "assessments": v["assessments"]}
                           for k, v in selection.items()}, handle, indent=1,
                          sort_keys=True, default=sorted)
        return 0
    if args.command == "table":
        architecture = str(catalog["architecture"])
        if architecture != args.architecture:
            say(f"REFUSED: the catalog is for architecture {architecture}, not "
                f"{args.architecture}: it would account for none of this table's keys")
            return 2
        cut = cut_ladder_table(selection, args.architecture, expected)
        with open(args.out_json, "w") as handle:
            json.dump(cut, handle, indent=1, sort_keys=True)
        with open(args.out_md, "w") as handle:
            handle.write(render_table(cut))
        say(f"table {args.architecture}: {len(cut['rows'])} rows, "
            f"{len(cut['missing'])} keys without a valid row")
        return 0
    if args.command == "report":
        import build_fused_timing_report  # noqa: PLC0415

        staged = stage_fused_report(selection, args.architecture, args.out_dir)
        lacking = []
        for key in sorted(set(expected) | set(selection)):
            parsed = parse_key(key)
            if parsed["kind"] != "fused" or parsed["architecture"] != args.architecture:
                continue
            chosen = selection.get(key)
            if chosen is None or chosen["selected"] is None or not measured(
                    chosen["selected"]):
                lacking.append(key)
        reached = {parse_key(key)["table"] for key, chosen in selection.items()
                   if parse_key(key)["kind"] == "fused" and chosen["selected"] is not None
                   and parse_key(key)["architecture"] == args.architecture}
        unreached = [f"{table} on {architecture}: no valid row of this table"
                     for table, architecture in named
                     if architecture == args.architecture and table not in reached]
        # THE RELEASE'S DRIVE CASES are known only by running the bench's estimate, so
        # they are not listed key by key here: said so, never left out in silence.
        tables_spec = (((catalog.get("fused") or {}).get("tables") or {})
                       if catalog and str(catalog["architecture"]) == args.architecture
                       else {})
        noted = [f"{table} on {args.architecture}: its DRIVE cases are the release's "
                 "and are not listed here; the table's record names every released "
                 "(product, case) without a row under its missing block"
                 for table, spec in sorted(tables_spec.items())
                 if (spec or {}).get("drive") == "release"]
        staged["missing_keys"] = len(lacking)
        staged["tables_not_reached"] = len(unreached)
        with open(os.path.join(args.out_dir, "missing_keys.txt"), "w") as handle:
            handle.write("".join(f"{key}\n" for key in lacking + unreached + noted))
        say(json.dumps(staged, sort_keys=True))
        if not staged["drive_rows"]:
            say(f"no selected DRIVE row of {args.architecture}: no report")
            return 2
        return build_fused_timing_report.main([
            "--rows", staged["drive"], "--sweep", staged["sweep"],
            "--out", os.path.join(args.out_dir, "fused_timing.md")])
    results = cut_records(selection, args.out_root, base=args.base, expected=expected,
                          named=named)
    for result in results:
        say(json.dumps(result, sort_keys=True))
    return 2 if any("refused" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
