"""The canonical form of a selected dispatch plan, and its digest.

A dispatch record (``driver.fast_path_report()``) says which kernel holds each sub-step
slot of a frozen configuration, among much else that varies from host to host
(refusal wording, device names, timings, paths). This module reduces a record to the
part that decides WHAT RUNS -- which arm holds which slot -- in one spelling, so two
runs, two hosts or two timing records can be asked "is this the same plan?" by
comparing a digest.

The canonical plan, schema ``meep_gpu.selected_plan/1``::

    {"schema":     "meep_gpu.selected_plan/1",
     "platform":   "nvidia" | "metal" | None,
     "decision":   the record's decision ("dispatched", "refused", ...) or None,
     "dispatched": [[slot, arm], ...]}

* ``dispatched`` lists every slot whose state is ``dispatched``, in the driver's step
  order (:data:`SLOT_ORDER`), then any other slot name in sorted order. Arms are the
  record's labels exactly: a hand-CUDA arm keeps its ``cuda:`` prefix, so on an NVIDIA
  plan the label alone says which table serves the slot. Array slots are the rest
  and are not listed; their refusal text is host wording, not plan.
* ``platform`` is ``metal`` for a plan composed on the Metal table and ``nvidia`` for
  one composed on the Triton or hand-CUDA tables; ``None`` when the record names no
  table (a configuration refused before a table was chosen).
* Nothing about the case (cells, run shape) is in the plan: a caller keys plans by
  case and carries those beside it.

The digest is the sha256 of ``json.dumps(plan, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)``.

The input may be a whole dispatch record (``slots`` per slot) or the plan summary the
route gates and the bench store (``gate_dispatch_end_to_end._plan_summary``:
``arms`` per dispatched slot, the table under ``composition``); both give the same
plan. Imports nothing from the package, so it reads records written by any tree.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "meep_gpu.selected_plan/1"

#: The driver's sub-step slots in step order (``fastpath.DRIVER_SLOTS``; the harness
#: test holds the two equal).
SLOT_ORDER = ("step_B", "fill_B", "update_H", "step_D", "fill_D", "update_E", "update_P")

#: The kernel tables of each platform.
PLATFORM_OF_TABLE = {"triton": "nvidia", "cuda": "nvidia", "metal": "metal"}


def _ordered(slots: Mapping[str, Any]) -> List[str]:
    known = [slot for slot in SLOT_ORDER if slot in slots]
    return known + sorted(str(slot) for slot in slots if slot not in SLOT_ORDER)


def selected_plan(report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The canonical plan of a dispatch record or of a stored plan summary."""
    report = report or {}
    table = report.get("table")
    if table is None:
        table = (report.get("composition") or {}).get("table")
    if "slots" in report and isinstance(report.get("slots"), Mapping):
        slots = report["slots"]
        arms = {slot: (entry or {}).get("arm") for slot, entry in slots.items()
                if (entry or {}).get("state") == "dispatched"}
    else:
        arms = dict(report.get("arms") or {})
    dispatched = [[slot, None if arms[slot] is None else str(arms[slot])]
                  for slot in _ordered(arms)]
    return {"schema": SCHEMA, "platform": PLATFORM_OF_TABLE.get(str(table)),
            "decision": report.get("decision"), "dispatched": dispatched}


def canonical_bytes(plan: Mapping[str, Any]) -> bytes:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def plan_sha256(plan: Mapping[str, Any]) -> str:
    """The digest of a canonical plan."""
    return hashlib.sha256(canonical_bytes(plan)).hexdigest()


def selected_plan_sha256(report: Optional[Mapping[str, Any]]) -> str:
    """The digest of a dispatch record's (or plan summary's) canonical plan."""
    return plan_sha256(selected_plan(report))
