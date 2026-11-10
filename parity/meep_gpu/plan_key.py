"""The plan a run dispatched, in one canonical form, and the data its planner read.

Why this exists. A route leg or an L3 row is evidence about the kernels one plan
dispatched: which product held each sub-step slot, from which table, launched how many
times a step. Two things decide that plan: the code (the planner and the kernels, which a
run's import record names file by file) and data the planner reads but never imports
(the timing records the dispatch preference arbitrates by, and the ledgers that admit
each kernel). Pinning a whole timing record by checksum makes every route leg stale when
any record is re-cut, even when no plan a leg dispatched changes. Keying the evidence on
the plan instead lets a re-cut record re-certify only the combinations it moves: a run
whose import record still holds byte for byte and whose plan, asked again under the new
data, is the same plan, is the same evidence.

The form (:data:`FORM`, ``meep_gpu.plan_key/1``). :func:`describe` reads a dispatch
record (``driver.fast_path_report()``) or the summary the route gates keep of one
(``gate_dispatch_end_to_end._plan_summary``) and keeps exactly the facts the freeze
decides and the launches they made:

* ``selected_plan``: the canonical selected plan, schema ``meep_gpu.selected_plan/1``
  (``{"schema", "platform", "decision", "dispatched": [[slot, arm], ...]}``, slots in the
  driver's step order, arms as the record labels them, array slots not listed; the form
  the dispatch-preference work reports plan changes in, so a key and a plan-change
  report name one plan the same way, and :func:`selected_plan_sha256` is that form's
  digest);
* ``launches_per_step``: for every dispatched slot, its ``dispatches`` and
  ``plan_launches`` from ``launch_counters`` PER STEP (an integer, or ``"n/d"`` when the
  run's count does not divide by its steps).

Nothing about the case is in it (a key is always held beside its case or row label),
and nothing that describes the process or the evidence rather than the selection: the
environment, the subnormal block (it names which freeze installed the policy, which
depends on case order), families' gates, timestamps and hosts, refusal prose, wall
times. Two runs of one plan on one device therefore digest equal whatever their step
count, case order or host label, and a plan that moved one arm or one launch per step
does not.

The digest (:func:`digest`) is the sha256 of the description as compact JSON with sorted
keys. :func:`key` returns ``{"form", "digest", "plan", "selected_plan_sha256"}``, the
block every recorder writes.

The planner's data (:func:`plan_inputs`). Read in-process when the plan is recorded:

* ``timing_records``: every shipped timing record (``<table>_kernels/timing_*.json``, the
  set ``dispatch_preference.record_files`` returns, from which the portable prior is
  derived), each by the sha256 of its bytes;
* ``ledgers``: each kernel table's ``fingerprints.json``, as the sha256 of its entries
  with ``driver_dispatch`` removed, in canonical JSON. ``driver_dispatch`` is the entry a
  route campaign's own re-cut writes, so leaving it out keeps a campaign from staling
  itself; every other entry, any admission included, is in the digest.

A default-preference control row that did not record the default keeps its key under
:data:`UNRECORDED_KEY_FIELD`, so a plan the default did not make is never offered as one.
An L3 row also records :func:`toolchain` (the Python version and the versions of the
array and kernel modules it imported), which a plan pass must share with it.

Standard library only, and no import of the package: whatever imports this module gains
this file in its import record and nothing else.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: The name and version of the canonical form. A change to what :func:`describe` keeps is
#: a new form, and two keys of different forms are never compared.
FORM = "meep_gpu.plan_key/1"
#: The block a recorder writes, and the field a plan-input statement is written under.
KEY_FIELD = "plan_key"
INPUTS_FIELD = "plan_inputs"
#: Where a default-preference row that did NOT record the default keeps its plan's key:
#: beside the row for a reader, and never under :data:`KEY_FIELD`, so no collector of
#: plan keys (:func:`route_leg_keys`) offers a plan the default did not make.
UNRECORDED_KEY_FIELD = "plan_key_of_an_unrecorded_default"
#: Where an L3 row records the versions its plan ran under (:func:`toolchain`).
TOOLCHAIN_FIELD = "plan_toolchain"
#: The modules whose versions :func:`toolchain` reads, if the process imported them.
TOOLCHAIN_MODULES = ("numpy", "cupy", "triton", "torch")
#: The kernel tables whose ledgers and timing records the planner reads.
TABLES = ("triton", "cuda", "metal")
#: The ledger entry a route campaign's re-cut writes, which no plan input may include.
SELF_WRITTEN_LEDGER_ENTRY = "driver_dispatch"
#: The sub-legs of a route case row that carry a whole-run plan summary.
ROUTE_LEG_LABELS = ("fused", "unfused", "array")
#: The control rows of a route leg that carry a one-step plan key of their own.
ROUTE_CONTROL_KEYS = ("default-preference",)


def _repo_root() -> str:
    """The nearest ancestor of this file that holds both ``meep_gpu/`` and ``parity/``."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    while root != os.path.dirname(root):
        if (os.path.isdir(os.path.join(root, "meep_gpu"))
                and os.path.isdir(os.path.join(root, "parity", "meep_gpu"))):
            return root
        root = os.path.dirname(root)
    raise RuntimeError(f"no repository root above {here}")


def per_step(count: Any, steps: int) -> Any:
    """``count`` per step: an integer when it divides, else the reduced ``"n/d"``."""
    if count is None or isinstance(count, bool):
        return None
    try:
        value = int(count)
    except (TypeError, ValueError):
        return str(count)
    steps = int(steps)
    if steps <= 0:
        return f"{value}/{steps}"
    if value % steps == 0:
        return value // steps
    common = math.gcd(value, steps)
    return f"{value // common}/{steps // common}"


# The canonical selected plan (``meep_gpu.selected_plan/1``) and its digest have one
# implementation, selected_plan.py beside this module, so a key and a plan-change report
# name one plan the same way.
try:
    import selected_plan as _selected_plan  # parity/meep_gpu on the path: the gates and tools
except ImportError:  # imported as parity.meep_gpu.plan_key: the L3 row
    from parity.meep_gpu import selected_plan as _selected_plan

SLOT_ORDER = _selected_plan.SLOT_ORDER
PLATFORM_OF_TABLE = _selected_plan.PLATFORM_OF_TABLE
SELECTED_PLAN_SCHEMA = _selected_plan.SCHEMA
selected_plan = _selected_plan.selected_plan
selected_plan_sha256 = _selected_plan.selected_plan_sha256


def describe(report: Optional[Mapping[str, Any]], steps: int) -> Dict[str, Any]:
    """The canonical description of the plan ``report`` records, over ``steps`` steps."""
    report = report or {}
    plan = selected_plan(report)
    counters = report.get("launch_counters")
    counters = counters if isinstance(counters, Mapping) else {}
    launches: Dict[str, Any] = {}
    for slot, _arm in plan["dispatched"]:
        counter = counters.get(slot)
        counter = counter if isinstance(counter, Mapping) else {}
        launches[slot] = {"dispatches": per_step(counter.get("dispatches"), steps),
                          "plan_launches": per_step(counter.get("plan_launches"), steps)}
    return {"form": FORM, "selected_plan": plan, "launches_per_step": launches}


def digest(description: Mapping[str, Any]) -> str:
    """sha256 of ``description`` as compact JSON with sorted keys."""
    text = json.dumps(description, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key(report: Optional[Mapping[str, Any]], steps: int) -> Dict[str, Any]:
    """``{"form", "digest", "plan", "selected_plan_sha256"}`` for ``report`` over ``steps``."""
    plan = describe(report, steps)
    return {"form": FORM, "digest": digest(plan), "plan": plan,
            "selected_plan_sha256": selected_plan_sha256(report)}


def key_digest(block: Any) -> Optional[str]:
    """The digest of a recorded key block of this form, or None (no block, other form)."""
    if not isinstance(block, Mapping) or block.get("form") != FORM:
        return None
    value = block.get("digest")
    plan = block.get("plan")
    if not isinstance(value, str) or not isinstance(plan, Mapping):
        return None
    # A block whose digest does not reproduce from its own description is not a key.
    return value if digest(plan) == value else None


# ---------------------------------------------------------------------------------------
# Route legs
# ---------------------------------------------------------------------------------------

def route_case_keys(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan key of every sub-leg of one route case row that recorded a plan."""
    out: Dict[str, Any] = {}
    legs = row.get("legs") or {}
    steps = row.get("steps_taken") or {}
    for label in ROUTE_LEG_LABELS:
        leg = legs.get(label) if isinstance(legs, Mapping) else None
        if not isinstance(leg, Mapping) or not isinstance(leg.get("plan"), Mapping):
            continue
        taken = steps.get(label) if isinstance(steps, Mapping) else None
        if not isinstance(taken, int) or isinstance(taken, bool) or taken <= 0:
            continue
        out[label] = key(leg["plan"], taken)
    return out


def route_leg_keys(rows: Iterable[Mapping[str, Any]],
                   controls: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    """``{case: {sub-leg or control: digest}}`` over a route leg's rows and controls.

    A control appears only when it recorded a key of this form; a case appears only when
    something of it did. The gate writes this under ``plan_keys`` in ``gate.json``.
    """
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        case = row.get("case")
        if not isinstance(case, str):
            continue
        for label, block in route_case_keys(row).items():
            out.setdefault(case, {})[label] = block["digest"]
    for control in controls:
        name = control.get("control")
        case = control.get("case")
        if name not in ROUTE_CONTROL_KEYS or not isinstance(case, str):
            continue
        value = key_digest(control.get(KEY_FIELD))
        if value is not None:
            out.setdefault(case, {})[str(name)] = value
    return {case: dict(sorted(entry.items())) for case, entry in sorted(out.items())}


# ---------------------------------------------------------------------------------------
# The planner's data
# ---------------------------------------------------------------------------------------

def _sha256_bytes(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def ledger_view_sha256(path: str) -> str:
    """sha256 of a ledger's entries other than ``driver_dispatch``, in canonical JSON."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, dict):
        document = {k: v for k, v in document.items() if k != SELF_WRITTEN_LEDGER_ENTRY}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def timing_record_files(root: str) -> List[str]:
    """Every shipped timing record, repository-relative and sorted (the set the
    dispatch preference derives its portable prior from)."""
    found: List[str] = []
    for table in TABLES:
        folder = os.path.join(root, "meep_gpu", f"{table}_kernels")
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        found.extend(f"meep_gpu/{table}_kernels/{name}" for name in names
                     if name.startswith("timing_") and name.endswith(".json"))
    return sorted(found)


def plan_inputs(root: Optional[str] = None) -> Dict[str, Any]:
    """The data the planner reads, by digest: the timing records and the ledgers' views.

    An unreadable file is recorded as ``{"error": ...}`` in place of a digest, which never
    equals a digest, so a comparison over it cannot pass.
    """
    root = root or _repo_root()
    records: Dict[str, Any] = {}
    for name in timing_record_files(root):
        try:
            records[name] = _sha256_bytes(os.path.join(root, name))
        except OSError as exc:
            records[name] = {"error": repr(exc)}
    ledgers: Dict[str, Any] = {}
    for table in TABLES:
        name = f"meep_gpu/{table}_kernels/fingerprints.json"
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        try:
            ledgers[name] = ledger_view_sha256(path)
        except (OSError, ValueError) as exc:
            ledgers[name] = {"error": repr(exc)}
    return {"timing_records": records, "ledgers": ledgers}


def toolchain() -> Dict[str, Any]:
    """The Python version and the version of each of :data:`TOOLCHAIN_MODULES` this
    process has imported (None for one it has not): read from ``sys.modules``, never
    importing anything, so the reading changes neither the import record nor the run."""
    out: Dict[str, Any] = {"python": platform.python_version()}
    for name in TOOLCHAIN_MODULES:
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None) if module is not None else None
        out[name] = None if version is None else str(version)
    return out


def inputs_moved(recorded: Any, live: Mapping[str, Any]) -> Dict[str, List[str]]:
    """``{"timing_records": [...], "ledgers": [...]}``: the names whose digest differs,
    or that one side holds and the other does not. A recorded statement that is not a
    mapping moves everything."""
    out: Dict[str, List[str]] = {}
    for part in ("timing_records", "ledgers"):
        have = (recorded or {}).get(part) if isinstance(recorded, Mapping) else None
        want = live.get(part) or {}
        if not isinstance(have, Mapping):
            out[part] = sorted(set(want)) or [f"<no {part} statement>"]
            continue
        moved = sorted(name for name in set(have) | set(want)
                       if not isinstance(have.get(name), str)
                       or have.get(name) != want.get(name))
        if moved:
            out[part] = moved
    return out
