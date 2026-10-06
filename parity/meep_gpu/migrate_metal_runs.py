"""Move each Metal weld's run under the GPU architecture it ran on. ONE TIME.

WHY THIS EXISTS. ``metal_kernels/fingerprints.json`` records one run per weld, with
the run's facts (``host``, ``records``, ``recorded_utc``, ``artifact_sha256``, ...)
beside the digests of the bytes it certified. That shape holds one Apple GPU
architecture: a second Mac's rebind could only replace the first Mac's certification.
This is the Metal half of ``migrate_capability_records.py``, keyed by the GPU
architecture (``applegpu_g13s``) instead of a compute capability.

WHAT IT WRITES. Per weld entry, one ``runs[<architecture>]`` record holding the run
fields (``meep_gpu.metal_runs.RUN_FIELDS``), the torch and Metal frontend the run's
``host`` line names, and the digest of the bytes the run certified
(``fastpath.bound_digest``). The digests stay where they are. The ``driver_dispatch``
record's route-run facts (``residency``, ``lift`` and the ``gate``, ``artifact`` and
``what_was_measured`` of ``exclusions.released_fused_arms``) move into a dated history
block with ``runs: {}``: this change moves ``metal_dispatch.py``, which that record
binds, so the next Metal route campaign writes ``runs[<architecture>]`` from the run
itself. A ``_architecture_runs_migration`` note records what was done.

WHAT IT REFUSES TO DO. It invents nothing. The architecture, torch and frontend are
READ from each entry's own ``host`` line; an entry whose line does not name all three,
or that records no run, refuses the whole ledger with nothing written. It refuses a
ledger it has already migrated and one it would reformat.

WHAT IT CHECKS BEFORE A BYTE IS WRITTEN, entry by entry: the move is reversible
(the entry and its run's fields are the entry it started as, and the route record with
its history block put back is the record it started as); every digest the ledger
held is still held, and the only digests added are one ``bound_sha256`` per run; each
``bound_sha256`` is ``fastpath.bound_digest`` of its entry and the run is live; the
run's ``host`` line is byte-equal to the one it replaces and its torch and frontend are
the ones that line names; every cited weld's live runs record the environment its host
line recorded, so the admission certifies what it certified before; the record walker
classifies every digest; and the history block carries no digest.

    python parity/meep_gpu/migrate_metal_runs.py            # report only
    python parity/meep_gpu/migrate_metal_runs.py --write

Progress reporting: one flushed line per step and per refusal, and the verification
numbers before anything is written.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _find_api_root(start: Path) -> Path:
    """BY NAME, never by ``parents[N]``: a moved harness must not resolve a wrong root."""
    for parent in [start, *start.parents]:
        if (parent / "meep_gpu").is_dir() and (parent / "parity").is_dir():
            return parent
    raise SystemExit("cannot find the tree root (a directory holding meep_gpu/ and parity/)")


_API = _find_api_root(Path(__file__).resolve())
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import fastpath                              # noqa: E402
from meep_gpu import metal_dispatch                        # noqa: E402
from meep_gpu import metal_runs                            # noqa: E402
from meep_gpu import weld_record_walk as walk              # noqa: E402

LEDGER = _API / "meep_gpu" / "metal_kernels" / "fingerprints.json"

#: The note the migrated ledger carries, and the key this tool refuses to run twice on.
NOTE_KEY = "_architecture_runs_migration"

#: The dated block inside ``driver_dispatch`` that keeps the route run cut before the
#: per-architecture records, as the NVIDIA migration keeps its own.
HISTORY_PREFIX = "_route_run_before_architecture_records_"

#: The subkeys of ``exclusions.released_fused_arms`` that describe the route RUN
#: (``recut_driver_dispatch_record.RELEASED_RUN_SUBKEYS``, spelled once there).
sys.path.insert(0, str(_API / "parity" / "meep_gpu"))
from recut_driver_dispatch_record import RELEASED_RUN_SUBKEYS  # noqa: E402

#: The fields a weld must record for it to describe a run at all.
REQUIRED_RUN_FIELDS = ("host", "records", "recorded_utc")

_HEX = re.compile(r"\b[0-9a-f]{64}\b")


def say(message: str) -> None:
    print(message, flush=True)


def _dump(document: Mapping[str, Any]) -> str:
    """The formatting every Metal ledger writer uses."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _digests(node: Any) -> List[str]:
    """Every 64-hex string in ``node``, values only, as the record walker finds them."""
    found: List[str] = []
    if isinstance(node, Mapping):
        for value in node.values():
            found.extend(_digests(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_digests(value))
    elif isinstance(node, str) and re.fullmatch(r"[0-9a-f]{64}", node):
        found.append(node)
    return found


def _migrate_weld(key: str, entry: Dict[str, Any]
                  ) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """``(entry rewritten, architecture, "")`` or ``(None, None, refusal)``."""
    host = entry.get("host")
    if not isinstance(host, str):
        return None, None, f"{key}: records no host line, so its run cannot be keyed"
    named = metal_runs.environment_of(host)
    unread = [fact for fact in metal_dispatch.ENVIRONMENT_FACTS if not named[fact]]
    if unread:
        return None, None, (f"{key}: its host line names no {unread}, so the run cannot "
                            f"be filed under one environment: {host!r}")
    architecture = named["architecture"]
    try:
        metal_runs.require_architecture(architecture)
    except metal_runs.ArchitectureRecordError as refused:
        return None, None, f"{key}: {refused}"
    moved = {field: entry[field] for field in metal_runs.RUN_FIELDS if field in entry}
    missing = [field for field in REQUIRED_RUN_FIELDS if field not in moved]
    if missing:
        return None, None, f"{key}: records no {missing}, so it describes no run"
    clash = [field for field in metal_runs.ENVIRONMENT_FIELDS if field in moved]
    if clash:
        return None, None, (f"{key}: already carries {clash} beside its digests; this "
                            f"tool reads them off the host line and will not choose")
    rest = {name: value for name, value in entry.items() if name not in moved}
    if not any(field in rest for field in fastpath.BOUND_FIELDS):
        return None, None, (f"{key}: binds none of {list(fastpath.BOUND_FIELDS)}, so no "
                            f"run of it could be bound to the bytes it certified")
    run = dict(moved)
    for field in metal_runs.ENVIRONMENT_FIELDS:
        run[field] = named[field]
    try:
        metal_runs.bind_architecture(rest, bound_before=None, architecture=architecture,
                                     run=run)
    except metal_runs.ArchitectureRecordError as refused:
        return None, None, f"{key}: {refused}"
    return rest, architecture, ""


def _migrate_dispatch(entry: Dict[str, Any], stamp: str) -> Tuple[Dict[str, Any], List[str]]:
    """``driver_dispatch``: its one route run becomes dated history, ``runs`` empty."""
    history: Dict[str, Any] = {}
    for field in metal_runs.DISPATCH_RUN_FIELDS:
        if field != "released_fused_arms" and field in entry:
            history[field] = entry.pop(field)
    released = (entry.get("exclusions") or {}).get("released_fused_arms")
    if isinstance(released, Mapping):
        run_part = {name: released[name] for name in RELEASED_RUN_SUBKEYS
                    if name in released}
        if run_part:
            history["released_fused_arms"] = run_part
            for name in run_part:
                del entry["exclusions"]["released_fused_arms"][name]
    history["_why_this_is_history"] = (
        "The route run this record described was cut before the Metal ledger kept one "
        "run per GPU architecture, and on bytes of metal_dispatch.py this record binds "
        "that the per-architecture admission edit moved. Kept as the provenance of the "
        "dispatch shape this record still states. The next Metal route campaign writes "
        f"{metal_runs.RUNS}[<architecture>] from the architecture its legs record, under "
        "the campaign metal_runs.route_campaign(metal_dispatch.METAL_DRIVER_ROUTE_GATE, "
        "<architecture>).")
    entry[metal_runs.RUNS] = {}
    entry[f"{HISTORY_PREFIX}{stamp}"] = history
    return entry, sorted(history)


def _unmigrate_dispatch(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """``driver_dispatch`` as it stood before :func:`_migrate_dispatch`: the reverse move.

    Used only to CHECK the move: the history block's fields return to the entry, the
    run's subkeys of ``released_fused_arms`` return to it, and the empty ``runs``
    container, the history key and its explanatory note are dropped.
    """
    rebuilt = json.loads(json.dumps(entry))
    history_keys = [key for key in rebuilt if key.startswith(HISTORY_PREFIX)]
    if rebuilt.get(metal_runs.RUNS) == {}:
        del rebuilt[metal_runs.RUNS]
    for key in history_keys:
        history = rebuilt.pop(key)
        for field, value in history.items():
            if field == "_why_this_is_history":
                continue
            if field == "released_fused_arms":
                released = rebuilt.setdefault("exclusions", {}).setdefault(
                    "released_fused_arms", {})
                released.update(value)
            else:
                rebuilt[field] = value
    return rebuilt


def migrate_document(ledger: Mapping[str, Any], text: str, *, stamp: str,
                     keys: Optional[Sequence[str]] = None,
                     cited: Optional[Sequence[str]] = None
                     ) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    """``(new text, refusals, report)``. Writes nothing.

    ``keys`` are the weld entries to move (default: every weld entry,
    ``metal_runs.weld_keys``) and ``cited`` the ones the admission reads (default:
    ``metal_dispatch.ARM_CERTIFICATION``'s); a test states both for a ledger it built.
    """
    report: Dict[str, Any] = {}
    if _dump(ledger) != text:
        return None, ["the ledger does not round-trip through json.dumps(indent=2, "
                      "sort_keys=True); this tool would reformat the whole file"], report
    if NOTE_KEY in ledger or any(
            isinstance(value, Mapping) and metal_runs.RUNS in value
            for value in ledger.values()):
        return None, [f"already migrated (it carries {NOTE_KEY!r} or a "
                      f"{metal_runs.RUNS!r} record)"], report

    keys = metal_runs.weld_keys(ledger) if keys is None else sorted(keys)
    cited = metal_runs.cited_keys() if cited is None else sorted(cited)
    report["entries"] = len(keys)
    refusals: List[str] = []
    migrated: Dict[str, Dict[str, Any]] = {}
    architectures: Dict[str, str] = {}
    for key in keys:
        entry = json.loads(json.dumps(ledger[key]))
        new_entry, architecture, why = _migrate_weld(key, entry)
        if new_entry is None:
            refusals.append(why)
            continue
        migrated[key] = new_entry
        architectures[key] = architecture
    if refusals:
        return None, refusals, report

    document = json.loads(json.dumps(ledger))
    for key, entry in migrated.items():
        document[key] = entry
    if isinstance(document.get("driver_dispatch"), Mapping):
        dispatch, moved = _migrate_dispatch(
            json.loads(json.dumps(document["driver_dispatch"])), stamp)
        document["driver_dispatch"] = dispatch
        report["driver_dispatch_moved"] = moved

    before = _cited_environments_before(ledger, cited)
    report["architectures"] = sorted(set(architectures.values()))
    report["admitted_before"] = before["uniform"]
    document[NOTE_KEY] = {
        "tool": "parity/meep_gpu/migrate_metal_runs.py",
        "when": stamp,
        "what": (f"Each Metal weld's run fields moved under {metal_runs.RUNS}[<GPU "
                 f"architecture>] with the torch and Metal frontend its host line names "
                 f"and the digest of the bytes that run certified "
                 f"(fastpath.bound_digest), so a second Apple GPU is certified BESIDE "
                 f"this one instead of replacing it."),
        "architecture_was_read_from": "the entry's own host line",
        "entries": len(migrated),
        "architectures": sorted(set(architectures.values())),
        "certified_before": before["uniform"],
    }

    problems, verified = _verify(ledger, document, migrated, architectures, cited, before)
    report.update(verified)
    if problems:
        return None, problems, report
    return _dump(document), [], report


def _cited_environments_before(ledger: Mapping[str, Any],
                               cited: Sequence[str]) -> Dict[str, Any]:
    """What each cited weld's ONE host line recorded: the rule the admission read before."""
    per_gate = {gate: metal_dispatch.host_environment(
        (ledger.get(gate) or {}).get("host") if isinstance(ledger.get(gate), Mapping)
        else None) for gate in cited}
    distinct = {tuple(env[fact] for fact in metal_dispatch.ENVIRONMENT_FACTS)
                for env in per_gate.values()}
    uniform = (dict(zip(metal_dispatch.ENVIRONMENT_FACTS, next(iter(distinct))))
               if len(distinct) == 1 else None)
    return {"per_gate": per_gate, "uniform": uniform}


def _verify(old: Mapping[str, Any], new: Mapping[str, Any],
            migrated: Mapping[str, Dict[str, Any]], architectures: Mapping[str, str],
            cited: Sequence[str], before: Mapping[str, Any]
            ) -> Tuple[List[str], Dict[str, Any]]:
    """Every check that has to hold before a byte is written, and the numbers it read."""
    problems: List[str] = []
    numbers: Dict[str, Any] = {}

    untouched = set(old) - set(migrated) - {"driver_dispatch"}
    for key in sorted(untouched):
        if json.dumps(old[key], sort_keys=True) != json.dumps(new.get(key), sort_keys=True):
            problems.append(f"{key}: changed but was not selected for migration")

    added: List[str] = []
    for key, entry in migrated.items():
        architecture = architectures[key]
        original = old[key]
        run = entry[metal_runs.RUNS][architecture]
        bound = run.get("bound_sha256")
        added.append(bound)
        moved = {name: value for name, value in run.items()
                 if name not in ("bound_sha256",) + metal_runs.ENVIRONMENT_FIELDS}
        rebuilt = {name: value for name, value in entry.items() if name != metal_runs.RUNS}
        rebuilt.update(moved)
        if json.dumps(rebuilt, sort_keys=True) != json.dumps(original, sort_keys=True):
            problems.append(f"{key}: not reversible (lost "
                            f"{sorted(set(original) - set(rebuilt))}, added "
                            f"{sorted(set(rebuilt) - set(original))})")
        if bound != fastpath.bound_digest(entry):
            problems.append(f"{key}: runs[{architecture!r}].bound_sha256 is not "
                            f"fastpath.bound_digest of the entry")
        if fastpath.live_capabilities(entry) != (architecture,):
            problems.append(f"{key}: its {architecture} run is not live after the move "
                            f"(live: {list(fastpath.live_capabilities(entry))})")
        if run.get("host") != original.get("host"):
            problems.append(f"{key}: the run's host line is not the line it replaces")
        named = metal_dispatch.host_environment(original.get("host"))
        for field in metal_runs.ENVIRONMENT_FIELDS:
            if run.get(field) != named[field]:
                problems.append(f"{key}: runs[{architecture!r}].{field} is "
                                f"{run.get(field)!r} and the host line names "
                                f"{named[field]!r}")
        reasons = metal_runs.shape_reasons(entry)
        if reasons:
            problems.append(f"{key}: not in the per-architecture shape: {reasons}")
    numbers["runs_written"] = len(added)

    # EVERY DIGEST KEPT, AND THE ONLY ONES ADDED ARE THE BOUNDS.
    old_digests = collections.Counter(_digests(old))
    new_digests = collections.Counter(_digests(new))
    lost = old_digests - new_digests
    gained = new_digests - old_digests
    numbers["digests_before"] = sum(old_digests.values())
    numbers["digests_after"] = sum(new_digests.values())
    if lost:
        problems.append(f"{sum(lost.values())} digests the ledger held are gone: "
                        f"{sorted(lost)[:4]}")
    if gained != collections.Counter(added):
        problems.append(f"the digests added are not exactly one bound_sha256 per run "
                        f"({sum(gained.values())} added, {len(added)} runs)")

    # THE ADMISSION CERTIFIES WHAT IT CERTIFIED BEFORE, weld by weld.
    after = dict(metal_dispatch.cited_environments(ledger=new, gates=cited))
    for gate in cited:
        wanted = before["per_gate"].get(gate)
        runs = after.get(gate)
        if runs is None:
            problems.append(f"{gate}: cited, and the migrated entry cannot be read as runs")
            continue
        if [dict(run) for run in runs] != [dict(wanted)]:
            problems.append(f"{gate}: its live runs record {list(runs)}; its host line "
                            f"recorded {wanted}")
    # AND THROUGH THE ADMISSION ITSELF, where the cited set is the table's: the verdicts
    # metal_dispatch.environment_block would write for a host in that environment.
    uniform = before["uniform"]
    if uniform is not None and cited and set(cited) == set(metal_runs.cited_keys()):
        verdicts = {fact: metal_dispatch.environment_verdict(
            fact, uniform[fact], uniform["architecture"], ledger=new)
            for fact in metal_dispatch.ENVIRONMENT_FACTS}
        numbers["admission_after"] = verdicts
        if not all(verdict is True for verdict in verdicts.values()):
            problems.append(f"the admission no longer certifies {uniform}: {verdicts}")
    numbers["recorded_environments_after"] = metal_dispatch.recorded_environments(
        ledger=new)

    # THE WALKER CLASSIFIES EVERY DIGEST.
    found = walk.census(new)
    unclassified = [walk.dotted(f.trail) for f in found if f.rule == walk.UNCLASSIFIED]
    numbers["walker_discovered"] = len(found)
    numbers["walker_run_bound_digests"] = sum(1 for f in found
                                              if f.rule == "run_bound_digest")
    if unclassified:
        problems.append(f"the record walker classifies {len(unclassified)} digests by no "
                        f"rule: {unclassified[:4]}")

    dispatch = new.get("driver_dispatch")
    if isinstance(dispatch, Mapping):
        for key, block in dispatch.items():
            if key.startswith(HISTORY_PREFIX) and _HEX.search(json.dumps(block)):
                problems.append("the dispatch history block carries a digest; a history "
                                "block states none")
        # THE ROUTE RECORD'S MOVE IS REVERSIBLE TOO, by the same rebuild the weld
        # entries get: put every history field back where it came from, drop the
        # container and the note, and the record must be the one this tool read.
        rebuilt = _unmigrate_dispatch(dispatch)
        if json.dumps(rebuilt, sort_keys=True) != json.dumps(
                old.get("driver_dispatch"), sort_keys=True):
            original = old.get("driver_dispatch") or {}
            problems.append(f"driver_dispatch: not reversible (lost "
                            f"{sorted(set(original) - set(rebuilt))}, added "
                            f"{sorted(set(rebuilt) - set(original))}, or a value moved)")
        for field in ("source_sha256", "code_sha256"):
            if json.dumps(dispatch.get(field), sort_keys=True) != json.dumps(
                    (old.get("driver_dispatch") or {}).get(field), sort_keys=True):
                problems.append(f"driver_dispatch.{field} moved")
    return problems, numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true",
                        help="apply; without it nothing is written")
    arguments = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d")

    say(f"reading {LEDGER.relative_to(_API)}")
    text = LEDGER.read_text(encoding="utf-8")
    new_text, refusals, report = migrate_document(json.loads(text), text, stamp=stamp)
    if refusals:
        say(f"REFUSED, nothing written ({len(refusals)}):")
        for refusal in refusals[:20]:
            say(f"    {refusal}")
        return 1
    say(f"{report['entries']} weld entries -> {metal_runs.RUNS}{report['architectures']}; "
        f"{report['runs_written']} runs written")
    say(f"digests: {report['digests_before']} before, {report['digests_after']} after "
        f"(every one kept; {report['runs_written']} bound_sha256 added)")
    say(f"walker: {report['walker_discovered']} discovered, 0 unclassified, "
        f"{report['walker_run_bound_digests']} run bounds")
    say(f"certified before: {report['admitted_before']}")
    say(f"admission after: {report.get('admission_after')}")
    say(f"recorded environments after: {report['recorded_environments_after']}")
    say(f"driver_dispatch: moved {report.get('driver_dispatch_moved', [])} into "
        f"{HISTORY_PREFIX}{stamp}; runs = {{}}")
    say(f"ledger bytes: {len(text.encode())} -> {len(new_text.encode())}")
    if not arguments.write:
        say("report only: pass --write to apply")
        return 0
    temporary = LEDGER.with_suffix(".json.migrating")
    temporary.write_text(new_text, encoding="utf-8")
    os.replace(temporary, LEDGER)
    say(f"wrote {LEDGER.relative_to(_API)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
