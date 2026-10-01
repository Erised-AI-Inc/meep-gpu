"""Move each weld's run evidence under the capability it ran on. ONE TIME, per ledger.

WHY THIS EXISTS. Both NVIDIA ledgers record one run per weld, with the run's facts
(``host``, ``records``, ``recorded_utc``, ...) beside the digests of the bytes it
certified. That shape can hold exactly one architecture: a rebind from a run on another
compute capability overwrites the evidence and leaves the label, which is how an entry
comes to say "RTX A6000, cc 8.6" over an artifact from another card. It also makes a
second architecture subtractive -- certifying 9.0 would un-certify 8.6 -- so the
validated list could never grow.

WHAT IT WRITES. Per entry, one ``runs`` record keyed by the capability that entry's run
reports, holding the run fields and the digest of the bytes they certified
(``fastpath.bound_digest``). The digests stay where they are. After this, a rebind on
UNCHANGED bytes adds a capability beside the ones already recorded, and a rebind on bytes
that MOVED refuses by name rather than stranding the others
(``fastpath.bind_capability``).

WHAT IT REFUSES TO DO. It invents nothing. The capability is READ -- from the Triton
entry's own ``host`` line, and from the CUDA campaign block's ``compute_capability`` --
and an entry whose capability cannot be read, or reads two values, refuses the whole run
with nothing written. It does not touch the Metal ledger, does not re-cut a digest, and
refuses a ledger it has already migrated.

    python parity/meep_gpu/migrate_capability_records.py            # report only
    python parity/meep_gpu/migrate_capability_records.py --write

Progress reporting: one flushed line per step and per refusal, and the verification
numbers before anything is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _find_api_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "meep_gpu").is_dir() and (parent / "parity").is_dir():
            return parent
    raise SystemExit("cannot find the tree root (a directory holding meep_gpu/ and parity/)")


_API = _find_api_root(Path(__file__).resolve())
sys.path.insert(0, str(_API))

from meep_gpu import fastpath                              # noqa: E402
from meep_gpu import fastpath_cuda                         # noqa: E402

#: The note each migrated ledger carries, and the key this tool refuses to run twice on.
NOTE_KEY = "_capability_runs_migration"

#: ``cc 8.6`` / ``compute capability 8.6`` inside a Triton weld's ``host`` line.
_HOST_CAPABILITY = re.compile(r"(?:cc|compute capability)\s+([0-9]+\.[0-9])")

#: The run fields of a table's ``driver_dispatch`` record, plus the subkeys of
#: ``released_fused_arms`` that describe the route run rather than the arms.
_RELEASED_RUN_SUBKEYS = ("gate", "artifact", "what_was_measured")


def say(message: str) -> None:
    print(message, flush=True)


def _load(path: Path) -> Tuple[Dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text), text


def _dump(document: Mapping[str, Any], sort_keys: bool) -> str:
    return json.dumps(document, indent=2, sort_keys=sort_keys, ensure_ascii=True) + "\n"


def _round_trips(document: Mapping[str, Any], text: str) -> Optional[bool]:
    """Which ``sort_keys`` reproduces this file byte for byte, or None if neither does."""
    for sort_keys in (False, True):
        if _dump(document, sort_keys) == text:
            return sort_keys
    return None


def _triton_capability(key: str, entry: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    host = entry.get("host")
    if not isinstance(host, str):
        return None, f"{key}: records no host line, so its capability cannot be read"
    found = {fastpath._normalized_capability(value)  # noqa: SLF001
             for value in _HOST_CAPABILITY.findall(host)}
    if len(found) != 1:
        return None, (f"{key}: its host line names {sorted(found)} compute "
                      f"capabilities, not exactly one: {host[:120]!r}")
    return found.pop(), ""


def _cuda_capability(key: str, entry: Mapping[str, Any],
                     blocks: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    """The capability of the campaign block this weld's own fields name."""
    names: List[str] = []
    campaign = entry.get("campaign")
    if isinstance(campaign, str) and "certification.json:" in campaign:
        names.append(campaign.split("certification.json:", 1)[1].split()[0].rstrip(":,."))
    notes = entry.get("_notes")
    if isinstance(notes, Mapping):
        cited = str(notes.get("_narrative_lives_in") or "")
        if ":" in cited:
            names.append(cited.rsplit(":", 1)[1])
    names.append(key)
    found = set()
    for name in names:
        block = blocks.get(name)
        if isinstance(block, Mapping) and block.get("compute_capability"):
            found.add(fastpath._normalized_capability(  # noqa: SLF001
                block["compute_capability"]))
    if len(found) != 1:
        return None, (f"{key}: the certification blocks it names ({names}) report "
                      f"{sorted(found)} compute capabilities, not exactly one")
    return found.pop(), ""


def _migrate_entry(key: str, entry: Dict[str, Any], capability: str,
                   run_fields: Tuple[str, ...]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``entry`` rewritten with its run under ``runs[capability]``, or a refusal."""
    moved = {field: entry[field] for field in run_fields if field in entry}
    for required in ("host", "records", "recorded_utc"):
        if required not in moved:
            return None, f"{key}: records no {required}, so it describes no run"
    rest = {name: value for name, value in entry.items() if name not in moved}
    if not any(field in rest for field in fastpath.BOUND_FIELDS):
        return None, (f"{key}: binds none of {list(fastpath.BOUND_FIELDS)}, so no run "
                      f"of it could be bound to the bytes it certified")
    try:
        fastpath.bind_capability(rest, bound_before=None, capability=capability,
                                 run=moved, run_fields=run_fields)
    except fastpath.CapabilityRecordError as refused:
        return None, f"{key}: {refused}"
    return rest, ""


def _migrate_dispatch(key: str, entry: Dict[str, Any],
                      stamp: str) -> Tuple[Dict[str, Any], List[str]]:
    """``driver_dispatch``: its one route run becomes dated history, ``runs`` empty.

    The route record is the one entry this tool cannot key by capability: neither
    table's record carries a readable device (the CUDA recut read a field no gate
    writes, and the Triton record states that no device run backs it), and the route
    legs live under the gitignored results tree. Both records are already stale against
    the shipped code -- their contract tests are listed as pending -- so the next route
    campaign writes ``runs[<capability>]`` from the run itself.
    """
    history: Dict[str, Any] = {}
    for field in fastpath.DISPATCH_RUN_FIELDS:
        if field == "released_fused_arms":
            continue
        if field in entry:
            history[field] = entry.pop(field)
    released = entry.get("released_fused_arms")
    if isinstance(released, Mapping):
        run_part = {name: released[name] for name in _RELEASED_RUN_SUBKEYS
                    if name in released}
        if run_part:
            history["released_fused_arms"] = run_part
            entry["released_fused_arms"] = {
                name: value for name, value in released.items()
                if name not in run_part} or None
            if entry["released_fused_arms"] is None:
                del entry["released_fused_arms"]
    for field in ("validated_compute_capabilities", "recorded_by"):
        if field in entry:
            history[field] = entry.pop(field)
    history["_why_this_is_history"] = (
        "The route run this record described was cut before the ledger kept one record "
        "per compute capability, and no device could be read back from it: this table's "
        "route legs live under the gitignored results tree. The next route campaign "
        f"writes {fastpath.RUNS}[<capability>] from the run it reads, under the "
        f"campaign {stamp}_cc<capability>. Kept because it is the provenance of the "
        "dispatch shape this record still states.")
    entry[fastpath.RUNS] = {}
    entry[f"_route_run_before_capability_records_{_today()}"] = history
    return entry, sorted(history)


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y_%m_%d")


def _owned_keys(ledger: Mapping[str, Any], table: str) -> List[str]:
    """Every weld entry this tool migrates, for one table."""
    if table == "cuda":
        return sorted(name for name, value in ledger.items()
                      if isinstance(value, Mapping) and not name.startswith("_")
                      and name != "driver_dispatch")
    sys.path.insert(0, str(_API / "parity" / "meep_gpu"))
    import rebind_triton_welds as rebind                     # noqa: PLC0415
    import recut_composition_records as composition          # noqa: PLC0415
    keys = set(rebind.CAMPAIGN_DIRS) | set(composition.LEGS)
    keys.add(rebind.BIT_IDENTITY_KEY if hasattr(rebind, "BIT_IDENTITY_KEY")
             else "bit_identity_gate")
    keys.add(fastpath.FAMILY_RECERT_GATE)
    return sorted(key for key in keys if isinstance(ledger.get(key), Mapping))


def _family_recert_bound(entry: Mapping[str, Any]) -> Tuple[Optional[Dict[str, str]], str]:
    """The nine families' recert maps, unioned, as that entry's bound set.

    Without one it pins nothing at entry level, so every capability's record would
    stay live through any edit -- the container would be decoration for the arms that
    cite it.
    """
    families = entry.get("families")
    if not isinstance(families, Mapping):
        return None, "records no families block"
    union: Dict[str, str] = {}
    for family, record in sorted(families.items()):
        digests = record.get("source_sha256_at_recert") if isinstance(record, Mapping) else None
        if not isinstance(digests, Mapping):
            continue
        for name, digest in digests.items():
            if name in union and union[name] != digest:
                return None, (f"{family} reports {name} at {digest[:12]} where another "
                              f"family reports {union[name][:12]}; the union is not "
                              f"single-valued and cannot be one bound set")
            union[name] = digest
    if not union:
        return None, "its families record no source_sha256_at_recert map"
    return union, ""


def migrate(table: str, path: Path, blocks: Mapping[str, Any],
            stamp: str) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    """``(new text, refusals, report)`` for one ledger ON DISK. Writes nothing."""
    ledger, _text = _load(path)
    return migrate_document(ledger, path, table=table, blocks=blocks, stamp=stamp,
                            keys=_owned_keys(ledger, table))


def migrate_document(ledger: Mapping[str, Any], path: Path, *, table: str,
                     blocks: Mapping[str, Any], stamp: str,
                     keys: Sequence[str]) -> Tuple[Optional[str], List[str],
                                                   Dict[str, Any]]:
    """The migration itself, over a loaded ledger and a stated key set.

    Split from :func:`migrate` so the refusals and the reversibility can be driven
    over ledgers built in a test rather than only over the two that ship.
    """
    text = path.read_text(encoding="utf-8")
    report: Dict[str, Any] = {"table": table, "path": str(path)}
    refusals: List[str] = []

    sort_keys = _round_trips(ledger, text)
    if sort_keys is None:
        return None, [f"{path.name}: does not round-trip through json.dumps(indent=2); "
                      f"this tool would reformat the whole file"], report
    report["sort_keys"] = sort_keys

    if NOTE_KEY in ledger or any(
            isinstance(value, Mapping) and fastpath.RUNS in value
            for value in ledger.values()):
        return None, [f"{path.name}: already migrated (it carries {NOTE_KEY!r} or a "
                      f"{fastpath.RUNS!r} record)"], report

    before = {
        "capabilities": (fastpath.validated_compute_capabilities() if table == "triton"
                         else fastpath_cuda.validated_compute_capabilities()),
    }
    keys = list(keys)
    report["entries"] = len(keys)
    migrated: Dict[str, Dict[str, Any]] = {}
    capabilities: Dict[str, str] = {}
    for key in keys:
        entry = json.loads(json.dumps(ledger[key]))
        if table == "triton":
            capability, why = _triton_capability(key, entry)
        else:
            capability, why = _cuda_capability(key, entry, blocks)
        if capability is None:
            refusals.append(why)
            continue
        if key == fastpath.FAMILY_RECERT_GATE:
            union, why = _family_recert_bound(entry)
            if union is None:
                refusals.append(f"{key}: {why}")
                continue
            entry["source_sha256_at_recert"] = union
        new_entry, why = _migrate_entry(key, entry, capability, fastpath.RUN_FIELDS)
        if new_entry is None:
            refusals.append(why)
            continue
        migrated[key] = new_entry
        capabilities[key] = capability

    if refusals:
        return None, refusals, report

    document = json.loads(json.dumps(ledger))
    for key, entry in migrated.items():
        document[key] = entry
    if isinstance(document.get("driver_dispatch"), Mapping):
        dispatch, moved = _migrate_dispatch(
            "driver_dispatch", json.loads(json.dumps(document["driver_dispatch"])), stamp)
        document["driver_dispatch"] = dispatch
        report["driver_dispatch_moved"] = moved
    for retired in ("validated_compute_capabilities",
                    "_validated_compute_capabilities_why"):
        if retired in document:
            report.setdefault("deleted", []).append(retired)
            del document[retired]

    report["capabilities"] = sorted(set(capabilities.values()))
    report["note"] = {
        "tool": "parity/meep_gpu/migrate_capability_records.py",
        "when": stamp,
        "what": (f"Each weld's run fields moved under {fastpath.RUNS}[<compute "
                 f"capability>] with the digest of the bytes that run certified "
                 f"(fastpath.bound_digest), so a second architecture is certified "
                 f"BESIDE this one instead of replacing it."),
        "capability_was_read_from": ("the entry's own host line" if table == "triton"
                                     else "the certification.json block the entry names"),
        "entries": len(migrated),
        "capabilities": sorted(set(capabilities.values())),
        "admitted_before": list(before["capabilities"] or ()),
    }
    document[NOTE_KEY] = report["note"]

    problems = _verify(ledger, document, migrated, capabilities, table)
    if problems:
        return None, problems, report
    return _dump(document, sort_keys), [], report


def _verify(old: Mapping[str, Any], new: Mapping[str, Any],
            migrated: Mapping[str, Any], capabilities: Mapping[str, str],
            table: str) -> List[str]:
    """Every check that has to hold before a byte is written."""
    problems: List[str] = []
    untouched = set(old) - set(migrated) - {"driver_dispatch",
                                            "validated_compute_capabilities",
                                            "_validated_compute_capabilities_why"}
    for key in sorted(untouched):
        if json.dumps(old[key], sort_keys=True) != json.dumps(new.get(key), sort_keys=True):
            problems.append(f"{key}: changed but was not selected for migration")

    for key, entry in migrated.items():
        capability = capabilities[key]
        run = entry[fastpath.RUNS][capability]
        moved = {name: value for name, value in run.items() if name != "bound_sha256"}
        lifted = ({"source_sha256_at_recert": entry["source_sha256_at_recert"]}
                  if key == fastpath.FAMILY_RECERT_GATE else {})
        rebuilt = {name: value for name, value in entry.items()
                   if name != fastpath.RUNS and name not in lifted}
        rebuilt.update(moved)
        original = {name: value for name, value in old[key].items()}
        if json.dumps(rebuilt, sort_keys=True) != json.dumps(original, sort_keys=True):
            lost = sorted(set(original) - set(rebuilt))
            added = sorted(set(rebuilt) - set(original))
            problems.append(f"{key}: not reversible (lost {lost}, added {added})")
        if fastpath.live_capabilities(entry) != (capability,):
            problems.append(f"{key}: its {capability} record is not live after the move")

    cited = {gate for _family, gate in
             (fastpath_cuda.CUDA_ARM_CERTIFICATION if table == "cuda"
              else fastpath.ARM_CERTIFICATION).values()}
    # Only the welds this run actually migrated can be asserted on: a ledger built by
    # a test cites none of them, and the shipped ones cite every key it owns.
    cited &= set(migrated)
    if cited:
        admitted = fastpath.table_capabilities(new, sorted(cited))
        expected = tuple(sorted({capabilities[key] for key in cited}))
        if admitted != expected:
            problems.append(f"the {table} table would admit {admitted}, not "
                            f"{expected}: the cited welds are not all migrated to "
                            f"the same capability")

    history = [value for key, value in new.items()
               if key.startswith("_route_run_before_capability_records")]
    for block in history:
        hexes = re.findall(r"\b[0-9a-f]{64}\b", json.dumps(block))
        if hexes:
            problems.append(f"the dispatch history block carries {len(hexes)} digests; "
                            f"a history block states no digest")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true",
                        help="apply; without it nothing is written")
    parser.add_argument("--tables", default="triton,cuda")
    arguments = parser.parse_args(argv)
    stamp = _today()

    blocks: Dict[str, Any] = {}
    certification = _API / "meep_gpu" / "cuda_kernels" / "certification.json"
    if certification.is_file():
        blocks = json.loads(certification.read_text(encoding="utf-8"))

    plans: List[Tuple[Path, str]] = []
    failed = False
    for table in arguments.tables.split(","):
        table = table.strip()
        if not table:
            continue
        path = _API / "meep_gpu" / f"{table}_kernels" / "fingerprints.json"
        say(f"[{table}] reading {path.relative_to(_API)}")
        text, refusals, report = migrate(table, path, blocks, stamp)
        if refusals:
            failed = True
            say(f"[{table}] REFUSED, nothing written ({len(refusals)}):")
            for refusal in refusals[:20]:
                say(f"    {refusal}")
            continue
        say(f"[{table}] {report['entries']} entries -> "
            f"{fastpath.RUNS}[{report['capabilities']}]; "
            f"sort_keys={report['sort_keys']}; "
            f"deleted {report.get('deleted', [])}; "
            f"driver_dispatch moved {len(report.get('driver_dispatch_moved', []))} fields")
        plans.append((path, text))

    if failed:
        say("REFUSED: no ledger was written")
        return 1
    if not arguments.write:
        say("report only: pass --write to apply")
        return 0
    for path, text in plans:
        temporary = path.with_suffix(".json.migrating")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        say(f"wrote {path.relative_to(_API)}")
    say("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
