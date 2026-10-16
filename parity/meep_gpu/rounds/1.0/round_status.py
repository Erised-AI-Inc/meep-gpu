#!/usr/bin/env python3
"""What a round's artifacts say: one reader for every verdict shape the round meets.

WHY A ROUND-SIDE READER. The fleet driver's row reads ``canonical_verdict.released``
alone, which is ``None`` for three shapes that are not refusals:

* ``fused_offdiag_electric`` writes ``verdict.clauses`` and ``verdict["REFUSAL
  ESTABLISHED"]``: its job is to ESTABLISH a refusal, so ``True`` there is the gate's
  pass (state ``refusal_established``). A reader of ``released`` alone scores it "not
  released", which is how the 1.0 pre-certification passes first read it.
* the five composition legs write ``summary.status == "passed"``.
* the route gate's ``summary.json`` and the end-to-end licence's ``summary.json``.

Widening ``gate_provenance.read_verdict`` instead would move a file every gate imports,
and with it every weld's ``imported_source_sha256``; so the reading lives here, in the
round, and the gates' bytes do not move.

States: ``released`` · ``refusal_established`` · ``not_released`` · ``unreadable``
(a payload with no recognised verdict) · ``missing``. Only the first two are a pass.

usage:
  round_status.py artifact <file>                    one artifact, as JSON
  round_status.py root <fleet root> [<root> ...]     one row per gate, N of D
  round_status.py lanes <lanes home>                 every lane's steps.jsonl, N of D
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PASSING = ("released", "refusal_established")
E2E_PASSING = ("PASS-DISPATCHED", "PASS-FELL-BACK")


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_payload(payload: Any) -> Dict[str, Any]:
    """One payload's verdict, read in the order the round trusts the spellings."""
    if not isinstance(payload, dict):
        return {"state": "unreadable", "read_from": None, "reasons": ["not a JSON object"]}
    canonical = payload.get("canonical_verdict") or {}
    if isinstance(canonical, dict) and canonical.get("released") is not None:
        return {"state": "released" if canonical["released"] else "not_released",
                "read_from": "canonical_verdict.released",
                "reasons": list(canonical.get("reasons") or [])[:20]}
    release = payload.get("release")
    if isinstance(release, dict) and release.get("released") is not None:
        return {"state": "released" if release["released"] else "not_released",
                "read_from": "release.released",
                "reasons": list(release.get("reasons") or [])[:20]}
    verdict = payload.get("verdict")
    if isinstance(verdict, dict) and "REFUSAL ESTABLISHED" in verdict:
        clauses = verdict.get("clauses") or {}
        false = sorted(k for k, v in clauses.items() if v is not True) if isinstance(
            clauses, dict) else []
        established = verdict["REFUSAL ESTABLISHED"] is True and not false
        return {"state": "refusal_established" if established else "not_released",
                "read_from": "verdict.REFUSAL ESTABLISHED",
                "reasons": [f"clause false: {k}" for k in false]}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        status = summary.get("status")
        if isinstance(status, str):
            ok = status.lower() in ("passed", "pass")
            return {"state": "released" if ok else "not_released",
                    "read_from": "summary.status", "reasons": [] if ok else [status]}
        if isinstance(summary.get("passed"), bool):
            return {"state": "released" if summary["passed"] else "not_released",
                    "read_from": "summary.passed", "reasons": []}
    if isinstance(payload.get("verdicts"), dict):           # the end-to-end summary.json
        verdicts = payload["verdicts"]
        bad = {c: v for c, v in verdicts.items() if v not in E2E_PASSING}
        div = {c: v for c, v in (payload.get("first_divergent_checkpoint") or {}).items()
               if v is not None}
        ok = bool(verdicts) and not bad and not div
        return {"state": "released" if ok else "not_released", "read_from": "verdicts",
                "reasons": ([f"not passing: {bad}"] if bad else [])
                + ([f"divergent: {div}"] if div else [])}
    for key in ("passed", "released"):
        if isinstance(payload.get(key), bool):
            return {"state": "released" if payload[key] else "not_released",
                    "read_from": key, "reasons": []}
    if payload.get("status") in ("PASS", "passed"):
        return {"state": "released", "read_from": "status", "reasons": []}
    return {"state": "unreadable", "read_from": None,
            "reasons": [canonical.get("unreadable") or "no recognised verdict shape"]}


def classify(path: Path) -> Dict[str, Any]:
    """An artifact file, or a directory holding gate.json / summary.json."""
    path = Path(path)
    if path.is_dir():
        for name in ("gate.json", "summary.json", "whole_step.json"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            return {"state": "missing", "read_from": None, "reasons": [f"{path} holds no artifact"],
                    "artifact": str(path)}
    if not path.is_file():
        return {"state": "missing", "read_from": None, "reasons": [f"{path} does not exist"],
                "artifact": str(path)}
    out = classify_payload(_load(path))
    out["artifact"] = str(path)
    return out


def is_pass(state: str) -> bool:
    return state in PASSING


def fleet_root(root: Path) -> List[Dict[str, Any]]:
    """One row per gate the fleet driver ran into ``root``, read off the artifacts."""
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    seen = set()
    jsonl = root / "gates.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            gate = row.get("gate")
            if not gate:
                continue
            seen.add(gate)
            artifact = Path(row.get("artifact") or root / gate / "gate.json")
            verdict = classify(artifact)
            rows.append({"gate": gate, "root": root.name, "elapsed_s": row.get("elapsed_s"),
                         "timed_out": row.get("timed_out"), "exit_code": row.get("exit_code"),
                         **verdict})
    for directory in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else ():
        if directory.name in seen or directory.name in ("caches",):
            continue
        if (directory / "gate.json").is_file():
            rows.append({"gate": directory.name, "root": root.name, **classify(directory)})
    return rows


def remaining_gates(roots: Iterable[Path], gates: Iterable[str]) -> List[str]:
    """The gates of a fleet step that no root holds a passing artifact for (resume)."""
    passed = set()
    for root in roots:
        for row in fleet_root(Path(root)):
            if is_pass(row["state"]):
                passed.add(row["gate"])
    return [g for g in gates if g not in passed]


def _print_rows(rows: List[Dict[str, Any]]) -> int:
    good = [r for r in rows if is_pass(r["state"])]
    for row in rows:
        why = "; ".join(str(x) for x in (row.get("reasons") or [])[:2])[:200]
        print(f"  {row['state']:<20} {row.get('root', ''):<44} {row['gate']:<48} {why}")
    print(f"{len(good)} of {len(rows)} passing (released or refusal established)")
    return 0 if len(good) == len(rows) and rows else 1


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    what, rest = argv[0], argv[1:]
    if what == "artifact":
        print(json.dumps(classify(Path(rest[0])), indent=1))
        return 0 if is_pass(classify(Path(rest[0]))["state"]) else 1
    if what == "root":
        rows: List[Dict[str, Any]] = []
        for root in rest:
            rows += fleet_root(Path(root))
        return _print_rows(rows)
    if what == "lanes":
        home = Path(rest[0])
        total = good = 0
        for steps in sorted(home.glob("*/steps.jsonl")):
            last: Dict[str, Dict[str, Any]] = {}
            for line in steps.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                last[row["id"]] = row
            n = sum(1 for r in last.values() if is_pass(r.get("state", "")))
            print(f"{steps.parent.name}: {n} of {len(last)} steps passing")
            for row in last.values():
                if not is_pass(row.get("state", "")):
                    print(f"    {row['id']}: {row.get('state')} {str(row.get('reasons'))[:160]}")
            total += len(last)
            good += n
        print(f"all lanes: {good} of {total} steps passing")
        return 0 if total and good == total else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
