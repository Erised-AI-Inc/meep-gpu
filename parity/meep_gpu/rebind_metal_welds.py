"""Rebind ``metal_kernels/fingerprints.json`` to a fresh round of gate artifacts.

WHY THIS EXISTS. A weld entry names the exact bytes its gate executed. When a
shared source changes -- ``metal_kernels/launch.py`` is the usual one, pinned by
39 of the 44 entries -- every weld that pins it is owed a re-run, and the ledger
is owed the new digests. Re-running the gates is only half of that: until the
ledger is rebound it still points at the previous round's artifacts, and
``test_the_metal_welds_are_bound_to_the_live_sources`` correctly refuses the
tree. The re-gate script had only ever done the first half.

WHAT IT REFUSES TO DO. It does not invent a weld, widen one, or promote a gate
that did not pass. It rebinds an entry only when the fresh artifact for that
family says ``released`` and ``passed``; anything else is reported and skipped,
leaving the stale entry in place to keep failing. It also keeps each entry's
CURATED key set -- the handful of paths that entry chose to pin -- rather than
replacing it with the artifact's full 42-file import set, because which files a
weld binds to is a claim someone made, not a thing to regenerate.

    python rebind_metal_welds.py --stamp 2026-08-27_routing            # report
    python rebind_metal_welds.py --stamp 2026-08-27_routing --write    # apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

LEDGER = _API / "meep_gpu" / "metal_kernels" / "fingerprints.json"
RESULTS = _HERE / "results"


def _family_of(entry_key: str) -> str:
    """``metal_ade_update_p_device_gate`` -> ``ade_update_p``."""
    name = entry_key
    for prefix in ("metal_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("_device_gate", "_gate"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _artifact_for(family: str, stamp: str) -> Path | None:
    directory = RESULTS / f"metal_{family}_{stamp}"
    if not directory.is_dir():
        return None
    for candidate in ("gate.json", f"{family}.json", "whole_step.json"):
        if (directory / candidate).is_file():
            return directory / candidate
    found = sorted(directory.glob("*.json"))
    return found[0] if found else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True, help="e.g. 2026-08-27_routing")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    rebound, skipped, missing = [], [], []

    for key in sorted(ledger):
        entry = ledger[key]
        if not isinstance(entry, dict) or "source_sha256" not in entry:
            continue
        family = _family_of(key)
        artifact = _artifact_for(family, args.stamp)
        if artifact is None:
            missing.append((key, family))
            continue
        fresh = json.loads(artifact.read_text(encoding="utf-8"))
        # ``canonical_verdict`` is the ONE verdict every Metal gate emits, and it
        # carries ``read_from`` naming the field it was derived from -- a verdict
        # someone can read rather than a word someone typed. ``passed`` is present
        # on only 2 of the 43 artifacts, so keying on it silently skipped 41 welds.
        # ``canonical_verdict`` is the verdict most Metal gates emit, and it carries
        # ``read_from`` naming the field it came from. It is NOT universal: the
        # whole_step gate writes an empty one and puts its verdict in ``release``.
        # Accept either, refuse when neither is present, and record which was read
        # so the rebind stays auditable. (``passed`` is on only 2 of 43 artifacts,
        # so keying on it silently skipped 41 welds.)
        # A key present with a null value is NOT a verdict -- whole_step writes
        # canonical_verdict.released = null -- so fall through on null, not just
        # on absence.
        #
        # NO PRESENT VERDICT MAY SAY FALSE, 2026-08-30, and this is a DEFECT FIX rather
        # than a tightening for its own sake. The order above was "canonical_verdict
        # first, fall through on null" -- so an artifact whose GATE passed but whose
        # RELEASE was refused rebound anyway, because the first verdict consulted said
        # True and the second was never read. THE TWO ANSWER DIFFERENT QUESTIONS:
        # `canonical_verdict` is the gate's own word on whether its legs passed;
        # `release` is `metal_gate_runner`'s word on whether the bytes that ran were
        # the shipping bytes and the source manifest agreed. A weld binds BYTES, so a
        # refused release is disqualifying no matter how green the legs were.
        #
        # MEASURED, not hypothetical: this round edited two gate scripts and re-ran
        # them in place. `metal_gate_runner` refused to release both -- "campaign
        # source manifest disagrees" -- while both wrote canonical_verdict.released =
        # True, and the previous form of this loop reported all 44 rebound with 0
        # skipped, two of them against artifacts the release contract had rejected.
        verdicts = {name: (fresh.get(name) or {}) for name in
                    ("canonical_verdict", "release")}
        stated = {name: value.get("released") for name, value in verdicts.items()
                  if value.get("released") is not None}
        if not stated:
            skipped.append((key, f"{artifact.name} carries neither "
                                 "canonical_verdict.released nor release.released"))
            continue
        refused = sorted(name for name, released in stated.items() if not released)
        if refused:
            reasons = [f"{name}: {verdicts[name].get('reasons') or 'no reason given'}"
                       for name in refused]
            skipped.append((key, f"not released ({', '.join(refused)}): {reasons}"))
            continue
        source_field = "release" if "release" in stated else "canonical_verdict"
        verdict = verdicts[source_field]

        imported = fresh.get("imported_source_sha256") or {}
        curated = list(entry["source_sha256"])
        unknown = [name for name in curated if name not in imported]
        if unknown:
            skipped.append((key, f"artifact does not record {unknown}"))
            continue

        entry["source_sha256"] = {name: imported[name] for name in curated}
        entry["code_sha256"] = {
            name: code_digest_of_path(_API / name) for name in curated
        }
        entry["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        entry["records"] = (f"apps/api/parity/meep_gpu/results/{artifact.parent.name}/ - "
                            f"{fresh.get('records', '?')} records")
        entry["recorded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry["status"] = "PASS"
        entry["verdict_read_from"] = (
            f"{artifact.parent.name}/{artifact.name}:{source_field}.released"
            + (f" (read_from={verdict['read_from']})" if verdict.get("read_from") else ""))
        rebound.append(key)

    for key in rebound:
        print(f"  rebound  {key}", flush=True)
    for key, why in skipped:
        print(f"  SKIPPED  {key}: {why}", flush=True)
    for key, family in missing:
        print(f"  no artifact for {key} (looked for metal_{family}_{args.stamp}/)", flush=True)
    print(f"\n  {len(rebound)} rebound, {len(skipped)} skipped, {len(missing)} without an artifact",
          flush=True)

    if args.write and rebound:
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  wrote {LEDGER}", flush=True)
    elif not args.write:
        print("  (report only -- pass --write to apply)", flush=True)
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
