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

ONE RUN RECORD PER GPU ARCHITECTURE. The digests stay beside the entry; the facts
about the RUN -- its artifact, records line, timestamp, host line, the environment
record it was read from, its torch, its Metal frontend and its subnormal policy -- go
into ``runs[<architecture>]`` (``meep_gpu.metal_runs``), and the architecture is READ
from the campaign's recorded environment (``metal_environment.campaign``), never
passed in. A campaign without both environment records is refused whole.

That makes a second Apple GPU ADDITIVE: a round on UNCHANGED bytes adds its
architecture's run beside the ones already there, and the admission certifies every
architecture all cited welds have a live run for. A round on bytes that MOVED cannot
be additive -- it would leave the other architectures' runs certifying bytes that no
longer ship -- so that entry is skipped and the other architectures are named;
``--supersede <architecture>[,...]`` is how a round says those certifications are
being dropped. An entry still in the one-run shape is refused: run
``migrate_metal_runs.py`` first.

    python rebind_metal_welds.py --stamp 2026-08-27_routing            # report
    python rebind_metal_welds.py --stamp 2026-08-27_routing --write    # apply
    python rebind_metal_welds.py --stamp <stamp> --supersede applegpu_g13s --write
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

from meep_gpu import fastpath  # noqa: E402
from meep_gpu import metal_runs  # noqa: E402
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

sys.path.insert(0, str(_HERE))
import metal_environment  # noqa: E402

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


#: The suffix every derived ``subnormal_policy`` line carries, as
#: ``mint_metal_weld.py`` writes it: the policy a Metal weld runs under is the table's.
POLICY_SUFFIX = metal_runs.POLICY_SUFFIX


def subnormal_policy(entry: dict, architecture: str, fresh: dict,
                     artifact_name: str) -> tuple:
    """``(policy, where it was read from)`` for this architecture's new run, or ``(None, why)``.

    THE LINE IS CARRIED, NOT RETYPED, in this order:

    1. this architecture's previous run of the gate: the curated sentences in the
       ledger are claims someone wrote about the gate, and a rebind of the same gate
       on the same GPU does not change them;
    2. the one line every other live run of the entry carries: the policy belongs to
       the table, MPS flushing float32 subnormals natively on every Apple GPU, so a
       second architecture's first run states what the first one's states, and the
       run says which architecture it was carried from;
    3. only for an entry no live run describes, the artifact's own statement, as
       ``mint_metal_weld.py`` writes it: its policy VALUE
       (``metal_runs.artifact_policy``), never the repr of a policy report.
    """
    runs = entry.get(metal_runs.RUNS) or {}
    previous = runs.get(architecture)
    if isinstance(previous, dict) and previous.get("subnormal_policy"):
        return (previous["subnormal_policy"],
                f"carried from {metal_runs.RUNS}[{architecture!r}], this "
                f"architecture's previous run of the gate")
    others = {name: run.get("subnormal_policy")
              for name, run in metal_runs.live_runs(entry).items()
              if name != architecture and run.get("subnormal_policy")}
    if len(set(others.values())) == 1:
        name = sorted(others)[0]
        return (others[name],
                f"carried from {metal_runs.RUNS}[{name!r}]: the policy is the table's "
                f"(MPS flushes float32 subnormals natively on every Apple GPU)")
    stated = metal_runs.artifact_policy(fresh)
    if stated:
        return f"{stated}{POLICY_SUFFIX}", f"derived from {artifact_name}"
    return None, (f"{artifact_name} states no subnormal policy and no other live run "
                  f"carries one line to take ({sorted(set(others.values()))})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True, help="e.g. 2026-08-27_routing")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    parser.add_argument(
        "--supersede", default="", metavar="ARCHITECTURE[,...]",
        help="GPU architectures whose runs this rebind retires. A rebind on bytes that "
             "MOVED leaves every other architecture's run certifying code that no "
             "longer ships; those architectures are named in the refusal and must be "
             "re-run on these bytes or listed here")
    args = parser.parse_args(argv)
    supersede = tuple(sorted({name.strip() for name in args.supersede.split(",")
                              if name.strip()}))
    for name in supersede:
        try:
            metal_runs.require_architecture(name)
        except metal_runs.ArchitectureRecordError as refused:
            raise SystemExit(f"--supersede: {refused}") from None

    environment = metal_environment.campaign(args.stamp)
    host = metal_environment.host_line(environment)
    # THE RUN IS KEYED AND DESCRIBED BY PARSING THE LINE IT RECORDS, so the line a
    # reader sees and the fields the admission compares cannot disagree.
    named = metal_runs.environment_of(host)
    try:
        architecture = metal_runs.require_architecture(named["architecture"])
    except metal_runs.ArchitectureRecordError as refused:
        raise SystemExit(f"the campaign's environment cannot key a run: {refused}") from None
    print(f"  campaign environment: {host}", flush=True)
    print(f"  writes runs[{architecture!r}]"
          + (f"; supersedes {list(supersede)}" if supersede else ""), flush=True)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    rebound, skipped, missing = [], [], []

    for key in metal_runs.weld_keys(ledger):
        entry = ledger[key]
        family = _family_of(key)
        artifact = _artifact_for(family, args.stamp)
        if artifact is None:
            missing.append((key, family))
            continue
        fresh = json.loads(artifact.read_text(encoding="utf-8"))
        # THE ARTIFACT'S OWN GPU, WHERE IT NAMES ONE, must be the campaign's: the run is
        # keyed by the environment records, and a gate artifact from another Mac's
        # results/ under this stamp would otherwise be filed under this GPU.
        ran_on = metal_runs.artifact_architecture(fresh)
        if ran_on is not None and ran_on != architecture:
            skipped.append((key, f"{artifact.name} records "
                                 f"environment.apple_gpu.architecture {ran_on!r} and the "
                                 f"campaign's environment records {architecture!r}"))
            continue
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

        # THE SHAPE IS REFUSED BY NAME, not read both ways: an entry still holding its
        # run beside the digests is migrated by migrate_metal_runs.py, never here.
        retired = metal_runs.shape_reasons(entry)
        if retired:
            skipped.append((key, f"not in the per-architecture shape "
                                 f"({'; '.join(retired)}) -- run migrate_metal_runs.py "
                                 f"first"))
            continue
        policy, policy_from = subnormal_policy(
            entry, architecture, fresh, f"{artifact.parent.name}/{artifact.name}")
        if policy is None:
            skipped.append((key, policy_from))
            continue

        # THE BOUND BEFORE THE REWRITE, read while the entry still holds the digests
        # its existing runs were cut against: it decides whether another
        # architecture's run survives this write.
        before = json.loads(json.dumps(entry))
        bound_before = fastpath.bound_digest(entry)
        entry["source_sha256"] = {name: imported[name] for name in curated}
        entry["code_sha256"] = {
            name: code_digest_of_path(_API / name) for name in curated
        }
        entry["status"] = "PASS"
        run = {
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "environment_read_from": f"metal_environment_{args.stamp}/start.json",
            "host": host,
            "metal_frontend": named["metal_frontend"],
            "records": (f"apps/api/parity/meep_gpu/results/{artifact.parent.name}/ - "
                        f"{fresh.get('records', '?')} records"),
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subnormal_policy": policy,
            "_subnormal_policy_read_from": policy_from,
            "torch": named["torch"],
            "verdict_read_from": (
                f"{artifact.parent.name}/{artifact.name}:{source_field}.released"
                + (f" (read_from={verdict['read_from']})"
                   if verdict.get("read_from") else "")),
        }
        try:
            staled = metal_runs.bind_architecture(
                entry, bound_before=bound_before, architecture=architecture, run=run,
                supersede=supersede)
        except metal_runs.ArchitectureRecordError as refused:
            # THE ENTRY IS PUT BACK: the refusal comes after the pins moved, and the
            # ledger object is what gets written if any other key binds.
            ledger[key] = before
            skipped.append((key, str(refused)))
            continue
        # NO OTHER ARCHITECTURE'S RUN MAY MOVE unless this round superseded it by
        # name -- the survival proof, in the tool rather than in a reviewer's head.
        # An explicit raise, not an ``assert``: ``python -O`` strips asserts, and this
        # is the check that keeps another Mac's certification intact.
        for name, record in (before.get(metal_runs.RUNS) or {}).items():
            if name in (architecture, *staled):
                continue
            if json.dumps(entry[metal_runs.RUNS].get(name), sort_keys=True) != \
                    json.dumps(record, sort_keys=True):
                raise SystemExit(f"{key}: the rebind of runs[{architecture!r}] moved "
                                 f"runs[{name!r}], which it did not supersede; nothing "
                                 f"written")
        rebound.append((key, staled))

    for key, staled in rebound:
        print(f"  rebound  {key}  runs[{architecture!r}]"
              + (f"  SUPERSEDED {list(staled)}" if staled else ""), flush=True)
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
