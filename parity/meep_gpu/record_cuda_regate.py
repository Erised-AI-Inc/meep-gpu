"""Transcribe a CUDA RE-GATE campaign into ``cuda_kernels/certification.json``.

WHY THIS EXISTS, AND WHY IT IS A SECOND TOOL. ``rebind_cuda_welds.py`` binds
``cuda_kernels/fingerprints.json`` to the artifacts a campaign left. It finds
those artifacts by asking ``certification.json`` where the campaign wrote them,
which is the right dependence -- the ledger's own ``_relation_to_certification_json``
says certification.json is authoritative for WHAT WAS CERTIFIED and the ledger
only for WHICH BYTES -- and it is also, exactly, why the rebind tool could not
clear the drift the ledger carried on 2026-08-30. Every one of the twenty CUDA
gates had been RE-RUN on device and RELEASED, into
``results/cuda_regate_2026-08-30/<family>/``. No block in certification.json
named that tree. So the rebind tool could see forty stale pins and could not see
the run that cleared them: the blocker was a MISSING WRITER, not a missing
measurement.

THE REPAIR THAT WAS TRIED FIRST, AND WHY IT IS NOT THE ONE. Installing the fresh
legs INTO each block's own dated directory took the ledger to zero drift and
broke four tests, because ``certification.json`` blocks carry an
``artifact_sha256`` over THEIR OWN directory and
``test_certification_metadata.py::test_the_recorded_artifact_digest_recomputes_
over_the_directory`` recomputes it. A record that disagrees with its own
directory is worse than an honest drift, so that attempt was reverted and both
trees kept (``results/_pre_regate_2026-08-30/`` holds the originals verbatim).

THE REPAIR THIS IS. A re-gate is a RUN, and a run belongs in the record. This
tool writes ONE block describing the campaign -- where it ran, when, under which
policies, which certification block each of its family subdirectories re-gated,
and the manifest digest of the whole tree -- leaving every dated family block and
every dated directory untouched. ``rebind_cuda_welds.py --campaign`` then binds
against that block, and refuses to bind against a campaign the record does not
name. Nothing is installed over anything.

WHAT IT REFUSES TO DO.

* IT NEVER WRITES A BLOCK OVER A RECORD THAT ALREADY DISAGREES WITH ITS OWN
  DIRECTORIES. Before writing anything it recomputes ``artifact_sha256`` for
  every block whose directory is on this machine, by the record's own rule, and
  refuses the whole run on the first mismatch. That is the 2026-08-30 lesson
  encoded where it can bite rather than remembered.
* IT NEVER INVENTS A VERDICT. Every canonical policy leg under every family
  subdirectory must carry ``canonical_verdict.released`` true. One unreleased
  leg refuses the whole campaign: a re-gate that covered nineteen of twenty
  families is not the campaign this block would describe.
* IT NEVER GUESSES WHICH BLOCK A SUBDIRECTORY RE-GATED. The subdirectory name
  must equal the BASENAME of some block's own declared artifact directory, and
  exactly one block must claim it. A subdirectory no block claims is a refusal,
  not a row to drop -- that is a re-run of something this record does not
  describe, and binding a weld to it would be binding to an unnamed run.
* IT DERIVES EVERY FACT AND TYPES NONE. The timestamps, host, device,
  capability, library versions, policies, per-family leg lists, gate names and
  the manifest digest all come out of the payloads. The prose below is the
  narrative half, which is what certification.json is for; the numbers beside it
  are transcription and are checked against the artifact by
  ``test_certification_metadata.py``.

    python record_cuda_regate.py --campaign results/cuda_regate_2026-08-30
    python record_cuda_regate.py --campaign results/cuda_regate_2026-08-30 --write

EXIT CODE. Non-zero on any refusal. A refusal here is a statement that the
campaign is not describable as it stands, and the remedy is a device run or a
correction to the tree -- never an edit to the record to make this pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

RECORD = _API / "meep_gpu" / "cuda_kernels" / "certification.json"
RESULTS = _HERE / "results"

#: The prefix every block spells its artifact path with. Extracted rather than
#: listed, the same way ``rebind_cuda_welds.py`` and
#: ``test_certification_metadata.py`` do it, so the three agree by construction.
ARTIFACT_PREFIX = "parity/meep_gpu/results/"

#: The payload names a CUDA gate writes its verdict into, and the two policy
#: directories one has to sit under. Kept identical to
#: ``rebind_cuda_welds.CANONICAL_LEG_NAMES`` / ``POLICY_DIRECTORIES`` -- imported
#: rather than re-typed below, so a family that grows a third spelling cannot be
#: recognised by one tool and missed by the other.
from rebind_cuda_welds import (  # noqa: E402
    CANONICAL_LEG_NAMES, POLICY_DIRECTORIES, RE_GATE_KEY, artifact_directory,
    canonical_legs, leg_payloads, manifest_digest, read_verdict,
    record_disagrees_with_its_own_directories,
)


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _stamps(directory: Path) -> dict:
    """When, where, on what device, and under which policies -- from the payloads.

    The same reader ``test_certification_metadata._artifact_stamps`` uses, over
    the same two homes (top-level ``host`` and ``environment.hostname``), because
    a fact this tool writes is a fact that test then recomputes; two readers that
    disagree would mean a block that cannot be written and checked by the same
    rule.
    """
    started, hosts, policies, environments = set(), set(), set(), set()
    for path in sorted(directory.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("started_utc"), str):
            started.add(payload["started_utc"])
        if isinstance(payload.get("host"), str) and payload["host"]:
            hosts.add(payload["host"])
        environment = payload.get("environment")
        if isinstance(environment, dict):
            if environment.get("hostname"):
                hosts.add(environment["hostname"])
            environments.add(json.dumps({
                key: environment.get(key) for key in
                ("device_name", "compute_capability", "cupy_version",
                 "nvrtc_version", "numpy_version", "python",
                 "cuda_runtime_version", "driver_version")}, sort_keys=True))
        for key in ("subnormal_policy_install", "subnormal_policy_stamp"):
            value = payload.get(key)
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("policy"), str):
                policies.add(value["policy"])
            stamp = value.get("stamp")
            if isinstance(stamp, dict) and isinstance(stamp.get("policy"), str):
                policies.add(stamp["policy"])
    return {"started_utc": started, "hostname": hosts,
            "subnormal_policy": policies, "environment": environments}


def blocks_by_artifact_basename(record: dict) -> dict:
    """``{directory basename: [block names]}`` off each block's own artifact key.

    THE JOIN IS THE RECORD'S OWN CLAIM, not string surgery on a name. The Triton
    rebind tool types its family table out precisely because stripping a prefix
    "happens to work this round"; here nothing is stripped -- a campaign
    subdirectory is matched to the block that says its artifacts live in a
    directory of that name. A basename claimed by two blocks is reported rather
    than resolved, because picking one would be a guess.
    """
    found: dict[str, list[str]] = {}
    for name, value in record.items():
        if not isinstance(value, dict):
            continue
        directory = artifact_directory(value)
        if directory is not None:
            found.setdefault(directory.name, []).append(name)
    return found


def survey(campaign: Path, record: dict) -> tuple[list[dict], list[str]]:
    """One row per (block, subdirectory), and every refusal the campaign earns."""
    claimed = blocks_by_artifact_basename(record)
    rows, refusals = [], []
    subdirectories = sorted(p for p in campaign.iterdir() if p.is_dir())
    if not subdirectories:
        refusals.append(f"{campaign.name}/ holds no family subdirectory")
    for directory in subdirectories:
        name = directory.name
        owners = claimed.get(name, [])
        if not owners:
            refusals.append(
                f"{name}/ re-gates nothing this record names: no block declares "
                f"an artifact directory of that name. A weld bound to it would "
                f"be bound to a run the record does not describe.")
            continue
        # A SUBDIRECTORY MAY BE CLAIMED BY SEVERAL BLOCKS, and that is not an
        # ambiguity when each of them says which legs are its own. One gate script
        # taking ``--family`` writes ``<family>_<policy>`` legs so three families can
        # share a campaign directory and one environment stamp, and each block names
        # its ``gate_family`` for exactly that reason --
        # ``rebind_cuda_welds.canonical_legs`` has scoped by it since the layout
        # appeared. Refusing the shape outright made a re-gate of the complex fused
        # pairs unrecordable and left three welds with no route back to PASS but a
        # hand edit. The refusal now falls only where the join really is a guess.
        families = {owner: (record.get(owner) or {}).get("gate_family")
                    for owner in owners}
        if len(owners) > 1 and not all(families.values()):
            unscoped = sorted(owner for owner, family in families.items()
                              if not family)
            refusals.append(
                f"{name}/ is claimed by {owners} and {unscoped} name no "
                f"gate_family, so which legs belong to which block is a guess")
            continue
        if len(owners) > 1 and len(set(families.values())) != len(owners):
            # THE SCOPING ONLY DISAMBIGUATES IF THE SCOPES DIFFER. Checking that every
            # claimant NAMES a gate_family, without checking that the names are
            # distinct, lets two blocks stand on the same two legs -- one measurement
            # credited twice, and canonical_policy_legs reporting twice the legs the
            # tree holds.
            shared = sorted({f for f in families.values()
                             if list(families.values()).count(f) > 1})
            refusals.append(
                f"{name}/ is claimed by {owners} and more than one names the same "
                f"gate_family {shared}, so the scoping does not separate them and "
                f"both would bind to one measurement")
            continue
        for owner in owners:
            family = families[owner]
            legs = canonical_legs(directory, family)
            if not legs:
                scope = f" for gate_family {family!r}" if family else ""
                refusals.append(f"{name}/ holds no canonical policy leg{scope} "
                                f"({'/'.join(POLICY_DIRECTORIES)} × "
                                f"{'/'.join(CANONICAL_LEG_NAMES)})")
                continue
            released, why = read_verdict(directory, legs)
            if not released:
                refusals.append(f"{name}/ is not released for {owner}: {why}")
                continue
            payloads = leg_payloads(directory, legs)
            gates = sorted({payload["gate"] for payload in payloads
                            if isinstance(payload.get("gate"), str)})
            started = sorted({payload["started_utc"] for payload in payloads
                              if isinstance(payload.get("started_utc"), str)})
            rows.append({
                "block": owner,
                "directory": name,
                "gate_family": family,
                "legs": legs,
                "released": True,
                "gate": gates[0] if len(gates) == 1 else (gates or None),
                "first_leg_started_utc": started[0] if started else None,
            })
    return rows, refusals


#: The narrative half. Facts are derived; this is the reasoning a reader needs to
#: know what the block MEANS, which is the job certification.json exists for. It
#: lives here rather than in the file so a later campaign gets the same statement
#: and a change to it is a change to a reviewable line of code.
_WHAT_THIS_IS = (
    "A RE-GATE, not a new certification. Every family block this record already "
    "carries was re-run on device against the bytes in the tree TODAY, under both "
    "float32 subnormal policies, and every canonical policy leg released. It adds "
    "no kernel, widens no predicate and closes no open gap: each family's claim is "
    "the one its own dated block states, and this block says only that the claim "
    "was re-measured on the current sources. Re-running is what licenses re-binding "
    "cuda_kernels/fingerprints.json; nothing else does. Why THIS campaign was run is "
    "the block's own _why_it_was_run."
)

#: The 2026-08-30 wording, kept as the default so the tool's first caller reproduces
#: the block it wrote. A campaign with a different cause passes ``--why``: the reason
#: a re-gate happened is the one thing about it that is never derivable from the tree.
_DEFAULT_WHY = (
    "cuda_kernels/fingerprints.json welds each verdict to the sha256 of every file "
    "the gate imported, and 40 of its 79 pins no longer matched the tree -- the "
    "modules and gate scripts had moved under verdicts that were still being "
    "reported as current."
)

_WHAT_IT_DOES_NOT_CLAIM = [
    "IT DOES NOT REPLACE ANY DATED BLOCK, AND IT DOES NOT TOUCH ANY DATED "
    "DIRECTORY. Each family block keeps its own artifacts, its own "
    "artifact_sha256 and its own prose, and each still digests to what it "
    "records. The legs this campaign wrote live in their own tree. An earlier "
    "attempt to close the same drift installed the fresh legs INTO the dated "
    "directories: the weld ledger went to zero drift and four certification "
    "tests went red, because a block's artifact digest is computed over its own "
    "directory. A record that disagrees with its own directory is worse than an "
    "honest drift, so that attempt was reverted; results/_pre_regate_2026-08-30/ "
    "holds the originals verbatim and this tree holds the re-gate.",
    "IT IS NOT A COVERAGE CLAIM. The kernels, cases, shapes and mutation legs "
    "are exactly the ones each family block already describes. Read the numbers "
    "there; the only new fact here is the date and the bytes.",
    "IT IS NOT A WIRING CLAIM. The record's 'dispatch' key still reads "
    "wired=false and this block does not change it.",
    "IT TIMES NOTHING. The box is shared and this is a correctness re-gate.",
]

_HOW_TO_READ_RE_GATED = (
    "One row per (block, subdirectory) pair: which certification block it re-gated, "
    "which canonical policy legs it wrote, and the gate each leg stamps. The join is "
    "the record's own claim -- a subdirectory is matched to the block whose declared "
    "artifact directory has that basename -- never string surgery on a block name. "
    "Several blocks may claim one subdirectory when each names a gate_family: the "
    "legs are then scoped to <gate_family>_<policy>, the same rule "
    "rebind_cuda_welds.py reads, so a block about one family never stands on "
    "another family's measurement. A subdirectory no block claims, or one several "
    "claim without scoping, is a refusal in record_cuda_regate.py, not a row that "
    "gets dropped."
)


def build_block(campaign: Path, rows: list[dict], stamps: dict, why: str) -> dict:
    """The block, every fact of it derived from the tree just surveyed."""
    environments = [json.loads(text) for text in sorted(stamps["environment"])]
    environment = environments[0] if len(environments) == 1 else None
    started = sorted(stamps["started_utc"])
    block = {
        "_what_this_is": _WHAT_THIS_IS,
        "_why_it_was_run": why,
        "_what_it_does_NOT_claim": _WHAT_IT_DOES_NOT_CLAIM,
        "track": "hand-CUDA",
        "recorded_utc": started[0],
        "_recorded_utc_is": (
            f"the EARLIEST started_utc in the tree, by the convention at the head "
            f"of this record: when the campaign's first device leg began. The last "
            f"leg began {started[-1]}."),
        "started_utc": started[0],
        "host": sorted(stamps["hostname"])[0] if len(stamps["hostname"]) == 1
                else sorted(stamps["hostname"]),
        "policies_cut_under": sorted(stamps["subnormal_policy"]),
        "artifacts": f"{ARTIFACT_PREFIX}{campaign.name} (gitignored)",
        "artifact_sha256": manifest_digest(campaign),
        "_artifact_sha256_is": (
            f"{len(list(campaign.rglob('*.json')))} verdict payloads across "
            f"{len({row['directory'] for row in rows})} campaign subdirectorie(s) "
            f"claimed by {len(rows)} certification block(s), by the rule at the head "
            f"of this record, computed against the directory as it stands in this "
            f"checkout. The two counts differ whenever several blocks share one "
            f"subdirectory, each scoped to its own gate_family."),
        # NAMED FOR WHAT EACH COUNTS. ``families_re_gated`` used to be ``len(rows)``,
        # which counted rows; after the join was widened a row became a (block,
        # subdirectory) pair, so the field named after families was the one field
        # counting something else. All three are published because they genuinely
        # differ once several blocks share a directory.
        "families_re_gated": len({row.get("gate_family") or row["block"]
                                  for row in rows}),
        "campaign_subdirectories": len({row["directory"] for row in rows}),
        "blocks_re_gated": len(rows),
        "canonical_policy_legs": sum(len(row["legs"]) for row in rows),
        "every_leg_released": True,
        "_how_to_read_re_gated": _HOW_TO_READ_RE_GATED,
        RE_GATE_KEY: rows,
    }
    if environment is not None:
        block["device"] = environment["device_name"]
        block["compute_capability"] = environment["compute_capability"]
        block["cupy_version"] = environment["cupy_version"]
        block["nvrtc_version"] = environment["nvrtc_version"]
        block["environment"] = environment
    else:
        # FAILS CLOSED, and loudly. A campaign whose legs disagree about the
        # device cannot be described by one environment block, and writing the
        # first one would make the record say something the run does not.
        block["_environments_disagree"] = environments
    return block


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True,
                        help="the campaign directory (relative to parity/meep_gpu "
                             "or absolute)")
    parser.add_argument("--block", default="",
                        help="the certification.json key to write; defaults to the "
                             "campaign directory name")
    parser.add_argument("--why", default=_DEFAULT_WHY,
                        help="why THIS campaign was run. The one fact about a "
                             "re-gate that cannot be derived from the tree.")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)

    campaign = Path(args.campaign)
    if not campaign.is_absolute():
        campaign = (_HERE / campaign).resolve()
    if not campaign.is_dir():
        raise SystemExit(f"no such campaign directory: {campaign}")
    if _API not in campaign.parents:
        raise SystemExit(
            f"{campaign} lives outside {_API}; a block's artifacts key must name a "
            "parity/meep_gpu/results/... path or it points nowhere resolvable")

    record = _record()
    key = args.block or campaign.name

    disagreeing = record_disagrees_with_its_own_directories(record)
    for line in disagreeing:
        print(f"  RECORD DISAGREES WITH ITS OWN DIRECTORY  {line}", flush=True)
    if disagreeing:
        print("\n  refusing to record a run into a record that no longer describes "
              "the trees it already names. Find out which side moved; do not re-cut "
              "a digest on sight.", flush=True)
        return 1

    rows, refusals = survey(campaign, record)
    stamps = _stamps(campaign)
    for row in rows:
        print(f"  re-gated  {row['block']:42s} <- {row['directory']}/ "
              f"({len(row['legs'])} legs)", flush=True)
    for why in refusals:
        print(f"  REFUSED   {why}", flush=True)
    if refusals:
        print(f"\n  {len(rows)} families surveyed, {len(refusals)} refusals — "
              f"nothing written", flush=True)
        return 1

    missing = [name for name in ("started_utc", "hostname", "subnormal_policy")
               if not stamps[name]]
    if missing:
        print(f"  REFUSED   the campaign stamps no {missing}; a block that cannot "
              f"say when, where or under which policy it ran is not describable",
              flush=True)
        return 1

    block = build_block(campaign, rows, stamps, args.why)
    if "_environments_disagree" in block:
        print("  REFUSED   the legs disagree about the device/environment: "
              f"{block['_environments_disagree']}", flush=True)
        return 1

    existing = record.get(key)
    verb = "would update" if isinstance(existing, dict) else "would create"
    print(f"\n  {verb} certification.json:{key}", flush=True)
    print(f"    host {block['host']}, device {block.get('device')}, "
          f"policies {block['policies_cut_under']}", flush=True)
    print(f"    {block['families_re_gated']} families, "
          f"{block['canonical_policy_legs']} canonical policy legs, all released",
          flush=True)
    print(f"    recorded_utc {block['recorded_utc']}  "
          f"artifact_sha256 {block['artifact_sha256']}", flush=True)

    if not args.write:
        print("  (report only — pass --write to apply)", flush=True)
        return 0

    record[key] = block
    # ROUND-TRIPPED, not reformatted: json.dumps(indent=2, ensure_ascii=False)
    # reproduces this file byte for byte, so appending a key leaves every other
    # byte of the record exactly where it was.
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"  wrote {RECORD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
