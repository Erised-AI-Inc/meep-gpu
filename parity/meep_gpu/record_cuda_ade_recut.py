"""Transcribe an ADE ``update_P`` RE-CUT into ``cuda_kernels/certification.json``.

WHY A SECOND RECORDER AND NOT ``record_cuda_regate.py``. That tool describes a
campaign whose subdirectories are FAMILIES, each joined to the block whose artifact
directory has that basename. An ADE re-cut has no family layer: ``run_cuda_ade_recut.sh``
writes ``<policy>/gate.json`` directly under the campaign, for one family, so the
regate tool refuses both directories by name -- correctly, because ``keep/`` and
``flush/`` are not families and matching them to a block by basename would be a
guess. The join here is DECLARED instead (``--re-cuts``), and then checked: the
declared block must already name this kernel module, these certified kernels and
these device-source digests, or the run is a new certification rather than a re-cut
and does not belong in a block that says otherwise.

WHAT A RE-CUT IS FOR. A dated block records what a release was cut against.
``test_ade_update_p.py`` then re-checks it against the tree, and a subject that moved
afterwards has to be DECLARED in ``post_certification_edits`` with the argument for
why it cannot reach the verdict. That mechanism is a substitute for a run, and it
gets weaker with every entry: the 2026-08-19 block carried five before
``meep_gpu/subnormal_policy.py`` grew the arm64 fenv lever, at which point the honest
answer was to run the gate again rather than write a sixth argument. This tool
records that run. It does NOT edit the dated block, whose value is precisely that it
still says what the original release was cut against.

WHAT IT REFUSES TO DO.

* IT NEVER INVENTS A VERDICT. Both policy legs must carry
  ``canonical_verdict.released`` true. One unreleased leg refuses the whole re-cut.
* IT NEVER RECORDS A RE-CUT THAT DID NOT RUN ON THE SHIPPING BYTES. The campaign's
  own ``SUBJECT_DIGESTS.txt`` is re-hashed against the working tree file by file, and
  the first disagreement is named and refuses. That check is the entire point of the
  block: without it, "re-cut on today's sources" is a sentence rather than a fact.
* IT NEVER CALLS A NEW CERTIFICATION A RE-CUT. The device-source digests both legs
  stamp must equal the ones the declared block already carries. Device code that
  moved means the kernels are not the certified ones, and that is a new block with
  new prose, not this.
* IT NEVER WRITES OVER A RECORD THAT ALREADY DISAGREES WITH ITS OWN DIRECTORIES, and
  never over an existing key.
* IT DERIVES EVERY FACT AND TYPES NONE. Times, host, device, library versions,
  policies, per-policy counts, gate name and digest, kernels, subject digests and the
  artifact manifest are all read out of the payloads and the tree.

USAGE

    python record_cuda_ade_recut.py --campaign results/cuda_ade_recut_2026-09-11T1829
    python record_cuda_ade_recut.py --campaign results/cuda_ade_recut_2026-09-11T1829 --write

EXIT CODE. Non-zero on any refusal, and a refusal is a statement that the run is not
describable as it stands -- the remedy is another run or a correction to the tree,
never an edit to the record to make this pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from record_cuda_regate import (  # noqa: E402
    _API, ARTIFACT_PREFIX, RECORD, RESULTS, _record, _stamps,
    record_disagrees_with_its_own_directories,
)
from rebind_cuda_welds import manifest_digest  # noqa: E402

#: The two policy legs a CUDA family is certified under, and the only two this
#: campaign writes. Spelled here rather than imported from ``POLICY_DIRECTORIES``
#: because that tuple is about a family subdirectory's INNER layout; here the policy
#: directory IS the campaign's top level, and a shared name would hide the difference.
POLICY_LEGS = ("keep", "flush")

#: The manifest the campaign writes before it compiles anything: ``sha256  path``
#: per subject, relative to ``the repository root``.
SUBJECT_MANIFEST = "SUBJECT_DIGESTS.txt"


def _read_subject_manifest(campaign: Path) -> dict:
    """``{repo-relative path: sha256}`` off the campaign's own manifest file."""
    path = campaign / SUBJECT_MANIFEST
    if not path.is_file():
        return {}
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            table[parts[1]] = parts[0]
    return table


def _subject_drift(subjects: dict) -> list[str]:
    """Which subjects the tree no longer agrees with. Empty is the whole claim."""
    drift = []
    for name, digest in sorted(subjects.items()):
        path = _API / name
        if not path.is_file():
            drift.append(f"{name}: not in this checkout")
            continue
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live != digest:
            drift.append(f"{name}: tree {live[:16]} vs run {digest[:16]}")
    return drift


def _declarations(pairs: list[str], subjects: dict,
                  device_module: str | None = None) -> tuple[dict, list[dict], list[str]]:
    """``--declare path=why`` -> revision digests and edit entries, plus refusals.

    THE ONE SUBJECT THAT CANNOT AVOID MOVING is this family's own test module: it is
    in the gate's subject list, and pointing it at a newly minted block is an edit to
    a subject of that block. There is no run that can precede itself, so this is where
    a DECLARATION is the honest instrument rather than a substitute for one -- and it
    is bounded: the tool computes the revision digest itself from the tree, refuses a
    path the run never read, and ``touches_device_code`` is False because the caller
    has already been required to show the device strings this run compiled are
    byte-identical to the ones the dated block certified.
    """
    revisions, entries, refusals = {}, [], []
    for pair in pairs:
        name, _, why = pair.partition("=")
        name, why = name.strip(), why.strip()
        if not why:
            refusals.append(f"--declare {pair!r} carries no reason after '='")
            continue
        if name not in subjects:
            refusals.append(f"--declare names {name!r}, which this run never read; "
                            f"only a subject of the run can be declared")
            continue
        if device_module and name == device_module:
            # THE ONE SUBJECT A DECLARATION CANNOT COVER, refused rather than asserted.
            # Every other subject is host-side: the gate's verdict is about the DEVICE
            # STRINGS, and the only file in the subject list that carries them is the
            # kernel module. An earlier revision wrote "The file carries no device
            # string" into every entry as a constant -- a sentence the tool never
            # established and which is flatly false for exactly this file.
            refusals.append(
                f"--declare names {name!r}, which is this block's kernel_module and "
                f"the one subject that CARRIES the device strings the verdict is "
                f"about. A prose declaration cannot cover it; re-run the gate")
            continue
        path = _API / name
        if not path.is_file():
            refusals.append(f"--declare names {name!r}, which is not in this checkout")
            continue
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live == subjects[name]:
            refusals.append(f"--declare names {name!r}, which has NOT moved since the "
                            f"run; a declaration with nothing to declare would read "
                            f"as drift that was tolerated")
            continue
        revisions[name] = live
        entries.append({"what": why, "touches_device_code": False,
                        "how_that_was_established": (
                            "record_cuda_ade_recut.py required both policy legs of "
                            "this run to stamp device-source digests byte-identical "
                            "to the ones the re-cut block certifies, recomputed this "
                            "file's digest from the tree rather than accepting one, "
                            "and REFUSES a declaration naming the kernel_module -- "
                            "the only subject that carries a device string. So "
                            "touches_device_code is a checked consequence here, not "
                            "a claim about this file's contents.")})
    return revisions, entries, refusals


def _leg_payload(campaign: Path, leg: str) -> dict | None:
    path = campaign / leg / "gate.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _released(payload: dict) -> tuple[bool, str]:
    verdict = payload.get("canonical_verdict")
    if isinstance(verdict, dict):
        if verdict.get("released") is True:
            return True, ""
        return False, f"canonical_verdict {json.dumps(verdict)[:160]}"
    if payload.get("passed") is True:
        return True, ""
    return False, "no canonical_verdict and passed is not true"


def _per_policy(payload: dict) -> dict:
    """The counts this leg measured, read off the payload's own summaries."""
    single = payload.get("bytes") or {}
    multi = payload.get("multistep") or {}
    mutations = payload.get("mutations") or {}
    stamp = payload.get("subnormal_policy_stamp")
    stamp = stamp if isinstance(stamp, dict) else {}
    guards = (single.get("summary") or {}).get("per_guard") or {}
    guarded = guards.get("fmad_false") or {}
    control = guards.get("default_no_options") or {}
    multi_guard = ((multi.get("summary") or {}).get("per_guard") or {}).get(
        "fmad_false") or {}
    return {
        "released": True,
        "started_utc": payload.get("started_utc"),
        "finished_utc": payload.get("finished_utc"),
        "policy_stamp": stamp.get("policy"),
        "single_launch": f"{guarded.get('identical')}/{guarded.get('ran')}",
        "multi_step": f"{multi_guard.get('identical')}/{multi_guard.get('ran')}",
        "guard_control_identical": f"{control.get('identical')}/{control.get('ran')}",
        "guard_control_diverged": bool(control.get("ran")) and
                                  control.get("identical") == 0,
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
        "every_leg_disarmed": mutations.get("every_leg_disarmed"),
        "cases_refused": {"single_launch": single.get("refused") or [],
                          "multi_step": multi.get("refused") or []},
    }


_WHAT_THIS_IS = (
    "A RE-CUT of the ADE update_P certification, not a new one. The gate named by the "
    "block this re-cuts was run again, on device, under both float32 subnormal "
    "policies, against the bytes in the tree TODAY. It adds no kernel, widens no "
    "predicate and closes none of the gaps that block lists; the device strings are "
    "byte-identical to the ones it recorded, which this tool refuses to proceed "
    "without. WHAT IT SAYS ABOUT SUBJECTS IS IN _subject_sha256_is AND NOWHERE ELSE: "
    "an earlier revision of this sentence asserted that EVERY subject hashes to what "
    "the checkout holds, which was the one fact the re-cut exists to state and was "
    "false for any block carrying a declared edit."
)

_WHAT_IT_DOES_NOT_CLAIM = [
    "IT DOES NOT REPLACE THE DATED BLOCK. That block records what the original "
    "release was cut against, including the post_certification_edits that carried "
    "it between runs, and it keeps its own artifact directory and its own digest. "
    "Read the coverage numbers, the fold accounting and the open gaps there.",
    "IT IS NOT A COVERAGE CLAIM. Same gate, same cases, same shapes, same mutation "
    "legs. The only new facts are the date and the bytes.",
    "IT IS NOT A WIRING CLAIM. Certification is a statement about bytes.",
    "IT TIMES NOTHING. The box is shared and this is a correctness re-cut.",
]


def build_block(campaign: Path, payloads: dict, subjects: dict,
                re_cuts: str, prior: dict, revisions: dict, edits: list) -> dict:
    stamps = _stamps(campaign)
    environments = [json.loads(text) for text in sorted(stamps["environment"])]
    environment = environments[0] if len(environments) == 1 else None
    started = sorted(stamps["started_utc"])
    gates = sorted({payload.get("gate") for payload in payloads.values()
                    if isinstance(payload.get("gate"), str)})
    gate_digests = sorted({payload.get("gate_sha256") for payload in payloads.values()
                           if isinstance(payload.get("gate_sha256"), str)})
    block = {
        "_what_this_is": _WHAT_THIS_IS,
        "_what_it_does_NOT_claim": _WHAT_IT_DOES_NOT_CLAIM,
        "track": "hand-CUDA",
        "re_cuts": re_cuts,
        "recorded_utc": started[0],
        "_recorded_utc_is": (
            f"the EARLIEST started_utc in the tree, by the convention at the head of "
            f"this record: when the first policy leg began. The last began "
            f"{started[-1]}."),
        "started_utc": started[0],
        "host": sorted(stamps["hostname"])[0] if len(stamps["hostname"]) == 1
                else sorted(stamps["hostname"]),
        "policies_cut_under": sorted(stamps["subnormal_policy"]),
        "artifacts": f"{ARTIFACT_PREFIX}{campaign.name} (gitignored)",
        "artifact_sha256": manifest_digest(campaign),
        "_artifact_sha256_is": (
            f"{len(list(campaign.rglob('*.json')))} verdict payloads across "
            f"{len(payloads)} policy legs, by the rule at the head of this record, "
            f"computed against the directory as it stands in this checkout."),
        "gate": gates[0] if len(gates) == 1 else gates,
        "gate_sha256": gate_digests[0] if len(gate_digests) == 1 else gate_digests,
        "kernel_module": prior.get("kernel_module"),
        # DERIVED FROM THE RUN, not copied from the block it re-cuts: the gate stamps
        # the names it actually compiled, and a re-cut that covered a different set is
        # exactly what this block must be able to say. main() refuses when the two
        # legs disagree or when the set is not the one the dated block certified.
        "certified_kernels": sorted(payloads[POLICY_LEGS[0]]["kernels"]),
        "multi_step_budget": sorted({payload.get("multi_step_budget")
                                     for payload in payloads.values()})[0],
        "seed": sorted({payload.get("seed") for payload in payloads.values()})[0],
        "resolution": sorted({payload.get("resolution")
                              for payload in payloads.values()})[0],
        "device_source_sha256": dict(sorted(
            payloads[POLICY_LEGS[0]]["device_source_sha256"].items())),
        "per_policy": {leg: _per_policy(payloads[leg]) for leg in POLICY_LEGS},
        "subject_sha256": dict(sorted(subjects.items())),
        "_subject_sha256_is": (
            f"the campaign's own {SUBJECT_MANIFEST}, written before anything compiled, "
            f"re-hashed against this checkout file by file by "
            f"record_cuda_ade_recut.py: {len(subjects) - len(revisions)} of "
            f"{len(subjects)} agree with the tree" +
            ("" if not revisions else
             f", and the remaining {len(revisions)} "
             f"({', '.join(sorted(revisions))}) moved AFTER the run and are declared "
             f"in revision_sha256 with a post_certification_edits entry each") +
            ". The tool refuses any drift that is not declared, so what this field "
            "states is the split, not an absence."),
        "revision_sha256": dict(sorted(revisions.items())),
        "post_certification_edits": edits,
        "passed": True,
    }
    if environment is not None:
        block["device"] = environment["device_name"]
        block["compute_capability"] = environment["compute_capability"]
        block["cupy_version"] = environment["cupy_version"]
        block["nvrtc_version"] = environment["nvrtc_version"]
        block["environment"] = environment
    else:
        block["_environments_disagree"] = environments
    return block


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True,
                        help="the re-cut directory, e.g. "
                             "results/cuda_ade_recut_2026-09-11T1829")
    parser.add_argument("--re-cuts", default="ade_2026-08-19",
                        help="the certification block this run re-cuts")
    parser.add_argument("--block", default="",
                        help="block key to write (default: the directory basename)")
    parser.add_argument("--declare", action="append", default=[],
                        metavar="PATH=WHY",
                        help="a subject edited AFTER the run, with the reason. "
                             "Repeatable. The digest is computed here, never given.")
    parser.add_argument("--write", action="store_true",
                        help="apply (default: report)")
    arguments = parser.parse_args(argv)

    campaign = Path(arguments.campaign)
    if not campaign.is_absolute():
        candidate = _HERE / campaign
        campaign = candidate if candidate.exists() else RESULTS / campaign.name
    if not campaign.is_dir():
        print(f"no such campaign directory: {campaign}", flush=True)
        return 2

    record = _record()
    disagreements = record_disagrees_with_its_own_directories(record)
    for line in disagreements:
        print(f"  RECORD DISAGREES WITH ITS OWN DIRECTORY  {line}", flush=True)
    if disagreements:
        print("\n  refusing to write into a record that already disagrees with its "
              "own artifacts", flush=True)
        return 1

    prior = record.get(arguments.re_cuts)
    if not isinstance(prior, dict):
        print(f"  REFUSED   the record has no block named {arguments.re_cuts!r}",
              flush=True)
        return 1

    refusals, payloads = [], {}
    for leg in POLICY_LEGS:
        payload = _leg_payload(campaign, leg)
        if payload is None:
            refusals.append(f"{leg}/gate.json is missing; a re-cut is a PAIR of "
                            f"policy legs and one of them says nothing about the other")
            continue
        released, why = _released(payload)
        if not released:
            refusals.append(f"{leg}/ is not released: {why}")
            continue
        payloads[leg] = payload

    subjects = _read_subject_manifest(campaign)
    revisions, edits = {}, []
    if not subjects:
        refusals.append(f"{campaign.name}/{SUBJECT_MANIFEST} is missing or empty, so "
                        f"nothing says which bytes this run read")
    else:
        revisions, edits, declaration_refusals = _declarations(
            arguments.declare, subjects, prior.get("kernel_module"))
        refusals.extend(declaration_refusals)
        for line in _subject_drift(subjects):
            if line.split(":")[0] in revisions:
                continue
            refusals.append(f"subject moved since the run -- {line}")

    if len(payloads) == len(POLICY_LEGS):
        kernel_sets = {leg: tuple(sorted(payload.get("kernels") or ()))
                       for leg, payload in sorted(payloads.items())}
        if len(set(kernel_sets.values())) != 1:
            refusals.append(f"the two legs compiled different kernel sets: "
                            f"{kernel_sets}")
        elif sorted(prior.get("certified_kernels") or ()) != \
                sorted(next(iter(kernel_sets.values()))):
            refusals.append(
                f"this run covered {sorted(next(iter(kernel_sets.values())))} while "
                f"{arguments.re_cuts} certifies "
                f"{sorted(prior.get('certified_kernels') or ())}. A different set is "
                f"a new certification, not a re-cut of this block.")
        recorded = prior.get("device_source_sha256") or {}
        for leg, payload in sorted(payloads.items()):
            for name, digest in sorted(
                    (payload.get("device_source_sha256") or {}).items()):
                if name in recorded and recorded[name] != digest:
                    refusals.append(
                        f"{leg}/ compiled a DIFFERENT {name}: {digest[:16]} against "
                        f"{recorded[name][:16]} in {arguments.re_cuts}. That is a new "
                        f"certification, not a re-cut of this block.")

    key = arguments.block or campaign.name
    if key in record:
        refusals.append(f"certification.json already has a block named {key!r}")

    for why in refusals:
        print(f"  REFUSED   {why}", flush=True)
    if refusals:
        print(f"\n  {len(refusals)} refusal(s) — nothing written", flush=True)
        return 1

    block = build_block(campaign, payloads, subjects, arguments.re_cuts, prior,
                        revisions, edits)
    print(f"\n  WOULD WRITE certification.json:{key}" if not arguments.write
          else f"\n  WRITING certification.json:{key}", flush=True)
    print(f"    re-cuts {block['re_cuts']}  on {block['host']}  {block['device']}",
          flush=True)
    print(f"    policies {block['policies_cut_under']}", flush=True)
    for leg in POLICY_LEGS:
        row = block["per_policy"][leg]
        print(f"    {leg:6s} single {row['single_launch']}  multi {row['multi_step']}"
              f"  control {row['guard_control_identical']}"
              f"  mutations {row['mutation_legs_as_required']}", flush=True)
    declared = len(block["revision_sha256"])
    print(f"    subjects {len(block['subject_sha256'])} — "
          f"{len(block['subject_sha256']) - declared} agree with the tree, "
          f"{declared} declared: {sorted(block['revision_sha256'])}", flush=True)
    print(f"    artifact_sha256 {block['artifact_sha256']}", flush=True)
    if not arguments.write:
        print("  (report only — pass --write to apply)", flush=True)
        return 0

    record[key] = block
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"  wrote {RECORD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
