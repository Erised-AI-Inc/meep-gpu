"""Mint the FIRST weld entry for a Metal family, from that family's gate artifact.

WHY THIS EXISTS, AND WHY IT IS NOT PART OF ``rebind_metal_welds.py``. That script
refuses, by design, to "invent a weld": it rebinds an entry that already exists and
keeps the CURATED path set someone chose for it. That refusal is correct and worth
keeping -- a rebinder that could also create entries would be one edit away from
minting a weld for a gate nobody read. But a NEW family has no entry to rebind, and
``test_every_metal_family_is_welded`` requires one, so the first entry has to come
from somewhere. It comes from here, and from the artifact rather than from a hand.

WHAT IT REFUSES TO DO:

* mint against an artifact that is not RELEASED and does not read PASS;
* mint over an entry that already exists, unless ``--replace`` is passed AND that
  entry pins EXACTLY the paths being written -- so a re-mint can correct this tool's
  own output but can never silently discard a curated path set someone chose;
* mint a path the artifact does not record, or whose recorded digest disagrees with
  the tree -- a weld must bind bytes the gate actually executed and that still ship;
* invent a device digest. ``device_identity.device_digests`` establishes it or the
  entry goes without one, which leaves the strict byte rule in force rather than
  substituting something weaker.

EVERY FIELD IS DERIVED. Nothing here is typed by a person except the ``--purpose``
line and the curated path list, and both are arguments rather than literals.

    python mint_metal_weld.py --family fused_electric_pair \\
        --stamp 2026-08-28_arity --purpose "..." [--write]
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
    """BY NAME, never by ``parents[N]``: a moved harness must not resolve a wrong root."""
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu.code_identity import code_digest_of_path  # noqa: E402
from meep_gpu.device_identity import device_digests  # noqa: E402

LEDGER = _API / "meep_gpu" / "metal_kernels" / "fingerprints.json"
RESULTS = _HERE / "results"


#: Campaign leg directories, in the order a weld should prefer them when a family's
#: run directory holds legs rather than one artifact.
#:
#: THE ORDER IS THE ARGUMENT. A multi-leg campaign has one leg whose licence is the
#: SHIPPED behaviour — the harness installs nothing and the ladder does its own work
#: — and the others are controls around it (a policy installed from outside, a
#: policy the table refuses, an expansion licence supplied). A weld binds the bytes
#: a gate executed, and the leg that executed them the way a user's process would is
#: the one to bind. Preferring any other would make the record's own verdict a
#: statement about a harness rather than about the product.
CAMPAIGN_LEGS = ("shipped", "harness_flush", "shipped_expansion_probe")


def _artifact_for(family: str, stamp: str) -> Path:
    directory = RESULTS / f"metal_{family}_{stamp}"
    if not directory.is_dir():
        # A DISPATCH CAMPAIGN IS NOT NAMED metal_<family>_<stamp>. The driver-route
        # campaign writes results/<family>_<stamp>/<leg>/gate.json — its gate is
        # deliberately outside the gate_metal_* glob and its --out is a leg
        # directory — and the family it welds under is already spelled
        # ``dispatch_metal_route``, so no second prefix is added. Both layouts are
        # looked for BY NAME, so a typo is a refusal naming every spelling tried
        # rather than a silent fall-through to whatever else is on disk.
        tried = [directory]
        for candidate in (RESULTS / f"{family}_{stamp}",
                          RESULTS / f"dispatch_{family}_{stamp}"):
            tried.append(candidate)
            if candidate.is_dir():
                return _leg_artifact(candidate)
        raise SystemExit("no artifact directory; tried "
                         + ", ".join(str(path) for path in tried))
    if not any((directory / name).is_file()
               for name in ("gate.json", f"{family}.json", "whole_step.json")):
        legs = [name for name in CAMPAIGN_LEGS if (directory / name).is_dir()]
        if legs:
            return _leg_artifact(directory)
    for candidate in ("gate.json", f"{family}.json", "whole_step.json"):
        if (directory / candidate).is_file():
            return directory / candidate
    raise SystemExit(f"{directory} holds no recognised gate artifact")


def _leg_artifact(campaign: Path) -> Path:
    """The one leg of a multi-leg campaign a weld may be minted from."""
    for leg in CAMPAIGN_LEGS:
        candidate = campaign / leg / "gate.json"
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"{campaign} holds no {'/'.join(CAMPAIGN_LEGS)} leg carrying gate.json; a "
        f"weld may only be minted from the leg that ran the shipped behaviour, and "
        f"--artifact is how a different one is named deliberately")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument(
        "--pins", nargs="*", default=None,
        help="repo-relative paths to bind. Defaults to the three every fused-pair "
             "weld binds: the family module, metal_kernels/launch.py, and the gate.")
    parser.add_argument(
        "--artifact", default=None, type=Path,
        help="the gate artifact to mint FROM, named explicitly. Overrides the "
             "layout search entirely, which is what a multi-leg campaign needs: "
             "the leg a weld binds is a judgement (the SHIPPED leg, where the "
             "harness installed nothing) and not something a directory walk should "
             "make on its own.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--replace", action="store_true",
        help="re-mint an entry this tool wrote (its pinned path set must match "
             "exactly); use after fixing a defect in this tool, never to retire a "
             "curated set")
    args = parser.parse_args(argv)

    if args.artifact is not None:
        artifact = args.artifact.resolve()
        if not artifact.is_file():
            raise SystemExit(f"--artifact {artifact} is not a file")
        try:
            artifact.relative_to(RESULTS)
        except ValueError:
            raise SystemExit(
                f"--artifact {artifact} is outside {RESULTS}; a weld records the "
                f"artifact's path relative to results/ and a path from anywhere "
                f"else would name a run this tree cannot show") from None
    else:
        artifact = _artifact_for(args.family, args.stamp)
    fresh = json.loads(artifact.read_text(encoding="utf-8"))

    # NO PRESENT VERDICT MAY SAY FALSE. The same defect fix `rebind_metal_welds.py`
    # carries, and for the same reason: `canonical_verdict` is the gate's word on its
    # legs and `release` is the runner's word on the BYTES, so consulting the first and
    # falling through only on null lets an artifact with a refused release mint a weld
    # on the strength of green legs. A weld binds bytes; a refused release disqualifies.
    verdicts = {name: (fresh.get(name) or {})
                for name in ("canonical_verdict", "release")}
    stated = {name: value.get("released") for name, value in verdicts.items()
              if value.get("released") is not None}
    if not stated:
        raise SystemExit(f"{artifact} carries neither canonical_verdict.released nor "
                         f"release.released; there is no verdict to bind")
    refused = sorted(name for name, released in stated.items() if not released)
    if refused:
        raise SystemExit(f"{artifact} is NOT released ({', '.join(refused)}): "
                         + "; ".join(str(verdicts[name].get('reasons')
                                         or 'no reason given') for name in refused))
    field = "release" if "release" in stated else "canonical_verdict"
    verdict = verdicts[field]

    key = f"metal_{args.family}_device_gate"
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    pins = args.pins or [
        f"meep_gpu/metal_kernels/{args.family}.py",
        "meep_gpu/metal_kernels/launch.py",
        f"parity/meep_gpu/gate_metal_{args.family}.py",
    ]
    if key in ledger:
        if not args.replace:
            raise SystemExit(
                f"{key} already exists; rebind it with rebind_metal_welds.py rather "
                f"than minting over its curated path set, or pass --replace to "
                f"re-mint an entry this tool wrote")
        held = sorted((ledger[key].get("source_sha256") or {}))
        if held != sorted(pins):
            raise SystemExit(
                f"{key} pins {held}, which is not the set being written "
                f"({sorted(pins)}); --replace may correct this tool's own output, "
                f"never retire a curated path set")
    imported = fresh.get("imported_source_sha256") or {}
    missing = [name for name in pins if name not in imported]
    if missing:
        raise SystemExit(f"the artifact does not record {missing}; a weld may only "
                         f"bind bytes the gate is recorded as having imported")
    drift = [name for name in pins
             if hashlib.sha256((_API / name).read_bytes()).hexdigest() != imported[name]]
    if drift:
        raise SystemExit(f"{drift} have moved since the gate ran; re-run the gate "
                         f"rather than minting a weld against bytes that no longer ship")

    entry = {
        "purpose": args.purpose,
        "status": "PASS",
        "source_sha256": {name: imported[name] for name in pins},
        "code_sha256": {name: code_digest_of_path(_API / name) for name in pins},
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        # THE PATH RELATIVE TO ``results``, NOT THE PARENT'S BARE NAME. A campaign
        # that separates its policies puts the artifact at
        # ``metal_<family>_<stamp>/flush/gate.json``, and the bare name is then
        # ``flush`` -- so the record said "results/flush/" and named no family at
        # all. Measured 2026-09-07 on the three H->D tail products, whose campaign
        # is the first Metal one to run both policies into one stamp.
        "records": (f"apps/api/parity/meep_gpu/results/"
                    f"{artifact.parent.relative_to(RESULTS)}/ - "
                    f"{fresh.get('records', '?')} records"),
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": (f"this machine: Apple MPS device, torch "
                 f"{fresh.get('torch_version', '?')}, "
                 f"{fresh.get('metal_frontend', '?')}"),
        # TWO SPELLINGS, because two generations of gate write it differently: the
        # older ones put the report under ``subnormal_policy`` and the 2026-09-07
        # tails gates under ``subnormal_policy_report``. Reading only the first
        # wrote a literal "?" into three ledger entries -- a weld whose policy line
        # says nothing, which is the one field a reader consults to know WHICH
        # arithmetic the entry binds.
        "subnormal_policy": (
            f"{fresh.get('subnormal_policy') or fresh.get('subnormal_policy_report', '?')}"
            f" - native and uncontrollable on MPS; the oracle flushes too"),
        "verdict_read_from": (
            f"{artifact.parent.relative_to(RESULTS)}/{artifact.name}:{field}.released"
            + (f" (read_from={verdict['read_from']})" if verdict.get("read_from") else "")),
        "_not_wired": ("Registered NOT WIRED. Records byte-correctness, not dispatch "
                       "eligibility."),
    }
    # ABSOLUTE, because `device_digests` dispatches on the path's DIRECTORY to decide
    # which of the three device-source spellings applies; a repo-relative path
    # resolves to kind "none" and silently drops the device pin. Measured: the first
    # mint of this family took the strict byte rule for exactly that reason.
    kind, digests = device_digests(_API / "meep_gpu" / "metal_kernels"
                                   / f"{args.family}.py")
    if digests:
        entry["device_sha256"] = {
            f"meep_gpu/metal_kernels/{args.family}.py": {"kind": kind,
                                                         "digests": digests}}
    else:
        print("  NOTE: no device digest could be established for this module; the "
              "entry keeps the strict byte rule", flush=True)

    print(json.dumps({key: entry}, indent=2, sort_keys=True), flush=True)
    if not args.write:
        print("\n  (report only -- pass --write to apply)", flush=True)
        return 0
    ledger[key] = entry
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"\n  wrote {key} into {LEDGER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
