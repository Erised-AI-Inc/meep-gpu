"""Create a Triton ledger entry for a family whose gate RELEASED and has none.

WHY THIS EXISTS, AND WHY IT IS NOT ``rebind_triton_welds.py``. That tool REFRESHES
the pins of an entry that already exists and refuses to invent a key -- correctly,
because inventing one is how a weld starts describing a run that never happened. But
that refusal left a real gap: 23 fused arms sit in
``fastpath.PENDING_DEVICE_GATE_ARMS`` with released gates and NO ledger entry, and
``the design notes (meep-gpu-dispatch-expansion-plan)`` name it as the phase's real cost -- "20
ledger entries have to be cut BY A GATE RUN. No in-tree tool creates one, and writing
one by hand is record forgery."

WHERE THE COUNT STANDS. 23 of the 29 pending rows were fused arms before the
2026-09-11 round; that round seeds three of them from the ``..._realarms_a`` fleet --
``triton_folded_dispersive_fused_pair_device_gate``,
``triton_conductive_fused_electric_pair_device_gate`` and
``triton_cylindrical_real_fused_electric_pair_device_gate`` -- leaving TWENTY pending
fused arms with no entry. The fourth arm released in that round, ``fused pair B
(cylindrical)``, is not seeded here at all: its entry
(``triton_cylindrical_real_fused_magnetic_pair_device_gate``) already exists and this
tool refuses it by name, because refreshing a standing pin is
``rebind_triton_welds.py``'s job and doing it here would walk one backwards.

THE DISTINCTION THIS TOOL RESTS ON. Deriving an entry from a RELEASED artifact is not
forgery; typing one is. Every field here comes out of the gate's own record: the pins
from ``imported_source_sha256``, the verdict from the artifact's own verdict field,
the host from its environment block, the digest from the artifact bytes. Nothing is
typed, and the refusals below are what keep it that way. It is the Triton twin of
``rebind_cuda_welds.py --seed``.

WHAT IT REFUSES.

* A gate whose artifact is not RELEASED. A pending arm is pending because nothing
  proved it; an entry cut from an unreleased run would say the opposite.
* A key that already exists. Refreshing is ``rebind_triton_welds.py``'s job, and
  doing it here would let a seed walk a standing pin backwards onto an older run.
* An artifact whose recorded digests no longer match the checkout. The entry would
  then pin bytes that neither ran nor ship.
* A curated set with no path under ``meep_gpu/``. A weld over harness scripts alone
  certifies nothing a user runs -- the same clause the rebind tool applies.
* An artifact that records no hostname, in its own environment block or in a
  ``device.json`` beside it. Until 2026-09-11 the machine was DEFAULTED to the string
  "the GPU host" -- the single typed claim in an otherwise derived entry, and typed
  about the very thing a later reader would go back to. The probes record it now; a
  gate that does not is refused by name rather than guessed at.
* A run whose Triton or CuPy version is outside the ones this ledger's live records
  stand on. A weld is a claim about generated PTX, so the compiler is part of it, and
  widening that declaration is a deliberate act rather than a side effect of a seed.

WHERE EACH FIELD LANDS. The pins and the ``status`` sit on the entry, because they
describe the BYTES. Everything that describes the RUN -- its artifact digest, records
line, timestamp, host, toolchain and policy -- goes into ``runs[<capability>]``, and
the capability is read off the run (``rebind_triton_welds.run_capability``) so a
seeded entry and a rebound one are the same shape and a second architecture is
additive rather than a rewrite.

Usage (from ``parity/meep_gpu``)::

    python seed_triton_welds.py --campaign results/<campaign>            # report
    python seed_triton_welds.py --campaign results/<campaign> --write    # apply
    python seed_triton_welds.py --campaign results/<campaign> --only a,b
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_API = _HERE.parent.parent
for _path in (str(_API), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rebind_triton_welds  # noqa: E402
from meep_gpu import fastpath                            # noqa: E402
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

LEDGER = _API / "meep_gpu" / "triton_kernels" / "fingerprints.json"

#: Shared package files a weld pins when the gate imported them. Kept as a list
#: rather than "everything imported" because a gate imports the whole package and a
#: weld that pinned all of it would go red on any unrelated edit.
SHARED = ("meep_gpu/driver.py", "meep_gpu/fields.py", "meep_gpu/stepping.py",
          "meep_gpu/grid.py", "meep_gpu/pml.py", "meep_gpu/subnormal_policy.py",
          "meep_gpu/triton_kernels/launch.py", "meep_gpu/triton_kernels/kernels.py",
          "meep_gpu/triton_kernels/coverage.py")


def ledger_key(directory: str) -> str:
    """``probe_triton_folded_fused_magnetic_pair`` -> the entry that names it."""
    stem = directory
    for prefix in ("probe_triton_", "gate_triton_", "triton_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return f"triton_{stem}_device_gate"


def released(fresh: dict) -> tuple[bool, str]:
    """The authoritative verdict, in the order this tree ranks the spellings."""
    for field in ("release", "canonical_verdict"):
        block = fresh.get(field)
        if isinstance(block, dict) and "released" in block:
            return bool(block["released"]), field
    return False, "<no verdict field>"


def curated_for(directory: str, imported: dict, fresh: dict | None = None) -> list[str]:
    """The pin set, DERIVED: the family module, the shared files it imported, the gate.

    Never the whole import map. A weld pins what it certifies plus the seam it runs
    through; pinning every import would make the entry go red on an unrelated edit,
    which is the failure mode that trains people to ignore drift.

    THE GATE SCRIPT IS READ OFF THE RECORD, NOT MATCHED BY NAME. The line below it
    keeps the name match, which finds a per-family probe like
    ``probe_triton_beta_fused_electric_pair.py``; what it cannot find is a gate named
    after the WELD GROUP rather than the product. Measured 2026-09-17:
    ``gate_triton_offdiag_stencil_welds.py`` certifies
    ``offdiag_fused_electric_pair`` and ``folded_offdiag_fused_electric_pair``, and
    contains neither stem, so both seeded entries pinned no script at all and
    ``test_every_triton_weld_pins_the_script_that_gated_it`` went red on them by name
    — 2 of 55. The artifact says which script ran in its own ``gate`` field, so that
    is what is pinned, and a heuristic is only the fallback.
    """
    stem = ledger_key(directory)[len("triton_"):-len("_device_gate")]
    keys = [n for n in (f"meep_gpu/triton_kernels/{stem}.py",) if n in imported]
    keys += [n for n in SHARED if n in imported]
    keys += [n for n in imported
             if n.startswith("parity/meep_gpu/")
             and (f"{stem}" in n)
             and n.endswith(".py")]
    named = (fresh or {}).get("gate")
    if isinstance(named, str) and named:
        script = f"parity/meep_gpu/gate_{named}.py"
        if script in imported:
            keys.append(script)
    return sorted(set(keys))


def policy_line(fresh: dict) -> str | None:
    """The policy this run was cut under, NAMED, in the shape the contract reads.

    ``test_every_triton_weld_names_the_policy_it_was_cut_under`` requires the string
    to NAME a policy rather than carry a blob, and a seeded entry that dumped the
    whole ``subnormal_policy`` object satisfied nothing: the resolved name is one key
    inside it. Reading `resolved` and reporting the stamp beside it is what the
    accepted entries do, so a seeded weld is indistinguishable from a rebound one.
    """
    block = fresh.get("subnormal_policy")
    if not isinstance(block, dict):
        # A WELD MUST NAME THE POLICY IT WAS CUT UNDER, and "?" is not a name: the
        # contract test greps this line for `keep` or `flush`, and a placeholder
        # would clear the presence check while failing the naming one -- a red the
        # seeding round could not explain. Refuse instead, by returning None.
        return None
    resolved = block.get("resolved") or block.get("requested")
    if not resolved:
        return None
    stamp = block.get("policy") or "?"
    # ABSENT AND EMPTY ARE DIFFERENT MEASUREMENTS. The keep policy is attainable only
    # with an `ftz_stripped` CuPy cache, so "carries no ftz token" is a claim about
    # the run; an artifact whose stamp recorded no cache directory at all supports
    # neither that claim nor its opposite, and says so.
    if "cache_dir" not in block:
        token = "the run recorded no CuPy cache directory"
    else:
        cache = str(block.get("cache_dir") or "")
        token = ("the CuPy cache directory carries ftz_stripped" if "ftz_stripped"
                 in cache else "the CuPy cache directory carries no ftz token")
    return (f"{resolved} (installed by the gate before its first device compile: "
            f"requested={block.get('requested', '?')} resolved={resolved}; stamp "
            f"{stamp}; {token})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True,
                        help="campaign directory, relative to parity/meep_gpu")
    parser.add_argument("--only", default="",
                        help="comma-separated family directories to consider")
    parser.add_argument("--supersede", default="",
                        help="comma-separated compute capabilities this round "
                             "DISCARDS. It is accepted so the seeder and "
                             "rebind_triton_welds.py take the same flag and a round "
                             "driver can pass it to both; on a seed it can only ever "
                             "be a no-op, because a key that already exists is "
                             "refused by name, so there is no other architecture's "
                             "record here to strand.")
    parser.add_argument("--write", action="store_true",
                        help="apply (default: report)")
    args = parser.parse_args(argv)
    supersede = tuple(s.strip() for s in args.supersede.split(",") if s.strip())

    campaign = (_HERE / args.campaign).resolve()
    if not campaign.is_dir():
        raise SystemExit(f"no such campaign directory: {campaign}")
    try:
        campaign.relative_to(_API)
    except ValueError:
        raise SystemExit(
            f"campaign {campaign} is outside {_API}: the entry's `records` line is "
            f"a repo-relative path and cannot be written for a tree the repo does "
            f"not contain") from None
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    record = json.loads(LEDGER.read_text(encoding="utf-8"))
    # THE TYPED DECLARATION IS GONE AND MAY NOT COME BACK: the capabilities a table
    # admits are now the intersection of the ones its cited welds have a LIVE record
    # for, so they follow from the records this tool writes. A typed key beside them
    # is a second answer to the same question.
    if "validated_compute_capabilities" in record:
        raise SystemExit(
            "the ledger still carries the typed validated_compute_capabilities key; "
            "run parity/meep_gpu/migrate_capability_records.py first")
    seeded, refused = {}, []

    for sub in sorted(p for p in campaign.iterdir() if p.is_dir()):
        if only and sub.name not in only:
            continue
        artifact = sub / "gate.json"
        if not artifact.is_file():
            refused.append((sub.name, "no gate.json")); continue
        fresh = json.loads(artifact.read_text(encoding="utf-8"))
        key = ledger_key(sub.name)
        if key in record:
            refused.append((sub.name, f"{key} already exists -- refreshing is "
                                      f"rebind_triton_welds.py's job")); continue
        ok, field = released(fresh)
        if not ok:
            refused.append((sub.name, f"not released ({field})")); continue
        imported = fresh.get("imported_source_sha256") or {}
        if not imported:
            refused.append((sub.name, "records no imported_source_sha256")); continue

        drifted = [n for n, want in imported.items()
                   if (_API / n).is_file()
                   and hashlib.sha256((_API / n).read_bytes()).hexdigest() != want]
        if drifted:
            refused.append((sub.name, f"{len(drifted)} of {len(imported)} recorded "
                                      f"digests no longer match the checkout, e.g. "
                                      f"{drifted[0]}")); continue

        curated = curated_for(sub.name, imported, fresh)
        if not any(n.startswith("meep_gpu/") for n in curated):
            refused.append((sub.name, "curated set pins no path under meep_gpu/"))
            continue

        # THE DEVICE IDENTITY IS REQUIRED, AND MOST GATES DO NOT RECORD IT.
        # `test_every_triton_weld_names_a_validated_triton_and_capability` holds a
        # weld to naming a validated Triton and a compute capability, and only the
        # `probe_triton_*` artifacts carry an `environment` block with `device` in
        # it -- the plain gate artifacts record `host` as versions alone. Measured
        # 2026-09-09 across this campaign: bfast, special_kz and
        # cylindrical_fused_magnetic_pair all lack it, and the 2026-09-25 fleet
        # writes `environment: null` for several families. A seeded weld that typed a
        # device would be exactly the forgery this tool exists to avoid, so the
        # identity is read from the run -- its own environment block, its
        # `device_info`, its provenance stamp, or a `device.json` the campaign left in
        # the family directory or at its root -- and a run that names none is refused
        # and reported as the gate-output gap it is. The capability the record is
        # FILED UNDER comes from the same read, because a per-capability record is
        # only worth keeping if its key is a measurement.
        capability, identity = rebind_triton_welds.run_capability(
            fresh, [sub, campaign])
        if capability is None:
            refused.append((sub.name, identity))
            continue
        # ONE SPELLING, and it is the rebind tool's. A seeded entry and a rebound one
        # must be indistinguishable, which two f-strings do not stay.
        policy = policy_line(fresh)
        if policy is None:
            refused.append((sub.name, "records no subnormal_policy the weld can name "
                                      "-- a weld that does not say `keep` or `flush` "
                                      "does not identify its own result"))
            continue
        # THE MACHINE IS MEASURED OR THE WELD IS NOT CUT. Until 2026-09-11 the host
        # line defaulted the hostname to the string "the GPU host" -- the one typed fact
        # in an otherwise derived entry, and typed about the very thing a reader
        # would go back to. It was defensible only while every gate ran on one box;
        # it is a forgery the moment one does not, and nothing in the entry would
        # say so. The gates record it now (parity/meep_gpu/triton_device_identity.py)
        # and a gate that does not is refused by name.
        host = rebind_triton_welds._host_line(identity)
        if host is None:
            refused.append((sub.name, "neither the artifact's own identity block nor "
                                      "a device.json beside it can compose a host "
                                      "line (hostname, device, compute_capability, "
                                      "triton and cupy must all be present)"))
            continue
        toolchain = rebind_triton_welds.toolchain_refusal(identity, record)
        if toolchain is not None:
            refused.append((sub.name, toolchain))
            continue
        # THE ENTRY IS THE BYTES; THE RECORD IS THE RUN. The pins and the status say
        # what this weld certifies and are what `fastpath.bound_digest` hashes;
        # everything that describes the run that earned it is filed under the
        # capability that run measured, so a later round on another architecture adds
        # a record beside this one instead of overwriting it.
        entry = {
            "_seeded": (
                "CREATED by parity/meep_gpu/seed_triton_welds.py from the run named "
                "in its own capability record, because this family's gate had "
                "RELEASED and the ledger carried no entry for it -- the gap dispatch "
                "expansion names as its real cost. WHAT IS DERIVED AND WHAT IS THE "
                "TOOL'S OWN STAMP, said plainly because `status` is what admits an "
                "entry into every weld test: `status` is read from the artifact's own "
                "release verdict (the run is refused unless it released, so PASS is a "
                "reading, not a default); `source_sha256` is read from the artifact, "
                "and so is every field of runs[<capability>] -- artifact_sha256, "
                "host, subnormal_policy, cupy_version, triton_version, records and "
                "verdict_read_from -- with the capability itself read off the run and "
                "never passed in; `code_sha256` is the CHECKOUT's docstring-stripped "
                "digest, licensed by the raw-digest guard above (every path the run "
                "recorded matched the checkout byte for byte) and disclosed in the "
                "record; `recorded_utc` is when this entry was cut, not when the gate "
                "ran."),
            # READ, NOT TYPED. `verdict` is the artifact's own release field, and the
            # candidate was refused above unless it was True -- so this records what
            # the run said rather than what the seeding round wanted it to say.
            "status": "PASS" if ok else "<unreleased>",
            "source_sha256": {n: imported[n] for n in curated},
            "code_sha256": {n: code_digest_of_path(_API / n) for n in curated},
        }
        run = {
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "host": host,
            # THE PATH A READER CAN ACTUALLY WALK. `campaign.name` is the leaf, so
            # the line read `parity/meep_gpu/triton_fleet_.../<sub>/` and
            # dropped the `results/` component the campaign really sits under -- a
            # records line pointing at a directory that does not exist. Relative to
            # the repository root it is the same key shape the digests are already recorded in.
            "records": (f"apps/api/{campaign.relative_to(_API).as_posix()}/"
                        f"{sub.name}/ - {field}.released"),
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subnormal_policy": policy,
            # ONE SPELLING EACH: the merge canonicalised them, and _host_line
            # returning a line is the proof both are present.
            "cupy_version": str(identity["cupy"]),
            "triton_version": str(identity["triton"]),
            "verdict_read_from": f"{sub.name}/gate.json:{field}.released",
            "_digests_taken_from_the_checkout": (
                "code_sha256 is computed from the checkout, not from the artifact: a "
                "gate records raw file digests, and the docstring-stripped AST digest "
                "this ledger pins is not among them. It is licensed by the drift "
                "guard this tool applies before seeding -- every path the run DID "
                "record matched the checkout byte for byte -- and is disclosed here "
                "for the same reason rebind_triton_welds.py discloses it."),
        }
        # ``bound_before=None`` because there is no earlier bound to compare against:
        # the key did not exist a moment ago, which is the one condition this tool
        # seeds under, so no other architecture's record can be stranded by the write.
        fastpath.bind_capability(entry, bound_before=None, capability=capability,
                                 run=run, supersede=supersede)
        seeded[key] = entry

    for name, why in refused:
        print(f"  REFUSED  {name}: {why}", flush=True)
    for key in sorted(seeded):
        print(f"  seed     {key}  runs"
              f"{sorted(seeded[key][fastpath.RUNS])} "
              f"({len(seeded[key]['source_sha256'])} pins)", flush=True)
    print(f"\n  {len(seeded)} seeded, {len(refused)} refused", flush=True)

    if not args.write:
        print("  (report only -- pass --write to apply)", flush=True)
        return 0
    if not seeded:
        return 0
    record.update(seeded)
    LEDGER.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"  wrote {LEDGER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
