"""Bind ``cuda_kernels/fingerprints.json`` to the artifacts the CUDA gates left.

WHY THIS EXISTS, AND WHY IT EXISTS NOW. The CUDA track carries
``cuda_kernels/certification.json``, a hand-written narrative record: 22 dated
blocks, each with its own key set, each describing one device campaign in prose
and numbers. It is the richest of the three tracks' records and it is the only
one that pins the device SOURCE STRINGS. What it has never carried is a weld --
a mapping from a verdict to the sha256 of the tree files the gate imported --
so nothing at the merge bar could notice when those files moved. Measured the
day this tool was written: 39 of the 79 files the 21 released blocks imported no
longer match the bytes their gate ran, and not one clause anywhere said so.

That is the Triton lesson arriving late. The Triton ledger has the weld and 28
of its 30 passing entries went red from a single edit to a shared module, and
only 12 of those could be rebound because the rest point at runs whose bytes no
longer ship. The remedy on that track was a tool exactly like this one, written
after the fact, over a ledger that had already accumulated hand-shaped entries.
Here the tool and the ledger are being created together, so the ledger has no
history to retrofit and every entry in it was written by this code.

WHAT IT REFUSES TO DO.

* It never invents a verdict. An entry is written only when EVERY canonical
  policy leg under the block's artifact directory carries
  ``canonical_verdict.released`` true. A missing key is not a verdict, and a key
  present with ``null`` in it is not a verdict either -- the 2026-08-15 PML
  re-cut predates ``canonical_verdict`` entirely and is therefore skipped, which
  is the correct answer and not a bug to work around.
* It never promotes a leg. CUDA gate directories are full of JSON payloads that
  carry ``canonical_verdict`` and are SUPPOSED to read false (nine caught
  mutations under ``cuda_constitutive_recut_2026-08-22_rename/keep/``) or
  SUPPOSED to read true while proving nothing (the ``cn*`` null controls beside
  them). Picking "the first .json in the directory" would bind a weld to a
  mutation leg. The canonical legs are named -- see :data:`CANONICAL_LEG_NAMES`
  -- and once an entry exists its ``legs`` list is a curated claim this tool
  preserves rather than re-derives.
* It never widens a curated key set. ``source_sha256`` on an existing entry is
  rebound key-for-key; a key the fresh artifact does not record is a refusal for
  that entry, not a key to drop. Which files a weld pins is a claim someone
  made.
* It never backfills ``code_sha256`` over drift. ``code_identity.py`` licenses
  one clause and states its precondition in the same breath: a weld whose
  ``source_sha256`` no longer matches the file may still stand on an unchanged
  ``code_sha256``, and backfilling that digest "is only honest for a weld whose
  ``source_sha256`` still matches, because only then is the file on disk the
  bytes the gate actually ran" (``meep_gpu/code_identity.py``, the closing
  paragraph). So a file whose live bytes have moved gets NO ``code_sha256`` here
  and is listed in ``source_drift`` instead, and the entry's ``status`` says
  ``DRIFTED``. Writing a live code digest beside a stale source digest would
  manufacture exactly the licence the clause withholds.

WHAT IT WRITES, AND WHAT IT WILL NOT TOUCH. Two blocks per entry. The mechanical
block is every key without a leading underscore and this tool owns all of it.
``_notes`` is a nested object for bespoke narrative; the tool writes it once,
when it creates an entry, with a pointer to the certification.json block that
holds that entry's prose, and never reads or rewrites it again. Any other
underscore key is likewise left alone. That is the Metal ledger's discipline
with its underscore-prefixed siblings gathered into one block, so "did a tool
write this?" is answered by nesting rather than by remembering a naming rule.

BINDING A RE-GATE, added 2026-08-30 — the MISSING WRITER this tool used to be.
Everything above finds a block's artifacts by asking certification.json where
that block wrote them. That is the right dependence and it is also why this tool
could not clear the forty stale pins it had reported since the day it was
written: all twenty gates HAD been re-run on device and released, into
``results/cuda_regate_2026-08-30/<family>/``, and no block named that tree. The
blocker was a missing writer, not a missing measurement.

``--campaign <dir>`` binds against such a tree. Three refusals make it safe, and
each is one half of a failure that has already happened here:

* THE CAMPAIGN MUST BE RECORDED. certification.json must carry a block whose own
  ``artifacts`` key names that directory and which carries a
  :data:`RE_GATE_KEY` table; ``record_cuda_regate.py`` writes it, and a run that
  happened is what licenses writing it. A weld bound to an unrecorded run points
  at evidence the narrative record has no entry for.
* THAT BLOCK MUST AGREE WITH ITS DIRECTORY, and so must every other block whose
  directory is on this machine — see
  :func:`record_disagrees_with_its_own_directories`. The first attempt to clear
  this drift installed the fresh legs INTO the dated directories: zero drift, and
  four certification tests red, because a block's ``artifact_sha256`` is computed
  over its own tree. Nothing is installed over anything here; the dated
  directories and their digests are untouched.
* THE JOIN IS THE RECORD'S CLAIM. Which block each subdirectory re-gated comes
  out of the campaign block's table, never out of string surgery on a name, and
  the re-gate must have cut the SAME canonical legs the entry stands on.

    python rebind_cuda_welds.py                  # report against the ledger
    python rebind_cuda_welds.py --seed           # report, including new entries
    python rebind_cuda_welds.py --seed --write   # create the ledger / apply
    python rebind_cuda_welds.py --campaign results/cuda_regate_2026-08-30 --write

EXIT CODE. Non-zero whenever anything was skipped, the same as the Metal tool.
Blocks are skipped every run and will be until each is re-cut, so a non-zero exit
here is the standing statement that device runs are owed, not a transient failure
to retry. Reading the printed reasons is the point; making the exit code green by
narrowing what is enumerated would be the wrong repair.
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

KERNELS = _API / "meep_gpu" / "cuda_kernels"
LEDGER = KERNELS / "fingerprints.json"
RECORD = KERNELS / "certification.json"
RESULTS = _HERE / "results"

#: The prefix every certification.json block writes its artifact path with. Some
#: blocks spell the key ``artifact``, some ``artifacts``, and several append
#: " (gitignored)"; the path is extracted rather than listed, for the same reason
#: the blocks are enumerated rather than named.
ARTIFACT_PREFIX = "parity/meep_gpu/results/"

#: The payload each CUDA gate writes its own release verdict into. Measured over
#: every block's artifact tree: ``gate.json`` for eighteen families, ``cgate.json``
#: for the constitutive re-cut, ``probe.json`` for the bit-identity probe. Every
#: other JSON beside them is a mutation leg, a null control, a policy-binary
#: sidecar or a summary -- several of which carry a ``canonical_verdict`` of
#: their own that means the opposite of a release.
CANONICAL_LEG_NAMES = ("gate.json", "cgate.json", "probe.json")

#: A canonical leg also has to sit under a policy directory. ``cuda_ade_2026-08-19``
#: ships a third ``smoke/gate.json`` that reads released=false by design; the two
#: policy legs beside it are the verdict. Every device block in the record is cut
#: under both policies, so a leg outside these two directories is not one.
#:
#: MATCHED BY PREFIX SINCE 2026-09-11, and the reason is a real four-leg layout
#: rather than tidiness: ``cuda_three_slot_weld_2026-09-03`` writes
#: ``keep_pml``/``flush_pml``/``keep_no_pml``/``flush_no_pml`` — two products, two
#: policies each — and an exact-name match reads that directory as having NO
#: canonical leg at all, so the block cannot be seeded and the failure looks like a
#: missing verdict rather than a naming one. The prefix keeps the rule intact (a leg
#: must sit under a POLICY directory, so a ``smoke/`` leg is still not one) while
#: admitting a campaign that ran more than one product per policy.
POLICY_DIRECTORIES = ("keep", "flush")


def _is_policy_directory(name: str) -> bool:
    """Is ``name`` a policy leg directory? Prefix OR SUFFIX match.

    THE SUFFIX FORM IS WHAT A MULTI-FAMILY CAMPAIGN WRITES. One gate script that
    takes ``--family`` puts its legs under ``<family>_<policy>`` -- ``complex_keep``,
    ``complex_electric_flush`` -- so that three families can share one campaign
    directory and one environment stamp, and ``record_fused_complex_pairs.py`` reads
    exactly that layout. Matching only the prefix form meant the seeder found no
    canonical leg at all and skipped the block with "no canonical policy leg under
    the artifact directory", leaving a certification row that resolves to nothing.
    """
    return any(name == policy or name.startswith(f"{policy}_")
               or name.endswith(f"_{policy}")
               for policy in POLICY_DIRECTORIES)

#: Shared package files a weld pins beside its own module, when the gate recorded
#: importing them. ``compile_cache.py`` is the CUDA analogue of the Metal ledger's
#: ``launch.py``: the one seam every family's launch passes through, and therefore
#: the one whose edit owes every family a re-run.
SHARED_PACKAGE_FILES = ("meep_gpu/cuda_kernels/compile_cache.py",)

#: Harness files a weld pins beside its gate script. The probe is the comparator
#: itself for the bit-identity block and the mutation driver for the rest.
SHARED_HARNESS_FILES = ("parity/meep_gpu/probe_fused_kernel_bit_identity.py",)

#: The key that makes a certification.json block a RE-GATE record: a table of
#: which block each of the campaign's family subdirectories re-ran. Written by
#: ``record_cuda_regate.py``, read here. A block carrying it is not itself a
#: family owing a weld -- its verdict IS the twenty family verdicts it re-cut --
#: so the main loop names it and moves on rather than reporting it absent from
#: the ledger forever.
RE_GATE_KEY = "re_gated"


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def record_disagrees_with_its_own_directories(record: dict) -> list[str]:
    """Blocks whose declared ``artifact_sha256`` no longer digests their own tree.

    THE GUARD THAT ENCODES THE 2026-08-30 LESSON, and the reason ``--campaign``
    is safe to have at all. The first attempt to clear this ledger's forty stale
    pins installed the fresh legs INTO each block's dated directory. The ledger
    went to zero drift and FOUR certification tests went red, because every
    block carries a manifest digest over its own directory and
    ``test_certification_metadata.py`` recomputes it. The verdict looked bound
    and the record no longer described the tree it named.

    So neither writer may run against a record that is already in that state.
    Checked over every block, not just the campaign's: a re-gate is exactly the
    moment someone is tempted to overwrite a dated directory, and the check has
    to be able to see that having happened somewhere else. results/ is
    gitignored, so a directory that is not on this machine is skipped -- absence
    is not evidence.
    """
    wrong = []
    for name, value in sorted(record.items()):
        if not isinstance(value, dict):
            continue
        directory = artifact_directory(value)
        declared = value.get("artifact_sha256")
        if directory is None or not directory.is_dir() or not isinstance(declared, str):
            continue
        live = manifest_digest(directory)
        if live != declared:
            wrong.append(f"{name}: {directory.name}/ digests to {live[:12]}, "
                         f"the block records {declared[:12]}")
    return wrong


def campaign_block(record: dict, campaign: Path) -> tuple[str, dict] | None:
    """The block that RECORDS this campaign, or ``None`` if the record names none.

    A CAMPAIGN THE RECORD DOES NOT NAME IS NOT BINDABLE, and that refusal is the
    whole design. The gap this closes was never "the tool cannot read a
    directory" -- it is that a weld bound to an unnamed run points at evidence
    the narrative record has no entry for, so a later reader can follow the
    ledger to a tree and find nothing that says what it was. The remedy is to
    record the run first (``record_cuda_regate.py``), which is licensed by the
    run having happened. Matching is by the block's OWN declared artifact
    directory, never by name surgery.
    """
    for name, value in sorted(record.items()):
        if not isinstance(value, dict) or RE_GATE_KEY not in value:
            continue
        if artifact_directory(value) == campaign:
            return name, value
    return None


def device_blocks(record: dict) -> dict:
    """Every top-level object in certification.json backed by a device run.

    ENUMERATED, never listed, so a block added to the record later cannot escape
    this tool by not being mentioned in it. TWO discriminators, because one is
    demonstrably not enough:

    1. the block NAMES a device, which is what ``test_certification_metadata``'s
       ``blocks()`` keys on; and
    2. the block names an artifact directory that holds a released canonical
       policy leg -- it is BACKED by a device run whether or not it says so.

    Clause 2 is not redundant. Measured 2026-08-28:
    ``complex_offdiag_update_e_2026-08-21`` writes its device inside its ``host``
    string (``"the GPU host, NVIDIA RTX A6000, CuPy 13.5.1"``) and carries neither a
    ``device`` key nor an ``environment``, so clause 1 alone misses it -- 21
    blocks found where 22 exist, and the one missed is a released two-policy
    verdict. A ledger that inherited that blindness would have been short an
    entry on the day it was created and nothing would have said which.
    ``mutations`` and ``lifted_cases`` name a file rather than a directory and
    spell the path without this prefix, so neither clause reaches them.
    """
    found = {}
    for name, value in record.items():
        if not isinstance(value, dict):
            continue
        environment = value.get("environment")
        if value.get("device") or (isinstance(environment, dict)
                                   and environment.get("device_name")):
            found[name] = value
            continue
        directory = artifact_directory(value)
        if (directory is not None and directory.is_dir()
                and canonical_legs(directory, value.get("gate_family"))):
            found[name] = value
    return found


def artifact_directory(block: dict) -> Path | None:
    """The results directory this block's verdict lives in, off its own text."""
    for key in ("artifact", "artifacts"):
        text = block.get(key)
        if not isinstance(text, str) or ARTIFACT_PREFIX not in text:
            continue
        return _API / text[text.index(ARTIFACT_PREFIX):].split()[0].rstrip("/")
    return None


def canonical_legs(directory: Path, family: str | None = None) -> list[str]:
    """Directory-relative paths of the policy legs, sorted.

    ``family`` SCOPES A SHARED CAMPAIGN DIRECTORY TO ONE BLOCK'S OWN LEGS. Three
    families writing into one directory would otherwise give every block six legs,
    and ``read_verdict`` would then refuse each of them unless all three families
    released -- a block about ``complex`` standing on a measurement of
    ``no_pml_complex_electric``. When the block names its ``gate_family`` the legs
    are the two directories whose names are exactly ``<family>_<policy>``.
    """
    wanted = ({f"{family}_{policy}" for policy in POLICY_DIRECTORIES}
              if family else None)
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.json")
        if path.name in CANONICAL_LEG_NAMES
        and (path.parent.name in wanted if wanted is not None
             else _is_policy_directory(path.parent.name))
    )


def manifest_digest(directory: Path) -> str:
    """The record's OWN artifact rule, so the two files cannot disagree.

    ``certification.json``'s ``_artifact_sha256_rule`` states it and
    ``test_certification_metadata.py`` recomputes it: every ``*.json`` under the
    directory, sorted by relative POSIX path, one ``'<path> <sha256>'`` line
    each, joined by single newlines, sha256 of the result. A second rule here
    would give the same directory two digests and no way to tell which was
    wrong.
    """
    table = {path.relative_to(directory).as_posix():
             hashlib.sha256(path.read_bytes()).hexdigest()
             for path in directory.rglob("*.json")}
    lines = [f"{name} {digest}" for name, digest in sorted(table.items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_verdict(directory: Path, legs: list[str]) -> tuple[bool, str]:
    """(released, why) over EVERY named leg. One unreleased leg refuses the entry.

    A CUDA verdict is a pair: the same measurement under ``ieee_keep_ftz_stripped``
    and under ``meep_x86_flush``. Reading one leg would let a family that released
    on keep and failed on flush rebind as a pass, which is the whole reason the
    record stamps ``policies_cut_under`` on every block.
    """
    if not legs:
        return False, "no canonical policy leg under the artifact directory"
    for leg in legs:
        path = directory / leg
        if not path.is_file():
            return False, f"{leg} is named by the entry but not in the artifact"
        payload = json.loads(path.read_text(encoding="utf-8"))
        verdict = payload.get("canonical_verdict") or {}
        released = verdict.get("released")
        if released is None:
            # Absent and null are both "this run recorded no verdict", not "no".
            # cuda_pml_recut_2026-08-15 predates canonical_verdict and lands here.
            return False, f"{leg} carries no canonical_verdict.released"
        if not released:
            return False, (f"{leg} is not released: "
                           f"{verdict.get('reasons') or 'no reason given'}")
    return True, ""


def leg_payloads(directory: Path, legs: list[str]) -> list[dict]:
    return [json.loads((directory / leg).read_text(encoding="utf-8")) for leg in legs]


def agreed_imports(payloads: list[dict]) -> dict:
    """Files every leg imported at the SAME digest.

    A file two policy legs recorded differently was edited mid-campaign, and a
    weld cannot name one of the two digests as the bytes the gate ran. Such a
    file simply is not available to pin, which turns into a refusal wherever a
    curated key set asks for it.
    """
    maps = [payload.get("imported_source_sha256") or {} for payload in payloads]
    if not maps or not all(maps):
        return {}
    shared = set(maps[0]).intersection(*(set(m) for m in maps[1:])) if len(maps) > 1 \
        else set(maps[0])
    return {name: maps[0][name] for name in shared
            if len({m[name] for m in maps}) == 1}


def _stamp(payloads: list[dict], *keys: str) -> list:
    """Distinct values of a machine-written leg field, in first-seen order."""
    seen = []
    for payload in payloads:
        value = payload
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None and value not in seen:
            seen.append(value)
    return seen


def seed_key_set(record: dict, name: str, block: dict, imports: dict) -> list[str]:
    """The curated set for an entry this tool is creating.

    Derived, never typed: the module the block says it certifies, the shared
    compile seam and harness where the gate recorded importing them, and every
    ``gate_cuda_*.py`` in the same import map. A block with no ``kernel_module``
    of its own inherits the record's top-level one -- ``bit_identity_gate`` is
    the verdict for the root block, which is itself a block describing
    ``step_curl_kernels.py``.

    Once written the list is a claim, and :func:`main` preserves it. Re-deriving
    it on every run would let a gate that stopped importing a file quietly stop
    pinning it.
    """
    # A BLOCK MAY CERTIFY MORE THAN ONE MODULE, and pinning one of them would be
    # exactly the drift this ledger exists to catch. The 2026-09-02 stencil round
    # is the first such campaign: one gate, one artifact directory, two kernels in
    # two family modules plus the shared lift they are both spliced from. A string
    # stays a string, so every existing entry's key set is unchanged.
    declared = block.get("kernel_module") or record.get("kernel_module") or ""
    modules = [declared] if isinstance(declared, str) else list(declared)
    keys = [f"meep_gpu/cuda_kernels/{name.split('/')[-1]}"
            for name in modules if name]
    keys += [name for name in SHARED_PACKAGE_FILES + SHARED_HARNESS_FILES
             if name in imports]
    keys += [name for name in imports
             if name.startswith("parity/meep_gpu/gate_cuda") and name.endswith(".py")]
    return sorted(set(keys))


def seed_kernels(record: dict, block: dict) -> list[str]:
    """The kernel names this block claims, in the block's own spelling.

    Two of the constitutive blocks spell it ``kernels`` rather than
    ``certified_kernels`` precisely because they are additional legs on kernels
    another block certifies; both spellings are read, and the distinction stays
    where the record already makes it.
    """
    names = block.get("certified_kernels") or block.get("kernels")
    if names is None and "kernel_module" not in block:
        names = record.get("certified_kernels")
    return sorted(names or [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true",
                        help="also create entries for released blocks not yet in the ledger")
    parser.add_argument("--campaign", default="",
                        help="bind against a RE-GATE campaign tree instead of each "
                             "block's own dated directory. The tree must already be "
                             "recorded by a certification.json block carrying a "
                             f"{RE_GATE_KEY!r} table — see record_cuda_regate.py — "
                             "and that block's artifact_sha256 must recompute over "
                             "it. Nothing is installed over anything: the dated "
                             "directories and their digests are untouched.")
    parser.add_argument("--only", default="",
                        help="comma-separated certification.json block names; every "
                             "other block is SKIPPED (reported by name, never bound). "
                             "A narrowing only: it cannot admit a block the rules "
                             "below refuse, and it exists so that seeding ONE newly "
                             "released block does not also seed every other absent "
                             "block from its dated directory -- several of which "
                             "would seed DRIFTED, which is a decision and not a side "
                             "effect.")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)
    only = {name.strip() for name in args.only.split(",") if name.strip()}

    record = _record()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}
    blocks = device_blocks(record)
    # THE ``driver_dispatch`` BLOCK IS NOT THIS TOOL'S TO WRITE, and saying so here
    # rather than relying on it never matching is the point. It is cut ONLY by
    # ``recut_driver_dispatch_record.py --backend cuda``, FROM the campaign's five
    # legs, which refuses unless every leg, every other leg and the live tree agree
    # on every bound digest — a check this tool does not make and must not appear to
    # have made. A weld ledger with two writers for one entry is how a record ends
    # up describing bytes nobody ran.
    if "driver_dispatch" in ledger:
        print("  OWNED ELSEWHERE  driver_dispatch — cut by "
              "recut_driver_dispatch_record.py --backend cuda from the campaign's "
              "legs; this tool reports it and never rewrites it", flush=True)
    unknown_only = sorted(name for name in only if name not in blocks)
    if unknown_only:
        raise SystemExit(f"--only names blocks this record does not carry as device "
                         f"blocks: {unknown_only}")

    # NEITHER WRITER RUNS AGAINST A RECORD THAT DISAGREES WITH ITS OWN
    # DIRECTORIES. See the function's docstring: this is the four-broken-tests
    # failure mode, encoded where it bites instead of remembered.
    disagreeing = record_disagrees_with_its_own_directories(record)
    for line in disagreeing:
        print(f"  RECORD DISAGREES WITH ITS OWN DIRECTORY  {line}", flush=True)
    if disagreeing:
        print("\n  refusing to bind a weld into a record that no longer describes the "
              "trees it already names. Find out which side moved; do not re-cut a "
              "digest on sight.", flush=True)
        return 1

    # THE CAMPAIGN, resolved and CHECKED AGAINST THE RECORD before any entry is
    # touched. ``re_gated`` gives the block-to-subdirectory join as the record's
    # own claim; this tool never derives it from a name.
    campaign, campaign_name, re_gated = None, "", {}
    if args.campaign:
        campaign = Path(args.campaign)
        if not campaign.is_absolute():
            campaign = (_HERE / campaign).resolve()
        if not campaign.is_dir():
            raise SystemExit(f"no such campaign directory: {campaign}")
        found = campaign_block(record, campaign)
        if found is None:
            raise SystemExit(
                f"certification.json names no block whose artifacts are "
                f"{campaign.name}/ and which carries a {RE_GATE_KEY!r} table. A weld "
                f"bound to an unrecorded run points at evidence the narrative record "
                f"has no entry for. Record the run first: "
                f"record_cuda_regate.py --campaign results/{campaign.name} --write")
        campaign_name, block = found
        declared = block.get("artifact_sha256")
        live = manifest_digest(campaign)
        if declared != live:
            raise SystemExit(
                f"certification.json:{campaign_name} records artifact_sha256 "
                f"{declared} and {campaign.name}/ digests to {live}. The block and "
                f"its directory disagree; re-run record_cuda_regate.py rather than "
                f"binding against either.")
        re_gated = {row["block"]: row for row in block[RE_GATE_KEY]}

    created, rebound, skipped, drifted = [], [], [], []

    for name in sorted(blocks):
        block = blocks[name]
        entry = ledger.get(name)
        if only and name not in only:
            skipped.append((name, "not selected by --only"))
            continue
        if RE_GATE_KEY in block:
            # A RE-GATE CAMPAIGN IS NOT A FAMILY. Its verdict is the family
            # verdicts it re-cut, each of which owns its own ledger entry, so it
            # owes no weld of its own and is named here rather than reported
            # absent from the ledger on every future run.
            skipped.append((name, f"a re-gate campaign block ({RE_GATE_KEY} table "
                                  f"over {len(block[RE_GATE_KEY])} families); its "
                                  f"welds are those families' entries"))
            continue
        if campaign is not None:
            row = re_gated.get(name)
            if row is None:
                skipped.append((name, f"{campaign_name} does not re-gate this block"))
                continue
            directory = campaign / row["directory"]
        else:
            directory = artifact_directory(block)
        if directory is None:
            skipped.append((name, "the block names no artifact directory"))
            continue
        if not directory.is_dir():
            skipped.append((name, f"{directory.name}/ is not on this machine"))
            continue

        family = (block or {}).get("gate_family") if isinstance(block, dict) else None
        legs = (list(entry["legs"]) if entry and entry.get("legs")
                else canonical_legs(directory, family))
        if campaign is not None:
            # THE RE-GATE MUST HAVE CUT THE SAME LEGS. ``legs`` is a curated
            # claim -- which measurements this weld stands on -- and a fresh run
            # that wrote a different set did not re-cut this verdict, it cut
            # another one. Requiring equality rather than containment: a run
            # missing a leg is short, and a run with an extra leg is a verdict
            # this entry has never described.
            fresh = canonical_legs(directory, family)
            if fresh != legs:
                skipped.append((name, f"the re-gate wrote legs {fresh}, and this "
                                      f"entry stands on {legs}"))
                continue
        released, why = read_verdict(directory, legs)
        if not released:
            skipped.append((name, why))
            continue

        # Checked AFTER the verdict, deliberately. Reporting "not in the ledger"
        # for a block that has no readable verdict would name the wrong cause and
        # invite someone to re-run with --seed and be puzzled.
        if entry is None and not args.seed:
            skipped.append((name, "released, but not in the ledger "
                                  "(pass --seed to create it)"))
            continue

        payloads = leg_payloads(directory, legs)
        imports = agreed_imports(payloads)
        if not imports:
            skipped.append((name, "the legs record no agreed imported_source_sha256"))
            continue

        curated = list(entry["source_sha256"]) if entry else \
            seed_key_set(record, name, block, imports)
        unknown = [key for key in curated if key not in imports]
        if unknown:
            skipped.append((name, f"the artifact does not record {unknown}"))
            continue

        # Live bytes vs the bytes the gate imported. Where they match, the live
        # code digest is honest to record; where they do not, it is withheld --
        # see the module docstring and code_identity.py's closing paragraph.
        moved = [key for key in curated
                 if hashlib.sha256((_API / key).read_bytes()).hexdigest() != imports[key]]
        code = {key: code_digest_of_path(_API / key) for key in curated if key not in moved}

        hosts = _stamp(payloads, "host")
        if hosts and len(hosts) == 1:
            host = f"{hosts[0]} (machine-stamped: {legs[0]}:host)"
        else:
            declared = block.get("host")
            environment = block.get("environment")
            if not declared and isinstance(environment, dict):
                declared = environment.get("hostname")
            host = (f"{declared} (NOT machine-stamped by this gate; from "
                    f"certification.json:{name})" if declared
                    else "unknown - no leg stamps a host and the block declares none")

        policies = _stamp(payloads, "subnormal_policy_install", "policy")
        # THE RUN THIS WELD READ, spelled so a reader can walk to it. Under
        # ``--campaign`` that is the campaign subdirectory and NOT the block's
        # own dated directory, so ``records`` names the tree the digests were
        # actually taken from -- the misattribution the Triton tool wrote twice
        # in one round by taking the path from one place and the artifact from
        # another. ``campaign`` names the certification block that records the
        # run, which is the join back to what it was.
        relative = directory.relative_to(_API / "parity" / "meep_gpu" / "results")
        entry = dict(entry or {})
        entry.update({
            "artifact_sha256": manifest_digest(directory),
            "code_sha256": code,
            "gate_started_utc": min(_stamp(payloads, "started_utc") or ["unknown"]),
            "host": host,
            "kernels": entry.get("kernels") or seed_kernels(record, block),
            "kernel_module": entry.get("kernel_module") or [
                key for key in curated if key.startswith("meep_gpu/cuda_kernels/")
                and key not in SHARED_PACKAGE_FILES],
            "legs": legs,
            "purpose": entry.get("purpose") or (
                f"Device byte gate welded to certification.json:{name}."),
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "records": (f"apps/api/parity/meep_gpu/results/{relative.as_posix()}/ - "
                        f"{len(legs)} canonical policy legs, all released"),
            "source_drift": sorted(moved),
            "source_sha256": {key: imports[key] for key in curated},
            "status": "PASS" if not moved else "DRIFTED",
            "subnormal_policy": "; ".join(policies) if policies else
                                "not stamped by any leg",
            "verdict_read_from": (f"{relative.as_posix()}/{{{','.join(legs)}}}"
                                  ":canonical_verdict.released"),
        })
        if campaign is not None:
            entry["campaign"] = (
                f"certification.json:{campaign_name} — the re-gate that cut this "
                f"verdict. The block's own dated artifacts and digest are unchanged; "
                f"this weld is bound to the re-gate tree, which that block records.")
        else:
            entry.pop("campaign", None)
        if name not in ledger:
            # Written once, at creation, and never again. Everything a human
            # later adds under _notes is invisible to this tool.
            entry["_notes"] = {
                "_narrative_lives_in":
                    f"meep_gpu/cuda_kernels/certification.json:{name}",
            }
            created.append(name)
        else:
            rebound.append(name)
        if moved:
            drifted.append((name, moved))
        ledger[name] = entry

    for key in created:
        print(f"  created  {key}", flush=True)
    for key in rebound:
        print(f"  rebound  {key}", flush=True)
    for key, why in skipped:
        print(f"  SKIPPED  {key}: {why}", flush=True)
    for key, moved in drifted:
        print(f"  DRIFTED  {key}: the tree has moved under "
              f"{len(moved)} pinned file(s): {moved}", flush=True)
    print(f"\n  {len(created)} created, {len(rebound)} rebound, {len(skipped)} skipped, "
          f"{len(drifted)} carrying source drift", flush=True)

    if args.write and (created or rebound):
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"  wrote {LEDGER}", flush=True)
    elif not args.write:
        print("  (report only -- pass --write to apply)", flush=True)
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
