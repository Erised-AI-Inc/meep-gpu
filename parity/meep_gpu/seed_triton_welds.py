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
from meep_gpu import subnormal_policy as _subnormal_policy  # noqa: E402
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


#: WHERE A GATE ARTIFACT STATES THE POLICY ITS OWN PROCESS INSTALLED, in the order
#: the line is read from. Enumerated over every ``gate.json`` of the 2026-10-03 Triton
#: round (three fleet roots, the stencil welds, the unified flush run, the composition
#: and bit-identity runs): the gates do not share one spelling, and until this table
#: the reader knew only the first row, so nine released runs -- seven of them cited by
#: dispatch arms -- could not be bound. Each row is a STAMP BLOCK, the dict
#: ``subnormal_policy.policy_stamp()`` or the install report returns; a dict counts as
#: a stamp only where it carries ``resolved`` or ``requested``, which is how ``policy``
#: is told apart: it is the stamp in ``no_pml_constitutive`` and a launch-configuration
#: dict (block size, warps, the REQUESTED policy) in the fused-pair probes.
#:
#: A STAMP ANSWERS ONLY WITH ``resolved``, AND ONLY WHEN IT WAS INSTALLED.
#: ``policy_stamp()`` fills ``resolved`` before anything is installed -- it is then a
#: preference, and the stamp says so with ``installed: False`` and ``policy:
#: "none_installed"`` -- and ``requested`` is the question, never the answer. Such a
#: stamp, and one whose ``resolved`` names something other than ``keep`` or ``flush``,
#: DENIES that a policy was attained: it is a statement (so it blocks the
#: ``_mixed_policy`` exception), it never answers, and any other location that does
#: answer beside it is a contradiction. A string at a stamp path is a stamp NAME
#: (``subnormal_policy: "keep"``): compared through :data:`STAMP_NAME_POLICY`, never an
#: answer.
#:
#: The top-level ``subnormal_policy`` row comes first and keeps the line it always
#: produced, so every record this reader could already write is written byte for byte
#: as before. A line read from any other row says where it was read from.
POLICY_STAMP_BLOCKS = (
    ("subnormal_policy",),
    ("subnormal_policy_stamp",),            # complex_ade, complex_no_pml_stored_e
    ("summary", "subnormal_policy"),        # cylindrical_complex
    ("policy",),                            # no_pml_constitutive
    ("policy_stamp",),                      # the complex conductive / cylindrical probes
    ("environment", "subnormal_policy_stamp"),
    ("subnormal_policy", "stamp"),
    ("subnormal_policy_at_start",),         # folded_offdiag, taken before the first leg
    ("subnormal", "subnormal_policy"),      # nonlinear, offdiag: the leg's own re-stamp
    ("synthetic", "subnormal", "subnormal_policy"),  # bfast
    ("subnormal_policy_install",),          # the bit-identity probe's install report
)

#: A bare string naming the policy the gate INSTALLED: the four no-PML gates write
#: ``subnormal_policy_installed = "keep"`` on the line after a strict
#: ``install_subnormal_policy("keep")`` returned, and record no stamp block at all.
POLICY_INSTALLED_NAMES = (("subnormal_policy_installed",),)

#: REQUESTS, read only to cross-check. ``policy.subnormal_policy`` is the probe's
#: ``--subnormal-policy`` argument and ``subnormal_policy_requested`` the stencil
#: gate's; a request is not an attainment, so neither ever answers. An explicit
#: ``keep`` or ``flush`` resolves to itself (``subnormal_policy.policy_resolution``),
#: so one that differs from the attained policy is a contradiction; ``match_meep`` is
#: a question the resolution answers and is compared with nothing.
POLICY_REQUESTS = (("policy", "subnormal_policy"), ("subnormal_policy_requested",))

#: STAMP NAMES, read only to cross-check, through :data:`STAMP_NAME_POLICY`. The
#: gates spell the mechanism (``ieee_keep_ftz_stripped``) where the record wants the
#: policy (``keep``), and one block spells ``policy: keep`` above a stamp spelling
#: ``ieee_keep_ftz_stripped`` -- the same claim in two vocabularies, so names are
#: compared by the policy they mean and never as strings. A name outside the table
#: says nothing comparable and is not compared. ``verdict.subnormal_policy`` is the
#: bit-identity gate's verdict line and ``subnormal.policy`` the folded off-diagonal
#: leg's own name for what it ran under.
POLICY_STAMP_NAMES = (("summary", "certified_under_subnormal_policy"),
                      ("verdict", "subnormal_policy"),
                      ("subnormal", "policy"))
STAMP_NAME_POLICY = {
    _subnormal_policy.CUPY_KEEP_POLICY_NAME: _subnormal_policy.KEEP,
    _subnormal_policy.FLUSH_POLICY_NAME: _subnormal_policy.FLUSH,
    _subnormal_policy.KEEP: _subnormal_policy.KEEP,
    _subnormal_policy.FLUSH: _subnormal_policy.FLUSH,
}

#: NOT READ, and why. ``policy_stamps`` holds BOTH policies of a probe that cuts each
#: in one run (its keep install is copied into the top-level block, which is read);
#: ``environment.subnormal_policy_stamps`` is the same two-policy record kept inside
#: the environment block. ``expansion``, ``expansion_probe``, ``expansion_license``,
#: ``unified_licenses`` and ``no_device_legs`` describe the expansion-probe artifact
#: the gate LOADED from disk to license an arm: their stamps are another run's policy,
#: not this one's. ``policies`` and ``rows[].policy`` are not read for the same reason
#: as ``policy_stamps``: in the one gate that writes them (cylindrical_fused_magnetic_pair,
#: a probe that cuts both policies in one run) they list each leg's policy, ``keep``
#: and ``flush`` side by side by design, and the run's own install is the top-level
#: block.
POLICY_NOT_READ = ("policy_stamps", "environment.subnormal_policy_stamps", "policies",
                   "rows[].policy", "expansion", "expansion_probe", "expansion_license",
                   "unified_licenses", "no_device_legs")

#: THE DECLARED EXCEPTION. Three entries carry ``_mixed_policy``: their gate never
#: installs a policy, so the CuPy oracle flushed while Triton kept, and their records
#: say so in exactly this string. ``test_triton_weld_contract.POLICY_UNNAMED_BUDGET``
#: admits it per weld, on every architecture, until the gate installs a policy.
MIXED_POLICY_FIELD = "_mixed_policy"
MIXED_POLICY_LINE = "NOT INSTALLED - see _mixed_policy"


class PolicyStampConflict(ValueError):
    """The artifact's own statements of its policy contradict each other, or contradict
    the entry it would be bound into. Refused by name: a record that picked one of two
    answers would describe a run the artifact does not."""


def _at(node, path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _where(path) -> str:
    return ".".join(path)


def _policy(value) -> str | None:
    """``value`` when it is literally ``keep`` or ``flush``, else ``None`` (a dict or a
    list in a policy slot is not a name, and must not reach a set lookup)."""
    return value if isinstance(value, str) and value in _subnormal_policy.POLICIES else None


def _meant(value) -> str | None:
    """The policy a stamp NAME means (:data:`STAMP_NAME_POLICY`), or ``None``."""
    return STAMP_NAME_POLICY.get(value) if isinstance(value, str) else None


def policy_line(fresh: dict, entry: dict | None = None) -> str | None:
    """The policy this run was cut under, NAMED, in the shape the contract reads.

    ``test_every_triton_weld_names_the_policy_it_was_cut_under`` requires the string
    to NAME a policy rather than carry a blob, and a seeded entry that dumped the
    whole ``subnormal_policy`` object satisfied nothing: the resolved name is one key
    inside it. Reading `resolved` and reporting the stamp beside it is what the
    accepted entries do, so a seeded weld is indistinguishable from a rebound one.

    EVERY SPELLING IS READ AND THEY MUST AGREE. The attained policy is read from each
    stamp block (:data:`POLICY_STAMP_BLOCKS`) and installed-name string
    (:data:`POLICY_INSTALLED_NAMES`) the artifact carries; requests and stamp names
    are read beside them as cross-checks. Two statements naming different policies
    raise :class:`PolicyStampConflict` naming both, rather than letting table order
    pick one.

    ONLY AN INSTALLED ``resolved`` ANSWERS. A stamp answers with its ``resolved`` key
    alone; ``requested`` is a cross-check. A stamp taken before any install
    (``installed: False`` or ``policy: "none_installed"``) or whose ``resolved`` names
    neither policy is a DENIAL: it counts as stated, never answers, and an answer
    found anywhere else beside it is refused as a contradiction.

    Returns ``None`` when the artifact states no attained ``keep`` or ``flush`` at all,
    unless ``entry`` declares :data:`MIXED_POLICY_FIELD` and the artifact states
    nothing about a policy anywhere this reader looks: that is the declared exception,
    and the line is :data:`MIXED_POLICY_LINE`, exactly as the existing records spell
    it. A location whose value is ``null`` counts as absent (the 2026-09-23 fleet
    wrote ``subnormal_policy: null`` for gates that stamped nothing), so it neither
    answers nor blocks the exception. An entry that declares the exception for a run
    that DID state a policy is
    refused: the binder may not edit the declaration, and a record naming ``keep``
    beside an entry that says it may not claim ``keep`` contradicts itself.
    """
    policies = set(_subnormal_policy.POLICIES)
    attained: list[tuple[str, str]] = []       # (where, keep|flush): answers
    checks: list[tuple[str, str]] = []         # (where, keep|flush): cross-checks
    denials: list[str] = []                    # stamps saying no policy was attained
    stated: list[str] = []                     # every enumerated location present
    primary = None                             # (path, block, resolved)

    for path in POLICY_STAMP_BLOCKS:
        block = _at(fresh, path)
        if block is None:
            continue
        if isinstance(block, str):
            # A stamp NAME where a stamp block is usually written: a statement, and
            # compared by what it means, but never an answer.
            stated.append(_where(path))
            meant = _meant(block)
            if meant is not None:
                checks.append((_where(path), meant))
            continue
        if not isinstance(block, dict) or not ("resolved" in block
                                               or "requested" in block):
            # ``policy`` as a launch-configuration dict, or a block that names no
            # resolution: present, but not a stamp.
            if path != ("policy",):
                stated.append(_where(path))
            continue
        stated.append(_where(path))
        resolved = block.get("resolved")
        uninstalled = (block.get("installed") is False
                       or block.get("policy") == _subnormal_policy.UNINSTALLED_POLICY_NAME)
        if uninstalled or (resolved is not None and _policy(resolved) is None):
            denials.append(
                f"{_where(path)} (installed={block.get('installed', '?')!r}, "
                f"policy={block.get('policy', '?')!r}, resolved={resolved!r})")
            continue
        requested = _policy(block.get("requested"))
        if requested is not None:
            checks.append((f"{_where(path)}.requested", requested))
        if resolved is None:
            # A request with no answer: compared, never read as the attained policy.
            continue
        attained.append((_where(path), resolved))
        meant = _meant(block.get("policy"))
        if meant is not None:
            checks.append((f"{_where(path)}.policy", meant))
        if primary is None:
            primary = (path, block, resolved)

    for path in POLICY_INSTALLED_NAMES:
        name = _at(fresh, path)
        if name is None:
            continue
        stated.append(_where(path))
        if _policy(name) is None:
            raise PolicyStampConflict(
                f"{_where(path)}={name!r} names no attained policy (one of "
                f"{sorted(policies)}), so the artifact's statement of what it "
                f"installed cannot be read")
        attained.append((_where(path), name))

    for path in POLICY_REQUESTS:
        name = _at(fresh, path)
        if name is None:
            continue
        stated.append(_where(path))
        if _policy(name) is not None:
            checks.append((_where(path), name))

    for path in POLICY_STAMP_NAMES:
        name = _at(fresh, path)
        if name is None:
            continue
        stated.append(_where(path))
        meant = _meant(name)
        if meant is not None:
            checks.append((_where(path), meant))

    named = {policy for _where_, policy in attained + checks}
    if len(named) > 1:
        raise PolicyStampConflict(
            "the artifact's policy spellings disagree: "
            + ", ".join(f"{where}={policy}" for where, policy in attained + checks))
    if denials and attained:
        raise PolicyStampConflict(
            f"the artifact both denies and states an attained policy: "
            f"{', '.join(denials)} says none was installed, while "
            + ", ".join(f"{where}={policy}" for where, policy in attained)
            + " answers; a record cannot pick one of the two")

    declared = (entry or {}).get(MIXED_POLICY_FIELD)
    if isinstance(declared, str) and declared.strip():
        if stated:
            raise PolicyStampConflict(
                f"the entry declares {MIXED_POLICY_FIELD} (its gate installs no "
                f"policy) but this run states one at {stated}; the declaration and "
                f"the run disagree, and retiring a declaration is a reviewed change "
                f"to the entry, not something a binder does")
        return MIXED_POLICY_LINE

    if primary is not None:
        path, block, resolved = primary
        stamp = block.get("policy") or "?"
        # ABSENT AND EMPTY ARE DIFFERENT MEASUREMENTS. The keep policy is attainable
        # only with an `ftz_stripped` CuPy cache, so "carries no ftz token" is a claim
        # about the run; an artifact whose stamp recorded no cache directory at all
        # supports neither that claim nor its opposite, and says so.
        if "cache_dir" not in block:
            token = "the run recorded no CuPy cache directory"
        else:
            cache = str(block.get("cache_dir") or "")
            token = ("the CuPy cache directory carries ftz_stripped" if "ftz_stripped"
                     in cache else "the CuPy cache directory carries no ftz token")
        where = "" if path == POLICY_STAMP_BLOCKS[0] else f"; read from {_where(path)}"
        return (f"{resolved} (installed by the gate before its first device compile: "
                f"requested={block.get('requested', '?')} resolved={resolved}; stamp "
                f"{stamp}; {token}{where})")
    if attained:
        # A WELD MUST NAME THE POLICY IT WAS CUT UNDER, and only an installed name is
        # left: the gate recorded no stamp, so the mechanism and the cache directory
        # are unstated and the line says so instead of supplying them.
        where, resolved = attained[0]
        return (f"{resolved} (the gate's own {where}={resolved!r}; the run recorded no "
                f"policy stamp, so neither the mechanism nor a CuPy cache directory is "
                f"stated)")
    # "?" is not a name: the contract test greps this line for `keep` or `flush`, and
    # a placeholder would clear the presence check while failing the naming one -- a
    # red the seeding round could not explain. Refuse instead, by returning None.
    return None


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
        # must be indistinguishable, which two f-strings do not stay. No entry is
        # passed: a seed runs only where the key does not exist, so no declaration
        # can apply.
        try:
            policy = policy_line(fresh)
        except PolicyStampConflict as exc:
            refused.append((sub.name, f"policy refused: {exc}"))
            continue
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
