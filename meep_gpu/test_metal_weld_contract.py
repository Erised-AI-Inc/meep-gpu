"""Every Metal weld is bound to the bytes its gate executed.

The Triton counterpart is test_dispatch_contract.py's weld test. This is the same
guarantee for the Metal track, and it is ENUMERATED for the reason that one was
rewritten on 2026-08-19: it named three gates by hand, so six welds added the same
day recorded a source_sha256 that nothing recomputed. A hand-maintained list is a
second place to forget a weld; the record IS the list.

WHY THIS BECAME POSSIBLE. A weld needs two things from a gate - which bytes ran,
and whether it passed. Neither was mechanically available on this track until
2026-08-19: the 30 gates now stamp imported_source_sha256 through gate_provenance,
and their verdicts are readable through read_verdict, which understands the ten
different spellings the fleet uses. The artifacts themselves share NO common
top-level key, so nothing gate-specific could ever have been required.

THE WALK, added 2026-08-30 (third round), and it is why this file grew. Until
today the enumerator here read `entry["source_sha256"]` for each top-level entry
and nothing else, so it compared 259 of the record's 694 digests and reported that
as "every Metal weld is bound to the live sources". The 435 it did not look at:

    259  code_sha256   compared by NOTHING at the merge bar. The code tier is the
                       one that licenses a comment-only edit to stand, so leaving
                       it unchecked is exactly how a stale record keeps looking
                       current - the raw pin goes red, the code pin is read as the
                       reason that is fine, and nothing ever established that the
                       code pin still describes anything.
     56  host_sha256   recomputed by test_metal_kernels.py - excluded, and the
     36  kernel slots  exclusion now says WHERE instead of being a missing key.
     44  artifact      digests of gitignored run artifacts.
     40  device_sha256 the shader source strings that actually reached the Metal
                       compiler, read by no laptop test at all.

meep_gpu.weld_record_walk now recurses the whole record - dicts and lists, at any
depth - and every 64-hex digest it finds is either CHECKED against this tree in a
named tier or EXCLUDED BY A NAMED RULE carrying its reason. A digest matching
neither is UNCLASSIFIED, which fails.

THE FOURTH ROUND, 2026-08-30, took the two "excluded, and the exclusion says
WHERE" lines above at their word and checked them. An exclusion is a DECISION NOT
TO COMPARE and rests on a reason; a reason that points at another test is a
CITATION, and until this round nothing established that the cited test made the
claim the citation advertised. It did not, on the other track: Triton's
`host_sha256` deferred to a test that recomputes each digest and then calls
`device_identity.weld_survives_edit` on a mismatch - the allowance both weld
contracts refuse by name. So:

  * the 56 `host_sha256` digests are CHECKED HERE now, raw, with no allowance;
    the citation is deleted rather than watched, because a reference that cannot
    go stale beats one that has to be. Measured after promotion: 56 of 56 match.
    `metal_kernels` therefore leaves NOT_A_GATE_RECORD and enters
    NOT_A_DEVICE_GATE - it owns checked digests, and it has no gate script.
  * the 36 generated kernel-source slots stay excluded, because nothing in this
    checkout is their subject - and their citation is now ASSERTED rather than
    trusted: test_the_generated_slot_delegation_is_real_and_admits_no_allowance
    requires the delegate to exist, to compare the whole dict, and to admit no
    allowance call.

Measured 2026-08-30, fourth round:

    discovered 694   checked 614   excluded  80   unclassified 0   drifted 0

and the DISCOVERED count is asserted too, so a digest that stops being found fails
here rather than shrinking every count derived from it.

Stdlib, pytest, and two in-package identity helpers. No device.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import pytest

from . import weld_record_walk as walk
from .code_identity import code_digest
from .device_identity import device_digests

PACKAGE = pathlib.Path(__file__).resolve().parent
RECORD = PACKAGE / "metal_kernels" / "fingerprints.json"
REPO = PACKAGE.parent


#: Entries that own NO checked digest, excluded BY NAME with the reason. There is
#: no predicate that quietly drops an entry - see the rule below.
#: EMPTY SINCE 2026-08-30 (fourth round), and empty on purpose rather than
#: deleted. Its one member, `metal_kernels`, moved to NOT_A_DEVICE_GATE when its
#: 56 host digests stopped being deferred and became checked here: every dict
#: entry in this record now owns a checked digest, so nothing needs excusing from
#: the byte check. The dict stays because the RULE stays - an entry that pins
#: nothing must be named here with a reason, never dropped by a predicate - and
#: `test_every_record_entry_is_checked_or_excluded_by_name` fails on the first
#: entry that tries.
NOT_A_GATE_RECORD = {}

#: Entries that own checked digests but are NOT device gate records, so the
#: gate-script requirement below does not apply to them. Named, with the reason,
#: for the same argument as everything else here: a predicate that decided this
#: by shape would be the next filter.
#:
#: `metal_kernels` moved OUT of NOT_A_GATE_RECORD on 2026-08-30 and into this
#: list, because its 56 `host_sha256` digests are now CHECKED here rather than
#: deferred. The old exclusion read "they ARE recomputed against the tree, by
#: test_metal_kernels.py - checking them twice is two places to disagree". That
#: argument holds only where the two places make the SAME claim, and on the
#: Triton side the corresponding delegate recomputes THROUGH
#: device_identity.weld_survives_edit - an allowance this contract refuses by
#: name. Rather than verify a citation per track, the citation is gone: the raw
#: bytes are compared in the record that makes the claim.
NOT_A_DEVICE_GATE = {
    "metal_kernels":
        "the package-level block (host_sha256, kernel_source_sha256, validated "
        "frontend/torch versions). It pins the shipped host side, which is why "
        "its digests are checked; it is not a device run, so it has no gate "
        "script to pin and requiring one would be a red for a shape that has "
        "none",
}

#: Measured 2026-08-30. THE DENOMINATOR IS PART OF THE CHECK: these fail when
#: coverage SHRINKS, so an entry cannot leave the strict comparison by dropping a
#: key. That is exactly how the Triton track lost ten entries and 88 pinned pairs
#: - 31 of them live drifts - to a `status == "PASS"` filter for eleven days.
#: 44 device gate records + the `metal_kernels` package block, which entered the
#: checked set on 2026-08-30 when its 56 host digests stopped being delegated.
ENTRY_FLOOR = 45
PIN_FLOOR = 315

#: THE TOTAL, and the tier floors under it. DISCOVERED is the one the other
#: floors cannot substitute for: a digest that stops being FOUND also stops being
#: counted by every floor derived from the walk, so only the total can report it.
#: Measured 2026-08-30 (third round) by recursing the record.
DISCOVERED_FLOOR = 694
CHECKED_FLOOR = 614          # 558 + the 56 host digests, promoted 2026-08-30
RAW_TIER_FLOOR = 315         # 259 source-map + 56 host-side
CODE_TIER_FLOOR = 259
DEVICE_TIER_FLOOR = 40

#: Which rules the walk fires on THIS record, and how many times at the floor.
#: BOTH DIRECTIONS: a rule that stops firing means its shape has left the record
#: (and an unexercised rule is where the next omission hides); a rule that starts
#: firing means a shape arrived that nobody wrote down.
EXPECTED_RULES = {
    walk.TIER_CODE + ":code_digest_map": 259,
    walk.TIER_RAW + ":source_digest_map": 259,
    walk.TIER_RAW + ":host_side_digest_map": 56,
    "artifact_or_log_digest": 44,
    walk.TIER_DEVICE + ":device_kernel_digest": 40,
    "generated_kernel_source_slot": 36,
    # ARRIVED 2026-09-12, on the FIRST recut this record has ever had. The recut
    # writes `driver_dispatch._source_sha256_recut_log`, a from/to pair per file
    # whose digest moved, and the walk already had a rule and a written reason for
    # the shape (weld_record_walk.RULE_REASONS["historical_recut_log"]): half of
    # every pair is stale by construction, because a log that had to match the tree
    # could not record a change. So it is EXCLUDED from the checked set, not pinned.
    #
    # THE FLOOR IS 2 AND NOT TODAY'S 4, which is the one judgement here. Every other
    # floor in this dict counts COVERAGE, and shrinking coverage is the failure they
    # exist to catch. This one counts how many files the LAST recut happened to move
    # — today two, hence four digests — which is not a coverage property at all. A
    # future recut that moves one file writes a well-formed log of 2 and must not
    # fail. 2 is the smallest well-formed log: one file, its from and its to. Zero
    # would be wrong too, since the `missing` check above already requires the rule
    # to fire at all, and a recut that moved nothing should not have been cut.
    "historical_recut_log": 2,
}


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _census(record=None):
    """EVERY digest in the record, classified. The denominator, structurally derived.

    THE ONE ENUMERATOR. Nothing in this file walks a shape any more - no
    `record.items()` loop that assumes a top level, no `entry["source_sha256"]`
    that assumes a key. What the record holds is what gets classified, and a
    digest the classifier has no rule for is UNCLASSIFIED: a failure, not silence.
    """
    return walk.census(_record() if record is None else record)


def _checked(found):
    return [f for f in found if f.tier is not None]


def _of_tier(found, tier):
    return [f for f in found if f.tier == tier]


def _resolve(name, root=REPO):
    """A pinned path, under both spellings this record uses.

    ELEVEN OF THE DEVICE PINS ARE ABSOLUTE - `<repo>/meep_gpu/
    metal_kernels/shaders.py` and friends, written by whichever machine minted the
    weld. That is a portability defect in the record (a clean checkout elsewhere
    would report eleven MISSING lines, a red for the wrong reason), and it is
    cleared by rebinding those blocks repo-relative with mint_metal_weld.py. Until
    then an absolute path that lies INSIDE this repo is rewritten to its
    repo-relative form - the same file under another name, never a different one -
    and one that does not is left alone so it reports MISSING rather than
    resolving to something on whichever machine happens to run this.

    A BARE BASENAME means this record's own directory, then the package root.
    The `host_sha256` map keys its 56 entries that way (`launch.py`,
    `shaders.py`), which is why promoting that map from excluded to CHECKED needs
    this branch and not a special case: `root / "launch.py"` is `<repo>/
    launch.py` and does not exist, so all 56 would have reported MISSING FROM THE
    TREE - a red for the wrong reason, and indistinguishable from a red for the
    right one. The FIRST EXISTING candidate wins and no digest is consulted, so
    resolution cannot be steered toward whatever would make a comparison pass.
    """
    if name.startswith("/"):
        absolute = pathlib.Path(name)
        try:
            return root / absolute.relative_to(REPO)
        except ValueError:
            return absolute
    if "/" not in name:
        for base in ("meep_gpu/metal_kernels", "meep_gpu"):
            candidate = root / base / name
            if candidate.is_file():
                return candidate
        return root / "meep_gpu" / "metal_kernels" / name
    return root / name


def _raw_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest_of(path):
    """``None`` for a file with no code identity, which walk.compare REPORTS.

    `code_digest` parses Python and nothing else, so a `code_sha256` recorded for
    a `.metal` shader or a JSON is a digest nothing can recompute. Returning None
    rather than falling back to a byte hash keeps that a report instead of a
    silently-passing comparison - the difference between a rule and a hole.
    """
    if path.suffix != ".py":
        return None
    return code_digest(path.read_text(encoding="utf-8"))


def _device_digests_of(path):
    try:
        _, digests = device_digests(path)
    except (SyntaxError, OSError, ValueError):
        return None
    return digests


def _drifted(found, root=REPO):
    """Every CHECKED digest the tree does not support, ACROSS ALL THREE TIERS.

    THE RULE IS THE RAW BYTES in the raw tier, WITH NO ALLOWANCE, and there is
    deliberately no fall-through to a laxer digest. A weld says "this gate ran
    THESE bytes on a device"; anything that decides an edit is harmless is a
    judgement made on the host about a claim only the device can settle, and a
    general allowance can license the very edit it is asked to judge - including
    an edit to the module that implements the allowance. The code tier is NOT such
    a fall-through: it is compared to the live file's own code identity, never
    consulted to excuse a raw mismatch. If a specific entry ever needs an
    exception, name that entry here and say why; a drifted weld is otherwise fixed
    by re-running its gate, never by relaxing this comparison or typing a digest
    in.

    Factored out because an assertion nobody has watched FAIL is a measurement
    that cannot fail: `test_the_drift_check_refuses_a_planted_digest` drives this
    same function with a deliberately wrong digest in each tier.
    """
    return walk.compare(
        found,
        resolve=lambda name: _resolve(name, root),
        raw_digest=_raw_digest,
        code_digest_of=_code_digest_of,
        device_digests_of=_device_digests_of,
        rebind_tool="parity/meep_gpu/rebind_metal_welds.py",
    )


def _pinned(record=None):
    """Every top-level entry owning at least one CHECKED digest.

    AN ENTRY IS CHECKED IF IT PINS A FILE DIGEST - anywhere in it, in any tier.
    Not whether it passed, not whether it carries a `status` key, and no longer
    whether that digest happens to sit under a top-level `source_sha256`. A
    recorded digest is a claim about this tree and can be checked against this
    tree whatever the run concluded; a filter on a field a gate record need not
    carry is a hand-maintained list wearing an enumeration's clothes.
    """
    record = _record() if record is None else record
    owners = {f.trail[0] for f in _checked(_census(record))}
    return {name: entry for name, entry in record.items()
            if name in owners and isinstance(entry, dict)}


def _welds():
    """The PASS subset - the METADATA tier. The byte tier is :func:`_pinned`."""
    record = _record()
    return {name: entry for name, entry in record.items()
            if isinstance(entry, dict) and entry.get("status") == "PASS"
            and entry.get("source_sha256")}


# ---------------------------------------------------------------------------
# THE DENOMINATOR
# ---------------------------------------------------------------------------


def test_every_discovered_digest_is_checked_or_excluded_by_a_named_rule():
    """THE WALK. Every digest in the record, wherever it lives, has a fate.

    THE TEST THAT REMOVES THE CLASS. The round before it removed a status filter
    and left a shape filter: an enumerator that knew about top-level entries and
    one key. Here the record is RECURSED - dicts and lists, at any depth - and
    each digest is either CHECKED in a named tier or EXCLUDED BY A NAMED RULE
    carrying its reason. There is no third outcome.

    THE TOTAL IS ASSERTED, not only the checked count: a digest that stops being
    DISCOVERED is the failure no count of what is checked can see.
    """
    found = _census()
    assert len(found) >= DISCOVERED_FLOOR, (
        f"the walk discovered {len(found)} digests, was {DISCOVERED_FLOOR} - "
        f"digests have stopped being FOUND, which is a shrunken denominator that "
        f"no count of what is CHECKED can report")
    unclassified = [walk.dotted(f.trail) for f in found
                    if f.rule == walk.UNCLASSIFIED]
    assert not unclassified, (
        f"{len(unclassified)} digest(s) matched no rule: {unclassified[:12]} - "
        f"write a rule in meep_gpu/weld_record_walk.py, in this change: check it, "
        f"or exclude it by name with the reason")
    checked = _checked(found)
    assert len(checked) >= CHECKED_FLOOR, (
        f"only {len(checked)} of {len(found)} discovered digests are checked, was "
        f"{CHECKED_FLOOR} - digests moved from checked to excluded, which is a "
        f"decision that has to be argued, not a count that may drift")
    fired = walk.rules_that_fired(found)
    reasonless = sorted(set(fired) - set(walk.RULE_REASONS))
    assert not reasonless, f"rules fired with no written reason: {reasonless}"
    missing = sorted(set(EXPECTED_RULES) - set(fired))
    assert not missing, (
        f"declared rules that no longer fire on this record: {missing} - the "
        f"shape each was written for has left, and a rule nothing exercises is "
        f"where the next real omission hides")
    surprise = sorted(set(fired) - set(EXPECTED_RULES) - {walk.UNCLASSIFIED})
    assert not surprise, (
        f"rules fired that this record never declared: {surprise} - a new shape "
        f"arrived; declare it in EXPECTED_RULES with its count")
    shrunk = {name: (fired[name], floor) for name, floor in EXPECTED_RULES.items()
              if fired[name] < floor}
    assert not shrunk, (
        f"rules whose digest count SHRANK (got, was): {shrunk} - coverage does "
        f"not fall silently")


def test_every_record_entry_is_checked_or_excluded_by_name():
    """Nothing leaves coverage by omitting a field, in either direction."""
    record = _record()
    dict_entries = {name for name, entry in record.items() if isinstance(entry, dict)}
    checked = set(_pinned(record))
    unaccounted = sorted(dict_entries - checked - set(NOT_A_GATE_RECORD))
    assert not unaccounted, (
        f"{len(unaccounted)} record entries are neither checked nor excluded by "
        f"name: {unaccounted} - pin the bytes the gate executed, or name each "
        f"here with the reason it is not a gate record")
    stale = sorted(set(NOT_A_GATE_RECORD) - dict_entries)
    assert not stale, f"the exclusion list names entries the record lacks: {stale}"
    assert not sorted(checked & set(NOT_A_GATE_RECORD)), (
        "an entry that pins bytes is checked; delete the exclusion, not the coverage")
    # THE SECOND EXCLUSION LIST, held to the same rule, in both directions. An
    # entry named here must EXIST and must be CHECKED - a name that has stopped
    # matching an entry, or that quietly moved out of the checked set, is a place
    # the next omission hides.
    absent = sorted(set(NOT_A_DEVICE_GATE) - dict_entries)
    assert not absent, f"NOT_A_DEVICE_GATE names entries the record lacks: {absent}"
    unchecked = sorted(set(NOT_A_DEVICE_GATE) - checked)
    assert not unchecked, (
        f"{unchecked} is excused the gate-script requirement while owning no "
        f"checked digest - that is two exemptions, not one; name it in "
        f"NOT_A_GATE_RECORD or pin its bytes")


def test_no_exclusion_on_this_record_rests_on_an_unevaluated_premise(tmp_path):
    """AN EXCLUSION IS ONLY AS GOOD AS ITS PREMISE - the fifth sub-form.

    BOTH DIRECTIONS, and on this record the first is a NEGATIVE result worth
    asserting: no digest here is excluded on a claim about the checkout, so the
    premise checker has nothing to report. A negative that is not also ARMED is
    indistinguishable from a checker that never ran, which is the failure mode
    this whole file exists to remove - so the same function is driven against a
    planted census in which the premise IS violated, and must report it.
    """
    found = _census()
    premised = [f for f in found if f.rule in walk.PREMISE_ABSENT_FROM_TREE]
    assert premised == [], (
        f"{len(premised)} digests now rest on a not-in-this-checkout premise on "
        f"the Metal record; assert it the way the Triton contract does")

    planted = walk.census(
        {"g": {"staging": {"staging_only_file":
                           {"path": "meep_gpu/metal_kernels/launch.py",
                            "sha256": "a" * 64}}}})
    armed = walk.premise_violations(
        planted,
        is_file=lambda name: _resolve(name).is_file(),
        inside_repo=lambda name: True,
    )
    assert len(armed) == 1 and "IS in this checkout" in armed[0][2], armed


def test_the_generated_slot_delegation_is_real_and_admits_no_allowance():
    """THE ONE EXCLUSION HERE THAT STILL CITES ANOTHER TEST, and it is asserted.

    36 `kernel_source_sha256` slots are digests of GENERATED shader source
    strings keyed by slot name. Nothing in this checkout is their subject, so
    there is no raw comparison to promote them to - which is exactly the argument
    `host_sha256` could NOT make, and why that map is now checked here instead.

    A citation is not a comparison. This one is therefore checked as a citation:
    the delegate module exists, the named test is in it, and the comparison it
    makes is a WHOLE-DICT equality with no allowance call. That last clause is the
    one that matters, because the Triton delegate `host_sha256` used to defer to
    recomputes each digest and then calls `device_identity.weld_survives_edit` on
    a mismatch - a strictly weaker claim than the exclusion advertised, and
    invisible from the citation alone.
    """
    # ONE delegating exclusion on this record, and it is this one. A second would
    # be a second unasserted citation, which is the shape the whole test is about.
    # Read from the walker's DELEGATED_COVERAGE register rather than sniffed out
    # of reason text: several reasons cross-reference a test that EARNS an
    # exclusion, and that is a different claim from "compared over there".
    delegating = sorted({f.rule for f in _census()
                         if f.tier is None and f.rule in walk.DELEGATED_COVERAGE})
    assert delegating == ["generated_kernel_source_slot"], (
        f"exclusions delegating their comparison: {delegating} - each one needs "
        f"its citation asserted here, or its digests checked")

    # THE CITATION IS READ FROM THE REGISTER, not restated here, so the delegate
    # this test drives is by construction the one the exclusion names.
    filename, function = walk.DELEGATED_COVERAGE[
        "generated_kernel_source_slot"].split("::")
    delegate = PACKAGE / filename
    assert delegate.is_file(), f"the cited delegate is not in the tree: {delegate}"
    tree = ast.parse(delegate.read_text(encoding="utf-8"))
    named = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == function]
    assert len(named) == 1, (
        f"{filename} no longer defines {function}, which 36 excluded digests are "
        f"cited to - check them here, or fix the citation")
    body = ast.dump(named[0])
    assert "'kernel_source_sha256'" in body, (
        "the cited test no longer compares kernel_source_sha256 - the 36 slots "
        "it is credited with covering are covered by nothing")
    allowances = sorted({node.func.id for node in ast.walk(named[0])
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id in ("weld_survives_edit", "code_digest")})
    assert not allowances, (
        f"the cited test now admits {allowances} - it makes a weaker claim than "
        f"this exclusion states, which is precisely how host_sha256 went "
        f"unchecked for eleven days. Check the slots here, or restate the "
        f"exclusion as the weaker claim it actually rests on")


# ---------------------------------------------------------------------------
# THE THREE TIERS
# ---------------------------------------------------------------------------


def test_every_metal_weld_pins_the_script_that_gated_it():
    """44 of 44 today. The sources say which implementation ran; the gate script
    says what was asked of it, and a gate that changes underneath an unpinned weld
    rewrites the result while every source digest still verifies."""
    naked = sorted(name for name, entry in _pinned().items()
                   if name not in NOT_A_DEVICE_GATE
                   and not any(p.startswith("parity/")
                               for p in (entry.get("source_sha256") or {})))
    assert not naked, (
        f"{len(naked)} Metal welds pin no gate or probe script under "
        f"parity/meep_gpu/: {naked}")


def test_the_metal_welds_are_bound_to_the_live_sources():
    welded = _pinned()
    raw = _of_tier(_census(), walk.TIER_RAW)
    assert len(welded) >= ENTRY_FLOOR, (
        f"only {len(welded)} Metal entries own a checked digest, was {ENTRY_FLOOR} "
        f"- coverage SHRANK; an entry does not leave this check by losing a key")
    assert len(raw) >= PIN_FLOOR, (
        f"only {len(raw)} raw-tier digests to compare across {len(welded)} "
        f"entries, was {PIN_FLOOR} - the shape changed and this test is inert")
    drift = _drifted(raw)
    assert not drift, (
        f"{len(drift)} of {len(raw)} pinned path(s) drifted from the bytes the "
        f"gate executed - re-run the gate, do not edit this record: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def test_the_metal_code_tier_recomputes_against_the_tree():
    """259 ``code_sha256`` pins, read by NOTHING at the merge bar until today.

    WHY THIS TIER IS THE DANGEROUS ONE TO LEAVE UNCHECKED. `code_sha256` is the
    digest that licenses a weld to stand after a comment-only edit - the one field
    whose whole job is to say "the raw bytes moved and it did not matter". An
    unchecked field with that job is how a stale record keeps looking current.

    THE RULE, and it is a rule rather than a skip: a `code_sha256` is compared to
    `code_identity.code_digest` OF THE LIVE FILE, never to the file's source
    digest. The two tiers MAY legitimately disagree, and exactly one thing makes
    that so - a comment or docstring edit since the gate ran, which is the entire
    clause code_identity licenses. Every other disagreement is a defect: a code
    digest that no longer recomputes describes neither the bytes that ran nor the
    bytes on disk.

    Measured 2026-08-30: 259 of 259 recompute.
    """
    code = _of_tier(_census(), walk.TIER_CODE)
    assert len(code) >= CODE_TIER_FLOOR, (
        f"only {len(code)} code_sha256 pins found, was {CODE_TIER_FLOOR} - the "
        f"tier left the comparison, which is how it went unread")
    drift = _drifted(code)
    assert not drift, (
        f"{len(drift)} of {len(code)} code digest(s) no longer recompute against "
        f"the tree - the recorded code identity describes neither the bytes the "
        f"gate ran nor the bytes on disk. Re-run the gate and rebind; do not "
        f"hand-edit the digest: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def test_no_entry_backfills_a_code_digest_beside_a_stale_source_digest():
    """THE CLAUSE `code_identity` WITHHOLDS, enforced on this record too.

    `code_identity.py` states the rule in its own docstring and it is a rule about
    every track: "Backfilling `code_sha256` is only honest for a weld whose
    `source_sha256` still matches, because only then is the file on disk the bytes
    the gate actually ran." A code digest recomputed today and written beside a
    source digest that has ALREADY moved describes the current tree, not the gate,
    and MANUFACTURES the licence the clause withholds - the next reader finds a
    green code pin beside a red source pin and reads the green one as the reason
    the red one is fine.

    Until 2026-08-30 (fourth round) only the CUDA contract enforced this, over its
    39 pins. A rule universal in its statement and partial in its coverage is the
    same shape as every other defect removed here, so it is brought across: 259
    pins here, 183 on the Triton side.

    MEASURED: 0 offenders, and 0 source pins drifted - so the green says nothing
    on its own, which is what the planted control is for.
    """
    found = _census()
    source = [f for f in found if f.rule == walk.TIER_RAW + ":source_digest_map"]
    code = _of_tier(found, walk.TIER_CODE)
    assert len(source) >= 259 and len(code) >= CODE_TIER_FLOOR, (len(source), len(code))
    moved_names = {where for where, _, _ in _drifted(source)}
    moved = {(f.trail[0], f.subject) for f in source
             if walk.dotted(f.trail) in moved_names}
    offenders = sorted(walk.dotted(f.trail) for f in code
                       if (f.trail[0], f.subject) in moved)
    assert not offenders, (
        f"{len(offenders)} code digest(s) sit beside a STALE source digest: "
        f"{offenders[:8]} - that manufactures the licence code_identity "
        f"withholds. Re-run the gate; do not backfill the code digest")

    # ARMED, in both directions, with a CURRENT code digest so the leg being
    # exercised is the pairing rule and not the code tier.
    victim = "meep_gpu/metal_kernels/launch.py"
    live_code = code_digest(_resolve(victim).read_text(encoding="utf-8"))
    plant = {"w": {"source_sha256": {victim: "a" * 64},
                   "code_sha256": {victim: live_code}}}
    planted = _census(plant)
    assert _drifted(_of_tier(planted, walk.TIER_CODE)) == []
    stale_names = {where for where, _, _ in _drifted(
        [f for f in planted if f.rule == walk.TIER_RAW + ":source_digest_map"])}
    stale = {(f.trail[0], f.subject) for f in planted
             if f.rule == walk.TIER_RAW + ":source_digest_map"
             and walk.dotted(f.trail) in stale_names}
    caught = sorted(walk.dotted(f.trail) for f in planted
                    if f.tier == walk.TIER_CODE and (f.trail[0], f.subject) in stale)
    assert caught == [f"w.code_sha256.{victim}"], caught

    plant["w"]["source_sha256"][victim] = _raw_digest(_resolve(victim))
    honest = _census(plant)
    assert not {where for where, _, _ in _drifted(
        [f for f in honest if f.rule == walk.TIER_RAW + ":source_digest_map"])}


def test_the_metal_device_tier_recomputes_against_the_tree():
    """40 ``device_sha256`` shader digests, likewise read by no laptop test.

    This is the identity that actually reached the Metal compiler:
    `device_identity.device_digests` parses the module and hashes the shader
    source string it emits. A module can be edited so its FILE moves while every
    shader it emits is unchanged, and so a shader changes while the file's code
    identity barely does - which is why this tier is neither of the other two.

    A declared shader the module no longer generates is REPORTED, not skipped: a
    digest naming a shader that has left the file is a claim about nothing.
    """
    device = _of_tier(_census(), walk.TIER_DEVICE)
    assert len(device) >= DEVICE_TIER_FLOOR, (
        f"only {len(device)} device_sha256 shader digests found, was "
        f"{DEVICE_TIER_FLOOR} - the tier left the comparison")
    drift = _drifted(device)
    assert not drift, (
        f"{len(drift)} of {len(device)} device shader digest(s) no longer "
        f"recompute - re-run the gate and rebind: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


# ---------------------------------------------------------------------------
# THE ARMED CONTROLS
# ---------------------------------------------------------------------------


def _plant(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def test_the_drift_check_refuses_a_planted_digest(tmp_path):
    """THE ARMED CONTROL. This record is green in all three tiers, so without a
    control its greens are indistinguishable from a comparison that never ran.

    Driven through the WALK rather than a hand-built pin list, so the classifier
    is exercised too: a rule that stopped claiming these shapes would take the
    planted digest out of the checked set and this test would go red.
    """
    victim = PACKAGE / "metal_kernels" / "shaders.py"
    rel = victim.relative_to(REPO).as_posix()
    source = victim.read_text(encoding="utf-8")
    raw_now = hashlib.sha256(victim.read_bytes()).hexdigest()
    code_now = code_digest(source)
    wrong = "f" * 64

    planted = {"a_synthetic_weld": {"source_sha256": {rel: wrong}}}
    honest = {"a_synthetic_weld": {"source_sha256": {rel: raw_now}}}
    assert _drifted(_census(planted)) == [
        (f"a_synthetic_weld.source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {raw_now[:12]}")]
    assert _drifted(_census(honest)) == []

    # A DIGEST INSIDE A LIST. This record holds none today, which is exactly why
    # the control is here: the shape that hid 22 Triton pins must fail the moment
    # it appears, not the day someone notices it appeared.
    in_list = {"g": {"runs": [{"source_sha256": {rel: wrong}}]}}
    assert _drifted(_census(in_list)) == [
        (f"g.runs[0].source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {raw_now[:12]}")]

    # A MISSING FILE IS REPORTED, not skipped.
    assert _drifted(_census({"g": {"source_sha256": {"meep_gpu/nope.py": raw_now}}})) == [
        ("g.source_sha256.meep_gpu/nope.py", "meep_gpu/nope.py",
         "MISSING FROM THE TREE")]


def test_the_code_tier_is_the_clause_code_identity_licenses(tmp_path):
    """THE ARMED CONTROL FOR THE CODE TIER, in both directions, on a real file.

      1. A COMMENT-ONLY EDIT: the raw digest MOVES and the code digest HOLDS.
         This is the whole clause code_identity licenses and the only reason the
         two tiers are allowed to disagree.
      2. A REAL CODE EDIT: both move. The licence does not stretch to it.
      3. A PLANTED CODE DIGEST on an unedited file: reported.
      4. A CODE DIGEST ON A FILE WITH NO CODE IDENTITY: reported, never skipped.
    """
    victim = PACKAGE / "metal_kernels" / "shaders.py"
    rel = victim.relative_to(REPO).as_posix()
    source = victim.read_text(encoding="utf-8")
    raw_now = hashlib.sha256(victim.read_bytes()).hexdigest()
    code_now = code_digest(source)
    record = {"w": {"source_sha256": {rel: raw_now}, "code_sha256": {rel: code_now}}}

    commented = _plant(tmp_path / "commented", rel,
                       source + "\n# a comment appended, changing no execution\n")
    assert hashlib.sha256((commented / rel).read_bytes()).hexdigest() != raw_now
    assert len(_drifted(_of_tier(_census(record), walk.TIER_RAW), root=commented)) == 1
    assert _drifted(_of_tier(_census(record), walk.TIER_CODE), root=commented) == [], (
        "a comment-only edit was reported as code drift - the tier is stricter "
        "than code_identity licenses and will make doc fixes expensive")

    edited = _plant(tmp_path / "edited", rel,
                    source + "\nA_NEW_MODULE_CONSTANT = 1\n")
    assert len(_drifted(_of_tier(_census(record), walk.TIER_RAW), root=edited)) == 1
    assert len(_drifted(_of_tier(_census(record), walk.TIER_CODE), root=edited)) == 1, (
        "a new module-level constant did not move the code digest - the tier "
        "would license a real code change as a comment")

    reported = _drifted(_of_tier(_census({"w": {"code_sha256": {rel: "f" * 64}}}),
                                 walk.TIER_CODE))
    assert len(reported) == 1 and reported[0][1] == rel, reported

    non_python = {"w": {"code_sha256": {"meep_gpu/metal_kernels/fingerprints.json":
                                        "f" * 64}}}
    reported = _drifted(_of_tier(_census(non_python), walk.TIER_CODE))
    assert len(reported) == 1 and "no code identity" in reported[0][2], reported


def test_the_walker_finds_a_digest_planted_at_a_new_nesting_depth():
    """THE ARMED CONTROL FOR THE WALK ITSELF. Plant one deeper than anything real.

    Without this, a walker that stopped at depth 4 would report zero unclassified
    digests and a perfectly stable discovered count, because this record's deepest
    real digest sits at depth 4. Both halves are required: the walk must SURFACE
    the planted digest, and it must classify it UNCLASSIFIED. The second is the
    load-bearing one - a walker that found it and quietly excluded it would be the
    original defect wearing this test as cover.
    """
    digest = "a" * 64
    deep = {"an_entry": {"a": [{"b": {"c": [{"d": {"a_key_nobody_wrote": digest}}]}}]}}
    found = walk.census(deep)
    assert [f.value for f in found] == [digest], (
        "the walk did not surface a digest nested eight levels down through two "
        "lists - it is not recursing")
    planted = found[0]
    assert walk.dotted(planted.trail) == "an_entry.a[0].b.c[0].d.a_key_nobody_wrote"
    assert len(planted.trail) > max(len(f.trail) for f in _census()), (
        "the planted digest is no deeper than the record's own - the control has "
        "stopped testing anything the real walk does not already reach")
    assert planted.rule == walk.UNCLASSIFIED and planted.tier is None, (
        f"a digest under an unanticipated key was classified {planted.rule!r} "
        f"instead of failing - an unknown shape must be a red somebody writes a "
        f"rule for, never a silent exclusion")


# ---------------------------------------------------------------------------
# WHAT A WELD MUST CARRY
# ---------------------------------------------------------------------------


def _declares_a_weld_is_owed():
    """Families whose own module declares ``WELD_OWED`` non-empty, by name and reason.

    READ OUT OF THE SOURCE, not by import: this file runs on hosts with no Metal
    frontend, and a partition that could only be evaluated on a Mac would be a
    partition nobody checks. The declaration is the Metal spelling of the CUDA
    track's ``UNCERTIFIED_KERNELS`` - a family may stand unwelded only by SAYING
    it does, in the module a reader of the family opens, with the measurement that
    refused it.
    """
    owed = {}
    for path in sorted((PACKAGE / "metal_kernels").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)):
                target = node.targets[0].id
            if target != "WELD_OWED" or node.value is None:
                continue
            reason = ast.literal_eval(node.value)
            if reason:
                owed[path.stem] = reason
    return owed


def test_every_metal_family_is_welded():
    """All 30. cylindrical_real was the last, and its history is why this test changed.

    It refused until 2026-08-19 on a NON-DETERMINISTIC needle: prefix_wall_row_dropped
    read past the end of the prefix volume, so it was caught 5/8, then 7/8 on the same
    bytes, then 8/8 alone. Re-armed in bounds it is 8/8 on five consecutive runs. This
    test used to pin its ABSENCE; pinning the absence of a product is exactly the shape
    that goes red when the gap closes, so it now pins the completeness instead.

    THE ONE EXEMPTION IS A DECLARATION, ADDED 2026-09-05, AND IT IS A PARTITION.
    ``fused_hd_pair``'s gate runs to completion and does not release, so the family
    must NOT be welded - and until this change the only way to say so was to leave the
    test red, which makes an honest refusal indistinguishable from a forgotten weld.
    A gated family may stand unwelded only by declaring ``WELD_OWED`` in its own
    module, and the two sets must be DISJOINT and must EXHAUST the fleet: a declaring
    family that is welded anyway is as much a failure as an unwelded one that declares
    nothing.
    """
    welded = _welds()
    gates = {p.stem[len("gate_metal_"):] for p in
             (PACKAGE.parent / "parity" / "meep_gpu").glob("gate_metal_*.py")}
    named = {name[len("metal_"):-len("_device_gate")] for name in welded}
    owed = _declares_a_weld_is_owed()
    missing = sorted(gates - named - set(owed))
    assert not missing, (
        f"{len(missing)} Metal families have a gate but no weld: {missing}. A family "
        f"whose gate REFUSES must stay unwelded - fix the gate, do not weld it - but it "
        f"must then declare WELD_OWED in its own module rather than passing unnoticed.")
    both = sorted(set(owed) & named)
    assert not both, (
        f"{both} declare WELD_OWED and are welded anyway. The declaration says the "
        f"gate refused; the weld says it released. Delete whichever is false - never "
        f"both, and never neither.")
    stray = sorted(set(owed) - gates)
    assert not stray, (
        f"{stray} declare WELD_OWED but have no gate_metal_*.py at all. WELD_OWED is "
        f"what a family whose GATE refused says; a family with no gate owes a gate.")
    for family, reason in owed.items():
        assert len(reason) > 200 and "results/" in reason, (
            f"{family}'s WELD_OWED must name the artifact that refused and what a "
            f"release still owes, not merely that one is owed: {reason[:120]!r}")


@pytest.mark.parametrize("field", ["recorded_utc", "host", "artifact_sha256",
                                   "subnormal_policy"])
def test_every_metal_weld_carries_its_metadata(field):
    """The gap the CUDA comparison identified in certification.json - 0 of 4 blocks
    carried recorded_utc - is not reintroduced here."""
    missing = [name for name, entry in _welds().items() if not entry.get(field)]
    assert not missing, f"{len(missing)} Metal welds lack {field}: {sorted(missing)[:5]}"
