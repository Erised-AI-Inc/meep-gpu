"""A Triton weld is bound to the live bytes, and must CARRY what identifies them.

FOUR HALVES NOW, AND THE FOURTH IS THE WALK — added 2026-08-30, third round.
The denominator work below was right and was NOT ENOUGH, and the way it fell short
is the reason this round exists. Dropping the ``status`` filter fixed a filter.
What replaced it still walked a FIXED SHAPE — top-level entry, then
``source_sha256``, then a sibling ``<name>_sha256`` — so the same class of defect
was sitting one level deeper the whole time:

    22 ``source_sha256`` pins inside ``cylindrical_composition_gate.superseded_runs``
       were invisible, because the enumerator never descended into a LIST;
    183 ``code_sha256`` pins were compared by NOTHING at the merge bar, and the
       code tier is the one that licenses a comment-only edit — so an unchecked
       code tier is exactly how a stale record keeps looking current;
    68 ``device_sha256`` digests were likewise read by no laptop test;
    6 ``specialized_kernel_sources`` pins name their file by the PARENT KEY, a
       shape no rule in this file had a name for;
    4 more sat under ``serial_driver`` in a spelling nobody had anticipated.

A status filter became a shape filter and the class was unchanged. So the shape
is gone. :mod:`meep_gpu.weld_record_walk` recurses the whole record — dicts and
lists, at any depth — collects EVERY 64-hex digest with the trail it was found
at, and every one of them is either CHECKED against this tree in a named tier or
EXCLUDED BY A NAMED RULE that carries its reason. A digest matching neither is
``UNCLASSIFIED``, which is a failure, so a key spelling nobody anticipated arrives
as a red somebody has to write a rule for rather than as silence.

A FIFTH HALF, and it is the PREMISE — added 2026-08-30, fourth round. The walk
above made every digest either checked or excluded by a named rule, which was
right and was not the end of it: an exclusion is a DECISION NOT TO COMPARE, and
every one of them rests on a reason. Three of those reasons here are claims about
the world rather than about the record, and nothing evaluated any of them.

    ``host_sha256``            10 digests, excluded on the citation "already
        recomputed against the tree by test_triton_kernels.py::test_the_
        fingerprint_record_matches_the_shipped_host_side". That test recomputes
        each digest and then, on a mismatch, calls
        ``device_identity.weld_survives_edit`` and continues — the allowance this
        module refuses BY NAME in the paragraphs below, and the one that let two
        comment-block edits to ``launch.py`` leave twenty-one welds describing
        bytes that no longer shipped. A delegation landing on a weaker claim than
        the exclusion states is an unchecked digest wearing a citation. All ten
        are now CHECKED HERE at the raw tier and the citation is deleted rather
        than watched; measured after promotion, 10 of 10 match, so this buys no
        green — it removes the path by which a later edit would have bought one.
    ``staging_only_file``      16 digests the record names as staged on the device
        host and never committed: ``parity/meep_gpu/cases.py``. It was committed in
        0.9.1 with exactly those bytes, so the exclusion has expired as its premise
        said it would, and the premise test now compares each digest with the
        committed file instead of excusing it.
    ``out_of_tree_sibling_path``  16 digests naming launchers under
        ``<staging-root>/``. Same treatment.

``serial_driver_script``'s four digests key a BARE BASENAME and name no
directory, so the premise cannot be put to the record at all. That limit is
written into the rule's reason rather than covered with a looser predicate — a
predicate past what is built is how the last four defects were introduced.

MEASURED 2026-08-30, fourth round, over this record:

    discovered 940   checked 557   excluded 383   unclassified 0   drifted 13

and the count DISCOVERED is asserted, not only the count checked — so a digest
that stops being FOUND fails here too. :func:`test_the_walker_finds_a_digest_
planted_at_a_new_nesting_depth` plants one deeper than anything the record holds
and requires the walk to surface it as unclassified, so the green above is a fact
about the record rather than a fact about the walk having stopped early.

THREE HALVES, AND THE THIRD IS THE DENOMINATOR — added 2026-08-30, second
round. The two below were both strict over a set that was 179 of the record's 290
pinned pairs, because both enumerators filtered ``status == "PASS"`` and ten
entries carrying complete device-gate records do not have that key. See the block
comment at :data:`NOT_A_GATE_RECORD` for the measurement and the rule that
replaced it. The short version: an entry is checked if it PINS A FILE DIGEST, an
entry that is deliberately not a weld is excluded BY NAME, and the number of
entries and pins compared is itself asserted — so nothing leaves coverage by
omitting a key, which is the mechanism that would have caught this.

TWO HALVES, AND UNTIL 2026-08-30 ONLY ONE OF THEM WAS STRICT HERE.

The first half is "which bytes ran", and it is
``test_the_triton_welds_are_bound_to_the_live_sources`` below: raw sha256 equality
against the tree, with NO allowance, which is
``test_metal_weld_contract.py::test_the_metal_welds_are_bound_to_the_live_sources``
brought across unchanged in rule and in reason.

WHY IT IS HERE AND NOT LEFT TO ``test_dispatch_contract.py``, whose
``test_the_no_absorber_gate_records_are_welded_to_the_live_sources`` enumerates the
same welds and recomputes the same digests. That test does not make this claim,
because it carries a GENERAL fall-through::

    live = hashlib.sha256(path.read_bytes()).hexdigest()
    if live == digest:
        continue
    if weld_survives_edit(path, entry, name):     # <-- the allowance
        continue

``device_identity.weld_survives_edit`` admits any edit an AST-level ``code_sha256``
calls equivalent — comments and docstrings, for a file carrying no device source.
MEASURED 2026-08-30: that is precisely why the deposit-carry round's two
COMMENT-BLOCK edits to ``triton_kernels/launch.py`` — pinned by 28 of the 30 PASS
welds — left twenty-one arm-level welds describing bytes that no longer shipped
while the whole suite stayed green. The allowance did its job as written; the
defect is that the job is the wrong one. A weld says "this gate ran THESE bytes on
a device". Whether an edit is harmless is a judgement made on the HOST about a
claim only the DEVICE can settle, and a general allowance can license the very
edit it is asked to judge — including an edit to the module that implements the
allowance, which ``device_identity.py`` is and which no weld pins. So the rule
here is the raw bytes and nothing else, and the twenty-one were fixed the only
honest way: re-run on the GPU host, rebind with ``rebind_triton_welds.py``, then
assert. See the campaign note at the foot of this file.

The second half is what a weld must CARRY, and it is the Metal track's other
guarantee: ``test_metal_weld_contract.py`` requires ``recorded_utc``, ``host``,
``artifact_sha256`` and ``subnormal_policy`` on every weld, and pins a gate that
REFUSED so it cannot be welded. Neither requirement existed on the Triton side.

MEASURED 2026-08-19, before any of this was written, against all ten Triton welds:

    field              missing
    recorded_utc         0/10
    host                 0/10
    artifact_sha256      0/10
    subnormal_policy     0/10

So the bare Metal requirement carries ZERO debt here and is asserted flat. The
debt appears one level down, where a field is present but does not answer its own
question: ``triton_complex_offdiag_device_gate`` records
``subnormal_policy: "see artifact"``, and the artifact it points at
(``weld_recut_2026-08-19/complex_offdiag.json``, digest confirmed against the
recorded ``artifact_sha256``) contains no policy stamp at all — only a source
digest for ``subnormal_policy.py``. Its nine siblings all name ``keep``. That one
is declared as debt below and may only shrink.

WHY THE VALUE CHECKS AND NOT ONLY PRESENCE. A weld exists so a later reader can
say what a device result covers. "Present" is not that: a policy that points
elsewhere, a host that names no Triton, or a timestamp that does not parse each
satisfy a truthiness test and answer nothing. The float32 subnormal policy is the
sharpest case on this track — every family in this record re-cuts under a
different one — so a weld that does not name the policy it ran under is a weld
whose bytes cannot be reproduced.

A SIXTH HALF, AND IT IS A LEVEL RATHER THAN A SHAPE — 0.9.1. Every requirement in
the paragraph above used to be asked of the ENTRY, because an entry held exactly
one run's facts beside its digests. It now holds ONE RECORD PER COMPUTE
CAPABILITY under ``runs``: the run fields moved into ``runs[<cc>]``, each record
carrying ``bound_sha256`` — the digest of the digests that run certified — and the
digests, ``status``, ``legs`` and the curated claims stayed at entry level. The
reason is additivity: a second architecture's round must be certified BESIDE the
first rather than overwrite it, and a record that outlives the bytes it certified
must expire on its own instead of leaving every architecture admitted. See
:mod:`meep_gpu.fastpath` (``RUNS``, ``RUN_FIELDS``, ``bound_digest``,
``live_capabilities``, ``retired_shape_reasons``).

SO THE DENOMINATOR HERE IS NOW ``(weld, capability)``, and the level a field is
read at is DERIVED from ``fastpath.RUN_FIELDS`` rather than typed twice below — a
field that moves level in ``fastpath.py`` goes red here instead of being read from
the level it used to live at, which is the retired-shape defect wearing this
file's own green. MEASURED 2026-10-01: 55 PASS welds, 55 run records, all keyed
``8.6``; every one carries all four required fields, names a validated Triton and
names ITS OWN key. A weld with NO record is reported rather than iterated over
zero times, which is the way a migration could have emptied this tier silently.

``validated_compute_capabilities`` LEFT THE RECORD in the same change, and that is
why the capability half of
:func:`test_every_triton_weld_names_a_validated_triton_and_capability` no longer
reads a declared list. It was one hand-typed key that no tool wrote, so a round on
another architecture moved every weld's evidence and left the declaration saying
what it had always said; it is now DERIVED
(``fastpath.validated_compute_capabilities``) from the capabilities each cited weld
has a live record for. Each record is therefore compared against ITS OWN KEY, which
is strictly stronger than membership of any list.

Stdlib, pytest, and three in-package modules — the record walk, the two identity
helpers, and :mod:`meep_gpu.fastpath` for where a run record lives and what may be
in it. Still a laptop merge-bar test: it touches no device.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import pathlib
import re

import pytest

from . import fastpath
from . import weld_record_walk as walk
from .code_identity import code_digest
from .device_identity import device_digests

PACKAGE = pathlib.Path(__file__).resolve().parent
RECORD = PACKAGE / "triton_kernels" / "fingerprints.json"
#: the repository root, the root every welded path in the record is relative to.
REPO = PACKAGE.parent

#: Non-vacuity floor. TEN welds measured 2026-08-19; the floor sits below that so
#: one deliberate retirement does not pressure anyone into editing the record,
#: and far enough above zero that a changed ``status``/``source_sha256`` shape
#: fails here instead of quietly making every assertion below enforce nothing.
#: Independent of — and stricter than — the floor of 3 in test_dispatch_contract.
WELD_FLOOR = 8

#: The four the Metal track requires. Same names, same track-independent reason.
#:
#: ALL FOUR ARE RUN FIELDS, so since 0.9.1 they are read from ``runs[<cc>]`` and not
#: from the entry. The level is not typed here — :func:`_weld_runs` reads whatever
#: ``fastpath.RUN_FIELDS`` calls a run field, and the subset relation is ASSERTED in
#: the metadata test — because a hand-typed level is a second place to be wrong and
#: would read a field from where it used to live while the ledger moved it.
REQUIRED_METADATA = ("recorded_utc", "host", "artifact_sha256",
                     "subnormal_policy")

#: The keys this table's arms cite, which is the denominator of the live-capability
#: test below: 44 measured 2026-10-01 over ``fastpath.ARM_CERTIFICATION``'s 75 arms
#: (many arms share a gate). A FLOOR, so an arm map that collapsed to one key fails
#: here rather than passing a one-element loop.
CITED_KEY_FLOOR = 44

#: One run record per PASS weld per architecture it ran on. 55 welds x one ``8.6``
#: record, measured 2026-10-01. The per-weld denominator the metadata tier used to
#: have was 55 entries; this is what replaced it, and it is asserted for the same
#: reason the pin floors are: a tier that stops being walked reports nothing.
WELD_RUN_RECORD_FLOOR = 55

#: DECLARED DEBT: welds whose ``subnormal_policy`` does not NAME a policy.
#: ONE, measured 2026-08-19. This set may only SHRINK — adding a name is a
#: visible edit here, which is the point; a new weld that cannot say what policy
#: it was cut under should fail rather than be admitted by widening this.
POLICY_UNNAMED_BUDGET = {
    "triton_complex_offdiag_device_gate":
        "records 'see artifact', and complex_offdiag.json carries no policy "
        "stamp — re-run the gate under a stamped policy, or transcribe the "
        "policy it actually ran under; do not widen this budget",
    # THE THREE MIXED-POLICY WELDS, added 2026-08-19. These are a DIFFERENT
    # debt from the one above and are named individually rather than waved
    # through: they do not merely fail to record a policy, they ran WITHOUT one.
    #
    # Measured on the GPU host with MEEP_GPU_SUBNORMAL_POLICY=keep exported but
    # install_ftz_strip never called (operand 0x00004000, x * 1.0):
    #     host -> 0x4000 KEPT        cupy -> 0x0 FLUSHED
    #     strip counters {calls: 0, removed: 0}
    # so the CuPy ORACLE was flushing while Triton natively keeps. The byte
    # identity each gate recorded still holds for that configuration; what is
    # withheld is any claim about WHICH policy it holds under. Clearing this
    # debt means making the gate call install_ftz_strip and re-running it —
    # not editing the field.
    "triton_cylindrical_device_gate":
        "gate never calls install_ftz_strip: mixed-policy comparison, see _mixed_policy",
    "triton_fused_electric_device_gate":
        "gate never calls install_ftz_strip: mixed-policy comparison, see _mixed_policy",
    "triton_no_pml_device_gate":
        "gate never calls install_ftz_strip: mixed-policy comparison, see _mixed_policy",
}

_POLICY_WORD = re.compile(r"\b(keep|flush)\b", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _welds(record=None):
    """The PASS subset of :func:`_pinned` — the METADATA tier, not the byte tier.

    ENUMERATED, NOT LISTED, for the reason already written down twice in this
    tree: a hand-maintained list is a second place to forget a weld, and the six
    welds added on 2026-08-19 were orphaned by exactly that. The record IS the
    list.

    THIS SET IS NOT THE DENOMINATOR OF THE BYTE CHECK any more, and that change
    is the point of the 2026-08-30 round — see :func:`_pinned`. It remains the
    denominator of the metadata requirements below (``artifact_sha256``,
    ``subnormal_policy``, a parsable ``recorded_utc``, a validated toolchain),
    because those are claims a PASS makes and a refusal does not. Which pinned
    entries are deliberately outside it is declared by name in
    :data:`PINNED_BUT_NOT_A_WELD` and pinned in both directions below, so the
    split is a written decision rather than a missing key.
    """
    record = _record() if record is None else record
    return {name: entry for name, entry in record.items()
            if isinstance(entry, dict) and entry.get("status") == "PASS"
            and entry.get("source_sha256")}


def _weld_runs(record=None, welded=None):
    """``[(weld, capability, run)]`` and the welds that carry NO record at all.

    THE METADATA DENOMINATOR SINCE 0.9.1. Each requirement below used to be asked of
    the entry; it is now asked of every per-capability record the entry holds, so a
    weld certified on two architectures answers twice and one certified on neither
    answers never.

    THE RECORDLESS LIST IS RETURNED RATHER THAN SKIPPED, and it is the half that
    makes the rest mean anything: a loop over ``runs`` is empty for an entry whose
    records were lost, dropped, or never written, so a migration that emptied this
    container would have turned every assertion below into a pass. Every caller
    asserts the list is empty and the row count clears
    :data:`WELD_RUN_RECORD_FLOOR`, which is the same mechanism as the pin floors:
    coverage does not fall silently.

    A record that is not a mapping is reported as recordless rather than read, for
    the reason ``retired_shape_reasons`` gives: a shape read two ways makes "which
    bytes did this run certify" a question with two answers.
    """
    welded = _welds(record) if welded is None else welded
    rows, recordless = [], []
    for name, entry in sorted(welded.items()):
        runs = entry.get(fastpath.RUNS)
        if not isinstance(runs, dict) or not runs:
            recordless.append(f"{name}: {fastpath.RUNS} is {runs!r}")
            continue
        for capability, run in sorted(runs.items()):
            if not isinstance(run, dict):
                recordless.append(f"{name}[{capability}] is {type(run).__name__}, "
                                  f"not a record")
                continue
            rows.append((name, capability, run))
    return rows, recordless


def _assert_the_run_tier_is_walkable(rows, recordless):
    """The two non-vacuity legs every per-record requirement below shares."""
    assert not recordless, (
        f"{len(recordless)} PASS weld(s) carry no per-capability run record: "
        f"{recordless[:8]} — a weld with no record certifies no architecture, and "
        f"every requirement in this tier would iterate over nothing. Re-run its "
        f"gate and bind it with parity/meep_gpu/rebind_triton_welds.py")
    assert len(rows) >= WELD_RUN_RECORD_FLOOR, (
        f"only {len(rows)} (weld, capability) run records to check, was "
        f"{WELD_RUN_RECORD_FLOOR} — the tier SHRANK; a weld does not leave the "
        f"metadata requirements by losing its record")


# ---------------------------------------------------------------------------
# WHAT MAKES AN ENTRY CHECKABLE — THE RULE, STATED ONCE
# ---------------------------------------------------------------------------
#
# AN ENTRY IS CHECKED IF IT PINS A FILE DIGEST. Nothing else is consulted: not
# ``status``, not ``host``, not whether an artifact was recorded, not whether the
# run passed. A recorded digest for a path is a CLAIM ABOUT THIS TREE — "the run
# this entry describes executed these bytes" — and a claim about the tree can be
# checked against the tree whatever the run concluded.
#
# WHY NOT ``status == "PASS"``, WHICH IS WHAT THIS FILE USED UNTIL 2026-08-30.
# MEASURED, by recomputing every digest in the record independently of these
# tests rather than by reading them:
#
#     status=PASS   welds=31  pairs=179  drifted= 0
#     status absent welds=10  pairs= 88  drifted=31   <-- CHECKED BY NOTHING
#
# Those ten entries are not stubs. Each carries a complete device-gate record —
# ``host`` "the GPU host, NVIDIA RTX A6000, compute capability 8.6", an
# ``artifact_sha256``, a ``records`` line, and a pinned source map — and eight of
# them pin ``triton_kernels/launch.py``, the very file the weld-close campaign
# existed to rebind. The check above read as "every weld is bound to the live
# bytes" and meant "179 of the record's 267 pinned pairs are". A strict assertion
# over a filtered subset is a weak check wearing a strong one's clothes, and the
# filter here was a key that a gate record is under no obligation to carry.
#
# THE COROLLARY, and it is the mechanism rather than the rule:
# ``test_every_record_entry_is_checked_or_excluded_by_name`` requires every dict
# entry in the record to be either CHECKED or NAMED in
# :data:`NOT_A_GATE_RECORD` with a reason. So a future entry cannot leave
# coverage by omitting a key — omitting ``source_sha256`` moves it into the
# named-exclusion set, where it has to be written down and justified. Absence of
# a field never again decides what gets checked.

#: What ``rebind_triton_welds.py`` has not bound yet. A curated path may be added
#: to a weld's ``source_sha256`` with this value to DECLARE that the weld must
#: cover it; the digest is then supplied by the rebind tool from the gate's own
#: ``imported_source_sha256``, never typed here. It is a failure until then, and
#: it is reported as UNBOUND rather than as drift so the next reader is not sent
#: to a device for something a rebind clears.
UNBOUND_PLACEHOLDER = "0" * 64

#: Dict entries that pin NO file digest, each excluded BY NAME with the reason.
#: This is the whole exclusion list: there is no predicate that quietly drops an
#: entry, so anything not here and not checked fails.
NOT_A_GATE_RECORD = {
    "_tranche_boundaries":
        "prose: which families each certification tranche covered",
    "guard":
        "prose: how ENABLE_FP_FUSION reaches the JIT launch; the VALUE is "
        "checked against kernels.py by test_triton_kernels.py::"
        "test_the_recorded_guard_is_the_launch_route_and_the_value_is_false",
    # ``host_sha256`` LEFT THIS LIST on 2026-08-30, FOURTH round, and its old
    # reason is the finding. It read: "already recomputed against the tree by
    # test_triton_kernels.py::test_the_fingerprint_record_matches_the_shipped_
    # host_side — checking it twice is two places to disagree." The citation is
    # real and the claim it lands on is NOT THE SAME CLAIM. That test recomputes
    # each host digest and then, on a mismatch, calls
    # ``device_identity.weld_survives_edit`` and continues — the allowance this
    # module refuses by name in its own docstring, and the one that let two
    # comment-block edits to ``launch.py`` leave twenty-one welds describing
    # bytes that no longer shipped while the suite stayed green. So ten digests
    # sat behind an exclusion that promised a strict comparison and delegated to
    # a lax one. They are now CHECKED HERE at the raw tier, the delegation is
    # deleted rather than watched, and the entry moves to PINNED_BUT_NOT_A_WELD.
    # Measured after promotion: 10 of 10 match, so this buys no green — it
    # removes a path by which a future edit would have bought one.
    # ``specialized_kernel_sources`` LEFT THIS LIST on 2026-08-30, third round,
    # and how it left is the point. Its exclusion said "not a digest map: each
    # value is a per-module block" — true, and it meant the six digests inside
    # those blocks were checked here by nothing. They name their file by the
    # PARENT KEY (``conductivity.py: {kernel, sha256}``), a shape the sibling rule
    # could not see. The structural walk surfaced all six; they are now compared
    # raw, and all six match. An exclusion whose reason was a description of the
    # shape rather than an argument about the claim is the same defect one level
    # down, so this one is recorded rather than quietly deleted.
    "subnormal_policy":
        "prose: which policy the track is certified under and what is impure "
        "about that label; carries no path and no digest",
    # ARRIVED 0.9.1 with the per-capability migration. A BLOCK, not a string —
    # which is why it is named here rather than left to the "not a dict" reading
    # the rest of this list used to be able to assume.
    "_capability_runs_migration":
        "the migration's own stamp: which tool moved the run fields under "
        "`runs[<cc>]`, when, where each entry's capability was read from, and how "
        "many entries it touched. It records a COUNT and a capability list and "
        "deliberately no digest, so there is nothing here to compare against the "
        "tree; that absence is asserted by test_capability_records.py::"
        "test_the_migration_is_recorded_in_both_ledgers_and_states_no_digest",
    "not_built_this_round":
        "prose: the PTX census of products this round did not build",
    # REWRITTEN 2026-09-07, for the reason the block comment below gives about the
    # next three: the old reason said this entry "names files but records no digest
    # for them, which is what makes it a debt note". That became FALSE the moment a
    # wiring round populated `files`, and a false reason tells the next reader to stop
    # looking. The entry is still excluded, on the true reason.
    # REWRITTEN AGAIN 2026-09-19, the same defect a second time: the 2026-09-14
    # reason opened "`files` CARRIES A DEBT TODAY" and said the debt rule "EXISTS
    # AGAIN", and both went false when the 2026-09-17 runs discharged the debt.
    # Measured 2026-09-19: `files` == {}, host_sha256["launch.py"] == the tree's
    # b4ffd899, and the rule fired on 0 of the three walked records.
    "pending_host_recut":
        "the standing debt list and what clears it. `files` is EMPTY today. The latest "
        "debt it carried was triton_kernels/launch.py, declared here on 2026-09-14 "
        "when the 2026-09-13 narrowing of specialized_family_owns_the_grid moved the "
        "file, and discharged on 2026-09-17 on a RUN: the Triton product-fleet re-gate "
        "(the fleet and tranche rebind) plus the composition-leg recut behind a staged "
        "manifest, which moved host_sha256[launch.py] to the tree's bytes and emptied "
        "`files` (parity/meep_gpu/results/round_2026-09-14_weldgrid_drivers/"
        "ROUND_NOTES.md:1058-1081). Before it, the 2026-09-07 cylindrical H->D wiring "
        "round declared launch.py here and discharged it by running the composition "
        "legs: running the device gates is the only thing that clears one. While a "
        "round HAS a debt declared, "
        "`files.<host module>` carries a {certified, live} pair whose polarity is "
        "inverted (`certified` is the superseded digest and must NOT match the tree, "
        "which is what makes it a debt), so those digests want an exclusion rule of "
        "their own in weld_record_walk -- written in the change that declares the debt "
        "and deleted with it, because a rule nothing exercises is where the next "
        "omission hides. The 2026-09-14 declaration's rule, `declared_host_debt_pair`, "
        "was deleted on 2026-09-19, once both walk tests reported it firing on none of "
        "the three records; until a future declaration writes its own, a declared pair "
        "walks as UNCLASSIFIED and fails test_every_discovered_digest_is_checked_or_"
        "excluded_by_a_named_rule below. Either way they are checked in both "
        "directions, against the tree and against host_sha256, by "
        "test_triton_planner_composition.test_every_drifted_host_file_"
        "is_named_in_the_pending_recut_record.",
    # THE NEXT THREE REASONS WERE REWRITTEN ON 2026-08-30, third round, because
    # the structural walk REFUTED what they said. Each claimed the entry pinned
    # nothing checkable; each pins something. The digests were excluded anyway —
    # correctly, for reasons that are now the true ones — but a reason that is
    # false is worse than a reason that is narrow: it tells the next reader to
    # stop looking. This is the same defect as the status filter, in prose.
    "family_recert_2026-08-14":
        "a recert LOG index. Its 64 digests are HISTORY, not results-tree files "
        "as this reason used to claim: 45 are source_sha256_at_recert snapshots "
        "of what each family gate imported when it was re-certified, and 19 more "
        "are a source_drift_since_recert log whose certified/live_at_declaration "
        "pairs are stale by construction. Excluded by the walker's "
        "historical_recert_snapshot and historical_recut_log rules; this entry is "
        "the one carrying history with no live claim of its own, which is why it "
        "must be named here — see "
        "test_a_historical_exclusion_is_earned_by_a_live_checked_claim",
    "planner_integration_recert_2026-08-13":
        "a run narrative (GPU census, staging, tenancy). It DOES pin two paths, "
        "contrary to what this reason said until 2026-08-30: serial_driver names "
        "chain.sh and run_gate.sh, the shell drivers that serialised the sweep "
        "and enforced empty-device placement. Both live only under the gitignored "
        "results tree — they were never committed — so a clean checkout has "
        "nothing to compare; excluded by the walker's serial_driver_script rule",
    "planner_integration_recert_2026-08-14":
        "as above: a run narrative that pins chain.sh and run_gate.sh, neither "
        "committed to the tree",
}

#: Entries that ARE checked for bytes but are deliberately NOT welds, so the
#: metadata tier does not apply to them. Named, with the reason, for the same
#: cause as everything else here: leaving them out by the absence of ``status``
#: is what hid them from the byte check for eleven days.
PINNED_BUT_NOT_A_WELD = {
    "no_pml_composition_gate":
        "recert_attempt_2026_08_14 records rc=1 and "
        "failure_measured_on_device: a REFUSAL, and welding a refusal makes the "
        "record lie — see test_no_triton_weld_claims_a_gate_that_refused_on_device",
    "conductivity_composition_gate":
        "recert_attempt_2026_08_14 records rc=1 and failure_measured_on_device",
    "cylindrical_composition_gate":
        "a COMPOSITION gate: it certifies which plans plan_step selects, not a "
        "kernel's bytes, so it carries no per-arm weld metadata",
    "symmetry_composition_gate": "a composition gate, as above",
    "dispersive_composition_gate": "a composition gate, as above",
    "source_seam_gate":
        "a composition gate over the fused-pair source partition, as above",
    "dispersive_fused_pair_gate":
        "the PRE-2026-08-19 spelling of a product gate: bare-basename pins and "
        "no subnormal_policy field; superseded as a weld by "
        "triton_dispersive_fused_pair_device_gate, and kept because its "
        "composition evidence is not carried there",
    "fused_ade_state_gate":
        "the pre-2026-08-19 spelling, superseded as a weld by "
        "triton_fused_ade_state_device_gate",
    "fused_electric_gate":
        "the pre-2026-08-19 spelling, superseded as a weld by "
        "triton_fused_electric_device_gate",
    "driver_dispatch":
        "the dispatch WIRING record — which consult sites exist and what the "
        "kill switch does — not a device run; it has no host and no artifact "
        "because no gate produced it. Its verdict is its route runs', filed "
        "under runs[<cc>], and the record is checked through them by "
        "test_triton_weld_contract.py::"
        "test_the_dispatch_record_stands_on_a_live_passing_route_run",
    "bit_identity_gate":
        "pins its harness (probe_sha256, adapter_sha256) rather than a package "
        "source map; its verdict shape is pinned by test_triton_kernels.py::"
        "test_the_recorded_gate_verdicts_are_pass_shaped",
    "specialized_kernel_sources":
        "the per-module SPECIALIZATION index — which kernel each module "
        "specialises and the sha256 of the module that holds it. Its six digests "
        "became checked on 2026-08-30 when the structural walk found the "
        "parent-key-named shape, but it is not a device run: no host, no "
        "artifact, no policy, because no gate produced it",
    "host_sha256":
        "the shipped HOST SIDE — launch.py, coverage.py, the per-family modules "
        "that pick the Yee sub-lattice and bind the coefficient pointers. Its ten "
        "digests became checked on 2026-08-30 (fourth round) when the delegation "
        "to test_triton_kernels.py was found to land on a weaker claim than the "
        "exclusion stated. It is a map, not a device run: no host, no artifact, "
        "no policy, so the metadata tier cannot apply to it",
}

#: Non-vacuity floors for the CHECKED DENOMINATOR, measured 2026-08-30 by
#: independent recomputation. These are the mechanism that would have caught the
#: status-filter defect: they fail when coverage SHRINKS, so an entry cannot slip
#: out by dropping a key, and a record whose shape changes fails here rather than
#: silently making every assertion below enforce nothing.
#: 31 PASS + 10 status-less + bit_identity_gate + specialized_kernel_sources, the
#: last of which the structural walk added on 2026-08-30. Counts DICT entries; the
#: record's one top-level ``kernel_source``/``kernel_source_sha256`` pair is a
#: checked digest with no owning block and is counted in the tier floors instead.
#: 44 since 2026-08-30 fourth round: ``host_sha256`` joined the checked set when
#: its delegation was deleted.
CHECKED_ENTRY_FLOOR = 44
SOURCE_PIN_FLOOR = 268          # (entry, path) pairs from live source_sha256 maps
#: <name>/<name>_sha256 pairs naming an in-tree file. 22 until release 0.9.1
#: (24c9b47), which removed the eight slurm_launcher/slurm_launcher_sha256 pairs:
#: they named run_triton_*.slurm batch launchers that this repository does not
#: track. 14 resolve in-tree on the shipped record (measured 2026-10-05), and a
#: record that loses one more fails here.
SIBLING_PIN_FLOOR = 14

#: THE TOTAL, and it is the floor the other three cannot substitute for. Those
#: count what is CHECKED; this counts what the walk FINDS, so a digest that stops
#: being discovered — moved under a key the classifier is not asked about, or
#: respelled out of 64-hex — fails here rather than shrinking the denominator
#: quietly. Measured 2026-08-30 (third round) by recursing the record.
DISCOVERED_FLOOR = 940
CHECKED_FLOOR = 557             # 547 + the 10 host digests, promoted 2026-08-30

#: Per-tier floors. Split because the tiers have different rules and different
#: remedies, and one mixed count would let a collapse in the code tier be masked
#: by growth in the raw tier.
RAW_TIER_FLOOR = 306            # 268 source-map + 22 sibling + 6 parent-key + 10 host
CODE_TIER_FLOOR = 183           # code_sha256 maps, under the merge bar since today
DEVICE_TIER_FLOOR = 68          # device_sha256 kernel digests, likewise


def _resolve(name, root=REPO):
    """A pinned path, under BOTH spellings this record uses.

    The post-2026-08-19 spelling is repo-relative (``meep_gpu/triton_kernels/
    launch.py``) and needs nothing. The pre-2026-08-19 entries pin a BARE
    BASENAME (``launch.py``, ``driver.py``), which is why dropping the status
    filter alone is not enough: ``REPO / "launch.py"`` does not exist, so all 88
    pairs would have reported MISSING FROM THE TREE — a red for the wrong reason,
    which is indistinguishable from a red for the right one and would have been
    cleared by deleting the check.

    THE RULE: a bare name means the record's own directory, then the package root.
    That order is not a preference, it is what the record says — the two
    directories share only ``__init__.py``, and the digests the entries carry for
    it (017f710e0002, and 8ae115cef0e6 from an older cut) are the
    ``triton_kernels`` file, never ``meep_gpu/__init__.py`` (ccf73f8cdbee). No
    digest is consulted to resolve a path; the first EXISTING candidate wins, so
    resolution cannot be steered by what would make the comparison pass.
    """
    if name.startswith("/"):
        # An absolute pin. Rewritten to repo-relative when it lies inside THIS
        # checkout — the same file under another name, never a different one —
        # and left alone otherwise so it reports MISSING rather than resolving to
        # something on whichever machine happens to run this.
        absolute = pathlib.Path(name)
        try:
            return root / absolute.relative_to(REPO)
        except ValueError:
            return absolute
    if "/" in name:
        return root / name
    for base in ("meep_gpu/triton_kernels", "meep_gpu"):
        candidate = root / base / name
        if candidate.is_file():
            return candidate
    return root / "meep_gpu" / "triton_kernels" / name


def _census(record=None):
    """EVERY digest in the record, classified. The denominator, structurally derived.

    THE ONE ENUMERATOR. Nothing in this file walks a shape any more: no
    ``record.items()`` loop that assumes a top level, no ``entry["source_sha256"]``
    that assumes a key, no recursion that descends dicts and stops at lists. What
    the record holds is what gets classified, and a digest the classifier has no
    rule for is ``UNCLASSIFIED`` — a failure, not a silence.
    """
    return walk.census(_record() if record is None else record)


def _checked(found):
    return [f for f in found if f.tier is not None]


def _of_tier(found, tier):
    return [f for f in found if f.tier == tier]


def _pinned(record=None):
    """Every top-level entry that owns at least one CHECKED digest.

    Derived from the census rather than from the presence of a ``source_sha256``
    key, so an entry whose only pins live in a ``code_sha256`` map, a
    ``device_sha256`` block, or a shape nobody has invented yet is in the
    denominator on the same terms as everything else.
    """
    record = _record() if record is None else record
    owners = {f.trail[0] for f in _checked(_census(record))}
    return {name: entry for name, entry in record.items() if name in owners
            and isinstance(entry, dict)}


def _source_pins(record=None):
    """``(owner, path, digest)`` for the LIVE ``source_sha256`` maps only.

    Kept as its own view because :data:`SOURCE_PIN_FLOOR` is about this shape
    specifically — it is the count the status filter was hiding, and a floor that
    silently absorbed the code and device tiers would stop measuring the thing it
    was written to measure.

    ``owner`` is the top-level entry name, not the trail, because
    ``test_dispatch_contract.py`` imports this and indexes the record by it. The
    full trail is what :func:`walk.compare` reports on a failure, which is where
    it is actually needed.
    """
    return [(f.trail[0], f.subject, f.value)
            for f in _census(record)
            if f.rule == walk.TIER_RAW + ":source_digest_map"]


def _sibling_pins(record=None):
    """``(trail, path, digest)`` for every ``<name>``/``<name>_sha256`` pair.

    A SECOND UNCHECKED DENOMINATOR, found the same way as the first: the record
    pins the gate script beside the sources under ``probe_sha256``,
    ``adapter_sha256``, ``slurm_launcher_sha256`` and friends, and nothing
    recomputed any of them. MEASURED 2026-08-30: 22 such pins resolve to files in
    this checkout and 6 had drifted, including both of ``bit_identity_gate``'s —
    an entry that pins no ``source_sha256`` at all and was therefore invisible to
    every enumerator in the suite.

    Nested blocks count: ``fused_electric_gate.throughput_gate.probe_sha256``
    names a real script and is checked like any other. Pins naming a path outside
    the checkout (``<staging-root>/...``) or under the gitignored results
    tree are excluded by the walker's ``out_of_tree_sibling_path`` rule, with the
    reason, rather than by a prefix test buried here.
    """
    return [(walk.dotted(f.trail), f.subject, f.value)
            for f in _census(record)
            if f.rule == walk.TIER_RAW + ":sibling_script_pin"]


def _records_a_device_failure(entry):
    """Recorded evidence that a run of this gate exited non-zero, nested at any depth.

    Mechanical rather than a name list, so a gate that is re-run clean and has its
    record updated leaves this set on its own — and one that refuses cannot be
    welded by adding a ``status`` key while the failure still stands beside it.
    """
    found = []
    rc = entry.get("rc")
    if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
        found.append(("rc", rc))
    if entry.get("failure_measured_on_device"):
        found.append(("failure_measured_on_device", True))
    for value in entry.values():
        if isinstance(value, dict):
            found.extend(_records_a_device_failure(value))
    return found


def _refused(record=None):
    record = _record() if record is None else record
    return {name: _records_a_device_failure(entry)
            for name, entry in record.items()
            if isinstance(entry, dict) and _records_a_device_failure(entry)}


def test_the_triton_weld_set_is_still_enumerable():
    welded = _welds()
    assert len(welded) >= WELD_FLOOR, (
        f"only {len(welded)} Triton welds discovered — the status/source_sha256 "
        f"shape changed and every requirement in this module would now enforce "
        f"nothing")


def _raw_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest_of(path):
    """``None`` for a file with no code identity, which :func:`walk.compare` reports.

    ``code_digest`` parses Python and nothing else, so a ``code_sha256`` recorded
    for a shader, a JSON or a ``.cu`` is a digest nothing can recompute. Returning
    ``None`` rather than falling back to a byte hash keeps that a REPORT instead of
    a silently-passing comparison — the difference between a rule and a hole.
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

    Takes the CENSUS rather than a pin list, so raw bytes, code identity and
    device-kernel identity all run through one comparison and there is a single
    place where "does the recorded digest equal what the tree says" is decided.
    The tier decides WHICH question is asked; nothing decides whether to ask.

    Factored out for one reason: an assertion nobody has watched FAIL is a
    measurement that cannot fail. ``test_the_drift_check_refuses_a_planted_digest``
    below drives this same function with a deliberately wrong digest in each tier
    and requires it to report one, so the green result above is a fact about the
    tree rather than a fact about the comparison being inert.

    There is deliberately NO ``weld_survives_edit`` here and no other fall-through
    to a laxer digest — see this module's docstring for the twenty-one welds that
    cost. The code tier is not such a fall-through: it is compared to the LIVE
    FILE'S CODE, never consulted to excuse a raw-byte mismatch.

    An UNBOUND PLACEHOLDER is reported as its own reason rather than as drift: the
    pin is a DECLARATION that a weld must cover the path, waiting for
    ``rebind_triton_welds.py`` to supply the digest from the gate's own
    ``imported_source_sha256``. It is still a failure — a weld carrying an unbound
    pin certifies nothing about that file — but the remedy is a rebind, not a
    device run, and saying so is the difference between a red someone can clear
    and a red someone deletes.
    """
    return walk.compare(
        found,
        resolve=lambda name: _resolve(name, root),
        raw_digest=_raw_digest,
        code_digest_of=_code_digest_of,
        device_digests_of=_device_digests_of,
        unbound_placeholder=UNBOUND_PLACEHOLDER,
        rebind_tool="parity/meep_gpu/rebind_triton_welds.py",
    )


def _measured_drift(name, entry, root=REPO):
    """The set of paths this entry pins in the RAW tier that no longer match the tree.

    Derived from the census over ``{name: entry}`` rather than from
    ``entry["source_sha256"]``, so a raw pin the entry grows in some other shape is
    measured here too. The same measurement, under the same name, as the CUDA
    contract's, so the route-record rule below reads drift the same way in all three.
    """
    raw = _of_tier(_census({name: entry}), walk.TIER_RAW)
    return {subject for _, subject, _ in _drifted(raw, root=root)}


def test_the_triton_welds_are_bound_to_the_live_sources():
    """The Metal rule, on the WHOLE Triton record. Raw bytes, no allowance.

    THE DENOMINATOR IS EVERY RAW-TIER DIGEST THE WALK FINDS — 296 across 44
    entries as measured on 2026-08-30 (third round), against 290 in the second
    round and 179 across 31 before that. The six it gained are the
    ``specialized_kernel_sources`` pins, which name their file by the PARENT KEY
    and so were invisible to a walker that only knew the sibling shape.

    THE RULE IS THE RAW BYTES, and if a specific entry ever needs an exception it
    is named HERE, as one entry, with the reason — never as a predicate that
    decides a class of edits harmless. A drifted pin is fixed by RE-RUNNING its
    gate; it is never fixed by relaxing this comparison, by adding a KNOWN_DRIFT
    set, or by typing a digest in.

    GREEN, 0 of 314, 2026-09-01. It was 13 across 3 on the morning of
    2026-08-30, 5 across 2 that evening, and 2 across 1 (``bit_identity_gate``)
    through 2026-08-31; ``dispersive_composition_gate`` cleared on 2026-08-31
    (see below) and ``bit_identity_gate`` cleared on 2026-09-01, by exactly the
    remedy the standing paragraph here prescribed — SOMEBODY WROTE THE BINDER:

      * ``bit_identity_gate`` (was 2: ``adapter_sha256`` ->
        ``parity/meep_gpu/track_triton_pml.py``, ``probe_sha256`` ->
        ``parity/meep_gpu/probe_fused_kernel_bit_identity.py``). The blocker had
        been structural — the entry carries ``status``-less sibling pins, no
        ``source_sha256`` map and (until now) no ``records`` line, so neither
        ``rebind_triton_welds.CAMPAIGN_DIRS`` nor
        ``recut_composition_records.LEGS`` could reach it. The clearing is
        ``rebind_triton_welds.bind_bit_identity`` (``--bind-bit-identity``),
        which refuses anything but a RELEASED re-run of the probe's identity-leg
        families through the Triton adapter whose every recorded digest matches
        this checkout, and then refreshes ONLY the two sibling pins from the
        artifact's own ``imported_source_sha256`` — plus ``records``/
        ``artifact_sha256``/``recorded_utc`` so the run is followable. The
        re-run itself: the GPU host GPU 3, 2026-09-01,
        ``results/triton_bit_identity_2026-09-01/gate.json`` — curl 120/120
        (subnormal band 120/120), constitutive 120/120 with its multi-step
        auxiliary floor, fused pair 60/60 against BOTH oracles, whole-step and
        every multi-step arm identical, ``verdict.pass`` normalised to
        ``canonical_verdict.released`` by ``gate_provenance``. The six curated
        legs and the mutation tables remain the 2026-08-09/10 campaign's
        measurements, attributed to it in the records line; what the fresh run
        binds is that the CURRENT harness bytes reproduce the released identity
        verdict. ``compile_time_census_A1`` is deliberately outside the
        binder's required experiment set — that experiment drives the hand-CUDA
        kernel-string contract and does not run on the Triton adapter; its
        numbers are a compile-time census of ``kernels.py``, whose bytes the
        package's other welds pin.

    CLEARED 2026-08-31 — ``dispersive_composition_gate`` (3: ``__init__.py``,
    ``coverage.py``, ``launch.py``), and how, because "the probe earned a verdict
    key" is the short version of a decision:

      * THE BLOCKER WAS REAL. ``probe_triton_engine_route.py`` wrote
        ``probe``/``cases``/``summary`` and no outcome in ANY of the seven shapes
        ``gate_provenance.read_verdict`` knows, so ``released()`` scored it
        ``no verdict key`` and correctly refused to bind. It also recorded no
        ``imported_source_sha256``.
      * GIVING IT A KEY MEANT ADJUDICATING ``refusals_returned_none`` 3/4, and the
        premise that had moved was the PROBE'S. Measured on the GPU host and again
        locally on a lifted ``mp.Simulation``: ``2d_cond_pml`` carries
        ``D_conductivity=0.4``, so ``fields.condfac_for`` answers non-None on
        Dx/Dy/Dz and None on Bx/By/Bz. ``pml_curl_coverage`` with no ``sub_step``
        takes the conservative AGGREGATE path over all six ``CURL_TARGETS``
        (coverage.py clause 8 says so in as many words) and refuses; asked for
        ``step_B`` it returns COVERED with ZERO reasons. ``plan_pml_curl`` is a
        builder and asks for its own named sub-step, so it returns a plan on
        ``step_B`` and None on ``step_D`` — and the probe's ``all(... is None)``
        over both is False for a configuration the composition serves in FULL
        (4/4 sub-steps replaced, ``refusals`` empty, six complete driver steps
        byte-identical).
      * SO THE COUNTER WAS KEPT, NOT REDEFINED. ``refusals_returned_none`` still
        means what it always meant and is re-cut 4/4 -> 3/4 with the reason
        declared in ``recut_composition_records.CLAIM_RECUTS``; it is RECORDED and
        is deliberately NOT a release clause, because a tree that grows a product
        is allowed to move it. The release rests on the arithmetic instead, and on
        a new counter that is strictly stronger than the one it replaces:
        ``builder_matches_its_predicate`` 8/8 is an EQUIVALENCE per curl sub-step,
        asked on all eight cases rather than the four listed as refusals, so it
        also catches a builder returning None where its predicate ADMITS — the
        direction ``all(... is None)`` never looked at.
      * ``covered_whole_step_identical`` moved 4/4 -> 8/8 in the same re-cut, and
        that one had been invisible: ``claim_disagreements`` joins by FIELD NAME
        and the probe's summary spelled it ``whole_step_identical``, so the record's
        claim was compared against the run by nothing. Renaming the probe's counters
        to the record's own field names is what surfaced it.

      * The re-run is ``results/triton_engine_route_regate_2026-08-31/``, on
        the GPU host GPU 6 against a staged tree verified byte-identical to this one on
        all 533 manifest rows, and it released: 4/4 covered curls, 8/8 composed
        whole steps, 8/8 builder-vs-predicate, ``status: passed``, ``reasons: []``,
        39 imported source digests, ``canonical_verdict.released`` true read from
        ``summary.status``.

    CLEARED EARLIER, 2026-08-30, and kept here because its shape is the template
    the entry above followed:

      * ``dispersive_fused_pair_gate`` (8). Its gate re-ran on the GPU host into
        ``results/triton_composition_regate_2026-08-30/dispersive_fused_pair/``
        and released, but its curated ``product`` block said ``complete_steps_exact
        36/36`` over three cases while the run measured 48/48 over four — the gate
        GAINED ``two_pole_electric_source``, the in-seam deposit case. Binding the
        digests while carrying 36/36 forward would have left a weld whose hashes
        verify and whose arithmetic does not, so the claim was re-cut from the run
        (``recut_composition_records.CLAIM_RECUTS``, which derives every counter
        from the artifact and cross-checks the per-case lines against its own
        total) and then bound.
    """
    raw = _of_tier(_census(), walk.TIER_RAW)
    entries = _pinned()
    # NON-VACUITY, in every dimension the defect could return through, and it is
    # not the same guard as WELD_FLOOR above. That one counts PASS welds; these
    # count what is actually COMPARED, so a record whose entries survive with
    # empty-but-present digest maps fails here instead of passing an empty loop,
    # and an entry that drops a key to escape coverage takes the count below the
    # floor rather than going quiet.
    source_pins, sibling_pins = _source_pins(), _sibling_pins()
    assert len(entries) >= CHECKED_ENTRY_FLOOR, (
        f"only {len(entries)} entries carry a checked digest, was "
        f"{CHECKED_ENTRY_FLOOR} — coverage SHRANK; an entry does not leave this "
        f"check by losing a key")
    assert len(raw) >= RAW_TIER_FLOOR, (
        f"only {len(raw)} raw-tier digests, was {RAW_TIER_FLOOR} — the shape "
        f"changed and this test is now inert")
    assert len(source_pins) >= SOURCE_PIN_FLOOR, (
        f"only {len(source_pins)} (entry, path) pairs from live source_sha256 "
        f"maps, was {SOURCE_PIN_FLOOR} — the count the status filter was hiding "
        f"has shrunk, and a total that absorbed it would not have said so")
    assert len(sibling_pins) >= SIBLING_PIN_FLOOR, (
        f"only {len(sibling_pins)} <name>/<name>_sha256 pins resolve in-tree, "
        f"was {SIBLING_PIN_FLOOR} — the gate scripts left the comparison")
    drift = _drifted(raw)
    assert not drift, (
        f"{len(drift)} of {len(raw)} raw-tier digest(s) across {len(entries)} "
        f"entries no longer match the bytes the gate executed — RE-RUN the gate "
        f"and rebind with parity/meep_gpu/rebind_triton_welds.py; do not edit "
        f"this record and do not relax this test: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def test_the_triton_code_tier_recomputes_against_the_tree():
    """183 ``code_sha256`` pins, read by NOTHING at the merge bar until today.

    WHY THIS TIER IS THE DANGEROUS ONE TO LEAVE UNCHECKED. ``code_sha256`` is the
    digest that licenses a weld to stand after a comment-only edit — it is the
    one field in the record whose whole job is to say "the raw bytes moved and it
    did not matter". An unchecked field with that job is exactly how a stale
    record keeps looking current: the raw pin goes red, somebody reads the code
    pin as the reason it is fine, and nothing ever established that the code pin
    still describes anything at all.

    THE RULE, and it is a rule rather than a skip: a ``code_sha256`` is compared
    to ``code_identity.code_digest`` OF THE LIVE FILE, never to the file's source
    digest. Those two tiers MAY legitimately disagree, and exactly one thing makes
    that so — a comment or docstring edit since the gate ran, which is the entire
    clause ``code_identity`` licenses. Every other disagreement is a defect. A
    code digest that no longer recomputes describes neither the bytes that ran nor
    the bytes on disk.

    Measured 2026-08-30: 183 of 183 recompute. That is not a reason to have
    skipped it — it is what makes the assertion cheap to keep and expensive to
    break, and ``test_the_code_tier_is_the_clause_code_identity_licenses`` drives
    it in both directions so this green is not a green loop.
    """
    code = _of_tier(_census(), walk.TIER_CODE)
    assert len(code) >= CODE_TIER_FLOOR, (
        f"only {len(code)} code_sha256 pins found, was {CODE_TIER_FLOOR} — the "
        f"tier left the comparison, which is how it went unread for eleven days")
    drift = _drifted(code)
    assert not drift, (
        f"{len(drift)} of {len(code)} code digest(s) no longer recompute against "
        f"the tree — the recorded code identity describes neither the bytes the "
        f"gate ran nor the bytes on disk. RE-RUN the gate and rebind with "
        f"parity/meep_gpu/rebind_triton_welds.py; do not hand-edit the digest: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def test_no_entry_backfills_a_code_digest_beside_a_stale_source_digest():
    """THE CLAUSE ``code_identity`` WITHHOLDS, enforced on this record too.

    ``code_identity.py`` states the rule in its own docstring and it is a rule
    about every track, not about one: "Backfilling ``code_sha256`` is only honest
    for a weld whose ``source_sha256`` still matches, because only then is the
    file on disk the bytes the gate actually ran." A code digest recomputed today
    and written beside a source digest that has ALREADY moved describes the
    current tree, not the gate — and it MANUFACTURES the licence the clause
    withholds: the next reader finds a green code pin beside a red source pin and
    reads the green one as the reason the red one is fine.

    UNTIL 2026-08-30 (fourth round) ONLY THE CUDA CONTRACT ENFORCED THIS, over its
    39 pins. That is the same shape as every other defect this file has removed —
    a rule that is universal in its statement and partial in its coverage — so it
    is brought across here, over all 183, and to the Metal contract over its 259.

    MEASURED at the time of writing: 0 offenders, with 10 source pins drifted. The
    green is therefore about the record rather than about the check, which is what
    the planted control below is for.
    """
    found = _census()
    stale = {(f.trail[0], f.subject) for f in _of_tier(found, walk.TIER_RAW)
             if f.rule == walk.TIER_RAW + ":source_digest_map"}
    drifted_names = {where for where, _, _ in _drifted(
        [f for f in found if f.rule == walk.TIER_RAW + ":source_digest_map"])}
    moved = {(f.trail[0], f.subject) for f in found
             if f.rule == walk.TIER_RAW + ":source_digest_map"
             and walk.dotted(f.trail) in drifted_names}
    assert len(stale) >= SOURCE_PIN_FLOOR, len(stale)
    code = [f for f in _of_tier(found, walk.TIER_CODE)]
    assert len(code) >= CODE_TIER_FLOOR, len(code)
    offenders = sorted(walk.dotted(f.trail) for f in code
                       if (f.trail[0], f.subject) in moved)
    assert not offenders, (
        f"{len(offenders)} code digest(s) sit beside a STALE source digest: "
        f"{offenders[:8]} — that manufactures the licence code_identity "
        f"withholds. RE-RUN the gate; do not backfill the code digest")

    # ARMED. The predicate is driven against a planted census in which the same
    # entry pins a code digest for a path whose source digest has moved, and must
    # report it — a check that has never rejected anything has not been shown to
    # work.
    victim = "meep_gpu/triton_kernels/launch.py"
    live_code = code_digest(_resolve(victim).read_text(encoding="utf-8"))
    plant = {"w": {"source_sha256": {victim: "a" * 64},
                   "code_sha256": {victim: live_code}}}
    planted = _census(plant)
    # The code pin is CURRENT — it recomputes — which is exactly the shape this
    # rule is about: a green code digest written beside a source digest that has
    # already moved. The code tier alone therefore reports nothing.
    assert _drifted(_of_tier(planted, walk.TIER_CODE)) == []
    planted_moved_names = {where for where, _, _ in _drifted(
        [f for f in planted if f.rule == walk.TIER_RAW + ":source_digest_map"])}
    planted_moved = {(f.trail[0], f.subject) for f in planted
                     if f.rule == walk.TIER_RAW + ":source_digest_map"
                     and walk.dotted(f.trail) in planted_moved_names}
    caught = sorted(walk.dotted(f.trail) for f in planted
                    if f.tier == walk.TIER_CODE
                    and (f.trail[0], f.subject) in planted_moved)
    assert caught == [f"w.code_sha256.{victim}"], caught

    # AND THE OTHER DIRECTION: with the source pin honest, the same current code
    # pin is not an offender. Otherwise this would refuse every code digest.
    plant["w"]["source_sha256"][victim] = _raw_digest(_resolve(victim))
    honest = _census(plant)
    honest_moved_names = {where for where, _, _ in _drifted(
        [f for f in honest if f.rule == walk.TIER_RAW + ":source_digest_map"])}
    assert honest_moved_names == set()


def test_the_triton_device_tier_recomputes_against_the_tree():
    """68 ``device_sha256`` kernel digests, likewise read by no laptop test.

    This is the identity that actually reached the compiler: ``device_identity.
    device_digests`` parses the module and hashes the device source string each
    ``@triton.jit`` kernel generates. A module can be edited so its FILE moves
    while every kernel it emits is unchanged, and it can be edited so a kernel
    changes while the file's code identity barely moves — which is why this tier
    is neither the raw one nor the code one and is checked separately.

    A declared kernel the module no longer generates is REPORTED, not skipped: a
    digest naming a kernel that has left the file is a claim about nothing.
    """
    device = _of_tier(_census(), walk.TIER_DEVICE)
    assert len(device) >= DEVICE_TIER_FLOOR, (
        f"only {len(device)} device_sha256 kernel digests found, was "
        f"{DEVICE_TIER_FLOOR} — the tier left the comparison")
    drift = _drifted(device)
    assert not drift, (
        f"{len(drift)} of {len(device)} device kernel digest(s) no longer "
        f"recompute — RE-RUN the gate and rebind with "
        f"parity/meep_gpu/rebind_triton_welds.py: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


#: Which rules the walk is expected to fire on THIS record, and how many times at
#: the floor. BOTH DIRECTIONS, because both are informative: a rule that stops
#: firing means the shape it was written for has left the record (and the rule is
#: now a place for the next omission to hide), and a rule that starts firing means
#: a shape arrived that nobody wrote down. Measured 2026-08-30, third round.
EXPECTED_RULES = {
    walk.TIER_RAW + ":source_digest_map": 268,
    walk.TIER_CODE + ":code_digest_map": 183,
    "historical_recert_snapshot": 178,
    "artifact_or_log_digest": 88,
    walk.TIER_DEVICE + ":device_kernel_digest": 68,
    "historical_recut_log": 41,
    "historical_superseded_run": 38,
    # Added in 0.9.1: one per weld per architecture it has a run record for
    # (66 today, all 8.6). It is the digest of the digests that run certified,
    # which is what makes an architecture's evidence expire with the bytes.
    "run_bound_digest": 66,
    # 22 until 0.9.1, which removed the eight launch-script pins
    # (``slurm_launcher``/``slurm_launcher_sha256``): a record pins what the gate
    # executed, not how it was launched (test_package_boundary.py).
    walk.TIER_RAW + ":sibling_script_pin": 14,
    "out_of_tree_sibling_path": 16,
    "staging_only_file": 16,
    walk.TIER_RAW + ":host_side_digest_map": 10,
    walk.TIER_RAW + ":parent_key_named_pin": 6,
    "serial_driver_script": 4,
    "pin_names_no_path": 2,
}

#: DECLARED DEBT, and it is a debt rather than a clean exclusion. TWO digests in
#: this record name no file at all: ``conductivity_composition_gate.
#: standalone_product_gate`` carries ``gate_sha256`` and ``shared_probe_sha256``
#: with no ``gate`` or ``shared_probe`` key beside them, so nothing can ever
#: verify either one. Found by the structural walk on 2026-08-30; invisible to
#: every enumerator before it, because the old sibling rule required a sibling
#: and silently dropped the pin when there wasn't one.
#:
#: This set may only SHRINK. It is cleared by adding the path key beside the
#: digest — after which the pin becomes a checked sibling pin and this budget
#: loses a name — never by deleting the digest.
PIN_NAMES_NO_PATH_BUDGET = 2


def test_every_discovered_digest_is_checked_or_excluded_by_a_named_rule():
    """THE WALK. Every digest in the record, wherever it lives, has a fate.

    THIS IS THE TEST THAT REMOVES THE CLASS. The three rounds before it each
    removed one hiding place and left the next: a status filter, then a shape that
    knew about top-level entries and sibling keys and nothing else. Here the
    record is RECURSED — dicts and lists, at any depth — every 64-hex string is
    collected with the trail it was found at, and each one is either CHECKED in a
    named tier or EXCLUDED BY A NAMED RULE carrying its reason. There is no third
    outcome: a digest no rule claims is ``UNCLASSIFIED``, and that is this
    assertion failing.

    THE TOTAL IS ASSERTED, NOT ONLY THE CHECKED COUNT. A digest that stops being
    DISCOVERED fails here — which is the failure the checked-count floors cannot
    see, because a digest that vanishes from the walk also vanishes from every
    count derived from it.
    """
    found = _census()
    assert len(found) >= DISCOVERED_FLOOR, (
        f"the walk discovered {len(found)} digests, was {DISCOVERED_FLOOR} — "
        f"digests have stopped being FOUND, which is a shrunken denominator that "
        f"no count of what is CHECKED can report")
    unclassified = [walk.dotted(f.trail) for f in found
                    if f.rule == walk.UNCLASSIFIED]
    assert not unclassified, (
        f"{len(unclassified)} digest(s) matched no rule: {unclassified[:12]} — "
        f"write a rule in meep_gpu/weld_record_walk.py, in this change: check it, "
        f"or exclude it by name with the reason. A digest with no rule is the "
        f"defect this walk exists to make impossible to introduce quietly")
    checked = _checked(found)
    assert len(checked) >= CHECKED_FLOOR, (
        f"only {len(checked)} of {len(found)} discovered digests are checked, was "
        f"{CHECKED_FLOOR} — digests moved from checked to excluded, which is a "
        f"decision that has to be argued, not a count that may drift")
    fired = walk.rules_that_fired(found)
    reasonless = sorted(set(fired) - set(walk.RULE_REASONS))
    assert not reasonless, (
        f"rules fired with no written reason: {reasonless}")
    missing = sorted(set(EXPECTED_RULES) - set(fired))
    assert not missing, (
        f"declared rules that no longer fire on this record: {missing} — the "
        f"shape each was written for has left, and a rule nothing exercises is "
        f"where the next real omission hides. Delete it, or find what moved")
    surprise = sorted(set(fired) - set(EXPECTED_RULES) - {walk.UNCLASSIFIED})
    assert not surprise, (
        f"rules fired that this record never declared: {surprise} — a new shape "
        f"arrived; declare it in EXPECTED_RULES with its count")
    shrunk = {name: (fired[name], floor) for name, floor in EXPECTED_RULES.items()
              if fired[name] < floor}
    assert not shrunk, (
        f"rules whose digest count SHRANK (got, was): {shrunk} — coverage does "
        f"not fall silently")
    assert fired["pin_names_no_path"] <= PIN_NAMES_NO_PATH_BUDGET, (
        f"{fired['pin_names_no_path']} digests name no file, budget is "
        f"{PIN_NAMES_NO_PATH_BUDGET} — a digest nothing can verify is a debt "
        f"that may only shrink. Add the path key beside it")


def _inside_repo(name, root=REPO):
    """Does this pinned name land inside THIS checkout?

    ``~`` is expanded rather than treated as a path component, because that is
    what the record means by it — the launcher directory on the device host. A
    name that expands outside the repo is outside it; everything relative is
    inside by construction, which is exactly the case the premise below is about.
    """
    candidate = (pathlib.Path(name).expanduser() if name.startswith("~")
                 else _resolve(name, root))
    try:
        candidate.resolve().relative_to(pathlib.Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def test_every_not_in_this_checkout_exclusion_still_has_its_premise(tmp_path):
    """AN EXCLUSION IS ONLY AS GOOD AS ITS PREMISE — the fifth sub-form.

    32 digests in this record are excluded on a claim about the world rather than
    about the record: 16 ``staging_only_file`` pins say ``parity/meep_gpu/
    cases.py`` was staged on the device host and never committed, and 16
    ``out_of_tree_sibling_path`` pins name launchers under
    ``<staging-root>/``. Both reasons are sentences about THIS CHECKOUT, and
    until today nothing evaluated either one. The day ``cases.py`` is committed,
    the old arrangement would have gone on excluding a digest that had become
    checkable — a filter nobody counted, in prose.

    A violation is NOT a licence to delete the digest. It says the digest has
    become checkable and must move into a tier, which is the only direction
    coverage is allowed to move. That day came in 0.9.1: ``cases.py`` is committed,
    its 16 staging-only premises have expired, and each of those digests is now
    compared with the committed bytes here and must match. Every other premise
    must still hold.

    THE ARMED CONTROL is the second half: the same census is driven against a
    planted tree in which the staged file DOES exist, and the check must report
    it. ``walk.premise_violations`` takes both predicates as arguments precisely
    so this can be done without writing into the real tree.
    """
    found = _census()
    premised = [f for f in found if f.rule in walk.PREMISE_ABSENT_FROM_TREE]
    assert len(premised) >= 32, (
        f"only {len(premised)} digests rest on a not-in-this-checkout premise, "
        f"was 32 — the shape left and this check is now inert")
    violations = walk.premise_violations(
        premised,
        is_file=lambda name: _resolve(name).is_file(),
        inside_repo=_inside_repo,
    )
    # A STAGED FILE THAT IS NOW COMMITTED IS CHECKED, NOT EXCUSED. ``cases.py`` was
    # committed in 0.9.1 with the bytes these records name as staged, so its 16
    # exclusions have expired exactly as the premise promised they would. Each digest
    # is compared here with the committed file, raw -- the comparison the raw tier
    # makes -- and a mismatch fails. Any OTHER expired premise still fails outright.
    by_where = {walk.dotted(f.trail): f for f in premised}
    committed = [(where, subject) for where, subject, why in violations
                 if why.startswith("declared staging-only, but the file IS in this")]
    others = [v for v in violations if (v[0], v[1]) not in committed]
    assert not others, (
        f"{len(others)} exclusion(s) rest on a premise the tree no longer "
        f"supports: {others[:6]} — the digest has become checkable; move it "
        f"into a tier, never delete it")
    assert sorted({subject for _, subject in committed}) == ["parity/meep_gpu/cases.py"] \
        and len(committed) == 16, committed
    mismatched = [(where, subject) for where, subject in committed
                  if _raw_digest(_resolve(subject)) != by_where[where].value]
    assert not mismatched, (
        f"{len(mismatched)} digest(s) of a committed staged file do not match the bytes "
        f"in the tree: {mismatched[:4]} — the record names one file and the commit "
        f"holds another")

    # THE CONTROL. Plant the staged file and require the check to notice.
    staged = sorted({f.subject for f in premised if f.rule == "staging_only_file"})
    assert staged == ["parity/meep_gpu/cases.py"], staged
    planted = tmp_path / staged[0]
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("X = 1\n", encoding="utf-8")
    armed = walk.premise_violations(
        premised,
        is_file=lambda name: (tmp_path / name).is_file(),
        inside_repo=lambda name: _inside_repo(name, tmp_path),
    )
    assert len(armed) == 16 and all("IS in this checkout" in why
                                    for _, _, why in armed), armed
    # ...and the digest comparison above is armed too: bytes other than the recorded
    # ones, planted at the committed path, must be reported as a mismatch, every one.
    planted_mismatches = [where for where, _, _ in armed
                          if _raw_digest(planted) != by_where[where].value]
    assert len(planted_mismatches) == 16, planted_mismatches


def test_no_exclusion_on_this_record_defers_to_another_test():
    """THE DELEGATION IS GONE, and this is what keeps it gone.

    ``host_sha256`` was excluded here for eleven days on the strength of a
    citation — "already recomputed against the tree by test_triton_kernels.py" —
    and the cited test recomputes those digests THROUGH
    ``device_identity.weld_survives_edit``, the allowance this module refuses by
    name. A delegation that lands on a weaker claim than the exclusion states is
    an unchecked digest wearing a citation, and the citation is the part that
    makes it hard to see.

    So no rule firing on THIS record may be one the walker's
    :data:`walk.DELEGATED_COVERAGE` register lists. The two that ever did are
    named there: ``host_side_digest_map``, now a raw tier, and
    ``generated_kernel_source_slot``, which does not fire here at all because this
    record's ``kernel_source_sha256`` is a sibling pin naming a real file.

    THE REGISTER RATHER THAN THE PROSE, deliberately. Several reasons in the
    walker cross-reference a test — ``historical_superseded_run`` points at the
    three legs that EARN it — and "mentions a test" is not the same claim as
    "these digests are compared over there instead of here". Sniffing the reason
    text would conflate the two and turn an earning citation into a red.
    """
    found = _census()
    excluding = {f.rule for f in found if f.tier is None}
    deferring = sorted(excluding & set(walk.DELEGATED_COVERAGE))
    assert not deferring, (
        f"{deferring} excuse digests on this record by citing another test. A "
        f"citation is not a comparison: check them here, or assert the citation "
        f"the way test_metal_weld_contract.py::"
        f"test_the_generated_slot_delegation_is_real_and_admits_no_allowance does")

    # ARMED. A predicate that has never rejected anything is a predicate nobody
    # has watched work, which is the whole complaint this file makes about the
    # eleven days host_sha256 spent behind a citation.
    planted = dict(walk.DELEGATED_COVERAGE)
    planted["artifact_or_log_digest"] = "other_test.py::test_something"
    assert sorted(excluding & set(planted)) == ["artifact_or_log_digest"]


def test_every_citation_in_an_exclusion_names_a_test_that_exists():
    """A CITATION IS A CLAIM, and an unresolvable one is worse than none.

    Two entries here are excused the metadata tier partly on the strength of
    another test: ``guard``'s VALUE is checked against ``kernels.py``, and
    ``bit_identity_gate``'s verdict SHAPE is pinned. Both citations were real and
    neither named a function — "by test_triton_kernels.py" cannot be resolved,
    so it cannot rot visibly either. They now name the function, and the function
    must exist.

    THE SWEEP IS OVER EVERY CITATION THIS RECORD IS SUBJECT TO — the two
    exclusion tables here AND the shared walker's rule reasons and delegation
    register — because a citation in the walker is read by the same person, on
    the same claim, and rots the same way.

    This is the same argument the digest-level delegation lost: what makes a
    citation safe is not that someone wrote it down, it is that something fails
    when it stops being true. Parsed rather than imported, so this stays a
    laptop-cheap structural check.
    """
    cited = set()
    sources = [NOT_A_GATE_RECORD.values(), PINNED_BUT_NOT_A_WELD.values(),
               # AND THE WALKER'S OWN REASONS, because the rule is about every
               # citation and not only the ones written in this file. Two of
               # those cross-reference the tests that EARN an exclusion rather
               # than covering it — a different claim, resolved the same way.
               walk.RULE_REASONS.values(), walk.DELEGATED_COVERAGE.values()]
    for reasons in sources:
        for reason in reasons:
            cited |= set(re.findall(r"(test_[a-z0-9_]+\.py)::(test_[a-z0-9_]+)",
                                    reason))
    assert cited, (
        "no exclusion cites a test any more — either the citations were dropped "
        "(fine, delete this) or they were rewritten into a form this cannot "
        "resolve, which is the shape the check is about")
    missing = []
    for filename, function in sorted(cited):
        module = PACKAGE / filename
        if not module.is_file():
            missing.append(f"{filename} (no such file)")
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        if not any(isinstance(node, ast.FunctionDef) and node.name == function
                   for node in ast.walk(tree)):
            missing.append(f"{filename}::{function}")
    assert not missing, (
        f"exclusions cite tests that do not exist: {missing} — an exclusion "
        f"resting on a test nobody can find is an unchecked claim; fix the "
        f"citation or check the thing here")

    # ARMED, both directions, on the resolver itself.
    assert ("test_triton_kernels.py", "test_a_function_nobody_wrote") not in cited
    tree = ast.parse((PACKAGE / "test_triton_kernels.py").read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}
    assert "test_a_function_nobody_wrote" not in names
    assert "test_the_recorded_gate_verdicts_are_pass_shaped" in names


#: The container keys whose digests are HISTORY. Each is excluded by the walker
#: with its reason; this names them again here because the exclusion is EARNED
#: rather than granted — see the test below.
HISTORICAL_RULES = ("historical_superseded_run", "historical_recert_snapshot",
                    "historical_recut_log")


def test_a_historical_exclusion_is_earned_by_a_live_checked_claim():
    """SUPERSEDED PINS ARE EXCLUDED — and here is the price of that decision.

    THE DECISION, stated once. 22 ``source_sha256`` pins live inside
    ``cylindrical_composition_gate.superseded_runs[0..1]``, and 217 more sit under
    ``*_at_recert`` and ``*_recut_log`` blocks. They are NOT checked against the
    tree. Both outcomes were defensible and this is the one the record itself
    argues for: ``why_superseded`` says of those bytes that the run "certified the
    previous bytes and is kept verbatim; it is not the record of the shipped host
    side any more", and the recert blocks say the surrounding fields "are the
    ORIGINAL job and are deliberately untouched: they are true about that job and
    its bytes". Comparing them to this tree would assert a claim the record
    explicitly withdraws, and — worse — the only way to clear the resulting red
    would be to DELETE THE HISTORY, which is the one repair that destroys evidence.
    A recut log is stronger still: half of every from/to pair is stale by
    construction, and a log that had to match the tree could not record a change.

    THE PRICE, which is what stops this being a hole. ``superseded_runs`` must not
    become the place a live drift goes to die, so the exemption is earned, in three
    legs, all checked here:

      1. A superseded run must SAY it is superseded — ``why_superseded`` or
         ``superseded_utc``, non-empty. A block that merely SITS in the list gets
         nothing.
      2. The entry containing history must carry a LIVE CHECKED digest of its own,
         or be named in :data:`NOT_A_GATE_RECORD`. So moving a gate's last live pin
         into its own history does not buy silence: it empties the entry, which
         then has to be written down in this file with a reason, in the same change.
      3. Every historical digest must be attributable to a top-level entry — no
         orphan history at the record root.

    Measured 2026-08-30: 3 superseded runs, all declaring; 11 entries carrying
    history, 10 live-checked and ``family_recert_2026-08-14`` excluded by name.
    """
    record = _record()
    found = _census(record)
    historical = [f for f in found if f.rule in HISTORICAL_RULES]
    assert len(historical) >= 257, (
        f"only {len(historical)} historical digests found — the containers this "
        f"decision is about have left the record, and this test now argues about "
        f"nothing")

    # LEG 1 — a superseded run declares itself, or it is not one.
    silent = []
    for name, entry in sorted(record.items()):
        runs = entry.get("superseded_runs") if isinstance(entry, dict) else None
        if not isinstance(runs, list):
            continue
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            if not (run.get("why_superseded") or run.get("superseded_utc")):
                silent.append(f"{name}.superseded_runs[{index}]")
    assert not silent, (
        f"{len(silent)} superseded run(s) claim the exemption without declaring "
        f"it: {silent} — carry why_superseded or superseded_utc, or have the "
        f"digests checked like everything else")

    # LEG 2 — history is only exempt where a current claim stands beside it.
    live_owners = {f.trail[0] for f in _checked(found)}
    unearned = sorted({f.trail[0] for f in historical}
                      - live_owners - set(NOT_A_GATE_RECORD))
    assert not unearned, (
        f"{unearned} carry HISTORICAL digests and no live checked digest at all — "
        f"an entry whose only pins are in its own past has stopped making a claim "
        f"about this tree. Either it still gates something, in which case pin the "
        f"live bytes, or it does not, in which case name it in NOT_A_GATE_RECORD "
        f"with the reason")

    # LEG 3 — no orphan history at the record root.
    rootless = sorted(walk.dotted(f.trail) for f in historical if len(f.trail) < 2)
    assert not rootless, (
        f"historical digests with no owning entry: {rootless}")

    # THE ARMED CONTROLS. All three legs are green over the real record, so each
    # is driven against a planted one — an exemption that has never been watched
    # to fail is an exemption granted, not earned.
    #
    # LEG 1 ARMED: a run that sits in superseded_runs saying nothing.
    mute = {"g": {"source_sha256": {"driver.py": "0" * 64},
                  "superseded_runs": [{"source_sha256": {"driver.py": "1" * 64}}]}}
    silent_planted = [f"g.superseded_runs[{i}]"
                      for i, run in enumerate(mute["g"]["superseded_runs"])
                      if not (run.get("why_superseded") or run.get("superseded_utc"))]
    assert silent_planted == ["g.superseded_runs[0]"]
    mute["g"]["superseded_runs"][0]["why_superseded"] = "it was re-cut"
    assert not [f"g.superseded_runs[{i}]"
                for i, run in enumerate(mute["g"]["superseded_runs"])
                if not (run.get("why_superseded") or run.get("superseded_utc"))]

    # LEG 2 ARMED: the exact evasion this leg exists to stop — a gate whose live
    # pins have all MOVED INTO its own history. It must be reported, and adding
    # back one live pin must clear it.
    evasive = {"g": {"superseded_runs": [{"why_superseded": "x",
                                          "source_sha256": {"driver.py": "1" * 64}}]}}
    evasive_found = _census(evasive)
    assert [f.rule for f in evasive_found] == ["historical_superseded_run"]
    assert not {f.trail[0] for f in _checked(evasive_found)}, (
        "an entry whose only digests are historical still counts as live-checked "
        "— leg 2 cannot fire and the exemption is free")
    evasive["g"]["source_sha256"] = {"driver.py": "0" * 64}
    assert {f.trail[0] for f in _checked(_census(evasive))} == {"g"}


def test_every_record_entry_is_checked_or_excluded_by_name():
    """THE MECHANISM. Nothing leaves coverage by omitting a field.

    This is what the status filter cost and what would have caught it: every dict
    entry in the record is either CHECKED for its bytes or written down in
    :data:`NOT_A_GATE_RECORD` with the reason it pins nothing. A new gate record
    that forgets ``status`` is checked anyway; one that pins no digest at all has
    to be justified in this file, in the same change, where a reviewer sees it.

    BOTH DIRECTIONS. An exclusion that outlives its entry is deleted here too — a
    stale name is a place where the next real omission can hide.
    """
    record = _record()
    dict_entries = {name for name, entry in record.items() if isinstance(entry, dict)}
    checked = set(_pinned(record))
    unaccounted = sorted(dict_entries - checked - set(NOT_A_GATE_RECORD))
    assert not unaccounted, (
        f"{len(unaccounted)} record entries are neither checked nor excluded by "
        f"name: {unaccounted} — either pin the bytes the run executed, or name "
        f"each here with the reason it is not a gate record")
    stale = sorted(set(NOT_A_GATE_RECORD) - dict_entries)
    assert not stale, (
        f"the exclusion list names entries the record no longer has: {stale}")
    overlap = sorted(checked & set(NOT_A_GATE_RECORD))
    assert not overlap, (
        f"{overlap} are excluded by name AND carry pins — an entry that pins "
        f"bytes is checked; delete the exclusion rather than the coverage")


def test_the_weld_tier_and_the_pinned_tier_differ_only_where_it_is_written_down():
    """Which pinned entries are not welds is a DECISION, not a missing key.

    The metadata requirements below run over :func:`_welds` — the PASS subset —
    because ``artifact_sha256``, a named subnormal policy and a validated
    toolchain are claims a PASS makes and a refusal does not. That is a narrower
    denominator than the byte check's, so it gets the same treatment the byte
    check just got: the gap is enumerated and named, in both directions, rather
    than left to the absence of ``status``.
    """
    pinned, welded = set(_pinned()), set(_welds())
    assert welded <= pinned, sorted(welded - pinned)
    undeclared = sorted(pinned - welded - set(PINNED_BUT_NOT_A_WELD))
    assert not undeclared, (
        f"{len(undeclared)} entries pin bytes but are not welds and say why "
        f"nowhere: {undeclared} — name each in PINNED_BUT_NOT_A_WELD, or give it "
        f"the metadata a weld carries")
    promoted = sorted(set(PINNED_BUT_NOT_A_WELD) & welded)
    assert not promoted, (
        f"{promoted} are declared 'not a weld' and carry status=PASS — one of "
        f"the two is wrong, and a refusal that acquires a PASS is the dangerous "
        f"direction")
    stale = sorted(set(PINNED_BUT_NOT_A_WELD) - pinned)
    assert not stale, f"declared non-welds that are not in the record: {stale}"


# ---------------------------------------------------------------------------
# THE ROUTE RECORD: checked through its runs, not as a weld
# ---------------------------------------------------------------------------
#
# The same rule, in the same words, as the other two weld contracts. What differs
# per suite is only where this suite declares the record a non-weld, how it spells a
# run's key, and which route-run fields its shape check reads.

#: This ledger's dispatch record, and the table that declares it a non-weld.
DISPATCH_RECORD = "driver_dispatch"
DISPATCH_RECORD_DECLARED_BY = PINNED_BUT_NOT_A_WELD

#: The key the armed control files a planted route run under: a compute capability.
PLANTED_RUN_KEY = "8.6"


def _dispatch_shape_reasons(entry):
    """Why the record is not in the per-capability shape, read against the
    ROUTE-run fields, which name ``status``; the weld run fields do not."""
    return fastpath.retired_shape_reasons(entry, run_fields=fastpath.DISPATCH_RUN_FIELDS)


def _tree_digest(name):
    """The live raw digest of a path the record pins, resolved as this suite's drift
    check resolves it."""
    return _raw_digest(_resolve(name))


def _dispatch_record_problems(record):
    """What ``weld_record_walk.route_record_problems`` reports for this ledger's
    dispatch record, from this suite's own measurements: liveness by
    ``fastpath.live_capabilities``, drift by :func:`_measured_drift` (the raw tier,
    as for a weld), and shape by :func:`_dispatch_shape_reasons`."""
    entry = record[DISPATCH_RECORD]
    return walk.route_record_problems(
        entry, live=fastpath.live_capabilities(entry),
        moved=sorted(_measured_drift(DISPATCH_RECORD, entry)),
        stranded=_dispatch_shape_reasons(entry))


def test_the_dispatch_record_stands_on_a_live_passing_route_run():
    """The dispatch record is not a weld: it stands on its route runs and the tree.

    ``driver_dispatch`` pins the bytes of the dispatch wiring, and no single gate
    produced it, so the weld status rule does not apply to it. Its verdict is the
    route campaign's: ``recut_driver_dispatch_record.py`` files each run under
    ``runs[<key>]``, one per environment, with that run's ``status``. The rule is the
    same in the three weld contracts (``weld_record_walk.route_record_problems``):

      (a) at least one run is live (``fastpath.live_capabilities``), so an empty
          ``runs``, or one whose every run binds bytes that have since moved, fails;
      (b) every live run reads ``status`` PASS;
      (c) every file the record pins in the raw tier matches the tree. A run stays
          live after the files move, because it binds the record's digests and not
          the tree, so liveness alone cannot say the record describes what ships;
      (d) the route-run fields, ``status`` among them, appear only under
          ``runs[<key>]``. A status beside the digests would be a second answer to
          what the route campaign measured, and is refused.

    Cleared by a route campaign that releases on these bytes and a recut that files
    its run; never by writing a status beside the digests.
    """
    record = _record()
    assert DISPATCH_RECORD in DISPATCH_RECORD_DECLARED_BY, (
        f"{DISPATCH_RECORD} is no longer declared a non-weld, so the weld rules would "
        f"apply to it as well as this one; declare it, or retire this test")
    assert isinstance(record.get(DISPATCH_RECORD), dict), (
        f"the ledger carries no {DISPATCH_RECORD} record")
    problems = _dispatch_record_problems(record)
    assert not problems, (
        f"{DISPATCH_RECORD}: {problems} - run the route campaign on these bytes and "
        f"cut its run with parity/meep_gpu/recut_driver_dispatch_record.py; never "
        f"write a status beside the digests")


def test_the_dispatch_record_rule_refuses_what_its_route_runs_do_not_support():
    """ARMED, on this ledger's own record, because the test above fails until a route
    campaign has run on these bytes, and a rule watched only while it fails has not
    been watched refusing what it is meant to refuse.

    The control first: a copy of the record re-pinned to the tree, holding one live
    run that reads PASS, is accepted, so the rule can pass. Then four plants on that
    control, each refused by its own clause and by nothing else: an entry-level
    ``status`` (d), an emptied ``runs`` (a), a run that reads PASS but binds bytes
    other than the record's, so that no run is live (a), and the live run reading
    FAIL (b). The stale run is what holds liveness to ``fastpath.live_capabilities``:
    a check that read every recorded run as live would accept it."""
    record = _record()
    source = {name: _tree_digest(name)
              for name in record[DISPATCH_RECORD]["source_sha256"]}
    bound = fastpath.bound_digest(dict(record[DISPATCH_RECORD], source_sha256=source))

    def problems_with(**fields):
        planted = json.loads(json.dumps(record))
        planted[DISPATCH_RECORD].update({"source_sha256": source, fastpath.RUNS: {
            PLANTED_RUN_KEY: {"status": "PASS", "bound_sha256": bound}}})
        planted[DISPATCH_RECORD].update(fields)
        return _dispatch_record_problems(planted)

    control = problems_with()
    assert control == [], control
    stated = problems_with(status="PASS")
    assert len(stated) == 1 and "beside the digests" in stated[0], stated
    assert "'status'" in stated[0], stated
    emptied = problems_with(**{fastpath.RUNS: {}})
    assert len(emptied) == 1 and emptied[0].startswith("no live run"), emptied
    other_bytes = "0" * 64
    assert other_bytes != bound
    stale = problems_with(**{fastpath.RUNS: {
        PLANTED_RUN_KEY: {"status": "PASS", "bound_sha256": other_bytes}}})
    assert len(stale) == 1 and stale[0].startswith("no live run"), stale
    failed = problems_with(**{fastpath.RUNS: {
        PLANTED_RUN_KEY: {"status": "FAIL", "bound_sha256": bound}}})
    assert len(failed) == 1 and failed[0].startswith(
        f"{fastpath.RUNS}[{PLANTED_RUN_KEY!r}] reads status 'FAIL', not PASS"), failed


def test_the_bare_basename_resolver_finds_the_files_the_old_entries_pin():
    """The resolver is new logic, so it gets a control of its own.

    Without this, ``_resolve`` returning a path nobody has could turn 88 real
    comparisons into 88 identical MISSING lines — a red that looks like a broken
    test rather than a stale record, which is the reading that gets a check
    deleted. Both branches are exercised: at least one bare name must land in
    ``triton_kernels/`` and at least one in the package root, and every bare name
    the record pins must resolve to a file that exists.
    """
    bare = {f.subject for f in _of_tier(_census(), walk.TIER_RAW)
            if f.subject and "/" not in f.subject}
    assert len(bare) >= 15, (
        f"only {len(bare)} bare-basename pins found — the pre-2026-08-19 spelling "
        f"left the record and this control is now vacuous")
    unresolved = sorted(n for n in bare if not _resolve(n).is_file())
    assert not unresolved, f"bare pins that resolve to nothing: {unresolved}"
    where = {_resolve(n).parent.relative_to(REPO).as_posix() for n in bare}
    assert where == {"meep_gpu", "meep_gpu/triton_kernels"}, (
        f"the two-root rule is not being exercised; bare names resolve into "
        f"{sorted(where)}")
    # AND THE ORDER IS THE RULE, not a coincidence of which files exist:
    # __init__.py is the one basename both directories hold, and the record means
    # the triton_kernels one.
    assert _resolve("__init__.py") == REPO / "meep_gpu" / "triton_kernels" / "__init__.py"
    assert _resolve("driver.py") == REPO / "meep_gpu" / "driver.py"


def test_the_drift_check_refuses_a_planted_digest(tmp_path):
    """THE ARMED CONTROL. Withhold nothing and plant one wrong byte-digest.

    Without this, ``test_the_triton_welds_are_bound_to_the_live_sources`` passing
    would be consistent with ``_drifted`` returning an empty list unconditionally.
    Both directions are pinned: the planted digest must be reported, and the same
    record with the TRUE digest must not be.

    DRIVEN THROUGH THE WALK, not through a hand-built pin list, so the control
    exercises the classifier as well as the comparison. A rule that stopped
    claiming these shapes would take the planted digest out of the checked set and
    this test would go red — which is the only way a control can cover a walker
    whose job is deciding what to compare.
    """
    victim = PACKAGE / "triton_kernels" / "launch.py"
    true_digest = hashlib.sha256(victim.read_bytes()).hexdigest()
    rel = victim.relative_to(REPO).as_posix()
    wrong = "f" * 64          # NOT the placeholder; that is a separate leg below
    planted = {"a_synthetic_weld": {"source_sha256": {rel: wrong}}}
    honest = {"a_synthetic_weld": {"source_sha256": {rel: true_digest}}}
    assert _drifted(_census(planted)) == [
        (f"a_synthetic_weld.source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {true_digest[:12]}")]
    assert _drifted(_census(honest)) == []

    # THE SAME CONTROL ON A BARE BASENAME, because the resolver is what makes the
    # 88 previously-unchecked pairs comparable and a resolver that quietly missed
    # would report every one of them as MISSING rather than as a match.
    bare_planted = {"a_synthetic_weld": {"source_sha256": {"launch.py": wrong}}}
    bare_honest = {"a_synthetic_weld": {"source_sha256": {"launch.py": true_digest}}}
    assert _drifted(_census(bare_planted)) == [
        ("a_synthetic_weld.source_sha256.launch.py", "launch.py",
         f"recorded {wrong[:12]} live {true_digest[:12]}")]
    assert _drifted(_census(bare_honest)) == []

    # AN UNBOUND PIN IS A FAILURE, and it is reported as its own reason rather
    # than as drift. Pinned in both directions so the placeholder can never
    # become a way to declare coverage without paying for it.
    unbound = {"a_synthetic_weld": {"source_sha256": {rel: UNBOUND_PLACEHOLDER}}}
    reported = _drifted(_census(unbound))
    assert len(reported) == 1 and reported[0][1] == rel
    assert "UNBOUND PLACEHOLDER" in reported[0][2]

    # A SIBLING SCRIPT PIN IS COMPARED THE SAME WAY. Without this the 22 sibling
    # pins could be collected and never compared.
    probe = "parity/meep_gpu/probe_triton_symmetry_composition.py"
    probe_digest = hashlib.sha256((REPO / probe).read_bytes()).hexdigest()
    sib_planted = {"g": {"probe": probe, "probe_sha256": wrong}}
    sib_honest = {"g": {"probe": probe, "probe_sha256": probe_digest}}
    assert _drifted(_census(sib_planted)) == [
        ("g.probe_sha256", probe, f"recorded {wrong[:12]} live {probe_digest[:12]}")]
    assert _drifted(_census(sib_honest)) == []

    # A PARENT-KEY-NAMED PIN, the shape that hid six digests until the walk. Same
    # comparison, and a control so the new rule cannot collect without comparing.
    pk_planted = {"specialized_kernel_sources": {"launch.py": {"sha256": wrong}}}
    pk_honest = {"specialized_kernel_sources": {"launch.py": {"sha256": true_digest}}}
    assert _drifted(_census(pk_planted)) == [
        ("specialized_kernel_sources.launch.py.sha256", "launch.py",
         f"recorded {wrong[:12]} live {true_digest[:12]}")]
    assert _drifted(_census(pk_honest)) == []

    # A DIGEST INSIDE A LIST, at the depth that hid 22 real pins for eleven days.
    # The old enumerator descended dicts and stopped at lists; this leg is the one
    # that would have failed then and must never stop failing on a walker that
    # forgets lists again.
    in_list = {"g": {"runs": [{"source_sha256": {rel: wrong}}]}}
    assert _drifted(_census(in_list)) == [
        (f"g.runs[0].source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {true_digest[:12]}")]

    # AND A COMMENT-ONLY EDIT IS STILL DRIFT IN THE RAW TIER. This is the exact
    # class ``weld_survives_edit`` waves through, so the difference between the
    # two rules is executed here rather than asserted in prose.
    commented = tmp_path / rel
    commented.parent.mkdir(parents=True, exist_ok=True)
    commented.write_bytes(victim.read_bytes() + b"\n# a comment appended\n")
    assert _drifted(_census(honest), root=tmp_path) == [
        (f"a_synthetic_weld.source_sha256.{rel}", rel,
         f"recorded {true_digest[:12]} live "
         f"{hashlib.sha256(commented.read_bytes()).hexdigest()[:12]}")]


def test_the_walker_finds_a_digest_planted_at_a_new_nesting_depth():
    """THE ARMED CONTROL FOR THE WALK ITSELF. Plant one deeper than anything real.

    ``test_every_discovered_digest_is_checked_or_excluded_by_a_named_rule``
    passing is only meaningful if the walk actually descends. Without this, a
    walker that stopped at depth 4 would report zero unclassified digests and a
    perfectly stable discovered count, and every one of the 940 would still be
    found — because the record's deepest real digest sits at depth 6.

    So: plant a digest at depth EIGHT — two levels past the record's deepest real
    one — through two lists and a key spelling that appears nowhere in the tree,
    and require BOTH that the walk surfaces it AND that it lands in
    ``UNCLASSIFIED``. The second half is the load-bearing one. A walker that found
    it and quietly classified it as excluded would be the original defect wearing
    this test as cover; the contract is that a shape nobody has written a rule
    for FAILS.
    """
    digest = "a" * 64
    deep = {"an_entry": {"a": [{"b": {"c": [{"d": {"a_key_nobody_wrote": digest}}]}}]}}
    found = walk.census(deep)
    assert [f.value for f in found] == [digest], (
        "the walk did not surface a digest nested eight levels down through two "
        "lists — it is not recursing, and every count it reports is a floor on "
        "how deep it happens to go rather than on what the record holds")
    planted = found[0]
    assert len(planted.trail) == 8, planted.trail
    assert len(planted.trail) > max(len(f.trail) for f in _census()), (
        "the planted digest is no deeper than the record's own — the control has "
        "stopped testing anything the real walk does not already reach")
    assert walk.dotted(planted.trail) == \
        "an_entry.a[0].b.c[0].d.a_key_nobody_wrote"
    assert planted.rule == walk.UNCLASSIFIED and planted.tier is None, (
        f"a digest under an unanticipated key was classified {planted.rule!r} "
        f"instead of failing — an unknown shape must be a red somebody writes a "
        f"rule for, never a silent exclusion")

    # AND THE REAL RECORD IS DEEPER THAN THE OLD WALKER WENT, so this is not a
    # hypothetical: 38 digests live inside superseded_runs, past a list.
    real = _census()
    assert max(len(f.trail) for f in real) >= 6
    assert sum(1 for f in real if any(isinstance(p, int) for p in f.trail)) >= 68, (
        "no digests found inside lists — the shape that hid 22 source pins has "
        "left the record and this control no longer covers it")


def _plant(root, rel, text):
    """Write ``text`` at ``root/rel`` and return ``root``, so a census can be
    compared against a tree that differs from this checkout in exactly one file."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def test_the_code_tier_is_the_clause_code_identity_licenses(tmp_path):
    """THE ARMED CONTROL FOR THE CODE TIER, in both directions, on a real file.

    The code tier is under the merge bar as of 2026-08-30, and 183 of 183 pins
    recompute — so without this the assertion could be inert and read identically.
    Three legs, each executed against a planted tree rather than argued in prose:

      1. A COMMENT-ONLY EDIT: the raw digest MOVES and the code digest HOLDS.
         This is the whole clause ``code_identity`` licenses, and it is the reason
         the two tiers are allowed to disagree at all.
      2. A REAL CODE EDIT: both move. The licence does not stretch to it.
      3. A PLANTED CODE DIGEST on an unedited file: reported. A code digest is
         compared to the live file's CODE, never excused by its source digest
         still matching.
    """
    victim = PACKAGE / "triton_kernels" / "launch.py"
    rel = victim.relative_to(REPO).as_posix()
    source = victim.read_text(encoding="utf-8")
    raw_now = hashlib.sha256(victim.read_bytes()).hexdigest()
    code_now = code_digest(source)

    record = {"w": {"source_sha256": {rel: raw_now}, "code_sha256": {rel: code_now}}}

    # LEG 1 — comments move the bytes and not the code.
    commented = _plant(tmp_path / "commented", rel,
                       source + "\n# a comment appended, changing no execution\n")
    assert hashlib.sha256((commented / rel).read_bytes()).hexdigest() != raw_now
    assert code_digest((commented / rel).read_text(encoding="utf-8")) == code_now, (
        "code_identity no longer treats an appended comment as the same code — "
        "the clause this tier rests on has changed underneath it")
    raw_drift = _drifted(_of_tier(_census(record), walk.TIER_RAW), root=commented)
    code_drift = _drifted(_of_tier(_census(record), walk.TIER_CODE), root=commented)
    assert len(raw_drift) == 1, raw_drift
    assert code_drift == [], (
        f"a comment-only edit was reported as code drift: {code_drift} — the tier "
        f"is stricter than code_identity licenses and will make doc fixes expensive")

    # LEG 2 — a real edit moves both. The licence does not stretch.
    edited = _plant(tmp_path / "edited", rel,
                    source + "\nA_NEW_MODULE_CONSTANT = 1\n")
    assert len(_drifted(_of_tier(_census(record), walk.TIER_RAW), root=edited)) == 1
    assert len(_drifted(_of_tier(_census(record), walk.TIER_CODE), root=edited)) == 1, (
        "a new module-level constant did not move the code digest — the tier "
        "would license a real code change as a comment")

    # LEG 3 — a planted code digest on the untouched tree is reported.
    wrong = {"w": {"code_sha256": {rel: "f" * 64}}}
    reported = _drifted(_of_tier(_census(wrong), walk.TIER_CODE))
    assert len(reported) == 1 and reported[0][1] == rel, reported
    assert reported[0][2].startswith("recorded ffffffffffff live "), reported

    # LEG 4 — a code digest on a file with NO code identity is reported, never
    # skipped. Silently passing an incomputable comparison is the defect class
    # this whole round exists to remove.
    non_python = {"w": {"code_sha256": {"meep_gpu/triton_kernels/fingerprints.json":
                                        "f" * 64}}}
    reported = _drifted(_of_tier(_census(non_python), walk.TIER_CODE))
    assert len(reported) == 1 and "no code identity" in reported[0][2], reported


@pytest.mark.parametrize("field", REQUIRED_METADATA)
def test_every_triton_weld_carries_its_metadata(field):
    """The Metal requirement, on the Triton record. Measured 0/55 missing today.

    Flat, with no budget, because there is no debt to declare: this is a floor a
    future weld must clear, not a backlog. The gap it forecloses is the one the
    CUDA comparison found in certification.json — 0 of 4 blocks carried
    recorded_utc — reaching a third track.

    ASKED OF EVERY RUN RECORD SINCE 0.9.1, not of the entry. All four of these name
    a fact about ONE RUN — when it ran, on what, which artifact it wrote, which
    float32 policy was installed — so after the per-capability migration they live
    in ``runs[<cc>]`` and an entry-level read would find nothing at all. The level
    is DERIVED from ``fastpath.RUN_FIELDS`` and the subset relation is asserted
    here, so a field that is reclassified in ``fastpath.py`` fails this test rather
    than being quietly read from a level the ledger no longer uses.

    THE KEY IS IN THE FAILURE, as ``weld[cc]``: a weld certified on two
    architectures and missing a field on one of them must name which one, because
    the remedy is a run of THAT architecture and nothing else.
    """
    assert field in fastpath.RUN_FIELDS, (
        f"{field!r} is no longer a run field in fastpath.RUN_FIELDS, so this test "
        f"is reading it from the wrong level — decide where it lives and move it "
        f"here in the same change")
    rows, recordless = _weld_runs()
    _assert_the_run_tier_is_walkable(rows, recordless)
    missing = [f"{name}[{capability}]" for name, capability, run in rows
               if not run.get(field)]
    assert not missing, (
        f"{len(missing)} of {len(rows)} Triton run records lack {field}: "
        f"{sorted(missing)}")


def test_every_triton_weld_records_a_timestamp_that_parses():
    """ISO-8601 UTC, not merely a non-empty string.

    A weld's date is what orders it against the source drift around it; a spelling
    no one can parse cannot be compared, and the no-absorber test already reads
    these with a ``>=`` on the raw string, which is only sound while the shape is
    fixed.

    PER RUN RECORD since 0.9.1. The stamp was never a property of the weld — it is
    when one architecture's gate ran — and once two records sit under one entry a
    single stamp could only describe one of them.
    """
    rows, recordless = _weld_runs()
    _assert_the_run_tier_is_walkable(rows, recordless)
    bad = {}
    for name, capability, run in rows:
        stamp = run.get("recorded_utc")
        try:
            datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            bad[f"{name}[{capability}]"] = stamp
    assert not bad, f"run records whose recorded_utc is not ISO-8601 UTC: {bad}"


def test_every_triton_weld_artifact_digest_is_a_sha256():
    """``artifact_sha256`` must be a digest, not a note about one.

    The artifacts live under the gitignored results tree, so nothing here can
    recompute them on a fresh clone — which is precisely why the recorded value
    has to be well formed: it is the only handle a later reader gets. All ten were
    resolved and recomputed by hand on 2026-08-19 and matched.

    ONE ARTIFACT PER RUN, so one digest per run record. This digest is deliberately
    NOT part of ``fastpath.bound_digest`` — the bound is over the bytes a run
    certified, and an artifact is what the run produced — so nothing else in the
    ledger would catch it degrading into a note.
    """
    rows, recordless = _weld_runs()
    _assert_the_run_tier_is_walkable(rows, recordless)
    bad = {f"{name}[{capability}]": run.get("artifact_sha256")
           for name, capability, run in rows
           if not _SHA256.match(str(run.get("artifact_sha256")))}
    assert not bad, f"run records whose artifact_sha256 is not a sha256: {bad}"


def test_every_triton_weld_names_a_validated_triton_and_capability():
    """A Triton weld is a claim about generated PTX, so the toolchain is part of it.

    The record declares ``validated_triton_versions``, and each record's host string
    is bound to it, so a weld cut on an unvalidated Triton fails until the
    declaration is widened deliberately, in the same change.

    THE CAPABILITY HALF NO LONGER READS A DECLARED LIST, and the reason is the whole
    point of the 0.9.1 migration. ``validated_compute_capabilities`` used to be one
    hand-typed key in this record that no tool wrote, so a certification round on
    another architecture moved every weld's evidence and left the declaration saying
    what it had always said. The key is DELETED; the answer is derived from the
    capabilities each cited weld has a live record for
    (``fastpath.validated_compute_capabilities``).

    SO EACH RECORD IS COMPARED AGAINST ITS OWN KEY — a record filed under ``9.0``
    whose host says ``cc 8.6`` is reported — which is strictly stronger than
    membership of any list: the old rule passed every weld in the record as long as
    one global list mentioned the cc each host happened to name, and it could not
    see a record filed under the wrong architecture at all. MEASURED 2026-10-01:
    55 of 55 records name a validated Triton AND their own key.
    """
    record = _record()
    versions = record["validated_triton_versions"]
    assert versions, "the record's own validated_triton_versions is empty"
    assert "validated_compute_capabilities" not in record, (
        "the hand-typed capability declaration is back in the record. It is the "
        "defect the per-capability records replaced — no tool writes it, so it "
        "cannot be wrong in a way anything notices. Delete it and let "
        "fastpath.validated_compute_capabilities derive the answer")
    rows, recordless = _weld_runs(record)
    _assert_the_run_tier_is_walkable(rows, recordless)
    silent = {}
    for name, capability, run in rows:
        host = str(run.get("host", ""))
        if not any(version in host for version in versions):
            silent[f"{name}[{capability}]"] = (
                f"names no validated Triton ({versions}): {host!r}")
        elif capability not in host:
            silent[f"{name}[{capability}]"] = (
                f"is filed under {capability!r} and its host line does not say so: "
                f"{host!r} — one of the two is wrong, and a record filed under an "
                f"architecture its own run never names is the dangerous direction")
    assert not silent, silent


def test_every_triton_weld_names_the_policy_it_was_cut_under():
    """Present is not enough: the field must say ``keep`` or ``flush``.

    Measured 2026-08-19: nine of ten name ``keep`` (four of them with the
    mechanism in parentheses, which is welcome and still names the policy); one
    says ``see artifact`` and the artifact does not say. Under the keep policy the
    kernels' bytes differ in the subnormal range from what a flush run produces,
    so a weld that does not name its policy does not identify its own result — and
    pairing it with the wrong probe is a broken comparison, not a platform verdict.

    PER RUN RECORD since 0.9.1, and the policy is the sharpest case for that: it is
    a property of the environment one run installed, so two architectures' records
    may legitimately disagree about it and a single entry-level field could only
    name one of them. Measured 2026-10-01: three welds short, the three mixed-policy
    gates the budget already names; the debt is unchanged by the migration.

    THE BUDGET IS KEYED BY WELD, not by ``weld[cc]``, deliberately. Each of the
    three is owed a gate that CALLS ``install_ftz_strip`` — see the block comment at
    :data:`POLICY_UNNAMED_BUDGET` — and that debt is about the gate, so it would be
    owed on every architecture it is ever run on. A per-record budget would have to
    grow a line per capability to say the same thing.
    """
    welded = _welds()
    rows, recordless = _weld_runs(welded=welded)
    _assert_the_run_tier_is_walkable(rows, recordless)
    unnamed = {name for name, _capability, run in rows
               if not _POLICY_WORD.search(str(run.get("subnormal_policy", "")))}
    assert unnamed <= set(POLICY_UNNAMED_BUDGET), (
        "welds whose subnormal_policy names no policy and are not declared debt: "
        f"{sorted(unnamed - set(POLICY_UNNAMED_BUDGET))}")
    stale = set(POLICY_UNNAMED_BUDGET) - set(welded)
    assert not stale, (
        f"the debt budget names something that is not a weld: {sorted(stale)} — "
        "a budget entry that outlived its weld hides the next real one")


def test_every_cited_triton_weld_has_a_live_record_for_some_architecture():
    """A CITED WELD WITH NO LIVE RECORD ADMITS NOTHING, and must say so here.

    WHAT LIVENESS IS. A run record carries ``bound_sha256``, the digest of the
    digests that run certified (``fastpath.bound_digest`` over ``BOUND_FIELDS``). A
    record is LIVE while that equals the entry's current bound, so the comparison is
    LEDGER AGAINST LEDGER: an edit that rebinds a weld's source map without
    re-running its gate expires every architecture's evidence at once. Whether the
    ledger matches the TREE is a different claim and stays the job of
    ``test_the_triton_welds_are_bound_to_the_live_sources`` above — this test would
    stay green on a record bound to bytes that have all moved, and that is the right
    split: one tells you the evidence expired, the other that the files did.

    WHY A TABLE'S ANSWER IS THE INTERSECTION. An arm dispatches from whichever
    certified family serves it, so a capability ONE cited family never ran on is a
    capability the table cannot claim. An empty intersection is not "unknown": it
    is a readable ledger whose cited welds share no live architecture, and it
    refuses every device.

    WHAT THIS ASSERTS THAT ``test_capability_records.py`` DOES NOT. That file drives
    the machinery over synthetic ledgers and, in
    ``test_capability_records.py::test_both_nvidia_tables_derive_the_architecture_
    they_were_certified_on``, pins both shipped tables at ``('8.6',)`` through the
    package's own loader. Two things are left over, and they are this file's job
    because this file owns this record: the DENOMINATOR — how many keys the arms
    cite, asserted as a floor, since a collapsed arm map would make that file's loop
    pass over one key — and that the claim holds for the record AS CHECKED IN HERE,
    read from the file rather than through ``fastpath._fingerprints``. The admitted
    set is deliberately NOT pinned to a literal: the next architecture's round must
    widen it by writing records, not by editing this test.

    MEASURED 2026-10-01: 75 arms cite 44 keys, every one live on ``8.6``, admitted
    non-empty, 0 blocked.
    """
    record = _record()
    cited = sorted({gate for _family, gate
                    in fastpath._arm_certification_map("triton").values()})
    assert len(cited) >= CITED_KEY_FLOOR, (
        f"this table's arms cite only {len(cited)} ledger keys, was "
        f"{CITED_KEY_FLOOR} — the arm map SHRANK, and a loop over fewer keys is a "
        f"weaker claim wearing this test's name")
    absent = [key for key in cited if not isinstance(record.get(key), dict)]
    assert not absent, (
        f"{len(absent)} cited key(s) are not entries in this record: {absent} — an "
        f"arm whose certification names nothing dispatches on evidence that is not "
        f"here")
    dead = {key: fastpath.capability_report(record, [key])["by_key"][key]["problem"]
            for key in cited if not fastpath.live_capabilities(record[key])}
    assert not dead, (
        f"{len(dead)} cited weld(s) have no LIVE record for any architecture: "
        f"{dead} — every architecture's evidence for them expired with the bytes "
        f"they bind, so the table admits no device at all. RE-RUN each gate and "
        f"rebind with parity/meep_gpu/rebind_triton_welds.py; do not hand-edit a "
        f"bound_sha256, which would re-point the evidence at bytes no run saw")
    report = fastpath.capability_report(record, cited)
    assert report["admitted"], (
        "this table's cited welds share no live architecture, so "
        "validated_compute_capabilities is empty and every device is refused. The "
        "blockers are named per key in capability_report's by_key")


def test_no_triton_entry_is_left_in_the_shape_the_run_records_replaced():
    """BOTH SHAPES ARE NEVER READ, so neither may be left in the record.

    THE RULE ``fastpath.retired_shape_reasons`` STATES, and the reason it is a
    refusal rather than a compatible read: an entry that carried a run field BESIDE
    its digests and a record under ``runs`` would make "which bytes did this run
    certify" a question with two answers, which is the defect the container closes.
    So a stranded run field, a record keyed by something that is not a normalised
    capability (``sm_90`` normalises to itself and would otherwise key one), a
    record that is not a mapping, a record with no ``bound_sha256``, and an entry
    that binds no bytes at all are each reported by name.

    THE DENOMINATOR IS WIDER THAN THE CITED SET, which is what this adds over
    ``test_capability_records.py::test_no_shipped_nvidia_weld_is_left_in_the_shape_
    this_replaced``: that one walks the 44 keys the arms cite, and this record holds
    55 PASS welds and 77 entries. An uncited weld left half-migrated is a weld the
    next round rebinds against, and a prose block that grows a ``host`` is how a run
    field gets stranded in the first place.

    SO EVERY ENTRY IS EITHER CLEAN OR NAMED, the mechanism this whole file rests on.
    The exemptions are the two tables already here — :data:`NOT_A_GATE_RECORD` and
    :data:`PINNED_BUT_NOT_A_WELD` — because an entry that is not a gate record has
    no run to file. MEASURED 2026-10-01: 55 of 55 welds clean; ten entries report,
    all ten named (eight prose or narrative blocks, plus the ``host_sha256`` and
    ``specialized_kernel_sources`` maps, which are indexes rather than runs).

    THE SLOT CONTENTS ARE CHECKED TOO, and nothing else checks them.
    ``bind_capability`` refuses a foreign key at WRITE time, so a ledger written by
    hand — or by an older tool — can hold a record carrying a field that describes
    the BYTES, which belongs beside the digests where the bound covers it. Measured:
    0 foreign keys over 66 records.
    """
    record = _record()
    welded = _welds(record)
    assert len(welded) >= WELD_FLOOR, len(welded)
    stranded = {name: fastpath.retired_shape_reasons(entry)
                for name, entry in sorted(welded.items())
                if fastpath.retired_shape_reasons(entry)}
    assert not stranded, (
        f"{len(stranded)} PASS weld(s) are still in the shape the per-capability "
        f"records replaced: {stranded} — migrate them with "
        f"parity/meep_gpu/migrate_capability_records.py; do not teach a reader to "
        f"accept both levels")

    declared = set(NOT_A_GATE_RECORD) | set(PINNED_BUT_NOT_A_WELD)
    undeclared = {}
    for name, entry in sorted(record.items()):
        if not isinstance(entry, dict) or name in welded or name in declared:
            continue
        reasons = fastpath.retired_shape_reasons(entry)
        if reasons:
            undeclared[name] = reasons
    assert not undeclared, (
        f"{len(undeclared)} entry(ies) are neither in the per-capability shape nor "
        f"declared a non-gate record: {undeclared} — either file the run under "
        f"runs[<cc>], or name it in NOT_A_GATE_RECORD / PINNED_BUT_NOT_A_WELD with "
        f"the reason it has no run to file")

    foreign = {}
    for name, capability, run in _weld_runs(record, welded)[0]:
        extra = sorted(set(run) - {"bound_sha256"} - set(fastpath.RUN_FIELDS))
        if extra:
            foreign[f"{name}[{capability}]"] = extra
    assert not foreign, (
        f"{len(foreign)} run record(s) hold a field that is not a run field: "
        f"{foreign} — a field describing the BYTES belongs beside the digests, "
        f"where fastpath.bound_digest covers it and the record expires with it; "
        f"inside a record it is a claim no bound protects")

    # ARMED, both legs, because a shape check that has never rejected anything is a
    # shape check nobody has watched work — the complaint this file makes about the
    # eleven days host_sha256 spent behind a citation, in another form.
    clean = {"source_sha256": {"launch.py": "a" * 64},
             fastpath.RUNS: {"8.6": {"bound_sha256": "b" * 64}}}
    assert fastpath.retired_shape_reasons(clean) == []
    pre_migration = {"source_sha256": {"launch.py": "a" * 64}, "host": "the GPU host",
                     "recorded_utc": "2026-09-25T16:09:24Z"}
    reasons = fastpath.retired_shape_reasons(pre_migration)
    assert any("run fields beside the digests" in reason for reason in reasons), reasons
    assert any(f"no {fastpath.RUNS!r} record" in reason for reason in reasons), reasons
    # AND A RECORD KEYED BY A SPELLING THAT IS NOT A CAPABILITY, which is the one
    # the normaliser would pass through unchanged rather than reject.
    mis_keyed = dict(clean, **{fastpath.RUNS: {"sm_90": {"bound_sha256": "b" * 64}}})
    assert any("not a normalised capability" in reason
               for reason in fastpath.retired_shape_reasons(mis_keyed))


def test_no_triton_weld_claims_a_gate_that_refused_on_device():
    """A weld is a record of a PASS. Welding a refusal makes the record lie.

    The Metal counterpart pins this by name (cylindrical_real, whose mutation was
    caught on 5 of 8 launches). The Triton track has its own two, and they are
    found MECHANICALLY rather than named, so the pin maintains itself:
    ``no_pml_composition_gate`` and ``conductivity_composition_gate`` both re-ran
    on 2026-08-14 against the four-family planner wiring and exited rc=1 on a
    stale contract.

    CORRECTED 2026-08-30. This docstring used to say those two were "kept out of
    the weld set by nothing but the ABSENCE of a ``status`` key". That was true,
    and it was the defect: the same absence kept them out of the BYTE check as
    well, where they had eight and three live drifts respectively. They are now
    checked for bytes like every other pinned entry, and their exclusion from the
    METADATA tier is written down by name in :data:`PINNED_BUT_NOT_A_WELD` with
    the rc=1 as the reason. What this test still stops is the other direction —
    stamping one ``PASS`` to silence a red would weld a device failure.
    """
    record = _record()
    refused = _refused(record)
    assert refused, (
        "no recorded device failure found at all — the rc/"
        "failure_measured_on_device shape changed and this pin is now vacuous")
    welded_refusals = sorted(set(refused) & set(_welds(record)))
    assert not welded_refusals, (
        f"welded despite a recorded device failure: "
        + "; ".join(f"{name} {refused[name]}" for name in welded_refusals))




def test_every_triton_weld_pins_the_script_that_gated_it():
    """A weld that does not pin its own probe cannot say what produced it.

    MEASURED 2026-08-30: 30 of 31 Triton welds and 44 of 44 Metal welds pin a
    script under ``parity/meep_gpu/``; exactly one did not, and it was the weld
    the previous round ADDED —
    ``triton_folded_deposit_closure_device_gate``. ``probe_triton_folded_deposit_
    closure.py`` was pinned by no entry in any of the three records, although the
    gate's own artifact recorded it in ``imported_source_sha256``.

    WHY IT MATTERS SEPARATELY FROM THE SOURCES. The sources say which
    implementation ran; the probe says what was ASKED of it. The closure gate's
    verdict is "eleven of twelve cases armed, leg B diverges on all eleven, the
    on-plane control vacuous" — every one of those numbers is a property of the
    probe, so a probe that changes underneath an unpinned weld rewrites the
    result the weld reports while every source digest still verifies. That is the
    same failure the source pins exist to prevent, on the other half of the run.
    """
    welded = _welds()
    assert len(welded) >= WELD_FLOOR, len(welded)
    naked = sorted(name for name, entry in welded.items()
                   if not any(path.startswith("parity/")
                              for path in entry["source_sha256"]))
    assert not naked, (
        f"{len(naked)} of {len(welded)} Triton welds pin no gate or probe script "
        f"under parity/meep_gpu/: {naked} — add the path the gate ran and let "
        f"rebind_triton_welds.py bind it from the run's own "
        f"imported_source_sha256; do not type the digest")


#: The weld that carries the deposit IMAGE CLOSURE evidence, and the two products
#: whose 2026-08-30 ``CARRIES_DEPOSIT_REPAIR`` flip it licenses. Named here rather
#: than derived, because the point of the assertion is that THESE two are cited.
CLOSURE_WELD = "triton_folded_deposit_closure_device_gate"
CLOSURE_PRODUCTS = ("meep_gpu/triton_kernels/folded_fused_pair.py",
                    "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py")


def test_the_deposit_closure_weld_cites_the_products_whose_flip_it_licenses():
    """The R1 flip's device evidence is IN the record, bound to the flipped bytes.

    The 2026-08-30 round flipped ``CARRIES_DEPOSIT_REPAIR`` False->True on three
    Triton products and cited ``probe_triton_symmetry_composition``, which is
    vacuous for the mechanism: it plans with ``fuse=False``, only two of its six
    cases declare a source, and its source sits at the grid centre — which on a
    mirrored axis IS the mirror plane, so the deposit is its own image and a
    point-only repair is exact there by construction. This pins that the record
    now carries evidence that actually exercises the closure, and that the weld
    is bound to the two products the evidence covers.

    ONLY TWO OF THE THREE, deliberately. ``complex_fused_magnetic_pair`` cannot be
    shown to need the closure — its predicate refuses every mirrored axis by name,
    so no configuration it admits runs a post-injection fill and the closure and a
    point repair are the same set there. Its flip is licensed by the POINT repair
    alone, which is correct because there are no images. That limit is pinned in
    ``meep_gpu/test_folded_deposit_closure.py``, not papered over here.
    """
    welded = _welds()
    assert CLOSURE_WELD in welded, (
        f"{CLOSURE_WELD} is not a PASS weld — the deposit-closure evidence is not "
        f"in the record, so the flip cites nothing that exercises the closure")
    pinned = welded[CLOSURE_WELD]["source_sha256"]
    missing = [name for name in CLOSURE_PRODUCTS if name not in pinned]
    assert not missing, (
        f"{CLOSURE_WELD} does not pin {missing} — a weld that does not name the "
        f"flipped bytes cannot be the evidence for flipping them")
    assert "meep_gpu/deposit_repair.py" in pinned, (
        f"{CLOSURE_WELD} does not pin deposit_repair.py, which IMPLEMENTS the "
        f"closure (fill_image_rules / repair_cells) the gate scores")
    # The strict byte comparison above already proves these digests are live; this
    # only refuses the placeholder an unbound skeleton would still be carrying.
    unbound = sorted(name for name, digest in pinned.items() if digest == "0" * 64)
    assert not unbound, (
        f"{CLOSURE_WELD} still carries unbound placeholder digests for {unbound} — "
        f"rebind_triton_welds.py did not bind it from a released artifact")


# ---------------------------------------------------------------------------
# THE CAMPAIGN THAT MADE THE STRICT CHECK LANDABLE — 2026-08-30
# ---------------------------------------------------------------------------
#
# The strict comparison above fired on twenty-one arm-level welds, all on one
# path (``triton_kernels/launch.py``, 2550edbc -> 45373307) and on nothing else.
# They were fixed by RE-RUNNING, never by editing the record. Campaign:
#
#     parity/meep_gpu/results/triton_weld_close_2026-08-30/
#
# Sixteen came from the deposit-carry round's own arms0/6/7, whose artifacts had
# already imported the current launch.py and agreed with this checkout on every
# digest they recorded. FIVE had refused there, and every one of the five turned
# out to be an ENVIRONMENT fault in the runner rather than anything about the
# kernels. All five released on re-run, each reporting exactly the imported-digest
# count its previous weld recorded (no_pml 40, complex_no_pml_curl 32,
# complex_no_pml_conductive 35, complex_offdiag 36, fused_ade_state 35):
#
#   * ``no_pml`` — its ENGINE leg does ``import cases``, the shared MEEP builder
#     that lives in a results script bundle and on no package path. The staged
#     tree excluded ``results/``, so the leg died with ModuleNotFoundError after
#     the other legs had passed, and the gate wrote an artifact with no
#     ``validation`` block at all — which read as UNREADABLE, not as refused.
#     Fixed by putting that scripts directory on PYTHONPATH. The driver's own
#     docstring already warned about this exact import.
#   * ``complex_no_pml_curl`` and ``complex_no_pml_conductive`` — these take NO
#     ``--probe`` flag; they load through ``probe_residual_group_bodies.
#     _load_expansion_probe``, which reads a fixed path under ``results/`` and
#     falls back to ``parity/meep_gpu/expansion_probe.json``. With ``results/``
#     unstaged, NEITHER existed, so the licence was absent and both refused with
#     "no EXPANSION constexpr was licensed from a probe artifact". Fixed by
#     staging a freshly measured keep-cut probe at the fallback path.
#
#     THE FRESH PROBE WAS NOT REQUIRED, and saying so matters because otherwise
#     the next reader concludes a fresh cut is a precondition. The LEGACY artifact
#     the constant points at is in the checkout and licenses the same arm:
#     ``expansion_license`` on it returns expansion=1, arm=FMA_V1, basis=measured,
#     refusals=[] over the base four patterns. Its ``_resolve_expansion`` reads
#     None on a laptop only because no policy is installed there, and the two
#     reasons it gives are both about the POLICY IN FORCE, not the artifact. A
#     full checkout would therefore have released these two as well; the staged
#     tree simply had neither file. The fresh probe was chosen because a probe cut
#     in this campaign, on this device, under this policy is stronger provenance
#     than one carried in from 2026-08-12 — not because it was needed.
#
#     CORRECTION TO WHAT THIS FILE PREVIOUSLY SAID. The earlier note here claimed
#     these two "need a probe carrying c8_mul_c8_parity_coefficient_left, which
#     only the unified expansion record classifies". That is wrong, and it was
#     wrong as a guess rather than as a measurement: both resolve their arm from
#     the BASE FOUR patterns and released against a probe chosen for being fresh,
#     not for being extended. Only ``complex_offdiag`` needs the parity pattern.
#   * ``complex_offdiag`` — the one the extended pattern set really does gate. Its
#     folded products bind ('c8_mul_c8', 'c8_mul_f4_field_left',
#     'f4_mul_c8_coefficient_left', 'python_float_left',
#     'c8_mul_c8_parity_coefficient_left') and the ``complex`` gate's probe
#     licenses only the base four, so the shipped builder refused every folded
#     product. Fixed by handing it the FOLDED_COMPLEX probe, which the driver's
#     own ``Gate`` note has recorded all along and does not apply by default.
#   * ``fused_ade_state`` — not a driver gate at all; it ships no
#     ``gate_triton_*.py`` and is measured by ``probe_triton_fused_ade_state.py``
#     run directly, so the campaign never invoked it. Run with the direct
#     runner's own environment (fresh private caches carrying the keep token).
#
# The board was re-cut on the rebound record and did not move: 262 served, and
# every aggregate identical to the carry board. That is the expected result and
# it is worth saying why — these twenty-one are ARM-level welds, credited by no
# board, which is exactly why ``assert_every_credit_is_bound_to_released_bytes``
# caught the fused products' drift on the same day and stayed silent about these.
# A number that had moved here would have meant the drift reached a credited cell.
