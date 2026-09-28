"""Every CUDA weld is bound to the bytes its gate executed — and says so honestly.

UNTIL 2026-08-30 THIS RECORD WAS CHECKED BY NOTHING. ``cuda_kernels/
fingerprints.json`` holds twenty entries, each naming a released device campaign,
its host, its two canonical policy legs, an ``artifact_sha256`` and the sha256 of
every tree file that campaign imported. No Python file in ``meep_gpu/`` read it:
there is no other ``test_cuda_weld_contract``, and ``cuda_kernels/
test_certification_*.py`` cover ``certification.json`` — a different record with a
different job — mentioning their Triton and Metal siblings only in prose. The only
reader in the tree was ``parity/meep_gpu/rebind_cuda_welds.py``, which WRITES it.
A ledger whose sole reader is its own writer is honest self-declaration with
nothing enforcing it, which is the state this file ends.

MEASURED THE DAY IT WAS WRITTEN, by recomputing every digest independently:

    entries  20      pins  79      drifted  40      status PASS  0

So this module landed RED, deliberately and on the whole denominator. That was the
correct result for a ledger nothing had been checking: the drift was real, it was
what the record already said about itself in nineteen of twenty entries, and it
was only ever clearable by RE-GATING on a CUDA host and rebinding with
``rebind_cuda_welds.py`` — never by relaxing anything here and never by typing a
digest.

CLEARED 2026-08-30, BY THE ONLY ROUTE THIS FILE ALLOWS. All twenty campaigns were
re-run on the GPU host (NVIDIA RTX A6000, both float32 subnormal policies, 44
canonical policy legs, every one released) into
``results/cuda_regate_2026-08-30/``. The measurement had in fact existed for most
of that day and could not be used: ``rebind_cuda_welds.py`` finds a campaign's
artifacts by asking ``certification.json`` where that campaign wrote them, and no
block named the re-gate tree. THE BLOCKER WAS A MISSING WRITER, NOT A MISSING
MEASUREMENT. It is closed by ``record_cuda_regate.py``, which transcribes a
re-gate campaign into a certification block — every fact derived from the
payloads — and by ``rebind_cuda_welds.py --campaign``, which refuses to bind
against a tree the record does not name. Nothing was installed over anything: an
earlier attempt to clear the same drift copied the fresh legs INTO the dated
directories, reached zero drift, and broke four certification tests, because each
block's ``artifact_sha256`` is computed over its own tree. Both writers now
recompute every present block's digest before writing a byte and refuse on the
first disagreement.

    entries  20      pins  79      drifted  0       status PASS  20

WHAT MAKES AN ENTRY CHECKABLE. The same rule as the Triton contract, for the same
reason: AN ENTRY IS CHECKED IF IT PINS A FILE DIGEST. Not whether it passed, not
whether it carries a status. Every one of the twenty here says ``DRIFTED``, so a
``status == "PASS"`` filter — the shape that hid ten Triton entries and 88 pinned
pairs — would have made this entire module vacuous while reading as strict. An
entry that is deliberately not a weld is excluded BY NAME in
:data:`NOT_A_WELD_LEDGER_ENTRY`, with the reason, and
``test_every_record_key_is_checked_or_excluded_by_name`` requires that of every
key in the file.

THE SECOND CLAIM, and it is the one that bites while the re-gate is outstanding.
``_what_status_means`` in the record says a DRIFTED entry lists the moved files in
``source_drift``, and that ``code_sha256`` deliberately omits them because
``code_identity`` licenses a weld to stand on an unchanged code digest ONLY where
``source_sha256`` still matches. Both halves are checkable against the tree
without a device, and both are checked below: a self-declaration that is allowed
to be WRONG is not a weaker check than no check, it is a stronger-looking one.

THE WALK, added 2026-08-30 (third round). The enumerator this file landed with
read ``entry["source_sha256"]`` per top-level key — a FIXED SHAPE, and the same
class of defect the status filter was, one level down. It compared 79 of the
record's 138 digests. The 59 it did not look at:

    39  code_sha256   which this file DID check, uniquely among the three tracks,
                      through test_no_entry_backfills_a_code_digest_beside_a_stale
                      _source_digest — but only as a self-declaration rule, and
                      only for entries the shape enumerator had already found.
    20  artifact      digests of gitignored run artifacts, excluded by name.

``meep_gpu.weld_record_walk`` now recurses the whole record — dicts and lists, at
any depth — and every 64-hex digest is either CHECKED against this tree in a named
tier or EXCLUDED BY A NAMED RULE carrying its reason. A digest matching neither is
``UNCLASSIFIED``, which fails. Measured 2026-08-30, third round and then fifth:

    discovered 138   checked 118   excluded 20   unclassified 0   drifted 40
    discovered 178   checked 158   excluded 20   unclassified 0   drifted  0

The second line is the re-gate. The 40 new digests are not new claims: they are
the ``code_sha256`` entries the tool had been WITHHOLDING on every drifted file,
because ``code_identity`` licenses a code digest beside a source digest only
while the source digest still matches. Binding to the run that imported those
bytes made all 79 honest to record at once.

Stdlib, pytest, and two in-package identity helpers: this is a laptop merge-bar
test and touches no device.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import pathlib
import re

import pytest

from . import weld_record_walk as walk
from .code_identity import code_digest
from .device_identity import device_digests

PACKAGE = pathlib.Path(__file__).resolve().parent
RECORD = PACKAGE / "cuda_kernels" / "fingerprints.json"
#: the repository root, the root every pinned path in this record is relative to. Unlike the
#: Triton ledger, every CUDA pin is already repo-relative — there is no
#: bare-basename spelling here — so no resolver is needed and none is provided.
REPO = PACKAGE.parent

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Top-level keys that carry no pin and are excluded BY NAME. Every one is a
#: narrative string rather than a block; they are listed individually anyway,
#: because "it is not a dict" is exactly the kind of shape test that stops being
#: true the day someone promotes a note to a block.
NOT_A_WELD_LEDGER_ENTRY = {
    "_a_hole_this_ledger_does_not_inherit":
        "prose: how certification.json's block enumerator can miss a block",
    "_relation_to_certification_json":
        "prose: one record, two jobs — which file is written by whom",
    "_the_two_block_layout":
        "prose: the mechanical/narrative split inside an entry",
    "_what_is_absent_and_why":
        "prose: absence is the claim; a block with no released verdict gets no "
        "entry rather than a pending one",
    "_what_status_means":
        "prose: the PASS/DRIFTED definitions — CHECKED as a rule by "
        "test_cuda_weld_contract.py::test_the_status_is_the_one_the_tree_supports "
        "and test_cuda_weld_contract.py::"
        "test_the_declared_source_drift_is_the_measured_source_drift",
    "_what_this_is": "prose: what the ledger is for",
    "_which_blocks_are_absent_today":
        "prose: which certification.json blocks have no entry here and why",
    "_which_is_authoritative_for_what":
        "prose: certification.json for WHAT was certified, this for which bytes",
    "_why_this_file_exists_at_all":
        "prose: why device source strings are a stronger claim than a host digest",
}

#: The metadata every entry must carry. Same four the Metal and Triton contracts
#: require, plus the three this ledger's own shape adds: which legs were read,
#: where the verdict was read from, and which kernels the campaign covers.
#:
#: THE INCOMING ``driver_dispatch`` ENTRY IS HELD TO ALL NINE, WITH NO CARVE-OUT, and
#: the absence of a carve-out is the decision rather than an oversight. That block
#: pins ``meep_gpu/`` source (``fastpath.py``, ``driver.py``, ``fields.py``,
#: ``fastpath_cuda.py``, three ``cuda_kernels`` files and every released arm's product
#: module), which is what makes an entry a WELD here; a "it is only a route record"
#: exemption would let the one entry that licenses SELECTION be the one entry that
#: names no device, no legs and no policy. Its ``subnormal_policy`` names BOTH legs
#: (the dispatching ``ieee_keep_ftz_stripped`` and the ``meep_x86_flush`` leg refused
#: by name at rung 8b), its ``legs`` lists all five leg paths, and it is cut by
#: ``recut_driver_dispatch_record.py --backend cuda`` from the campaign artifact and
#: by nothing else — ``rebind_cuda_welds.py`` reports it and refuses to rewrite it.
#: The floors below rise with it and with the campaign's seeds; they are FLOORS, so
#: they fail when coverage shrinks and never when it grows.
REQUIRED_METADATA = ("recorded_utc", "host", "artifact_sha256", "subnormal_policy",
                     "records", "legs", "verdict_read_from", "kernels",
                     "kernel_module")

#: Non-vacuity floors, measured 2026-08-30 by independent recomputation. THE
#: DENOMINATOR IS PART OF THE CHECK: these fail when coverage SHRINKS, so an
#: entry cannot leave the strict comparison by dropping a key — the mechanism
#: that would have caught the Triton status filter eleven days earlier.
ENTRY_FLOOR = 20
PIN_FLOOR = 79

#: THE TOTAL, and the tier floors under it. DISCOVERED is the one the others
#: cannot substitute for: a digest that stops being FOUND also stops being counted
#: by every floor derived from the walk, so only the total can report it.
#:
#: RAISED 2026-08-30 (fifth round) when the re-gate cleared the drift, and the
#: SHAPE of the rise is the point. Nothing was added to the ledger: the code tier
#: went 39 -> 79 because ``code_sha256`` had been WITHHELD on every drifted file
#: — ``code_identity`` licenses a code digest beside a source digest only while
#: the source digest still matches, so 40 of the 79 pins carried no code digest
#: at all. Binding them to the run that actually imported those bytes made all 79
#: honest to record, and the discovered total rose by exactly the same 40. A
#: floor left at 39 would have let the code tier fall back to a third of the
#: record without anything saying so.
DISCOVERED_FLOOR = 178
CHECKED_FLOOR = 158
RAW_TIER_FLOOR = 79
CODE_TIER_FLOOR = 79

#: Which rules the walk fires on THIS record, and how many times at the floor.
#: BOTH DIRECTIONS: a rule that stops firing means its shape has left the record,
#: and one that starts firing means a shape arrived that nobody wrote down.
#: ``historical_recut_log`` ARRIVED 2026-09-11, and it is the first history this
#: ledger has ever carried. ``recut_driver_dispatch_record.py`` writes one
#: ``_source_sha256_recut_log`` entry per re-cut, each row a ``from``/``to`` pair over
#: a file whose digest moved between two cuts of the SAME weld — so half of every
#: pair is stale BY CONSTRUCTION and a log that had to match the tree could not
#: record a change at all. The exemption is earned by
#: ``test_a_historical_exclusion_is_earned_by_a_live_checked_claim`` below, ported
#: from the Triton contract, which is what the old
#: ``test_this_ledger_carries_no_superseded_or_historical_digests`` asked for by name
#: on the day this shape appeared.
EXPECTED_RULES = {
    walk.TIER_RAW + ":source_digest_map": 79,
    walk.TIER_CODE + ":code_digest_map": 79,
    "artifact_or_log_digest": 20,
    "historical_recut_log": 6,
}


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _census(record=None):
    """EVERY digest in the record, classified. The denominator, structurally derived.

    THE ONE ENUMERATOR. No ``record.items()`` loop that assumes a top level, no
    ``entry["source_sha256"]`` that assumes a key. What the record holds is what
    gets classified, and a digest the classifier has no rule for is
    ``UNCLASSIFIED``: a failure, not silence.
    """
    return walk.census(_record() if record is None else record)


def _checked(found):
    return [f for f in found if f.tier is not None]


def _of_tier(found, tier):
    return [f for f in found if f.tier == tier]


#: Ledger entries that own checked digests but are NOT welds, and why. A weld is a
#: mapping from a DEVICE VERDICT to the bytes its gate ran, so it carries a host, an
#: artifact digest and a timestamp; these carry pins without a run behind them and
#: the metadata floor does not apply to them. Named individually, with the reason, so
#: a future entry cannot join the exemption by accident.
#:
#: The Triton ledger's contract exempts the same block by the same argument
#: (``test_triton_weld_contract`` NOT_A_WELD: "the dispatch WIRING record ... it has
#: no host and no artifact because no gate produced it"). ``driver_dispatch`` reached
#: this ledger on 2026-09-11 when the CUDA route campaign first cut one, and the
#: exemption had never been written here because the block had never existed.
NOT_A_WELD = {
    "driver_dispatch":
        "the dispatch WIRING record -- which consult sites exist, which slots the "
        "driver can run, and what the arbitration between the two NVIDIA tables "
        "measured. Its source digests ARE checked, on the same terms as a weld's; "
        "what it has no business carrying is a host, an artifact digest or a gate "
        "timestamp, because no single gate produced it: "
        "recut_driver_dispatch_record.py cuts it from a five-leg campaign.",
}


def _entries(record=None):
    """Every entry that owns a CHECKED digest. No status filter — see the docstring.

    Derived from the census rather than from the presence of a ``source_sha256``
    key, so an entry whose only pins live in a ``code_sha256`` map or a shape
    nobody has invented yet is in the denominator on the same terms.
    """
    record = _record() if record is None else record
    owners = {f.trail[0] for f in _checked(_census(record))}
    return {name: entry for name, entry in record.items()
            if name in owners and isinstance(entry, dict)}


def _weld_entries(record=None):
    """The entries the WELD METADATA floor applies to: :func:`_entries` less
    :data:`NOT_A_WELD`. Their digests stay in every byte-level denominator -- a
    record that pins bytes is checked on the same terms as a weld -- and only the
    host/artifact/timestamp floor, which presumes a single gate run, is lifted."""
    return {name: entry for name, entry in _entries(record).items()
            if name not in NOT_A_WELD}


def _pins(record=None):
    """``(owner, path, digest)`` for every raw-tier digest — the byte denominator.

    ``owner`` is the top-level entry name; the full trail is what
    :func:`walk.compare` reports on a failure, which is where it is needed.
    """
    return [(f.trail[0], f.subject, f.value)
            for f in _of_tier(_census(record), walk.TIER_RAW)]


def _raw_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest_of(path):
    """``None`` for a file with no code identity, which :func:`walk.compare` reports.

    ``code_digest`` parses Python and nothing else, so a ``code_sha256`` recorded
    for a ``.cu``, a shader or a JSON is a digest nothing can recompute. Returning
    ``None`` rather than falling back to a byte hash keeps that a REPORT instead of
    a silently-passing comparison.
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
    """Every CHECKED digest the tree does not support, across every tier present.

    THE RULE IS THE RAW BYTES in the raw tier, with no allowance and no
    fall-through to a laxer digest — the Metal rule, brought across unchanged.
    ``weld_survives_edit`` is deliberately NOT called: whether an edit is harmless
    is a judgement made on the host about a claim only the device can settle.
    ``code_sha256`` still has a job in this record, and it is compared to the LIVE
    FILE'S CODE below, but it is never a licence to skip the byte comparison.

    Factored out because an assertion nobody has watched FAIL is a measurement
    that cannot fail: ``test_the_drift_check_refuses_a_planted_digest`` drives
    this same function with a deliberately wrong digest and requires a report.
    """
    return walk.compare(
        found,
        resolve=lambda name: root / name,
        raw_digest=_raw_digest,
        code_digest_of=_code_digest_of,
        device_digests_of=_device_digests_of,
        rebind_tool="parity/meep_gpu/rebind_cuda_welds.py",
    )


def _measured_drift(name, entry, root=REPO):
    """The set of paths this entry pins in the RAW tier that no longer match the tree.

    Derived from the census over ``{name: entry}`` rather than from
    ``entry["source_sha256"]``, so a raw pin the entry grows in some other shape is
    measured here too. ``source_drift`` claims to be the whole truth about this
    entry's exposure; a measurement that only looked at one key could not check
    that claim, it could only check the part of it the same key describes.
    """
    raw = _of_tier(_census({name: entry}), walk.TIER_RAW)
    return {subject for _, subject, _ in _drifted(raw, root=root)}


# ---------------------------------------------------------------------------
# THE DENOMINATOR
# ---------------------------------------------------------------------------


def test_every_record_key_is_checked_or_excluded_by_name():
    """Nothing leaves coverage by omitting a field.

    A future entry that forgets ``status`` is checked anyway; one that pins no
    digest has to be justified here, in the same change, where a reviewer sees
    it. Both directions: an exclusion that outlives its key is deleted too,
    because a stale name is where the next real omission hides.
    """
    record = _record()
    checked = set(_entries(record))
    unaccounted = sorted(set(record) - checked - set(NOT_A_WELD_LEDGER_ENTRY))
    assert not unaccounted, (
        f"{len(unaccounted)} ledger keys are neither checked nor excluded by "
        f"name: {unaccounted} — pin the bytes the campaign imported, or name "
        f"each here with the reason it is not a weld")
    stale = sorted(set(NOT_A_WELD_LEDGER_ENTRY) - set(record))
    assert not stale, f"the exclusion list names keys the record no longer has: {stale}"
    overlap = sorted(checked & set(NOT_A_WELD_LEDGER_ENTRY))
    assert not overlap, (
        f"{overlap} are excluded by name AND pin bytes — delete the exclusion, "
        f"not the coverage")


def test_the_checked_denominator_has_not_shrunk():
    """The count itself is asserted, not just the comparison over whatever is left.

    A strict assertion over a filtered subset is a weak check wearing a strong
    one's clothes. This is the guard that makes the strictness above mean
    something: twenty entries and seventy-nine pins were measured on the day this
    file was written, and a record that presents fewer fails here rather than
    passing a shorter loop.
    """
    entries, pins = _entries(), _pins()
    assert len(entries) >= ENTRY_FLOOR, (
        f"only {len(entries)} CUDA weld entries own a checked digest, was "
        f"{ENTRY_FLOOR} — coverage SHRANK")
    assert len(pins) >= PIN_FLOOR, (
        f"only {len(pins)} (entry, path) pairs to compare across "
        f"{len(entries)} entries, was {PIN_FLOOR} — the shape changed and this "
        f"module is now inert")


def test_every_discovered_digest_is_checked_or_excluded_by_a_named_rule():
    """THE WALK. Every digest in the ledger, wherever it lives, has a fate.

    THE TEST THAT REMOVES THE CLASS, and this module needed it as much as its
    siblings did: it landed with a shape enumerator that read one key per
    top-level entry, and so compared 79 of the ledger's 138 digests while reading
    as strict. Here the record is RECURSED — dicts and lists, at any depth — and
    each digest is either CHECKED in a named tier or EXCLUDED BY A NAMED RULE
    carrying its reason. There is no third outcome: a digest no rule claims is
    ``UNCLASSIFIED``, and that is this assertion failing.

    THE TOTAL IS ASSERTED, not only the checked count: a digest that stops being
    DISCOVERED is the failure no count of what is checked can see, because it
    leaves every count derived from the walk at the same time.
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
        f"or exclude it by name with the reason")
    checked = _checked(found)
    assert len(checked) >= CHECKED_FLOOR, (
        f"only {len(checked)} of {len(found)} discovered digests are checked, was "
        f"{CHECKED_FLOOR} — digests moved from checked to excluded, which is a "
        f"decision that has to be argued, not a count that may drift")
    fired = walk.rules_that_fired(found)
    reasonless = sorted(set(fired) - set(walk.RULE_REASONS))
    assert not reasonless, f"rules fired with no written reason: {reasonless}"
    missing = sorted(set(EXPECTED_RULES) - set(fired))
    assert not missing, (
        f"declared rules that no longer fire on this record: {missing} — the "
        f"shape each was written for has left, and a rule nothing exercises is "
        f"where the next real omission hides")
    surprise = sorted(set(fired) - set(EXPECTED_RULES) - {walk.UNCLASSIFIED})
    assert not surprise, (
        f"rules fired that this record never declared: {surprise} — a new shape "
        f"arrived; declare it in EXPECTED_RULES with its count")
    shrunk = {name: (fired[name], floor) for name, floor in EXPECTED_RULES.items()
              if fired[name] < floor}
    assert not shrunk, (
        f"rules whose digest count SHRANK (got, was): {shrunk} — coverage does "
        f"not fall silently")


def test_a_historical_exclusion_is_earned_by_a_live_checked_claim():
    """HISTORY IS EXEMPT FROM THE TREE — and here is the price of that exemption.

    UNTIL 2026-09-11 THIS LEDGER CARRIED NO HISTORY AT ALL, and the test in this
    slot asserted exactly that: an entry that grew a ``superseded_runs`` list would
    have inherited the Triton exclusion silently, because the rule lives in the
    shared walker and fires on SHAPE. Its failure message named the remedy — "bring
    the earning legs across from test_triton_weld_contract.py" — and this is that
    change, made on the day the shape arrived.

    WHAT ARRIVED, and why it cannot be checked against the tree.
    ``recut_driver_dispatch_record.py`` writes ``_source_sha256_recut_log``: one
    block per re-cut of the driver-dispatch weld, each row a ``from``/``to`` pair
    over a file whose digest moved between two cuts. Half of every pair is stale BY
    CONSTRUCTION — a log that had to match the tree could not record a change — and
    the only way to clear a red raised against it would be to DELETE THE HISTORY,
    which is the one repair that destroys evidence.

    THE PRICE, which is what stops this being a hole. Three legs, all checked here:

      1. A recut block must SAY what it re-cut, when, and from which artifact
         (``why``, ``recut_utc``, ``cut_from``, each non-empty) and must name at
         least one moved file. A block that merely SITS in the list gets nothing.
      2. The entry containing history must carry a LIVE CHECKED digest of its own.
         So moving a weld's last live pin into its own log does not buy silence: it
         empties the entry, which then has to be argued for in this file.
      3. Every historical digest must be attributable to a top-level entry — no
         orphan history at the record root.

    Measured 2026-09-11: 1 recut block over 3 files, 6 digests, under
    ``driver_dispatch``, which carries 4 live raw pins and 4 live code pins.
    """
    found = _census()
    historical = [f for f in found if f.rule.startswith("historical_")]
    assert historical, (
        "no historical digests found — the container this test argues about has "
        "left the record, and with it the exemption. Restore the assertion that "
        "this ledger carries NO history, or find where the recut log went")

    record = _record()

    # LEG 1 — a recut block says what it re-cut, when, and from what.
    silent = []
    for name, entry in sorted(record.items()):
        if not isinstance(entry, dict):
            continue
        for key, value in sorted(entry.items()):
            if not key.endswith("_recut_log") or not isinstance(value, list):
                continue
            for index, block in enumerate(value):
                if not isinstance(block, dict):
                    silent.append(f"{name}.{key}[{index}] is not a block")
                    continue
                missing = [field for field in ("why", "recut_utc", "cut_from")
                           if not block.get(field)]
                if missing:
                    silent.append(f"{name}.{key}[{index}] declares no {missing}")
                    continue
                # SHAPE, NOT PRESENCE. A truthy string satisfies "declares itself"
                # while saying nothing, so each field is held to the shape its writer
                # produces: an ISO-8601 Z stamp that parses, a path under the results
                # tree, and rows that are {file, from, to} with sha256-shaped digests
                # on both sides. Without this the leg admitted `why: "x"`,
                # `recut_utc: "soon"` and `cut_from: "somewhere"`.
                stamp = str(block["recut_utc"])
                try:
                    datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    silent.append(f"{name}.{key}[{index}].recut_utc {stamp!r} is not "
                                  f"an ISO-8601 Z stamp, so when it was cut is not "
                                  f"actually recorded")
                if "parity/meep_gpu/results/" not in str(block["cut_from"]):
                    silent.append(f"{name}.{key}[{index}].cut_from "
                                  f"{block['cut_from']!r} names no run under "
                                  f"parity/meep_gpu/results/")
                if len(str(block["why"]).split()) < 8:
                    silent.append(f"{name}.{key}[{index}].why is {len(str(block['why']).split())} "
                                  f"words; a reason that short is a placeholder")
                rows = block.get("files")
                if not rows:
                    silent.append(f"{name}.{key}[{index}] names no moved file, so "
                                  f"it records no change")
                    continue
                for position, moved in enumerate(rows):
                    if not isinstance(moved, dict):
                        silent.append(f"{name}.{key}[{index}].files[{position}] is "
                                      f"not a block")
                        continue
                    if not moved.get("file"):
                        silent.append(f"{name}.{key}[{index}].files[{position}] "
                                      f"names no file")
                    for side in ("from", "to"):
                        digest = moved.get(side)
                        if not (isinstance(digest, str) and len(digest) == 64
                                and all(c in "0123456789abcdef" for c in digest)):
                            silent.append(
                                f"{name}.{key}[{index}].files[{position}].{side} is "
                                f"{digest!r}, not a sha256 -- a from/to pair that is "
                                f"not two digests records no move")
    assert not silent, (
        f"{len(silent)} recut block(s) claim the exemption without declaring it: "
        f"{silent} — carry why, recut_utc, cut_from and the files that moved, or "
        f"have the digests checked like everything else")

    # LEG 2 — history is only exempt where a current claim stands beside it.
    live_owners = {f.trail[0] for f in _checked(found)}
    unearned = sorted({f.trail[0] for f in historical} - live_owners)
    assert not unearned, (
        f"{unearned} carry HISTORICAL digests and no live checked digest at all — "
        f"an entry whose only pins are in its own past has stopped making a claim "
        f"about this tree. THE REMEDY IS TO RE-GATE AND PIN THE LIVE BYTES. Naming it "
        f"in NOT_A_WELD_LEDGER_ENTRY does NOT clear this leg -- that table is read by "
        f"the entry-accounting test, not here, and the Triton original's "
        f"corresponding exemption set (NOT_A_GATE_RECORD) was deliberately not ported "
        f"because this ledger has no entry that has stopped gating. If one ever does, "
        f"the exemption has to be added HERE, in the same change, with its reason")

    # LEG 3 — every historical digest is OWNED BY AN ENTRY, so leg 2 can reach it.
    #
    # WIDENED 2026-09-11 by its own planted control, which is the reason the controls
    # exist. The ported form asked only ``len(trail) < 2`` — a digest literally at
    # depth one — and a recut log hung at the record ROOT produces
    # ``_source_sha256_recut_log.0.files.0.from``, depth five, which sailed past it
    # while being exactly the orphan the leg is named for. What leg 2 needs is not
    # depth but an OWNER: a top-level key whose value is an entry (a dict), because
    # leg 2's live-checked set is keyed on ``trail[0]``. A list or a string at the
    # root owns nothing and cannot be asked for a live pin.
    rootless = sorted(walk.dotted(f.trail) for f in historical
                      if len(f.trail) < 2
                      or not isinstance(record.get(f.trail[0]), dict))
    assert not rootless, (
        f"{rootless} are not owned by a top-level entry, so no live checked digest "
        f"can ever stand beside them and leg 2 cannot ask anything of them")

    # LEG 4 — THE EXEMPTION IS SHAPED, so it cannot be used as a hiding place.
    #
    # THE HOLE THIS CLOSES, and it was opened by the same change that admitted the
    # rule. ``weld_record_walk`` classifies on the TRAIL: any digest whose path
    # contains a key ending ``_recut_log`` is historical, decided FIRST, before every
    # tier rule, with no requirement on where under that key it sits. So a live pin
    # whose file had moved could be cleared by RELOCATING it — lift ``source_sha256``
    # out of the entry and drop it inside the entry's own recut log, and the drift
    # leaves the tree comparison while legs 1-3 and every floor stay green. Legs 1-3
    # ask whether the log DECLARES itself; none of them asks what is IN it.
    #
    # A recut log has exactly one legal home for a digest: ``[i].files[j].from`` and
    # ``[i].files[j].to`` — the pair recording what one file moved between. Any other
    # position is refused BY PATH, which is a statement about structure rather than
    # about the value, so it cannot be aimed at a particular digest.
    misplaced = []
    for entry in historical:
        trail = [str(part) for part in entry.trail]
        cut = next((i for i, key in enumerate(trail)
                    if key.endswith("_recut_log")), None)
        if cut is None:          # a superseded/recert container, not this leg's business
            continue
        tail = trail[cut + 1:]
        # The writer's shape, verbatim: ['0', 'files', '2', 'to'] — a block index, the
        # files list, a row index, and one side of the from/to pair. List indices come
        # through the walker as bare decimal strings, not as '[i]'.
        legal = (len(tail) == 4 and tail[0].isdigit() and tail[1] == "files"
                 and tail[2].isdigit() and tail[3] in ("from", "to"))
        if not legal:
            misplaced.append(walk.dotted(entry.trail))
    assert not misplaced, (
        f"{len(misplaced)} digest(s) sit inside a recut log somewhere other than "
        f"files[].from / files[].to: {misplaced[:8]} — the historical exemption is "
        f"granted by PATH, so a digest parked anywhere else under that key would "
        f"leave the tree comparison without any leg noticing. Put the pin back where "
        f"it is checked, or the log back into the shape the writer produces")


def test_the_earning_legs_can_each_be_made_to_fire():
    """THE CONTROL THE PORT DROPPED. A leg nobody has watched fail proves nothing.

    The Triton original ends with planted-record controls for exactly this reason, and
    the first version of this port brought none of them across — four assertions, all
    green over the real record, none ever shown able to go red. That is the shape of
    a floor that has quietly stopped floor-ing, and this file arms its other new
    tests, so it has no excuse.

    Each leg is planted INTO A COPY of the real record, so the controls exercise the
    same code path the real assertions do, against the same walker.
    """
    import copy  # noqa: PLC0415

    record = _record()
    owner = "driver_dispatch"
    assert owner in record, "the entry these controls plant into has left the record"

    def legs_over(planted):
        """Re-run the four legs over a planted record; return which fired."""
        found = _census(planted)
        historical = [f for f in found if f.rule.startswith("historical_")]
        fired = []
        if not historical:
            fired.append("no-history")
        for name, entry in sorted(planted.items()):
            if not isinstance(entry, dict):
                continue
            for key, value in sorted(entry.items()):
                if not key.endswith("_recut_log") or not isinstance(value, list):
                    continue
                for block in value:
                    if not isinstance(block, dict):
                        fired.append("leg1"); continue
                    if not all(block.get(f) for f in ("why", "recut_utc", "cut_from")):
                        fired.append("leg1")
                    elif not block.get("files"):
                        fired.append("leg1")
        live = {f.trail[0] for f in _checked(found)}
        if {f.trail[0] for f in historical} - live:
            fired.append("leg2")
        if any(len(f.trail) < 2 or not isinstance(planted.get(f.trail[0]), dict)
               for f in historical):
            fired.append("leg3")
        for f in historical:
            trail = [str(part) for part in f.trail]
            cut = next((i for i, k in enumerate(trail)
                        if k.endswith("_recut_log")), None)
            if cut is None:
                continue
            tail = trail[cut + 1:]
            if not (len(tail) == 4 and tail[0].isdigit() and tail[1] == "files"
                    and tail[2].isdigit() and tail[3] in ("from", "to")):
                fired.append("leg4")
        return set(fired)

    assert legs_over(record) == set(), (
        "a leg fires on the UNPLANTED record, so these controls cannot show anything")

    # LEG 1 — a recut block that declares nothing.
    mute = copy.deepcopy(record)
    mute[owner]["_source_sha256_recut_log"][0]["why"] = ""
    assert "leg1" in legs_over(mute), "leg 1 does not fire on an undeclared recut block"

    # LEG 2 — an entry whose ONLY digests are historical.
    emptied = copy.deepcopy(record)
    for key in ("source_sha256", "code_sha256"):
        emptied[owner].pop(key, None)
    assert "leg2" in legs_over(emptied), (
        "leg 2 does not fire on an entry whose live pins have all been moved into its "
        "own past — the exact evasion it exists to refuse")

    # LEG 3 — history at the record root, owned by nothing.
    orphan = copy.deepcopy(record)
    orphan["_source_sha256_recut_log"] = copy.deepcopy(
        record[owner]["_source_sha256_recut_log"])
    assert "leg3" in legs_over(orphan), "leg 3 does not fire on rootless history"

    # LEG 4 — a live pin RELOCATED into the recut log, the laundering path.
    laundered = copy.deepcopy(record)
    moved = laundered[owner].pop("source_sha256")
    laundered[owner]["_source_sha256_recut_log"][0]["source_sha256"] = moved
    assert "leg4" in legs_over(laundered), (
        "leg 4 does not fire on a live digest map parked inside a recut log, which is "
        "the whole reason the exemption is granted by path and not by container")


def test_no_exclusion_here_rests_on_a_premise_or_a_citation():
    """AN EXCLUSION IS ONLY AS GOOD AS ITS PREMISE — asserted as a negative, ARMED.

    Two shapes of exclusion in the shared walker rest on something other than the
    record: a claim about this checkout (``staging_only_file``,
    ``out_of_tree_sibling_path``) and a claim about another test
    (``generated_kernel_source_slot``). Neither fires here, and that is worth
    pinning rather than leaving to the absence of a key — the rules live in the
    shared walker and fire on SHAPE, so an entry that grew one of those blocks
    would inherit the exemption without anyone arguing for it.

    THE NEGATIVE IS ARMED. A premise check with nothing to check is
    indistinguishable from one that never runs, so the same function is driven
    against a planted census that does violate the premise, and must say so.
    """
    found = _census()
    premised = [walk.dotted(f.trail) for f in found
                if f.rule in walk.PREMISE_ABSENT_FROM_TREE]
    assert not premised, (
        f"{len(premised)} digest(s) here are now excluded on a claim about this "
        f"checkout: {premised[:8]} — assert the premise the way "
        f"test_triton_weld_contract.py does, or check the digests")

    excluding = {f.rule for f in found if f.tier is None}
    deferring = sorted(excluding & set(walk.DELEGATED_COVERAGE))
    assert not deferring, (
        f"{deferring} excuse digests here by citing another test. A citation is "
        f"not a comparison — check them here, or assert the citation the way "
        f"test_metal_weld_contract.py::"
        f"test_the_generated_slot_delegation_is_real_and_admits_no_allowance does")

    planted = walk.census(
        {"g": {"staging": {"staging_only_file":
                           {"path": "meep_gpu/cuda_kernels/registry.py",
                            "sha256": "a" * 64}}}})
    armed = walk.premise_violations(planted, is_file=lambda name: (REPO / name).is_file(),
                                    inside_repo=lambda name: True)
    assert len(armed) == 1 and "IS in this checkout" in armed[0][2], armed


def test_every_citation_in_an_exclusion_names_a_test_that_exists():
    """A CITATION IS A CLAIM, and an unresolvable one is worse than none.

    ``_what_status_means`` is excused a pin because two tests here check the rule
    it states. That citation named no file until 2026-08-30 — "checked by
    test_the_status_is_the_one_the_tree_supports" — which is a sentence, not a
    reference: nothing failed if the function were renamed or deleted, and the
    exclusion would have gone on reading as covered. It now names file and
    function, and both must resolve.

    Parsed rather than imported, so this stays a laptop-cheap structural check.
    """
    cited = set()
    for reasons in (NOT_A_WELD_LEDGER_ENTRY.values(), walk.RULE_REASONS.values(),
                    walk.DELEGATED_COVERAGE.values()):
        for reason in reasons:
            cited |= set(re.findall(r"(test_[a-z0-9_]+\.py)::(test_[a-z0-9_]+)",
                                    reason))
    assert cited, "no exclusion cites a test any more — delete this, or fix the form"
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
        f"resting on a test nobody can find is an unchecked claim")

    # ARMED on the resolver: a name nobody wrote must NOT resolve.
    tree = ast.parse((PACKAGE / "test_cuda_weld_contract.py").read_text(
        encoding="utf-8"))
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}
    assert "test_a_function_nobody_wrote" not in names
    assert "test_the_status_is_the_one_the_tree_supports" in names


# ---------------------------------------------------------------------------
# THE BYTES
# ---------------------------------------------------------------------------


def test_the_cuda_welds_are_bound_to_the_live_sources():
    """The Metal rule, on the CUDA record, over every pin in it.

    RED AS LANDED, 40 of 79, which was the honest reading of a ledger nineteen of
    whose twenty entries already said ``DRIFTED`` about themselves. GREEN SINCE
    2026-08-30, 0 of 79, and the only way it was allowed to get there: every
    campaign re-run on a CUDA host into ``results/cuda_regate_2026-08-30/`` and
    rebound with ``parity/meep_gpu/rebind_cuda_welds.py --campaign``, against a
    tree ``record_cuda_regate.py`` had first made ``certification.json`` name.
    Nothing here was relaxed and no digest was typed; the denominator RAISED
    (``CODE_TIER_FLOOR`` 39 -> 79) rather than staying where a shrink could hide.

    If a specific entry ever needs an exception it is named HERE, as one entry,
    with the reason — never as a predicate that decides a class of edits harmless.
    """
    raw = _of_tier(_census(), walk.TIER_RAW)
    entries = _entries()
    assert len(raw) >= RAW_TIER_FLOOR, len(raw)
    drift = _drifted(raw)
    assert not drift, (
        f"{len(drift)} of {len(raw)} pinned path(s) across {len(entries)} "
        f"entries no longer match the bytes the campaign imported — RE-GATE on a "
        f"CUDA host and rebind with parity/meep_gpu/rebind_cuda_welds.py; do not "
        f"edit this record and do not relax this test: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def test_the_cuda_code_tier_recomputes_against_the_tree():
    """Every ``code_sha256`` pin — 39, now 79 — compared to the LIVE FILE'S CODE.

    THIS IS NOT THE SAME CHECK AS ``test_no_entry_backfills_a_code_digest_beside_a
    _stale_source_digest``, and the difference matters. That one enforces the
    ledger's SELF-DECLARATION — the rule the record writes against itself, that a
    code digest may not sit beside a source digest that has moved. This one
    enforces the tree: a recorded code digest must equal what ``code_identity.
    code_digest`` computes from the file today.

    THE RULE, and it is a rule rather than a skip: the comparison is against the
    live file's CODE, never against its source digest. The two tiers MAY
    legitimately disagree, and exactly one thing makes that so — a comment or
    docstring edit since the gate ran, which is the entire clause code_identity
    licenses. Every other disagreement is a defect: a code digest that no longer
    recomputes describes neither the bytes that ran nor the bytes on disk, which
    is precisely how a stale record keeps looking current.

    Measured 2026-08-30, before the re-gate: 39 of 39 recompute, on a ledger 40 of
    whose 79 raw pins had drifted. That combination is exactly what the code tier
    is FOR — the drifted files moved by comment and docstring — and it is also why
    leaving the tier unread was dangerous: it is the field a reader consults to
    decide the raw red is benign. After the re-gate: 79 of 79, because the tier is
    no longer being withheld on 40 files whose source digests had moved.
    """
    code = _of_tier(_census(), walk.TIER_CODE)
    assert len(code) >= CODE_TIER_FLOOR, (
        f"only {len(code)} code_sha256 pins found, was {CODE_TIER_FLOOR} — the "
        f"tier left the comparison")
    drift = _drifted(code)
    assert not drift, (
        f"{len(drift)} of {len(code)} code digest(s) no longer recompute against "
        f"the tree — RE-GATE and rebind with "
        f"parity/meep_gpu/rebind_cuda_welds.py; do not hand-edit the digest: "
        + "; ".join(f"{where} -> {name} ({why})" for where, name, why in drift[:12]))


def _plant(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def test_the_drift_check_refuses_a_planted_digest(tmp_path):
    """THE ARMED CONTROL. Withhold nothing and plant one wrong byte-digest.

    Without this, the test above failing would be consistent with ``_drifted``
    reporting unconditionally, and — once the re-gate turns it green — its passing
    would be consistent with ``_drifted`` returning an empty list unconditionally.
    Both directions are pinned, and a comment-only edit is pinned as raw drift so
    the difference from ``weld_survives_edit``'s rule is EXECUTED rather than
    asserted in prose.

    Driven through the WALK rather than a hand-built pin list, so the classifier
    is exercised too: a rule that stopped claiming these shapes would take the
    planted digest out of the checked set and this test would go red.
    """
    victim = PACKAGE / "cuda_kernels" / "ade_kernels.py"
    source = victim.read_text(encoding="utf-8")
    true_digest = hashlib.sha256(victim.read_bytes()).hexdigest()
    rel = victim.relative_to(REPO).as_posix()
    wrong = "f" * 64
    planted = {"a_synthetic_weld": {"source_sha256": {rel: wrong}}}
    honest = {"a_synthetic_weld": {"source_sha256": {rel: true_digest}}}
    assert _drifted(_census(planted)) == [
        (f"a_synthetic_weld.source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {true_digest[:12]}")]
    assert _drifted(_census(honest)) == []

    commented = _plant(tmp_path / "commented", rel, source + "\n# appended\n")
    assert _drifted(_census(honest), root=commented) == [
        (f"a_synthetic_weld.source_sha256.{rel}", rel,
         f"recorded {true_digest[:12]} live "
         f"{hashlib.sha256((commented / rel).read_bytes()).hexdigest()[:12]}")]

    absent = {"a_synthetic_weld": {"source_sha256": {"meep_gpu/nope.py": true_digest}}}
    assert _drifted(_census(absent), root=tmp_path / "nowhere") == [
        ("a_synthetic_weld.source_sha256.meep_gpu/nope.py", "meep_gpu/nope.py",
         "MISSING FROM THE TREE")]

    # A DIGEST INSIDE A LIST. This ledger holds none today, which is exactly why
    # the control is here: the shape that hid 22 Triton pins must fail the moment
    # it appears, not the day somebody notices it appeared.
    in_list = {"g": {"runs": [{"source_sha256": {rel: wrong}}]}}
    assert _drifted(_census(in_list)) == [
        (f"g.runs[0].source_sha256.{rel}", rel,
         f"recorded {wrong[:12]} live {true_digest[:12]}")]


def test_the_code_tier_is_the_clause_code_identity_licenses(tmp_path):
    """THE ARMED CONTROL FOR THE CODE TIER, in both directions, on a real file.

      1. A COMMENT-ONLY EDIT: the raw digest MOVES and the code digest HOLDS.
         This is the whole clause code_identity licenses and the only reason the
         two tiers are allowed to disagree — and on this ledger it was not
         hypothetical: until the 2026-08-30 re-gate it was what 40 drifted raw
         pins beside 39 clean code pins meant. The re-gate removed the
         disagreement rather than the rule, so this control is now the ONLY place
         the clause is exercised, which makes it load-bearing rather than
         illustrative.
      2. A REAL CODE EDIT: both move. The licence does not stretch to it.
      3. A PLANTED CODE DIGEST on an unedited file: reported.
      4. A CODE DIGEST ON A FILE WITH NO CODE IDENTITY: reported, never skipped.
         Silently passing an incomputable comparison is the defect class this
         round exists to remove.
    """
    victim = PACKAGE / "cuda_kernels" / "ade_kernels.py"
    rel = victim.relative_to(REPO).as_posix()
    source = victim.read_text(encoding="utf-8")
    raw_now = hashlib.sha256(victim.read_bytes()).hexdigest()
    code_now = code_digest(source)
    record = {"w": {"source_sha256": {rel: raw_now}, "code_sha256": {rel: code_now}}}

    commented = _plant(tmp_path / "commented", rel,
                       source + "\n# a comment appended, changing no execution\n")
    assert len(_drifted(_of_tier(_census(record), walk.TIER_RAW), root=commented)) == 1
    assert _drifted(_of_tier(_census(record), walk.TIER_CODE), root=commented) == [], (
        "a comment-only edit was reported as code drift — the tier is stricter "
        "than code_identity licenses and will make doc fixes expensive")

    edited = _plant(tmp_path / "edited", rel, source + "\nA_NEW_CONSTANT = 1\n")
    assert len(_drifted(_of_tier(_census(record), walk.TIER_RAW), root=edited)) == 1
    assert len(_drifted(_of_tier(_census(record), walk.TIER_CODE), root=edited)) == 1, (
        "a new module-level constant did not move the code digest — the tier "
        "would license a real code change as a comment")

    reported = _drifted(_of_tier(_census({"w": {"code_sha256": {rel: "f" * 64}}}),
                                 walk.TIER_CODE))
    assert len(reported) == 1 and reported[0][1] == rel, reported

    non_python = {"w": {"code_sha256": {"meep_gpu/cuda_kernels/certification.json":
                                        "f" * 64}}}
    reported = _drifted(_of_tier(_census(non_python), walk.TIER_CODE))
    assert len(reported) == 1 and "no code identity" in reported[0][2], reported


def test_the_walker_finds_a_digest_planted_at_a_new_nesting_depth():
    """THE ARMED CONTROL FOR THE WALK ITSELF. Plant one deeper than anything real.

    Without this, a walker that stopped at depth 3 would report zero unclassified
    digests and a perfectly stable discovered count, because this ledger's deepest
    real digest sits at depth 3. Both halves are required: the walk must SURFACE
    the planted digest, and it must classify it ``UNCLASSIFIED``. The second is
    the load-bearing one — a walker that found it and quietly excluded it would be
    the original defect wearing this test as cover.
    """
    digest = "a" * 64
    deep = {"an_entry": {"a": [{"b": {"c": [{"d": {"a_key_nobody_wrote": digest}}]}}]}}
    found = walk.census(deep)
    assert [f.value for f in found] == [digest], (
        "the walk did not surface a digest nested eight levels down through two "
        "lists — it is not recursing")
    planted = found[0]
    assert walk.dotted(planted.trail) == "an_entry.a[0].b.c[0].d.a_key_nobody_wrote"
    assert len(planted.trail) > max(len(f.trail) for f in _census()), (
        "the planted digest is no deeper than the ledger's own — the control has "
        "stopped testing anything the real walk does not already reach")
    assert planted.rule == walk.UNCLASSIFIED and planted.tier is None, (
        f"a digest under an unanticipated key was classified {planted.rule!r} "
        f"instead of failing — an unknown shape must be a red somebody writes a "
        f"rule for, never a silent exclusion")


# ---------------------------------------------------------------------------
# THE SELF-DECLARATION — checkable without a device
# ---------------------------------------------------------------------------


def test_the_declared_source_drift_is_the_measured_source_drift():
    """``source_drift`` must be the whole truth, not a snapshot of it.

    THE DEFECT THIS CATCHES, measured 2026-08-30 on
    ``cuda_fused_magnetic_pair_2026-08-27``: it declared two moved files and
    three had moved — ``parity/meep_gpu/gate_cuda_fused_magnetic_pair.py``
    drifted after the declaration was written and nothing noticed, because
    nothing read the declaration. An under-declared drift list is worse than none:
    it invites a reader to treat the two named files as the whole exposure and to
    trust every pin the list does not mention.

    IT WAS CLEARED BY RE-RUNNING THE WRITER, WHICH IS THE ONLY WAY. The entry now
    declares ``source_drift: []`` and all four of its pins are bound to the
    2026-08-30 re-gate, so the list is the whole truth by being empty. That is a
    stronger outcome than a corrected list of three, and it is worth naming which
    one happened: the declaration was never hand-edited to match.

    BOTH DIRECTIONS. A file named in ``source_drift`` that matches the tree again
    is also reported — a stale entry there means the next real one can hide
    behind it, and the remedy is the same rebind either way.
    """
    wrong = {}
    for name, entry in sorted(_entries().items()):
        declared = set(entry.get("source_drift") or [])
        measured = _measured_drift(name, entry)
        if declared != measured:
            wrong[name] = {"declared not measured": sorted(declared - measured),
                           "measured not declared": sorted(measured - declared)}
    assert not wrong, (
        f"{len(wrong)} entries' source_drift does not describe the tree: {wrong} "
        f"— rebind with parity/meep_gpu/rebind_cuda_welds.py; do not hand-edit "
        f"the list to match")


def test_the_status_is_the_one_the_tree_supports():
    """PASS and DRIFTED mean what ``_what_status_means`` says they mean.

    The record defines them itself: PASS is "every canonical policy leg released,
    AND every file this entry pins still matches"; DRIFTED is "every leg still
    released, but at least one pinned file has moved". The second half of each is
    a fact about the tree, so it is checked here. The consequence that matters is
    the one direction this makes impossible: an entry cannot be stamped PASS
    while a pin it carries has moved.
    """
    record = _record()
    assert "PASS:" in record["_what_status_means"], (
        "the record no longer defines its own status vocabulary — this test is "
        "checking a rule that is no longer written down")
    wrong = {}
    for name, entry in sorted(_entries(record).items()):
        status, moved = entry.get("status"), _measured_drift(name, entry)
        if status not in ("PASS", "DRIFTED"):
            wrong[name] = f"status {status!r} is neither PASS nor DRIFTED"
        elif status == "PASS" and moved:
            wrong[name] = f"PASS, but {len(moved)} pinned file(s) moved: {sorted(moved)[:3]}"
        elif status == "DRIFTED" and not moved:
            wrong[name] = "DRIFTED, but every pinned file matches — rebind it"
    assert not wrong, wrong


def test_no_entry_backfills_a_code_digest_beside_a_stale_source_digest():
    """The licence ``code_identity`` withholds must not be manufactured here.

    The record states the rule against itself: "code_sha256 deliberately omits
    the moved files: code_identity.py licenses a weld to stand on an unchanged
    code digest only where source_sha256 still matches, 'because only then is the
    file on disk the bytes the gate actually ran', so backfilling a live code
    digest beside a stale source digest would manufacture the licence that clause
    withholds."

    MEASURED 2026-08-30: one entry violated it —
    ``cuda_fused_magnetic_pair_2026-08-27`` carried a ``code_sha256`` for
    ``parity/meep_gpu/gate_cuda_fused_magnetic_pair.py`` whose source digest was
    stale. It was the SAME file the entry failed to declare in ``source_drift``,
    found by an independent route, which is what made it a defect rather than a
    bookkeeping preference. Both are cleared by the same re-gate: that file is now
    pinned to the bytes the run imported, so the code digest beside it is licensed
    rather than manufactured.

    A ``code_sha256`` that no longer recomputes is reported too. A digest that is
    neither current nor accompanied by a matching source digest describes nothing.
    """
    offenders = {}
    for name, entry in sorted(_entries().items()):
        moved = _measured_drift(name, entry)
        for path, digest in sorted((entry.get("code_sha256") or {}).items()):
            if path in moved:
                offenders[f"{name}:{path}"] = (
                    "carries a code digest beside a STALE source digest")
                continue
            live = REPO / path
            if not live.is_file():
                offenders[f"{name}:{path}"] = "code_sha256 names a path not in the tree"
            elif live.suffix == ".py" and code_digest(
                    live.read_text(encoding="utf-8")) != digest:
                offenders[f"{name}:{path}"] = "code_sha256 no longer recomputes"
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# WHAT AN ENTRY MUST CARRY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_METADATA)
def test_every_cuda_entry_carries_its_metadata(field):
    """The Metal requirement, widened by this ledger's own shape.

    The gap it forecloses is the one the CUDA comparison found in
    ``certification.json`` — 0 of 4 blocks carried ``recorded_utc`` — returning
    through the ledger that was written to close it. Flat, with no budget: all
    twenty carry all nine today, so this is a floor a future entry must clear
    rather than a backlog to be worked down.
    """
    missing = [name for name, entry in _weld_entries().items()
               if not entry.get(field)]
    assert not missing, f"{len(missing)} CUDA entries lack {field}: {sorted(missing)}"


def test_every_cuda_entry_records_a_timestamp_that_parses():
    """ISO-8601 UTC, not merely a non-empty string.

    An entry's date is what orders it against the source drift around it, and
    every one of these entries is currently claiming a run that predates the tree.
    A spelling no one can parse cannot be compared. ``gate_started_utc`` is held
    to the same rule: it is the stamp the artifact carries, so a mismatch of
    spelling between the two is a mismatch of provenance.
    """
    bad = {}
    for name, entry in _weld_entries().items():
        for field in ("recorded_utc", "gate_started_utc"):
            stamp = entry.get(field)
            try:
                datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, ValueError):
                bad[f"{name}.{field}"] = stamp
    assert not bad, f"stamps that are not ISO-8601 UTC: {bad}"


def test_every_cuda_entry_artifact_digest_is_a_sha256():
    """``artifact_sha256`` must be a digest, not a note about one.

    The artifacts live under the gitignored results tree, so nothing here can
    recompute them on a fresh clone — which is precisely why the recorded value
    has to be well formed: it is the only handle a later reader gets.
    """
    bad = {name: entry["artifact_sha256"] for name, entry in _weld_entries().items()
           if not _SHA256.match(str(entry["artifact_sha256"]))}
    assert not bad, f"entries whose artifact_sha256 is not a sha256: {bad}"


def test_every_cuda_entry_names_both_canonical_policies():
    """A CUDA weld is a claim about two legs, so both must be named.

    This track's whole comparison is stock MEEP's x86 flush behaviour against an
    FTZ-stripped IEEE keep run; an entry that names one policy, or names none,
    does not identify its own result. Present is not enough — the field must say
    which two, and the ``legs`` list must have one entry per leg it names.
    """
    bad = {}
    for name, entry in sorted(_entries().items()):
        policy = str(entry.get("subnormal_policy", ""))
        if "flush" not in policy or "keep" not in policy:
            bad[name] = f"subnormal_policy names fewer than both legs: {policy!r}"
        elif not isinstance(entry.get("legs"), list) or len(entry["legs"]) < 2:
            bad[name] = f"legs is not a list of at least two: {entry.get('legs')!r}"
    assert not bad, bad


def test_every_cuda_entry_pins_both_the_kernels_and_the_script_that_gated_them():
    """A weld over harness scripts alone certifies nothing a user runs — and a
    weld over package bytes alone cannot say what was asked of them.

    MEASURED 2026-08-30: all twenty pin at least one ``meep_gpu/cuda_kernels/``
    module and at least one ``parity/meep_gpu/`` script, which is the same
    coverage the Metal record has at 44 of 44 and the Triton record reached at
    31 of 31. The half that is easy to lose is the second one: the sources say
    which implementation ran, the gate script says what it was asked to prove, and
    a gate that changes underneath an unpinned weld rewrites the result while
    every source digest still verifies.

    The module the entry NAMES in ``kernel_module`` must also be among the paths
    it pins — a record that names one module and welds another is unfollowable.
    """
    bad = {}
    for name, entry in sorted(_entries().items()):
        pinned = set(entry["source_sha256"])
        if not any(p.startswith("meep_gpu/cuda_kernels/") for p in pinned):
            bad[name] = "pins no path under meep_gpu/cuda_kernels/"
            continue
        if not any(p.startswith("parity/") for p in pinned):
            bad[name] = "pins no gate or probe script under parity/meep_gpu/"
            continue
        named = entry.get("kernel_module")
        named = [named] if isinstance(named, str) else list(named or [])
        unpinned = sorted(set(named) - pinned)
        if unpinned:
            bad[name] = f"names kernel_module {unpinned} but does not pin it"
    assert not bad, bad


def test_no_cuda_entry_claims_a_campaign_that_failed_on_device():
    """An entry is a record of a RELEASED campaign. Welding a refusal makes it lie.

    Found MECHANICALLY rather than by a name list, so the pin maintains itself:
    any ``rc`` that is a non-zero integer, or a ``failure_measured_on_device``
    flag, at any depth. The Triton counterpart has two such records today and
    this one has none, so the control below is what keeps that zero honest — it
    plants a failure in a copy of a real entry and requires it to be found.
    """
    def failures(entry, trail=""):
        found = []
        rc = entry.get("rc")
        if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
            found.append((trail + "rc", rc))
        if entry.get("failure_measured_on_device"):
            found.append((trail + "failure_measured_on_device", True))
        for key, value in entry.items():
            if isinstance(value, dict):
                found.extend(failures(value, f"{trail}{key}."))
        return found

    entries = _entries()
    welded_refusals = {name: failures(entry) for name, entry in entries.items()
                       if failures(entry)}
    assert not welded_refusals, (
        f"entries welded despite a recorded device failure: {welded_refusals}")

    # THE ARMED CONTROL, because the assertion above is currently vacuous over
    # the real record and an empty result must be a fact about the ledger rather
    # than about the finder.
    victim = json.loads(json.dumps(entries["ade_2026-08-19"]))
    assert failures(victim) == []
    victim["_notes"]["rc"] = 1
    assert failures(victim) == [("_notes.rc", 1)]
    victim["_notes"].pop("rc")
    victim["failure_measured_on_device"] = True
    assert failures(victim) == [("failure_measured_on_device", True)]


def test_the_records_line_names_a_campaign_directory():
    """A weld a reader cannot follow to its run is a weld they must take on trust.

    Only the SHAPE is checked, not the directory's presence: ``results/`` is
    gitignored, so requiring it to exist would make this laptop test depend on
    which artifacts happen to be on the machine. What must hold is that the line
    names an ``parity/meep_gpu/results/...`` path — the spelling that
    resolves from a checkout — rather than a bare directory name or a note.
    """
    bad = {name: entry["records"] for name, entry in _entries().items()
           if not str(entry["records"]).startswith(
               "apps/api/parity/meep_gpu/results/")}
    assert not bad, f"entries whose records line resolves nowhere: {bad}"
