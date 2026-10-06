"""Find EVERY digest in a weld record, wherever it lives, and say what decides its fate.

WHY THIS EXISTS. FIVE times now a check in this tree has covered less than it appeared
to, and every one of them was the same mistake wearing a different hat:

    a status filter hid 88 pinned pairs (31 of them live drifts) for eleven days;
    a shape filter hid 22 more inside ``superseded_runs[0..1]``;
    442 ``code_sha256`` pins were compared by NOTHING at the merge bar;
    108 ``device_sha256`` digests were likewise read by no laptop test;
    66 ``host_sha256`` digests were EXCLUDED as "checked elsewhere", and on the
        Triton side "elsewhere" made a weaker claim than the exclusion did.

THE FIFTH SUB-FORM, and it is the one this round removes: AN EXCLUSION IS ONLY AS
GOOD AS ITS PREMISE, and a premise nobody checks is the same defect as a filter
nobody counted. Every exclusion below rests on a claim about the world —
"this file is not in this checkout", "these digests are recomputed by that test
over there" — and until 2026-08-30 not one of those claims was asserted. Two were
already false or weaker than written:

  * ``host_side_digest_map`` deferred 10 Triton digests to
    ``test_triton_kernels.py::test_the_fingerprint_record_matches_the_shipped_host_side``,
    which admits any edit ``device_identity.weld_survives_edit`` calls equivalent
    — the exact allowance ``test_triton_weld_contract`` refuses by name, and the
    one that let two comment-block edits to ``launch.py`` leave twenty-one welds
    describing bytes that no longer shipped. A delegation that arrives at a weaker
    claim than the exclusion states is an unchecked digest wearing a citation. All
    66 are now CHECKED HERE, at the raw tier, and the delegation is gone rather
    than verified — a reference that cannot go stale is better than one that is
    watched.
  * the remaining "not in this checkout" exclusions now carry a PREMISE that
    :func:`premise_violations` decides mechanically, so a staged file that gets
    committed stops being excludable instead of staying quiet.

The 2026-08-30 round removed the *status* filter and asserted the denominator, which was
right and was not enough: the replacement still walked a FIXED SHAPE — top-level entry,
then ``source_sha256``, then a sibling ``<name>_sha256`` beside a ``<name>`` — so a digest
one level deeper, or inside a list, or under a key spelling nobody anticipated, was still
invisible. Naming the next hiding place and patching it is how a defect CLASS survives
being fixed four times.

THE RULE THAT REPLACES THE SHAPE. Recurse the whole record — dicts and lists, at any
depth — and collect every 64-hex string with the path it was found at. Then EVERY digest
is either

    CHECKED   against this tree, in one of three tiers (raw bytes, code identity,
              device-side kernel identity), or
    EXCLUDED  by a NAMED rule that carries a written reason,

and a digest matching neither is :data:`UNCLASSIFIED`, which is a test failure. There is
no third outcome and no predicate that quietly drops a digest. A new key spelling does not
leave coverage — it arrives as an unclassified digest that somebody has to write a rule
for, in the same change, where a reviewer sees it.

WHAT THIS MODULE DOES NOT DO. It does not read the tree and it does not decide whether a
record is honest. :func:`census` classifies; the contract tests compare. That split is
deliberate: the classifier consults no digest and no file, so it cannot be steered toward
a shape that happens to pass.

ONE RECORD IN EACH LEDGER IS NOT A WELD: a table's ``driver_dispatch`` record, whose
verdict belongs to the route runs filed under ``runs[<key>]``. :func:`route_record_problems`
states the one rule all three weld contracts hold it to. Like :func:`compare`, it decides
only from what the caller passes in: liveness, drift and shape are measured by the
contract and handed over, so this module still reads no file.

Stdlib plus two in-package identity helpers. No device, no third-party imports: this runs
at the laptop merge bar.
"""

from __future__ import annotations

import re
from typing import (Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

#: The per-capability run container, spelled here rather than imported: this module is
#: a pure classifier over a loaded record and imports nothing from the package, so a
#: ledger can be walked in a checkout where the engine will not import.
#: ``test_weld_record_walk.py`` asserts this equals ``fastpath.RUNS``, so the two
#: cannot drift apart.
RUNS = "runs"

__all__ = [
    "Found", "discover", "census", "dotted", "rules_that_fired",
    "TIER_RAW", "TIER_CODE", "TIER_DEVICE", "UNCLASSIFIED", "RULE_REASONS",
    "PREMISE_ABSENT_FROM_TREE", "GITIGNORED_PREFIX", "premise_violations",
    "DELEGATED_COVERAGE", "ROUTE_RUN_PASS", "route_run_problems", "route_record_problems",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: sha256 of the file's BYTES. The strongest tier and the default: it is what a weld
#: means by "the gate executed these bytes".
TIER_RAW = "raw"

#: ``code_identity.code_digest`` of the file — the parsed code with docstrings, comments
#: and layout removed. See :data:`RULE_REASONS` for the rule this tier carries.
TIER_CODE = "code"

#: ``device_identity.device_digests`` of the file, per kernel name — the identity of the
#: device-side source string the module generates, not of the file that generates it.
TIER_DEVICE = "device"

#: The fate of a digest no rule claimed. Not a tier and not an exclusion: a failure.
UNCLASSIFIED = "unclassified"


class Found(NamedTuple):
    """One 64-hex digest, where it was found, and what decides its fate.

    ``tier`` is one of the three ``TIER_*`` constants when the digest is CHECKED, and
    ``None`` when it is excluded. ``rule`` always names the rule that decided, so a
    census can be reported as "discovered / checked / excluded" with every exclusion
    attributable to a written reason.
    """

    trail: Tuple[Any, ...]
    value: str
    rule: str
    tier: Optional[str]
    #: The path this digest claims, when its shape names one. ``None`` when the record
    #: pins a digest without saying which file it is about — which is itself a finding.
    subject: Optional[str]
    #: For :data:`TIER_DEVICE` only: which kernel inside ``subject`` the digest names.
    kernel: Optional[str] = None


def dotted(trail: Tuple[Any, ...]) -> str:
    """``a.b[0].c`` — a trail rendered so a failure message can be pasted into a search."""
    out = ""
    for part in trail:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += (f".{part}" if out else str(part))
    return out


def discover(node: Any, trail: Tuple[Any, ...] = ()) -> Iterator[Tuple[Tuple[Any, ...], str]]:
    """Every 64-hex string in ``node``, at any depth, through dicts AND lists.

    THE WHOLE POINT IS THAT THIS KNOWS NOTHING about weld records. It does not look for
    ``source_sha256``, it does not stop at a level, and it does not skip a container it
    does not recognise. A digest cannot escape it by being somewhere new; it can only
    escape by not being a 64-hex string, and a record that stops spelling digests in hex
    fails the discovered-count floor rather than going quiet.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from discover(value, trail + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from discover(value, trail + (index,))
    elif isinstance(node, str) and _SHA256.match(node):
        yield trail, node


# ---------------------------------------------------------------------------
# THE RULES — every one of them named, and every name carrying its reason
# ---------------------------------------------------------------------------
#
# ORDER IS THE RULE, not an implementation detail, and it runs from the most specific
# claim to the least. The historical containers come FIRST because they hold digests
# whose inner shape is identical to a live one — ``superseded_runs[0].source_sha256`` is
# a ``source_sha256`` map in every respect except that the record says out loud it is not
# about this tree. If ``source_digest_map`` matched first, those 22 pins would be checked
# as if they were current, and the record's own ``why_superseded`` would be overruled by
# the order two rules happen to sit in.

#: Why each rule excludes what it excludes, and — for the tiers — what it compares.
#: The contract tests assert that every rule here FIRES at least once on the record it
#: guards, so a rule cannot outlive the shape it was written for and sit around as a
#: place for the next real omission to hide.
RULE_REASONS: Dict[str, str] = {
    # ---- CHECKED -----------------------------------------------------------
    TIER_RAW + ":source_digest_map":
        "a source_sha256 map: path -> sha256 of the file's bytes. THE tier. Compared "
        "raw against the tree with no allowance, because a weld says a device executed "
        "these bytes and nothing on the host can settle that claim by judging an edit "
        "harmless.",
    TIER_RAW + ":sibling_script_pin":
        "a <name>/<name>_sha256 pair naming an in-tree script — the probe, launcher or "
        "adapter the gate ran. Compared raw, same rule: a gate script that changes "
        "underneath a weld rewrites the result while every source digest still verifies.",
    TIER_RAW + ":parent_key_named_pin":
        "a block keyed BY THE FILE, carrying a bare 'sha256' inside it — the shape "
        "specialized_kernel_sources uses. The subject is the parent key, so nothing "
        "names a path and the old sibling rule could not see these at all. Compared raw.",
    TIER_RAW + ":host_side_digest_map":
        "a host_sha256 map: filename -> sha256 of the host module the gate shipped. "
        "CHECKED HERE SINCE 2026-08-30, and it was excluded before that on the "
        "strength of a citation. The old reason said these were 'recomputed against "
        "the tree already' by the track's kernels test, and on the Triton side that "
        "test recomputes them THROUGH device_identity.weld_survives_edit — the "
        "allowance this contract refuses by name, and the one that let two "
        "comment-block edits to launch.py leave twenty-one welds describing bytes "
        "that no longer shipped. A delegation that lands on a weaker claim than the "
        "exclusion states is an unchecked digest wearing a citation, so the "
        "delegation is deleted rather than watched: raw bytes, no allowance, in the "
        "record that makes the claim. The track's own kernels test keeps its own, "
        "different, allowance-bearing claim for its own purpose — two claims, not "
        "two spellings of one.",
    TIER_CODE + ":code_digest_map":
        "a code_sha256 map: path -> code_identity.code_digest of the file. THE RULE IS "
        "THAT IT RECOMPUTES AGAINST THE TREE, and it is compared to the live file's "
        "CODE, never to the file's source digest. Those two tiers MAY differ, and "
        "exactly one thing makes that legitimate: a comment or docstring edit since the "
        "gate ran, which is the whole clause code_identity licenses. Every other "
        "disagreement is a defect — a code digest that no longer recomputes describes "
        "neither the bytes that ran nor the bytes on disk, and is precisely how a stale "
        "record keeps looking current.",
    TIER_DEVICE + ":device_kernel_digest":
        "a device_sha256 block: path -> {digests: {kernel: sha256}}. Recomputed with "
        "device_identity.device_digests, which parses the module and hashes the device "
        "source string each kernel generates. This is the identity that actually "
        "reached the compiler, so it is checked here rather than trusted.",

    # ---- EXCLUDED, each by name, each with what clears it -------------------
    "historical_superseded_run":
        "inside a superseded_runs list. The record says of these bytes, in its own "
        "why_superseded, that the run 'certified the previous bytes and is kept "
        "verbatim; it is not the record of the shipped host side any more'. Comparing "
        "them to the tree would assert a claim the record explicitly withdraws, and the "
        "only way to clear the resulting red would be to delete the history. THE "
        "EXCLUSION IS EARNED, NOT FREE: see test_triton_weld_contract.py::"
        "test_a_historical_exclusion_is_earned_by_a_live_checked_claim.",
    "historical_recert_snapshot":
        "under a *_at_recert key: the digests as they were when an earlier run was "
        "re-certified. The record says the surrounding fields 'are the ORIGINAL job and "
        "are deliberately untouched: they are true about that job and its bytes'. Same "
        "earning requirement as a superseded run.",
    "historical_recut_log":
        "inside a *_recut_log or source_drift_since_recert block: a LOG of what a digest "
        "moved from and to, so half of every pair is stale by construction. A log that "
        "had to match the tree could not record a change. Same earning requirement.",
    "staging_only_file":
        "a staging_only_file block. The record declares this file untracked — staged on "
        "the device host and never committed — so there is nothing in this checkout to "
        "compare it to. Clearing it means tracking the file, not editing the digest. "
        "THE PREMISE IS ASSERTED: the block names its path, and premise_violations "
        "requires that path to be absent from this checkout, so the day the file is "
        "committed the exclusion expires loudly instead of covering a live digest.",
    "run_bound_digest":
        "a bound_sha256 inside runs[<capability>]: the digest of the digests that run "
        "certified, which is what decides whether that capability's evidence still "
        "describes the shipped bytes. It names no file, so there is nothing to "
        "recompute from a checkout; it is compared against the ENTRY by "
        "fastpath.live_capabilities, and the weld contracts assert that every cited "
        "entry has at least one live capability. A run whose bound no longer matches is "
        "history by design — the capability stops being admitted — so this is not "
        "delegated coverage waiting on a re-cut.",
    "artifact_or_log_digest":
        "an artifact_sha256 or log_sha256: the digest of a run artifact or log under the "
        "gitignored parity/meep_gpu/results/ tree. It cannot be recomputed from a clean "
        "checkout, so requiring it would make this laptop test depend on which results "
        "tree happens to be on the machine. WHAT IS ACTUALLY CHECKED, stated narrowly "
        "because the previous wording ('that artifacts exist and are named is checked') "
        "claimed more than any test made good on: each track asserts that the field is "
        "PRESENT and sha256-SHAPED on the entries its metadata tier covers — the PASS "
        "subset on Triton and Metal, every entry on CUDA. Nothing asserts the artifact "
        "is on this machine, and nothing asserts a shape-valid digest names a real file. "
        "Clearing that means checking against a results tree, which is a device-host "
        "job, not a merge-bar one.",
    "generated_kernel_source_slot":
        "a kernel_source_sha256 slot: the digest of a GENERATED shader source string "
        "keyed by SLOT NAME (pml_curl_step/step_B/011/contract-off.sha256), not of any "
        "file. Nothing in this checkout is the subject, so there is no raw comparison "
        "to make and promoting it the way host_sha256 was promoted would be inventing "
        "one. It is recomputed — by generating the shader source again — in "
        "metal_kernels/launch.py's compute_fingerprints, read by "
        "test_metal_kernels.py::test_the_checked_in_fingerprints_match_the_tree, which "
        "compares the WHOLE dict raw and admits no allowance. That delegation is "
        "asserted rather than trusted: see test_metal_weld_contract.py::"
        "test_the_generated_slot_delegation_is_real_and_admits_no_allowance.",
    "out_of_tree_sibling_path":
        "a <name>/<name>_sha256 pair whose path is absolute, under ~, or under the "
        "gitignored parity/meep_gpu/results/ tree. Not in this checkout by construction "
        "— and THE PREMISE IS ASSERTED rather than assumed: premise_violations requires "
        "each such path to be either outside this repo entirely or under the gitignored "
        "results tree, so a repo-relative path that comes back into the checkout stops "
        "being excludable instead of escaping on a prefix.",
    "serial_driver_script":
        "inside a serial_driver block: the two shell drivers (chain.sh, run_gate.sh) "
        "that serialised a recert sweep and enforced empty-device placement. They exist "
        "only under the gitignored parity/meep_gpu/results/ tree — they were never "
        "committed — so a clean checkout has nothing to compare. The digests are kept "
        "because they name the bytes that placed the gates; cleared by committing the "
        "drivers, never by deleting the digests. NO PREMISE IS ASSERTED FOR THIS RULE "
        "AND THAT IS STATED RATHER THAN GLOSSED: the block keys a BARE BASENAME "
        "('chain.sh') and names no directory, so 'this path is absent from the "
        "checkout' is not a question this record can be asked. Widening it to 'no file "
        "of that name exists anywhere' would be a predicate past what is built. The "
        "remedy is to make the block name its path, after which it becomes an "
        "out_of_tree_sibling_path with a premise like every other.",
    "pin_names_no_path":
        "a *_sha256 leaf with NO sibling naming what it is the digest of. DECLARED DEBT, "
        "not a clean exclusion: the record carries a digest and cannot say which file it "
        "is about, so nothing can ever verify it. Cleared by adding the path key beside "
        "the digest, after which it becomes a checked sibling pin — never by deleting "
        "the digest.",
    UNCLASSIFIED:
        "no rule claimed this digest. NOT an exclusion: a failure. Write a rule for it — "
        "check it, or exclude it by name with the reason — in the change that introduces "
        "the shape.",
}


#: The one directory in this checkout whose contents are gitignored, so a
#: repo-relative path under it is legitimately absent from a clean tree while
#: being present on whichever machine ran the gate.
GITIGNORED_PREFIX = "parity/meep_gpu/results/"

#: EXCLUSIONS WHOSE PREMISE IS A CLAIM ABOUT THIS CHECKOUT, and which
#: :func:`premise_violations` therefore decides mechanically instead of taking on
#: trust. Both of these say, in different words, "there is nothing here to
#: compare to". That is a fact about the tree, it can stop being true — a staged
#: file gets committed, a results path is promoted into the package — and when it
#: does, the exclusion must EXPIRE rather than go on covering a live digest.
#:
#: ``serial_driver_script`` is deliberately NOT in this set: its block keys a bare
#: basename and names no directory, so the question cannot be put to the record.
#: That limit is written into its reason rather than papered over with a looser
#: predicate. ``artifact_or_log_digest`` is likewise absent — most of those pins
#: are bare leaves that name no path at all, and the ones that do are already
#: covered here through the sibling shape.
PREMISE_ABSENT_FROM_TREE = ("staging_only_file", "out_of_tree_sibling_path")

#: EXCLUSIONS WHOSE DIGESTS ARE CLAIMED TO BE COVERED SOMEWHERE ELSE, mapped to
#: the test that is supposed to cover them. This is the CITATION register, and it
#: is a register rather than a phrase in a docstring because the distinction it
#: draws is one string-sniffing cannot: several reasons above cross-reference a
#: test (``historical_superseded_run`` points at the legs that EARN it), and only
#: the entries here say "these digests are compared over there instead of here".
#:
#: THE RULE THAT GOES WITH IT. A contract may leave a rule's digests unchecked
#: only if the rule is in this map AND that contract asserts the citation — the
#: delegate exists, still makes the comparison, and admits no allowance the
#: exclusion does not. Anything else is checked in the record that makes the
#: claim. ``host_side_digest_map`` was in this map in spirit until 2026-08-30 and
#: is what the rule was written from: its Triton delegate recomputed each digest
#: and then admitted ``device_identity.weld_survives_edit``, so the citation
#: landed on a strictly weaker claim than the exclusion advertised. It is now a
#: checked raw tier and the map is down to one.
DELEGATED_COVERAGE: Dict[str, str] = {
    "generated_kernel_source_slot":
        "test_metal_kernels.py::test_the_checked_in_fingerprints_match_the_tree",
}


def premise_violations(found: List["Found"], is_file: Callable[[str], bool],
                       inside_repo: Callable[[str], bool]) -> List[Tuple[str, str, str]]:
    """``(where, subject, why)`` for every exclusion whose premise no longer holds.

    THE ARGUMENT THIS FUNCTION MAKES. An exclusion is a decision not to compare a
    digest, and every one of them below rests on a reason. Two of those reasons
    are claims about this checkout rather than about the record, and a claim about
    the world that nothing evaluates is exactly the shape of the four defects this
    module was written for: it reads as coverage and is not.

      ``staging_only_file``       the named path must be ABSENT from the checkout.
      ``out_of_tree_sibling_path``  the named path must be outside this repo, or
                                    under the gitignored results tree.

    A violation is not a licence to delete the digest. It is a report that the
    digest has become checkable and must move into a tier — which is the direction
    coverage is allowed to move.

    The two predicates are injected for the same reason the comparators are: this
    module reads no file, so an armed control can drive it against a planted tree
    and watch it fail.
    """
    problems: List[Tuple[str, str, str]] = []
    for f in found:
        if f.rule not in PREMISE_ABSENT_FROM_TREE:
            continue
        where = dotted(f.trail)
        if f.subject is None:
            problems.append((where, "?", f"excluded as {f.rule} but names no path, "
                                         f"so its premise cannot be evaluated"))
            continue
        if f.rule == "staging_only_file":
            if is_file(f.subject):
                problems.append((where, f.subject,
                                 "declared staging-only, but the file IS in this "
                                 "checkout — the exclusion has expired; check the "
                                 "digest, do not delete it"))
            continue
        # out_of_tree_sibling_path
        if f.subject.startswith(GITIGNORED_PREFIX):
            continue
        if inside_repo(f.subject) and is_file(f.subject):
            problems.append((where, f.subject,
                             "excluded as out-of-tree, but the path resolves to a "
                             "file inside this checkout — check the digest"))
    return problems


def _keys(trail: Tuple[Any, ...]) -> List[str]:
    return [p for p in trail if isinstance(p, str)]


def _nearest_sha_key(trail: Tuple[Any, ...]) -> Optional[str]:
    """The closest ancestor-or-self key that ends in ``_sha256`` (or is ``sha256``)."""
    for part in reversed(trail):
        if isinstance(part, str) and (part.endswith("_sha256") or part == "sha256"):
            return part
    return None


def _classify(trail: Tuple[Any, ...], parent: Any) -> Tuple[str, Optional[str],
                                                            Optional[str], Optional[str]]:
    """``(rule, tier, subject, kernel)`` for one discovered digest.

    ``parent`` is the dict or list the digest sits directly in, needed only for the
    sibling-pin shape. NO DIGEST IS CONSULTED and no file is read: classification is a
    fact about the record's structure, so it cannot be aimed at an outcome.
    """
    leaf = trail[-1] if trail else None
    ancestors = _keys(trail[:-1])
    keys = _keys(trail)

    # --- historical containers, FIRST. See the block comment above on order. ---
    if "superseded_runs" in keys:
        return "historical_superseded_run", None, None, None
    if any(k.endswith("_at_recert") for k in keys):
        return "historical_recert_snapshot", None, None, None
    if any(k.endswith("_recut_log") or k == "source_drift_since_recert" for k in keys):
        return "historical_recut_log", None, None, None
    if "staging_only_file" in keys:
        # THE SUBJECT IS CARRIED so the premise can be decided. The block is
        # {path, sha256, why}; without the path the exclusion would rest on a
        # claim about a file nobody could name, which is where the last four
        # defects lived.
        claimed = parent.get("path") if isinstance(parent, dict) else None
        return ("staging_only_file", None,
                claimed if isinstance(claimed, str) else None, None)
    if "serial_driver" in ancestors:
        return "serial_driver_script", None, (leaf if isinstance(leaf, str) else None), None

    # --- shapes whose digests are about something other than a tree file's bytes ---
    if "device_sha256" in ancestors:
        # device_sha256.<path>.digests.<kernel>
        subject = next((k for k in ancestors[ancestors.index("device_sha256") + 1:]), None)
        return (TIER_DEVICE + ":device_kernel_digest", TIER_DEVICE, subject,
                leaf if isinstance(leaf, str) else None)
    if "host_sha256" in ancestors:
        # CHECKED, not delegated. The map key IS the file — a bare basename in
        # both records, resolved by the track's own resolver, which consults no
        # digest.
        return (TIER_RAW + ":host_side_digest_map", TIER_RAW,
                leaf if isinstance(leaf, str) else None, None)
    if "kernel_source_sha256" in ancestors:
        return "generated_kernel_source_slot", None, None, None

    nearest = _nearest_sha_key(trail)
    if nearest in ("artifact_sha256", "log_sha256"):
        return "artifact_or_log_digest", None, None, None

    # --- the per-capability run's bound: a digest OF DIGESTS, naming no file ---
    # It is what decides whether that run's evidence still describes the bytes the
    # entry pins, so it is compared against the entry rather than against a file, and
    # a run whose bound no longer matches is history BY DESIGN — the capability simply
    # stops being live. Checked by ``fastpath.live_capabilities`` and by the weld
    # contracts' liveness clause, not by this walk, and not delegated for that reason.
    if len(trail) >= 3 and trail[-1] == "bound_sha256" and trail[-3] == RUNS:
        return "run_bound_digest", None, None, None

    # --- the checked maps: <map key> IS the path ---
    if len(trail) >= 2 and isinstance(trail[-2], str) and isinstance(leaf, str):
        if trail[-2] == "source_sha256":
            return TIER_RAW + ":source_digest_map", TIER_RAW, leaf, None
        if trail[-2] == "code_sha256":
            return TIER_CODE + ":code_digest_map", TIER_CODE, leaf, None

    # --- the sibling shape: <name> beside <name>_sha256 ---
    if isinstance(leaf, str) and (leaf.endswith("_sha256") or leaf == "sha256") \
            and isinstance(parent, dict):
        stem = leaf[: -len("_sha256")] if leaf.endswith("_sha256") else "path"
        sibling = parent.get(stem) if stem else None
        if isinstance(sibling, str):
            if sibling.startswith(("~", "/")) or sibling.startswith(GITIGNORED_PREFIX):
                return "out_of_tree_sibling_path", None, sibling, None
            return TIER_RAW + ":sibling_script_pin", TIER_RAW, sibling, None
        # --- the block-keyed-by-file shape: <file>: {sha256: ...} ---
        # LAST, so it never steals a digest a more specific rule claims. The subject
        # is the PARENT KEY, which is why no sibling exists to find and why the old
        # walker saw nothing here at all.
        if leaf == "sha256" and len(trail) >= 2 and isinstance(trail[-2], str):
            return TIER_RAW + ":parent_key_named_pin", TIER_RAW, trail[-2], None
        return "pin_names_no_path", None, None, None

    return UNCLASSIFIED, None, None, None


def _parent_of(record: Any, trail: Tuple[Any, ...]) -> Any:
    node = record
    for part in trail[:-1]:
        node = node[part]
    return node


def census(record: Any) -> List[Found]:
    """Every digest in ``record``, classified. The denominator, structurally derived.

    Sorted by trail so a report reads in record order and two runs over the same file
    produce the same list.
    """
    out: List[Found] = []
    for trail, value in discover(record):
        rule, tier, subject, kernel = _classify(trail, _parent_of(record, trail))
        out.append(Found(trail, value, rule, tier, subject, kernel))
    out.sort(key=lambda f: dotted(f.trail))
    return out


def rules_that_fired(found: List[Found]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in found:
        counts[f.rule] = counts.get(f.rule, 0) + 1
    return counts


def compare(found: List[Found], resolve: Callable[[str], Any],
            raw_digest: Callable[[Any], str],
            code_digest_of: Callable[[Any], Optional[str]],
            device_digests_of: Callable[[Any], Optional[Dict[str, str]]],
            unbound_placeholder: Optional[str] = None,
            rebind_tool: str = "the track's rebind_*_welds.py") -> List[Tuple[str, str, str]]:
    """``(where, subject, why)`` for every CHECKED digest that the tree does not support.

    One place decides "does this recorded digest equal what the tree says", for all three
    tiers and all three records. The callables are injected rather than imported so this
    module reads no file itself and the armed controls can drive it against a planted
    tree — an assertion nobody has watched FAIL is a measurement that cannot fail.

    A tier that cannot be computed for a subject (a non-Python file has no code identity;
    a module that declares no device kernels has no device digest) is reported, never
    skipped: silently passing an incomputable comparison is the same defect class this
    module exists to remove.
    """
    problems: List[Tuple[str, str, str]] = []
    for f in found:
        if f.tier is None:
            continue
        where = dotted(f.trail)
        if f.subject is None:
            problems.append((where, "?", "CHECKED but names no subject"))
            continue
        if unbound_placeholder is not None and f.value == unbound_placeholder:
            problems.append((where, f.subject,
                             f"UNBOUND PLACEHOLDER — bind it with {rebind_tool}"))
            continue
        path = resolve(f.subject)
        if path is None or not path.is_file():
            problems.append((where, f.subject, "MISSING FROM THE TREE"))
            continue
        if f.tier == TIER_RAW:
            live = raw_digest(path)
        elif f.tier == TIER_CODE:
            live = code_digest_of(path)
            if live is None:
                problems.append((where, f.subject,
                                 "carries a code_sha256 for a file that has no code "
                                 "identity — code_digest parses Python and nothing else"))
                continue
        else:
            digests = device_digests_of(path)
            if digests is None or f.kernel not in digests:
                problems.append((where, f.subject,
                                 f"declares device kernel {f.kernel!r}, which the module "
                                 f"no longer generates"))
                continue
            live = digests[f.kernel]
        if live != f.value:
            problems.append((where, f.subject,
                             f"recorded {f.value[:12]} live {live[:12]}"))
    return problems


# ---------------------------------------------------------------------------
# THE ROUTE RECORD: a table's driver_dispatch entry, which is not a weld
# ---------------------------------------------------------------------------

#: The one verdict a route run may record and still license what it released.
ROUTE_RUN_PASS = "PASS"


def route_run_problems(record: Any, live: Sequence[str]) -> List[str]:
    """Why a ``driver_dispatch`` record's route runs do not carry it. Empty means they do.

    THE RECORD IS NOT A WELD. It pins the dispatch wiring's bytes, and its verdict is the
    route campaign's, filed per environment under ``runs[<key>]`` (a compute capability
    on the NVIDIA tables, a GPU architecture on Metal) by
    ``recut_driver_dispatch_record.py``. So it is asked two things about those runs:

      (a) at least one run is LIVE, so an empty ``runs`` or one whose every run binds
          bytes that have since moved fails rather than passing a loop over nothing;
      (b) every live run reads ``status`` PASS. A run that is not live is history and
          is not read: what it says certifies bytes that no longer ship.

    ``live`` is ``fastpath.live_capabilities(record)``, the one liveness rule, passed in
    so this module imports nothing from the package. The tree and the record's shape are
    :func:`route_record_problems`'s other two clauses; this half is exposed alone for the
    dispatch contract, which compares the tree its own way.
    """
    runs = record.get(RUNS) if isinstance(record, Mapping) else None
    problems: List[str] = []
    if not live:
        recorded = sorted(runs, key=str) if isinstance(runs, Mapping) else runs
        problems.append(f"no live run under {RUNS} (recorded: {recorded!r}): no route "
                        f"campaign has released on the bytes this record binds")
    for key in live:
        run = runs.get(key) if isinstance(runs, Mapping) else None
        status = run.get("status") if isinstance(run, Mapping) else None
        if status != ROUTE_RUN_PASS:
            problems.append(f"{RUNS}[{key!r}] reads status {status!r}, not "
                            f"{ROUTE_RUN_PASS}: a route run that did not release "
                            f"licenses nothing")
    return problems


def route_record_problems(record: Any, live: Sequence[str], moved: Sequence[str],
                          stranded: Sequence[str]) -> List[str]:
    """Why a ``driver_dispatch`` record is not supported by its route runs and the tree.

    Empty means it is. The rule all three weld contracts apply to their table's record,
    in place of the weld status rule, which asks an entry-level ``status`` of a weld:

      (a), (b)  :func:`route_run_problems`;
      (c)       ``moved``: the pinned paths the contract's raw-tier drift measurement
                reports. A live run certifies the record's digests, not the tree, so a
                file that moved since the cut is reported here even while the run
                stays live;
      (d)       ``stranded``: the contract's shape reasons over the route-run fields
                (``fastpath.DISPATCH_RUN_FIELDS``, ``metal_runs.DISPATCH_RUN_FIELDS``).
                ``status`` is one of them, so an entry-level status is refused: the
                verdict has one home, the run that measured it.

    ``moved`` and ``stranded`` are required: a default of nothing would let a caller
    skip a clause and still read as having checked it.
    """
    problems = route_run_problems(record, live)
    if moved:
        problems.append(f"{len(moved)} pinned file(s) moved since the record was cut: "
                        f"{sorted(moved)}")
    problems.extend(stranded)
    return problems
