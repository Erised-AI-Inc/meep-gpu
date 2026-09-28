"""The walker itself, held to the rules it exists to enforce.

WHY THIS FILE. :mod:`meep_gpu.weld_record_walk` is now the single denominator for
three weld records — 1,772 digests across the Triton, Metal and CUDA ledgers — and
a defect in it would be invisible in exactly the way the four defects before it
were: every contract test would still report a stable count and a green tier,
because every count they report comes from this module.

So the walker is tested here, not only through the records:

  * the WALK reaches digests in dicts, in lists, and at depths past anything the
    real records hold;
  * the ORDER of the rules is asserted, because it is load-bearing — a
    ``superseded_runs`` block holds a ``source_sha256`` map identical in shape to
    a live one, and if ``source_digest_map`` matched first the record's own
    ``why_superseded`` would be overruled by which rule happens to be written
    higher up;
  * NO RULE IS DEAD: every rule in :data:`RULE_REASONS` fires on at least one of
    the three real records, so a rule cannot outlive the shape it was written for
    and sit around as a place for the next omission to hide;
  * and the CLASSIFIER READS NO FILE AND NO DIGEST, which is what makes it
    impossible to aim at an outcome.

Stdlib and pytest. No device, no tree writes outside tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

from . import weld_record_walk as walk

PACKAGE = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE.parent

RECORDS = {
    "triton": PACKAGE / "triton_kernels" / "fingerprints.json",
    "metal": PACKAGE / "metal_kernels" / "fingerprints.json",
    "cuda": PACKAGE / "cuda_kernels" / "fingerprints.json",
}

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _all_records():
    return {name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in RECORDS.items()}


# ---------------------------------------------------------------------------
# THE WALK
# ---------------------------------------------------------------------------


def test_the_walk_reaches_dicts_lists_and_arbitrary_depth():
    """Every container, and no floor on depth. The whole premise, executed."""
    record = {
        "flat": A,
        "in_a_dict": {"deeper": {"deepest": B}},
        "in_a_list": [{"x": C}, ["nested", {"y": A}]],
    }
    found = dict((walk.dotted(t), v) for t, v in walk.discover(record))
    assert found == {
        "flat": A,
        "in_a_dict.deeper.deepest": B,
        "in_a_list[0].x": C,
        "in_a_list[1][1].y": A,
    }, found


def test_the_walk_ignores_everything_that_is_not_a_64_hex_string():
    """A digest is 64 lowercase hex characters. Nothing else is collected, so the
    counts the contracts assert are counts of digests rather than of strings."""
    record = {
        "short": "a" * 63,
        "long": "a" * 65,
        "uppercase": "A" * 64,
        "not_hex": "g" * 64,
        "an_int": 12345,
        "none": None,
        "a_real_one": A,
    }
    assert [v for _, v in walk.discover(record)] == [A]


def test_a_digest_at_a_depth_no_record_reaches_is_still_found_and_still_fails():
    """THE ARMED CONTROL, shared by all three contracts and executed once here too.

    Both halves matter. A walker that stopped early would report a stable count
    and zero unclassified digests, which is indistinguishable from a correct one.
    A walker that found the digest and quietly classified it as excluded would be
    the original defect wearing this test as cover.
    """
    deep = {"e": {"a": [{"b": {"c": [{"d": {"a_key_nobody_wrote": A}}]}}]}}
    found = walk.census(deep)
    assert len(found) == 1
    assert walk.dotted(found[0].trail) == "e.a[0].b.c[0].d.a_key_nobody_wrote"
    assert found[0].rule == walk.UNCLASSIFIED
    assert found[0].tier is None

    deepest_real = max(len(f.trail)
                       for record in _all_records().values()
                       for f in walk.census(record))
    assert len(found[0].trail) > deepest_real, (
        f"the control plants at depth {len(found[0].trail)} and the records reach "
        f"{deepest_real} — it has stopped covering anything the real walk does "
        f"not already reach")


# ---------------------------------------------------------------------------
# THE RULES, AND THE ORDER THEY RUN IN
# ---------------------------------------------------------------------------


def test_a_superseded_source_map_is_history_and_a_live_one_is_checked():
    """THE ORDER IS THE RULE. Same inner shape, opposite fates.

    ``superseded_runs[0].source_sha256`` is a ``source_sha256`` map in every
    respect except the one that matters: the record says out loud it is not about
    this tree. If ``source_digest_map`` were consulted first, 22 real Triton pins
    would be compared as though current and the only way to clear the red would be
    to delete the history.
    """
    live = walk.census({"g": {"source_sha256": {"meep_gpu/driver.py": A}}})
    assert [f.rule for f in live] == [walk.TIER_RAW + ":source_digest_map"]
    assert live[0].tier == walk.TIER_RAW and live[0].subject == "meep_gpu/driver.py"

    history = walk.census(
        {"g": {"superseded_runs": [{"source_sha256": {"meep_gpu/driver.py": A}}]}})
    assert [f.rule for f in history] == ["historical_superseded_run"]
    assert history[0].tier is None


@pytest.mark.parametrize("record,rule", [
    ({"g": {"recert_x": {"source_sha256_at_recert": {"a.py": A}}}},
     "historical_recert_snapshot"),
    ({"g": {"source_sha256_recut_log": [{"from": A, "to": B}]}},
     "historical_recut_log"),
    ({"g": {"source_drift_since_recert": {"a.py": {"certified": [A]}}}},
     "historical_recut_log"),
    ({"g": {"staging": {"staging_only_file": {"path": "p.py", "sha256": A}}}},
     "staging_only_file"),
    ({"g": {"serial_driver": {"chain.sh": A}}}, "serial_driver_script"),
    ({"g": {"device_sha256": {"m.py": {"digests": {"k": A}}}}},
     walk.TIER_DEVICE + ":device_kernel_digest"),
    ({"host_sha256": {"a.py": A}}, walk.TIER_RAW + ":host_side_digest_map"),
    ({"g": {"kernel_source_sha256": {"slot": {"sha256": A}}}},
     "generated_kernel_source_slot"),
    ({"g": {"artifact_sha256": A}}, "artifact_or_log_digest"),
    ({"g": {"log_sha256": A}}, "artifact_or_log_digest"),
    ({"g": {"code_sha256": {"a.py": A}}}, walk.TIER_CODE + ":code_digest_map"),
    ({"g": {"probe": "parity/meep_gpu/p.py", "probe_sha256": A}},
     walk.TIER_RAW + ":sibling_script_pin"),
    ({"g": {"probe": "~/elsewhere/p.py", "probe_sha256": A}},
     "out_of_tree_sibling_path"),
    ({"g": {"probe": "parity/meep_gpu/results/r/p.py", "probe_sha256": A}},
     "out_of_tree_sibling_path"),
    ({"g": {"gate_sha256": A}}, "pin_names_no_path"),
    ({"g": {"a.py": {"sha256": A}}}, walk.TIER_RAW + ":parent_key_named_pin"),
    ({"g": {"a_key_nobody_wrote": A}}, walk.UNCLASSIFIED),
])
def test_each_rule_claims_the_shape_it_was_written_for(record, rule):
    """One minimal record per rule. A rule whose predicate stops matching its own
    shape would otherwise show up only as a count that quietly moved."""
    found = walk.census(record)
    assert found, "the shape yielded no digest at all"
    assert {f.rule for f in found} == {rule}, [f.rule for f in found]


def test_every_rule_has_a_written_reason_and_none_is_dead():
    """BOTH DIRECTIONS on the rule table itself.

    A rule with no reason is an exclusion nobody argued for. A rule that fires on
    none of the three real records is a rule whose shape has left the tree, and an
    unexercised rule is exactly where the next omission hides — the same argument
    the contracts make about a stale exclusion NAME, applied to the rules.
    """
    fired = set()
    for record in _all_records().values():
        fired |= set(walk.rules_that_fired(walk.census(record)))
    assert walk.UNCLASSIFIED not in fired, (
        f"a real record holds a digest no rule claims — the contract tests name "
        f"which; this one only reports that the rule table is incomplete")
    reasonless = sorted(fired - set(walk.RULE_REASONS))
    assert not reasonless, f"rules that fire with no written reason: {reasonless}"
    # UNCLASSIFIED is exercised by the planted control above rather than by a
    # record, so it is the one name allowed to be absent from `fired`.
    dead = sorted(set(walk.RULE_REASONS) - fired - {walk.UNCLASSIFIED})
    assert not dead, (
        f"rules that fire on none of the three records: {dead} — the shape each "
        f"was written for has left the tree. Delete the rule, or find what moved")
    empty = sorted(name for name, reason in walk.RULE_REASONS.items()
                   if not reason or len(reason) < 40)
    assert not empty, f"rules whose reason says nothing usable: {empty}"


def test_a_path_naming_exclusion_carries_the_path_it_excuses():
    """An exclusion that cannot name its subject cannot have its premise checked.

    ``staging_only_file`` used to return ``subject=None``, so "the record declares
    this file untracked" was a sentence in a docstring with nothing behind it.
    Both path-naming exclusions now carry the path, which is what makes
    :func:`walk.premise_violations` able to decide anything at all.
    """
    staged = walk.census(
        {"g": {"staging": {"staging_only_file":
                           {"path": "parity/meep_gpu/cases.py", "sha256": A}}}})
    assert [(f.rule, f.subject) for f in staged] == [
        ("staging_only_file", "parity/meep_gpu/cases.py")]

    oot = walk.census({"g": {"probe": "~/elsewhere/p.py", "probe_sha256": A}})
    assert [(f.rule, f.subject) for f in oot] == [
        ("out_of_tree_sibling_path", "~/elsewhere/p.py")]


def test_the_premise_of_a_not_in_this_checkout_exclusion_is_decided_not_assumed():
    """THE ARMED CONTROL FOR THE FIFTH SUB-FORM, driven in both directions.

    An exclusion is a decision NOT to compare a digest, and these two rest on a
    claim about the tree that can stop being true. Watched failing here, because
    a premise check nobody has seen fail is the same measurement problem as an
    assertion nobody has seen fail — and it is the problem this whole module was
    written for.
    """
    staged = walk.census(
        {"g": {"staging": {"staging_only_file":
                           {"path": "parity/meep_gpu/cases.py", "sha256": A}}}})
    absent = walk.premise_violations(staged, is_file=lambda p: False,
                                     inside_repo=lambda p: True)
    assert absent == [], absent
    present = walk.premise_violations(staged, is_file=lambda p: True,
                                      inside_repo=lambda p: True)
    assert len(present) == 1 and "the file IS in this checkout" in present[0][2], present

    # A results-tree path stays excluded even when the file is present: the
    # premise there is "gitignored", not "absent", and the two are different
    # claims. Pinned so the rule cannot quietly become the stricter one.
    ignored = walk.census({"g": {"probe": walk.GITIGNORED_PREFIX + "r/p.py",
                                 "probe_sha256": A}})
    assert walk.premise_violations(ignored, lambda p: True, lambda p: True) == []

    # A repo-relative path OUTSIDE the results tree that comes back into the
    # checkout must stop being excludable.
    returned = walk.census({"g": {"probe": "~/gone/p.py", "probe_sha256": A}})
    back = walk.premise_violations(returned, is_file=lambda p: True,
                                   inside_repo=lambda p: True)
    assert len(back) == 1 and "resolves to a file inside this checkout" in back[0][2]

    # And an exclusion that names nothing is reported as unevaluable rather than
    # passing: silence is what this module exists to remove.
    anonymous = [walk.Found(("g", "probe_sha256"), A, "out_of_tree_sibling_path",
                            None, None)]
    unnamed = walk.premise_violations(anonymous, lambda p: False, lambda p: False)
    assert len(unnamed) == 1 and "names no path" in unnamed[0][2]


def test_only_the_premise_bearing_rules_are_evaluated():
    """Both directions on the SET itself. A rule that quietly joined
    PREMISE_ABSENT_FROM_TREE would start reporting; one that quietly left would
    stop — and stopping is the direction that goes unnoticed."""
    assert set(walk.PREMISE_ABSENT_FROM_TREE) == {
        "staging_only_file", "out_of_tree_sibling_path"}, walk.PREMISE_ABSENT_FROM_TREE
    # serial_driver_script names a BARE BASENAME and is deliberately outside the
    # set; its reason says so. Evaluating it would be a predicate past what the
    # record can answer.
    serial = walk.census({"g": {"serial_driver": {"chain.sh": A}}})
    assert walk.premise_violations(serial, lambda p: True, lambda p: True) == []
    checked = walk.census({"g": {"source_sha256": {"meep_gpu/driver.py": A}}})
    assert walk.premise_violations(checked, lambda p: True, lambda p: True) == []


def test_the_delegation_register_names_real_rules_and_only_excluded_ones():
    """THE CITATION REGISTER, pinned in both directions.

    :data:`walk.DELEGATED_COVERAGE` is the list of exclusions whose digests are
    claimed to be compared somewhere else, and it is a REGISTER rather than a
    phrase in a docstring for a reason string-sniffing cannot serve: several
    reasons cross-reference a test that EARNS an exclusion
    (``historical_superseded_run`` points at its three legs), and "mentions a
    test" is not "is compared over there". A predicate that could not tell those
    apart would turn an earning citation into a red and teach the next author to
    stop writing them.

    Three invariants, each a way the register could go quietly wrong:

      1. every key is a REAL rule — a register entry for a rule that does not
         exist excuses nothing and hides that it excuses nothing;
      2. every key is an EXCLUSION — a checked tier in this map would be a rule
         that is both compared here and claimed to be compared elsewhere;
      3. every value is a RESOLVABLE ``file.py::function`` citation, which the
         Triton and CUDA contracts then resolve against the tree.
    """
    unknown = sorted(set(walk.DELEGATED_COVERAGE) - set(walk.RULE_REASONS))
    assert not unknown, f"the register names rules that do not exist: {unknown}"
    checked_tiers = {name for name in walk.RULE_REASONS
                     if name.startswith((walk.TIER_RAW + ":", walk.TIER_CODE + ":",
                                         walk.TIER_DEVICE + ":"))}
    both = sorted(set(walk.DELEGATED_COVERAGE) & checked_tiers)
    assert not both, (
        f"{both} are compared HERE and also claimed to be compared elsewhere — "
        f"one of the two is not true; delete the register entry or the tier")
    malformed = sorted(name for name, citation in walk.DELEGATED_COVERAGE.items()
                       if not re.fullmatch(r"test_[a-z0-9_]+\.py::test_[a-z0-9_]+",
                                           citation))
    assert not malformed, (
        f"citations that cannot be resolved to a file and a function: {malformed}")


def test_the_classifier_reads_no_file_and_no_digest(monkeypatch, tmp_path):
    """WHAT MAKES THE CLASSIFICATION UNSTEERABLE.

    If classification consulted the tree, a shape's fate could depend on whether
    the comparison would pass — which is the mechanism behind every "green bought
    by narrowing a check" this campaign has removed. Two legs:

      1. Changing only the DIGEST VALUE changes no rule, no tier and no subject.
      2. Running the census with the working directory moved to an empty tree
         yields the identical classification, so nothing was read off disk.
    """
    record = {"g": {"source_sha256": {"meep_gpu/driver.py": A},
                    "code_sha256": {"meep_gpu/driver.py": A},
                    "probe": "parity/meep_gpu/p.py", "probe_sha256": A}}
    other = json.loads(json.dumps(record).replace(A, B))
    assert [(f.rule, f.tier, f.subject) for f in walk.census(record)] == \
           [(f.rule, f.tier, f.subject) for f in walk.census(other)]

    before = [(walk.dotted(f.trail), f.rule, f.tier, f.subject)
              for f in walk.census(record)]
    monkeypatch.chdir(tmp_path)
    after = [(walk.dotted(f.trail), f.rule, f.tier, f.subject)
             for f in walk.census(record)]
    assert before == after


# ---------------------------------------------------------------------------
# THE COMPARISON
# ---------------------------------------------------------------------------


def _comparators(root):
    def resolve(name):
        return root / name

    def raw(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return resolve, raw


def test_compare_reports_each_tier_and_never_skips_an_incomputable_one(tmp_path):
    """A tier that CANNOT be computed is reported, not passed over.

    Silently passing an incomputable comparison is the same defect class as a
    filtered denominator: the loop runs, the count looks right, and the claim was
    never checked. Four legs, one per way a comparison can fail to happen.
    """
    (tmp_path / "meep_gpu").mkdir()
    victim = tmp_path / "meep_gpu" / "m.py"
    victim.write_text("X = 1\n", encoding="utf-8")
    true_raw = hashlib.sha256(victim.read_bytes()).hexdigest()
    resolve, raw = _comparators(tmp_path)

    def never(_):
        return None

    honest = walk.census({"g": {"source_sha256": {"meep_gpu/m.py": true_raw}}})
    assert walk.compare(honest, resolve, raw, never, never) == []

    planted = walk.census({"g": {"source_sha256": {"meep_gpu/m.py": A}}})
    assert walk.compare(planted, resolve, raw, never, never) == [
        ("g.source_sha256.meep_gpu/m.py", "meep_gpu/m.py",
         f"recorded {A[:12]} live {true_raw[:12]}")]

    absent = walk.census({"g": {"source_sha256": {"meep_gpu/gone.py": A}}})
    assert walk.compare(absent, resolve, raw, never, never) == [
        ("g.source_sha256.meep_gpu/gone.py", "meep_gpu/gone.py",
         "MISSING FROM THE TREE")]

    # A CODE PIN WHOSE FILE HAS NO CODE IDENTITY: reported, never skipped.
    code = walk.census({"g": {"code_sha256": {"meep_gpu/m.py": A}}})
    reported = walk.compare(code, resolve, raw, never, never)
    assert len(reported) == 1 and "no code identity" in reported[0][2], reported

    # A DEVICE PIN NAMING A KERNEL THE MODULE NO LONGER GENERATES: reported.
    device = walk.census(
        {"g": {"device_sha256": {"meep_gpu/m.py": {"digests": {"k": A}}}}})
    reported = walk.compare(device, resolve, raw, never,
                            lambda _: {"a_different_kernel": A})
    assert len(reported) == 1 and "no longer generates" in reported[0][2], reported


def test_compare_ignores_excluded_digests_and_only_those(tmp_path):
    """The excluded tier is not compared — and nothing else is exempt.

    Pinned in both directions so "excluded" cannot silently widen: a record made
    entirely of excluded shapes yields no comparison at all, and adding ONE
    checked digest to the same record yields exactly one report.
    """
    resolve, raw = _comparators(tmp_path)

    def never(_):
        return None

    excluded_only = walk.census({"g": {"artifact_sha256": A, "log_sha256": B,
                                       "superseded_runs": [
                                           {"source_sha256": {"meep_gpu/m.py": C}}]}})
    assert len(excluded_only) == 3
    assert all(f.tier is None for f in excluded_only)
    assert walk.compare(excluded_only, resolve, raw, never, never) == []

    plus_one = walk.census({"g": {"artifact_sha256": A, "log_sha256": B,
                                  "superseded_runs": [
                                      {"source_sha256": {"meep_gpu/m.py": C}}],
                                  "source_sha256": {"meep_gpu/m.py": C}}})
    reported = walk.compare(plus_one, resolve, raw, never, never)
    assert len(reported) == 1 and reported[0][2] == "MISSING FROM THE TREE"


def test_the_unbound_placeholder_is_reported_as_itself(tmp_path):
    """A pin waiting on a rebind is a failure with a DIFFERENT remedy, and saying
    which is the difference between a red someone clears and a red someone
    deletes. Both directions: with no placeholder declared, the same value is
    ordinary drift."""
    (tmp_path / "meep_gpu").mkdir()
    (tmp_path / "meep_gpu" / "m.py").write_text("X = 1\n", encoding="utf-8")
    resolve, raw = _comparators(tmp_path)
    zeroes = "0" * 64
    found = walk.census({"g": {"source_sha256": {"meep_gpu/m.py": zeroes}}})

    reported = walk.compare(found, resolve, raw, lambda _: None, lambda _: None,
                            unbound_placeholder=zeroes, rebind_tool="rebind_x.py")
    assert len(reported) == 1
    assert "UNBOUND PLACEHOLDER" in reported[0][2] and "rebind_x.py" in reported[0][2]

    plain = walk.compare(found, resolve, raw, lambda _: None, lambda _: None)
    assert len(plain) == 1 and plain[0][2].startswith("recorded 000000000000 live ")
