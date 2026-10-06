"""The Triton binders read a run's subnormal policy from every spelling its gate uses.

``seed_triton_welds.policy_line`` is the one reader behind ``seed_triton_welds.py``,
``rebind_triton_welds.py`` (fleet rebind and ``--bind-bit-identity``) and
``recut_composition_records.py --family-recert``. Until it read more than the
top-level ``subnormal_policy`` block, nine released gates of the 2026-10-03 round --
seven of them cited by dispatch arms -- could not be bound, because they state the
policy as ``subnormal_policy_installed``, ``subnormal_policy_stamp``,
``summary.subnormal_policy`` or a ``policy`` stamp, and one states none at all under
a declared ``_mixed_policy`` exception.

Every artifact here is synthetic and shaped after the real one named beside it, so
the suite needs no round directory. What each test pins:

* each answering spelling, alone, yields a line that names the policy and says
  where it was read from;
* the top-level spelling yields exactly the line it always did, so no record the
  reader could already write moves;
* requests and stamp names cross-check but never answer;
* a stamp answers only with its ``resolved`` key, and only when it was installed: an
  uninstalled stamp, or one resolving to neither policy, denies that a policy was
  attained and is refused beside any answer;
* two statements naming different policies are refused by name;
* the ``_mixed_policy`` exception yields the existing records' exact string only
  where the artifact states nothing, and is refused where the run states a policy.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import seed_triton_welds as seed  # noqa: E402

LEDGER = HERE.parent.parent / "meep_gpu" / "triton_kernels" / "fingerprints.json"

#: The contract test's own pattern (``test_triton_weld_contract._POLICY_WORD``).
POLICY_WORD = re.compile(r"\b(keep|flush)\b", re.IGNORECASE)

#: The shape ``subnormal_policy.policy_stamp()`` writes, as the fleet gates carry it.
KEEP_STAMP = {
    "requested": "keep", "resolved": "keep", "policy": "ieee_keep_ftz_stripped",
    "cache_dir": "run_root/caches/cupy_cache_ftz_stripped_family", "installed": True,
}
FLUSH_STAMP = {
    "requested": "flush", "resolved": "flush", "policy": "meep_x86_flush",
    "cache_dir": "run_root/caches/cupy_cache_flush_family",
}


def nest(path, value) -> dict:
    """``("summary", "subnormal_policy")`` -> ``{"summary": {"subnormal_policy": value}}``."""
    out = value
    for key in reversed(path):
        out = {key: out}
    return out


def artifact(**extra) -> dict:
    """A released gate artifact that states nothing about a policy until told to."""
    base = {"gate": "synthetic", "canonical_verdict": {"released": True},
            "imported_source_sha256": {"meep_gpu/subnormal_policy.py": "0" * 64}}
    base.update(extra)
    return base


# --- the top-level spelling is unchanged ---------------------------------------------


def test_the_top_level_block_yields_exactly_the_line_it_always_did():
    """Pinned to the string the shipped records already carry, character for character.

    Every one of the 36 welds the 2026-10-03 rebind could already bind reads this
    branch; a change here would move all of their records, not just the nine new ones.
    """
    line = seed.policy_line(artifact(subnormal_policy=dict(KEEP_STAMP)))
    assert line == (
        "keep (installed by the gate before its first device compile: requested=keep "
        "resolved=keep; stamp ieee_keep_ftz_stripped; the CuPy cache directory carries "
        "ftz_stripped)")


def test_a_top_level_block_without_a_cache_directory_says_so():
    """The bit-identity probe's shape: ``resolved`` with no ``requested`` and no cache key."""
    block = {"resolved": "keep", "policy": "ieee_keep_ftz_stripped", "installed": True}
    assert seed.policy_line(artifact(subnormal_policy=block)) == (
        "keep (installed by the gate before its first device compile: requested=? "
        "resolved=keep; stamp ieee_keep_ftz_stripped; the run recorded no CuPy cache "
        "directory)")


# --- each answering spelling, alone ---------------------------------------------------

#: (path, the real artifact it is shaped after)
ANSWERING_BLOCKS = [
    (("subnormal_policy_stamp",), "complex_ade, complex_no_pml_stored_e"),
    (("summary", "subnormal_policy"), "cylindrical_complex"),
    (("policy",), "no_pml_constitutive"),
    (("policy_stamp",), "complex_conductive_fused_pair"),
    (("environment", "subnormal_policy_stamp"), "cylindrical_fused_electric_pair"),
    (("subnormal_policy_at_start",), "folded_offdiag"),
    (("subnormal", "subnormal_policy"), "nonlinear, offdiag"),
    (("synthetic", "subnormal", "subnormal_policy"), "bfast"),
    (("subnormal_policy_install",), "the bit-identity probe"),
]


@pytest.mark.parametrize("path,shaped_after", ANSWERING_BLOCKS,
                         ids=[".".join(p) for p, _ in ANSWERING_BLOCKS])
def test_each_stamp_block_answers_alone_and_says_where(path, shaped_after):
    """A run whose only statement is this block is bound, and its line names the block."""
    line = seed.policy_line(artifact(**nest(path, dict(KEEP_STAMP))))
    assert line is not None, f"{'.'.join(path)} ({shaped_after}) was not read"
    assert POLICY_WORD.match(line) and line.startswith("keep ("), line
    assert line.endswith(f"; read from {'.'.join(path)})"), line
    assert "requested=keep resolved=keep; stamp ieee_keep_ftz_stripped" in line
    assert "the CuPy cache directory carries ftz_stripped" in line


def test_a_stamp_nested_under_a_top_level_block_that_states_no_resolution_answers():
    """``subnormal_policy.stamp``: the outer block names no resolution, the inner one does."""
    fresh = artifact(subnormal_policy={"policy": "keep", "stamp": dict(KEEP_STAMP)})
    line = seed.policy_line(fresh)
    assert line is not None and line.startswith("keep (")
    assert line.endswith("; read from subnormal_policy.stamp)"), line


@pytest.mark.parametrize("policy", ["keep", "flush"])
def test_an_installed_name_alone_answers_and_claims_no_mechanism(policy):
    """The four no-PML gates: ``subnormal_policy_installed`` and no stamp block at all.

    The line names the policy for the contract test and says, rather than supplies,
    that the mechanism and the cache directory are unstated.
    """
    line = seed.policy_line(artifact(subnormal_policy_installed=policy))
    assert line == (
        f"{policy} (the gate's own subnormal_policy_installed='{policy}'; the run "
        f"recorded no policy stamp, so neither the mechanism nor a CuPy cache "
        f"directory is stated)")
    assert "cache_dir" not in line and "/" not in line


def test_the_line_never_embeds_the_cache_path():
    """A cache directory holds an account's home path; the line reports only its token."""
    line = seed.policy_line(artifact(subnormal_policy_stamp=dict(KEEP_STAMP)))
    assert KEEP_STAMP["cache_dir"] not in line and "run_root" not in line


def test_the_flush_policy_is_read_the_same_way():
    line = seed.policy_line(artifact(summary={"subnormal_policy": dict(FLUSH_STAMP)}))
    assert line.startswith("flush (") and "stamp meep_x86_flush" in line
    assert "the CuPy cache directory carries no ftz token" in line


# --- requests and stamp names never answer --------------------------------------------


def test_a_request_alone_is_not_an_attainment():
    """The fused-pair probes' launch-configuration ``policy`` dict carries the
    ``--subnormal-policy`` ARGUMENT; a run stating only that has not said what it
    installed, and the reader refuses to say it for them."""
    fresh = artifact(policy={"block": "kernels.DEFAULT_BLOCK", "num_warps": 1,
                             "subnormal_policy": "keep"},
                     subnormal_policy_requested="keep")
    assert seed.policy_line(fresh) is None


def test_a_stamp_name_alone_is_not_an_attainment():
    fresh = artifact(summary={"certified_under_subnormal_policy": "ieee_keep_ftz_stripped"})
    assert seed.policy_line(fresh) is None


def test_a_block_that_only_asks_match_meep_names_no_policy():
    """``match_meep`` is a question; without ``resolved`` the block names no answer."""
    fresh = artifact(subnormal_policy={"requested": "match_meep",
                                       "cache_dir": "run_root/c"})
    assert seed.policy_line(fresh) is None


def test_a_match_meep_request_is_compared_with_nothing():
    """A run that asked ``match_meep`` and measured ``flush`` is consistent."""
    block = dict(FLUSH_STAMP, requested="match_meep")
    fresh = artifact(subnormal_policy=block,
                     policy={"num_warps": 1, "subnormal_policy": "match_meep"})
    line = seed.policy_line(fresh)
    assert line.startswith("flush (installed by the gate before its first device "
                           "compile: requested=match_meep resolved=flush")


def test_two_vocabularies_for_one_policy_agree():
    """cylindrical_fused_electric_pair: ``policy: keep`` above a stamp spelling
    ``ieee_keep_ftz_stripped``, plus the request and the certified-under name. Names
    are compared by the policy they mean, so this binds."""
    fresh = artifact(
        subnormal_policy={"requested": "keep", "resolved": "keep", "policy": "keep",
                          "cache_dir": "run_root/cupy_cache_ftz_stripped_x",
                          "stamp": dict(KEEP_STAMP)},
        policy_stamp=dict(KEEP_STAMP),
        environment={"subnormal_policy_stamp": dict(KEEP_STAMP)},
        policy={"num_warps": 1, "subnormal_policy": "keep"},
        summary={"certified_under_subnormal_policy": "ieee_keep_ftz_stripped"},
        subnormal_policy_requested="keep")
    assert seed.policy_line(fresh).startswith(
        "keep (installed by the gate before its first device compile: requested=keep "
        "resolved=keep; stamp keep;")


def test_the_containers_that_describe_another_run_are_not_read():
    """``policy_stamps`` (and its copy under ``environment``), ``policies`` and
    ``rows[].policy`` hold both cuts of a two-policy probe; ``expansion_probe`` and
    ``unified_licenses`` describe the licence artifact the gate LOADED. A flush stamp
    in any of them beside this run's keep is not a contradiction."""
    fresh = artifact(
        subnormal_policy=dict(KEEP_STAMP),
        policy_stamps={"keep": {"install": dict(KEEP_STAMP)},
                       "flush": {"install": dict(FLUSH_STAMP)}},
        environment={"subnormal_policy_stamps": {"flush": dict(FLUSH_STAMP)}},
        policies=["keep", "flush"],
        rows=[{"policy": "keep"}, {"policy": "flush"}],
        expansion_probe={"subnormal_policy": dict(FLUSH_STAMP)},
        expansion={"subnormal_policy": dict(FLUSH_STAMP)},
        unified_licenses={"base": {"policy": "meep_x86_flush",
                                   "policy_resolved": "flush"}})
    assert seed.policy_line(fresh).startswith("keep (")
    for name in ("policy_stamps", "environment.subnormal_policy_stamps", "policies",
                 "rows[].policy", "expansion", "expansion_probe", "unified_licenses"):
        assert name in seed.POLICY_NOT_READ


# --- a stamp answers only with an installed resolution ---------------------------------

#: ``subnormal_policy.policy_stamp()`` called BEFORE any install: ``resolved`` is then a
#: preference, and the stamp says so twice.
UNINSTALLED_STAMP = {
    "installed": False, "policy": "none_installed", "would_be_policy": "ieee_keep_ftz_stripped",
    "requested": "match_meep", "resolved": "keep", "cache_dir": "run_root/c",
}


def test_an_uninstalled_stamp_alone_answers_nothing():
    """The stamp's ``resolved`` is a preference nothing in the process earned; a line
    built from it would claim the gate installed a policy it never installed."""
    assert seed.policy_line(artifact(subnormal_policy=dict(UNINSTALLED_STAMP))) is None
    assert seed.policy_line(artifact(subnormal_policy_at_start=dict(UNINSTALLED_STAMP))) is None


@pytest.mark.parametrize("uninstalled", [
    {"installed": False},
    {"policy": "none_installed"},
], ids=["installed-false", "policy-none-installed"])
def test_either_mark_of_an_uninstalled_stamp_is_enough(uninstalled):
    block = dict(KEEP_STAMP, **uninstalled)
    assert seed.policy_line(artifact(subnormal_policy_stamp=block)) is None


def test_an_uninstalled_stamp_beside_an_answer_is_refused():
    fresh = artifact(subnormal_policy_at_start=dict(UNINSTALLED_STAMP),
                     subnormal_policy_installed="keep")
    with pytest.raises(seed.PolicyStampConflict) as caught:
        seed.policy_line(fresh)
    message = str(caught.value)
    assert "denies and states" in message and "subnormal_policy_at_start" in message
    assert "subnormal_policy_installed=keep" in message, message


def test_an_uninstalled_stamp_counts_as_stated_for_the_declared_exception():
    with pytest.raises(seed.PolicyStampConflict, match="_mixed_policy"):
        seed.policy_line(artifact(subnormal_policy=dict(UNINSTALLED_STAMP)), MIXED_ENTRY)


def test_a_request_only_block_answers_nothing():
    """``resolved: null`` beside ``requested: keep``: the question was asked and the
    stamp records no answer, so the reader records none either."""
    fresh = artifact(subnormal_policy_stamp={"requested": "keep", "resolved": None,
                                             "cache_dir": "run_root/c"})
    assert seed.policy_line(fresh) is None
    fresh = artifact(subnormal_policy={"requested": "flush", "cache_dir": "run_root/c"})
    assert seed.policy_line(fresh) is None


def test_a_request_only_block_still_cross_checks():
    fresh = artifact(subnormal_policy_stamp={"requested": "flush", "resolved": None},
                     subnormal_policy_installed="keep")
    with pytest.raises(seed.PolicyStampConflict, match="subnormal_policy_stamp.requested=flush"):
        seed.policy_line(fresh)


def test_a_resolution_naming_no_policy_answers_nothing_alone():
    assert seed.policy_line(artifact(subnormal_policy=dict(KEEP_STAMP, resolved="none"))) is None


def test_a_resolution_naming_no_policy_beside_an_answer_is_refused():
    fresh = artifact(subnormal_policy=dict(KEEP_STAMP, resolved="none"),
                     subnormal_policy_installed="keep")
    with pytest.raises(seed.PolicyStampConflict, match="resolved='none'"):
        seed.policy_line(fresh)


# --- strings at stamp paths, and the stamp names, are compared ---------------------------


@pytest.mark.parametrize("extra,named", [
    ({"subnormal_policy": "flush"}, "subnormal_policy=flush"),
    ({"subnormal_policy": "meep_x86_flush"}, "subnormal_policy=flush"),
    ({"policy": "flush"}, "policy=flush"),
    ({"verdict": {"subnormal_policy": "meep_x86_flush"}}, "verdict.subnormal_policy=flush"),
    ({"subnormal": {"policy": "flush"}}, "subnormal.policy=flush"),
], ids=["top-string", "top-stamp-name", "policy-string", "verdict", "subnormal-policy"])
def test_a_policy_named_as_a_string_is_compared_with_the_answer(extra, named):
    fresh = artifact(subnormal_policy_installed="keep", **extra)
    with pytest.raises(seed.PolicyStampConflict) as caught:
        seed.policy_line(fresh)
    assert named in str(caught.value), str(caught.value)


def test_a_policy_named_as_a_string_never_answers():
    for extra in ({"subnormal_policy": "keep"}, {"policy": "keep"},
                  {"verdict": {"subnormal_policy": "ieee_keep_ftz_stripped"}},
                  {"subnormal": {"policy": "keep"}}):
        assert seed.policy_line(artifact(**extra)) is None, extra


def test_names_that_agree_bind_and_a_string_counts_as_stated():
    fresh = artifact(subnormal_policy_installed="keep",
                     verdict={"subnormal_policy": "ieee_keep_ftz_stripped"},
                     subnormal={"policy": "keep"})
    assert seed.policy_line(fresh).startswith("keep (the gate's own ")
    with pytest.raises(seed.PolicyStampConflict, match="_mixed_policy"):
        seed.policy_line(artifact(verdict={"subnormal_policy": "keep"}), MIXED_ENTRY)


def test_a_container_in_a_policy_slot_is_not_a_name():
    """A dict where a name is usually written neither crashes the reader nor answers."""
    fresh = artifact(subnormal_policy=dict(KEEP_STAMP, requested={"argument": "keep"}),
                     subnormal_policy_requested={"argument": "flush"},
                     verdict={"subnormal_policy": {"keep": True}})
    assert seed.policy_line(fresh).startswith("keep (")
    with pytest.raises(seed.PolicyStampConflict, match="names no attained policy"):
        seed.policy_line(artifact(subnormal_policy_installed={"name": "keep"}))


# --- disagreement is refused by name ---------------------------------------------------


@pytest.mark.parametrize("extra,named", [
    ({"subnormal_policy_installed": "flush"}, "subnormal_policy_installed=flush"),
    ({"subnormal_policy_stamp": dict(FLUSH_STAMP)}, "subnormal_policy_stamp=flush"),
    ({"summary": {"subnormal_policy": dict(FLUSH_STAMP)}},
     "summary.subnormal_policy=flush"),
    ({"subnormal_policy_requested": "flush"}, "subnormal_policy_requested=flush"),
    ({"policy": {"num_warps": 1, "subnormal_policy": "flush"}},
     "policy.subnormal_policy=flush"),
    ({"summary": {"certified_under_subnormal_policy": "meep_x86_flush"}},
     "summary.certified_under_subnormal_policy=flush"),
], ids=["installed", "stamp", "summary", "request", "launch-config", "certified-under"])
def test_spellings_that_disagree_are_refused_naming_both(extra, named):
    fresh = artifact(subnormal_policy=dict(KEEP_STAMP), **extra)
    with pytest.raises(seed.PolicyStampConflict) as caught:
        seed.policy_line(fresh)
    message = str(caught.value)
    assert "disagree" in message and named in message, message
    assert "subnormal_policy=keep" in message, message


def test_a_block_whose_stamp_name_contradicts_its_resolution_is_refused():
    block = dict(KEEP_STAMP, policy="meep_x86_flush")
    with pytest.raises(seed.PolicyStampConflict) as caught:
        seed.policy_line(artifact(subnormal_policy_stamp=block))
    assert "subnormal_policy_stamp.policy=flush" in str(caught.value)
    assert "subnormal_policy_stamp=keep" in str(caught.value)


def test_a_block_whose_explicit_request_contradicts_its_resolution_is_refused():
    """An explicit ``keep`` or ``flush`` resolves to itself, so this cannot be a run."""
    block = dict(KEEP_STAMP, requested="flush")
    with pytest.raises(seed.PolicyStampConflict, match="requested=flush"):
        seed.policy_line(artifact(policy=block))


def test_an_installed_name_that_is_not_a_policy_is_refused():
    with pytest.raises(seed.PolicyStampConflict, match="names no attained policy"):
        seed.policy_line(artifact(subnormal_policy_installed="match_meep"))


def test_the_disagreement_is_refused_before_any_table_order_could_pick_one():
    """The first block is keep and would have been the line; the refusal wins."""
    fresh = artifact(subnormal_policy=dict(KEEP_STAMP),
                     subnormal_policy_at_start=dict(FLUSH_STAMP))
    with pytest.raises(seed.PolicyStampConflict):
        seed.policy_line(fresh)


# --- the declared _mixed_policy exception ---------------------------------------------

MIXED_ENTRY = {
    "_mixed_policy": "NO POLICY INSTALLED. This gate never calls install_ftz_strip.",
    "status": "PASS", "source_sha256": {"meep_gpu/triton_kernels/kernels.py": "1" * 64},
}


def test_a_declared_exception_with_nothing_stated_yields_the_records_exact_string():
    """fused_electric's shape: the gate installs nothing and its artifact says nothing."""
    assert seed.policy_line(artifact(), MIXED_ENTRY) == "NOT INSTALLED - see _mixed_policy"
    assert seed.MIXED_POLICY_LINE == "NOT INSTALLED - see _mixed_policy"
    assert not POLICY_WORD.search(seed.MIXED_POLICY_LINE), (
        "the exception must not name a policy: the record withholds that claim")


def test_without_the_declaration_a_silent_artifact_is_still_refused():
    assert seed.policy_line(artifact()) is None
    assert seed.policy_line(artifact(), {"status": "PASS"}) is None
    assert seed.policy_line(artifact(), dict(MIXED_ENTRY, _mixed_policy="  ")) is None


@pytest.mark.parametrize("extra", [
    {"subnormal_policy": dict(KEEP_STAMP)},
    {"subnormal_policy_installed": "keep"},
    {"policy": {"num_warps": 1, "subnormal_policy": "keep"}},
    {"subnormal_policy": {"cache_dir": "run_root/c"}},
], ids=["stamp", "installed", "request", "partial-block"])
def test_a_declared_exception_for_a_run_that_states_a_policy_is_refused(extra):
    """The binder may not edit the declaration, and a record naming ``keep`` beside an
    entry that says it may not claim ``keep`` contradicts itself. Anything the reader
    looks at, even a request or a block that resolves nothing, counts as stated."""
    with pytest.raises(seed.PolicyStampConflict, match="_mixed_policy"):
        seed.policy_line(artifact(**extra), MIXED_ENTRY)


def test_the_exception_does_not_alter_the_entry_it_reads():
    entry = copy.deepcopy(MIXED_ENTRY)
    seed.policy_line(artifact(), entry)
    assert entry == MIXED_ENTRY


def test_the_shipped_records_spell_the_exception_as_the_reader_does():
    """Read-only, against this tree's ledger: every run record of every entry that
    declares the exception carries the reader's string, so a rebind leaves the
    field exactly as the existing record expresses it."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    declaring = {key: entry for key, entry in ledger.items()
                 if isinstance(entry, dict) and entry.get(seed.MIXED_POLICY_FIELD)}
    assert declaring, ("no entry declares _mixed_policy: if the debt was cleared, "
                       "remove the exception from policy_line in the same change")
    spelled = {f"{key}[{capability}]": run.get("subnormal_policy")
               for key, entry in declaring.items()
               for capability, run in (entry.get("runs") or {}).items()}
    assert spelled and set(spelled.values()) == {seed.MIXED_POLICY_LINE}, spelled
