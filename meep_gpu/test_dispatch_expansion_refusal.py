"""A refused expansion probe artifact may not license an arm at ANY later rung.

THE DEFECT THESE PIN, in one sentence: dispatch refused a flush-cut expansion
probe artifact against the 'keep' policy it requires, and then dispatched anyway,
because the refusal was spelled ``None`` — the same value as "no artifact was
offered" — so the very next rung read the environment variable the refused
artifact came from and handed it to every complex arm.

Measured on a CUDA host with an attributing control, four legs. Each leg names
the environment variable's target and what the run must do:

    leg        artifact                       must
    keepkeep   cut under 'keep'               DISPATCH  <- proves this is not a
                                                           blanket refusal
    xferflush  cut under 'flush'              FALL BACK <- the defect
    noprobe    variable unset                 REFUSE BY NAME (absence, not policy)
    mutant     'flush', backend cupy->numpy   FALL BACK <- the attributing control

The mutant is a ONE-FIELD change to the xferflush artifact. It is dropped by the
SAME clause for the SAME reason, and pre-fix its verdict FLIPPED — which nothing
but a second read of the file can explain. Keeping it green is what keeps the
attribution.

WHAT IS DRIVEN. ``plan_fast_path`` end to end with the REAL composer: the
artifact is read by the shipped reader, judged by the shipped rung (4c), carried
by the shipped composition window and resolved by the shipped ``plan_step``. The
only double is the observation point — ``launch.complex_pml_curl_coverage``, the
seam the composer hands ``resolved_probe`` to — which records what the arm was
offered and what that offer licenses AT THAT MOMENT, then refuses so nothing
downstream runs. No CuPy, no Triton, no device.
"""

from __future__ import annotations

import json

import pytest

import meep_gpu.fastpath as fastpath
from meep_gpu.expansion_refusal import RefusedExpansionProbe
from meep_gpu.triton_kernels import complex_fields as complex_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.coverage import Coverage
from meep_gpu.test_dispatch_contract import cupy_like_grid, stub_triton


@pytest.fixture(autouse=True)
def _a_dispatch_that_reaches_the_composer(monkeypatch):
    """The ladder's first rungs put where every leg here needs them.

    Dispatch is opt-in and the subnormal policy is process state its own gate
    installs, so a leg that inherited either from the shell would report the
    shell rather than the artifact. Rungs 0-4b are not what these tests are
    about; rung (4c) and the composer below it are.
    """
    from meep_gpu import subnormal_policy

    fastpath.reset_dispatch_announcements()
    subnormal_policy._reset_for_tests()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
    monkeypatch.delenv(subnormal_policy.POLICY_ENV, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    yield
    subnormal_policy._reset_for_tests()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    fastpath.reset_dispatch_announcements()

#: The four probe patterns the base complex families score.
_PATTERNS = ("c8_mul_c8", "c8_mul_f4_field_left",
             "f4_mul_c8_coefficient_left", "python_float_left")


def _artifact(tmp_path, resolved, *, backend="cupy", name="probe.json"):
    """A well-formed, fully discriminating record cut under ``resolved``.

    Everything except the policy stamp is what a passing artifact looks like, so
    a refusal here can only be about the policy — or, for the mutant, about the
    one field the mutant changes.
    """
    record = {
        "backend": backend,
        "patterns": {p: "FMA_V1" for p in _PATTERNS},
        "vectors": {p: 2792 for p in _PATTERNS},
        "detail": {p: {"licensable_arms_disagreement_words": 128,
                       "discriminates": True,
                       "matches": {"FMA_V1": True, "NAIVE": False,
                                   "PLANEWISE_diagnostic": False}}
                   for p in _PATTERNS},
        "subnormal_policy": {"policy": f"stamp_{resolved}", "resolved": resolved},
        "candidates": {"policy": resolved},
        "environment": {"backend": "cupy", "machine": "x86_64",
                        "cupy_version": "13.5.1"},
    }
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


class _ComplexFields:
    """Enough of a fields object to make ``complex_storage_active`` True.

    That gate is what decides whether the composer resolves a probe at all, so a
    stub that fails it would make every leg pass by never asking the question.
    """

    class grid:  # noqa: N801 - a stand-in, not a type the package defines
        shape = (4, 4, 4)
        has_bloch = False

    force_complex_fields = True


def _run_leg(monkeypatch, probe_path):
    """Drive a whole dispatch and report what the complex arm was offered.

    Returns the recorded offers. Each is what ``plan_step`` handed the complex
    curl seam, together with the licence that offer earns AT THE RUNG THAT WOULD
    BIND IT — computed inside the composition window, because that window is
    where the declared policy is in force and where a plan builder would ask.
    """
    if probe_path is None:
        monkeypatch.delenv(fastpath.COMPLEX_PROBE_ENV, raising=False)
    else:
        monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(probe_path))

    offers = []

    def recording_coverage(fields, pml, sub_step, probe=None):
        offers.append({
            "probe": probe,
            "expansion": complex_module._resolve_expansion(probe),
            "reasons": list(complex_module._expansion_reasons(probe)),
        })
        return Coverage(False, ("observed, not admitted",))

    monkeypatch.setattr(launch_module, "complex_pml_curl_coverage",
                        recording_coverage)
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(_ComplexFields(), None, cupy_like_grid())
    assert offers, ("the complex curl seam was never consulted, so this leg "
                    "measured nothing about what it is offered")
    return offers


def _one_offer(offers):
    """The single value every complex arm shares — resolved once, per the design."""
    first = offers[0]["probe"]
    assert all(offer["probe"] is first for offer in offers), (
        "the composer resolved the probe more than once; the whole point of "
        "resolving it once is that one verdict reaches every arm")
    return offers[0]


# ---------------------------------------------------------------------------
# The four legs
# ---------------------------------------------------------------------------

def test_keepkeep_a_matching_artifact_still_reaches_the_arms_and_licenses_one(
        monkeypatch, tmp_path):
    """DISPATCH. A fix that refuses everything is not a fix.

    This is the leg that proves the refusal is a JUDGEMENT and not a switch: the
    artifact is cut under the same policy the run requires, so it must arrive at
    the arm intact and license the constexpr it always licensed.
    """
    path, record = _artifact(tmp_path, "keep")
    offer = _one_offer(_run_leg(monkeypatch, path))

    assert offer["probe"] == record, "the matching artifact did not reach the arm"
    assert offer["reasons"] == [], offer["reasons"]
    assert offer["expansion"] == complex_module.EXPANSIONS["FMA_V1"], (
        "the arm that was licensed before the refusal machinery existed is no "
        "longer licensed: this is a broken dispatch, not a fix")


def test_xferflush_a_refused_artifact_never_reaches_an_arm_as_a_record(
        monkeypatch, tmp_path):
    """THE DEFECT. Refused at rung (4c), and refused at every rung below it.

    Pre-fix the offer here was the flush-cut RECORD, re-read from the same
    environment variable the refusal was about, and it licensed FMA_V1.
    """
    path, record = _artifact(tmp_path, "flush")
    offer = _one_offer(_run_leg(monkeypatch, path))

    assert isinstance(offer["probe"], RefusedExpansionProbe), (
        f"the arm was offered {type(offer['probe']).__name__}, not a refusal")
    assert offer["probe"] != record
    assert offer["expansion"] is None, "the refused artifact licensed an arm"
    assert offer["reasons"], "a refused artifact produced no refusal reason"
    assert any("flush" in reason and "keep" in reason
               for reason in offer["reasons"]), offer["reasons"]

    environment = fastpath.last_dispatch_report()["environment"]
    assert environment["complex_expansion_probe_resolved"] is False
    assert environment["complex_expansion_probe_dropped"]


def test_noprobe_an_absent_artifact_refuses_for_ABSENCE_not_for_policy(
        monkeypatch):
    """THE NULL CONTROL, and it is a control over the reason, not just the verdict.

    Nothing was offered, so the arm may legitimately look for an artifact and
    must refuse BY NAME when it finds none. Reporting this as a policy refusal —
    or reporting a refused artifact as an absent one — would be the original
    conflation told backwards.
    """
    offer = _one_offer(_run_leg(monkeypatch, None))

    assert offer["probe"] is None, (
        "'nothing was offered' must stay distinguishable from 'offered and "
        f"refused'; the arm saw {offer['probe']!r}")
    assert offer["expansion"] is None
    assert offer["reasons"] and "no complex-multiply expansion probe artifact " \
                                "is available" in offer["reasons"][0]
    assert not any("REFUSED by this dispatch" in reason
                   for reason in offer["reasons"]), offer["reasons"]


def test_mutant_the_attributing_control_still_falls_back(monkeypatch, tmp_path):
    """THE ONE-FIELD CONTROL. Same clause, same reason, and pre-fix the verdict
    FLIPPED — which only a second read of the file could explain.

    ``backend`` is changed from 'cupy' to 'numpy' and nothing else. Rung (4c)
    judges the POLICY, so this artifact is dropped for exactly the reason the
    xferflush one is; the backend clause is a second, independent refusal below.
    Both legs must now fall back, and for this leg to go on falling back while
    xferflush dispatches would be the defect back again.
    """
    path, _ = _artifact(tmp_path, "flush", backend="numpy")
    offer = _one_offer(_run_leg(monkeypatch, path))

    assert isinstance(offer["probe"], RefusedExpansionProbe)
    assert offer["expansion"] is None
    assert offer["reasons"]


# ---------------------------------------------------------------------------
# The two routes a refusal has to survive, separately
# ---------------------------------------------------------------------------

def test_the_environment_variable_cannot_hand_back_a_refused_artifact(
        monkeypatch, tmp_path):
    """The route the argument does not cover.

    Passing a refusal binds the rungs the value reaches. Seventeen re-acquisition
    sites read the environment variable instead, and the composer was one of
    them, so the refusal is ALSO recorded against the variable for the length of
    the composition. Read the variable inside that window and the refusal is what
    comes back — not the artifact.
    """
    path, record = _artifact(tmp_path, "flush")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))

    assert complex_module.load_expansion_probe() == record, (
        "outside a refusal the reader must return the artifact unchanged")

    reasons = fastpath._complex_probe_policy_reasons(record)
    assert reasons, "the artifact under test is not one the policy clause refuses"

    with fastpath._composition_window(record, reasons) as offered:
        assert isinstance(offered, RefusedExpansionProbe)
        for reader in (complex_module.load_expansion_probe,
                       launch_module.load_expansion_probe):
            back = reader()
            assert isinstance(back, RefusedExpansionProbe), (
                f"{reader.__module__} handed the refused artifact back")
        # The sibling families bind the reader with ``from ... import`` at import
        # time, so they are a separate question from the module attribute.
        from meep_gpu.triton_kernels import folded_complex, special_kz
        for module in (folded_complex, special_kz):
            assert isinstance(module.load_expansion_probe(),
                              RefusedExpansionProbe), module.__name__

    assert complex_module.load_expansion_probe() == record, (
        "the refusal outlived the dispatch that made it")


def test_a_refusal_passed_explicitly_refuses_at_every_licence_funnel(tmp_path):
    """The route the environment record does not cover.

    A rung handed the refusal directly must refuse too — including the three
    extended-pattern funnels, which reach the same arbiter by different names.
    """
    refusal = RefusedExpansionProbe(complex_module.PROBE_PATH_ENVIRONMENT,
                                    ["cut under 'flush', run requires 'keep'"])
    from meep_gpu.triton_kernels import folded_complex, special_kz

    for licence in (complex_module.expansion_license,
                    special_kz.beta_expansion_license,
                    folded_complex.parity_expansion_license,
                    folded_complex.folded_beta_expansion_license):
        verdict = licence(refusal)
        assert verdict["expansion"] is None, licence.__name__
        assert verdict["arm"] is None and verdict["basis"] is None, licence.__name__
        assert any("REFUSED by this dispatch" in reason
                   for reason in verdict["refusals"]), licence.__name__
        assert verdict["refused_by_policy"], licence.__name__

    for reasons_of in (complex_module._expansion_reasons,
                       special_kz._beta_expansion_reasons,
                       folded_complex._parity_expansion_reasons,
                       folded_complex._folded_beta_expansion_reasons):
        found = reasons_of(refusal)
        assert found and "REFUSED by this dispatch" in found[0], reasons_of.__name__


# ---------------------------------------------------------------------------
# ENVIRONMENT_DEFAULTS — the second, independent route into an arm
# ---------------------------------------------------------------------------

def _all_ambiguous_flush(tmp_path):
    """A record that reaches clause 6 — every pattern blind, with the evidence.

    Cut under 'flush'. Nothing here discriminates, so ``expansion_license``
    consults ENVIRONMENT_DEFAULTS, whose one row licenses FMA_V1 for exactly this
    claimed environment. That row is matched against the RECORD's own environment
    block and never against the live machine, so it is reachable from any host.
    """
    ambiguous = complex_module.AMBIGUOUS_BOTH
    record = {
        "backend": "cupy",
        "patterns": {p: ambiguous for p in _PATTERNS},
        "vectors": {p: 2792 for p in _PATTERNS},
        "detail": {p: {"licensable_arms_disagreement_words": 0,
                       "discriminates": False,
                       "matches": {"FMA_V1": True, "NAIVE": True,
                                   "PLANEWISE_diagnostic": False,
                                   "FMA_V2_diagnostic": False},
                       "mismatch_words": {"FMA_V1": 0, "NAIVE": 0,
                                          "PLANEWISE_diagnostic": 71,
                                          "FMA_V2_diagnostic": 71}}
                   for p in _PATTERNS},
        "subnormal_policy": {"policy": "meep_x86_flush", "resolved": "flush"},
        "candidates": {"policy": "flush"},
        "environment": {"backend": "cupy", "machine": "x86_64",
                        "cupy_version": "13.5.1"},
    }
    path = tmp_path / "ambiguous_flush.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


def test_the_environment_default_table_is_reachable_at_all(tmp_path):
    """The premise of the next test, asserted rather than assumed.

    If this record stopped reaching clause 6 the refusal below would pass for the
    wrong reason and pin nothing.
    """
    _, record = _all_ambiguous_flush(tmp_path)
    verdict = complex_module.expansion_license(record)
    assert verdict["basis"] == "environment_default", verdict["refusals"]
    assert verdict["expansion"] == complex_module.EXPANSIONS["FMA_V1"]


def test_a_refused_run_cannot_reach_the_environment_default_table(
        monkeypatch, tmp_path):
    """A SECOND, INDEPENDENT DEFEAT OF THE SAME CLAUSE, closed.

    Clause 6 licenses an arm from a table row rather than from the record's own
    measurements, and it is INSIDE ``expansion_license`` — so a record that
    discriminates nothing takes an arm without any pattern having named one.
    Closing only the re-read idiom would leave this open: the record binds when
    passed EXPLICITLY too. The refusal has to be honoured by the arbiter itself,
    ahead of every clause including this one.
    """
    path, record = _all_ambiguous_flush(tmp_path)
    offer = _one_offer(_run_leg(monkeypatch, path))

    assert isinstance(offer["probe"], RefusedExpansionProbe)
    assert offer["expansion"] is None, (
        "a policy-refused run took an arm from ENVIRONMENT_DEFAULTS")

    verdict = complex_module.expansion_license(offer["probe"])
    assert verdict["basis"] is None and verdict["environment_default"] is None


# ---------------------------------------------------------------------------
# The abstention question: a clause that cannot judge REFUSES
# ---------------------------------------------------------------------------

def test_a_policy_that_can_be_neither_read_nor_declared_refuses(tmp_path):
    """FAIL CLOSED, and the three-valued contract that makes it affordable.

    ``None`` means the caller is NOT ASKING — artifact-reading tooling reads a
    record as a document and binds nothing, so there is no run whose policy could
    disagree. ``UNREADABLE_POLICY`` means the caller ASKED AND GOT NO ANSWER, and
    that refuses: consuming the licence would assert that it TRANSFERS, which is
    exactly what POLICY_CONDITIONAL_LICENCE denies.
    """
    _, record = _artifact(tmp_path, "keep")

    assert complex_module.expansion_policy_reasons(record, None) == []
    assert complex_module.expansion_policy_reasons(record, "keep") == []
    refused = complex_module.expansion_policy_reasons(
        record, complex_module.UNREADABLE_POLICY)
    assert refused and "could not be read and was not declared" in refused[0]


def test_the_declared_policy_is_what_makes_an_honest_run_judgeable(tmp_path):
    """Fail-closed refuses every dispatch unless dispatch can say what it will do.

    ``policy_is_installed()`` is False at every rung that binds an arm — the
    install is four rungs below — so without the declaration this clause would
    have nothing to judge on the shipped path and would refuse the keepkeep leg
    too. The declaration is a fact dispatch already owns, never a guess.
    """
    from meep_gpu import subnormal_policy
    from meep_gpu.expansion_refusal import declaring_run_policy

    assert not subnormal_policy.policy_is_installed()
    assert complex_module._policy_in_force() is complex_module.UNREADABLE_POLICY

    _, keep_cut = _artifact(tmp_path, "keep")
    assert complex_module._expansion_reasons(keep_cut), (
        "with nothing installed and nothing declared this must fail closed")

    with declaring_run_policy("keep"):
        assert complex_module._policy_in_force() == "keep"
        assert complex_module._expansion_reasons(keep_cut) == []
    assert complex_module._policy_in_force() is complex_module.UNREADABLE_POLICY

    with declaring_run_policy("flush"):
        assert complex_module._expansion_reasons(keep_cut), (
            "a keep-cut licence was consumed by a run declaring flush")


@pytest.mark.parametrize("reasons_of,builds", [
    ("_expansion_reasons", "the base four patterns"),
    ("_parity_expansion_reasons", "K2's mirror-ghost fill"),
    ("_beta_expansion_reasons", "K3a/K3b's beta curl"),
])
def test_every_extended_pattern_seam_applies_the_policy_clause(
        tmp_path, reasons_of, builds):
    """The third defeat, and it was independent of the other two.

    ``complex_fields._expansion_reasons`` ended in the policy clause;
    ``folded_complex._parity_expansion_reasons`` and
    ``special_kz._beta_expansion_reasons`` did not call it at all. Measured on the
    same flush-stamped record with the policy readable as 'keep': base 1 reason,
    the other two zero — so K2 and K3a/K3b licensed what the base family refused,
    with no re-read and no environment default involved. Extending the PATTERN
    SET never shrinks the question asked about the artifact.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy
    from meep_gpu.triton_kernels import folded_complex, special_kz

    seams = {
        "_expansion_reasons": complex_module._expansion_reasons,
        "_parity_expansion_reasons": folded_complex._parity_expansion_reasons,
        "_beta_expansion_reasons": special_kz._beta_expansion_reasons,
    }
    seam = seams[reasons_of]

    flush_cut = _extended_pattern_record(tmp_path, "flush")
    keep_cut = _extended_pattern_record(tmp_path, "keep")
    with declaring_run_policy("keep"):
        refused = seam(flush_cut)
        assert refused and any("does not transfer" in reason
                               for reason in refused), (f"{builds}: {refused}")
        assert seam(keep_cut) == [], (
            f"{builds} refuses a record cut under the policy it declares")


def _extended_pattern_record(tmp_path, resolved):
    """A record carrying the base four patterns AND both extended ones.

    K2 and K3a/K3b each add one operand orientation; a record missing either
    refuses for THAT reason and would hide whether the policy clause fired.
    """
    from meep_gpu.triton_kernels import folded_complex, special_kz

    names = set(_PATTERNS)
    names.update(special_kz.BETA_PROBE_PATTERNS)
    names.update(folded_complex.PARITY_PROBE_PATTERNS)
    return {
        "backend": "cupy",
        "patterns": {p: "FMA_V1" for p in names},
        "vectors": {p: 2792 for p in names},
        "detail": {p: {"licensable_arms_disagreement_words": 128,
                       "discriminates": True,
                       "matches": {"FMA_V1": True, "NAIVE": False,
                                   "PLANEWISE_diagnostic": False}}
                   for p in names},
        "subnormal_policy": {"policy": f"stamp_{resolved}", "resolved": resolved},
        "candidates": {"policy": resolved},
        "environment": {"backend": "cupy", "machine": "x86_64",
                        "cupy_version": "13.5.1"},
    }


# ---------------------------------------------------------------------------
# The two spellings of the variable name must stay one variable
# ---------------------------------------------------------------------------

def test_the_refusal_is_recorded_against_the_variable_the_reader_consults():
    """``fastpath`` names the variable for its audit block and the family module
    names it for its reader. A refusal recorded against one and consulted through
    the other would be a refusal that silently does nothing."""
    assert fastpath.COMPLEX_PROBE_ENV == complex_module.PROBE_PATH_ENVIRONMENT


# ---------------------------------------------------------------------------
# A DECLARATION MAY NOT OVERRULE ANOTHER DECLARATION
# ---------------------------------------------------------------------------
#
# The declaration exists because the dispatch path cannot READ a policy at the
# rungs that bind arms. That makes it the softest thing in the licence chain,
# and until 2026-08-16 it was a single unkeyed slot an inner caller could
# overwrite. MEASURED on the five seams that gate a plan builder, with a
# flush-cut record and a dispatch that had declared 'keep':
#
#   route                                            pre-fix         post-fix
#   nested declare('flush'), record passed           BOUND FMA_V1    refused
#   nested declare('flush'), record via the reader   BOUND FMA_V1    refused
#   the same inside a standing refusal, raw record   BOUND FMA_V1    refused
#   declare('flush') alone, flush record             bound           bound
#
# The last row is the point of the fix's shape. A backend whose device cannot
# keep float32 subnormals HAS to be able to declare 'flush' and consume a
# flush-cut artifact; forbidding the name would refuse the truth. What is
# refused is the CONTRADICTION — two open declarations about a fact this process
# has one of — because that, and not the name, is what laundering looks like.

_GATING_SEAMS = ("complex_fields._expansion_reasons",
                 "folded_complex._parity_expansion_reasons",
                 "folded_complex._base_expansion_reasons",
                 "folded_complex._folded_beta_expansion_reasons",
                 "special_kz._beta_expansion_reasons")


def _seam(name):
    """One of the five predicates a plan builder is gated on, by name."""
    from meep_gpu.triton_kernels import folded_complex, special_kz

    module, attribute = name.split(".")
    home = {"complex_fields": complex_module,
            "folded_complex": folded_complex,
            "special_kz": special_kz}[module]
    return getattr(home, attribute)


@pytest.mark.parametrize("seam_name", _GATING_SEAMS)
def test_a_nested_declaration_cannot_launder_a_record_the_outer_one_refuses(
        tmp_path, seam_name):
    """The laundering move, at every seam that gates a plan builder.

    Declare the policy the artifact was cut under, from inside the dispatch that
    requires the other one. Pre-fix the inner declaration simply replaced the
    outer one and all five seams went silent on the record ``fastpath``'s own
    policy clause had refused.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    flush_cut = _extended_pattern_record(tmp_path, "flush")
    seam = _seam(seam_name)

    with declaring_run_policy("keep"):
        assert seam(flush_cut), "control: the outer declaration must refuse it"
        with declaring_run_policy("flush"):
            reasons = seam(flush_cut)
            assert reasons, (
                f"{seam_name} licensed a flush-cut record because a nested "
                f"declaration said 'flush' inside a dispatch that said 'keep'")
            assert any("disagree" in reason for reason in reasons), reasons


def test_a_nested_declaration_cannot_bind_the_constexpr_through_the_reader(
        tmp_path, monkeypatch):
    """The same route with ``probe=None``, which re-reads the variable.

    Two rungs had to be defeated for the pre-fix measurement, and this is the
    second: passing nothing makes ``_resolve_expansion`` open the artifact
    itself, so a fix that only guarded the explicitly-passed record would leave
    the environment route open.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    path, _ = _artifact(tmp_path, "flush")
    monkeypatch.setenv(complex_module.PROBE_PATH_ENVIRONMENT, str(path))

    with declaring_run_policy("keep"):
        assert complex_module._resolve_expansion() is None
        with declaring_run_policy("flush"):
            assert complex_module._resolve_expansion() is None, (
                "a nested declaration bound the EXPANSION constexpr off the "
                "environment variable the outer declaration refuses")


def test_re_declaring_the_same_policy_stays_legal(tmp_path):
    """Nesting is not the defect; DISAGREEING is. This nesting exists in-tree.

    ``conftest``'s ``run_policy_declared_keep`` fixture wraps
    ``fastpath._composition_window`` and both declare 'keep'. A fix that refused
    all nesting would refuse the shipped dispatch path under the test harness.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    keep_cut = _extended_pattern_record(tmp_path, "keep")
    with declaring_run_policy("keep"):
        with declaring_run_policy("keep"):
            assert complex_module._policy_in_force() == "keep"
            for seam_name in _GATING_SEAMS:
                assert _seam(seam_name)(keep_cut) == [], seam_name
        assert complex_module._policy_in_force() == "keep", (
            "the inner window's exit retired the outer declaration")


def test_a_backend_that_cannot_keep_subnormals_may_still_declare_flush(tmp_path):
    """The legitimate case the fix must not break, pinned at the RIGHT seam.

    A device that flushes float32 subnormals has no 'keep' to offer, so its
    dispatch declares 'flush' and consumes a flush-cut artifact. Declaring it
    stays legal and the artifact-vs-run question still answers YES.

    WHAT THIS TEST USED TO ASSERT, AND WHY IT WAS WRONG (2026-08-16). It looped
    over ``_GATING_SEAMS`` — the five TRITON predicates — and required them to
    accept the flush-cut record. That is not "flush is usable by the backend that
    needs it", it is "flush is usable by a backend certified only under keep",
    and it was the defect written down as a requirement. ``_DECLARATIONS`` is
    process-wide and unkeyed by design, so the flush a Metal dispatch legitimately
    declares is the flush these Triton seams read; with only the artifact-vs-run
    comparison in place that declaration licensed arms whose gates never ran under
    flush. The two questions are separate and both must hold — see
    ``complex_fields.expansion_certification_reasons``.

    So the legitimate case is pinned where it actually lives: the DECLARATION is
    legal, the POLICY IN FORCE is what was declared, and the ARTIFACT agrees with
    the run. What the Triton seams then do with it is a statement about Triton's
    certification, and it is the next test.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    flush_cut = _extended_pattern_record(tmp_path, "flush")
    with declaring_run_policy("flush"):
        assert complex_module._policy_in_force() == "flush", (
            "declaring flush must stay legal; the fix cannot be 'forbid flush'")
        assert complex_module.expansion_policy_reasons(flush_cut, "flush") == [], (
            "the artifact-vs-run question must still answer YES for a flush-cut "
            "record under a flush run; refusing here would be refusing the truth")


def test_a_foreign_flush_declaration_cannot_bind_a_keep_certified_triton_arm(
        tmp_path):
    """Defect 2. The other half of the test above, and the one nothing pinned.

    Metal MUST declare 'flush' — the MPS executor flushes float32 subnormals
    natively and exposes no lever — and declarations are process-wide and
    deliberately not segregated by backend, because a subnormal policy is one
    process-wide fact. Every Triton complex family, meanwhile, was certified
    under 'keep' and only 'keep'.

    Pre-fix, MEASURED 2026-08-16: a top-level ``declaring_run_policy('flush')``
    with a flush-cut artifact scored ZERO reasons at all five gating seams and
    ``_resolve_expansion`` bound FMA_V1 — arriving explicitly or through the
    reader off the environment variable, both. The funnel every EXPANSION
    constexpr passes through was a choke point for artifact-vs-run agreement
    ONLY; nothing anywhere asked whether the KERNEL had ever been validated for
    the arithmetic in force.

    The mirror direction (a keep-cut record under a declared flush) was already
    refused, and is pinned above by
    ``test_the_declared_policy_is_what_makes_an_honest_run_judgeable``. This
    direction had no pin at all.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    flush_cut = _extended_pattern_record(tmp_path, "flush")
    with declaring_run_policy("flush"):
        # The BEHAVIOUR first, and asserted without naming anything the fix
        # introduced: pre-fix this must fail because the seam LICENSED the
        # record, not because a constant is missing.
        licensed = [name for name in _GATING_SEAMS if not _seam(name)(flush_cut)]
        assert not licensed, (
            f"{licensed} licensed a flush-cut record under a flush declared by "
            f"some other backend; every complex arm here was certified under "
            f"'keep' only, so no gate ever ran these kernels in this arithmetic")
        assert complex_module._resolve_expansion(flush_cut) is None, (
            "the constexpr bound under a policy no gate ran these kernels in")
        # And the refusal must be the CERTIFICATION one, not the artifact one
        # arriving by luck — the artifact and the run agree here.
        for seam_name in _GATING_SEAMS:
            assert any("CERTIFIED under" in reason
                       for reason in _seam(seam_name)(flush_cut)), (
                f"{seam_name} refused, but not for the certification reason")


@pytest.mark.parametrize("policy,label", [
    (complex_module.UNREADABLE_POLICY, "a policy that could not be read"),
    ("flush", "a policy these arms were not certified under"),
    ("banana", "a name that is not a policy at all"),
    (object(), "a value that is not a policy name"),
])
def test_the_certification_clause_refuses_on_its_own(policy, label):
    """``expansion_certification_reasons`` is SELF-SUFFICIENT, pinned directly.

    It is always called beside ``expansion_policy_reasons``, and that sibling
    already refuses an unreadable or contradicted policy — so a mutation that
    makes this clause abstain on anything non-``str`` changes NOTHING about the
    combined seam and no test of ``_expansion_reasons`` can see it. Measured: that
    exact mutation SURVIVED the battery while every other certification mutation
    was caught.

    A clause whose documented property is only true by its neighbour's accident
    is a clause that will be wrong the first time it is called alone. This asks
    it alone. ``None`` is the one abstention, and it is the next test.
    """
    from meep_gpu.expansion_refusal import ContradictedRunPolicy

    reasons = complex_module.expansion_certification_reasons(policy)
    assert reasons, f"the certification clause abstained on {label}"
    assert "CERTIFIED under" in reasons[0]

    contradicted = ContradictedRunPolicy(("keep", "flush"))
    assert complex_module.expansion_certification_reasons(contradicted), (
        "the certification clause abstained on a contradiction between two open "
        "declarations")


def test_the_certification_clause_abstains_only_for_a_caller_not_asking():
    """``None`` means "I am not asking" and is the ONLY abstention.

    Artifact-reading tooling — a gate script, a licence audit block — reads a
    record as a document, binds no arm and steps no field, so there is no run
    whose certification could be in question. That is the same four-valued
    contract ``expansion_policy_reasons`` keeps, and keeping the two clauses
    parallel is what lets one call site pass one policy to both.
    """
    assert complex_module.expansion_certification_reasons(None) == []
    assert complex_module.expansion_certification_reasons(
        complex_module.CERTIFIED_UNDER_SUBNORMAL_POLICY) == []


def test_the_certification_policy_and_what_dispatch_installs_agree(tmp_path):
    """The weld. Two constants, one fact, and a test rather than an argument.

    ``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`` is what the gates ran
    these kernels under; ``fastpath.CERTIFICATION_SUBNORMAL_POLICY`` is what
    dispatch installs and refuses to run without. They are the same value BECAUSE
    dispatch installs what the arms were certified under — and they are two
    constants because the library may not import the dispatcher (a caller that
    reaches ``plan_step`` without going through ``fastpath`` must still get the
    certification check). If they ever diverge, either dispatch is installing an
    arithmetic these kernels were never certified in, or the library is refusing
    every honest dispatch.
    """
    assert (complex_module.CERTIFIED_UNDER_SUBNORMAL_POLICY
            == fastpath.CERTIFICATION_SUBNORMAL_POLICY)

    # And the whole point of the weld: under the policy dispatch installs, an
    # artifact cut under it is licensed. A weld that only compared two strings
    # would pass with both set to a policy nothing was certified under.
    keep_cut = _extended_pattern_record(
        tmp_path, fastpath.CERTIFICATION_SUBNORMAL_POLICY)
    from meep_gpu.expansion_refusal import declaring_run_policy
    with declaring_run_policy(fastpath.CERTIFICATION_SUBNORMAL_POLICY):
        for seam_name in _GATING_SEAMS:
            assert _seam(seam_name)(keep_cut) == [], seam_name


def test_declaring_nothing_does_not_retract_the_dispatchs_declaration():
    """``declaring_run_policy(None)`` says nothing; it does not unsay.

    Under the single-slot implementation an inner ``None`` blanked the
    dispatch's declaration for the whole of its body, which made the policy
    unreadable at every rung inside it.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    with declaring_run_policy("keep"):
        with declaring_run_policy(None):
            assert complex_module._policy_in_force() == "keep"


def test_a_contradiction_does_not_outlive_the_window_that_made_it():
    """Including when the body raises: a leaked contradiction would refuse every
    later dispatch in the process, which is a denial of service wearing a
    fail-closed hat."""
    from meep_gpu.expansion_refusal import declaring_run_policy

    with declaring_run_policy("keep"):
        try:
            with declaring_run_policy("flush"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert complex_module._policy_in_force() == "keep"
    assert complex_module._policy_in_force() is complex_module.UNREADABLE_POLICY


def test_an_installed_policy_outranks_a_declaration_that_disagrees_with_it(
        tmp_path):
    """The ordering ``_policy_in_force`` argues at length and nothing pinned.

    An install is a MEASUREMENT of this process's arithmetic; a declaration is
    an intent. Inverting the two lets a declaration answer a licence question
    the machine has already answered, which is how a process that installed
    'flush' keeps consuming a 'keep'-cut licence because some rung above it said
    'keep'. Both directions are driven, because only checking the direction that
    happens to refuse would pass with the sources swapped.
    """
    from meep_gpu import subnormal_policy
    from meep_gpu.expansion_refusal import declaring_run_policy

    keep_cut = _extended_pattern_record(tmp_path, "keep")
    flush_cut = _extended_pattern_record(tmp_path, "flush")

    subnormal_policy.install_subnormal_policy("keep")
    assert subnormal_policy.policy_is_installed()
    with declaring_run_policy("flush"):
        assert complex_module._policy_in_force() == "keep", (
            "the declaration outranked the installed policy")
        assert complex_module._expansion_reasons(keep_cut) == [], (
            "the INSTALLED keep must license the keep-cut record")
        assert complex_module._expansion_reasons(flush_cut), (
            "the declared flush licensed a flush-cut record while keep was "
            "installed")


def _open_declarations():
    """The policy of every declaration currently open, outermost first.

    Reads the container without assuming its SHAPE, so these tests report the
    behaviour they are about — which entries are open — rather than failing on
    the type when the container changes. It has already changed once (a list of
    names became a token-keyed mapping when removal by position turned out to
    leak), and a test that fails with ``'list' object has no attribute 'values'``
    says nothing about whether a declaration outlived its window.
    """
    from meep_gpu import expansion_refusal

    entries = expansion_refusal._DECLARATIONS
    return list(entries.values()) if hasattr(entries, "values") else list(entries)


def test_an_abandoned_generator_does_not_leave_its_declaration_standing():
    """Defect 1. A declaration that OUTLIVES EVERY WINDOW, by an ordinary shape.

    ``declaring_run_policy`` used to remove its entry by the POSITION that entry
    took::

        depth = len(_DECLARATIONS)          # at entry
        if len(_DECLARATIONS) >= depth:     # at exit
            del _DECLARATIONS[depth - 1]

    Positions SHIFT when an earlier window closes, so a window whose entry had
    moved down no longer indexed itself, the guard declined to fire, and the
    entry stood for the LIFE OF THE PROCESS.

    A generator holding a declaration across a yield is the ordinary shape that
    reaches it — a ``for`` loop that breaks, a ``next()`` never followed up — and
    Python finalizes it when it is collected, which is after the outer window has
    gone. MEASURED pre-fix: ``_DECLARATIONS == ['flush']`` with nothing open,
    ``_policy_in_force()`` answering 'flush', all five gating seams silent on a
    flush-cut record and ``_resolve_expansion`` binding FMA_V1.

    That is the round-1 laundering hole restored by the cleanup meant to protect
    it: the 'keep' that made the pair a CONTRADICTION is erased on the way out,
    leaving the foreign 'flush' standing alone and uncontradicted.
    """
    import gc

    from meep_gpu.expansion_refusal import declaring_run_policy

    assert not _open_declarations(), "a leak arrived from elsewhere"

    def declaring_generator(policy):
        with declaring_run_policy(policy):
            yield "first"
            yield "second"

    outer = declaring_run_policy("keep")
    outer.__enter__()
    generator = declaring_generator("flush")
    next(generator)
    assert complex_module._policy_in_force().__class__.__name__ == (
        "ContradictedRunPolicy"), "both windows must be open and disagreeing"

    outer.__exit__(None, None, None)      # the OUTER one closes FIRST
    del generator
    gc.collect()                          # the generator's window closes here

    assert _open_declarations() == [], (
        "a declaration outlived every window that wrote it")
    assert complex_module._policy_in_force() is complex_module.UNREADABLE_POLICY


@pytest.mark.parametrize("close_first,label", [
    (0, "the OUTERMOST window closes first"),
    (1, "the MIDDLE window closes first"),
])
def test_a_window_finalized_out_of_order_retires_its_own_entry(close_first,
                                                               label):
    """The OTHER half, and the reason the fix could not simply be to pop the end.

    The position guard was defending something real: a generator finalized out of
    order pops somebody else's declaration and leaves its own behind, which is the
    same leak arriving from the other side. Both properties are required and this
    pins the second — each window retires ITS OWN entry, whatever order they close
    in and whatever moved underneath them.

    Driven with three windows so the entry that moves is distinguishable from the
    entry that was retired: with two, "retired the right one" and "retired one"
    are the same observation.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy

    # Pre-fix this file's earlier cases LEAK, so clear rather than assert — the
    # entries left standing are another test's finding, and inheriting them here
    # would make this test fail for somebody else's reason.
    from meep_gpu import expansion_refusal
    expansion_refusal._DECLARATIONS.clear()

    windows = [declaring_run_policy(policy)
               for policy in ("keep", "keep", "flush")]
    for window in windows:
        window.__enter__()
    assert _open_declarations() == ["keep", "keep", "flush"]

    victim = windows.pop(close_first)
    victim.__exit__(None, None, None)
    remaining = ["keep", "keep", "flush"]
    remaining.pop(close_first)
    assert _open_declarations() == remaining, (
        f"{label}: it retired somebody else's declaration")

    for window in reversed(windows):
        window.__exit__(None, None, None)
    assert _open_declarations() == [], (
        f"{label}: an entry was left behind with no window open")


def test_a_declaration_must_be_a_policy_name_and_not_merely_spell_one():
    """``str(policy)`` is a conversion, not a check.

    An object whose ``__str__`` returns 'keep' declared 'keep' and bound an arm;
    ``True`` became the string 'True' and refused. Neither is a policy, and the
    first is the shape that matters — a proxy that SPELLS a real name. The type
    check needs nothing imported; the membership check reads
    ``subnormal_policy.POLICIES`` so the list of names has one home.
    """
    from meep_gpu import subnormal_policy
    from meep_gpu.expansion_refusal import declaring_run_policy

    class SpellsKeep:
        def __str__(self):
            return "keep"

    for value in (SpellsKeep(), True, 1, ("keep",)):
        with pytest.raises(ValueError):
            with declaring_run_policy(value):
                pass  # pragma: no cover - the raise is the assertion

    with pytest.raises(ValueError):
        with declaring_run_policy("banana"):
            pass  # pragma: no cover - the raise is the assertion

    # Every real policy name stays declarable — the check is a membership test,
    # not a whitelist of the one policy this package happens to be certified in.
    for policy in subnormal_policy.POLICIES:
        with declaring_run_policy(policy):
            assert complex_module._policy_in_force() == policy


def test_a_non_string_is_refused_even_when_the_policy_NAMES_cannot_be_read(
        monkeypatch):
    """The type check, pinned where it is the ONLY thing doing the work.

    ``declaring_run_policy`` makes two checks and one subsumes the other in every
    configuration but one. A proxy whose ``__str__`` spells 'keep' is not IN
    ``subnormal_policy.POLICIES`` either, so the membership test rejects it and
    the type check appears redundant — MEASURED: deleting the ``isinstance``
    guard left the whole battery green.

    The configuration where it is not redundant is the one ``_known_policies``
    documents as its own fallback: when the sibling module cannot be read it
    returns None, the membership test is skipped, and the type check is all that
    is left. Without it a proxy would be stored AS THE POLICY, and every reader
    downstream would be comparing against an object rather than a name.

    This drives that configuration directly rather than trusting the argument.
    """
    from meep_gpu import expansion_refusal
    from meep_gpu.expansion_refusal import declaring_run_policy

    monkeypatch.setattr(expansion_refusal, "_known_policies", lambda: None)

    class SpellsKeep:
        def __str__(self):
            return "keep"

    with pytest.raises(ValueError):
        with declaring_run_policy(SpellsKeep()):
            pass  # pragma: no cover - the raise is the assertion

    # And the fallback still does what it says: an unrecognised STRING is
    # accepted when the names cannot be read, and fails closed downstream
    # because it matches neither an artifact stamp nor a certification policy.
    with declaring_run_policy("banana"):
        assert complex_module._policy_in_force() == "banana"
        assert complex_module.expansion_certification_reasons("banana")


def test_a_refusal_may_not_be_minted_without_a_reason():
    """A refusal carrying no reason is indistinguishable from an accident.

    Every funnel that honours this value quotes ``reasons`` into the refusal it
    reports, so an empty tuple produces a refusal whose audit line says nothing
    happened — the same conflation between "refused" and "absent" the value was
    built to end, arriving from the other side. All three ways to mint one are
    driven: the class, the minting helper, and the recording window.
    """
    from meep_gpu.expansion_refusal import (RefusedExpansionProbe,
                                            refuse_expansion_probe,
                                            refusing_expansion_probe)

    with pytest.raises(ValueError, match="at least one"):
        RefusedExpansionProbe("MEEP_GPU_ANY_PROBE", [])
    with pytest.raises(ValueError, match="at least one"):
        refuse_expansion_probe("MEEP_GPU_ANY_PROBE", ())
    with pytest.raises(ValueError, match="at least one"):
        with refusing_expansion_probe("MEEP_GPU_ANY_PROBE", iter(())):
            pass

    # And the reasons a well-formed refusal carries reach the funnel's output,
    # which is what makes the emptiness matter rather than being cosmetic.
    refusal = RefusedExpansionProbe("MEEP_GPU_ANY_PROBE", ["cut under flush"])
    reasons = complex_module._expansion_reasons(refusal)
    assert reasons and "cut under flush" in reasons[0]
