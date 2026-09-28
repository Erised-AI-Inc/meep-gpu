"""Is the off-diagonal fold refusal's boundary the boundary that was MEASURED?

WHAT THIS GATE IS, AND WHAT IT IS NOT. It makes NO device measurement and
launches NO kernel. ``gate_cuda_folded_offdiag.py`` did that on 2026-08-20
(the GPU host GPU 7, both float32 subnormal policies, 960 cases per policy) and the
artifact is
``parity/meep_gpu/results/cuda_folded_offdiag_2026-08-21_rename/{keep,flush}/gate.json``
-- REPOINTED 2026-08-22 from the 2026-08-20 run of the same gate, because the
CUDA kernel rename moved every device string this family emits and the older
artifact therefore replays a kernel that no longer ships. Both legs of the
replacement released and its recorded emitter corpus digest equals live
emission.
That run measured WHERE the shipped ``update_E_pml_real_offdiag`` stops
reproducing ``stepping.update_E`` on a mirror-folded grid. It did not answer the
question a predicate has to answer, which is whether the RULE now written into
``coverage.py`` draws its line in the same place — not one row wider, not one row
narrower — and what that line is worth in corpus slots.

So this gate answers four things, and each is scored separately:

  REPLAY      ``coverage.offdiag_fold_roles`` against every guarded case of the
              device artifact, per policy. Cheap, and only as good as the record.
  RE-MEASURE  the same boundary derived AGAIN, here, by running the emitted
              device source against ``stepping.update_E`` on real folded
              ``Grid``/``Fields``/``PML`` triples in float32 NumPy. This leg
              consults no artifact and is what makes the verdict independent of
              the record. The device run and this one build their fixtures
              differently — different cells, courants, plane parities and value
              classes — so agreement is a second draw.
  WORTH       the corpus slot arithmetic, recomputed from the all-families
              census: union before, union after, per-family attribution and
              disjointness. This is the number that decides whether the arm is
              installed, and it is ZERO.
  FALSIFY     a mutation battery on the rule and on the predicate, each leg
              scored CAUGHT or NULL CONFIRMED, and the release verdict itself
              re-run under each planted defect and required to FLIP.

WHY THE ARM IS NOT INSTALLED, which is this gate's finding rather than its
premise: the measured-exact arm admits zero corpus slots (the widened predicate
serves the SAME 16 ``update_E`` slots as the shipped one on all 186 rows), and
``offdiag_constitutive_kernels.offdiag_boundary_codes`` has no code to hand a
folded axis at all — while the obvious stand-in, reading the fold as PERIODIC, is
measured WRONG on 32 of the 480 guarded cases per policy. A clause that gains
nothing and depends on a launcher decision this module does not own can only
leak, so the refusal stays at its present width and the measurement is recorded
instead.

Run (nothing here needs a GPU)::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/gate_cuda_folded_offdiag_rowmask.py \\
        --out parity/meep_gpu/results/cuda_folded_offdiag_rowmask_<date>
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402
import replay_folded_offdiag_slots as replay  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import coverage, offdiag_emitter  # noqa: E402
# The laptop evaluator and its fixture, imported rather than re-written: the
# merge-bar file already executes the emitted device source in float32 NumPy and
# already pins every line of it as executed or transcribed. A second copy here
# would be a second thing to keep in step, and the two records could then
# disagree about what "the emitted source" is.
from meep_gpu.cuda_kernels import (  # noqa: E402
    test_offdiag_constitutive_pml_real as bench)

#: The device artifact this gate replays. Not produced here.
ARTIFACT = os.path.join(_HERE, "results", "cuda_folded_offdiag_2026-08-21_rename")

#: The census the slot arithmetic is recomputed from. 186 rows, 759 slots.
CENSUS = os.path.join(_HERE, "results",
                      "cuda_predicate_coverage_2026-08-20_all_families")

#: The refusal under examination, verbatim from ``coverage.py``.
MIRROR_REFUSAL = ("mirror symmetry: the fold changes the stored extent, and the "
                  "extent is what turns a cell index into a coefficient index")

#: Only the GUARDED leg is scored. ``--fmad=false`` is correctness on this
#: sub-step, not tuning: the unguarded control is 0/240 identical at the inexact
#: courant in the device record, so an unguarded case measures the contraction
#: and not the fold.
GUARD = "fmad_false"

#: Which sub-step each family may serve, for the union recount. Transcribed from
#: the census analyzer's ``UNION_FAMILIES`` so the two cannot count differently.
UNION_FAMILIES: Dict[str, Tuple[Tuple[str, Optional[str]], ...]] = {
    "step_B": (("cuda_curl", "step_B"), ("cuda_complex", "step_B"),
               ("cuda_cylindrical", "step_B")),
    "step_D": (("cuda_curl", "step_D"), ("cuda_complex", "step_D"),
               ("cuda_cylindrical", "step_D")),
    "update_H": (("cuda_constitutive", "update_H"), ("cuda_complex", "update_H"),
                 ("cuda_no_pml", "update_H")),
    "update_E": (("cuda_constitutive", "update_E"), ("cuda_offdiag", None),
                 ("cuda_complex", "update_E"), ("cuda_no_pml", "update_E")),
    "update_P": (("cuda_ade", "update_P"),),
}

#: The mutations. Each is a WRONG statement of the boundary that the reading
#: invites, and each must be caught by the recorded table, by the local
#: measurement, or by the corpus arithmetic — which one catches it is itself
#: reported, because a leg caught only by the artifact is a weaker leg.
#:
#: ``refuse_every_fold_as_the_predicate_does_today`` is the shipped clause's OWN
#: width armed as a rule, and it is the informative one: the boundary legs catch
#: it while the corpus leg does NOT (its slot delta stays 0), which is precisely
#: why this record exists rather than the census alone. ``rename_the_roles`` is
#: expected to change nothing, and a battery with no predicted null cannot show
#: that the scoring distinguishes a catch from a no-op.
RULE_MUTATIONS: Tuple[str, ...] = (
    "drop_the_partner_leg",
    "drop_the_own_axis_leg",
    "ignore_the_code_handed_the_fold",
    "read_the_partner_axis_as_the_own_axis",
    "require_both_roles",
    "refuse_every_fold_as_the_predicate_does_today",
    "admit_every_fold",
    "rename_the_roles",
)

#: Mutations expected to change NOTHING. Scored NULL CONFIRMED, and a null that
#: turns out to be caught is a finding, not a pass.
PREDICTED_NULLS: Tuple[str, ...] = ("rename_the_roles",)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# THE RULE, and the mutations of it
# ---------------------------------------------------------------------------

def rule_says_identical(folded: Sequence[int], mask: Sequence[int],
                        substitution: str, mutation: Optional[str] = None) -> bool:
    """Does the shipped rule predict byte-identity for this configuration?

    UNMUTATED, this is exactly ``coverage.offdiag_fold_roles(...)
    ["changes_a_byte"][substitution]``, negated — the gate does not carry a second
    statement of the boundary, it carries the shipped one. The mutations below
    are applied to the ROLES the shipped function returns, so every leg is armed
    on the real answer rather than on a re-derivation of it.
    """
    roles = coverage.offdiag_fold_roles(
        mask, tuple(axis in tuple(folded) for axis in range(3)))
    partner = roles["fold_is_a_live_partner_axis"]
    own = roles["fold_is_a_live_own_axis"]
    periodic = substitution == "periodic"
    if mutation is None:
        return not roles["changes_a_byte"][substitution]
    if mutation == "drop_the_partner_leg":
        return not (own and periodic)
    if mutation == "drop_the_own_axis_leg":
        return not partner
    if mutation == "ignore_the_code_handed_the_fold":
        return not (partner or own)
    if mutation == "read_the_partner_axis_as_the_own_axis":
        return not (own or (partner and periodic))
    if mutation == "require_both_roles":
        # "The fold has to be reachable through BOTH legs" — a conjunction where
        # the arithmetic has a disjunction.
        return not (partner and own)
    if mutation == "refuse_every_fold_as_the_predicate_does_today":
        # THE SHIPPED CLAUSE'S OWN WIDTH, armed as a rule. It must be caught by
        # the boundary legs (it calls the 96 measured-exact cases divergent) and
        # must NOT be caught by the corpus leg, whose delta stays 0 — which is the
        # whole reason this record exists rather than the census alone.
        return not tuple(folded)
    if mutation == "admit_every_fold":
        return True
    if mutation == "rename_the_roles":
        # THE PREDICTED NULL: the same two booleans read out of the record under
        # their other names. Nothing about the boundary moves.
        return not (roles["folded_partner_axes"]
                    or (roles["folded_own_axes"] and periodic))
    raise AssertionError(f"unknown mutation {mutation!r}")


# ---------------------------------------------------------------------------
# LEG 1 — REPLAY the device artifact
# ---------------------------------------------------------------------------

def load_artifact_cases(policy: str,
                        leg: str = "sweep") -> List[Dict[str, Any]]:
    path = os.path.join(ARTIFACT, policy, "gate.json")
    with open(path) as handle:
        payload = json.load(handle)
    return [case for case in payload[leg]["cases"] if case["guard"] == GUARD]


def score_against_the_rule(cases: Sequence[Dict[str, Any]],
                           mutation: Optional[str]) -> Dict[str, Any]:
    """One recorded leg against the rule: agreement, and the two-sided split."""
    agree = disagree = folded_exact = folded_divergent = unfolded = 0
    offenders: List[str] = []
    for case in cases:
        folded = tuple(axis for axis in range(3) if case["mirrored"][axis])
        substitution = ("metallic" if case["substitution"] == "mirror_as_metallic"
                        else "periodic")
        predicted = rule_says_identical(folded, case["row_mask"], substitution,
                                        mutation)
        measured = bool(case["bit_identical"])
        if predicted == measured:
            agree += 1
        else:
            disagree += 1
            if len(offenders) < 12:
                offenders.append(case["key"])
        if folded:
            folded_exact += measured
            folded_divergent += not measured
        else:
            unfolded += 1
    return {"cases": len(cases), "agree": agree, "disagree": disagree,
            "folded_bit_identical": folded_exact,
            "folded_divergent": folded_divergent,
            "unfolded_controls": unfolded, "first_offenders": offenders}


def artifact_table(cases: Sequence[Dict[str, Any]]) -> Dict[Tuple, Tuple[int, int]]:
    """``(folded axes, row mask, code) -> (ran, bit-identical)``, off the record."""
    tally: Dict[Tuple, List[int]] = collections.defaultdict(lambda: [0, 0])
    for case in cases:
        key = (tuple(axis for axis in range(3) if case["mirrored"][axis]),
               tuple(case["row_mask"]),
               "metallic" if case["substitution"] == "mirror_as_metallic"
               else "periodic")
        tally[key][0] += 1
        tally[key][1] += bool(case["bit_identical"])
    return {key: (value[0], value[1]) for key, value in tally.items()}


def leg_replay(mutation: Optional[str] = None) -> Dict[str, Any]:
    """The rule against every guarded device case, per policy."""
    out: Dict[str, Any] = {"policies": {}, "artifact": ARTIFACT,
                           "artifact_present": os.path.isdir(ARTIFACT)}
    if not out["artifact_present"]:
        out["why_not_measurable"] = f"{ARTIFACT} is absent"
        return out
    tables = {}
    for policy in ("keep", "flush"):
        cases = load_artifact_cases(policy, "sweep")
        # THE 60-LAUNCH LEG IS A DIFFERENT CLAIM AND IS SCORED SEPARATELY: ``f_w``
        # is state, so a tree that gets E right and ``f_w`` wrong is exact for one
        # launch and wrong forever after. A boundary that held only at n=1 would
        # be a boundary about one launch.
        multi = load_artifact_cases(policy, "multistep")
        single_leg = score_against_the_rule(cases, mutation)
        multi_leg = score_against_the_rule(multi, mutation)
        tables[policy] = artifact_table(cases)
        out["policies"][policy] = {
            "one_launch": single_leg, "sixty_launches": multi_leg,
            "guarded_cases": single_leg["cases"],
            "agree": single_leg["agree"], "disagree": single_leg["disagree"],
            "folded_bit_identical": single_leg["folded_bit_identical"],
            "folded_divergent": single_leg["folded_divergent"],
            "unfolded_controls": single_leg["unfolded_controls"],
        }
        log(f"  replay[{policy}] n=1 {single_leg['agree']}/{single_leg['cases']} "
            f"agree, {single_leg['disagree']} disagree (folded exact "
            f"{single_leg['folded_bit_identical']}, folded divergent "
            f"{single_leg['folded_divergent']}); n=60 "
            f"{multi_leg['agree']}/{multi_leg['cases']} agree, "
            f"{multi_leg['disagree']} disagree (folded exact "
            f"{multi_leg['folded_bit_identical']})")
    out["policies_agree_with_each_other"] = tables["keep"] == tables["flush"]
    recorded = {(tuple(folded), tuple(mask), substitution): (ran, identical)
                for folded, mask, substitution, ran, identical
                in coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["combinations"]}
    out["record_matches_the_artifact"] = all(
        tables[policy] == recorded for policy in ("keep", "flush"))
    out["combinations_in_the_record"] = len(recorded)
    out["combinations_in_the_artifact"] = len(tables["keep"])
    # NON-VACUITY: a table that is all-divergent, or one whose combinations are
    # MIXED, cannot measure a rule that reads neither courant nor value class.
    out["mixed_combinations"] = sorted(
        str(key) for key, (ran, identical) in tables["keep"].items()
        if identical not in (0, ran))
    out["has_both_verdicts"] = (
        any(identical == ran for ran, identical in tables["keep"].values())
        and any(identical == 0 for _, identical in tables["keep"].values()))
    out["multi_step_budget_replayed"] = min(
        leg["sixty_launches"]["cases"] for leg in out["policies"].values())
    out["ok"] = bool(
        out["artifact_present"]
        and all(leg["one_launch"]["disagree"] == 0
                and leg["sixty_launches"]["disagree"] == 0
                and leg["sixty_launches"]["folded_bit_identical"] > 0
                for leg in out["policies"].values())
        and out["policies_agree_with_each_other"]
        and out["record_matches_the_artifact"]
        and not out["mixed_combinations"]
        and out["has_both_verdicts"])
    return out


# ---------------------------------------------------------------------------
# LEG 2 — RE-MEASURE the boundary here, with no artifact
# ---------------------------------------------------------------------------

def leg_local(mutation: Optional[str] = None,
              steps: int = 2) -> Dict[str, Any]:
    """The emitted source against ``stepping.update_E`` on real folded grids.

    The FLOORS are per case and are the device gate's own, for its reasons: an
    "exact" verdict on a fixture whose folded axis does not absorb, or whose
    mirror ghost equals the metallic zero it is compared against, is a fact about
    the fixture. A case that fails a floor is scored REFUSED rather than passed.
    """
    cases: List[Dict[str, Any]] = []
    agree = disagree = refused = 0
    for mask, symmetry, substitution in bench.FOLD_CASES:
        folded = tuple(axis for axis in range(3) if "xyz"[axis] in symmetry)
        oracle, produced, moved, deviation, ghost = bench.run_both_paths_folded(
            mask, symmetry, substitution, steps=steps)
        identical = oracle == produced
        differing = sum(
            sum(a != b for a, b in zip(oracle[name], produced[name]))
            for name in oracle)
        floors = {
            "oracle_moved_outputs": int(moved),
            "folded_axis_deviation_from_identity": deviation,
            "mirror_ghost_min_abs": ghost,
            "meets_floor": bool(
                moved == 6
                and (not folded or (deviation > 0.0 and (ghost or 0.0) > 0.0))),
        }
        predicted = rule_says_identical(folded, mask, substitution, mutation)
        entry = {
            "folded_axes": list(folded), "row_mask": list(mask),
            "code_handed_the_fold": substitution,
            "bit_identical": bool(identical), "differing_bytes": int(differing),
            "rule_predicted_identical": bool(predicted),
            "agrees": bool(predicted == identical), "floors": floors,
        }
        cases.append(entry)
        if not floors["meets_floor"]:
            refused += 1
        elif predicted == identical:
            agree += 1
        else:
            disagree += 1
        if len(cases) % 13 == 0:
            log(f"  local {len(cases)}/{len(bench.FOLD_CASES)} measured "
                f"({agree} agree, {disagree} disagree, {refused} refused)")
    folded_cases = [case for case in cases if case["folded_axes"]]
    out = {
        "cases": cases, "scored": len(cases), "agree": agree,
        "disagree": disagree, "refused_as_vacuous": refused,
        "steps_per_case": steps,
        "folded_bit_identical": sum(case["bit_identical"] for case in folded_cases),
        "folded_divergent": sum(not case["bit_identical"] for case in folded_cases),
        "unfolded_controls": len(cases) - len(folded_cases),
        "every_divergent_case_moved_words": all(
            case["differing_bytes"] > 0 for case in cases
            if not case["bit_identical"]),
    }
    out["has_both_verdicts"] = (out["folded_bit_identical"] > 0
                                and out["folded_divergent"] > 0)
    out["ok"] = bool(out["disagree"] == 0 and out["refused_as_vacuous"] == 0
                     and out["has_both_verdicts"]
                     and out["every_divergent_case_moved_words"])
    log(f"  local: {agree}/{len(cases)} agree, {disagree} disagree, "
        f"{refused} refused; folded exact {out['folded_bit_identical']}, "
        f"folded divergent {out['folded_divergent']}")
    return out


# ---------------------------------------------------------------------------
# LEG 3 — WHAT THE ARM IS WORTH, recomputed from the census
# ---------------------------------------------------------------------------

def census_rows() -> List[Dict[str, Any]]:
    """The analyzer's row set, built the analyzer's way: 186 rows.

    Deliberately NOT ``replay.load_rows``, which globs every ``*.jsonl`` and
    returns 205 — the parameterised legs re-emitted into ``tests_param``. The
    union denominator is 759 over 186 rows, and counting a slot against the wrong
    row set is how a delta gets invented.
    """
    rows: List[Dict[str, Any]] = []
    for leg in ("examples", "tests"):
        path = os.path.join(CENSUS, f"{leg}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    recovered = os.path.join(CENSUS, "tests_param_matched.jsonl")
    if os.path.exists(recovered):
        with open(recovered) as handle:
            replacements = {}
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    replacements[(row.get("leg"), row.get("row"))] = row
        rows = [replacements.get((row.get("leg"), row.get("row")), row)
                for row in rows]
    return rows


def _entry(row, family, key):
    block = row.get(family)
    if block is None:
        return None
    return block if key is None else block.get(key)


def _sub_steps(row) -> List[str]:
    return ["step_B", "step_D", "update_H", "update_E"] + (
        ["update_P"] if (row.get("polarization") or []) else [])


def widened_verdict(configuration: Dict[str, Any], mask: Sequence[int],
                    mutation: Optional[str] = None) -> Tuple[bool, str]:
    """The predicate as it WOULD read with the fold clause replaced by the rule.

    The fold is rewritten away and the SHIPPED predicate then answers everything
    else, so this is a widening of one clause rather than a second transcription
    of twenty. The folded axis is declared metallic for the boundary-kind clause
    alone — that is the only clause with no ``mirror`` code — and that is also
    the code the arm was measured exact under. The wall flags are untouched:
    ``offdiag_wall_mask_flags`` already asks ``is_metallic and not is_mirrored``
    and already returns 0 on a folded axis, which is what
    ``stepping._mask_metallic_wall_coupling`` does.
    """
    mirrored = configuration.get("mirrored", [False] * 3)
    folded = tuple(axis for axis in range(3) if mirrored[axis])
    if folded and not rule_says_identical(folded, mask, "metallic", mutation):
        return False, ("a folded axis is the PARTNER axis of a surviving row "
                       "slot: the partner-axis shift reads the mirror ghost "
                       "(parity * g[2]) and the kernel has no mirror branch")
    if folded:
        configuration = dict(configuration)
        configuration["has_symmetry"] = False
        configuration["mirrored"] = [False] * 3
        metallic = configuration.get("metallic", [False] * 3)
        configuration["metallic"] = [bool(metallic[a] or a in folded)
                                     for a in range(3)]
    fields, pml, grid = replay.stand_ins(configuration, mask)
    return coverage.covers_real_pml_offdiag_constitutive(fields, pml, grid)


def leg_corpus(mutation: Optional[str] = None) -> Dict[str, Any]:
    """Union before and after, recomputed. The number that decides the install."""
    rows = census_rows()
    out: Dict[str, Any] = {"census": CENSUS, "rows": len(rows)}
    if not rows:
        out["ok"] = False
        out["why_not_measurable"] = f"{CENSUS} carries no rows"
        return out

    # THE REPLAY IS VALIDATED BEFORE IT IS TRUSTED WITH A NUMBER: every recorded
    # verdict and every recorded first-refusal string must reproduce against the
    # LIVE predicate, or the stand-ins are not standing in for this census.
    validation = replay.validate(
        [row for row in rows
         if "configuration" in row and "cuda_offdiag" in row])
    out["validation"] = {key: validation[key] for key in (
        "slots", "verdict_reproduced", "refusal_reproduced", "unfolded_slots",
        "unfolded_widened_agrees", "validated")}

    slots = before = after = 0
    offdiag_before = offdiag_after = 0
    gained: List[str] = []
    overlaps: List[str] = []
    unserved_update_E = 0
    mirror_first = live_partner = dead_mask = 0
    for row in rows:
        mask = replay.row_mask_of(row)
        for sub_step in _sub_steps(row):
            slots += 1
            serving = [family for family, key in UNION_FAMILIES[sub_step]
                       if (entry := _entry(row, family, key))
                       and entry.get("covered_modulo_backend")]
            before += bool(serving)
            if sub_step != "update_E":
                after += bool(serving)
                continue
            shipped = (row.get("cuda_offdiag") or {}).get(
                "covered_modulo_backend", False)
            offdiag_before += bool(shipped)
            widened = shipped
            if mask is not None:
                widened = bool(widened_verdict(row["configuration"], mask,
                                               mutation)[0])
            offdiag_after += bool(widened)
            others = [family for family in serving if family != "cuda_offdiag"]
            if widened and others:
                overlaps.append(f"{row.get('leg')}/{row.get('row')}: {others}")
            if widened and not serving:
                gained.append(f"{row.get('leg')}/{row.get('row')}")
            after += bool(serving or widened)
            if not serving:
                unserved_update_E += 1
                refusal = (row.get("cuda_offdiag") or {}).get("first_refusal")
                if refusal == MIRROR_REFUSAL:
                    mirror_first += 1
                    folded = tuple(
                        axis for axis in range(3)
                        if row["configuration"].get("mirrored", [False] * 3)[axis])
                    roles = coverage.offdiag_fold_roles(
                        mask or (0,) * 6,
                        tuple(axis in folded for axis in range(3)))
                    if roles["fold_is_a_live_partner_axis"]:
                        live_partner += 1
                    elif not any(mask or ()):
                        dead_mask += 1
    out.update({
        "slots": slots,
        "union_admitted_before": before,
        "union_admitted_after": after,
        "slot_delta": after - before,
        "offdiag_update_E_slots_admitted_before": offdiag_before,
        "offdiag_update_E_slots_admitted_after": offdiag_after,
        "slots_gained": gained,
        "overlaps_introduced": overlaps,
        "update_E_slots_no_family_serves": unserved_update_E,
        "of_those_refused_first_by_the_mirror_clause": mirror_first,
        "of_those_a_fold_is_a_live_partner_axis": live_partner,
        "of_those_no_off_diagonal_row_survived_at_all": dead_mask,
        "counts_are_an_upper_bound": True,
        "why_upper_bound": (
            "the census record carries no dtypes, strides, contiguity or base "
            "addresses, so the stand-ins pass that tail by construction"),
    })
    # NON-VACUITY: if the census carried no folded off-diagonal row at all, a
    # delta of zero would be a fact about the corpus and not about the arm.
    out["folded_offdiagonal_rows_present"] = mirror_first > 0
    out["ok"] = bool(out["validation"]["validated"]
                     and not overlaps
                     and out["folded_offdiagonal_rows_present"]
                     and slots == 759 and len(rows) == 186)
    log(f"  corpus: union {before} -> {after} of {slots} "
        f"(delta {after - before}); offdiag update_E {offdiag_before} -> "
        f"{offdiag_after}; gained {len(gained)}, overlaps {len(overlaps)}")
    return out


# ---------------------------------------------------------------------------
# The verdict, and the battery that has to move it
# ---------------------------------------------------------------------------

def run_legs(mutation: Optional[str] = None) -> Dict[str, Any]:
    return {
        "replay": leg_replay(mutation),
        "local": leg_local(mutation),
        "corpus": leg_corpus(mutation),
    }


def verdict_of(legs: Dict[str, Any]) -> Dict[str, Any]:
    """Conjunctive over every leg, and over the finding the arm is worth zero.

    THE LAST TWO CLAUSES ARE THE POINT OF THE GATE. ``slot_delta == 0`` and
    ``arm_is_not_installed`` are not bookkeeping: this gate releases a REFUSAL,
    so a run in which the arm suddenly bought slots, or in which the predicate had
    quietly been widened, must NOT release on the strength of the boundary legs
    alone — it needs a new decision.
    """
    replay_ok = bool(legs["replay"].get("ok"))
    local_ok = bool(legs["local"].get("ok"))
    corpus = legs["corpus"]
    corpus_ok = bool(corpus.get("ok"))
    delta_zero = corpus.get("slot_delta") == 0
    not_installed = (
        coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["installed"] is False)
    return {
        "boundary_replayed_against_the_device_record": replay_ok,
        "boundary_re_measured_locally": local_ok,
        "corpus_arithmetic_recomputed": corpus_ok,
        "slot_delta_is_zero": bool(delta_zero),
        "arm_is_not_installed": bool(not_installed),
        "released": bool(replay_ok and local_ok and corpus_ok and delta_zero
                         and not_installed),
    }


def leg_mutations(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Every mutation, scored CAUGHT or NULL CONFIRMED — and by WHICH leg.

    ARMED ON A PASSING BASELINE, which is the folded-curl gate's own lesson: a
    leg planted on an already-failing baseline scores CAUGHT for free and every
    null reads as violated.
    """
    out: Dict[str, Any] = {"baseline_released": bool(baseline["released"]),
                           "legs": {}}
    if not baseline["released"]:
        out["ok"] = False
        out["why_not_measurable"] = (
            "the unmutated verdict did not release, so no mutation is armed on a "
            "passing baseline and every result would be free")
        return out
    for mutation in RULE_MUTATIONS:
        legs = run_legs(mutation)
        verdict = verdict_of(legs)
        caught_by = sorted(
            name for name in ("replay", "local", "corpus")
            if not legs[name].get("ok", False))
        if legs["corpus"].get("slot_delta") not in (0, None):
            caught_by.append("corpus:slot_delta")
        predicted_null = mutation in PREDICTED_NULLS
        caught = not verdict["released"]
        out["legs"][mutation] = {
            "verdict_released": verdict["released"],
            "caught": caught,
            "caught_by": caught_by,
            "replay_disagreements": sum(
                leg["disagree"] for leg in
                legs["replay"].get("policies", {}).values()),
            "local_disagreements": legs["local"].get("disagree"),
            "slot_delta": legs["corpus"].get("slot_delta"),
            "predicted": "NULL" if predicted_null else "CAUGHT",
            "scored": ("NULL CONFIRMED" if (predicted_null and not caught)
                       else "CAUGHT" if caught else "NULL"),
            "conforms": bool(caught != predicted_null),
        }
        log(f"  mutation {mutation:38s} -> "
            f"{out['legs'][mutation]['scored']:14s} "
            f"(replay {out['legs'][mutation]['replay_disagreements']}, local "
            f"{out['legs'][mutation]['local_disagreements']}, caught_by "
            f"{caught_by})")
    # WHAT A FOLD-BLIND WIDENING WOULD BUY, measured here rather than quoted:
    # ``admit_every_fold`` is that widening, and its corpus delta is the number
    # the record carries. It is NOT zero, and every slot it buys is one the
    # device gate measured divergent — which is why "the refusal is too broad"
    # does not imply "drop the clause".
    fold_blind = out["legs"].get("admit_every_fold", {}).get("slot_delta")
    recorded = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["corpus"].get(
        "slots_a_fold_blind_widening_would_gain")
    out["fold_blind_widening_slot_delta"] = fold_blind
    out["record_agrees_about_the_fold_blind_delta"] = fold_blind == recorded
    out["conforming"] = sum(leg["conforms"] for leg in out["legs"].values())
    out["scored"] = len(out["legs"])
    out["nulls_confirmed"] = sum(
        leg["scored"] == "NULL CONFIRMED" for leg in out["legs"].values())
    out["ok"] = bool(out["conforming"] == out["scored"]
                     and out["record_agrees_about_the_fold_blind_delta"])
    # THE VERDICT HAS TO BE ABLE TO SAY NO, and this is where that is measured
    # rather than asserted: at least one planted defect must flip it.
    out["verdict_flips_against_a_planted_defect"] = any(
        not leg["verdict_released"] for leg in out["legs"].values())
    return out


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

def save(payload: Dict[str, Any], path: str) -> None:
    """Stamp, then serialise. Never the other way round."""
    gate_provenance.stamp(payload)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def subject_digests() -> Dict[str, str]:
    out = {}
    for name in ("meep_gpu/cuda_kernels/coverage.py",
                 "meep_gpu/cuda_kernels/offdiag_emitter.py",
                 "meep_gpu/cuda_kernels/offdiag_constitutive_kernels.py",
                 "meep_gpu/cuda_kernels/test_offdiag_constitutive_pml_real.py",
                 "meep_gpu/stepping.py",
                 "parity/meep_gpu/gate_cuda_folded_offdiag.py",
                 "parity/meep_gpu/gate_cuda_folded_offdiag_rowmask.py"):
        path = os.path.join(_REPO_API, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                out[name] = hashlib.sha256(handle.read()).hexdigest()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True,
                        help="results directory; must not already hold a verdict")
    parser.add_argument("--skip-mutations", action="store_true",
                        help="the boundary legs only — for a smoke run, not for a "
                             "record: without the battery the verdict is unfalsified")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "gate.json")
    if os.path.exists(path):
        print(f"{path} already exists; write every run into a FRESH directory",
              file=sys.stderr)
        return 2

    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "cuda_folded_offdiag_rowmask",
        "question": ("is covers_real_pml_offdiag_constitutive's fold refusal "
                     "drawn where the 2026-08-20 device gate measured the "
                     "boundary, and what is the measured-safe arm worth?"),
        "makes_a_device_measurement": False,
        "device_artifact_replayed": ARTIFACT,
        "census": CENSUS,
        "started_utc": _now(),
        "emitter_corpus_digest": offdiag_emitter.corpus_digest(),
        "mirror_source_index": int(stepping.MIRROR_SOURCE_INDEX),
        "numpy": np.__version__,
        "subject_sha256": subject_digests(),
    }
    save(payload, path)

    log("leg 1/3 REPLAY — the rule against the device record")
    legs = {"replay": leg_replay()}
    payload["replay"] = legs["replay"]
    save(payload, path)

    log("leg 2/3 RE-MEASURE — the emitted source against stepping, here")
    legs["local"] = leg_local()
    payload["local"] = legs["local"]
    save(payload, path)

    log("leg 3/3 WORTH — the corpus slot arithmetic")
    legs["corpus"] = leg_corpus()
    payload["corpus"] = legs["corpus"]
    save(payload, path)

    payload["verdict"] = verdict_of(legs)
    save(payload, path)
    log(f"unmutated verdict: released={payload['verdict']['released']}")

    if args.skip_mutations:
        payload["mutations"] = {"ok": False, "why_not_measurable":
                                "--skip-mutations was passed"}
    else:
        log("FALSIFY — the mutation battery")
        payload["mutations"] = leg_mutations(payload["verdict"])
    payload["release"] = {
        "released": bool(payload["verdict"]["released"]
                         and payload["mutations"].get("ok")
                         and payload["mutations"].get(
                             "verdict_flips_against_a_planted_defect")),
        "verdict": payload["verdict"],
        "mutations_conform": payload["mutations"].get("ok"),
        "finding": ("the fold refusal is BROADER than the measured boundary and "
                    "stays that way: the measured-safe arm admits ZERO corpus "
                    "slots, and offdiag_boundary_codes has no code to hand a "
                    "folded axis"),
        "does_not_license": (
            "no predicate widening, no device claim, and no statement about the "
            "unguarded (--fmad default) leg, complex storage, dispersion, Dcyl "
            "or a fold on more than three planes"),
    }
    payload["seconds"] = round(time.time() - started, 1)
    save(payload, path)
    log(f"released={payload['release']['released']} -> {path}")
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
