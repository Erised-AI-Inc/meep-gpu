"""Re-cut the two COMPOSITION weld records FROM a composition re-run.

WHY THIS EXISTS, and it is the same defect report ``recut_driver_dispatch_record.py``
opens with. ``fingerprints.json``'s ``cylindrical_composition_gate`` and
``symmetry_composition_gate`` each pin a set of source digests, a probe digest, an
artifact digest and a log digest. Every one of them names bytes that ran on a GPU, so
none may be re-typed from a laptop -- but until now they were maintained by hand, which
means they could drift from the tree AND from the run independently, and nothing
compared the two. The 2026-08-28 round's own ``_this_recut`` prose says as much: the
digests it recorded were the launcher's staged hashes, transcribed.

WHAT IT REFUSES TO DO. It does not release, widen, or invent anything.

* the run directory must hold BOTH legs, each with a ``gate.json`` whose
  ``summary.status`` is ``passed``;
* the run's own digests -- ``staged_source_sha256.txt`` where the launcher wrote one,
  the leg artifact's ``imported_source_sha256`` otherwise -- must agree with the LIVE
  tree on every file the record pins; a record cut against bytes that have already
  moved on is the thing this tool exists to prevent, not a state it may write;
* every key the existing record pins must be present in that manifest, so the
  rewrite can never QUIETLY drop a file from a weld's coverage;
* the probe digest is read from the tree and cross-checked against the one the run
  reported staging.

A RECORD PINS WHAT THE GATE EXECUTED, NOT HOW IT WAS LAUNCHED. The package code, the
gate or probe script, the artifact and the log are digested; a site's launch script and
its batch-scheduler job are not, because this repository does not carry them and no
reader could check them.

Any of those failing prints the disagreement and exits non-zero with nothing written.

WHICH ARCHITECTURE THE RUN CERTIFIED IS READ, NEVER TYPED. Every run fact goes into
``entry["runs"][<compute capability>]`` through :func:`meep_gpu.fastpath.bind_capability`,
and the capability comes from the run itself -- the leg artifact's own identity block,
or a ``device.json`` written beside it by ``triton_device_identity.py --write``. The
legs of one run must agree on it, because one re-cut describes one campaign on one
device. A second architecture is therefore ADDITIVE: re-running these gates on it adds
a record beside the existing one as long as the bytes the entry binds have not moved,
and ``--supersede`` is the only way to leave another architecture's record behind.

``--family-recert`` is the second mode, and it rebuilds the nine family records of
``family_recert_2026-08-14`` from a fleet campaign rather than from the retired 2026-08-14
launcher's logs. That entry is cited by 22 Triton arms, so without a writer for it no
round could ever admit a second architecture: the intersection the admission rests on
would be missing one key forever.

WHAT IT DELIBERATELY LEAVES ALONE. Every hand-authored key -- ``purpose``,
``dispatch``, ``fusion_boundary``, ``device_policy``, the ``recert_*`` blocks, the
``product`` measurement -- is a claim someone made about what the gate MEANS, and is
asserted byte-identical after the rewrite. Only the digests move at entry level, and
only the run facts (:data:`SLOT_FIELDS`) inside the record for the capability that ran.

    python recut_composition_records.py --run results/triton_composition_2026-08-30_carry
    python recut_composition_records.py --run ... --write
    python recut_composition_records.py --family-recert results/triton_fleet_2026-09-25_roundb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "meep_gpu" / "triton_kernels").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


API = find_api_root(Path(__file__).resolve())
HERE = API / "parity" / "meep_gpu"
LEDGER = API / "meep_gpu" / "triton_kernels" / "fingerprints.json"

# THE ENGINE AND THE SIBLING TOOLS BESIDE THIS FILE, not whichever copy is installed.
# This tool reads the ledger out of ``API`` and the capability rules out of
# ``meep_gpu.fastpath``; if those came from two different checkouts the staleness rule
# would be applied by one tree to another tree's digests. ``HERE`` is for
# ``rebind_triton_welds._host_line``, ``seed_triton_welds.policy_line`` and
# ``triton_device_identity`` -- one spelling of the host line, the policy line and the
# capability across every writer, imported rather than restated.
for _path in (str(HERE), str(API)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: entry key -> (leg directory names this tool accepts, probe path)
#:
#: THE SEVEN BELOW ARE THE COMPOSITION AND FUSED-PROBE WELDS, and the five added
#: 2026-08-30 are the entries the ``status == "PASS"`` filter used to hide. Each
#: carries a full device record and a set of pinned source digests, and none is in
#: ``rebind_triton_welds.py``'s ``CAMPAIGN_DIRS`` -- that table enumerates the 31
#: PASS welds only, so before this round nothing could re-cut these at all and the
#: only way to move one was by hand. Every probe path here is read off the entry's own
#: ``probe`` key, not derived from the leg name.
#:
#: TWO LAUNCHERS, TWO DIRECTORY NAMES, 2026-09-30. The private composition launcher
#: writes the first name of each pair; the SHIPPED fleet driver
#: (``drive_triton_weld_gates.GATES``) writes the second, and for four of the nine they
#: differ -- ``engine_route``, ``probe_triton_dispersive_fused_pair``,
#: ``probe_triton_fused_ade_state`` and ``no_pml``. Only the fleet ships, so a round run
#: on a rented host could not be consumed at all until both spellings were named here.
#: EXACTLY ONE of a pair may be present in a run: a root holding both is refused by
#: name rather than resolved by precedence, because which one cut the record is not a
#: question this tool may answer by sort order.
#:
#: THE TWO COLLIDING NAMES ARE SAFE BECAUSE THE ARTIFACT IS CHECKED, not because the
#: directory name is trusted. ``drive_triton_weld_gates`` warns that its own ``no_pml``
#: and ``fused_electric`` rows run the same scripts under non-composition arguments.
#: Measured 2026-09-30 on ``results/triton_fleet_2026-09-23_sparse``: the fleet's
#: ``no_pml/gate.json`` carries ``canonical_verdict.released = null`` and no
#: ``validation`` block, so :func:`released` reads "no verdict key" and the leg is
#: refused before anything binds -- the composition invocation is what writes the
#: ``engine``/``validation`` legs the record's claims are about. The fleet's
#: ``fused_electric/gate.json`` is the same invocation as the composition launcher's
#: (same ``gate`` name, same summary counters, 32 imported digests on both), so there
#: is nothing to distinguish and nothing to lose.
LEGS = {
    "cylindrical_composition_gate": (
        ("cylindrical_composition",),
        "parity/meep_gpu/probe_triton_cylindrical_composition.py"),
    "symmetry_composition_gate": (
        ("symmetry_composition",),
        "parity/meep_gpu/probe_triton_symmetry_composition.py"),
    "conductivity_composition_gate": (
        ("conductivity_composition",),
        "parity/meep_gpu/probe_triton_conductivity_composition.py"),
    "dispersive_composition_gate": (
        ("dispersive_composition", "engine_route"),
        "parity/meep_gpu/probe_triton_engine_route.py"),
    "dispersive_fused_pair_gate": (
        ("dispersive_fused_pair", "probe_triton_dispersive_fused_pair"),
        "parity/meep_gpu/probe_triton_dispersive_fused_pair.py"),
    "fused_ade_state_gate": (
        ("fused_ade_state", "probe_triton_fused_ade_state"),
        "parity/meep_gpu/probe_triton_fused_ade_state.py"),
    "source_seam_gate": (
        ("source_seams",),
        "parity/meep_gpu/probe_triton_source_seams.py"),
    "no_pml_composition_gate": (
        ("no_pml_composition", "no_pml"),
        "parity/meep_gpu/gate_triton_no_pml.py"),
    "fused_electric_gate": (
        ("fused_electric",),
        "parity/meep_gpu/gate_triton_fused_electric.py"),
}

#: The nine families ``family_recert_2026-08-14`` certifies, which are also nine gate
#: names in ``drive_triton_weld_gates.GATES`` -- so a fleet campaign re-runs all nine
#: and ``--family-recert`` can rebuild the record from artifacts instead of from the
#: retired 2026-08-14 launcher's logs. TYPED HERE AND CHECKED AGAINST THE LEDGER at
#: run time (:func:`family_recert`), because a family silently dropped from this tuple
#: would narrow a 22-arm weld's coverage without any digest changing.
FAMILY_RECERT_FAMILIES = ("bfast", "complex", "cylindrical_complex", "folded_complex",
                          "folded_offdiag", "no_pml_constitutive", "nonlinear",
                          "offdiag", "special_kz")

#: The blocks a declared re-cut may REWRITE. Reading is structural -- see
#: :func:`claim_disagreements`, which walks the whole entry -- so this list no
#: longer decides what gets COMPARED; it bounds what may be OVERWRITTEN, which is
#: the narrower and more dangerous act. ``product`` was the only one the two
#: original entries carried; the composition welds state theirs in
#: ``real_engine_route``, and reading only ``product`` would have let a
#: composition entry rebind its digests while its stated route measurement
#: silently stopped being true.
CLAIM_BLOCKS = ("product", "real_engine_route", "single_launch", "composition")

#: An "n/m" measurement. The record may extend it with prose ("36/36 against
#: both oracles"); it may not disagree about n or m.
COUNTER = re.compile(r"^\d+/\d+")

#: Curated claims DELIBERATELY RE-CUT from a fresh run, each with the reason.
#:
#: THE DEFAULT IS THE OPPOSITE, and it stays the default. A curated block is
#: carried forward verbatim and the RUN is checked against IT: what a gate
#: measured is a measurement, and a tool that overwrote it from every fresh
#: summary would turn "the record describes the run" into "the record is whatever
#: the last run said", which is how a claim quietly loses the specificity someone
#: put in it. So a disagreement REFUSES the entry, and clearing that refusal is a
#: decision a person makes, in this table, where a reviewer sees it beside what
#: moved.
#:
#: THE ONE ENTRY, 2026-08-30. ``dispersive_fused_pair_gate.product`` says
#: ``complete_steps_exact 36/36`` over three named cases at 12 steps each. The
#: gate GAINED A CASE -- ``two_pole_electric_source``, the in-seam deposit case --
#: and the fresh run measures 48/48 over four. Both halves of that matter: the
#: total is stale, AND the three per-case lines no longer add up to it, so
#: carrying the block forward beside freshly bound digests would leave a weld
#: whose hashes verify and whose arithmetic does not. The re-cut is mechanical
#: (see :func:`recut_claim`) -- every counter comes out of the artifact, the
#: per-case lines are derived from its own ``cases`` list and cross-checked
#: against the summary's total, and the record's trailing prose on each field is
#: preserved.
CLAIM_RECUTS = {
    ("dispersive_fused_pair_gate", "product"):
        "the gate gained a case. 2026-08-11 measured three pole products at 12 "
        "complete steps each (36/36); the 2026-08-30 re-run measures four, adding "
        "two_pole_electric_source -- the in-seam deposit case -- for 48/48 and "
        "1/1 in_seam_deposit_cases. Carrying 36/36 forward beside fresh digests "
        "would leave a weld whose hashes verify and whose claim is stale, and "
        "whose three 12/12 lines sum to 36 beside a total of 48.",
    # THE SECOND ENTRY, 2026-08-31. TWO counters move on this block and they move
    # for two DIFFERENT reasons, which is why the declaration names both rather
    # than one sentence covering the pair.
    #
    # `covered_whole_step_identical` 4/4 -> 8/8. The gate did not gain a case; the
    # WHOLE-STEP LEG gained six of them. It runs on every case with a non-empty
    # `replaces` list, and in the 2026-08-11 dispersive-composition run the four non-PML-curl cases
    # composed to an EMPTY plan, so the leg skipped them. Every one of those four
    # now composes through wired arms, so six complete driver steps are compared
    # as uint32 on all eight cases instead of four -- strictly more measured, and
    # all eight identical. This counter was invisible to `claim_disagreements`
    # until the probe's summary was renamed to the record's own field names in the
    # same change, so 2026-08-30's re-run could not report it at all.
    #
    # `refusals_returned_none` 4/4 -> 3/4. NOT A REGRESSION, and it is the numerator
    # the whole re-cut turns on. `2d_cond_pml` carries `D_conductivity=0.4`, so
    # `fields.condfac_for` answers non-None on Dx/Dy/Dz and None on Bx/By/Bz.
    # `pml_curl_coverage` with NO sub_step takes the conservative aggregate path over
    # all six CURL_TARGETS and refuses; asked for `step_B` it names no conductivity
    # reason at all. `plan_pml_curl` is a builder and asks for its own named
    # sub-step, so it returns a plan on step_B and None on step_D -- and the probe's
    # `all(... is None)` over both is False. Job 2298 ran a `coverage.py` with no
    # conductivity clause in it at all (grep count 0) and an engine that composed
    # NOTHING on that case; the conductive family landed hours later, on the conductivity-composition run,
    # which asserted `plan_step` selects the conductive products on this same case
    # and passed. Today the case is served in full: four of four sub-steps replaced,
    # `refusals` empty, six complete steps byte-identical. Carrying 4/4 forward would
    # be a weld claiming the engine still declines a configuration it now serves.
    ("dispersive_composition_gate", "real_engine_route"):
        "two counters, two reasons. covered_whole_step_identical 4/4 -> 8/8: the "
        "whole-step leg runs on every case that composes a plan, and the four cases "
        "that composed an EMPTY plan on 2026-08-11 now compose through wired arms, "
        "so six complete driver steps are compared as uint32 on eight cases rather "
        "than four -- strictly more measured, all eight identical. "
        "refusals_returned_none 4/4 -> 3/4: 2d_cond_pml carries a D-side "
        "conductivity only, so pml_curl_coverage refuses it in AGGREGATE (all six "
        "CURL_TARGETS) while its step_B verdict names no conductivity reason, and "
        "plan_pml_curl -- which asks for its own named sub-step -- returns a plan "
        "there and None on step_D. The conductive PML family landed between job "
        "2298 and now; the case is served in full today (4/4 sub-steps, no "
        "refusals, six steps byte-identical), so 4/4 would claim the engine still "
        "declines a configuration it serves.",
}


def released(payload: dict):
    """(verdict, where) over the three spellings these gates actually write.

    ENUMERATED BECAUSE THEY DISAGREE, measured over the seven artifacts:
    ``canonical_verdict.released`` (dispersive_fused_pair, fused_ade_state,
    fused_electric, no_pml_composition), ``summary.status == "passed"``
    (cylindrical, symmetry, conductivity, source_seams), and
    ``validation.status`` (no_pml_composition, whose summary is empty). Reading
    one spelling scores the others ``None``, and a ``None`` that is treated as a
    failure blocks a passing weld while a ``None`` treated as a pass binds a weld
    to a run that never released. Both directions are wrong, so all three are
    named here rather than guessed at the call site.
    """
    verdict = payload.get("canonical_verdict")
    if isinstance(verdict, dict) and verdict.get("released") is not None:
        return bool(verdict["released"]), "canonical_verdict.released"
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("status") is not None:
        return summary["status"] == "passed", "summary.status"
    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("status") is not None:
        return validation["status"] == "passed", "validation.status"
    return None, "no verdict key"

#: Keys the rewrite is allowed to move. Anything else on an entry is asserted
#: byte-identical afterwards, which is what keeps this a re-cut and not a rewrite.
#:
#: ``product`` IS NOT HERE, and the first cut of this tool got that wrong. The record's
#: product block reads "1/1: on-axis Ez" where the artifact's summary reads "1/1" --
#: the record says WHICH source case was non-vacuous and the summary only counts. A
#: re-cut that overwrote it from the artifact silently deleted the more specific
#: claim, and the weld test caught it. What the artifact measures is asserted against
#: the record instead, below.
#: The ENTRY-level keys the rewrite may move: the digests that say which bytes the
#: weld binds. ``fastpath.BOUND_FIELDS`` is computed over exactly these two, so moving
#: one is what can stale another architecture's record -- see :func:`apply_updates`.
ENTRY_MUTABLE = {"probe_sha256", "source_sha256"}

#: The RUN facts, which live in ``runs[<capability>]`` and nowhere else. Every one is
#: in ``fastpath.RUN_FIELDS``, asserted at import below, because ``bind_capability``
#: refuses a field that belongs beside the digests.
SLOT_FIELDS = {"artifact_sha256", "log_sha256", "records", "recorded_utc", "elapsed",
               "_this_recut", "host", "device_policy"}

#: The two slot fields that are PINS -- they name bytes. A record whose existing
#: capabilities carry one and whose fresh run cannot produce it is REFUSED rather than
#: written without it: dropping a pin narrows a weld's coverage silently, which is the
#: defect the "a key the entry does not carry is not a key to add" rule guards in the
#: other direction. ``elapsed`` and ``device_policy`` are not pins (a duration and a
#: curated claim), so they are derived where the run states them and omitted where it
#: does not.
SLOT_PINS = ("artifact_sha256", "log_sha256")

#: The run facts of ``family_recert_2026-08-14``'s record. ``families`` holds the nine
#: per-family blocks, which are facts about nine runs and so belong inside the record
#: for the architecture they ran on, not beside the entry's digests.
FAMILY_SLOT_FIELDS = {"host", "records", "recorded_utc", "families", "_this_recut"}

#: What a family record says when its gate's artifact states no step budget.
#:
#: AN ABSENT BUDGET IS A HANDLED CASE, NOT A REASON TO REFUSE, and the alternative was
#: measured before this string was written. Of the nine family gates, four state
#: ``step_budgets`` (complex, cylindrical_complex, bfast, special_kz), one states the
#: singular ``step_budget`` (folded_complex), one states it under ``provenance``
#: (no_pml_constitutive) and three state none at all (offdiag, nonlinear,
#: folded_offdiag) -- measured 2026-09-30 over results/triton_fleet_2026-09-23_sparse,
#: whose campaign ran all nine. A writer that refused the three would therefore refuse
#: in EVERY round, which would leave the 22 arms citing this entry unbindable forever
#: and the Triton table unable to admit a second architecture. The record already reads
#: this way for three families from the 2026-08-14 launcher, and
#: ``fastpath._certification_for`` already quotes an absent budget as absent.
BUDGET_NOT_STATED = (
    "not stated by this gate's artifact: it writes no step_budget(s) key, so the "
    "budget is in the run artifacts named beside this line and not here. Never read "
    "this as 'the campaign-wide budget applies'.")

#: The same, for the policy, and for the same reason.
#:
#: ``seed_triton_welds.policy_line`` is the shared spelling and is asked FIRST. On the
#: 09-23 fleet it answered for seven of the nine families and returned ``None`` for
#: cylindrical_complex and no_pml_constitutive, whose artifacts stamped
#: ``subnormal_policy: null``; their 2026-10-03 artifacts stamp the policy under
#: ``summary.subnormal_policy`` and ``policy``, which it now reads, so this string is
#: for an artifact that states no attained policy anywhere. The campaign row records
#: what the driver REQUESTED, which is not what the gate attained, so it is not
#: substituted here: a record claiming a policy no stamp attests is the forgery this
#: tool exists to avoid. Refusing instead would make such a family unbindable in
#: every round -- the same dead end as the step budget. Spellings that DISAGREE are
#: different: they raise ``PolicyStampConflict`` and the re-cut refuses that family.
POLICY_NOT_STATED = (
    "not stated by this gate's artifact: it writes no resolved subnormal_policy "
    "stamp, so this family's record makes no claim about the policy it ran under. "
    "The campaign's requested policy is recorded in the campaign artifact, and a "
    "request is not an attainment.")

#: WHERE A RUN SAYS WHICH DEVICE IT RAN ON is ``triton_device_identity.run_capability``
#: and is not restated here. Measured 2026-09-30 over the 09-23 and 09-25 fleets and
#: the 09-25 composition run: not one composition leg records a compute capability in
#: its own artifact -- the four probes that write an ``environment`` block record
#: device/cupy/triton/time and no capability or hostname, and ``engine_route``'s block
#: is empty -- so in practice every leg's capability comes from a ``device.json``
#: written beside the run by ``triton_device_identity.py --write``, which is the fourth
#: source that helper reads.
DEVICE_STAMP_HINT = ("python parity/meep_gpu/triton_device_identity.py --write "
                     "{directory}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fastpath():
    """The engine's capability-record rules, imported LAZILY.

    This module is imported for its ``LEGS`` table by ``migrate_capability_records.py``
    and by ``test_regate_writers.py``, and the same argument
    ``rebind_triton_welds._host_line`` makes applies here: a tool that must be able to
    REPORT on a checkout where ``meep_gpu`` will not import may not import it at module
    scope. Every write goes through this, so there is no second definition of the
    container or of the staleness rule.
    """
    from meep_gpu import fastpath  # noqa: PLC0415
    missing = sorted((SLOT_FIELDS | FAMILY_SLOT_FIELDS) - set(fastpath.RUN_FIELDS))
    if missing:
        raise SystemExit(f"{missing} are not fastpath.RUN_FIELDS, so bind_capability "
                         f"would refuse them; the two tables have drifted")
    return fastpath


#: Key spellings that mark a container as HISTORY, with the reason. A counter
#: inside one describes a run that is over; requiring it to match today's summary
#: would demand that the past change, and the only way to clear the resulting red
#: would be to delete the evidence.
#:
#: The first four are the exact spellings ``meep_gpu/weld_record_walk.py``
#: excludes DIGESTS by (``historical_superseded_run``,
#: ``historical_recert_snapshot``, ``historical_recut_log``); the fifth covers the
#: dated ``recert_2026_08_13`` / ``family_recert_2026-08-14`` blocks, which hold
#: counters rather than digests and so never reached that walker. Held to the same
#: both-directions rule as every other exclusion here: see
#: ``test_regate_writers.py::test_the_curated_claim_walk_excludes_history_and_says_so``.
HISTORICAL_CONTAINERS = ("superseded_runs", "_at_recert", "_recut_log",
                         "source_drift_since_recert", "recert")


def _is_historical(trail) -> bool:
    return any(isinstance(key, str)
               and (key == "superseded_runs" or key.endswith("_at_recert")
                    or key.endswith("_recut_log")
                    or key == "source_drift_since_recert" or "recert" in key)
               for key in trail)


def _counters(node, trail=()):
    """``(trail, field, value)`` for every ``n/m`` string ANYWHERE in an entry.

    STRUCTURAL, not a list of block names. ``CLAIM_BLOCKS`` names four spellings
    and the record holds THIRTY-THREE distinct key names carrying an ``n/m``
    counter, several of them at the entry's own top level. The two happen to
    agree today -- both find the same 13 comparable counters -- and that
    coincidence is exactly the shape this project keeps being bitten by: a
    denominator that is right by accident stops being right without saying so.
    Recursing costs nothing and cannot go stale.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and COUNTER.match(value):
                yield trail + (key,), key, value
            else:
                yield from _counters(value, trail + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _counters(value, trail + (index,))


def claim_disagreements(entry: dict, payload: dict, block: str | None = None):
    """Counters where a curated measurement and the fresh run's summary differ.

    THE COMPARISON IS COUNTERS ONLY, and that narrowing is the check getting
    STRONGER rather than weaker. A shared key name does not mean a shared
    meaning: ``controls`` is prose in the record and a list of case names in the
    summary, and ``oracles_per_step`` is prose on both sides whose WORDING
    changed while its content did not. Comparing those by ``startswith`` reports
    a difference that is not one, and a check that cries wolf on prose is a check
    someone deletes. What must not move is the measurement: an ``n/m`` counter,
    whose DENOMINATOR is the coverage the weld claims.

    THE JOIN IS THE FIELD NAME, and the denominator that produces is small and
    worth stating rather than implying: 13 counters across the 38 entries whose
    ``records`` line resolves to a payload in this checkout. Most curated fields
    share no name with any summary, so they are not compared here by ANYTHING --
    which is a fact about what the runs report, not a filter this function
    applies.

    Returns ``(where, field, recorded, measured)`` with ``where`` the dotted trail
    above the field -- ``"product"`` for a top-level block, so the CLAIM_RECUTS
    key is unchanged, and something longer for a nested one, which then has no
    declaration and correctly refuses.
    """
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        return []
    found = []
    for trail, field, recorded in _counters(entry):
        if _is_historical(trail):
            continue
        where = ".".join(str(part) for part in trail[:-1])
        if block is not None and where != block:
            continue
        measured = summary.get(field)
        if not isinstance(measured, str) or not COUNTER.match(measured):
            continue
        if not recorded.startswith(measured):
            found.append((where, field, recorded, measured))
    return found


def per_case_counters(payload: dict):
    """``{case name: (exact, total)}`` from the artifact's own ``cases`` list.

    ``None`` when the payload is not that shape, so a run that does not report
    per-case detail refuses the derivation rather than getting an invented one.
    A case is EXACT at a step only if every comparison that step recorded is
    ``bit_identical`` -- the fused-vs-array and fused-vs-separate oracles both --
    so a run that dropped one oracle cannot score the same as one that kept it.
    """
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return None
    out = {}
    for case in cases:
        if not isinstance(case, dict):
            return None
        name, steps = case.get("case"), case.get("per_step")
        if not isinstance(name, str) or not isinstance(steps, list) or not steps:
            return None
        exact = 0
        for step in steps:
            if not isinstance(step, dict):
                return None
            comparisons = [value for value in step.values()
                           if isinstance(value, dict) and "bit_identical" in value]
            if comparisons and all(value["bit_identical"] for value in comparisons):
                exact += 1
        out[name] = (exact, len(steps))
    return out


def recut_claim(entry: dict, block: str, payload: dict):
    """The curated block, RESTATED FROM THE RUN. Returns ``(fresh, moved, why)``.

    Three mechanical moves and no judgement in any of them:

    1. EVERY SHARED COUNTER FOLLOWS THE RUN, and the record's trailing prose is
       kept. ``"36/36"`` beside a measured ``"48/48"`` becomes ``"48/48"``;
       ``"36/36 against both oracles"`` becomes ``"48/48 against both oracles"``.
       The prose is the record's more specific claim and deleting it would be the
       same defect in the other direction. A recorded value that is NOT
       counter-prefixed cannot be split and refuses.
    2. EVERY COUNTER THE RUN REPORTS AND THE RECORD LACKS IS ADDED, verbatim.
       Without this a gate that grows a leg has its new measurement dropped on
       the floor by the very tool that was told to follow the run --
       ``in_seam_deposit_cases 1/1 (two_pole_electric_source)`` is exactly that
       case here.
    3. PER-CASE LINES ARE DERIVED FROM THE ARTIFACT'S OWN ``cases`` LIST, one
       field per case name, and CROSS-CHECKED against the summary's own total: if
       the per-case numerators and denominators do not sum to it, the derivation
       and the summary disagree and this refuses rather than writing either. That
       cross-check is what makes the derivation evidence instead of arithmetic.
    """
    summary = payload.get("summary") or {}
    recorded = dict(entry.get(block) or {})
    fresh, moved = dict(recorded), []

    for field, value in sorted(recorded.items()):
        measured = summary.get(field)
        if measured is None or not isinstance(value, str):
            continue
        match = COUNTER.match(str(measured))
        if match is None or str(value).startswith(str(measured)):
            continue
        old = COUNTER.match(value)
        if old is None:
            return None, [], (f"{block}[{field}] is {value!r}, which carries no "
                              f"leading counter to replace; a claim this tool "
                              f"cannot split it may not rewrite")
        fresh[field] = str(measured) + value[old.end():]
        moved.append(f"{block}[{field}] {value!r} -> {fresh[field]!r}")

    for field, measured in sorted(summary.items()):
        if field in fresh or not isinstance(measured, str):
            continue
        if COUNTER.match(measured):
            fresh[field] = measured
            moved.append(f"{block}[{field}] ADDED {measured!r} — the run reports a "
                         f"counter this block did not carry")

    cases = per_case_counters(payload)
    if cases and any(name in recorded for name in cases):
        exact = sum(value[0] for value in cases.values())
        total = sum(value[1] for value in cases.values())
        stated = fresh.get("complete_steps_exact")
        if stated is None or not COUNTER.match(str(stated)):
            return None, [], (f"{block} carries per-case lines and no "
                              f"complete_steps_exact counter to cross-check them "
                              f"against")
        if COUNTER.match(str(stated)).group(0) != f"{exact}/{total}":
            return None, [], (
                f"{block}: the artifact's own cases sum to {exact}/{total} and its "
                f"summary states {stated!r}. The derivation and the summary "
                f"disagree; neither is written.")
        for name, (case_exact, case_total) in sorted(cases.items()):
            counter = f"{case_exact}/{case_total}"
            previous = recorded.get(name)
            if isinstance(previous, str) and COUNTER.match(previous):
                value = counter + previous[COUNTER.match(previous).end():]
            elif previous is None:
                value = f"{counter} complete steps exact"
            else:
                return None, [], (f"{block}[{name}] is {previous!r}, which carries "
                                  f"no leading counter to replace")
            if value != previous:
                fresh[name] = value
                moved.append(f"{block}[{name}] {previous!r} -> {value!r}")
    return fresh, moved, ""


def pinned_path(name: str) -> str:
    """The repo-relative path a pinned BASENAME names — ONE resolution, both sides.

    The record pins bare basenames (``launch.py``, ``__init__.py``), resolved
    against ``meep_gpu/triton_kernels/`` first and the package root second. The
    staged manifest, by contrast, carries full repo-relative paths. Whichever way
    the two are joined, they must be joined ONCE: this is the function that says
    which file a pinned name means, and both the tree read and the manifest
    lookup below go through it.
    """
    candidate = f"meep_gpu/triton_kernels/{name}"
    return candidate if (API / candidate).is_file() else f"meep_gpu/{name}"


def read_staged(run: Path) -> Optional[Dict[str, str]]:
    """``{repo-relative path: digest}`` from the run's own staged manifest.

    KEYED BY PATH, NEVER BY BASENAME, and that is a DEFECT FIX rather than a
    tidy-up. This map was ``{basename: digest}``, so every manifest row sharing a
    basename with another collapsed onto one entry and the survivor was decided
    by the manifest's SORT ORDER. Measured 2026-09-08 on
    ``results/triton_composition_2026-09-08_wired``: 889 rows, 7 of them named
    ``__init__.py``, and the last one in path order is
    ``parity/meep_gpu/test_shims/parameterized/__init__.py``. So the tool compared
    the tree's ``meep_gpu/triton_kernels/__init__.py`` against a test shim's
    digest, reported "staged d76e07fc9107 but the tree holds 87f8dee235ed", and
    REFUSED 8 of the 9 composition entries — while the file the record actually
    pins was in the manifest at 87f8dee235ed and equalled the tree exactly.

    The 2026-09-07 manifest held the same seven rows in the opposite order, so the
    same last-wins map happened to pick the right file and this tool passed BY
    ACCIDENT OF ORDERING. A comparison whose answer depends on how a manifest was
    sorted is not a comparison, in either direction: the ordering that refused a
    correct re-cut here could as easily have licensed an incorrect one.

    Sixty-six of the 889 basenames in that manifest are ambiguous, so this is not
    a single-file quirk; ``__init__.py`` is only the one the pinned set reached.
    """
    manifest = run / "staged_source_sha256.txt"
    if not manifest.is_file():
        # ABSENT IS NOT AN ERROR ANY MORE, 2026-09-30. No SHIPPED tool writes this
        # file -- only the private composition launcher does -- so raising here meant
        # a round driven by the fleet driver could not re-cut a single one of these
        # nine welds. The caller falls back to each leg artifact's own
        # ``imported_source_sha256`` (:func:`leg_digests`), which is a per-leg
        # manifest rather than a campaign-wide one and therefore cannot stand in for
        # the ``host_sha256`` licence below.
        return None
    out: Dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def leg_directory(run: Path, names: Sequence[str]) -> Tuple[Optional[Path], str]:
    """The ONE directory of ``names`` this run holds, or a named refusal.

    Two launchers spell four of these legs differently (:data:`LEGS`), and a root
    holding both spellings is ambiguous about which run cut the record -- so it is
    refused rather than resolved. A directory with no ``gate.json`` is NOT a leg: a
    gate that died leaves its directory behind with a log and no artifact, and
    matching on ``is_dir()`` alone would report the dead leg's absence of a verdict as
    the run's answer.
    """
    held = [run / name for name in names if (run / name / "gate.json").is_file()]
    if len(held) > 1:
        return None, (f"the run holds {[p.name for p in held]}, both of which this "
                      f"record names; which one cut it is not for this tool to guess")
    if not held:
        return None, (f"the run holds none of {list(names)} with a gate.json in it")
    return held[0], ""


def leg_digests(payload: dict, staged: Optional[Dict[str, str]],
                directory: Path) -> Tuple[Optional[Dict[str, str]], str]:
    """``{repo-relative path: digest}`` for one leg, and WHERE it came from.

    The campaign-wide staged manifest when the launcher wrote one, otherwise the leg
    artifact's own ``imported_source_sha256``. Measured 2026-09-30 across the 09-20,
    09-23 and 09-25 fleets: five of the nine legs record an import set (32 to 83
    files), and four -- ``cylindrical_composition``, ``symmetry_composition``,
    ``conductivity_composition`` and ``source_seams`` -- record NONE in either
    launcher, because those probes never call the provenance import tracker. Those
    four are therefore re-cuttable only from a run that wrote a staged manifest, and
    they refuse by name under the fleet rather than being bound from the tree.
    """
    if staged is not None:
        return staged, "the run's staged_source_sha256.txt"
    imported = payload.get("imported_source_sha256")
    if isinstance(imported, dict) and imported:
        return dict(imported), f"{directory.name}/gate.json:imported_source_sha256"
    return None, (f"the run wrote no staged_source_sha256.txt and "
                  f"{directory.name}/gate.json records no imported_source_sha256, so "
                  f"nothing in this run says which bytes it executed")


def leg_capability(payload: dict, directories: Sequence[Path]
                   ) -> Tuple[Optional[str], Any]:
    """``(capability, identity)`` for one leg, or ``(None, reason)``.

    ``triton_device_identity.run_capability`` is the one rule -- four sources, every
    one that answers must agree, and the ``device.json`` a separate process wrote is
    read last and never guessed past. It is CALLED rather than restated for the reason
    its own docstring gives: a writer that typed the key would weld a claim about an
    architecture no measurement names. Both directories are offered because the fleet
    driver stamps the campaign root once while a single-leg re-run stamps its own
    directory.
    """
    from triton_device_identity import run_capability  # noqa: PLC0415
    return run_capability(payload, [str(path) for path in directories])


def derived_host(identity: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """The slot's ``host`` line, DERIVED on every write, in the one shared spelling.

    ``rebind_triton_welds._host_line`` composes it from the merged identity, so a
    composition record and a device gate's record are indistinguishable; this adds the
    toolchain check. CARRYING THE OLD LINE FORWARD IS THE DEFECT, not the convenience:
    a re-run of the same gate on a Triton 3.2 host would otherwise inherit a record
    reading 3.1.0, and the contract test, which greps the carried string, would pass it.
    """
    from rebind_triton_welds import _host_line  # noqa: PLC0415
    host = _host_line(identity)
    if host is None:
        return None, ("the run's identity is incomplete: a host line needs the "
                      "machine, the device, the capability, CuPy and Triton, and this "
                      "run states " + str(sorted(identity)))
    validated = _fastpath().validated_triton_versions()
    if str(identity.get("triton")) not in validated:
        return None, (f"the run reports Triton {identity.get('triton')!r}, which is "
                      f"outside the record's validated list {list(validated)}; a weld "
                      f"cut on an unvalidated toolchain must widen that declaration "
                      f"deliberately, not inherit it from a rebind")
    return host, ""


def derived_elapsed(payload: dict) -> Optional[str]:
    """The leg's wall time, from the artifact's OWN stamps, or ``None``.

    Measured 2026-09-30: ``fused_electric`` states ``elapsed_seconds``,
    ``engine_route`` states only ``started_utc``/``finished_utc``, and the rest state
    neither. A duration is a run fact, so it is derived where the run states one and
    LEFT OUT where it does not -- never carried from the previous record, which is
    where every other field's staleness came from.
    """
    seconds = payload.get("elapsed_seconds")
    if isinstance(seconds, (int, float)):
        return f"{float(seconds):.1f} s (the artifact's own elapsed_seconds)"
    started, finished = payload.get("started_utc"), payload.get("finished_utc")
    if isinstance(started, str) and isinstance(finished, str):
        try:
            span = (datetime.fromisoformat(finished.replace("Z", "+00:00"))
                    - datetime.fromisoformat(started.replace("Z", "+00:00")))
        except ValueError:
            return None
        whole = int(span.total_seconds())
        return (f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d} wall "
                f"({started} to {finished}, the artifact's own stamps)")
    return None


def apply_updates(ledger: dict, updates: Dict[str, dict], capability: str,
                  supersede: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Move the entry digests and bind ``capability``'s record. ``(notes, problems)``.

    RUN ON A COPY IN REPORT MODE, so a refusal this can only find at bind time -- a
    write that would strand another architecture's record -- is reported before
    ``--write`` rather than after it.

    THE FROZEN SET IS ``ENTRY_MUTABLE`` PLUS WHAT THIS RUN DECLARED, and not one key
    more. A curated block moves only where ``CLAIM_RECUTS`` names ``(entry, block)``
    AND this run actually re-cut it; a declaration that did not fire leaves the block
    frozen like everything else, so the table cannot become a standing licence for a
    block to drift. ``moves`` is the one widening, and only ``--family-recert`` sets
    it: that entry's bound is its ``source_sha256_at_recert`` map rather than a
    ``source_sha256``, and its drift declaration and dated log move with it, so the
    movable set is named per call instead of being bolted onto the shared constant.
    ``runs`` is excluded from the comparison and proved separately:
    every OTHER capability's record must come out byte-identical unless this write
    superseded it, which is the whole point of keeping one record per architecture.
    """
    fastpath = _fastpath()
    notes: List[str] = []
    problems: List[str] = []
    for key, fresh in sorted(updates.items()):
        entry = ledger[key]
        may_move = (ENTRY_MUTABLE
                    | {block for block in fresh["claims"]
                       if (key, block) in CLAIM_RECUTS}
                    | set(fresh.get("moves", ())))
        frozen = {name: value for name, value in entry.items()
                  if name not in may_move and name != fastpath.RUNS}
        others = json.loads(json.dumps(
            {name: record for name, record in entry[fastpath.RUNS].items()
             if name != capability}))
        # BEFORE the digests move: this is the bound the OTHER records were written
        # against, and the only moment both values are in hand.
        bound_before = fastpath.bound_digest(entry)
        entry.update(fresh["entry"])
        entry.update(fresh["claims"])
        try:
            staled = fastpath.bind_capability(
                entry, bound_before=bound_before, capability=capability,
                run=fresh["run"],
                run_fields=sorted(fresh.get("run_fields") or SLOT_FIELDS),
                supersede=supersede)
        except fastpath.CapabilityRecordError as exc:
            problems.append(f"{key}: {exc}")
            continue
        after = {name: value for name, value in entry.items()
                 if name not in may_move and name != fastpath.RUNS}
        if after != frozen:
            moved = sorted({name for name in set(after) | set(frozen)
                            if after.get(name) != frozen.get(name)})
            raise SystemExit(f"{key}: {moved} moved and is outside ENTRY_MUTABLE; "
                             f"refusing to write")
        survived = {name: record for name, record in entry[fastpath.RUNS].items()
                    if name != capability and name not in staled}
        expected = {name: record for name, record in others.items()
                    if name not in staled}
        if json.dumps(survived, sort_keys=True) != json.dumps(expected,
                                                              sort_keys=True):
            raise SystemExit(f"{key}: another capability's record moved; refusing "
                             f"to write")
        if staled:
            notes.append(f"{key}: superseded {list(staled)} -- those architectures' "
                         f"records bind bytes this run moved")
    return notes, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",
                      help="composition run directory, parity/meep_gpu "
                           "relative (e.g. results/triton_composition_2026-08-30_carry)")
    mode.add_argument("--family-recert", nargs="+", default=(), metavar="ROOT",
                      help="campaign roots holding the nine family gates, "
                           "parity/meep_gpu relative, in precedence order: the FIRST "
                           "root with a released artifact for a family wins, so a "
                           "re-run of two families goes first and the full fleet second")
    parser.add_argument("--why", default="",
                        help="one sentence for _this_recut: what moved and why the "
                             "re-run was owed")
    parser.add_argument("--only", default="",
                        help="comma-separated entry keys to re-cut (default: all "
                             "of LEGS). The ledger write is all-or-nothing over the "
                             "SELECTED entries, so naming them is how an operator "
                             "re-cuts the welds a run covers without being blocked "
                             "by one it does not")
    parser.add_argument("--supersede", default="",
                        help="comma-separated compute capabilities whose records this "
                             "write may leave behind. Needed ONLY when the run moves "
                             "the bytes the entry binds: those architectures' records "
                             "then certify bytes that no longer ship, and the refusal "
                             "names them. Re-running them on the same tree is the "
                             "other way out, and the one that keeps them admitted")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)
    supersede = tuple(part for part in args.supersede.split(",") if part)

    if args.family_recert:
        if args.only:
            raise SystemExit("--only selects among LEGS; --family-recert writes one "
                             "entry and takes its selection from the roots")
        return family_recert(list(args.family_recert), args.why, supersede, args.write)
    return recut_run(args.run, args.why, args.only, supersede, args.write)


def recut_run(run_rel: str, why: str, only: str, supersede: Sequence[str],
              write: bool) -> int:
    """Re-cut the composition welds a single composition or fleet run covers."""
    fastpath = _fastpath()
    run = (HERE / run_rel).resolve()
    if not run.is_dir():
        raise SystemExit(f"no such run directory: {run}")
    staged = read_staged(run)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    problems: List[str] = []
    updates: Dict[str, dict] = {}
    capabilities: Dict[str, str] = {}
    selected = set(only.split(",")) if only else set(LEGS)
    unknown = selected - set(LEGS)
    if unknown:
        raise SystemExit(f"--only names {sorted(unknown)}, which this tool does not cover")
    for key, (leg_names, probe_rel) in LEGS.items():
        if key not in selected:
            continue
        # WHERE THIS KEY'S PROBLEMS START. Until 2026-08-30 the success line below
        # printed unconditionally, so a run whose curated claim had gone stale
        # reported "curated blocks agree with the fresh summary" on one line and
        # the disagreement on another. A tool that says both is a tool whose
        # cheerful line stops being read.
        mark = len(problems)
        entry = ledger.get(key)
        if entry is None:
            problems.append(f"{key}: absent from the ledger")
            continue
        # BOTH RULES, and the narrower one is not the authority. ``SLOT_FIELDS`` is what
        # THIS tool writes, so it catches a field this writer stranded; the engine's
        # default ``RUN_FIELDS`` is what the CONTAINER means by a run fact, and the two
        # differ by the fields other writers own. If a stranded fact named in
        # CHALLENGE #7 (``versions``, ``run_id``, ``throughput``) is later added to
        # RUN_FIELDS, a check that only asked about this tool's set would bind a record
        # beside an entry the engine still reads as retired.
        retired = (fastpath.retired_shape_reasons(entry, run_fields=sorted(SLOT_FIELDS))
                   + fastpath.retired_shape_reasons(entry))
        if retired:
            problems.append(f"{key}: {'; '.join(retired)}. This record is still in the "
                            f"pre-2026-09-30 shape, with one run's facts beside the "
                            f"digests; migrate_capability_records.py moves them into "
                            f"runs[<capability>] and this tool writes nowhere else")
            continue
        directory, why_not = leg_directory(run, leg_names)
        if directory is None:
            problems.append(f"{key}: {why_not}")
            continue
        leg = directory.name
        artifact = directory / "gate.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        verdict, read_from = released(payload)
        if verdict is not True:
            problems.append(f"{key}: the fresh leg reports released={verdict!r} "
                            f"(read from {read_from}); nothing is rebound from a "
                            f"run that did not release")
            continue

        # WHICH ARCHITECTURE THIS LEG RAN ON, read from the run. The legs must agree:
        # one re-cut describes one campaign, and a record keyed by whichever leg was
        # read last would be a claim about a device nobody chose.
        capability, identity = leg_capability(payload, [directory, run])
        if capability is None:
            problems.append(f"{key}: {identity}. Stamp the run with '"
                            + DEVICE_STAMP_HINT.format(directory=f"{run_rel}/{leg}")
                            + "' in its own environment.")
            continue
        capabilities[key] = capability

        # EVERY FILE THIS ENTRY PINS MUST BE IN THE RUN'S OWN MANIFEST *AND* MATCH THE
        # TREE. Missing from the manifest = the run cannot speak for it. Different
        # from the tree = the run measured bytes that have since moved on.
        manifest, digest_source = leg_digests(payload, staged, directory)
        if manifest is None:
            problems.append(f"{key}: {digest_source}")
            continue
        fresh_sources: Dict[str, str] = {}
        for name in entry["source_sha256"]:
            relative = pinned_path(name)
            if relative not in manifest:
                problems.append(f"{key}: {name} ({relative}) is pinned by the record "
                                f"but absent from {digest_source}")
                continue
            live_path = API / relative
            if not live_path.is_file():
                problems.append(f"{key}: {name} pinned but not found in the tree")
                continue
            live = sha256_of(live_path)
            if live != manifest[relative]:
                problems.append(
                    f"{key}: {relative} ran as {manifest[relative][:12]} per "
                    f"{digest_source} but the tree holds {live[:12]}; the run and the "
                    f"tree disagree and this record may not be cut against either")
                continue
            fresh_sources[name] = live

        host, why_not = derived_host(identity)
        if host is None:
            problems.append(f"{key}: {why_not}")
            continue

        probe = API / probe_rel
        # A KEY THE RECORD DOES NOT CARRY IS NOT A KEY TO ADD. Which digests an
        # entry pins is a claim someone made; fused_electric_gate's record pins no
        # artifact digest, and inventing one here would widen its weld under cover of
        # a re-cut. The reference set for the RUN facts is every field the entry's
        # existing records carry, so a second architecture's record pins exactly what
        # the first one did -- and a pin this run cannot produce refuses the entry
        # rather than being dropped from it.
        owned = {name for record in entry[fastpath.RUNS].values()
                 if isinstance(record, dict) for name in record} - {"bound_sha256"}
        logs = [directory / name for name in ("gate.log", "run.log")
                if (directory / name).is_file()]
        run_facts: Dict[str, Any] = {
            "host": host,
            "records": f"apps/api/parity/meep_gpu/{run_rel}/{leg}/gate.json",
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "_this_recut": (why or "re-cut from a composition re-run") +
                           f" Digests READ from {run_rel}/{leg}/ via {digest_source} "
                           f"and cross-checked against the live tree by "
                           f"parity/meep_gpu/recut_composition_records.py, which "
                           f"refuses unless the run and the tree agree on every pinned "
                           f"file. Bound to compute capability {capability} from the "
                           f"run's own identity.",
        }
        produced = {"artifact_sha256": sha256_of(artifact),
                    "log_sha256": sha256_of(logs[0]) if logs else None}
        for pin in SLOT_PINS:
            value = produced[pin]
            if pin not in owned:
                continue
            if value is None:
                problems.append(
                    f"{key}: the record pins {pin} and {leg}/ holds neither gate.log "
                    f"nor run.log, so this run cannot produce it. Carrying the "
                    f"previous run's digest would put another run's bytes in this "
                    f"one's record")
                continue
            run_facts[pin] = value
        elapsed = derived_elapsed(payload)
        if elapsed is not None:
            run_facts["elapsed"] = elapsed
        elif "elapsed" in owned:
            print(f"    NOTE {key}: {leg}/gate.json states no elapsed time, so this "
                  f"record carries none", flush=True)
        # ``device_policy`` is NOT carried from the previous record, for the reason
        # ``rebind_triton_welds.py`` gives for not carrying it either: it reads "one
        # Slurm-assigned device (CUDA_VISIBLE_DEVICES=0), verified empty before
        # import ..." -- a description of how THAT run was hosted, which a re-run on
        # another host or scheduler would inherit as a false account of its own. It
        # used to be carried unconditionally, even from a record bound to other bytes,
        # and that also made the two binding orders of one commit disagree.

        updates[key] = {
            "entry": {name: value for name, value in
                      (("probe_sha256", sha256_of(probe)),
                       ("source_sha256", fresh_sources))
                      if name in entry or name == "source_sha256"},
            "run": run_facts,
            "claims": {},
        }
        # THE CURATED CLAIM MUST STILL BE TRUE OF THE FRESH RUN. These blocks are
        # kept verbatim (they are more specific than the artifact's summary), so the
        # run is checked against them rather than the other way round: every counter
        # the two share must agree, and the record's extra prose is allowed to be
        # longer. A block that no longer describes the run REFUSES the entry unless
        # the re-cut is DECLARED in CLAIM_RECUTS -- it never gets quietly
        # overwritten, because what a gate measured is a measurement and re-cutting
        # it is a decision someone has to make, in that table, with the reason.
        # dispersive_fused_pair's 36/36 -> 48/48 is the one declared today.
        recut_notes = []
        for block, field, recorded, measured in claim_disagreements(entry, payload):
            if (key, block) not in CLAIM_RECUTS:
                problems.append(
                    f"{key}: {block}[{field}]={recorded!r} but the fresh run "
                    f"measured {measured!r}; the curated claim no longer describes "
                    f"the run and may not be carried forward. If the run is right "
                    f"and the claim is stale, declare ({key!r}, {block!r}) in "
                    f"CLAIM_RECUTS with what moved.")
                continue
            if block in updates.get(key, {}).get("claims", {}):
                continue    # already re-cut on an earlier field of the same block
            fresh_block, moved, why = recut_claim(entry, block, payload)
            if fresh_block is None:
                problems.append(f"{key}: {why}")
                continue
            # THE POST-CONDITION, asserted rather than assumed: after the re-cut
            # the block must no longer disagree with the run on ANY counter. A
            # re-cut that fixed the field that tripped the check and left another
            # stale would be the same defect wearing this table as cover.
            after = dict(entry, **{block: fresh_block})
            remaining = claim_disagreements(after, payload, block=block)
            if remaining:
                problems.append(f"{key}: the re-cut of {block} still disagrees with "
                                f"the run on {[r[1] for r in remaining]}")
                continue
            updates[key]["claims"][block] = fresh_block
            recut_notes.extend(moved)
        if len(problems) > mark:
            continue
        if recut_notes:
            # WHAT MOVED GOES INTO THE RECORD, not only into this tool's stdout.
            # A curated measurement that changes without the record saying so is
            # the stale-claim defect with a fresh timestamp on it.
            updates[key]["run"]["_this_recut"] += (
                " CURATED CLAIM RE-CUT FROM THIS RUN, declared in "
                "recut_composition_records.CLAIM_RECUTS: "
                + "; ".join(sorted(
                    CLAIM_RECUTS[(key, block)]
                    for block in sorted(updates[key]["claims"])
                    if (key, block) in CLAIM_RECUTS))
                + " Fields: " + "; ".join(recut_notes) + ".")
        for note in recut_notes:
            print(f"    RE-CUT {note}", flush=True)
        print(f"  {key}: released (read from {read_from}) on cc {capability}, "
              f"{len(fresh_sources)} sources verified run==tree via {digest_source}, "
              f"curated blocks "
              + ("re-cut from the fresh summary" if recut_notes
                 else "agree with the fresh summary"), flush=True)

    # ONE RUN, ONE ARCHITECTURE. Legs that disagree are two campaigns in one
    # directory, and a record keyed by either of them would describe bytes the other
    # leg never ran on.
    if len(set(capabilities.values())) > 1:
        problems.append(
            "the legs disagree about the compute capability they ran on: "
            + ", ".join(f"{key}={cc}" for key, cc in sorted(capabilities.items()))
            + ". One re-cut describes one campaign on one device")

    if problems:
        print("\nREFUSED, nothing written:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        return 3

    # --- host_sha256, licensed BY THESE SAME RUNS -------------------------------
    #
    # ``host_sha256`` holds ten host files the bit-identity gate certified, and
    # ``test_the_fingerprint_record_matches_the_shipped_host_side`` refuses any drift
    # its shared rule cannot clear. The 2026-08-28 precedent
    # (``_host_sha256_recut_2026_08_28``) is the only sanctioned way to move one: the
    # composition gates are RE-RUN against the drifted bytes and the entry is re-cut
    # from that. This does the same thing, with the licence checked rather than
    # asserted -- a file may only be re-cut here if BOTH legs above passed and the
    # legs staged exactly the bytes the tree holds.
    #
    # ONLY A CAMPAIGN-WIDE MANIFEST LICENSES IT, and that is why the per-leg import
    # fallback is not read here. ``host_sha256`` is a claim about the WHOLE host side,
    # licensed by the composition legs together; a leg's ``imported_source_sha256`` is
    # what that one probe imported, and the two legs that historically carried this
    # licence (cylindrical, symmetry) record no import set at all. A fleet-driven round
    # therefore cannot move ``host_sha256``: where a host file has drifted it SAYS SO
    # and leaves the pin, because this block binds no weld and refusing on it would
    # refuse every fleet-driven run (4 of 10 host files have drifted on today's tree,
    # which is what ``pending_host_recut`` records). The run that may move it is one
    # whose launcher staged the tree.
    host_updates: Dict[str, str] = {}
    host = ledger.get("host_sha256") or {}
    for name, digest in host.items():
        # THE HOST BLOCK IS TRITON_KERNELS AND NOTHING ELSE -- it pins the shipped
        # host side of this backend -- so its path is stated rather than searched
        # for. The staged lookup uses the SAME repo-relative path, so the manifest
        # row consulted is the file being hashed and not a namesake elsewhere in
        # the tree.
        relative = f"meep_gpu/triton_kernels/{name}"
        path = API / relative
        live = sha256_of(path)
        if live == digest:
            continue
        if staged is None:
            # REPORTED, NOT REFUSED. A leg's own import set licenses nothing about the
            # shipped host side, so this run may not MOVE the pin -- but it binds no
            # weld either, and refusing here would refuse every run a fleet driver
            # produces (4 of 10 host files have drifted on today's tree, and the
            # pending_host_recut block is what records that). The drift and that block
            # stay for a launcher-staged run to clear; this run simply leaves them.
            print(f"    NOTE host_sha256[{name}] has drifted and this run wrote no "
                  f"staged_source_sha256.txt, so the pin is left where it is; a "
                  f"composition run whose launcher staged the tree is what moves it",
                  flush=True)
            continue
        if relative not in staged:
            problems.append(
                f"host_sha256[{name}] has drifted but the composition run did not "
                f"stage it, so these runs license nothing about it")
            continue
        if staged[relative] != live:
            problems.append(
                f"host_sha256[{name}]: the run staged {staged[relative][:12]} and "
                f"the tree holds {live[:12]}")
            continue
        host_updates[name] = live
        print(f"  host_sha256[{name}]: {digest[:12]} -> {live[:12]}, licensed by "
              f"both composition legs", flush=True)

    if problems:
        print("\nREFUSED, nothing written:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        return 3

    # THE BIND RUNS IN BOTH MODES, on a copy when reporting. A write that would strand
    # another architecture's record is only discoverable with both bounds in hand, and
    # discovering it after --write would mean discovering it in the ledger.
    capability = next(iter(set(capabilities.values())), "")
    target = ledger if write else json.loads(json.dumps(ledger))
    notes, bind_problems = apply_updates(target, updates, capability, supersede)
    for note in notes:
        print(f"  {note}", flush=True)
    if bind_problems:
        print("\nREFUSED, nothing written:", flush=True)
        for problem in bind_problems:
            print(f"  {problem}", flush=True)
        return 3

    if not write:
        print(f"\n  (report only -- pass --write to apply; {len(updates)} records "
              f"would bind cc {capability})", flush=True)
        return 0

    if host_updates:
        stamp = f"_host_sha256_recut_{datetime.now(timezone.utc):%Y_%m_%d}"
        ledger["host_sha256"].update(host_updates)
        ledger[stamp] = (
            ", ".join(f"{n} -> {d[:12]}..." for n, d in sorted(host_updates.items()))
            + f". Licensed by the composition re-runs recorded at "
              f"apps/api/parity/meep_gpu/{run_rel}/, both on the GPU host against these "
              f"exact digests -- the same mechanism as _host_sha256_recut_2026_08_28. "
              f"Digests READ from the tree only after this tool verified the runs "
              f"staged the same bytes; kernel_source_sha256 is untouched, so no "
              f"device source changed. " + (why or ""))
        # The pending-recut declaration is about drift AWAITING a device re-cut. This
        # IS that re-cut, so a file it names must not still be declared: a
        # declaration that outlives its drift fails the C19 test in the other
        # direction, and that is the direction that quietly accumulates.
        pending = (ledger.get("pending_host_recut") or {}).get("files")
        if isinstance(pending, dict):
            for name in host_updates:
                pending.pop(name, None)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"\n  wrote {LEDGER}", flush=True)
    return 0


def campaign_row(root: Path, gate: str) -> Tuple[Optional[dict], str]:
    """The campaign artifact's row for one gate, or a named refusal.

    The row is where the EXIT CODE lives: a gate artifact records its own verdict and
    not the status its process exited with, and a released artifact beside a non-zero
    exit is a contradiction this tool must not transcribe. ``drive_triton_weld_gates``
    writes one row per gate under ``rows``, carrying ``exit_code``, ``released``,
    ``artifact_sha256``, ``started_utc`` and the pinned ``gpu``.
    """
    manifest = root / "campaign.json"
    if not manifest.is_file():
        return None, (f"{root.name}/campaign.json is missing, so nothing in this run "
                      f"states the exit code its gate process returned")
    rows = json.loads(manifest.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list):
        return None, f"{root.name}/campaign.json carries no rows list"
    for row in rows:
        if isinstance(row, dict) and row.get("gate") == gate:
            return row, ""
    return None, (f"{root.name}/campaign.json has no row for {gate!r}, so its gate "
                  f"artifact is there and its run is not accounted for")


def _step_budget(payload: dict) -> Tuple[Any, Optional[str]]:
    """The budget the artifact STATES, and where, or :data:`BUDGET_NOT_STATED`.

    The three places are the three MEASURED ones (see :data:`BUDGET_NOT_STATED`), read
    in that order and never searched for recursively: a deep search would pick up any
    block that happens to carry the word and record it as the gate's claim.
    """
    for trail in (("step_budgets",), ("step_budget",), ("provenance", "step_budget")):
        node: Any = payload
        for part in trail:
            node = node.get(part) if isinstance(node, dict) else None
        if node not in (None, {}, "", []):
            return node, ".".join(trail)
    return BUDGET_NOT_STATED, None


def family_recert(root_rels: List[str], why: str, supersede: Sequence[str],
                  write: bool) -> int:
    """Rebuild ``family_recert_2026-08-14``'s nine family records FROM A FLEET CAMPAIGN.

    WHY THIS MODE EXISTS. Those nine records were transcribed from the 2026-08-14
    launcher's logs, which no longer run, and 22 Triton arms cite the entry -- so under
    per-capability records it was the one cited key nothing could re-cut, and a key with
    no live record for an architecture is a key that keeps the whole table from
    admitting it. The nine families are nine gate names in the SHIPPED fleet driver, so
    the evidence a round already produces is enough to rebuild the record.

    WHAT IT REFUSES. A family whose artifact is missing or did not release; a family
    whose campaign row is missing or exited non-zero, or whose row names a different
    artifact digest than the file holds; a run whose imported digests differ from the
    tree; two families disagreeing about the compute capability or the host; two
    families recording different digests for one file; and a union that does not cover
    every file the entry declares, because the FILE SET is the invariant
    ``test_the_family_modules_drift_declaration_matches_the_shipped_bytes`` enforces and
    a narrower map would shrink a 22-arm weld's coverage without any digest moving.
    """
    fastpath = _fastpath()
    from seed_triton_welds import PolicyStampConflict, policy_line  # noqa: PLC0415

    key = fastpath.FAMILY_RECERT_GATE
    roots: List[Tuple[str, Path]] = []
    for rel in root_rels:
        root = (HERE / rel).resolve()
        if not root.is_dir():
            raise SystemExit(f"no such campaign root: {root}")
        roots.append((rel.rstrip("/"), root))

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = ledger.get(key)
    if not isinstance(entry, dict):
        raise SystemExit(f"{key} is absent from the ledger")
    # Both rules, for the reason :func:`recut_run` states: this tool's own set and the
    # engine's definition of a run fact, so a narrower check cannot license a write.
    retired = (fastpath.retired_shape_reasons(entry,
                                              run_fields=sorted(FAMILY_SLOT_FIELDS))
               + fastpath.retired_shape_reasons(entry))
    if retired:
        raise SystemExit(f"{key}: {'; '.join(retired)}. This record is still in the "
                         f"pre-2026-09-30 shape; migrate_capability_records.py moves "
                         f"its run facts into runs[<capability>] first")
    # THE TYPED TUPLE IS CHECKED AGAINST THE RECORD, not trusted. A family dropped from
    # the tuple would quietly narrow the coverage of a 22-arm weld with every digest
    # still verifying, which is the failure mode this whole file is built around.
    declared = sorted(FAMILY_RECERT_FAMILIES)
    for capability, record in sorted(entry[fastpath.RUNS].items()):
        recorded = sorted(record.get("families") or {})
        if recorded != declared:
            raise SystemExit(
                f"{key}: runs[{capability}].families names {recorded} and "
                f"FAMILY_RECERT_FAMILIES names {declared}; the record and this tool "
                f"disagree about which families the entry certifies")
    pinned = list(entry.get("source_sha256_at_recert") or {})
    if not pinned:
        raise SystemExit(f"{key} declares no source_sha256_at_recert, so it binds no "
                         f"bytes and a record written here could never go stale")

    problems: List[str] = []
    families: Dict[str, dict] = {}
    hosts: Dict[str, str] = {}
    capabilities: Dict[str, str] = {}
    union: Dict[str, str] = {}
    witness: Dict[str, str] = {}
    for family in FAMILY_RECERT_FAMILIES:
        # THE FIRST ROOT HOLDING A RELEASED ARTIFACT WINS -- not the first holding a
        # directory, and not the first holding an artifact. The 09-25 fleet's
        # nonlinear leg released=False while the 09-23 fleet's released: picking by
        # directory would bind the record to the run that refused, and reporting
        # "missing" would hide a run that is there and did not pass.
        chosen: Optional[Tuple[str, Path, Path, dict, str]] = None
        looked: List[str] = []
        for rel, root in roots:
            artifact = root / family / "gate.json"
            if not artifact.is_file():
                looked.append(f"{rel} holds no {family}/gate.json")
                continue
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            verdict, read_from = released(payload)
            if verdict is not True:
                looked.append(f"{rel} reports released={verdict!r} "
                              f"(read from {read_from})")
                continue
            chosen = (rel, root, artifact, payload, read_from)
            break
        if chosen is None:
            problems.append(f"{family}: no root holds a released gate artifact -- "
                            + "; ".join(looked))
            continue
        rel, root, artifact, payload, read_from = chosen

        row, why_not = campaign_row(root, family)
        if row is None:
            problems.append(f"{family}: {why_not}")
            continue
        if row.get("exit_code") != 0:
            problems.append(f"{family}: {rel}'s campaign row exited "
                            f"{row.get('exit_code')!r} beside a released artifact; "
                            f"a verdict from a process that failed is not a verdict")
            continue
        if row.get("released") is not True:
            problems.append(f"{family}: {rel}'s campaign row reads "
                            f"released={row.get('released')!r} and the artifact reads "
                            f"released=True ({read_from}); the driver and the gate "
                            f"disagree about the same run")
            continue
        digest = sha256_of(artifact)
        if row.get("artifact_sha256") not in (None, digest):
            problems.append(f"{family}: {rel}'s campaign row names artifact "
                            f"{str(row.get('artifact_sha256'))[:12]} and the file "
                            f"holds {digest[:12]}; the artifact moved after the run")
            continue
        log = root / family / "gate.log"
        if not log.is_file():
            problems.append(f"{family}: {rel}/{family}/ holds no gate.log, and the "
                            f"record pins the log it was cut from")
            continue

        imported = payload.get("imported_source_sha256") or {}
        if not imported:
            problems.append(f"{family}: {rel}/{family}/gate.json records no "
                            f"imported_source_sha256, so the run says nothing about "
                            f"which bytes it executed")
            continue
        # THE RUN'S DIGESTS MUST BE THE TREE'S. Everything the gate imported is
        # compared, not only the thirteen files this entry declares: a recert states
        # "these were the bytes at recert", and a run that executed any other version
        # of the package cannot support that sentence for the files it happens to share.
        drifted = sorted(name for name, recorded in imported.items()
                         if (API / name).is_file()
                         and sha256_of(API / name) != recorded)
        if drifted:
            problems.append(
                f"{family}: {len(drifted)} of the {len(imported)} files this run "
                f"imported have moved in the tree since ({drifted[:3]}); a recert cut "
                f"against them would declare digests the shipped bytes do not hold")
            continue
        at_recert = {name: imported[pinned_path(name)] for name in pinned
                     if pinned_path(name) in imported}
        if not at_recert:
            problems.append(f"{family}: the run imported none of the files this entry "
                            f"declares, so it contributes no digest to the recert")
            continue
        for name, recorded in sorted(at_recert.items()):
            if name in union and union[name] != recorded:
                problems.append(
                    f"{name}: {witness[name]} imported {union[name][:12]} and "
                    f"{family} imported {recorded[:12]}; one declaration cannot hold "
                    f"two digests for one file")
            union.setdefault(name, recorded)
            witness.setdefault(name, family)

        capability, identity = leg_capability(payload, [root / family, root])
        if capability is None:
            problems.append(f"{family}: {identity}. Stamp the campaign with '"
                            + DEVICE_STAMP_HINT.format(directory=rel)
                            + "' in its own environment.")
            continue
        host, why_not = derived_host(identity)
        if host is None:
            problems.append(f"{family}: {why_not}")
            continue
        capabilities[family] = capability
        hosts[family] = host

        budget, budget_from = _step_budget(payload)
        # A CONTRADICTION IS NOT AN ABSENCE. ``None`` means the artifact states no
        # attained policy and is recorded as POLICY_NOT_STATED; spellings that disagree
        # are a defect in the run's own record and stop the re-cut by name.
        try:
            policy = policy_line(payload)
        except PolicyStampConflict as exc:
            problems.append(f"{family}: policy refused: {exc}")
            continue
        record = {
            "artifact_sha256": digest,
            "certified_under_subnormal_policy": policy or POLICY_NOT_STATED,
            "log": f"parity/meep_gpu/{rel}/{family}/gate.log",
            "log_sha256": sha256_of(log),
            "rc": row["exit_code"],
            "records": f"parity/meep_gpu/{rel}/{family}/gate.json",
            "run_id": root.name,
            "source_sha256_at_recert": {name: at_recert[name]
                                        for name in sorted(at_recert)},
            "step_budget": budget,
            "verdict": payload.get("canonical_verdict"),
        }
        # Recorded where the campaign states them, omitted where it does not: the
        # 2026-08-14 records carried both and a blank beside a real one reads as a
        # measurement that came out empty.
        if row.get("started_utc"):
            record["started_utc"] = row["started_utc"]
        if row.get("gpu") is not None:
            record["pinned_gpu_index"] = row["gpu"]
        families[family] = {name: record[name] for name in sorted(record)}
        print(f"  {family}: released (read from {read_from}) on cc {capability} from "
              f"{rel}, {len(at_recert)} declared digests, step budget "
              + (f"read from {budget_from}" if budget_from else "NOT STATED"),
              flush=True)

    missing = [name for name in pinned if name not in union]
    if missing:
        problems.append(
            f"the nine families' import sets cover {len(union)} of the {len(pinned)} "
            f"files this entry declares, missing {missing}. The FILE SET is the "
            f"invariant the drift test enforces, and a narrower declaration shrinks "
            f"the coverage of every arm citing this weld")
    if len(set(capabilities.values())) > 1:
        problems.append("the families disagree about the compute capability: "
                        + ", ".join(f"{name}={cc}"
                                    for name, cc in sorted(capabilities.items())))
    if len(set(hosts.values())) > 1:
        problems.append("the families disagree about the host they ran on: "
                        + "; ".join(f"{name}: {line}"
                                    for name, line in sorted(hosts.items())))
    if problems:
        print("\nREFUSED, nothing written:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        return 3

    capability = next(iter(set(capabilities.values())))
    named = ", ".join(f"apps/api/parity/meep_gpu/{rel}" for rel, _ in roots)
    moved: Dict[str, Any] = {
        "source_sha256_at_recert": {name: union[name] for name in pinned},
    }
    # THE DRIFT DECLARATION IS ABOUT BYTES THAT MOVED SINCE THE RECERT. This IS the
    # recert, and every family's imported digests were just proved equal to the tree,
    # so the declaration is empty by measurement. Its previous contents describe a
    # state that is over and move into a dated log beside it; the walker reads a
    # ``*_recut_log`` key as history, which is what they now are.
    #
    # ONLY WHEN THERE IS SOMETHING TO MOVE, and that is the additive protocol rather
    # than tidiness: an architecture X is certified by re-running the same tree AFTER
    # the 8.6 re-cut, usually the same day, and the second run finds the declaration
    # already empty. Writing a log unconditionally made that second bind refuse on its
    # own dated key -- measured, and the one case the whole mode exists to serve. A
    # same-day key beside a NON-empty declaration is still refused: that is two
    # different drifts cleared in one day, and the second would overwrite the first's
    # evidence.
    declaration = entry.get("source_drift_since_recert") or {}
    if declaration:
        stamp = f"family_recert_{datetime.now(timezone.utc):%Y_%m_%d}_recut_log"
        if stamp in entry:
            raise SystemExit(f"{key} already carries {stamp} and still declares drift "
                             f"on {sorted(declaration)}; a second log on the same day "
                             f"would overwrite the first one's evidence")
        moved["source_drift_since_recert"] = {}
        moved[stamp] = {
            "superseded_declaration": declaration,
            "why": (f"the nine family gates were re-run on the shipped fleet and "
                    f"re-cut from their own artifacts by "
                    f"parity/meep_gpu/recut_composition_records.py --family-recert "
                    f"({named}), which proved every imported digest equal to the tree "
                    f"before writing. The drift the superseded declaration described "
                    f"is what that re-run cleared. source_drift_note is left as "
                    f"authored: it explains how the declaration is read, not what it "
                    f"contained."),
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    run_facts = {
        "families": {name: families[name] for name in sorted(families)},
        "host": next(iter(set(hosts.values()))),
        "records": (f"{named} -- nine family gate artifacts, one root per family; each "
                    f"family's own root is named in families[<family>].records"),
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_this_recut": ((why or "re-cut from a fleet re-run of the nine family gates")
                        + f" READ from the artifacts under {named} by "
                          f"parity/meep_gpu/recut_composition_records.py "
                          f"--family-recert, which refuses a family that did not "
                          f"release, a campaign row that exited non-zero, a run whose "
                          f"imported digests differ from the tree, and a declaration "
                          f"that would not cover every file the entry names. Bound to "
                          f"compute capability {capability} from the runs' own "
                          f"identity."),
    }
    updates = {key: {"entry": moved, "run": run_facts, "claims": {},
                     "moves": tuple(moved),
                     "run_fields": sorted(FAMILY_SLOT_FIELDS)}}
    target = ledger if write else json.loads(json.dumps(ledger))
    notes, bind_problems = apply_updates(target, updates, capability, supersede)
    for note in notes:
        print(f"  {note}", flush=True)
    if bind_problems:
        print("\nREFUSED, nothing written:", flush=True)
        for problem in bind_problems:
            print(f"  {problem}", flush=True)
        return 3
    print(f"\n  {key}: {len(families)} families, {len(union)} declared digests, "
          f"cc {capability}", flush=True)
    if not write:
        print("\n  (report only -- pass --write to apply)", flush=True)
        return 0
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"  wrote {LEDGER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
