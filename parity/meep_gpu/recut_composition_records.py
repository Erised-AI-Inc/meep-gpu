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
* the run's ``staged_source_sha256.txt`` must agree with the LIVE tree on every file
  the record pins -- a record cut against bytes that have already moved on is the
  thing this tool exists to prevent, not a state it may write;
* every key the existing record pins must be present in the staged manifest, so the
  rewrite can never QUIETLY drop a file from a weld's coverage;
* the probe and launcher digests are read from the tree and cross-checked against the
  ones the run reported staging.

Any of those failing prints the disagreement and exits non-zero with nothing written.

WHAT IT DELIBERATELY LEAVES ALONE. Every hand-authored key -- ``purpose``,
``dispatch``, ``fusion_boundary``, ``device_policy``, the ``recert_*`` blocks, the
``product`` measurement -- is a claim someone made about what the gate MEANS, and is
asserted byte-identical after the rewrite. Only the digests, the artifact and log
pointers, ``records``, ``recorded_utc``, ``elapsed`` and ``_this_recut`` move.

    python recut_composition_records.py --run results/triton_composition_2026-08-30_carry
    python recut_composition_records.py --run ... --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "meep_gpu" / "triton_kernels").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


API = find_api_root(Path(__file__).resolve())
HERE = API / "parity" / "meep_gpu"
LEDGER = API / "meep_gpu" / "triton_kernels" / "fingerprints.json"

#: entry key -> (leg directory, probe path, launcher path or None)
#:
#: THE SEVEN BELOW ARE THE COMPOSITION AND FUSED-PROBE WELDS, and the five added
#: 2026-08-30 are the entries the ``status == "PASS"`` filter used to hide. Each
#: carries a full device record and a set of pinned source digests, and none is in
#: ``rebind_triton_welds.py``'s ``CAMPAIGN_DIRS`` -- that table enumerates the 31
#: PASS welds only, so before this round nothing could re-cut these at all and the
#: only way to move one was by hand. Every path here is read off the entry's own
#: ``probe``/``slurm_launcher`` keys, not derived from the leg name.
#:
#: ``fused_electric_gate`` pins NO launcher: it was run directly rather than
#: through Slurm, and its entry has no ``slurm_launcher_sha256`` key. ``None``
#: means "this entry pins no launcher", and the rewrite must not invent one --
#: see the key-set filter in :func:`main`.
LEGS = {
    "cylindrical_composition_gate": (
        "cylindrical_composition",
        "parity/meep_gpu/probe_triton_cylindrical_composition.py",
        "parity/meep_gpu/run_triton_cylindrical_composition.slurm"),
    "symmetry_composition_gate": (
        "symmetry_composition",
        "parity/meep_gpu/probe_triton_symmetry_composition.py",
        "parity/meep_gpu/run_triton_symmetry_composition.slurm"),
    "conductivity_composition_gate": (
        "conductivity_composition",
        "parity/meep_gpu/probe_triton_conductivity_composition.py",
        "parity/meep_gpu/run_triton_conductivity_composition.slurm"),
    "dispersive_composition_gate": (
        "dispersive_composition",
        "parity/meep_gpu/probe_triton_engine_route.py",
        "parity/meep_gpu/run_triton_dispersive_composition.slurm"),
    "dispersive_fused_pair_gate": (
        "dispersive_fused_pair",
        "parity/meep_gpu/probe_triton_dispersive_fused_pair.py",
        "parity/meep_gpu/run_triton_dispersive_fused_pair.slurm"),
    "fused_ade_state_gate": (
        "fused_ade_state",
        "parity/meep_gpu/probe_triton_fused_ade_state.py",
        "parity/meep_gpu/run_triton_fused_ade_state.slurm"),
    "source_seam_gate": (
        "source_seams",
        "parity/meep_gpu/probe_triton_source_seams.py",
        "parity/meep_gpu/run_triton_source_seams.slurm"),
    "no_pml_composition_gate": (
        "no_pml_composition",
        "parity/meep_gpu/gate_triton_no_pml.py",
        "parity/meep_gpu/run_triton_no_pml_composition.slurm"),
    "fused_electric_gate": (
        "fused_electric",
        "parity/meep_gpu/gate_triton_fused_electric.py",
        None),
}

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
    # `replaces` list, and in 2026-08-11's job 2298 the four non-PML-curl cases
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
    # NOTHING on that case; the conductive family landed hours later, on job 2317,
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
MUTABLE = {"artifact_sha256", "log_sha256", "probe_sha256", "slurm_launcher_sha256",
           "source_sha256", "records", "recorded_utc", "elapsed", "_this_recut"}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_staged(run: Path) -> Dict[str, str]:
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
        raise SystemExit(f"{manifest} is missing: the run recorded no staged hashes, "
                         f"so there is nothing to re-cut FROM")
    out: Dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True,
                        help="composition run directory, parity/meep_gpu "
                             "relative (e.g. results/triton_composition_2026-08-30_carry)")
    parser.add_argument("--why", default="",
                        help="one sentence for _this_recut: what moved and why the "
                             "re-run was owed")
    parser.add_argument("--only", default="",
                        help="comma-separated entry keys to re-cut (default: all "
                             "of LEGS). The ledger write is all-or-nothing over the "
                             "SELECTED entries, so naming them is how an operator "
                             "re-cuts the welds a run covers without being blocked "
                             "by one it does not")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)

    run = (HERE / args.run).resolve()
    if not run.is_dir():
        raise SystemExit(f"no such run directory: {run}")
    staged = read_staged(run)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    problems, updates = [], {}
    selected = set(args.only.split(",")) if args.only else set(LEGS)
    unknown = selected - set(LEGS)
    if unknown:
        raise SystemExit(f"--only names {sorted(unknown)}, which this tool does not cover")
    for key, (leg, probe_rel, launcher_rel) in LEGS.items():
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
        artifact = run / leg / "gate.json"
        log = run / leg / "run.log"
        if not artifact.is_file():
            problems.append(f"{key}: {artifact} is missing")
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        verdict, read_from = released(payload)
        if verdict is not True:
            problems.append(f"{key}: the fresh leg reports released={verdict!r} "
                            f"(read from {read_from}); nothing is rebound from a "
                            f"run that did not release")
            continue

        # EVERY FILE THIS ENTRY PINS MUST BE IN THE RUN'S MANIFEST *AND* MATCH THE
        # TREE. Missing from the manifest = the run cannot speak for it. Different
        # from the tree = the run measured bytes that have since moved on.
        fresh_sources: Dict[str, str] = {}
        for name in entry["source_sha256"]:
            relative = pinned_path(name)
            if relative not in staged:
                problems.append(f"{key}: {name} ({relative}) is pinned by the record "
                                f"but absent from the run's staged manifest")
                continue
            live_path = API / relative
            if not live_path.is_file():
                problems.append(f"{key}: {name} pinned but not found in the tree")
                continue
            live = sha256_of(live_path)
            if live != staged[relative]:
                problems.append(
                    f"{key}: {relative} staged {staged[relative][:12]} but the tree "
                    f"holds {live[:12]}; the run and the tree disagree and this "
                    f"record may not be cut against either")
                continue
            fresh_sources[name] = live

        probe = API / probe_rel
        # A KEY THE ENTRY DOES NOT CARRY IS NOT A KEY TO ADD. Which digests an
        # entry pins is a claim someone made; fused_electric_gate pins no launcher
        # and no artifact digest, and inventing them here would widen its weld
        # under cover of a re-cut. Built in full, then filtered to what is there.
        candidate = {
            "artifact_sha256": sha256_of(artifact),
            "log_sha256": sha256_of(log) if log.is_file() else entry.get("log_sha256"),
            "probe_sha256": sha256_of(probe),
            "source_sha256": fresh_sources,
        }
        # A TREE WITHOUT THE SITE LAUNCHERS (the public repository does not carry
        # them) cannot digest one. The entry's recorded launcher digest is then
        # carried forward unchanged, and the re-cut says so in ``_this_recut``,
        # rather than the tool failing on a file this tree never held.
        launcher_absent = launcher_rel is not None and not (API / launcher_rel).is_file()
        if launcher_rel is not None and not launcher_absent:
            candidate["slurm_launcher_sha256"] = sha256_of(API / launcher_rel)
        updates[key] = {name: value for name, value in candidate.items()
                        if name in entry or name == "source_sha256"}
        updates[key].update({
            "records": f"apps/api/parity/meep_gpu/{args.run}/{leg}/gate.json",
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "_this_recut": (args.why or "re-cut from a composition re-run") +
                           f" Digests READ from {args.run}/{leg}/ and cross-checked "
                           f"against the live tree by "
                           f"parity/meep_gpu/recut_composition_records.py, which "
                           f"refuses unless the run's staged manifest and the tree "
                           f"agree on every pinned file." +
                           (f" The launcher {launcher_rel} is not in this tree; its "
                            f"recorded digest is carried forward unverified."
                            if launcher_absent else ""),
        })
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
            if block in updates.get(key, {}):
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
            updates.setdefault(key, {})[block] = fresh_block
            recut_notes.extend(moved)
        if len(problems) > mark:
            continue
        if recut_notes:
            # WHAT MOVED GOES INTO THE RECORD, not only into this tool's stdout.
            # A curated measurement that changes without the record saying so is
            # the stale-claim defect with a fresh timestamp on it.
            updates[key]["_this_recut"] += (
                " CURATED CLAIM RE-CUT FROM THIS RUN, declared in "
                "recut_composition_records.CLAIM_RECUTS: "
                + "; ".join(sorted(
                    CLAIM_RECUTS[(key, block)] for block in sorted(updates[key])
                    if (key, block) in CLAIM_RECUTS))
                + " Fields: " + "; ".join(recut_notes) + ".")
        for note in recut_notes:
            print(f"    RE-CUT {note}", flush=True)
        print(f"  {key}: released (read from {read_from}), {len(fresh_sources)} "
              f"sources verified run==tree, curated blocks "
              + ("re-cut from the fresh summary" if recut_notes
                 else "agree with the fresh summary"), flush=True)

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

    if not args.write:
        print("\n  (report only -- pass --write to apply)", flush=True)
        return 0

    for key, fresh in updates.items():
        entry = ledger[key]
        # THE FROZEN SET IS MUTABLE PLUS WHAT THIS RUN DECLARED, and not one key
        # more. A curated block moves only where CLAIM_RECUTS names (entry, block)
        # AND this run actually re-cut it; a declaration that did not fire leaves
        # the block frozen like everything else, so the table cannot become a
        # standing licence for a block to drift.
        may_move = MUTABLE | {block for block in fresh if (key, block) in CLAIM_RECUTS}
        frozen = {k: v for k, v in entry.items() if k not in may_move}
        entry.update(fresh)
        after = {k: v for k, v in entry.items() if k not in may_move}
        if after != frozen:
            raise SystemExit(f"{key}: a key outside MUTABLE moved; refusing to write")
    if host_updates:
        stamp = f"_host_sha256_recut_{datetime.now(timezone.utc):%Y_%m_%d}"
        ledger["host_sha256"].update(host_updates)
        ledger[stamp] = (
            ", ".join(f"{n} -> {d[:12]}..." for n, d in sorted(host_updates.items()))
            + f". Licensed by the composition re-runs recorded at "
              f"apps/api/parity/meep_gpu/{args.run}/, both on the GPU host against these "
              f"exact digests -- the same mechanism as _host_sha256_recut_2026_08_28. "
              f"Digests READ from the tree only after this tool verified the runs "
              f"staged the same bytes; kernel_source_sha256 is untouched, so no "
              f"device source changed. " + (args.why or ""))
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


if __name__ == "__main__":
    raise SystemExit(main())
