"""The Metal composer's disjointness, MEASURED — configurations built, predicates called.

WHAT THIS ANSWERS, and why nothing else can. ``plan_step`` is fail-closed: two arms
admitting one slot leaves the slot UNSELECTED and names both. That contract is only
worth anything if the ambiguity is RARE — a composer that refuses every slot is
fail-closed and useless — so the question is not "does it fail closed" but "over the
configurations this backend can meet, how often do two products admit the same
slot". That is a measurement over constructed configurations, not a reading of
clause lists, because the clause lists are exactly what is being checked.

THE METHOD IS THE TRITON PLANNER SWEEP'S, unchanged: build a configuration matrix
that puts each family near its neighbours' boundaries, then for EVERY row and EVERY
slot call EVERY registered predicate — wired or not — and count. Three numbers come
out and all three are reported:

* **evaluations** — rows x slots x arms-on-that-slot. The size of the measurement.
* **admissions** — how many (row, slot, arm) triples returned covered.
* **overlaps** — how many (row, slot) pairs had two or more admitters. Every one is
  listed by name, with its claimants, and every one must be a PLANTED row.

THE SWEEP CONSULTS UNWIRED ARMS TOO. As of tranche 2 every certified family is
wired, so the two sets coincide and the distinction costs nothing — but the loop is
written over ``arms.registered()`` rather than ``arms.arms_for()`` deliberately, so
that the next family to be built-and-gated-before-wiring is swept the moment it
registers instead of the moment somebody remembers to add it here.

WHAT THIS IS NOT. It is not a byte claim: no kernel launches here and no field moves.
It is not a coverage number either — that is the corpus battery, on lifted rows. It
is the composition's own safety property and nothing more.

Rule 7: one flushed line per row, the artifact rewritten as each row lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu.metal_kernels import arms, device, launch  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def provenance() -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    import numpy  # noqa: PLC0415

    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "environment": dict(ENVIRONMENT),
        "metal_frontend": None,
        "torch": None,
        "mps_available": None,
    }
    try:
        import torch  # noqa: PLC0415

        record["torch"] = str(torch.__version__)
        record["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 - a probe may run without torch
        record["torch"] = f"unavailable: {exc!r}"
    try:
        record["metal_frontend"] = launch.metal_frontend_version()
    except Exception as exc:  # noqa: BLE001
        record["metal_frontend"] = f"unavailable: {exc!r}"
    return record


def table_snapshot() -> List[Dict[str, Any]]:
    """The arm table as data, so an artifact records the SUBJECT it measured."""
    rows: List[Dict[str, Any]] = []
    for slot in arms.registered_slots():
        for spec in arms.registered(slot):
            rows.append({"slot": slot, "family": spec.family,
                         "label": spec.label, "wired": bool(spec.wired)})
    return rows


def probe_kwargs() -> Dict[str, Any]:
    """The measured expansion records, read once and passed to every arm."""
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields,
        cylindrical_complex,
        folded_complex,
        special_kz,
    )

    return {
        "complex_probe": complex_fields.load_expansion_probe(),
        "beta_probe": special_kz.load_expansion_probe(),
        "folded_complex_probe": folded_complex.load_expansion_probe(),
        # PASSED EXPLICITLY rather than left to the arm's env-var fallback. The
        # fallback works and is why this family composed before the argument
        # existed, but "the sweep handed the composer the artifact it measured
        # against" and "the arm found one somewhere" are different claims, and only
        # the first can be recorded.
        "cylindrical_complex_probe": cylindrical_complex.load_expansion_probe(),
    }


def sweep(out_path: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "provenance": provenance(),
        "arm_table": table_snapshot(),
        "rows": [],
        "totals": {},
    }
    probes = probe_kwargs()
    if probes["folded_complex_probe"] is None:
        raise SystemExit(
            "no folded-complex expansion probe artifact; the folded complex arms "
            "would refuse by name on all six of their slots and the sweep would "
            "measure their absence rather than their disjointness")
    if probes["cylindrical_complex_probe"] is None:
        raise SystemExit(
            "no cylindrical-complex expansion probe artifact; that family's four "
            "arms would refuse by name and the sweep would record their absence "
            "as disjointness")
    if probes["complex_probe"] is None or probes["beta_probe"] is None:
        # A missing artifact is a REFUSAL for four of the eight products, so the
        # sweep would report disjointness it never tested. Fail here rather than
        # produce a clean-looking record of an untested tree.
        raise SystemExit(
            "no expansion probe artifact for "
            f"{'complex' if probes['complex_probe'] is None else ''}"
            f"{' and ' if not any(probes.values()) else ''}"
            f"{'beta' if probes['beta_probe'] is None else ''}"
            "; the complex and beta arms would refuse by name and the sweep would "
            "measure their absence rather than their disjointness")

    slots = tuple(arms.registered_slots())
    evaluations = 0
    admissions = 0
    overlaps: List[Dict[str, Any]] = []
    winners: Dict[str, int] = {}
    started = time.time()

    for index, (label, build, expected, expected_ambiguous) in enumerate(
            matrix.MATRIX, start=1):
        fields, pml = build()
        residency = device.Residency()
        context = arms.StepContext(
            fields, pml, residency, (launch.shaders.CONTRACT_OFF,),
            sources=(), synced=(),
            extra={"complex_probe": probes["complex_probe"],
                   "beta_probe": probes["beta_probe"],
                   "folded_complex_probe": probes["folded_complex_probe"],
                   "cylindrical_complex_probe":
                       probes["cylindrical_complex_probe"]})

        per_slot: Dict[str, List[str]] = {}
        refusals: Dict[str, Dict[str, List[str]]] = {}
        for slot in slots:
            admitted: List[str] = []
            for spec in arms.registered(slot):
                evaluations += 1
                try:
                    verdict = spec.coverage(context, slot)
                except Exception as exc:  # noqa: BLE001 - a raise is a refusal
                    refusals.setdefault(slot, {})[spec.label] = [f"RAISED {exc!r}"]
                    continue
                if getattr(verdict, "covered", False):
                    admitted.append(spec.label)
                    admissions += 1
                else:
                    refusals.setdefault(slot, {})[spec.label] = list(
                        getattr(verdict, "reasons", ()))[:2]
            per_slot[slot] = admitted
            if len(admitted) > 1:
                overlaps.append({"row": label, "slot": slot,
                                 "claimants": list(admitted)})

        # The SHIPPED composer on the same objects, so the sweep's own bookkeeping
        # is checked against the thing that actually decides.
        plan = launch.plan_step(
            fields, pml, residency=residency, sources=(),
            complex_probe=probes["complex_probe"],
            beta_probe=probes["beta_probe"],
            folded_complex_probe=probes["folded_complex_probe"],
            cylindrical_complex_probe=probes["cylindrical_complex_probe"])
        ambiguous = tuple(sorted(
            slot for slot, reasons in plan.reasons.items()
            if any("admit this configuration" in reason for reason in reasons)))
        for arm_label in plan.selected.values():
            winners[arm_label] = winners.get(arm_label, 0) + 1

        agrees = all(
            (per_slot.get(slot, []) == [plan.selected[slot]]) if slot in plan.selected
            else (len(per_slot.get(slot, [])) != 1)
            for slot in slots)

        row = {
            "row": label,
            "admitters": per_slot,
            "selected": dict(plan.selected),
            "expected_selected": dict(expected),
            "ambiguous": list(ambiguous),
            "expected_ambiguous": list(expected_ambiguous),
            "matches_expectation": (dict(plan.selected) == dict(expected)
                                    and ambiguous == tuple(expected_ambiguous)),
            "sweep_agrees_with_composer": bool(agrees),
            "residency_covered": bool(plan.residency and plan.residency.covered),
            "sample_refusals": refusals,
        }
        payload["rows"].append(row)
        payload["totals"] = {
            "configurations": index,
            "slots_per_configuration": len(slots),
            "predicate_evaluations": evaluations,
            "admissions": admissions,
            "overlaps": len(overlaps),
            "overlap_detail": overlaps,
            "winners": dict(winners),
        }
        save(payload, out_path)
        log(f"[sweep {index:>2}/{len(matrix.MATRIX)}] {label:<32} "
            f"selected={ {k: v for k, v in plan.selected.items()} } "
            f"ambiguous={list(ambiguous)} "
            f"{'OK' if row['matches_expectation'] else '*** MISMATCH ***'} "
            f"({time.time() - started:.1f}s)")
        assert row["sweep_agrees_with_composer"], (
            f"{label}: the sweep's own admitter count disagrees with the shipped "
            f"composer's selection — {per_slot} vs {plan.selected}")

    payload["totals"]["configurations"] = len(matrix.MATRIX)
    payload["totals"]["mismatches"] = [
        row["row"] for row in payload["rows"] if not row["matches_expectation"]]
    payload["totals"]["arms_never_winning"] = [
        arm for arm in matrix.EXPECTED_WINNERS if arm not in winners]

    # --- THE OTHER HALF OF THE NON-VACUITY FLOOR -----------------------------
    # `arms_never_winning` catches an arm that fires nowhere. This catches an arm
    # that fires WHERE NO PRODUCT EXISTS, which is the worse of the two: an arm
    # quietly admitting a cylindrical, nonlinear, folded-beta or folded-BFAST row
    # would step a recurrence this backend does not implement, and the sweep would
    # report it as coverage. Only the four ARITHMETIC slots are checked — a seam
    # fill reads none of those terms and correctly still composes.
    uncarried: List[Dict[str, Any]] = []
    by_label = {row["row"]: row for row in payload["rows"]}
    for label, family in sorted(matrix.UNCARRIED.items()):
        row = by_label.get(label)
        if row is None:
            uncarried.append({"row": label, "family": family,
                              "violation": "ROW ABSENT FROM THE MATRIX"})
            continue
        selected = {slot: arm for slot, arm in row["selected"].items()
                    if slot in matrix.ARITHMETIC_SLOTS}
        if selected:
            uncarried.append({"row": label, "family": family,
                              "violation": f"selected {selected}"})
    payload["totals"]["uncarried_families_checked"] = len(matrix.UNCARRIED)
    payload["totals"]["uncarried_violations"] = uncarried
    payload["totals"]["unregistered_slots_violations"] = [
        slot for slot in matrix.UNREGISTERED_SLOTS
        if slot in arms.registered_slots()]
    payload["totals"]["registered_arms"] = len(arms.registered())
    payload["totals"]["registered_families"] = sorted(
        {spec.family for spec in arms.registered()})
    save(payload, out_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        API_ROOT, "parity/meep_gpu/results/metal_composition_2026-08-16"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "composition_sweep.json")

    payload = sweep(out_path)
    totals = payload["totals"]

    log("")
    log("METAL COMPOSITION SWEEP")
    log(f"  configurations           : {totals['configurations']}")
    log(f"  slots per configuration  : {totals['slots_per_configuration']}")
    log(f"  registered arms          : {totals['registered_arms']} over "
        f"{len(totals['registered_families'])} families")
    log(f"  PREDICATE EVALUATIONS    : {totals['predicate_evaluations']}")
    log(f"  admissions               : {totals['admissions']}")
    log(f"  OVERLAPS (2+ admitters)  : {totals['overlaps']}")
    for entry in totals["overlap_detail"]:
        log(f"      {entry['row']:<34} {entry['slot']:<10} "
            f"{', '.join(entry['claimants'])}")
    log(f"  rows disagreeing w/ pin  : {totals['mismatches']}")
    log(f"  arms that never won      : {totals['arms_never_winning']}")
    log(f"  uncarried families checked: {totals['uncarried_families_checked']}")
    log(f"  UNCARRIED VIOLATIONS     : {totals['uncarried_violations']}")
    log(f"  unregistered-slot breach : {totals['unregistered_slots_violations']}")
    log("")
    for arm in matrix.EXPECTED_WINNERS:
        log(f"      {arm:<24} won {totals['winners'].get(arm, 0):>2} rows")

    failed = (bool(totals["mismatches"])
              or bool(totals["arms_never_winning"])
              or bool(totals["uncarried_violations"])
              or bool(totals["unregistered_slots_violations"]))
    # THE ONLY OVERLAPS THIS SWEEP TOLERATES, and both are the SAME engine coupling
    # one level apart: an arm refusing on `has_offdiagonal_epsilon` and an arm
    # requiring a LIVE ROW SLOT agree only when the flag and the dict disagree, which
    # the engine's read-only property makes unreachable. The matrix names both rows
    # as plants; this set is the second place that has to say so, and a row missing
    # here reads as a predicate defect rather than as a plant.
    planted = {"offdiag_flag_false_slot_live", "fold_offdiag_flag_false_slot_live"}
    unplanted = [entry for entry in totals["overlap_detail"]
                 if entry["row"] not in planted]
    if unplanted:
        log(f"  *** UNPLANTED OVERLAP: {unplanted}")
        failed = True
    log(f"  artifact: {out_path}")
    log("VERDICT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
