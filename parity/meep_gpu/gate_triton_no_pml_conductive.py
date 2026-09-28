"""Device byte gate for the real no-PML conductive curl family.

This gate drives the shipped engine-object predicate and builder.  It covers
both ordinary two-sided conductivity and the one-sided configuration that owns
the family's most easily missed slots: a D-only run must still build ``step_B``
with all three per-target ``COND`` flags false, and conversely for B-only
``step_D``.  The incumbent plain family refuses both slots per run, so those
lossless specialisations belong here.

Every case compares the complete allocated field inventory as uint32 words over
multiple launches.  Inputs are perturbed between launches by the audited shared
probe, and each row carries the shared probe's moved-word and signed-zero vacuity
checks.  Results and progress are written incrementally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for path in (str(HERE), str(API)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_residual_group_bodies as P  # noqa: E402
from meep_gpu import stepping  # noqa: E402

CYCLES = 8
STEPS = {"step_B": stepping.step_B, "step_D": stepping.step_D}


def log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def shipped_plan(fields: Any, pml: Any, sub_step: str):
    from meep_gpu.triton_kernels import no_pml_conductive as arm

    plan = arm.plan_conductive_plain_curl(fields, pml, sub_step)
    if plan is None:
        verdict = arm.conductive_plain_curl_coverage(fields, pml, sub_step)
        raise RuntimeError(
            f"shipped conductive builder refused admitted {sub_step}: "
            f"{verdict.reasons}")
    return plan


def plain_tail_mutation(fields: Any, pml: Any, sub_step: str):
    """Compile every target through the lossless tail."""
    plan = shipped_plan(fields, pml, sub_step)
    plan.cond = (0, 0, 0)
    return plan


def swapped_factors_mutation(fields: Any, pml: Any, sub_step: str):
    """Exchange condfac and condinv while retaining the shipped stencil."""
    plan = shipped_plan(fields, pml, sub_step)
    plan._condfac, plan._condinv = plan._condinv, plan._condfac
    return plan


def wrong_derive_mutation(fields: Any, pml: Any, sub_step: str):
    """Derive E from a plan whose sources are already the stored E volumes."""
    plan = shipped_plan(fields, pml, sub_step)
    if plan.derive != 0:
        raise RuntimeError("mutation fixture unexpectedly does not store E")
    plan.derive = 1
    return plan


def make_builder(**config):
    def make(offset: int):
        import cupy as cp

        return P.build(cp, seed_offset=offset, pml_thickness=0, **config)
    return make


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    product = (
        ("both_no_poles", dict(conductivity=0.20,
                               magnetic_conductivity=0.17, poles=0)),
        ("both_two_poles", dict(conductivity=0.20,
                                magnetic_conductivity=0.17, poles=2)),
        ("D_only", dict(conductivity=0.20, poles=0)),
        ("B_only", dict(magnetic_conductivity=0.17, poles=0)),
    )
    for tag, config in product:
        for sub_step in ("step_B", "step_D"):
            cases.append({
                "name": f"product_{tag}_{sub_step}",
                "group": "E/H", "sub_step": sub_step,
                "prediction": "IDENTICAL",
                "clause": "conductivity exists per run; COND flags are per target",
                "make": make_builder(**config),
                "array_path": STEPS[sub_step],
                "kernel_plan": lambda f, p, s=sub_step: shipped_plan(f, p, s),
            })

    mutations = (
        ("MUTATION_plain_tail", "step_B",
         dict(conductivity=0.20, magnetic_conductivity=0.17, poles=2),
         plain_tail_mutation),
        ("MUTATION_swap_condfac_condinv", "step_D",
         dict(conductivity=0.20, magnetic_conductivity=0.17, poles=2),
         swapped_factors_mutation),
        ("MUTATION_force_derive_against_stored_E", "step_B",
         dict(conductivity=0.20, magnetic_conductivity=0.17, poles=0),
         wrong_derive_mutation),
    )
    for tag, sub_step, config, builder in mutations:
        cases.append({
            "name": f"{tag}_{sub_step}", "group": "mutation",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": tag, "make": make_builder(**config),
            "array_path": STEPS[sub_step],
            "kernel_plan": lambda f, p, s=sub_step, b=builder: b(f, p, s),
        })
    return cases


def refusal_leg() -> Dict[str, Any]:
    import cupy as cp
    from meep_gpu.triton_kernels import no_pml as plain
    from meep_gpu.triton_kernels import no_pml_conductive as arm

    rows = []
    configurations = (
        ("lossless", dict(pml_thickness=0), False, True),
        ("both", dict(pml_thickness=0, conductivity=0.2,
                      magnetic_conductivity=0.17), True, False),
        ("D_only", dict(pml_thickness=0, conductivity=0.2), True, False),
        ("B_only", dict(pml_thickness=0, magnetic_conductivity=0.17), True, False),
    )
    for name, config, expected_arm, expected_plain in configurations:
        fields, pml, _ = P.build(cp, **config)
        for sub_step in ("step_B", "step_D"):
            mine = arm.conductive_plain_curl_coverage(fields, pml, sub_step).covered
            theirs = plain.plain_curl_coverage(fields, pml, sub_step).covered
            rows.append({"case": f"{name}:{sub_step}", "conductive": mine,
                         "plain": theirs, "expected_conductive": expected_arm,
                         "expected_plain": expected_plain,
                         "agrees": (mine == expected_arm and
                                    theirs == expected_plain),
                         "exclusive": mine != theirs})
    fields, pml, _ = P.build(cp, pml_thickness=2, conductivity=0.2,
                             magnetic_conductivity=0.17)
    for sub_step in ("step_B", "step_D"):
        verdict = arm.conductive_plain_curl_coverage(fields, pml, sub_step)
        rows.append({"case": f"active_layer:{sub_step}",
                     "conductive": verdict.covered,
                     "expected_conductive": False,
                     "agrees": not verdict.covered,
                     "exclusive": True,
                     "reasons": list(verdict.reasons)})
    return {"cases": rows,
            "all_agree": all(row["agrees"] for row in rows),
            "all_exclusive": all(row["exclusive"] for row in rows)}


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in str(row.get("case", ""))]
    if len(products) != 8:
        reasons.append(f"expected 8 product rows, found {len(products)}")
    if len(mutations) != 3:
        reasons.append(f"expected 3 mutations, found {len(mutations)}")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(f"{row.get('case')}: {row.get('verdict')} != "
                           f"{row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    refusals = payload.get("refusals") or {}
    if not refusals.get("all_agree"):
        reasons.append("a predicate answer disagreed with its expected owner")
    if not refusals.get("all_exclusive"):
        reasons.append("the conductive and plain predicates overlap or leave a gap")
    return {"released": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_log = open(out.parent / "driver.log", "a", encoding="utf-8")
    jsonl = open(out.parent / "cases.jsonl", "a", encoding="utf-8")
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "no_pml_conductive", "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from meep_gpu.subnormal_policy import install_subnormal_policy
    install_subnormal_policy("keep")
    payload["subnormal_policy_installed"] = "keep"

    rows = []
    built = build_cases()
    for index, case in enumerate(built, 1):
        log(run_log, f"case {index}/{len(built)} {case['name']} "
                     f"predict={case['prediction']}")
        try:
            row = P.run_case(cycles=args.cycles, **case)
        except BaseException as exc:  # noqa: BLE001 - a gate records and continues
            row = {"case": case["name"], "prediction": case["prediction"],
                   "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:500],
                   "traceback": traceback.format_exc()[-1600:]}
        rows.append(row)
        emit(jsonl, row)
        log(run_log, f"  -> {row.get('verdict', 'ERROR')} "
                     f"differing_max={row.get('differing_max', '?')} "
                     f"elapsed={time.time() - started:.1f}s")
    payload["cases"] = rows
    payload["refusals"] = refusal_leg()
    payload["seconds"] = round(time.time() - started, 2)
    payload["release"] = validate(payload)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    log(run_log, f"RELEASED={payload['release']['released']}")
    for reason in payload["release"]["reasons"]:
        log(run_log, f"  refused: {reason}")
    run_log.close()
    jsonl.close()
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
