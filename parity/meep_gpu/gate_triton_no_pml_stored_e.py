"""Device byte gate for no-PML stored-E ``update_E`` (Arm S).

The product leg drives the shipped engine-object predicate and builder against
``stepping.update_E``.  A separate rotation row advances polarization state after
each E update; this is the load-bearing test for the plan's live pole binding,
because ``PolarizationState.update`` rotates the P/P_prev/scratch buffers.

The mutation battery reverses the pole order, drops the final pole, and freezes
the initial pole pointers across those rotations.  The one-pole reversal is an
expected-identical control: reversing one item cannot change arithmetic, so a
divergence there would indict the harness.  Comparisons are whole-inventory
uint32 comparisons through the shared audited probe, with incremental progress
and JSONL output.
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


def log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def shipped_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    plan = arm.plan_stored_e_constitutive(fields, pml)
    if plan is None:
        verdict = arm.stored_e_constitutive_coverage(fields, pml)
        raise RuntimeError(f"shipped stored-E builder refused: {verdict.reasons}")
    return plan


class EThenPPlan:
    """Run an E plan and then rotate P exactly as a complete electric half-step."""

    def __init__(self, inner: Any, fields: Any, pml: Any) -> None:
        self.inner = inner
        self.fields = fields
        self.pml = pml

    def run(self, guard=None) -> None:
        self.inner.run(guard)
        stepping.update_P(self.fields, self.pml)

    def __repr__(self) -> str:
        return f"EThenPPlan({self.inner!r})"


def array_e_then_p(fields: Any, pml: Any) -> None:
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def static_groups(fields: Any, *, reverse: bool = False, drop: int = 0):
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    order = arm.poles_per_component(fields)
    groups = []
    for component, _source in arm.E_TERMS:
        arrays = [state.P[component] for state in order[component]]
        if reverse:
            arrays.reverse()
        if drop:
            arrays = arrays[:-drop] if drop <= len(arrays) else []
        groups.append(tuple(arrays))
    return groups


def reordered_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticPoleBinding(static_groups(fields, reverse=True))
    return plan


def dropped_pole_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticPoleBinding(static_groups(fields, drop=1))
    plan.counts = plan._poles.counts
    return plan


def frozen_rotation_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticPoleBinding(static_groups(fields))
    return EThenPPlan(plan, fields, pml)


def live_rotation_plan(fields: Any, pml: Any):
    return EThenPPlan(shipped_plan(fields, pml), fields, pml)


def make_builder(**config):
    def make(offset: int):
        import cupy as cp

        return P.build(cp, seed_offset=offset, pml_thickness=0, **config)
    return make


def case(name: str, prediction: str, config: Dict[str, Any], builder,
         *, rotate: bool = False) -> Dict[str, Any]:
    return {
        "name": name, "group": "S", "sub_step": "update_E",
        "prediction": prediction,
        "clause": ("live polarization pointer rotation" if rotate
                   else "stored E, inert layer, ordered D-minus-P chain"),
        "make": make_builder(**config),
        "array_path": array_e_then_p if rotate else stepping.update_E,
        "kernel_plan": builder,
    }


def build_cases() -> List[Dict[str, Any]]:
    return [
        case("product_3d_two_poles", "IDENTICAL", {"poles": 2}, shipped_plan),
        case("product_1d_five_poles", "IDENTICAL",
             {"poles": 5, "cell": (0.0, 0.0, 4.0), "dimensions": 1,
              "boundaries": "periodic"},
             shipped_plan),
        case("product_no_poles", "IDENTICAL", {"poles": 0}, shipped_plan),
        case("product_conductive_two_poles", "IDENTICAL",
             {"poles": 2, "conductivity": 0.20,
              "magnetic_conductivity": 0.17}, shipped_plan),
        case("product_live_pointer_rotation", "IDENTICAL", {"poles": 2},
             live_rotation_plan, rotate=True),
        case("MUTATION_reverse_pole_order", "DIVERGENT", {"poles": 2},
             reordered_plan),
        case("MUTATION_drop_last_pole", "DIVERGENT", {"poles": 2},
             dropped_pole_plan),
        case("MUTATION_freeze_rotating_pointers", "DIVERGENT", {"poles": 2},
             frozen_rotation_plan, rotate=True),
        case("CONTROL_reverse_one_pole", "IDENTICAL", {"poles": 1},
             reordered_plan),
    ]


def refusal_leg() -> Dict[str, Any]:
    import cupy as cp
    from meep_gpu.triton_kernels import no_pml_constitutive as null_family
    from meep_gpu.triton_kernels import no_pml_stored_e as arm

    rows = []
    for name, config, expected in (
        ("target", dict(pml_thickness=0, poles=2), True),
        ("degenerate_no_poles", dict(pml_thickness=0, poles=0), True),
        ("active_layer", dict(pml_thickness=2, poles=2), False),
        ("complex_storage", dict(pml_thickness=0, poles=2,
                                 complex_storage=True), False),
        ("offdiagonal", dict(pml_thickness=0, poles=2, offdiag=True), False),
    ):
        fields, pml, _ = P.build(cp, **config)
        mine = arm.stored_e_constitutive_coverage(fields, pml)
        null = null_family.null_constitutive_coverage(fields, pml, "E")
        rows.append({"case": name, "covered": mine.covered,
                     "expected": expected, "agrees": mine.covered == expected,
                     "null_covered": null.covered,
                     "exclusive_where_targeted": not (mine.covered and null.covered),
                     "reasons": list(mine.reasons)})
    return {"cases": rows,
            "all_agree": all(row["agrees"] for row in rows),
            "exclusive": all(row["exclusive_where_targeted"] for row in rows)}


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in str(row.get("case", ""))]
    controls = [row for row in rows if "CONTROL" in str(row.get("case", ""))]
    if len(products) != 5:
        reasons.append(f"expected 5 product rows, found {len(products)}")
    if len(mutations) != 3:
        reasons.append(f"expected 3 mutation rows, found {len(mutations)}")
    if len(controls) != 1:
        reasons.append(f"expected 1 control row, found {len(controls)}")
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
        reasons.append("a predicate answer disagreed with its expectation")
    if not refusals.get("exclusive"):
        reasons.append("stored-E and null families overlap")
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
        "gate": "no_pml_stored_e", "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from meep_gpu.subnormal_policy import install_subnormal_policy
    install_subnormal_policy("keep")
    payload["subnormal_policy_installed"] = "keep"

    rows = []
    built = build_cases()
    for index, spec in enumerate(built, 1):
        log(run_log, f"case {index}/{len(built)} {spec['name']} "
                     f"predict={spec['prediction']}")
        try:
            row = P.run_case(cycles=args.cycles, **spec)
        except BaseException as exc:  # noqa: BLE001 - a gate records and continues
            row = {"case": spec["name"], "prediction": spec["prediction"],
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
