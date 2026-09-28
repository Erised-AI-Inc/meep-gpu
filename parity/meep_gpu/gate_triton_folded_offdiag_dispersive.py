"""CUDA byte gate for folded tensor-epsilon dispersive ``update_E``.

The product rows compare the shipped engine-object builder against
``stepping.update_E`` over the complete field/polarization inventory.  The
rotation row additionally advances ``P`` after every E update, proving that the
plan resolves live pole pointers.  Mutation rows drop/reorder poles, freeze the
initial pointers, flip the mirror-ghost sign, and bind the wrong PML Yee lattice.
Progress and partial results are flushed after every row so a scheduled run is
observable and resumable.

Run from ``the repository root`` on an allocated CUDA node::

    PYTHONPATH=. python -u parity/meep_gpu/gate_triton_folded_offdiag_dispersive.py \
        --out parity/meep_gpu/results/<run>/folded_offdiag_dispersive.json
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

import probe_residual_group_bodies as probe  # noqa: E402
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
    from meep_gpu.triton_kernels import folded_offdiag_dispersive_update_e as arm

    plan = arm.plan_folded_offdiag_dispersive_constitutive(fields, pml)
    if plan is None:
        verdict = arm.folded_offdiag_dispersive_constitutive_coverage(fields, pml)
        raise RuntimeError(f"shipped builder refused: {verdict.reasons}")
    return plan


def static_groups(fields: Any, *, reverse: bool = False, drop: int = 0):
    from meep_gpu.triton_kernels.dispersive_update_e import poles_per_component

    order = poles_per_component(fields)
    groups = []
    for component in ("Ex", "Ey", "Ez"):
        arrays = [state.P[component] for state in order[component]]
        if reverse:
            arrays.reverse()
        if drop:
            arrays = arrays[:-drop] if drop <= len(arrays) else []
        groups.append(tuple(arrays))
    return groups


def reordered_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels.dispersive_update_e import StaticPoleBinding

    plan = shipped_plan(fields, pml)
    plan._poles = StaticPoleBinding(static_groups(fields, reverse=True))
    return plan


def dropped_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels.dispersive_update_e import StaticPoleBinding

    plan = shipped_plan(fields, pml)
    plan._poles = StaticPoleBinding(static_groups(fields, drop=1))
    plan.counts = plan._poles.counts
    return plan


def wrong_ghost_plan(fields: Any, pml: Any):
    plan = shipped_plan(fields, pml)
    weights = list(plan._base.ghost_weights)
    for axis, flag in enumerate(plan._base.ghost_axes):
        if flag:
            weights[axis] = -weights[axis]
    plan._base.ghost_weights = tuple(weights)
    return plan


def wrong_coefficient_lattice_plan(fields: Any, pml: Any):
    """Mutation: bind integer-site PML tables where E needs half-integer ones."""
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat

    plan = shipped_plan(fields, pml)
    plan._base._coefficients = tuple(
        CupyPointer(_flat(getattr(pml, f"{stem}_{axis}")))
        for axis in "xyz" for stem in ("kps", "kms"))
    return plan


class EThenP:
    def __init__(self, inner: Any, fields: Any, pml: Any) -> None:
        self.inner = inner
        self.fields = fields
        self.pml = pml

    def run(self, guard=None) -> None:
        self.inner.run(guard)
        stepping.update_P(self.fields, self.pml)

    def __repr__(self) -> str:
        return f"EThenP({self.inner!r})"


def live_rotation_plan(fields: Any, pml: Any):
    return EThenP(shipped_plan(fields, pml), fields, pml)


def frozen_rotation_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels.dispersive_update_e import StaticPoleBinding

    plan = shipped_plan(fields, pml)
    plan._poles = StaticPoleBinding(static_groups(fields))
    return EThenP(plan, fields, pml)


def array_e_then_p(fields: Any, pml: Any) -> None:
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def make_builder(**config):
    def make(offset: int):
        import cupy as cp

        merged = {
            "cell": (1.6, 1.6, 1.6),
            "symmetry": ("X",),
            "offdiag": True,
            "poles": 2,
            "pml_thickness": 2,
        }
        merged.update(config)
        return probe.build(cp, seed_offset=offset, **merged)
    return make


def case(name: str, prediction: str, builder, *, config=None,
         rotate: bool = False) -> Dict[str, Any]:
    return {
        "name": name,
        "group": "folded_offdiag_dispersive",
        "sub_step": "update_E",
        "prediction": prediction,
        "clause": ("live P pointer rotation" if rotate else
                   "ordered D-minus-P at folded off-diagonal Yee reads"),
        "make": make_builder(**(config or {})),
        "array_path": array_e_then_p if rotate else stepping.update_E,
        "kernel_plan": builder,
    }


def build_cases() -> List[Dict[str, Any]]:
    return [
        case("product_fold_x_two_poles", "IDENTICAL", shipped_plan),
        case("product_fold_y_one_pole", "IDENTICAL", shipped_plan,
             config={"symmetry": ("Y",), "poles": 1}),
        case("product_absorbed_power_density_shape", "IDENTICAL", shipped_plan,
             config={"cell": (2.4, 2.4, 0.0), "dimensions": 2,
                     "boundaries": {"x": "metallic", "y": "metallic",
                                    "z": "periodic"},
                     "symmetry": ("Y",)}),
        case("product_live_pointer_rotation", "IDENTICAL", live_rotation_plan,
             rotate=True),
        case("MUTATION_reverse_pole_order", "DIVERGENT", reordered_plan),
        case("MUTATION_drop_last_pole", "DIVERGENT", dropped_plan),
        case("MUTATION_flip_mirror_ghost_sign", "DIVERGENT", wrong_ghost_plan),
        case("MUTATION_use_integer_pml_tables", "DIVERGENT",
             wrong_coefficient_lattice_plan),
        case("MUTATION_freeze_rotating_pointers", "DIVERGENT",
             frozen_rotation_plan, rotate=True),
        case("CONTROL_reverse_one_pole", "IDENTICAL", reordered_plan,
             config={"poles": 1}),
    ]


def refusal_leg() -> Dict[str, Any]:
    import cupy as cp
    from meep_gpu.triton_kernels import folded_dispersive_update_e as diagonal
    from meep_gpu.triton_kernels import folded_offdiag_dispersive_update_e as arm
    from meep_gpu.triton_kernels import folded_offdiag_update_e as tensor

    rows = []
    configs = (
        ("target", {}, True),
        ("no_fold", {"symmetry": ()}, False),
        ("no_poles", {"poles": 0}, False),
        ("no_rows", {"offdiag": False}, False),
        ("inactive_pml", {"pml_thickness": 0}, False),
        ("complex_storage", {"complex_storage": True}, False),
        ("nonlinear", {}, False),
    )
    for name, changes, expected in configs:
        config = {"cell": (1.6, 1.6, 1.6), "symmetry": ("X",),
                  "offdiag": True, "poles": 2, "pml_thickness": 2}
        config.update(changes)
        fields, pml, _ = probe.build(cp, **config)
        if name == "nonlinear":
            # chi3 is mirror-compatible on every component. The shared fixture's
            # nonlinear=True also installs chi2 on the mirror-odd component and
            # is correctly rejected by Fields before this predicate is reached.
            fields.set_nonlinear_volumes(
                {}, {component: 0.05 for component in probe.E_NAMES})
        verdict = arm.folded_offdiag_dispersive_constitutive_coverage(fields, pml)
        tensor_verdict = tensor.folded_offdiag_constitutive_coverage(fields, pml)
        diagonal_verdict = diagonal.folded_dispersive_constitutive_coverage(
            fields, pml)
        rows.append({
            "case": name,
            "covered": verdict.covered,
            "expected": expected,
            "agrees": verdict.covered == expected,
            "tensor_parent_covered": tensor_verdict.covered,
            "diagonal_parent_covered": diagonal_verdict.covered,
            "exclusive": not (verdict.covered and
                              (tensor_verdict.covered or diagonal_verdict.covered)),
            "reasons": list(verdict.reasons),
        })
    return {
        "cases": rows,
        "all_agree": all(row["agrees"] for row in rows),
        "exclusive": all(row["exclusive"] for row in rows),
    }


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(
                f"{row.get('case')}: {row.get('verdict')} != "
                f"{row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    if len([row for row in rows if str(row.get("case", "")).startswith("product_")]) != 4:
        reasons.append("the four product rows were not all recorded")
    if len([row for row in rows if "MUTATION" in str(row.get("case", ""))]) != 5:
        reasons.append("the five mutation rows were not all recorded")
    refusals = payload.get("refusals") or {}
    if not refusals.get("all_agree"):
        reasons.append("predicate refusal battery disagreed")
    if not refusals.get("exclusive"):
        reasons.append("the new family overlaps a parent predicate")
    return {"released": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    driver_log = open(out.parent / "driver.log", "a", encoding="utf-8")
    jsonl = open(out.parent / "cases.jsonl", "a", encoding="utf-8")
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "folded_offdiag_dispersive_update_E",
        "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from meep_gpu.subnormal_policy import install_subnormal_policy
    install_subnormal_policy("keep")
    payload["subnormal_policy_installed"] = "keep"

    rows = []
    cases = build_cases()
    for index, spec in enumerate(cases, 1):
        log(driver_log, f"case {index}/{len(cases)} {spec['name']} "
                        f"predict={spec['prediction']}")
        try:
            row = probe.run_case(cycles=args.cycles, **spec)
        except BaseException as exc:  # noqa: BLE001 - record and continue
            row = {"case": spec["name"], "prediction": spec["prediction"],
                   "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:500],
                   "traceback": traceback.format_exc()[-2000:]}
        rows.append(row)
        emit(jsonl, row)
        log(driver_log, f"  -> {row.get('verdict', 'ERROR')} "
                        f"differing_max={row.get('differing_max', '?')} "
                        f"elapsed={time.time() - started:.1f}s")
    payload["cases"] = rows
    payload["refusals"] = refusal_leg()
    payload["seconds"] = round(time.time() - started, 2)
    payload["release"] = validate(payload)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    log(driver_log, f"RELEASED={payload['release']['released']}")
    for reason in payload["release"]["reasons"]:
        log(driver_log, f"  refused: {reason}")
    driver_log.close()
    jsonl.close()
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
