"""CUDA byte gate for the shared complex off-diagonal ``update_E`` core.

The five product rows are the residual corpus shapes: three folded/PML rows and
two no-PML Bloch rows.  Each is stepped through the shipped predicate and
builder for eight perturbed cycles and compared over the complete stored
inventory as uint32 words.  Mutation rows drop a live tensor slot, disable a
live phase, flip mirror parity, or select the wrong storage tail.

The run is incremental: every case is printed unbuffered and fsync'd to JSONL.
Use one Slurm allocation and one idle GPU; this script never selects a device.
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


def load_probe(path: str | None) -> Any:
    from meep_gpu.triton_kernels import complex_fields

    if path:
        return json.loads(Path(path).read_text())
    return complex_fields.load_expansion_probe()


def makers():
    def folded_3d(offset):
        import cupy as cp
        return P.build(cp, cell=(1.6, 1.6, 1.6), symmetry=("X",),
                       complex_storage=True, offdiag=True, pml_thickness=2,
                       seed_offset=offset)

    def folded_twofold(offset):
        import cupy as cp
        return P.build(
            cp, cell=(2.4, 2.4, 0.0), dimensions=2,
            boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
            symmetry=("X", "Y"), complex_storage=True, offdiag=True,
            pml_thickness=2, seed_offset=offset)

    def folded_bloch(offset):
        import cupy as cp
        return P.build(cp, cell=(2.4, 2.4, 0.0), dimensions=2,
                       boundaries="periodic", symmetry=("Y",),
                       k_point=(3.5, 0.0, 0.0), complex_storage=True,
                       offdiag=True, pml_thickness=2, seed_offset=offset)

    def no_pml_3d(offset):
        import cupy as cp
        return P.build(cp, cell=(2.5, 2.5, 2.5), dimensions=3,
                       boundaries="periodic", k_point=(0.23, -0.17, 0.35),
                       complex_storage=True, offdiag=True, pml_thickness=0,
                       seed_offset=offset)

    def no_pml_2d(offset):
        import cupy as cp
        return P.build(cp, cell=(2.5, 2.5, 0.0), dimensions=2,
                       boundaries="periodic", k_point=(0.3892, 0.1597, 0.0),
                       complex_storage=True, offdiag=True, pml_thickness=0,
                       seed_offset=offset)

    return {
        "folded_3d": folded_3d,
        "folded_twofold": folded_twofold,
        "folded_bloch": folded_bloch,
        "no_pml_3d": no_pml_3d,
        "no_pml_2d": no_pml_2d,
    }


def shipped_plan(fields: Any, pml: Any, probe: Any, folded: bool):
    from meep_gpu.triton_kernels import complex_offdiag_update_e as arm

    builder = (arm.plan_complex_folded_offdiag_update_e if folded else
               arm.plan_complex_no_pml_offdiag_update_e)
    plan = builder(fields, pml, probe=probe)
    if plan is None:
        predicate = (arm.complex_folded_offdiag_update_e_coverage if folded else
                     arm.complex_no_pml_offdiag_update_e_coverage)
        raise RuntimeError(f"shipped builder refused: "
                           f"{predicate(fields, pml, probe).reasons}")
    return plan


def mutate_drop_row(fields, pml, probe, folded):
    plan = shipped_plan(fields, pml, probe, folded)
    mask = list(plan.row_mask)
    mask[next(index for index, live in enumerate(mask) if live)] = 0
    plan.row_mask = tuple(mask)
    return plan


def mutate_wrong_tail(fields, pml, probe, folded):
    plan = shipped_plan(fields, pml, probe, folded)
    plan.pml = not plan.pml
    return plan


def mutate_phase_off(fields, pml, probe, folded):
    plan = shipped_plan(fields, pml, probe, folded)
    plan.phase_flags = (0, 0, 0)
    return plan


def mutate_mirror_sign(fields, pml, probe, folded):
    plan = shipped_plan(fields, pml, probe, folded)
    weights = list(plan.ghost_weights)
    axis = next(index for index, live in enumerate(plan.mirror_axes) if live)
    weights[axis] *= -1.0
    plan.ghost_weights = tuple(weights)
    return plan


def build_cases(probe: Any) -> List[Dict[str, Any]]:
    make = makers()
    rows: List[Dict[str, Any]] = []
    for name, folded in (("folded_3d", True), ("folded_twofold", True),
                         ("folded_bloch", True), ("no_pml_3d", False),
                         ("no_pml_2d", False)):
        rows.append({
            "name": "product_" + name, "group": "complex_offdiag",
            "sub_step": "update_E", "prediction": "IDENTICAL",
            "clause": "shared complex tensor row plus the correct storage tail",
            "make": make[name], "array_path": stepping.update_E,
            "kernel_plan": lambda f, p, folded=folded: shipped_plan(
                f, p, probe, folded),
        })
    rows.extend([
        {"name": "MUTATION_drop_live_row", "group": "complex_offdiag",
         "sub_step": "update_E", "prediction": "DIVERGENT",
         "clause": "one live tensor coefficient slot is omitted",
         "make": make["folded_3d"], "array_path": stepping.update_E,
         "kernel_plan": lambda f, p: mutate_drop_row(f, p, probe, True)},
        {"name": "MUTATION_wrong_folded_tail", "group": "complex_offdiag",
         "sub_step": "update_E", "prediction": "DIVERGENT",
         "clause": "direct-store tail substituted for PML accumulation",
         "make": make["folded_3d"], "array_path": stepping.update_E,
         "kernel_plan": lambda f, p: mutate_wrong_tail(f, p, probe, True)},
        {"name": "MUTATION_wrong_no_pml_tail", "group": "complex_offdiag",
         "sub_step": "update_E", "prediction": "DIVERGENT",
         "clause": "PML accumulation substituted for direct storage",
         "make": make["no_pml_3d"], "array_path": stepping.update_E,
         "kernel_plan": lambda f, p: mutate_wrong_tail(f, p, probe, False)},
        {"name": "MUTATION_phase_off", "group": "complex_offdiag",
         "sub_step": "update_E", "prediction": "DIVERGENT",
         "clause": "Bloch rotations compiled out on both shift directions",
         "make": make["no_pml_3d"], "array_path": stepping.update_E,
         "kernel_plan": lambda f, p: mutate_phase_off(f, p, probe, False)},
        {"name": "MUTATION_mirror_sign", "group": "complex_offdiag",
         "sub_step": "update_E", "prediction": "DIVERGENT",
         "clause": "partner-axis mirror parity sign flipped",
         "make": make["folded_3d"], "array_path": stepping.update_E,
         "kernel_plan": lambda f, p: mutate_mirror_sign(f, p, probe, True)},
    ])
    return rows


def refusal_leg(probe: Any) -> Dict[str, Any]:
    import cupy as cp
    from meep_gpu.triton_kernels import complex_offdiag_update_e as arm

    cases = []
    configurations = (
        ("no_pml_target", dict(cell=(1.2, 1.2, 1.2), boundaries="periodic",
                                k_point=(0.2, 0.1, 0.3), complex_storage=True,
                                offdiag=True, pml_thickness=0), "plain", True),
        ("folded_target", dict(cell=(1.6, 1.6, 1.6), symmetry=("X",),
                               complex_storage=True, offdiag=True,
                               pml_thickness=2), "folded", True),
        ("real_storage", dict(cell=(1.2, 1.2, 1.2), offdiag=True,
                              pml_thickness=0), "plain", False),
        ("no_rows", dict(cell=(1.2, 1.2, 1.2), boundaries="periodic",
                         k_point=(0.2, 0.1, 0.3), complex_storage=True,
                         offdiag=False, pml_thickness=0), "plain", False),
        ("active_on_plain", dict(cell=(1.2, 1.2, 1.2), boundaries="periodic",
                                 k_point=(0.2, 0.1, 0.3), complex_storage=True,
                                 offdiag=True, pml_thickness=2), "plain", False),
    )
    for name, config, tail, expected in configurations:
        fields, pml, _ = P.build(cp, **config)
        verdict = (arm.complex_folded_offdiag_update_e_coverage(
            fields, pml, probe) if tail == "folded" else
            arm.complex_no_pml_offdiag_update_e_coverage(fields, pml, probe))
        cases.append({"case": name, "tail": tail, "covered": verdict.covered,
                      "expected": expected, "agrees": verdict.covered == expected,
                      "reasons": list(verdict.reasons)})
    return {"cases": cases, "all_agree": all(row["agrees"] for row in cases)}


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases", [])
    products = [row for row in rows if row.get("case", "").startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in row.get("case", "")]
    if len(products) != 5:
        reasons.append(f"expected 5 products, got {len(products)}")
    if len(mutations) != 5:
        reasons.append(f"expected 5 mutations, got {len(mutations)}")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(f"{row.get('case')}: {row.get('verdict')} != "
                           f"{row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    if not (payload.get("refusals") or {}).get("all_agree"):
        reasons.append("a refusal predicate disagreed with its expectation")
    # A RELEASE THAT CANNOT NAME ITS POLICY DOES NOT IDENTIFY ITS BYTES: under keep
    # the kernels differ in the subnormal range from a flush run, and the weld cut
    # from this artifact takes its policy from this stamp.
    stamp = payload.get("subnormal_policy") or {}
    if not (stamp.get("installed") and stamp.get("resolved") == "keep"):
        reasons.append(f"the subnormal policy stamp does not record keep installed: "
                       f"installed={stamp.get('installed')!r} "
                       f"resolved={stamp.get('resolved')!r}")
    return {"released": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe")
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_log = open(out.parent / "driver.log", "a", encoding="utf-8")
    jsonl = open(out.parent / "cases.jsonl", "a", encoding="utf-8")
    payload: Dict[str, Any] = {
        "gate": "triton_complex_offdiag", "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe": args.probe,
    }
    from meep_gpu.subnormal_policy import install_subnormal_policy, policy_stamp
    install_subnormal_policy("keep")
    payload["subnormal_policy_installed"] = "keep"
    probe = load_probe(args.probe)
    if probe is None:
        raise RuntimeError("no complex expansion probe was supplied or configured")

    rows = []
    cases = build_cases(probe)
    for index, spec in enumerate(cases, 1):
        log(run_log, f"case {index}/{len(cases)} {spec['name']} "
                     f"predict={spec['prediction']}")
        try:
            row = P.run_case(cycles=args.cycles, **spec)
        except BaseException as exc:  # noqa: BLE001 - record and continue
            row = {"case": spec["name"], "prediction": spec["prediction"],
                   "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1600:]}
        rows.append(row)
        emit(jsonl, row)
        log(run_log, f"  -> {row.get('verdict', 'ERROR')} "
                     f"diff={row.get('differing_max', '?')} "
                     f"seconds={row.get('seconds', '?')}")
    payload["cases"] = rows
    payload["refusals"] = refusal_leg(probe)
    # THE WELD'S TWO IDENTITY FIELDS, written by the run rather than typed into the
    # ledger afterwards. Until 2026-09-15 this artifact carried neither -- the
    # batch_D run of 2026-09-13 has no `environment` and no `subnormal_policy` key --
    # so triton_complex_offdiag_device_gate read "see artifact" for its policy and
    # rebind_triton_welds.py could derive nothing. Stamped AFTER the legs, while the
    # device is open: the identity then reads the device the cases ran on, and the
    # stamp's strip counters include every compile the cases made.
    import triton_device_identity  # noqa: PLC0415
    payload["environment"] = triton_device_identity.record({})
    payload["subnormal_policy"] = policy_stamp()
    payload["release"] = validate(payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    log(run_log, f"released={payload['release']['released']} "
                 f"reasons={payload['release']['reasons']}")
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

