"""CUDA byte gate for complex64/Bloch conductive curls without active PML.

The eight primary rows are the four lifted Group-I ``TestLoadDump`` cases at
both ``step_B`` and ``step_D``.  All carry the exact lifted shape and Bloch
point, B and D conductivity, and a live Lorentz pole.  Additional product rows
pin D-only and B-only reach on both sub-steps.

The gate is adversarial, not a smoke test.  It must catch:

* PHASEOFF — the wrapped-plane Bloch rotation was compiled away;
* DROP_CONDUCTIVITY — the current sub-step took the lossless tail;
* D_SIGMA_LEAKS_TO_STEP_B — D coefficients were illegally applied to B;
* WRONG_EXPANSION — the opposite complex-multiply expansion was compiled;
* STALE_SOURCE — the plan bound a copy rather than the live stored source.

Every case is flushed and fsync'd to JSONL as it lands, and the summary is
atomically replaced.  Run this only on one confirmed-idle GPU with the installed
``keep`` policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for path in (str(HERE), str(API)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_residual_group_bodies as P  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

CYCLES = 8
GROUP_I_ROWS = (
    "TestLoadDump.test_load_dump_chunk_layout_file_3d",
    "TestLoadDump.test_load_dump_chunk_layout_sim_3d",
    "TestLoadDump.test_load_dump_structure_3d",
    "TestLoadDump.test_load_dump_structure_sharded_3d",
)
SUB_STEPS = {"step_B": stepping.step_B, "step_D": stepping.step_D}


def log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def save(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_configuration(
    xp: Any, *, seed_offset: int = 0, conductivity: str = "both",
    kind: str = "lorentzian", poles: int = 1, active_pml: bool = False,
    complex_storage: bool = True,
):
    """The lifted Group-I grid with independently selectable B/D sigma."""
    grid = Grid(
        resolution=15.0, cell_size=(2.3, 2.1, 2.7), dimensions=3,
        boundaries="periodic", courant=0.5,
        k_point=(0.4, -1.3, 0.7), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    dtype = np.complex64 if complex_storage else np.float32
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(
                frequency=1.0 + 0.2 * index, gamma=0.1, kind=kind),
            {"Ex": 0.30, "Ey": 0.25, "Ez": 0.20}, grid, dtype)
        fields.polarizations.append(state)

    shape = tuple(grid.shape)
    host = np.linspace(
        0.12, 0.72, np.prod(shape), dtype=np.float32).reshape(shape)
    d_volume = xp.asarray(host)
    b_volume = xp.asarray((host * np.float32(0.75)).astype(np.float32))
    if conductivity in ("both", "d_only"):
        fields.set_d_conductivity(d_volume)
    elif conductivity == "mixed":
        fields.set_d_conductivity({"Dx": d_volume, "Dz": d_volume})
    if conductivity in ("both", "b_only"):
        fields.set_b_conductivity(b_volume)
    elif conductivity == "mixed":
        fields.set_b_conductivity({"By": b_volume})
    if conductivity not in {"both", "d_only", "b_only", "mixed", "none"}:
        raise ValueError(f"unknown conductivity selection {conductivity!r}")

    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    P.seed_state(fields, xp, np.random.default_rng(P.SEED + seed_offset),
                 amplitude=0.4)
    notes = {
        "shape": list(shape), "k_point": list(grid.k_point),
        "complex_storage": complex_storage, "conductivity": conductivity,
        "pml_active": bool(pml.is_active), "kind": kind, "poles": poles,
        "stores_E": bool(fields.stores_E),
    }
    return fields, pml, notes


def make_factory(xp: Any, **configuration):
    return lambda offset: build_configuration(
        xp, seed_offset=offset, **configuration)


def shipped_plan(fields: Any, pml: Any, sub_step: str, probe: Dict[str, Any]):
    from meep_gpu.triton_kernels import complex_no_pml_conductive as arm

    verdict = arm.complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step, probe=probe)
    if not verdict.covered:
        raise RuntimeError(f"shipped predicate refused: {verdict.reasons}")
    plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, sub_step, probe=probe)
    if plan is None:
        raise RuntimeError("shipped builder returned None after coverage admitted")
    return plan


def phase_off_plan(fields: Any, pml: Any, sub_step: str, probe: Dict[str, Any]):
    plan = shipped_plan(fields, pml, sub_step, probe)
    plan.phased = (0, 0, 0)
    plan.phase_values = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    return plan


def drop_conductivity_plan(
    fields: Any, pml: Any, sub_step: str, probe: Dict[str, Any],
):
    plan = shipped_plan(fields, pml, sub_step, probe)
    plan.cond = (0, 0, 0)
    return plan


def leak_d_sigma_into_b_plan(
    fields: Any, pml: Any, _sub_step: str, probe: Dict[str, Any],
):
    """The exact forbidden cross-side reach: D sigma applied during step_B."""
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat

    plan = shipped_plan(fields, pml, "step_B", probe)
    plan.cond = (1, 1, 1)
    plan._condfac = tuple(
        CupyPointer(_flat(fields.condfac_for(name)))
        for name in ("Dx", "Dy", "Dz"))
    plan._condinv = tuple(
        CupyPointer(_flat(fields.condinv_for(name)))
        for name in ("Dx", "Dy", "Dz"))
    return plan


def wrong_expansion_plan(
    fields: Any, pml: Any, sub_step: str, probe: Dict[str, Any],
):
    plan = shipped_plan(fields, pml, sub_step, probe)
    plan.expansion = 1 - int(plan.expansion)
    return plan


def stale_source_plan(
    fields: Any, pml: Any, sub_step: str, probe: Dict[str, Any],
):
    from meep_gpu.triton_kernels import complex_fields
    from meep_gpu.triton_kernels.complex_no_pml_curl import source_arrays

    plan = shipped_plan(fields, pml, sub_step, probe)
    frozen = [array.copy() for array in source_arrays(fields, sub_step)]
    plan._sources = tuple(
        complex_fields.CupyPointer(complex_fields._word_view(array))
        for array in frozen)
    return plan


def case(
    name: str, sub_step: str, prediction: str, make, builder, clause: str,
) -> Dict[str, Any]:
    return {
        "name": name, "group": "I", "sub_step": sub_step,
        "prediction": prediction, "clause": clause, "make": make,
        "array_path": SUB_STEPS[sub_step], "kernel_plan": builder,
    }


def build_cases(xp: Any, probe: Dict[str, Any]) -> List[Dict[str, Any]]:
    target = make_factory(xp, conductivity="both", poles=1)
    cases: List[Dict[str, Any]] = []
    for row in GROUP_I_ROWS:
        for sub_step in SUB_STEPS:
            cases.append(case(
                "product_" + row.replace(".", "__") + "_" + sub_step,
                sub_step, "IDENTICAL", target,
                lambda f, p, s=sub_step, record=probe:
                    shipped_plan(f, p, s, record),
                "one-to-one lifted Group-I curl slot with a live Lorentz pole"))

    # The current sub-step flags may all be false even though run-wide admission
    # selects this family.  Both directions are product claims, not mutations.
    for selection in ("d_only", "b_only", "mixed"):
        make = make_factory(xp, conductivity=selection, poles=2,
                            kind="drude" if selection == "mixed" else "lorentzian")
        for sub_step in SUB_STEPS:
            cases.append(case(
                f"product_{selection}_{sub_step}", sub_step, "IDENTICAL", make,
                lambda f, p, s=sub_step, record=probe:
                    shipped_plan(f, p, s, record),
                "per-target conductivity reach with registered poles"))

    for sub_step in SUB_STEPS:
        cases.append(case(
            f"CONTROL_PHASEOFF_{sub_step}", sub_step, "DIVERGENT", target,
            lambda f, p, s=sub_step, record=probe:
                phase_off_plan(f, p, s, record),
            "control: Bloch wrapped-plane rotation compiled away"))
        cases.append(case(
            f"MUTATION_DROP_CONDUCTIVITY_{sub_step}", sub_step, "DIVERGENT",
            target,
            lambda f, p, s=sub_step, record=probe:
                drop_conductivity_plan(f, p, s, record),
            "mutation: current-side conductive tail compiled away"))
        cases.append(case(
            f"MUTATION_WRONG_EXPANSION_{sub_step}", sub_step, "DIVERGENT",
            target,
            lambda f, p, s=sub_step, record=probe:
                wrong_expansion_plan(f, p, s, record),
            "mutation: opposite measured complex-multiply expansion"))

    d_only = make_factory(xp, conductivity="d_only", poles=1)
    cases.append(case(
        "MUTATION_D_SIGMA_LEAKS_TO_STEP_B", "step_B", "DIVERGENT", d_only,
        lambda f, p, record=probe:
            leak_d_sigma_into_b_plan(f, p, "step_B", record),
        "mutation: D conductivity illegally reaches a B curl term"))
    cases.append(case(
        "CONTROL_STALE_SOURCE_step_B", "step_B", "DIVERGENT", target,
        lambda f, p, record=probe:
            stale_source_plan(f, p, "step_B", record),
        "control: source bound as a plan-time copy"))
    return cases


def refusal_leg(xp: Any, probe: Dict[str, Any]) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_no_pml_conductive as arm
    from meep_gpu.triton_kernels import complex_no_pml_curl as incumbent

    rows: List[Dict[str, Any]] = []
    fixtures = (
        ("target", {}, True),
        ("active_pml", {"active_pml": True}, False),
        ("real_storage", {"complex_storage": False}, False),
        ("no_conductivity", {"conductivity": "none"}, False),
    )
    for label, configuration, expected in fixtures:
        fields, pml, _ = build_configuration(xp, **configuration)
        for sub_step in SUB_STEPS:
            verdict = arm.complex_conductive_no_pml_curl_coverage(
                fields, pml, sub_step, probe=probe)
            old = incumbent.complex_no_pml_curl_coverage(
                fields, pml, sub_step, probe=probe)
            row = {
                "case": f"{label}:{sub_step}", "covered": verdict.covered,
                "expected": expected,
                "agrees": verdict.covered == expected,
                "reasons": list(verdict.reasons),
                "incumbent_covered": old.covered,
                "both_admit": verdict.covered and old.covered,
            }
            rows.append(row)

    # The public Susceptibility constructor correctly rejects unsupported
    # kinds, so plant the malformed state only after constructing a valid one.
    # This asks the predicate's defensive clause without inventing a public
    # configuration the engine itself would accept.
    from types import SimpleNamespace

    fields, pml, _ = build_configuration(xp)
    fields.polarizations[0].susceptibility = SimpleNamespace(kind="gyrotropic")
    for sub_step in SUB_STEPS:
        verdict = arm.complex_conductive_no_pml_curl_coverage(
            fields, pml, sub_step, probe=probe)
        rows.append({
            "case": f"unsupported_pole:{sub_step}",
            "covered": verdict.covered, "expected": False,
            "agrees": not verdict.covered, "reasons": list(verdict.reasons),
            "incumbent_covered": False, "both_admit": False,
        })

    fields, pml, _ = build_configuration(xp)
    missing = arm.complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_B", probe={})
    rows.append({
        "case": "missing_expansion_probe:step_B", "covered": missing.covered,
        "expected": False, "agrees": not missing.covered,
        "reasons": list(missing.reasons), "incumbent_covered": False,
        "both_admit": False,
    })
    return {
        "cases": rows,
        "all_agree": all(row["agrees"] for row in rows),
        "disjoint": not any(row["both_admit"] for row in rows),
    }


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows
                if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in str(row.get("case", ""))]
    controls = [row for row in rows if "CONTROL" in str(row.get("case", ""))]
    if len(products) != 14:
        reasons.append(f"expected 14 products (8 corpus + 6 reach), found {len(products)}")
    if len(mutations) != 5:
        reasons.append(f"expected 5 mutations, found {len(mutations)}")
    if len(controls) != 3:
        reasons.append(f"expected 3 controls, found {len(controls)}")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(
                f"{row.get('case')}: {row.get('verdict')} != {row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    if payload.get("expansion") not in (0, 1):
        reasons.append("the complex expansion probe licensed no arm")
    refusal = payload.get("refusals") or {}
    if not refusal.get("all_agree"):
        reasons.append("a coverage/refusal expectation disagreed")
    if not refusal.get("disjoint"):
        reasons.append("the new and incumbent predicates overlap")
    return {"released": not reasons, "reasons": reasons}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_log = (out.parent / "driver.log").open("a", encoding="utf-8")
    jsonl = (out.parent / "cases.jsonl").open("a", encoding="utf-8")
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "triton_complex_no_pml_conductive",
        "cycles": args.cycles, "group_i_rows": list(GROUP_I_ROWS),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import cupy as cp
        from meep_gpu.subnormal_policy import install_subnormal_policy
        from meep_gpu.triton_kernels import complex_fields

        install_subnormal_policy("keep")
        payload["subnormal_policy_installed"] = "keep"
        probe = P._load_expansion_probe()
        payload["expansion"] = complex_fields._resolve_expansion(probe)
        if payload["expansion"] is None:
            raise RuntimeError("the installed expansion artifact licenses no arm")
        cases = build_cases(cp, probe)
        rows = []
        for index, spec in enumerate(cases, 1):
            log(run_log, f"case {index}/{len(cases)} {spec['name']} "
                         f"predict={spec['prediction']}")
            try:
                row = P.run_case(cycles=args.cycles, **spec)
            except BaseException as exc:  # noqa: BLE001 - artifact and continue
                row = {
                    "case": spec["name"], "prediction": spec["prediction"],
                    "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:500],
                    "traceback": traceback.format_exc()[-1800:],
                }
            rows.append(row)
            emit(jsonl, row)
            log(run_log, f"  -> {row.get('verdict', 'ERROR')} "
                         f"diff={row.get('differing_max', '?')} "
                         f"elapsed={time.time() - started:.1f}s")
            payload["cases"] = rows
            save(out, payload)
        payload["refusals"] = refusal_leg(cp, probe)
    except BaseException as exc:  # noqa: BLE001 - setup failure is an artifact
        payload["setup_error"] = f"{type(exc).__name__}: {exc}"
        payload["setup_traceback"] = traceback.format_exc()[-3000:]
        log(run_log, f"SETUP ERROR: {payload['setup_error']}")
    payload["seconds"] = round(time.time() - started, 2)
    payload["release"] = validate(payload)
    if payload.get("setup_error"):
        payload["release"]["released"] = False
        payload["release"]["reasons"].append(payload["setup_error"])
    save(out, payload)
    log(run_log, f"RELEASED={payload['release']['released']}")
    for reason in payload["release"]["reasons"]:
        log(run_log, f"  refused: {reason}")
    jsonl.close()
    run_log.close()
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
