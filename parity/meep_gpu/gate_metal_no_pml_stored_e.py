#!/usr/bin/env python3
"""MPS byte gate for the real no-PML ``update_E → update_P`` dispersive pair.

This gate is intentionally not a curl benchmark.  It isolates the two shared-
residency plans that turn stored E into the next ADE drive, runs the NumPy
``stepping.update_E`` and ``PolarizationState.update`` oracle, and emits one
fsynced JSONL result per product or mutation.  It must be run in a process where
PyTorch reports an available MPS device; an unavailable device is a preflight
failure, never a passing empty gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, no_pml_stored_e, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _differences(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def build(shape: tuple[int, int, int], poles: int, seed: int, kind: str,
          ) -> tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(n / 8.0 for n in shape),
                courant=0.35, boundaries="periodic", xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(0.9 + index * 0.1, 0.05, kind),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.float32)
        fields.polarizations.append(state)
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=0)


def _record_difference(reference: Fields, actual: Fields) -> int:
    total = sum(_differences(getattr(reference, c), getattr(actual, c))
                for c in ("Ex", "Ey", "Ez"))
    for expected, observed in zip(reference.polarizations, actual.polarizations):
        for component in expected.driven():
            total += _differences(expected.P[component], observed.P[component])
            total += _differences(expected.P_prev[component], observed.P_prev[component])
        total += _differences(expected._scratch, observed._scratch)
    return total


def _normal_functions() -> dict[tuple[str, int], Any]:
    return {
        (shaders.CONTRACT_OFF, count):
        compile_source(no_pml_stored_e.stored_e_source(count)).stored_e_component
        for count in (0, 1, 2)
    }


def _mutant_functions() -> dict[str, dict[tuple[str, int], Any]]:
    """Build independently wrong two-pole bodies for the exact shipped ABI."""
    source = no_pml_stored_e.stored_e_source(2)
    mutations = {
        "drop_second_pole": source.replace(
            "source = source - p1[idx];", "// dropped p1", 1),
        "reverse_pole_order": source.replace(
            "source = source - p0[idx];\n    source = source - p1[idx];",
            "source = source - p1[idx];\n    source = source - p0[idx];", 1),
        "omit_inverse_epsilon": source.replace(
            "e_out[idx] = source * inv_e[idx];", "e_out[idx] = source;", 1),
    }
    normal = _normal_functions()
    return {
        label: {
            **normal,
            (shaders.CONTRACT_OFF, 2): compile_source(mutant).stored_e_component,
        }
        for label, mutant in mutations.items()
    }


def run_case(label: str, shape: tuple[int, int, int], poles: int, kind: str,
             cycles: int, functions: dict[tuple[str, int], Any] | None = None,
             mutation: bool = False) -> dict[str, Any]:
    reference, _ = build(shape, poles, 1800 + poles, kind)
    actual, pml = build(shape, poles, 1800 + poles, kind)
    residency = Residency()
    e_plan = no_pml_stored_e.plan_metal_stored_e(
        actual, pml, residency,
        functions=functions)
    p_plan = ade_update_p.plan_metal_ade_update_p(actual, pml, residency)
    if e_plan is None or p_plan is None:
        return {"label": label, "passed": False, "reason": "planner refused product"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        e_plan.run()
        p_plan.run()
        residency.sync_out()
        differing = _record_difference(reference, actual)
        reference_subnormals = sum(
            subnormal.census(getattr(reference, component))
            for component in ("Ex", "Ey", "Ez"))
        for state in reference.polarizations:
            for component in state.driven():
                reference_subnormals += subnormal.census(state.P[component])
                reference_subnormals += subnormal.census(state.P_prev[component])
            reference_subnormals += subnormal.census(state._scratch)
        per_cycle.append({"cycle": cycle, "differing_words": differing,
                          "reference_subnormals": reference_subnormals})
        if differing or reference_subnormals:
            break
    differing = per_cycle[-1]["differing_words"]
    subnormals = per_cycle[-1]["reference_subnormals"]
    passed = differing > 0 if mutation else (
        len(per_cycle) == cycles and differing == 0 and subnormals == 0)
    return {"label": label, "passed": passed, "differing_words": differing,
            "reference_subnormals": subnormals, "per_cycle": per_cycle,
            "e_launches": e_plan.launches, "p_launches": p_plan.launches,
            "residency_differences": residency.verify()}


def _emit(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{record['index']}/{record['total']} {record['label']}: "
          f"diff={record.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--out", type=Path, default=(Path(__file__).parent / "results" /
                        "metal_no_pml_stored_e_pending" / "gate.json"))
    args = parser.parse_args()
    try:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS is unavailable in this process")
    except Exception as exc:
        print(f"PRECONDITION FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.out.with_suffix(".jsonl")
    products = (("two_pole_lorentz", (17, 11, 9), 2, LORENTZIAN),
                ("two_pole_drude", (17, 11, 9), 2, DRUDE),
                ("collapsed_axis", (19, 1, 13), 2, LORENTZIAN),
                ("one_pole_control", (17, 11, 9), 1, LORENTZIAN))
    mutations = _mutant_functions()
    cases = tuple((label, shape, poles, kind, None, False)
                  for label, shape, poles, kind in products) + tuple(
                      (label, (17, 11, 9), 2, LORENTZIAN, functions, True)
                      for label, functions in mutations.items())
    started = time.perf_counter()
    records = []
    with records_path.open("w", encoding="utf-8") as handle:
        for index, (label, shape, poles, kind, functions, mutation) in enumerate(cases, 1):
            record = {"index": index, "total": len(cases), "mutation": mutation,
                      "kind": kind,
                      **run_case(label, shape, poles, kind, args.cycles,
                                 functions, mutation)}
            _emit(handle, record)
            records.append(record)
    summary = {"passed": all(row["passed"] for row in records), "records": records,
               "cycles": args.cycles, "elapsed_seconds": time.perf_counter() - started,
               "source_sha256": {
                   "family": hashlib.sha256(
                       Path(no_pml_stored_e.__file__).read_bytes()).hexdigest(),
                   "ade": hashlib.sha256(
                       Path(ade_update_p.__file__).read_bytes()).hexdigest(),
                   "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               }}
    summary["jsonl"] = str(records_path)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(summary)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
