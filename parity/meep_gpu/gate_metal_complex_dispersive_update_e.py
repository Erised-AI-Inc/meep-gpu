#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch PML dispersive ``update_E → update_P``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import ade_update_p, complex_dispersive_update_e as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source, metal_frontend_version  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

PROBE = {"backend": "numpy", "patterns": {name: "FMA_V1" for name in (
    "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
    "f4_mul_c8_coefficient_left", "python_float_left")}}


def words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def build(*, shape: Sequence[int], poles: int, kind: str, seed: int) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(float(n) / 8.0 for n in shape),
                courant=0.35, boundaries="periodic", k_point=(0.2, -0.1, 0.15), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    states = []
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(0.9 + index * 0.1, 0.05 + index * 0.01, kind),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.complex64)
        fields.polarizations.append(state)
        states.append(state)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=1)
    rng = np.random.default_rng(seed)

    def values(scale: float) -> np.ndarray:
        return (rng.uniform(-scale, scale, grid.shape).astype(np.float32)
                + 1j * rng.uniform(-scale, scale, grid.shape).astype(np.float32))

    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = values(0.4)
    for state in states:
        for component in state.driven():
            state.P[component][...] = values(0.2)
            state.P_prev[component][...] = values(0.2)
    return fields, pml


def named(fields: Fields) -> Tuple[str, ...]:
    values = ["Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            values.extend((f"P[{index}].{component}", f"P_prev[{index}].{component}"))
        values.append(f"scratch[{index}]")
    return tuple(values)


def array(fields: Fields, label: str) -> Any:
    if label in {"Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"}:
        return getattr(fields, label)
    if label.startswith("scratch["):
        return fields.polarizations[int(label[8:-1])]._scratch
    head, component = label.split(".", 1)
    index = int(head[head.index("[") + 1:head.index("]")])
    return (fields.polarizations[index].P_prev if head.startswith("P_prev")
            else fields.polarizations[index].P)[component]


def execute(*, shape: Sequence[int], poles: int, kind: str, seed: int, cycles: int,
            functions: Mapping[Tuple[str, int, int, str], Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = build(shape=shape, poles=poles, kind=kind, seed=seed)
    actual, actual_pml = build(shape=shape, poles=poles, kind=kind, seed=seed)
    names = named(actual)
    before = {name: np.array(array(actual, name), copy=True) for name in names}
    residency = Residency()
    e_plan = family.plan_metal_complex_dispersive_e(
        actual, actual_pml, residency, probe=PROBE, functions=functions)
    p_plan = ade_update_p.plan_metal_ade_update_p(actual, actual_pml, residency)
    if e_plan is None or p_plan is None:
        return {"passed": False, "reason": "declared dispersive product was refused"}
    residency.sync_in()
    rows = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        e_plan.run()
        p_plan.run()
        residency.sync_out()
        diff = sum(different(array(reference, name), array(actual, name)) for name in names)
        census = sum(subnormal.census(array(reference, name)) for name in names)
        rows.append({"cycle": cycle, "differing_words": diff, "reference_subnormals": census})
        if diff or census:
            break
    moved = sum(different(before[name], array(actual, name)) for name in names)
    return {"passed": len(rows) == cycles and all(
        item["differing_words"] == item["reference_subnormals"] == 0 for item in rows)
        and moved > 0 and e_plan.runs == p_plan.runs == len(rows),
        "per_cycle": rows, "differing_words": rows[-1]["differing_words"],
        "reference_subnormals": rows[-1]["reference_subnormals"], "moved_words": moved,
        "e_launches": e_plan.launches, "p_launches": p_plan.launches,
        "mirrors": len(residency.names)}


def mutants() -> Dict[str, Mapping[Tuple[str, int, int, str], Any]]:
    normal = {(shaders.CONTRACT_OFF, count, axis, "FMA_V1"):
              compile_source(family.complex_dispersive_e_source(count, axis, "FMA_V1"))
              .complex_dispersive_e_component
              for count in (0, 2) for axis in range(3)}
    base = family.complex_dispersive_e_source(2, 0, "FMA_V1")
    sources = {
        "second_pole_dropped": base.replace("source = source - p1[idx];", "// dropped p1", 1),
        "fw_store_dropped": base.replace("fw[idx] = src;", "// stale fw", 1),
        "pml_kps_kms_order_reversed": base.replace(
            "value = value + c_mul_coefficient_left(kps[i], src);\n    value = value - c_mul_coefficient_left(kms[i], prev);",
            "value = value - c_mul_coefficient_left(kms[i], src);\n    value = value + c_mul_coefficient_left(kps[i], prev);", 1),
    }
    out = {}
    for label, source in sources.items():
        functions = dict(normal)
        functions[(shaders.CONTRACT_OFF, 2, 0, "FMA_V1")] = (
            compile_source(source).complex_dispersive_e_component)
        out[label] = functions
    return out


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['label']}: diff={row.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    products = (("two_pole_lorentz", (9, 7, 5), 2, LORENTZIAN),
                ("two_pole_drude", (9, 7, 5), 2, DRUDE),
                ("narrow_y", (11, 3, 7), 2, LORENTZIAN))
    defects = mutants()
    total = len(products) + len(defects)
    rows = []
    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, shape, poles, kind) in enumerate(products, 1):
            result = execute(shape=shape, poles=poles, kind=kind, seed=28000 + index, cycles=args.cycles)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "shape": shape, "poles": poles, "kind": kind, "cycles": args.cycles, **result}
            rows.append(row)
            emit(handle, row)
        for index, (label, functions) in enumerate(defects.items(), len(products) + 1):
            result = execute(shape=(9, 7, 5), poles=2, kind=LORENTZIAN, seed=29000 + index,
                             cycles=1, functions=functions)
            diff = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "differing_words": diff, "e_launches": result.get("e_launches"),
                   "caught": diff > 0, "passed": result.get("e_launches") == 3 and diff > 0}
            rows.append(row)
            emit(handle, row)
    result = {"verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
              "elapsed_seconds": time.perf_counter() - started, "product_rows": len(products),
              "mutation_rows": len(defects), "rows": rows, "torch_version": torch.__version__,
              "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl)}
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
