#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch lossless no-PML curls.

The gate runs the NumPy ``stepping.step_B``/``step_D`` oracle beside persistent
Metal mirrors.  It intentionally covers only the curl pair: a whole-driver
residency seam remains a distinct certification step.  One JSONL row is flushed
after every product or mutation so an interrupted Apple-GPU run retains all
completed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import complex_no_pml_curl as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

PROBE = {
    "backend": "numpy",
    "patterns": {
        name: "FMA_V1"
        for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left",
        )
    },
}


def words(value: Any) -> np.ndarray:
    """View all float32 words without changing complex layout."""
    array = np.ascontiguousarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.complex64)):
        raise TypeError(f"expected float32 or complex64, got {array.dtype}")
    return array.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build(*, seed: int, cell_size: Sequence[float], phase: Sequence[float],
          boundaries: Sequence[str]) -> Tuple[Fields, PML]:
    """Build one live complex lossless state with nontrivial curl operands."""
    grid = Grid(resolution=8.0, cell_size=tuple(cell_size), courant=0.3,
                boundaries=tuple(boundaries), k_point=tuple(phase), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (
                rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
                + 1j * rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32))
    return fields, PML(grid=grid, thickness=0)


def compared_names(fields: Fields) -> Tuple[str, ...]:
    return ("Bx", "By", "Bz", "Dx", "Dy", "Dz")


def subnormal_words(fields: Fields, names: Iterable[str]) -> int:
    return sum(subnormal.census(getattr(fields, name)) for name in names)


def execute_case(*, seed: int, cycles: int, cell_size: Sequence[float],
                 phase: Sequence[float], boundaries: Sequence[str],
                 b_functions: Mapping[str, Any] | None = None,
                 ) -> Dict[str, Any]:
    """Run B then D beside their exact NumPy sub-step oracle."""
    reference, reference_pml = build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    actual, actual_pml = build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    names = compared_names(actual)
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    residency = Residency()
    b_plan = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_B", residency, probe=PROBE, functions=b_functions)
    d_plan = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_D", residency, probe=PROBE)
    if b_plan is None or d_plan is None:
        return {"passed": False, "reason": "a declared no-PML product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        b_plan.run()
        import torch  # noqa: PLC0415

        torch.mps.synchronize()
        residency.sync_out(names)
        stepping.step_B(reference, reference_pml)
        b_diff = sum(differing(getattr(reference, name), getattr(actual, name))
                     for name in names)
        d_plan.run()
        torch.mps.synchronize()
        stepping.step_D(reference, reference_pml)
        residency.sync_out(names)
        diff = sum(differing(getattr(reference, name), getattr(actual, name))
                   for name in names)
        census = subnormal_words(reference, names)
        per_cycle.append({"cycle": cycle, "b_differing_words": b_diff,
                          "d_differing_words": diff - b_diff,
                          "differing_words": diff,
                          "reference_subnormals": census})
        if diff or census:
            break
    moved = sum(differing(before[name], getattr(actual, name)) for name in names)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == 0
                           and row["reference_subnormals"] == 0 for row in per_cycle)
                   and moved > 0 and b_plan.launches == len(per_cycle)
                   and d_plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved,
        "b_launches": b_plan.launches,
        "d_launches": d_plan.launches,
        "mirrors": len(residency.names),
    }


def mutation_sources() -> Dict[str, str]:
    """Independent source faults, all exercised on a phased periodic B row."""
    source = family.complex_no_pml_curl_source(
        (0, 0, 0), False, (1, 1, 0), "FMA_V1")
    wrong_tail = source
    for index in range(3):
        wrong_tail = wrong_tail.replace(
            f"f{index}[ii] = f{index}[ii] - curl{index};",
            f"f{index}[ii] = f{index}[ii] + curl{index};", 1)
    flat_curl = source.replace(
        "float2 t0 = ((c_y - c) + (b - b_z));",
        "float2 t0 = (c_y - c + b - b_z);", 1)
    dropped_phase = source.replace(
        "b_x = wx ? c_mul(b_x, px) : b_x;", "// dropped b_x phase", 1)
    return {
        "curl_subtraction_reversed": wrong_tail,
        "curl_grouping_flattened": flat_curl,
        "bloch_x_rotation_dropped": dropped_phase,
    }


def source_provenance() -> Dict[str, Any]:
    specializations = {}
    for codes, phased in (((0, 0, 0), (1, 1, 0)),
                          ((1, 0, 1), (0, 1, 0)),
                          ((0, 0, 0), (0, 0, 0))):
        for sub_step in ("step_B", "step_D"):
            source = family.complex_no_pml_curl_source(
                codes, sub_step == "step_D", phased, "FMA_V1")
            specializations[f"{sub_step}/{codes}/{phased}"] = sha256_bytes(
                source.encode("utf-8"))
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "shared_stencil": sha256_file(Path(family.__file__).resolve().with_name(
                "complex_no_pml_conductive.py")),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": specializations,
        "mutation_metal_sources": {
            label: sha256_bytes(source.encode("utf-8"))
            for label, source in mutation_sources().items()
        },
    }


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['label']}: "
          f"diff={row.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Final JSON artifact; the sibling JSONL is incremental")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be positive")
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    products = (
        ("two_axis_bloch", (1.0, 0.875, 0.75), (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("zero_phase_control", (1.0, 0.875, 0.75), (0.0, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
        ("metallic_xz_bloch_y", (1.0, 0.875, 0.75), (0.0, 0.125, 0.0),
         ("metallic", "periodic", "metallic")),
        ("collapsed_y", (1.25, 0.0625, 0.75), (0.2, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
    )
    mutations = mutation_sources()
    total = len(products) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, cell_size, phase, boundaries) in enumerate(products, 1):
            result = execute_case(seed=20000 + index, cycles=args.cycles,
                                  cell_size=cell_size, phase=phase,
                                  boundaries=boundaries)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "cell_size": cell_size, "phase": phase, "boundaries": boundaries,
                   "cycles": args.cycles, **result}
            rows.append(row)
            emit(handle, row)
        offset = len(products)
        for number, (label, source) in enumerate(mutations.items(), 1):
            function = compile_source(source).complex_no_pml_conductive_curl_step
            result = execute_case(seed=21000 + number, cycles=1,
                                  cell_size=(1.0, 0.875, 0.75),
                                  phase=(0.2, -0.125, 0.0),
                                  boundaries=("periodic", "periodic", "periodic"),
                                  b_functions={shaders.CONTRACT_OFF: function})
            diff = int(result.get("differing_words", 0))
            row = {"index": offset + number, "total": total, "leg": "mutation",
                   "label": label, "differing_words": diff,
                   "b_launches": result.get("b_launches"),
                   "caught": diff > 0,
                   "passed": bool(result.get("b_launches") == 1 and diff > 0)}
            rows.append(row)
            emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products),
        "mutation_rows": len(mutations),
        "rows": rows,
        "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": subnormal.mps_policy_report("flush"),
        "source_provenance": source_provenance(),
        "jsonl": str(jsonl),
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
          f"artifact {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
