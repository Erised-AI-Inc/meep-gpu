#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch no-PML tensor-row ``update_E``."""

from __future__ import annotations

import argparse
import hashlib
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
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_no_pml_offdiag_update_e as family, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

PROBE = {"backend": "numpy", "patterns": {
    name: "FMA_V1" for name in (
        "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
        "f4_mul_c8_coefficient_left", "python_float_left")}}


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, seed: int, cell_size: Sequence[float], phase: Sequence[float],
           boundaries: Sequence[str]) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(cell_size), courant=0.35,
                boundaries=tuple(boundaries), k_point=tuple(phase), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = np.full(shape, np.float32(2.25))
    inverse = np.full(shape, np.float32(1.0 / 2.25))
    rows = {
        "Ex": {"Ey": (0.04 * rng.standard_normal(shape)).astype(np.float32),
               "Ez": (0.03 * rng.standard_normal(shape)).astype(np.float32)},
        "Ey": {"Ez": (0.02 * rng.standard_normal(shape)).astype(np.float32),
               "Ex": (0.05 * rng.standard_normal(shape)).astype(np.float32)},
        "Ez": {"Ex": (0.01 * rng.standard_normal(shape)).astype(np.float32),
               "Ey": (0.06 * rng.standard_normal(shape)).astype(np.float32)},
    }
    fields.set_epsilon_volumes(
        {name: epsilon for name in ("Ex", "Ey", "Ez")},
        {name: inverse for name in ("Ex", "Ey", "Ez")}, rows)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = (
            rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32))
    return fields, PML(grid=grid, thickness=0)


def _run(*, seed: int, cycles: int, cell_size: Sequence[float], phase: Sequence[float],
         boundaries: Sequence[str], functions: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = _build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    actual, actual_pml = _build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    names = ("Ex", "Ey", "Ez")
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    residency = Residency()
    plan = family.plan_metal_complex_no_pml_offdiag(
        actual, actual_pml, residency, probe=PROBE, functions=functions)
    if plan is None:
        return {"passed": False, "reason": "a declared complex tensor product was refused"}
    residency.sync_in()
    per_cycle = []
    import torch

    for cycle in range(1, cycles + 1):
        plan.run()
        torch.mps.synchronize()
        stepping.update_E(reference, reference_pml)
        residency.sync_out(names)
        difference = sum(_different(getattr(reference, name), getattr(actual, name))
                         for name in names)
        census = sum(subnormal.census(getattr(reference, name)) for name in names)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], getattr(actual, name)) for name in names)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == row["reference_subnormals"] == 0
                           for row in per_cycle)
                   and moved > 0 and plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved, "launches": plan.launches,
        "mirrors": len(residency.names),
    }


def _mutants() -> Dict[str, Mapping[str, Any]]:
    base = family.complex_no_pml_offdiag_source(
        (1, 1, 1, 1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 0), "FMA_V1")
    sources = {
        "first_tensor_row_dropped": base.replace(
            "float2 total0 = term_00;", "float2 total0 = float2(0.0f, 0.0f);", 1),
        "partner_down_phase_dropped": base.replace(
            "down_00 = (j == 0) ? c_mul(down_00, dpy) : down_00;", "// dropped down phase", 1),
        "own_up_product_phase_dropped": base.replace(
            "far_product_00 = (i == nxi - 1) ? c_mul(far_product_00, upx) : far_product_00;",
            "// dropped product up phase", 1),
    }
    return {label: {shaders.CONTRACT_OFF: compile_source(source).complex_no_pml_offdiag_update_e}
            for label, source in sources.items()}


def _emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['label']}: "
          f"diff={row.get('differing_words', '-')}", flush=True)


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

    products = (
        ("two_axis_bloch", (1.125, 0.875, 0.75), (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("zero_phase_control", (1.125, 0.875, 0.75), (0.0, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
        ("metallic_xz_bloch_y", (1.125, 0.875, 0.75), (0.0, 0.125, 0.0),
         ("metallic", "periodic", "metallic")),
        ("collapsed_y", (1.125, 0.125, 0.75), (0.2, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
    )
    mutations = _mutants()
    total = len(products) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, cell_size, phase, boundaries) in enumerate(products, 1):
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "cell_size": cell_size, "phase": phase, "boundaries": boundaries,
                   "cycles": args.cycles,
                   **_run(seed=34000 + index, cycles=args.cycles, cell_size=cell_size,
                          phase=phase, boundaries=boundaries)}
            rows.append(row)
            _emit(handle, row)
        for index, (label, functions) in enumerate(mutations.items(), len(products) + 1):
            result = _run(seed=35000 + index, cycles=1, cell_size=(1.125, 0.875, 0.75),
                          phase=(0.2, -0.125, 0.0),
                          boundaries=("periodic", "periodic", "periodic"),
                          functions=functions)
            difference = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "differing_words": difference, "launches": result.get("launches"),
                   "caught": difference > 0,
                   "passed": result.get("launches") == 1 and difference > 0}
            rows.append(row)
            _emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started, "product_rows": len(products),
        "mutation_rows": len(mutations), "rows": rows, "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}",
          flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
