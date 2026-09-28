#!/usr/bin/env python3
"""Native MPS byte gate for complex folded tensor-row PML ``update_E``.

This is a sub-step gate: it certifies the exact complex, Bloch, folded ghost,
off-diagonal Yee average, and stored-E PML arithmetic together.  It does not claim
that a whole FDTD step is device-resident; the arm remains deliberately unwired
until composer residency covers the neighbouring curl and fill sub-steps too.
"""

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
PARITY_ROOT = Path(__file__).resolve().parent
if str(PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARITY_ROOT))

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_folded_offdiag_update_e as family, folded_complex, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402


COMPARED = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, axes: str, mirror_phases: Sequence[int], boundaries: Any,
           k_point: Sequence[float], rows: Mapping[str, Sequence[str]],
           seed: int) -> Tuple[Fields, PML]:
    grid = Grid(
        resolution=8.0,
        cell_size=(2.0, 1.5, 1.25),
        courant=0.35,
        boundaries=boundaries,
        k_point=tuple(k_point),
        symmetry=tuple(Mirror(axis, int(phase))
                       for axis, phase in zip(axes, mirror_phases)),
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: np.full(shape, np.float32(1.0) / value, np.float32)
               for name, value in epsilon.items()}
    tensor_rows = {
        row: {partner: rng.uniform(-0.18, 0.18, shape).astype(np.float32)
              for partner in partners}
        for row, partners in rows.items()
    }
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=tensor_rows)
    folded = {"XYZ".index(axis) for axis in axes}
    pml = PML(grid=grid, thickness=tuple(
        (0, 2) if axis in folded else (2, 2) for axis in range(3)))
    for name in ("Dx", "Dy", "Dz", *COMPARED):
        getattr(fields, name)[...] = (
            rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32))
    return fields, pml


def _run(*, axes: str, mirror_phases: Sequence[int], boundaries: Any,
         k_point: Sequence[float], rows: Mapping[str, Sequence[str]], seed: int,
         cycles: int, probe: Mapping[str, Any],
         functions: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = _build(
        axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
        k_point=k_point, rows=rows, seed=seed)
    actual, actual_pml = _build(
        axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
        k_point=k_point, rows=rows, seed=seed)
    before = {name: np.array(getattr(actual, name), copy=True) for name in COMPARED}
    residency = Residency()
    plan = family.plan_metal_complex_folded_offdiag(
        actual, actual_pml, residency, probe=probe, functions=functions)
    if plan is None:
        return {"passed": False, "reason": "a declared folded complex product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        plan.run()
        residency.sync_out(COMPARED)
        difference = sum(_different(getattr(reference, name), getattr(actual, name))
                         for name in COMPARED)
        census = sum(subnormal.census(getattr(reference, name)) for name in COMPARED)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], getattr(actual, name)) for name in COMPARED)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == row["reference_subnormals"] == 0
                           for row in per_cycle)
                   and moved > 0 and plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved,
        "launches": plan.launches,
        "mirrors": len(residency.names),
    }


def _mutants(expansion: str) -> Dict[str, Mapping[str, Any]]:
    base = family.complex_folded_offdiag_source(
        (1, 1, 1, 1, 1, 1), (0, 3, 0), (0, 0, 0), (1, 0, 1), (0, 1, 0),
        expansion)
    sources = {
        "folded_ghost_sign_dropped": base.replace(
            "down_00 = (at_y ? c_mul_coefficient_left(-1.0f, down_00) : down_00);",
            "// dropped folded parity", 1),
        "partner_down_phase_dropped": base.replace(
            "down_10 = (k == 0) ? c_mul(down_10, ph[2]) : down_10;",
            "// dropped partner down phase", 1),
        "own_up_phase_dropped": base.replace(
            "far_product_00 = (i == nxi - 1) ? c_mul(far_product_00, ph[3]) : far_product_00;",
            "// dropped own up phase", 1),
        "pml_accumulations_reversed": base.replace(
            "a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n    a0 = a0 - c_mul_coefficient_left(km_0, prev0);",
            "a0 = a0 - c_mul_coefficient_left(km_0, src0);\n    a0 = a0 + c_mul_coefficient_left(kp_0, prev0);", 1),
    }
    return {label: {shaders.CONTRACT_OFF: compile_source(source).complex_folded_offdiag_update_e}
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
    probe = folded_complex.load_expansion_probe()
    expansion = folded_complex.expansion_from_probe(probe)
    if expansion is None:
        raise SystemExit("no valid folded-complex expansion probe is available")

    all_rows = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")}
    products = (
        ("periodic_even_two_axis_bloch", "Y", (1,), "periodic", (0.2, 0.0, 0.125), all_rows),
        ("metallic_odd", "Y", (-1,), {"y": "metallic"}, (0.2, 0.0, 0.125),
         {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}),
        ("two_axis_fold_bloch_z", "XY", (1, -1), "periodic", (0.0, 0.0, 0.125),
         {"Ex": ("Ey",), "Ey": ("Ex", "Ez")}),
        ("sparse_row_control", "Y", (1,), "periodic", (0.2, 0.0, 0.125),
         {"Ey": ("Ez",)}),
    )
    mutations = _mutants(expansion)
    total = len(products) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, axes, mirror_phases, boundaries, k_point, tensor_rows) in enumerate(products, 1):
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "axes": axes, "mirror_phases": mirror_phases, "boundaries": boundaries,
                   "k_point": k_point, "rows": tensor_rows, "cycles": args.cycles,
                   **_run(axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
                          k_point=k_point, rows=tensor_rows, seed=37000 + index,
                          cycles=args.cycles, probe=probe)}
            rows.append(row)
            _emit(handle, row)
        mutation_rows = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex")}
        for index, (label, functions) in enumerate(mutations.items(), len(products) + 1):
            result = _run(axes="Y", mirror_phases=(1,), boundaries="periodic",
                          k_point=(0.2, 0.0, 0.125), rows=mutation_rows,
                          seed=37100 + index, cycles=1, probe=probe, functions=functions)
            difference = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "differing_words": difference, "launches": result.get("launches"),
                   "caught": difference > 0,
                   "passed": result.get("launches") == 1 and difference > 0}
            rows.append(row)
            _emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products), "mutation_rows": len(mutations), "rows": rows,
        "environment": ENVIRONMENT, "torch_version": torch.__version__,
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
