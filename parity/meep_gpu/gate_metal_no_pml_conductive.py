#!/usr/bin/env python3
"""Native MPS byte gate for real conductive no-PML B/D curls.

The products run the shipped plans (each bound to the one-rank chunk record of its
grid, ``chunk_map.chunk_bounds_record``, whose presence word picks the curl's stencil
form) against ``stepping.step_B``/``step_D`` word for word, per cycle. Three 3-D rows
(both partners on every axis) and one 2-D row (the z partner absent, so the one-partner
forms run). The mutations must each diverge: the coefficient pair swapped, a target
store dropped, the curl summed PAIRWISE (``dtdx * ((g1s - g1) + (g2 - g2s))``, the form
this kernel carried before it took MEEP's left-to-right sum; a flattened sum is no longer
a defect, since it parses as MEEP's grouping), and, on the 2-D row, the z partner forced
present.
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

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import no_pml_conductive as family  # noqa: E402
from meep_gpu.metal_kernels import chunk_bounds as record_block  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source, metal_frontend_version  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


def words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def build(*, sides: Sequence[str], mixed: bool, stored_e: bool, seed: int,
          boundaries: Sequence[str], cell_size: Sequence[float],
          dimensions: int = 3) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(cell_size), courant=0.3,
                boundaries=tuple(boundaries), xp=np, dimensions=dimensions)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    if stored_e:
        fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity({"Bx": sigma if not mixed else None,
                                   "By": sigma,
                                   "Bz": sigma if not mixed else None})
    if "D" in sides:
        fields.set_d_conductivity({"Dx": sigma,
                                   "Dy": None if mixed else sigma,
                                   "Dz": None if mixed else sigma})
    return fields, PML(grid=grid, thickness=0)


def execute(*, sides: Sequence[str], mixed: bool, stored_e: bool, seed: int,
            cycles: int, boundaries: Sequence[str], cell_size: Sequence[float],
            dimensions: int = 3,
            b_functions: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = build(
        sides=sides, mixed=mixed, stored_e=stored_e, seed=seed,
        boundaries=boundaries, cell_size=cell_size, dimensions=dimensions)
    actual, actual_pml = build(
        sides=sides, mixed=mixed, stored_e=stored_e, seed=seed,
        boundaries=boundaries, cell_size=cell_size, dimensions=dimensions)
    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    residency = Residency()
    # THE ENGINE'S OWN RECORD for this grid: what the composer hands the arm.
    record = chunk_bounds_record(ChunkLayout.one_rank(actual.grid, actual_pml))
    b = family.plan_metal_conductive_plain_curl(
        actual, actual_pml, "step_B", residency, functions=b_functions,
        chunk_bounds=record)
    d = family.plan_metal_conductive_plain_curl(actual, actual_pml, "step_D", residency,
                                                chunk_bounds=record)
    if b is None or d is None:
        return {"passed": False, "reason": "declared conductive product was refused"}
    residency.sync_in()
    rows = []
    for cycle in range(1, cycles + 1):
        b.run()
        d.run()
        stepping.step_B(reference, reference_pml)
        stepping.step_D(reference, reference_pml)
        residency.sync_out(names)
        diff = sum(differing(getattr(reference, name), getattr(actual, name)) for name in names)
        census = sum(subnormal.census(getattr(reference, name)) for name in names)
        rows.append({"cycle": cycle, "differing_words": diff, "reference_subnormals": census})
        if diff or census:
            break
    moved = sum(differing(before[name], getattr(actual, name)) for name in names)
    return {"passed": len(rows) == cycles and all(
        row["differing_words"] == row["reference_subnormals"] == 0 for row in rows)
        and moved > 0 and b.launches == d.launches == len(rows),
        "per_cycle": rows, "differing_words": rows[-1]["differing_words"],
        "reference_subnormals": rows[-1]["reference_subnormals"], "moved_words": moved,
        "b_flags": list(b.conductive), "d_flags": list(d.conductive),
        "presence_word": int(np.asarray(record)[record_block.RECORD_PRESENCE_WORD]),
        "b_launches": b.launches, "d_launches": d.launches, "mirrors": len(residency.names)}


def _needle(source: str, old: str, new: str) -> str:
    """``source`` with ``old`` replaced ONCE; a needle that misses is a harness error."""
    if source.count(old) != 1:
        raise SystemExit(f"mutation needle {old!r} matches {source.count(old)} times, "
                         f"expected 1")
    return source.replace(old, new, 1)


def mutations() -> Dict[str, Tuple[str, int]]:
    """``label -> (mutated source, dimensions of the row it runs on)``."""
    source = family.conductive_plain_curl_source((0, 0, 0), False, True, (True, True, True))
    swapped = source.replace("cf0[ii]", "TEMP_CF0", 1).replace(
        "ci0[ii]", "cf0[ii]", 1).replace("TEMP_CF0", "ci0[ii]", 1)
    drop_store = _needle(source, "f0[ii] = value0;", "// dropped target store")
    pairwise = _needle(source, "dtdx * (((c_y - c) + b) - b_z)",
                       "dtdx * ((c_y - c) + (b - b_z))")
    present_z = _needle(source, record_block.presence_lines((2,)),
                        "    const bool has_z = true;")
    return {"condfac_condinv_swapped": (swapped, 3), "target_store_dropped": (drop_store, 3),
            "pairwise_grouping": (pairwise, 3), "z_partner_forced_present": (present_z, 2)}


def emit(handle, row: Dict[str, Any]) -> None:
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
        ("two_sided_derived", ("B", "D"), False, False,
         ("periodic", "periodic", "periodic"), (1.0, 0.875, 0.75), 3),
        ("D_only_derived", ("D",), False, False,
         ("periodic", "periodic", "periodic"), (1.0, 0.875, 0.75), 3),
        ("mixed_stored_e_metallic", ("B", "D"), True, True,
         ("metallic", "periodic", "metallic"), (1.25, 0.0625, 0.75), 3),
        ("two_sided_derived_2d", ("B", "D"), False, False,
         ("periodic", "periodic", "periodic"), (1.0, 0.875, 0.0), 2),
    )
    defects = mutations()
    total = len(products) + len(defects)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, sides, mixed, stored_e, boundaries, cell_size,
                    dimensions) in enumerate(products, 1):
            result = execute(sides=sides, mixed=mixed, stored_e=stored_e, seed=26000 + index,
                             cycles=args.cycles, boundaries=boundaries, cell_size=cell_size,
                             dimensions=dimensions)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "sides": sides, "mixed": mixed, "stored_e": stored_e,
                   "boundaries": boundaries, "cell_size": cell_size,
                   "dimensions": dimensions, **result}
            rows.append(row)
            emit(handle, row)
        for number, (label, (source, dimensions)) in enumerate(defects.items(),
                                                              len(products) + 1):
            function = compile_source(source).no_pml_conductive_curl_step
            result = execute(sides=("B", "D"), mixed=False, stored_e=False,
                             seed=27000 + number, cycles=1,
                             boundaries=("periodic", "periodic", "periodic"),
                             cell_size=(1.0, 0.875, 0.75 if dimensions == 3 else 0.0),
                             dimensions=dimensions,
                             b_functions={shaders.CONTRACT_OFF: function})
            diff = int(result.get("differing_words", 0))
            row = {"index": number, "total": total, "leg": "mutation", "label": label,
                   "dimensions": dimensions,
                   "differing_words": diff, "b_launches": result.get("b_launches"),
                   "caught": diff > 0, "passed": result.get("b_launches") == 1 and diff > 0}
            rows.append(row)
            emit(handle, row)
    result = {"verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
              "elapsed_seconds": time.perf_counter() - started, "product_rows": len(products),
              "mutation_rows": len(defects), "rows": rows, "torch_version": torch.__version__,
              "metal_frontend": metal_frontend_version(),
              "subnormal_policy": subnormal.mps_policy_report("flush"),
              "source_provenance": {"family": hashlib.sha256(
                  Path(family.__file__).read_bytes()).hexdigest(), "gate": hashlib.sha256(
                  Path(__file__).read_bytes()).hexdigest()}, "jsonl": str(jsonl)}
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
