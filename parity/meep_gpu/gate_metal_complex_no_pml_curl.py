#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch lossless no-PML curls.

The gate runs the NumPy ``stepping.step_B``/``step_D`` oracle beside persistent
Metal mirrors.  It intentionally covers only the curl pair: a whole-driver
residency seam remains a distinct certification step.  One JSONL row is flushed
after every product or mutation so an interrupted Apple-GPU run retains all
completed evidence.

Every plan binds the chunk record of its case's grid
(``complex_no_pml_gate_common.case_record``: the composer's one-rank
division), whose partner-presence word decides the stencil form. The cases, the
planted signed-zero state and the mutation rows are the common module's, shared
with the conductive gate: the products cover 3-D, 2-D and 1-D grids, a 3-D grid
storing one cell on a live axis, phased, Brillouin-edge and metallic axes; the
planted leg requires the shipped kernel to reproduce a state holding ``-0`` on every
target part; the mutations invert the old flattened-grouping row to the previous
pairwise bracket (KERNELS.md §7.1) and arm the absent-partner, connection and
presence rows, each absent-arm row beside a declared-null twin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import complex_no_pml_gate_common as common  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.metal_kernels import complex_no_pml_curl as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)

words = common.words
differing = common.differing


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def compared_names(fields: Fields) -> Sequence[str]:
    return ("Bx", "By", "Bz", "Dx", "Dy", "Dz")


def subnormal_words(fields: Fields, names: Iterable[str]) -> int:
    return sum(subnormal.census(getattr(fields, name)) for name in names)


def execute_case(*, case: str, seed: int, cycles: int, planted: bool = False,
                 b_functions: Mapping[str, Any] | None = None,
                 every_axis_present: bool = False) -> Dict[str, Any]:
    """Run B then D beside their exact NumPy sub-step oracle on one case."""
    reference, reference_pml = common.build(case, seed=seed, planted=planted)
    actual, actual_pml = common.build(case, seed=seed, planted=planted)
    names = compared_names(actual)
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    record = common.case_record(actual, actual_pml)
    bound = common.all_axes_present(record) if every_axis_present else record
    residency = Residency()
    b_plan = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_B", residency, functions=b_functions,
        chunk_bounds=bound)
    d_plan = family.plan_metal_complex_no_pml_curl(
        actual, actual_pml, "step_D", residency, chunk_bounds=bound)
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
                   and (moved > 0 or planted) and b_plan.launches == len(per_cycle)
                   and d_plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "compared_words": sum(2 * getattr(actual, name).size for name in names),
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "reference_negative_zeros": sum(common.negative_zeros(getattr(reference, name))
                                        for name in names),
        "moved_words": moved,
        "record_presence": int(np.asarray(bound)[-1]),
        "b_launches": b_plan.launches,
        "d_launches": d_plan.launches,
        "mirrors": len(residency.names),
    }


def _shipped_b_source(case: str) -> str:
    codes, kinds = common.b_specialization(case)
    return family.complex_no_pml_curl_source(codes, False, kinds)


def mutation_rows():
    """(label, case, mutated step_B source, expectation, planted, every-axis-present).

    The lossless tail row is this gate's own; the stencil, connection and presence
    rows are the common module's (shared with the conductive gate).
    """
    rows = []

    def reversed_tail(text: str) -> str:
        for index in range(3):
            text = common.needle(text, f"f{index}[ii] = f{index}[ii] - curl{index};",
                                 f"f{index}[ii] = f{index}[ii] + curl{index};")
        return text

    table = [("curl_subtraction_reversed", "two_axis_bloch", reversed_tail, "caught",
              False, False)] + common.curl_mutations()
    for label, case, transform, expectation, planted, present in table:
        rows.append((label, case, transform(_shipped_b_source(case)), expectation,
                     planted, present))
    return rows


def source_provenance(rows) -> Dict[str, Any]:
    specializations = {}
    for case in common.CASES:
        codes, _ = common.b_specialization(case)
        for sub_step in ("step_B", "step_D"):
            grid = common.grid_for(case)
            resolved = stepping._boundary_kinds(grid, None)
            from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

            kinds, _values = complex_fields.phase_arguments(
                complex_fields.bloch_phase_table(grid, resolved),
                backward=sub_step == "step_D")
            source = family.complex_no_pml_curl_source(
                codes, sub_step == "step_D", kinds)
            specializations[f"{sub_step}/{codes}/{kinds}"] = sha256_bytes(
                source.encode("utf-8"))
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "shared_stencil": sha256_file(Path(family.__file__).resolve().with_name(
                "complex_no_pml_conductive.py")),
            "gate": sha256_file(Path(__file__).resolve()),
            "common": sha256_file(Path(common.__file__).resolve()),
        },
        "cases": common.provenance_cases(),
        "specialized_metal_sources": specializations,
        "mutation_metal_sources": {
            label: sha256_bytes(source.encode("utf-8"))
            for label, _case, source, *_rest in rows
        },
    }


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['leg']} {row['label']}: "
          f"diff={row.get('differing_words', '-')} passed={row.get('passed')}",
          flush=True)


PRODUCTS = ("two_axis_bloch", "zero_phase_control", "metallic_xz_bloch_y",
            "collapsed_y", "brillouin_edge_x", "planar_2d_bloch", "planar_2d_edge_y",
            "planar_2d_metallic_x", "line_1d_z")


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
    mutations = mutation_rows()
    total = len(PRODUCTS) + len(common.PLANTED_CASES) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for number, case in enumerate(PRODUCTS, 1):
            index += 1
            result = execute_case(case=case, seed=20000 + number, cycles=args.cycles)
            row = {"index": index, "total": total, "leg": "product", "label": case,
                   "case": case, "cycles": args.cycles, **result}
            rows.append(row)
            emit(handle, row)
        for number, (label, case) in enumerate(common.PLANTED_CASES, 1):
            index += 1
            result = execute_case(case=case, seed=20500 + number, cycles=args.cycles,
                                  planted=True)
            row = {"index": index, "total": total, "leg": "planted_zero",
                   "label": label, "case": case, "cycles": args.cycles, **result}
            row["passed"] = bool(result["passed"]
                                 and result.get("reference_negative_zeros", 0) > 0)
            rows.append(row)
            emit(handle, row)
        for number, (label, case, source, expectation, planted, present) in enumerate(
                mutations, 1):
            index += 1
            function = compile_source(source).complex_no_pml_conductive_curl_step
            result = execute_case(case=case, seed=21000 + number, cycles=1,
                                  planted=planted,
                                  b_functions=(None if present else
                                               {shaders.CONTRACT_OFF: function}),
                                  every_axis_present=present)
            diff = int(result.get("differing_words", 0))
            caught = diff > 0
            launched = result.get("b_launches") == 1
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": label, "case": case, "expectation": expectation,
                   "planted": planted, "differing_words": diff,
                   "compared_words": result.get("compared_words"),
                   "b_launches": result.get("b_launches"), "caught": caught,
                   "passed": bool(launched and (caught if expectation == "caught"
                                                else not caught))}
            rows.append(row)
            emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(PRODUCTS),
        "planted_rows": len(common.PLANTED_CASES),
        "mutation_rows": len(mutations),
        "rows": rows,
        "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": subnormal.mps_policy_report("flush"),
        "source_provenance": source_provenance(mutations),
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
