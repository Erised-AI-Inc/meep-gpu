#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch conductive no-PML B/D curls.

The two curl plans share persistent mirrors and are compared to the array-path
``stepping`` oracle after every cycle.  Product and mutation rows are emitted to
a flushed JSONL sibling before the next GPU launch.  This establishes only this
curl-pair's arithmetic and live B/H mirror sharing, not a driver dispatch route.

Every plan binds the chunk record of its case's grid
(``complex_no_pml_gate_common.case_record``), whose partner-presence word
decides the stencil form. The cases, the planted signed-zero state and the stencil,
connection and presence mutation rows are the common module's, shared with the
lossless gate; the conductive rows (the two coefficient volumes swapped, the target
store dropped) are this gate's own. The flattened-grouping row is inverted to the
previous pairwise bracket (KERNELS.md §7.1) and stays must-catch.
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
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import complex_no_pml_gate_common as common  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.metal_kernels import complex_no_pml_conductive as family  # noqa: E402
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


def install_conductivity(fields: Fields, sides: Sequence[str], mixed: bool) -> None:
    """Install all-target or deliberately per-target conductivity profiles."""
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), fields.grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, fields.grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(
            {"Bx": sigma if not mixed else None,
             "By": sigma,
             "Bz": sigma if not mixed else None})
    if "D" in sides:
        fields.set_d_conductivity(
            {"Dx": sigma if mixed else sigma,
             "Dy": None if mixed else sigma,
             "Dz": None if mixed else sigma})


_F_COND = tuple("f_cond_" + target for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))


def build(*, case: str, seed: int, sides: Sequence[str], mixed: bool,
          planted: bool = False) -> Tuple[Fields, Any]:
    """Build a live complex conductive no-PML state on one case's grid."""
    return common.build(case, seed=seed, planted=planted,
                        install=lambda fields: install_conductivity(fields, sides, mixed),
                        names=_F_COND)


def compared_names(fields: Fields) -> Tuple[str, ...]:
    names = ["Bx", "By", "Bz", "Dx", "Dy", "Dz"]
    names.extend(name for name in _F_COND if getattr(fields, name) is not None)
    return tuple(names)


def subnormal_words(fields: Fields, names: Iterable[str]) -> int:
    return sum(subnormal.census(getattr(fields, name)) for name in names)


def execute_case(*, case: str, seed: int, sides: Sequence[str], mixed: bool,
                 cycles: int, planted: bool = False,
                 b_functions: Mapping[str, Any] | None = None,
                 every_axis_present: bool = False) -> Dict[str, Any]:
    """Run one complete B/D curl pair beside the NumPy oracle."""
    reference, reference_pml = build(case=case, seed=seed, sides=sides, mixed=mixed,
                                     planted=planted)
    actual, actual_pml = build(case=case, seed=seed, sides=sides, mixed=mixed,
                               planted=planted)
    names = compared_names(actual)
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    record = common.case_record(actual, actual_pml)
    bound = common.all_axes_present(record) if every_axis_present else record
    residency = Residency()
    b_plan = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_B", residency, functions=b_functions,
        chunk_bounds=bound)
    d_plan = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_D", residency, chunk_bounds=bound)
    if b_plan is None or d_plan is None:
        return {"passed": False, "reason": "a declared conductive product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        b_plan.run()
        d_plan.run()
        import torch  # noqa: PLC0415

        torch.mps.synchronize()
        stepping.step_B(reference, reference_pml)
        stepping.step_D(reference, reference_pml)
        residency.sync_out(names)
        diff = sum(differing(getattr(reference, name), getattr(actual, name))
                   for name in names)
        census = subnormal_words(reference, names)
        per_cycle.append({"cycle": cycle, "differing_words": diff,
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
        "b_flags": list(b_plan.conductive),
        "d_flags": list(d_plan.conductive),
        "b_launches": b_plan.launches,
        "d_launches": d_plan.launches,
        "mirrors": len(residency.names),
    }


def _shipped_b_source(case: str, flags=(True, True, True)) -> str:
    codes, kinds = common.b_specialization(case)
    return family.complex_conductive_no_pml_curl_source(codes, False, kinds, flags)


def mutation_rows():
    """(label, case, mutated step_B source, expectation, planted, every-axis-present).

    Armed on an all-conductive B product (both sides lossy, every component): the
    conductive rows are this gate's own, the rest the common module's.
    """
    def swapped(text: str) -> str:
        text = common.needle(text, "value0 = c_mul_field_left(value0, cf0[ii]);",
                             "value0 = c_mul_field_left(value0, TEMP_CF0);")
        text = common.needle(text, "value0 = c_mul_field_left(value0, ci0[ii]);",
                             "value0 = c_mul_field_left(value0, cf0[ii]);")
        return common.needle(text, "value0 = c_mul_field_left(value0, TEMP_CF0);",
                             "value0 = c_mul_field_left(value0, ci0[ii]);")

    def store_dropped(text: str) -> str:
        return common.needle(text, "    f0[ii] = value0;", "    // dropped f write")

    table = [("condfac_condinv_swapped", "two_axis_bloch", swapped, "caught", False,
              False),
             ("target_store_dropped", "two_axis_bloch", store_dropped, "caught", False,
              False)] + common.curl_mutations()
    return [(label, case, transform(_shipped_b_source(case)), expectation, planted,
             present) for label, case, transform, expectation, planted, present in table]


def source_provenance(rows) -> Dict[str, Any]:
    from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

    specializations = {}
    for case in common.CASES:
        grid = common.grid_for(case)
        resolved = stepping._boundary_kinds(grid, None)
        codes = tuple(1 if kind == "metallic" else 0 for kind in resolved)
        for flags in ((True, True, True), (False, True, False)):
            for sub_step in ("step_B", "step_D"):
                kinds, _values = complex_fields.phase_arguments(
                    complex_fields.bloch_phase_table(grid, resolved),
                    backward=sub_step == "step_D")
                source = family.complex_conductive_no_pml_curl_source(
                    codes, sub_step == "step_D", kinds, flags)
                specializations[f"{sub_step}/{codes}/{kinds}/{flags}"] = sha256_bytes(
                    source.encode("utf-8"))
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
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


#: (label, case, conductive sides, per-target mix).
PRODUCTS: Tuple[Tuple[str, str, Tuple[str, ...], bool], ...] = (
    ("two_sided_bloch", "two_axis_bloch", ("B", "D"), False),
    ("D_only_bloch", "two_axis_bloch", ("D",), False),
    ("B_only_x_phase", "collapsed_y", ("B",), False),
    ("mixed_components", "two_axis_bloch", ("B", "D"), True),
    ("metallic_xz", "metallic_xz_bloch_y", ("B", "D"), False),
    ("brillouin_edge_x", "brillouin_edge_x", ("B", "D"), False),
    ("planar_2d_bloch", "planar_2d_bloch", ("B", "D"), True),
    ("planar_2d_metallic_x", "planar_2d_metallic_x", ("B", "D"), False),
    ("line_1d_z", "line_1d_z", ("B", "D"), False),
)


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
        for number, (label, case, sides, mixed) in enumerate(PRODUCTS, 1):
            index += 1
            result = execute_case(case=case, seed=22000 + number, sides=sides,
                                  mixed=mixed, cycles=args.cycles)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "case": case, "sides": sides, "mixed": mixed, "cycles": args.cycles,
                   **result}
            rows.append(row)
            emit(handle, row)
        for number, (label, case) in enumerate(common.PLANTED_CASES, 1):
            index += 1
            result = execute_case(case=case, seed=22500 + number, sides=("B", "D"),
                                  mixed=True, cycles=args.cycles, planted=True)
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
            result = execute_case(case=case, seed=23000 + number, sides=("B", "D"),
                                  mixed=False, cycles=1, planted=planted,
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
