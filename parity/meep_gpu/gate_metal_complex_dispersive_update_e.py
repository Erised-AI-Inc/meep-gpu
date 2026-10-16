#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch PML dispersive ``update_E → update_P``.

The oracle is ``stepping.update_E`` then ``PolarizationState.update`` on the
one-rank region plan of each case's grid and layer; both plans bind that division's
chunk record (the E plan selects its form on it, the ADE plan its drive). Every
product row records, per component, the cells of each form and requires both on
every split component. Mutations (KERNELS.md §7.1, §7.2), each on Ex: the previous
accumulation grouping planted (must-catch), the coefficient order reversed, the
select removed, the auxiliary stored on a plain cell, the integer lattice read and
the bound record moved by one row; the unconditional auxiliary load is a declared
null.
"""

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

from gate_chunk_records import layer_record  # noqa: E402
from gate_metal_dispersive_update_e import RECORD_MUTATIONS, _forms  # noqa: E402


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
    # The region plan a driver installs with the layer: the one-rank division.
    stepping._region_plan(fields, pml)
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
            functions: Mapping[Tuple[str, int, int], Any] | None = None,
            record_mutation: Any = None) -> Dict[str, Any]:
    reference, reference_pml = build(shape=shape, poles=poles, kind=kind, seed=seed)
    actual, actual_pml = build(shape=shape, poles=poles, kind=kind, seed=seed)
    names = named(actual)
    before = {name: np.array(array(actual, name), copy=True) for name in names}
    residency = Residency()
    record = layer_record(actual.grid, actual_pml)
    e_record = record if record_mutation is None else record_mutation(record)
    e_plan = family.plan_metal_complex_dispersive_e(
        actual, actual_pml, residency, functions=functions,
        chunk_bounds=e_record)
    p_plan = ade_update_p.plan_metal_ade_update_p(actual, actual_pml, residency,
                                                  chunk_bounds=record)
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
    forms = _forms(actual, actual_pml)
    split_ok = all(row["accumulate"] and row["plain"] for row in forms.values()
                   if row["mode"] not in ("outside_only", "inside_only"))
    return {"passed": len(rows) == cycles and all(
        item["differing_words"] == item["reference_subnormals"] == 0 for item in rows)
        and moved > 0 and e_plan.runs == p_plan.runs == len(rows) and split_ok,
        "forms": forms,
        "per_cycle": rows, "differing_words": rows[-1]["differing_words"],
        "reference_subnormals": rows[-1]["reference_subnormals"], "moved_words": moved,
        "e_launches": e_plan.launches, "p_launches": p_plan.launches,
        "mirrors": len(residency.names)}


def _planted(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise RuntimeError(f"mutation needle {needle!r} is not unique in the source")
    return source.replace(needle, replacement, 1)


_ACCUMULATE = ("    if (dw) { value = e_out[idx] + (c_mul_coefficient_left(kps[i], src) "
               "- c_mul_coefficient_left(kms[i], prev)); }")


def mutant_sources() -> Dict[str, Tuple[str, str]]:
    """label -> (expected, Ex source at two poles)."""
    base = family.complex_dispersive_e_source(2, 0)
    head, tail = base.split("// chunk bounds: begin", 1)
    accumulate_everywhere = _planted(
        _planted(_planted(base, "    float2 prev = dw ? fw[idx] : float2(0.0f, 0.0f);",
                          "    float2 prev = fw[idx];"),
                 "    if (dw) { fw[idx] = src; }", "    fw[idx] = src;"),
        _ACCUMULATE, _ACCUMULATE.replace("    if (dw) { value =", "    { value ="))
    return {
        "second_pole_dropped": ("caught", _planted(
            base, "source = source - p1[idx];", "// dropped p1")),
        "fw_store_dropped": ("caught", _planted(
            base, "    if (dw) { fw[idx] = src; }", "    // stale fw")),
        # §7.1, inverted: the package's previous grouping, (f + kps*src) - kms*prev.
        "previous_accumulation_grouping": ("caught", _planted(
            base, _ACCUMULATE,
            "    if (dw) { value = (e_out[idx] + c_mul_coefficient_left(kps[i], src)) "
            "- c_mul_coefficient_left(kms[i], prev); }")),
        "pml_kps_kms_order_reversed": ("caught", _planted(
            base, _ACCUMULATE,
            "    if (dw) { value = e_out[idx] + (c_mul_coefficient_left(kms[i], src) "
            "- c_mul_coefficient_left(kps[i], prev)); }")),
        "select_removed": ("caught", accumulate_everywhere),
        "auxiliary_stored_on_a_plain_cell": ("caught", _planted(
            base, "    if (dw) { fw[idx] = src; }", "    fw[idx] = src;")),
        "half_integer_line_read_on_the_integer_lattice": ("caught", head
            + "// chunk bounds: begin" + tail.replace("cbh_", "cbi_").replace("in_h_", "in_i_")),
        "auxiliary_load_unconditional": ("null", _planted(
            base, "    float2 prev = dw ? fw[idx] : float2(0.0f, 0.0f);",
            "    float2 prev = fw[idx];")),
    }


def mutants() -> Dict[str, Tuple[str, Mapping[Tuple[str, int, int], Any]]]:
    """label -> (expected, function map keyed as the plan reads it: (mode, poles,
    axis)). A key the plan does not read never reaches a launch: the builder then
    compiles the shipped body for that key and the mutant is silently disarmed."""
    normal = {(shaders.CONTRACT_OFF, count, axis):
              compile_source(family.complex_dispersive_e_source(count, axis))
              .complex_dispersive_e_component
              for count in (0, 2) for axis in range(3)}
    out = {}
    for label, (expected, source) in mutant_sources().items():
        functions = dict(normal)
        functions[(shaders.CONTRACT_OFF, 2, 0)] = (
            compile_source(source).complex_dispersive_e_component)
        out[label] = (expected, functions)
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
    total = len(products) + len(defects) + len(RECORD_MUTATIONS)
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
        index = len(products)
        planned = [(label, expected, {"functions": functions})
                   for label, (expected, functions) in defects.items()]
        planned += [(label, "caught", {"record_mutation": mutate})
                    for label, mutate in RECORD_MUTATIONS.items()]
        for label, expected, keywords in planned:
            index += 1
            result = execute(shape=(9, 7, 5), poles=2, kind=LORENTZIAN, seed=29000 + index,
                             cycles=1, **keywords)
            diff = int(result.get("differing_words", 0))
            caught = diff > 0
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "expected": expected, "differing_words": diff,
                   "e_launches": result.get("e_launches"), "caught": caught,
                   "passed": (result.get("e_launches") == 3
                              and caught == (expected == "caught"))}
            rows.append(row)
            emit(handle, row)
    result = {"verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
              "elapsed_seconds": time.perf_counter() - started, "product_rows": len(products),
              "mutation_rows": len(defects) + len(RECORD_MUTATIONS), "rows": rows,
              "torch_version": torch.__version__,
              "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl)}
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
