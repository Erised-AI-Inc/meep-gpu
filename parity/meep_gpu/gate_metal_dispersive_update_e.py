#!/usr/bin/env python3
"""Native MPS byte gate for real PML dispersive ``update_E → update_P``.

The gate owns only the pointwise PML constitutive/ADE handoff.  Curl, source,
and whole-driver residency are explicitly outside this gate's scope.  Each row
is appended and fsynced before the following Metal product is launched.

The oracle is ``stepping.update_E`` then ``PolarizationState.update`` on the
one-rank region plan of each case's grid and layer, and the plans bind that
division's chunk record. Every product row records, per component, the cells of
each constitutive form (the form-coverage rule, DESIGN.md D13) and requires both
forms on every split component. Mutations (KERNELS.md §7.1, §7.2): the previous
package accumulation grouping is planted and must be caught; so are the select
removed, the auxiliary stored on a plain cell, the integer lattice read for the
half-integer one and the bound record moved by one row; the auxiliary load made
unconditional is a declared null (the store stays guarded, so no word moves).
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
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, dispersive_update_e as family, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

from gate_chunk_records import layer_record  # noqa: E402
from meep_gpu.chunk_map import record_index  # noqa: E402


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, shape: Sequence[int], poles: int, kind: str, seed: int) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(float(n) / 8.0 for n in shape),
                courant=0.35, boundaries="periodic", xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(0.9 + index * 0.1, 0.05 + index * 0.01, kind),
            {"Ex": 0.2 + index * 0.01, "Ey": 0.0, "Ez": 0.3 + index * 0.01},
            grid, np.float32)
        fields.polarizations.append(state)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=1)
    # The region plan a driver installs with the layer: the one-rank division.
    stepping._region_plan(fields, pml)
    rng = np.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml


def _names(fields: Fields) -> Tuple[str, ...]:
    names = ["Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.extend((f"P[{index}].{component}", f"P_prev[{index}].{component}"))
        names.append(f"scratch[{index}]")
    return tuple(names)


def _array(fields: Fields, name: str) -> Any:
    if name in {"Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"}:
        return getattr(fields, name)
    if name.startswith("scratch["):
        return fields.polarizations[int(name[8:-1])]._scratch
    head, component = name.split(".", 1)
    index = int(head[head.index("[") + 1:head.index("]")])
    state = fields.polarizations[index]
    return state.P_prev[component] if head.startswith("P_prev") else state.P[component]


def _forms(fields: Fields, pml: PML) -> Dict[str, Any]:
    """Per component: (accumulating cells, plain cells) of the region plan stepped."""
    plan = stepping._region_plan(fields, pml)
    out: Dict[str, Any] = {}
    for component in ("Ex", "Ey", "Ez"):
        mask = np.broadcast_to(plan.mask("constitutive", component, np), fields.grid.shape)
        inside = int(np.count_nonzero(mask))
        out[component] = {"mode": plan.spec("constitutive", component).mode,
                          "accumulate": inside, "plain": int(mask.size) - inside}
    return out


def _run(*, shape: Sequence[int], poles: int, kind: str, seed: int, cycles: int,
         functions: Mapping[Tuple[str, int, int], Any] | None = None,
         record_mutation: Any = None) -> Dict[str, Any]:
    reference, reference_pml = _build(shape=shape, poles=poles, kind=kind, seed=seed)
    actual, actual_pml = _build(shape=shape, poles=poles, kind=kind, seed=seed)
    names = _names(actual)
    before = {name: np.array(_array(actual, name), copy=True) for name in names}
    residency = Residency()
    record = layer_record(actual.grid, actual_pml)
    e_record = record if record_mutation is None else record_mutation(record)
    e_plan = family.plan_metal_dispersive_e(actual, actual_pml, residency, functions=functions,
                                            chunk_bounds=e_record)
    p_plan = ade_update_p.plan_metal_ade_update_p(actual, actual_pml, residency,
                                                  chunk_bounds=record)
    if e_plan is None or p_plan is None:
        return {"passed": False, "reason": "a declared product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        e_plan.run()
        p_plan.run()
        residency.sync_out()
        difference = sum(_different(_array(reference, name), _array(actual, name))
                         for name in names)
        census = sum(subnormal.census(_array(reference, name)) for name in names)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], _array(actual, name)) for name in names)
    forms = _forms(actual, actual_pml)
    split_ok = all(row["accumulate"] and row["plain"] for row in forms.values()
                   if row["mode"] not in ("outside_only", "inside_only"))
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == row["reference_subnormals"] == 0
                           for row in per_cycle)
                   and moved > 0 and e_plan.runs == p_plan.runs == len(per_cycle)
                   and split_ok),
        "forms": forms,
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved, "e_launches": e_plan.launches,
        "p_launches": p_plan.launches, "mirrors": len(residency.names),
    }


def _planted(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise RuntimeError(f"mutation needle {needle!r} is not unique in the source")
    return source.replace(needle, replacement, 1)


#: The mutated component: Ex (axis 0), a split component in every mutation case.
_ACCUMULATE = "    if (dw) { value = e_out[idx] + ((kps[i] * src) - (kms[i] * prev)); }"


def _mutant_sources() -> Dict[str, Tuple[str, str]]:
    """label -> (expected, Ex source at two poles)."""
    base = family.dispersive_e_source(2, 0)
    head, tail = base.split("// chunk bounds: begin", 1)
    return {
        "second_pole_dropped": ("caught", _planted(
            base, "source = source - p1[idx];", "// dropped p1")),
        "fw_store_dropped": ("caught", _planted(
            base, "    if (dw) { fw[idx] = src; }", "    // stale fw")),
        # §7.1, inverted: the package's previous grouping, (f + kps*src) - kms*prev.
        "previous_accumulation_grouping": ("caught", _planted(
            base, _ACCUMULATE,
            "    if (dw) { value = (e_out[idx] + (kps[i] * src)) - (kms[i] * prev); }")),
        "pml_kps_kms_order_reversed": ("caught", _planted(
            base, _ACCUMULATE,
            "    if (dw) { value = e_out[idx] + ((kms[i] * src) - (kps[i] * prev)); }")),
        # §7.2: today's previous whole-grid form, the accumulation on every cell.
        "select_removed": ("caught", _planted(
            _planted(_planted(base, "    float prev = dw ? fw[idx] : 0.0f;",
                              "    float prev = fw[idx];"),
                     "    if (dw) { fw[idx] = src; }", "    fw[idx] = src;"),
            _ACCUMULATE,
            "    value = e_out[idx] + ((kps[i] * src) - (kms[i] * prev));")),
        "auxiliary_stored_on_a_plain_cell": ("caught", _planted(
            base, "    if (dw) { fw[idx] = src; }", "    fw[idx] = src;")),
        "half_integer_line_read_on_the_integer_lattice": ("caught", head
            + "// chunk bounds: begin" + tail.replace("cbh_", "cbi_").replace("in_h_", "in_i_")),
        # Declared null: the load is unconditional but nothing it reads is stored on a
        # plain cell, so no word moves; the module test pins the guard statically.
        "auxiliary_load_unconditional": ("null", _planted(
            base, "    float prev = dw ? fw[idx] : 0.0f;", "    float prev = fw[idx];")),
    }


def _mutants() -> Dict[str, Tuple[str, Mapping[Tuple[str, int, int], Any]]]:
    normal = {
        (shaders.CONTRACT_OFF, count, axis): compile_source(
            family.dispersive_e_source(count, axis)).dispersive_e_component
        for count in (0, 1, 2) for axis in range(3)
    }
    return {
        label: (expected, {
            **normal,
            (shaders.CONTRACT_OFF, 2, 0): compile_source(source).dispersive_e_component,
        })
        for label, (expected, source) in _mutant_sources().items()
    }


def _record_moved(word: int, delta: int):
    """b3: one bound of the Ex half-integer x line moved by one row."""
    def mutate(record: np.ndarray) -> np.ndarray:
        moved = np.array(record, copy=True)
        moved[record_index(True, 0, word)] += delta
        return moved
    return mutate


RECORD_MUTATIONS = {
    "record_low_run_end_plus_one": _record_moved(1, +1),
    "record_high_run_start_minus_one": _record_moved(2, -1),
}


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
        ("two_pole_lorentz", (9, 7, 5), 2, LORENTZIAN),
        ("two_pole_drude", (9, 7, 5), 2, DRUDE),
        ("narrow_y", (11, 3, 7), 2, LORENTZIAN),
        ("one_pole_control", (9, 7, 5), 1, LORENTZIAN),
    )
    mutations = _mutants()
    total = len(products) + len(mutations) + len(RECORD_MUTATIONS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows = []
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, shape, poles, kind) in enumerate(products, 1):
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "shape": shape, "poles": poles, "kind": kind, "cycles": args.cycles,
                   **_run(shape=shape, poles=poles, kind=kind, seed=32000 + index,
                          cycles=args.cycles)}
            rows.append(row)
            _emit(handle, row)
        index = len(products)
        planned = [(label, expected, {"functions": functions})
                   for label, (expected, functions) in mutations.items()]
        planned += [(label, "caught", {"record_mutation": mutate})
                    for label, mutate in RECORD_MUTATIONS.items()]
        for label, expected, keywords in planned:
            index += 1
            result = _run(shape=(9, 7, 5), poles=2, kind=LORENTZIAN,
                          seed=33000 + index, cycles=1, **keywords)
            difference = int(result.get("differing_words", 0))
            caught = difference > 0
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "expected": expected, "differing_words": difference,
                   "e_launches": result.get("e_launches"), "caught": caught,
                   "passed": (result.get("e_launches") == 3
                              and caught == (expected == "caught"))}
            rows.append(row)
            _emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started, "product_rows": len(products),
        "mutation_rows": len(mutations) + len(RECORD_MUTATIONS), "rows": rows,
        "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "ade": hashlib.sha256(Path(ade_update_p.__file__).read_bytes()).hexdigest(),
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
