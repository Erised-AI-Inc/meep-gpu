#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch no-PML stored-E and ADE handoff.

Each product advances the NumPy ``update_E → update_P`` reference beside the
complex stored-E plan—and, when a pole is live, the shared ADE plan—on one
persistent residency. The zero-pole product establishes the direct ``D * inv_eps``
stored-E specialization used by the full-step complex conductive composition; the
pole-bearing products establish the ordered ``D - P0 - P1 ...`` update, live pole
rotation and E-to-ADE handoff. One JSONL row is flushed before the next GPU product
or mutation begins.
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
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import ade_update_p, complex_no_pml_stored_e as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source, metal_frontend_version  # noqa: E402
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


def build(*, shape: Sequence[int], poles: int, kind: str, seed: int,
          phase: Sequence[float], components: Sequence[str]) -> Tuple[Fields, PML]:
    """Construct complex stored-E state with live, nontrivial pole arrays."""
    grid = Grid(resolution=8.0, cell_size=tuple(float(n) / 8.0 for n in shape),
                courant=0.35, boundaries="periodic", k_point=tuple(phase), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)

    def random_complex(scale: float) -> np.ndarray:
        return (rng.uniform(-scale, scale, grid.shape).astype(np.float32)
                + 1j * rng.uniform(-scale, scale, grid.shape).astype(np.float32))

    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = random_complex(0.45)
    for index in range(poles):
        sigma = {
            component: np.float32(0.18 + 0.02 * index + 0.01 * position)
            for position, component in enumerate(components)
        }
        state = PolarizationState(
            Susceptibility(0.9 + index * 0.1, 0.05 + index * 0.01, kind),
            sigma, grid, np.complex64)
        fields.polarizations.append(state)
        for component in state.driven():
            state.P[component][...] = random_complex(0.22)
            state.P_prev[component][...] = random_complex(0.22)
    return fields, PML(grid=grid, thickness=0)


def compared_names(fields: Fields) -> Tuple[str, ...]:
    names = ["Ex", "Ey", "Ez"]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.extend((f"P[{index}].{component}", f"P_prev[{index}].{component}"))
        names.append(f"scratch[{index}]")
    return tuple(names)


def get_named(fields: Fields, name: str) -> Any:
    if name in ("Ex", "Ey", "Ez"):
        return getattr(fields, name)
    if name.startswith("scratch["):
        index = int(name[name.index("[") + 1:name.index("]")])
        return fields.polarizations[index]._scratch
    head, component = name.split(".", 1)
    index = int(head[head.index("[") + 1:head.index("]")])
    state = fields.polarizations[index]
    if head.startswith("P_prev["):
        return state.P_prev[component]
    if head.startswith("P["):
        return state.P[component]
    raise ValueError(name)


def subnormal_words(fields: Fields, names: Iterable[str]) -> int:
    return sum(subnormal.census(get_named(fields, name)) for name in names)


def execute_case(*, shape: Sequence[int], poles: int, kind: str, seed: int,
                 phase: Sequence[float], components: Sequence[str], cycles: int,
                 e_functions: Mapping[Tuple[str, int, str], Any] | None = None,
                 ) -> Dict[str, Any]:
    """Run stored E, plus ADE when a live pole exists, beside the strict oracle."""
    reference, reference_pml = build(
        shape=shape, poles=poles, kind=kind, seed=seed, phase=phase, components=components)
    actual, actual_pml = build(
        shape=shape, poles=poles, kind=kind, seed=seed, phase=phase, components=components)
    names = compared_names(actual)
    before = {name: np.array(get_named(actual, name), copy=True) for name in names}
    residency = Residency()
    e_plan = family.plan_metal_complex_stored_e(
        actual, actual_pml, residency, probe=PROBE, functions=e_functions)
    p_plan = ade_update_p.plan_metal_ade_update_p(actual, actual_pml, residency)
    if e_plan is None or (poles and p_plan is None):
        return {"passed": False,
                "reason": "a declared stored-E/ADE product was refused"}
    if not poles and p_plan is not None:
        return {"passed": False,
                "reason": "a zero-pole stored-E product unexpectedly built ADE"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        e_plan.run()
        if p_plan is not None:
            p_plan.run()
        residency.sync_out()
        diff = sum(differing(get_named(reference, name), get_named(actual, name))
                   for name in names)
        census = subnormal_words(reference, names)
        per_cycle.append({"cycle": cycle, "differing_words": diff,
                          "reference_subnormals": census})
        if diff or census:
            break
    moved = sum(differing(before[name], get_named(actual, name)) for name in names)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == 0
                           and row["reference_subnormals"] == 0 for row in per_cycle)
                   and moved > 0 and e_plan.runs == len(per_cycle)
                   and (p_plan is None or p_plan.runs == len(per_cycle))),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved,
        "e_launches": e_plan.launches,
        "p_launches": 0 if p_plan is None else p_plan.launches,
        "mirrors": len(residency.names),
    }


def mutation_functions() -> Dict[str, Mapping[Tuple[str, int, str], Any]]:
    """Compile independently wrong two-pole stored-E bodies for the same ABI."""
    base = family.complex_stored_e_source(2, "FMA_V1")
    dropped_pole = base.replace("source = source - p1[idx];", "// dropped p1", 1)
    wrong_order = base.replace(
        "float2 source = d_in[idx];\n    source = source - p0[idx];\n    source = source - p1[idx];",
        "float2 source = p0[idx];\n    source = source - d_in[idx];\n    source = source - p1[idx];", 1)
    omitted_inverse = base.replace(
        "e_out[idx] = c_mul_field_left(source, inv_e[idx]);",
        "e_out[idx] = source;", 1)
    normal = {
        (shaders.CONTRACT_OFF, count, "FMA_V1"):
            compile_source(family.complex_stored_e_source(count, "FMA_V1"))
            .complex_stored_e_component
        for count in (0, 1, 2)
    }
    out = {}
    for label, source in (("second_pole_dropped", dropped_pole),
                          ("ordered_subtraction_reversed", wrong_order),
                          ("inverse_epsilon_omitted", omitted_inverse)):
        functions = dict(normal)
        functions[(shaders.CONTRACT_OFF, 2, "FMA_V1")] = (
            compile_source(source).complex_stored_e_component)
        out[label] = functions
    return out


def source_provenance() -> Dict[str, Any]:
    sources = {}
    for count in (0, 1, 2):
        source = family.complex_stored_e_source(count, "FMA_V1")
        sources[str(count)] = sha256_bytes(source.encode("utf-8"))
    mutations = mutation_functions()
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "ade": sha256_file(Path(ade_update_p.__file__).resolve()),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "stored_e_metal_sources": sources,
        "mutation_labels": tuple(mutations),
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
        ("zero_pole_stored_e_control", (9, 7, 5), 0, LORENTZIAN,
         (0.2, -0.1, 0.15), ()),
        ("two_pole_bloch", (9, 7, 5), 2, LORENTZIAN, (0.2, -0.1, 0.15), ("Ex", "Ez")),
        ("two_pole_drude", (9, 7, 5), 2, DRUDE, (0.2, -0.1, 0.15), ("Ex", "Ez")),
        ("collapsed_axis", (11, 1, 7), 2, LORENTZIAN, (0.0, 0.125, 0.0), ("Ex", "Ez")),
        ("one_pole_control", (9, 7, 5), 1, LORENTZIAN, (0.2, 0.0, 0.0), ("Ex", "Ez")),
    )
    mutations = mutation_functions()
    total = len(products) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, shape, poles, kind, phase, components) in enumerate(products, 1):
            result = execute_case(shape=shape, poles=poles, kind=kind, seed=24000 + index,
                                  phase=phase, components=components, cycles=args.cycles)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "shape": shape, "poles": poles, "kind": kind, "phase": phase,
                   "components": components, "cycles": args.cycles, **result}
            rows.append(row)
            emit(handle, row)
        offset = len(products)
        for number, (label, functions) in enumerate(mutations.items(), 1):
            result = execute_case(shape=(9, 7, 5), poles=2, kind=LORENTZIAN,
                                  seed=25000 + number, phase=(0.2, -0.1, 0.15),
                                  components=("Ex", "Ez"), cycles=1,
                                  e_functions=functions)
            diff = int(result.get("differing_words", 0))
            row = {"index": offset + number, "total": total, "leg": "mutation",
                   "label": label, "differing_words": diff,
                   "e_launches": result.get("e_launches"),
                   "caught": diff > 0,
                   "passed": bool(result.get("e_launches") == 3 and diff > 0)}
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
