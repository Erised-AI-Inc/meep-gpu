#!/usr/bin/env python3
"""Native MPS byte gate for folded tensor-row dispersive ``update_E``.

The gate drives the new 31-binding packed-pole shader through its real plan.  It
checks every stored-E output plus rotated live pole buffers after each cycle, so a
plan that kept a stale pre-ADE pointer cannot pass.  It deliberately does not claim
a fully device-resident ADE/pack composition; that is outside this sub-step gate.
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
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import folded_offdiag_dispersive_update_e as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source, metal_frontend_version  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, cell: Sequence[float], axes: str, phases: Sequence[int], boundaries: Any,
           rows: Mapping[str, Sequence[str]], counts: Sequence[int], kind: str,
           seed: int) -> Tuple[Fields, PML]:
    dimensions = 2 if float(cell[2]) == 0.0 else 3
    grid = Grid(resolution=8.0, cell_size=tuple(cell), dimensions=dimensions,
                courant=0.35, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in zip(axes, phases)), xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: (np.float32(1.0) / value).astype(np.float32)
               for name, value in epsilon.items()}
    tensor_rows = {row: {partner: rng.uniform(-0.18, 0.18, shape).astype(np.float32)
                         for partner in partners}
                   for row, partners in rows.items()}
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=tensor_rows)
    for index in range(max(counts)):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + index * 0.1, 0.06 + index * 0.01, kind),
            {name: 0.18 + index * 0.025 if counts[axis] > index else 0.0
             for axis, name in enumerate(("Ex", "Ey", "Ez"))}, grid, np.float32))
    fields.enable_pml_storage()
    folded = {"XYZ".index(axis) for axis in axes}
    thickness = []
    for axis, size in enumerate(shape):
        if size < 6:
            thickness.append((0, 0))
        elif axis in folded:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    pml = PML(grid=grid, thickness=tuple(thickness))
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey",
                 "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
    return fields, pml


def _names(fields: Fields) -> Tuple[str, ...]:
    names = ["Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.extend((f"P[{index}].{component}", f"P_prev[{index}].{component}"))
    return tuple(names)


def _array(fields: Fields, name: str) -> Any:
    if name in {"Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"}:
        return getattr(fields, name)
    head, component = name.split(".", 1)
    state = fields.polarizations[int(head[head.index("[") + 1:head.index("]")])]
    return state.P_prev[component] if head.startswith("P_prev") else state.P[component]


def _advance_poles(fields: Fields) -> None:
    for state in fields.polarizations:
        state.update(fields.drive_field, fields.grid.dt)


def _run(*, cell: Sequence[float], axes: str, phases: Sequence[int], boundaries: Any,
         rows: Mapping[str, Sequence[str]], counts: Sequence[int], kind: str,
         seed: int, cycles: int, functions: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = _build(cell=cell, axes=axes, phases=phases,
                                      boundaries=boundaries, rows=rows, counts=counts,
                                      kind=kind, seed=seed)
    actual, actual_pml = _build(cell=cell, axes=axes, phases=phases,
                                boundaries=boundaries, rows=rows, counts=counts,
                                kind=kind, seed=seed)
    names = _names(actual)
    before = {name: np.array(_array(actual, name), copy=True) for name in names}
    residency = Residency()
    plan = family.plan_folded_offdiag_dispersive(actual, actual_pml, residency,
                                                 functions=functions)
    if plan is None:
        return {"passed": False, "reason": "the declared product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        plan.run()
        residency.sync_out()
        _advance_poles(reference)
        _advance_poles(actual)
        difference = sum(_different(_array(reference, name), _array(actual, name))
                         for name in names)
        census = sum(subnormal.census(_array(reference, name)) for name in names)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], _array(actual, name)) for name in names)
    return {"passed": (len(per_cycle) == cycles
                         and all(row["differing_words"] == row["reference_subnormals"] == 0
                                 for row in per_cycle)
                         and moved > 0 and plan.runs == len(per_cycle)),
            "per_cycle": per_cycle, "differing_words": per_cycle[-1]["differing_words"],
            "reference_subnormals": per_cycle[-1]["reference_subnormals"],
            "moved_words": moved, "launches": plan.launches, "counts": plan.counts,
            "stored_shape": plan.shape, "mirrors": len(residency.names)}


def _mutants(rows: Mapping[str, Sequence[str]], counts: Sequence[int]) -> Dict[str, Any]:
    kwargs = dict(row_mask=(1, 1, 1, 0, 0, 0), codes=(0, 3, 0), walls=(0, 0, 0),
                  negate=(0, 1, 0), counts=counts)
    base = family.folded_offdiag_dispersive_source(**kwargs)
    sources = {
        "second_pole_dropped": base.replace(
            "value = value - p[1u * n_elem + uint(index)];", "// dropped pole", 1),
        "partner_pole_dropped": base.replace(
            "dmp1(g1, p1, ii, n_elem)", "g1[ii]", 1),
        "mirror_ghost_sign_dropped": base.replace("at_y ? -dn_00 : dn_00", "dn_00", 1),
        "pml_accumulations_reversed": base.replace(
            "a0 = a0 + kp_0 * src0; a0 = a0 - km_0 * prev0;",
            "a0 = a0 - km_0 * src0; a0 = a0 + kp_0 * prev0;", 1),
    }
    return {label: {shaders.CONTRACT_OFF: compile_source(source)
                    .folded_offdiag_dispersive_step}
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
        ("periodic_even_all_rows", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey", "Ez"), "Ey": ("Ez",)}, (2, 1, 2), LORENTZIAN),
        ("metallic_odd", (2.0, 1.5, 1.25), "Y", (-1,), {"y": "metallic"},
         {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}, (2, 2, 1), DRUDE),
        ("two_axis_fold", (2.0, 1.5, 1.25), "XY", (1, -1), "periodic",
         {"Ex": ("Ey",), "Ey": ("Ex", "Ez")}, (2, 1, 2), LORENTZIAN),
        ("sparse_row_control", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey",)}, (2, 1, 2), LORENTZIAN),
        # The signature's reason for existing: eight poles on EACH component still
        # fit only because each component is packed into one persistent buffer.
        ("max_eight_poles_per_component", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")},
         (8, 8, 8), LORENTZIAN),
    )
    mutation_rows = {"Ex": ("Ey", "Ez"), "Ey": ("Ez",)}
    mutation_counts = (2, 1, 2)
    mutations = _mutants(mutation_rows, mutation_counts)
    total = len(products) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    report = []
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, cell, axes, phases, boundaries, rows, counts, kind) in enumerate(products, 1):
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "cell": cell, "axes": axes, "phases": phases, "boundaries": boundaries,
                   "rows": rows, "counts": counts, "kind": kind, "cycles": args.cycles,
                   **_run(cell=cell, axes=axes, phases=phases, boundaries=boundaries,
                          rows=rows, counts=counts, kind=kind, seed=35100 + index,
                          cycles=args.cycles)}
            report.append(row)
            _emit(handle, row)
        for index, (label, functions) in enumerate(mutations.items(), len(products) + 1):
            result = _run(cell=(2.0, 1.5, 1.25), axes="Y", phases=(1,),
                          boundaries="periodic", rows=mutation_rows,
                          counts=mutation_counts, kind=LORENTZIAN, seed=35200 + index,
                          cycles=1, functions=functions)
            difference = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "differing_words": difference, "launches": result.get("launches"),
                   "caught": difference > 0,
                   "passed": result.get("launches") == 1 and difference > 0}
            report.append(row)
            _emit(handle, row)
    result = {"verdict": "PASS" if all(row["passed"] for row in report) else "FAIL",
              "elapsed_seconds": time.perf_counter() - started,
              "product_rows": len(products), "mutation_rows": len(mutations),
              "rows": report, "torch_version": torch.__version__,
              "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl),
              "source_sha256": {"family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
                                "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}",
          flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
