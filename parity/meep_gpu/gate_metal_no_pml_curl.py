#!/usr/bin/env python3
"""Byte gate for the real no-absorber Metal curl family.

The oracle is assembled exclusively from ``meep_gpu.stepping``'s term table,
stencil gather, grouped curl and ownership mask.  Each product row is written to
JSONL and flushed before the next row begins, so an interrupted MPS run retains
its completed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import no_pml_curl as family  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402


class OracleGrid:
    """The three boundary queries the stepping oracle's ownership mask reads."""

    def __init__(self, shape: Sequence[int], codes: Sequence[int]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self._codes = tuple(int(value) for value in codes)

    def is_metallic(self, axis: int) -> bool:
        return self._codes[axis] == shaders.METALLIC

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002
        return False

    def is_axis(self, axis: int) -> bool:  # noqa: ARG002
        return False


def words(array: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_provenance() -> Dict[str, Any]:
    """Bind the result to every source specialization compiled by this gate."""
    specializations: Dict[str, str] = {}
    for codes in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        for sub_step, derive in (("step_B", True), ("step_B", False),
                                 ("step_D", False)):
            backward = bool(family.SUB_STEPS[sub_step]["backward"])
            label = f"{sub_step}/derive-{int(derive)}/{codes}"
            source = family.plain_curl_source(codes, backward, derive)
            specializations[label] = sha256_bytes(source.encode("utf-8"))
    mutation_hashes = {
        name: sha256_bytes(source.encode("utf-8"))
        for name, source, _derive in mutation_sources()
    }
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "launch": sha256_file(Path(metal_launch.__file__).resolve()),
            "residency_coverage": sha256_file(
                API_ROOT / "meep_gpu" / "metal_kernels" / "coverage.py"),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": specializations,
        "mutation_metal_sources": mutation_hashes,
    }


def build_arrays(shape: Sequence[int], seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays = {
        name: rng.uniform(-0.45, 0.45, shape).astype(np.float32)
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")
    }
    for name in ("Ex", "Ey", "Ez"):
        arrays["inv_eps_" + name] = rng.uniform(0.25, 0.85, shape).astype(np.float32)
    return arrays


def oracle_once(arrays: Dict[str, np.ndarray], sub_step: str,
                codes: Sequence[int], dtdx: float, derive: bool) -> None:
    spec = family.SUB_STEPS[sub_step]
    if derive:
        snapshot: Dict[str, np.ndarray] = {}
        for e_name, d_name in zip(("Ex", "Ey", "Ez"), spec["displacement"]):
            value = np.empty_like(arrays[d_name])
            np.multiply(arrays[d_name], arrays["inv_eps_" + e_name], out=value)
            snapshot[e_name] = value
    elif sub_step == "step_D":
        # ``Fields.get_H`` serves B directly when no absorber is active, but the
        # D term table names the logical H components. Preserve both facts rather
        # than renaming the term table or allocating stored H arrays.
        snapshot = {h_name: arrays[b_name]
                    for h_name, b_name in zip(("Hx", "Hy", "Hz"),
                                              spec["sources"])}
    else:
        snapshot = {name: arrays[name] for name in spec["sources"]}
    boundaries = tuple("metallic" if code else "periodic" for code in codes)
    terms = stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS
    grid = OracleGrid(arrays[spec["targets"][0]].shape, codes)
    for term in terms:
        operands = stepping._curl_operands(  # noqa: SLF001 - this is the oracle
            np, snapshot, term, boundaries,
            (None, None, None), (None, None, None),
            backward=bool(spec["backward"]),
        )
        curl = stepping._curl_from_operands(operands, dtdx)  # noqa: SLF001
        stepping._mask_non_owned_cells(curl, grid, term.iyee)  # noqa: SLF001
        arrays[term.target] -= curl


def run_plan(arrays: Dict[str, np.ndarray], sub_step: str,
             codes: Sequence[int], dtdx: float, derive: bool, cycles: int,
             functions: Dict[str, Any] | None = None) -> Tuple[Any, Residency]:
    residency = Residency()
    plan = family.plan_metal_plain_curl_from_arrays(
        sub_step, arrays, codes, dtdx, residency, derive=derive,
        functions=functions,
    )
    residency.sync_in()
    for _ in range(cycles):
        plan.run()
    residency.sync_out(plan.volumes)
    return plan, residency


def emit(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    label = payload.get("label", payload.get("leg", "row"))
    print(f"{payload['index']}/{payload['total']} {label}: "
          f"diff={payload.get('differing_words', '-')}", flush=True)


def product_cases():
    shapes = ((17, 13, 9), (19, 1, 23))
    boundaries = ((0, 0, 0), (1, 0, 1), (1, 1, 1))
    courants = (0.35, 0.5)
    cases = []
    for shape in shapes:
        for codes in boundaries:
            for dtdx in courants:
                cases.append((shape, codes, dtdx, "step_B", True))
                cases.append((shape, codes, dtdx, "step_B", False))
                cases.append((shape, codes, dtdx, "step_D", False))
    return tuple(cases)


def mutation_sources() -> Tuple[Tuple[str, str, bool], ...]:
    base = family.plain_curl_source((1, 0, 1), False, True)
    add = base
    for index in range(3):
        add = add.replace(
            f"f{index}[ii] = f{index}[ii] - curl{index};",
            f"f{index}[ii] = f{index}[ii] + curl{index};")
    flat = base.replace(
        "dtdx * ((c_y - c) + (b - b_z))", "dtdx * (c_y - c + b - b_z)")
    flat = flat.replace(
        "dtdx * ((a_z - a) + (c - c_x))", "dtdx * (a_z - a + c - c_x)")
    flat = flat.replace(
        "dtdx * ((b_x - b) + (a - a_y))", "dtdx * (b_x - b + a - a_y)")
    no_derive = family.plain_curl_source((1, 0, 1), False, False)
    return (("add_instead_of_subtract", add, True),
            ("flatten_curl_grouping", flat, True),
            ("drop_derived_epsilon", no_derive, True))


def build_engine(seed: int) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75),
                courant=0.35, xp=np)
    fields = Fields(grid=grid)
    fields.set_epsilon_volumes(
        {name: np.full(grid.shape, value, np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
        {name: np.full(grid.shape, np.float32(1.0 / value), np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
    )
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, grid.shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=0)


def run_engine_composition(cycles: int) -> Dict[str, Any]:
    reference, reference_pml = build_engine(10401)
    actual, actual_pml = build_engine(10401)
    residency = Residency()
    plan = metal_launch.plan_step(
        actual, actual_pml, residency=residency, sources=())
    expected = {"step_B": "no-PML curl", "update_H": "no-PML null",
                "step_D": "no-PML curl", "update_E": "no-PML null"}
    if plan.selected != expected or not plan.residency.covered:
        return {"differing_words": -1, "moved_words": 0,
                "selected": plan.selected,
                "residency_reasons": list(plan.residency.reasons),
                "passed": False}
    before = {name: np.array(getattr(actual, name), copy=True)
              for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}
    residency.sync_in()
    for _ in range(cycles):
        stepping.step_B(reference, reference_pml)
        stepping.update_H(reference, reference_pml)
        stepping.step_D(reference, reference_pml)
        stepping.update_E(reference, reference_pml)
        for slot in ("step_B", "update_H", "step_D", "update_E"):
            plan.plans[slot].run()
    residency.sync_out()
    names = tuple(before)
    diff = sum(differing(getattr(reference, name), getattr(actual, name))
               for name in names)
    moved = sum(differing(before[name], getattr(actual, name)) for name in names)
    census = sum(subnormal.census(getattr(reference, name)) for name in names)
    launches = {slot: getattr(plan.plans[slot], "launches",
                              getattr(plan.plans[slot], "runs", None))
                for slot in expected}
    return {"differing_words": diff, "moved_words": moved,
            "reference_subnormals": census, "selected": plan.selected,
            "residency_reasons": list(plan.residency.reasons),
            "launches": launches,
            "passed": diff == 0 and moved > 0 and census == 0
            and all(value == cycles for value in launches.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Final JSON artifact; a sibling .jsonl is incremental")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()

    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    products = product_cases()
    mutations = mutation_sources()
    total = len(products) + len(mutations) + 1
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as stream:
        for index, (shape, codes, dtdx, sub_step, derive) in enumerate(products, 1):
            label = f"product/{sub_step}/derive-{int(derive)}/{shape}/{codes}/{dtdx}"
            reference = build_arrays(shape, 8100 + index)
            actual = {name: np.array(value, copy=True)
                      for name, value in reference.items()}
            before = {name: np.array(actual[name], copy=True)
                      for name in family.SUB_STEPS[sub_step]["targets"]}
            for _ in range(args.cycles):
                oracle_once(reference, sub_step, codes, dtdx, derive)
            plan, _ = run_plan(actual, sub_step, codes, dtdx, derive, args.cycles)
            targets = family.SUB_STEPS[sub_step]["targets"]
            diff = sum(differing(reference[name], actual[name]) for name in targets)
            moved = sum(differing(before[name], actual[name]) for name in targets)
            census = sum(subnormal.census(reference[name]) for name in targets)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "shape": shape, "codes": codes, "dtdx": dtdx,
                   "sub_step": sub_step, "derive": derive,
                   "cycles": args.cycles, "launches": plan.launches,
                   "differing_words": diff, "moved_words": moved,
                   "reference_subnormals": census,
                   "passed": diff == 0 and moved > 0 and census == 0
                   and plan.launches == args.cycles}
            rows.append(row)
            emit(stream, row)

        engine = run_engine_composition(args.cycles)
        engine.update({"index": len(products) + 1, "total": total,
                       "leg": "engine", "label": "complete_no_pml_arithmetic",
                       "cycles": args.cycles})
        rows.append(engine)
        emit(stream, engine)

        offset = len(products) + 1
        for number, (name, source, derive) in enumerate(mutations, 1):
            index = offset + number
            shape, codes, dtdx, sub_step = (17, 13, 9), (1, 0, 1), 0.35, "step_B"
            reference = build_arrays(shape, 9200 + number)
            actual = {key: np.array(value, copy=True)
                      for key, value in reference.items()}
            oracle_once(reference, sub_step, codes, dtdx, derive)
            function = compile_source(source).no_pml_curl_step
            plan, _ = run_plan(actual, sub_step, codes, dtdx, derive, 1,
                               {shaders.CONTRACT_OFF: function})
            targets = family.SUB_STEPS[sub_step]["targets"]
            diff = sum(differing(reference[target], actual[target])
                       for target in targets)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "differing_words": diff,
                   "launches": plan.launches, "caught": diff > 0,
                   "passed": diff > 0 and plan.launches == 1}
            rows.append(row)
            emit(stream, row)

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products),
        "mutation_rows": len(mutations),
        "engine_rows": 1,
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
