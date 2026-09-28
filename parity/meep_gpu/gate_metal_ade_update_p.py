#!/usr/bin/env python3
"""Local MPS byte gate for the complete Lorentz/Drude ``update_P`` slot.

The oracle is :meth:`PolarizationState.update` itself.  Each product is emitted
to JSONL, flushed, and fsynced before the next product starts.  The result is
bound to the exact host planner, gate, and specialized Metal source bytes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu.dispersion import (  # noqa: E402
    DRUDE,
    LORENTZIAN,
    PolarizationState,
    Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import ade_update_p as family  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def words(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return array.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def mutation_sources() -> Dict[str, str]:
    base = family.ade_source("float32", False)
    return {
        "drop_previous_history": base.replace(
            "(c_prev * q)", "(0.0f * q)", 1),
        "reassociate_three_terms": base.replace(
            "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))",
            "(p * c_now) + ((c_prev * q) + (c_drive * (s * w)))", 1),
        "drop_sigma": base.replace("(s * w)", "(1.0f * w)", 1),
    }


def source_provenance() -> Dict[str, Any]:
    sources = {
        f"{dtype}/sigma-{kind}": sha256_bytes(
            family.ade_source(dtype, kind == "volume").encode("utf-8"))
        for dtype in ("float32", "complex64")
        for kind in ("scalar", "volume")
    }
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "launch": sha256_file(Path(metal_launch.__file__).resolve()),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": sources,
        "mutation_metal_sources": {
            name: sha256_bytes(source.encode("utf-8"))
            for name, source in mutation_sources().items()
        },
    }


def build(*, shape: Sequence[int], active_pml: bool, complex_storage: bool,
          kind: str, volume_sigma: bool, states: int, counts: Sequence[int],
          seed: int) -> tuple[Fields, PML]:
    resolution = 8.0
    cell = tuple(float(n) / resolution for n in shape)
    grid = Grid(
        resolution=resolution,
        cell_size=cell,
        courant=0.35,
        boundaries="periodic",
        k_point=(0.11, -0.07, 0.03) if complex_storage else (0.0, 0.0, 0.0),
        xp=np,
    )
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"requested {tuple(shape)}, Grid built {grid.shape}")
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    dtype = np.complex64 if complex_storage else np.float32
    rng = np.random.default_rng(seed)

    for state_index in range(states):
        sigma: Dict[str, Any] = {}
        for axis, component in enumerate(("Ex", "Ey", "Ez")):
            if not counts[axis]:
                sigma[component] = 0.0
            elif volume_sigma:
                sigma[component] = rng.uniform(
                    0.08 + 0.01 * state_index, 0.62, shape).astype(np.float32)
            else:
                sigma[component] = 0.19 + 0.04 * axis + 0.01 * state_index
        susceptibility = Susceptibility(
            0.85 + 0.08 * state_index,
            0.09 + 0.01 * state_index,
            kind if state_index % 2 == 0 else (
                DRUDE if kind == LORENTZIAN else LORENTZIAN),
        )
        state = PolarizationState(susceptibility, sigma, grid, dtype)
        fields.polarizations.append(state)

    for component in ("Ex", "Ey", "Ez"):
        drive = fields.drive_field(component)
        drive.real[...] = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        if np.iscomplexobj(drive):
            drive.imag[...] = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        if active_pml:
            stored = getattr(fields, component)
            stored.real[...] = rng.uniform(0.65, 0.95, shape).astype(np.float32)
            if np.iscomplexobj(stored):
                stored.imag[...] = rng.uniform(0.65, 0.95, shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                array = table[component]
                array.real[...] = rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                if np.iscomplexobj(array):
                    array.imag[...] = rng.uniform(-0.25, 0.25, shape).astype(np.float32)
    return fields, pml


def product_cases() -> tuple[Dict[str, Any], ...]:
    cases = []
    index = 0
    for active in (False, True):
        for complex_storage in (False, True):
            for kind in (LORENTZIAN, DRUDE):
                for volume in (False, True):
                    index += 1
                    cases.append({
                        "label": f"cartesian-{index:02d}",
                        "shape": (17, 11, 9),
                        "active_pml": active,
                        "complex_storage": complex_storage,
                        "kind": kind,
                        "volume_sigma": volume,
                        "states": 1,
                        "counts": (1, 0, 1),
                    })
    cases.extend((
        {"label": "all-components-two-pole", "shape": (13, 9, 7),
         "active_pml": True, "complex_storage": False,
         "kind": LORENTZIAN, "volume_sigma": True, "states": 2,
         "counts": (1, 1, 1)},
        {"label": "collapsed-axis-complex", "shape": (19, 1, 13),
         "active_pml": False, "complex_storage": True,
         "kind": DRUDE, "volume_sigma": False, "states": 2,
         "counts": (0, 1, 1)},
        {"label": "group-i-lifted", "shape": (35, 32, 41),
         "active_pml": False, "complex_storage": True,
         "kind": LORENTZIAN, "volume_sigma": True, "states": 2,
         "counts": (1, 1, 1)},
        {"label": "complex-signed-zero", "shape": (16, 8, 8),
         "active_pml": False, "complex_storage": True,
         "kind": DRUDE, "volume_sigma": False, "states": 1,
         "counts": (1, 1, 1), "adversarial": True},
    ))
    return tuple(cases)


def plant_adversarial_complex(fields: Fields) -> None:
    """Install normal finite values plus both zero signs on every complex plane."""
    base = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
                    dtype=np.float32)
    total = int(np.prod(fields.grid.shape))
    forward = np.resize(base, total).reshape(fields.grid.shape)
    reverse = np.resize(base[::-1], total).reshape(fields.grid.shape)
    for component_index, component in enumerate(("Ex", "Ey", "Ez")):
        drive = fields.drive_field(component)
        drive.real[...] = np.roll(forward, component_index, axis=0)
        drive.imag[...] = np.roll(reverse, component_index, axis=1)
    for state_index, state in enumerate(fields.polarizations):
        for component_index, component in enumerate(state.driven()):
            state.P[component].real[...] = np.roll(
                forward, state_index + component_index, axis=0)
            state.P[component].imag[...] = np.roll(
                reverse, state_index + component_index, axis=1)
            state.P_prev[component].real[...] = np.roll(
                reverse, state_index + component_index, axis=2)
            state.P_prev[component].imag[...] = np.roll(
                forward, state_index + component_index, axis=0)


def compare_states(reference: Fields, actual: Fields) -> tuple[int, int]:
    diff = moved = 0
    for expected, observed in zip(reference.polarizations, actual.polarizations):
        for component in expected.driven():
            diff += differing(expected.P[component], observed.P[component])
            diff += differing(expected.P_prev[component], observed.P_prev[component])
            moved += int(np.count_nonzero(words(observed.P[component])))
        diff += differing(expected._scratch, observed._scratch)
    return diff, moved


# ---------------------------------------------------------------------------
# THE ROTATION LEG — a HOST defect, and the one this family's fused successor
# will be built on top of
# ---------------------------------------------------------------------------
#
# The four source mutations above are all ARITHMETIC: they patch the Metal shader
# string. The rotation is not in the shader. ``MetalAdePlan.run`` does it in
# Python after each launch (ade_update_p.py:389-392):
#
#     state.P[component] = scratch     # this step's result
#     state.P_prev[component] = p      # what P held
#     state._scratch = p_prev          # the retired history
#
# so no shader mutation can reach it and, before this leg, NOTHING here pinned it.
#
# WHY IT MATTERS NOW. A fused kernel folds update_P into the launch that produced
# its drive field, and its whole aliasing discipline exists because that rotation
# makes one arm's destination another arm's source. Fusing on top of a sub-step
# gate that never planted a rotation defect would inherit an assumption nobody
# measured.
#
# IT DOES NOT HIDE HERE, AND THAT IS A MEASURED DIFFERENCE FROM THE CUDA TWIN.
# On CUDA (2026-08-19) a stale-pointer defect moved 0 of 7293 words at ONE launch
# and 7293 of 7293 at sixty, so that gate needs a budget to see it. This gate does
# not: ``compare_states`` compares P, P_prev AND _scratch (see :227-229), so a
# rotation defect is visible in the state after a SINGLE cycle. Measured here:
#
#     freeze_rotation   8415 differing words at 1 cycle and at 5 - identical
#     double_rotation   6732 at 1 cycle, 8415 at 5
#
# The one-cycle number is therefore recorded as EVIDENCE ABOUT THE COMPARATOR, not
# as a discriminator: a leg that fires at one cycle here means the comparator
# reaches the rotated state, which is the stronger position. The borrowed CUDA
# premise ("must be inert at one cycle") was wrong for this backend and is not
# asserted. What IS asserted is that both defects are caught.


def run_rotation_mutation(name: str, case: Dict[str, Any], cycles: int,
                          seed: int) -> Dict[str, Any]:
    """Plant a rotation defect in the PLAN OBJECT and require the bytes to move."""
    build_args = {k: v for k, v in case.items()
                  if k not in ("label", "adversarial")}
    reference, _ = build(seed=seed, **build_args)
    actual, pml = build(seed=seed, **build_args)
    residency = Residency()
    plan = family.plan_metal_ade_update_p(actual, pml, residency)
    if plan is None:
        return {"mutation": name, "caught": False, "differing_words": -1,
                "reason": "planner refused the product this leg needs"}

    original = type(plan).run

    def frozen(self, contract=None):
        """Rotate nothing: P, P_prev and the scratch keep their identities."""
        keep = [(state, component,
                 state.P[component], state.P_prev[component], state._scratch)
                for entry in self.entries
                for state, component in ((entry.state, entry.component),)]
        original(self, contract)
        for state, component, p, p_prev, scratch in keep:
            state.P[component], state.P_prev[component] = p, p_prev
            state._scratch = scratch

    def stale(self, contract=None):
        """Rotate twice: the history advances past the value the next step needs."""
        original(self, contract)
        for entry in self.entries:
            state, component = entry.state, entry.component
            p, p_prev = state.P[component], state.P_prev[component]
            state.P[component], state.P_prev[component] = p_prev, p

    planted = {"freeze_rotation": frozen, "double_rotation": stale}[name]
    type(plan).run = planted
    try:
        residency.sync_in()
        for _ in range(cycles):
            for state in reference.polarizations:
                state.update(reference.drive_field, reference.grid.dt)
            plan.run()
        residency.sync_out(plan.volumes)
        diff, moved = compare_states(reference, actual)
    finally:
        type(plan).run = original          # restored even if the leg raises
    return {"mutation": name, "cycles": cycles, "caught": diff > 0,
            "differing_words": diff, "moved_words": moved}


def run_product(case: Dict[str, Any], cycles: int, seed: int) -> Dict[str, Any]:
    build_args = {k: v for k, v in case.items()
                  if k not in ("label", "adversarial")}
    reference, _ = build(seed=seed, **build_args)
    actual, pml = build(seed=seed, **build_args)
    if case.get("adversarial"):
        plant_adversarial_complex(reference)
        plant_adversarial_complex(actual)
    residency = Residency()
    plan = family.plan_metal_ade_update_p(actual, pml, residency)
    if plan is None:
        return {"passed": False, "differing_words": -1,
                "reason": "planner refused a declared product"}
    residency.sync_in()
    for _ in range(cycles):
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
    residency.sync_out(plan.volumes)
    diff, moved = compare_states(reference, actual)
    expected_launches = cycles * sum(
        len(state.driven()) for state in actual.polarizations)
    return {
        "passed": diff == 0 and moved > 0 and plan.launches == expected_launches,
        "differing_words": diff,
        "moved_words": moved,
        "launches": plan.launches,
        "expected_launches": expected_launches,
        "variants": list(plan.variants),
    }


def run_mutation(name: str, source: str, cycles: int) -> Dict[str, Any]:
    case = {"shape": (17, 11, 9), "active_pml": False,
            "complex_storage": False, "kind": LORENTZIAN,
            "volume_sigma": False, "states": 1, "counts": (1, 1, 1)}
    reference, _ = build(seed=9917, **case)
    actual, pml = build(seed=9917, **case)
    residency = Residency()
    mutant = compile_source(source).ade_update_p_step
    plan = family.plan_metal_ade_update_p(
        actual, pml, residency, functions={("off", False, "float32"): mutant})
    if plan is None:
        return {"passed": False, "differing_words": -1,
                "reason": "mutation plan refused"}
    residency.sync_in()
    for _ in range(cycles):
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
    residency.sync_out(plan.volumes)
    diff, _ = compare_states(reference, actual)
    return {"passed": diff > 0 and plan.launches > 0,
            "differing_words": diff, "launches": plan.launches,
            "mutation": name}


def run_wrong_drive_mutation(cycles: int) -> Dict[str, Any]:
    case = {"shape": (17, 11, 9), "active_pml": True,
            "complex_storage": False, "kind": DRUDE,
            "volume_sigma": True, "states": 1, "counts": (1, 0, 0)}
    reference, _ = build(seed=9121, **case)
    actual, pml = build(seed=9121, **case)
    residency = Residency()
    plan = family.plan_metal_ade_update_p(actual, pml, residency)
    assert plan is not None
    wrong = actual.Ex
    wrong_tensor = residency.mirror("mutation:stored-Ex", wrong, dtype=np.float32)
    entry = dataclasses.replace(plan.entries[0], drive=wrong)
    plan.entries = (entry,)
    plan._tensor_by_host[id(wrong)] = wrong_tensor
    residency.sync_in()
    for _ in range(cycles):
        reference.polarizations[0].update(reference.drive_field, reference.grid.dt)
        plan.run()
    residency.sync_out(plan.volumes)
    diff, _ = compare_states(reference, actual)
    return {"passed": diff > 0 and plan.launches > 0,
            "differing_words": diff, "launches": plan.launches,
            "mutation": "stored_E_instead_of_f_w"}


def emit(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{payload['index']}/{payload['total']} {payload['label']}: "
          f"diff={payload.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--out", type=Path, default=(
        Path(__file__).resolve().parent / "results" /
        "metal_ade_update_p_2026-08-17" / "gate.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.out.with_suffix(".jsonl")
    products = product_cases()
    mutations = mutation_sources()
    total = len(products) + len(mutations) + 1
    records = []
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(products, 1):
            result = run_product(case, args.cycles, 7300 + index)
            record = {"index": index, "total": total, "leg": "product",
                      **case, **result}
            emit(handle, record)
            records.append(record)
        offset = len(products)
        for sub_index, (name, source) in enumerate(mutations.items(), 1):
            result = run_mutation(name, source, args.cycles)
            record = {"index": offset + sub_index, "total": total,
                      "leg": "mutation", "label": name, **result}
            emit(handle, record)
            records.append(record)
        result = run_wrong_drive_mutation(args.cycles)
        record = {"index": total, "total": total, "leg": "mutation",
                  "label": "stored_E_instead_of_f_w", **result}
        emit(handle, record)
        records.append(record)

        # THE ROTATION LEG, with its own discriminator. Each defect must be
        # CAUGHT at the gate's budget and INERT at one cycle: that pair is what
        # makes the budget the thing doing the catching rather than the
        # arithmetic, and it is the only leg in this gate that reaches the host
        # state machine the fused successor will be built on.
        rotation_case = dict(products[0])
        rotation_case.pop("adversarial", None)
        for name in ("freeze_rotation", "double_rotation"):
            caught = run_rotation_mutation(name, rotation_case, args.cycles, 7777)
            inert = run_rotation_mutation(name, rotation_case, 1, 7777)
            record = {"index": total, "total": total, "leg": "rotation",
                      "label": name,
                      "caught_at_budget": bool(caught.get("caught")),
                      "differing_at_budget": caught.get("differing_words"),
                      "differing_at_one_cycle": inert.get("differing_words"),
                      "budget": args.cycles,
                      "passed": bool(caught.get("caught")),
                      "one_cycle_note": (
                          "recorded as evidence about the COMPARATOR, not as a "
                          "discriminator: this gate compares P, P_prev and _scratch, "
                          "so a rotation defect is visible after one cycle. The CUDA "
                          "twin needs a 60-launch budget for the same defect class")}
            emit(handle, record)
            records.append(record)

    summary = {
        "passed": all(record["passed"] for record in records),
        "records": len(records),
        "product_records": len(products),
        "mutation_records": len(mutations) + 1,
        "rotation_records": 2,
        "cycles": args.cycles,
        "elapsed_seconds": time.perf_counter() - started,
        "platform": {"metal_frontend": metal_frontend_version()},
        "source_provenance": source_provenance(),
        "jsonl": str(records_path),
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(summary)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
