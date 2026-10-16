#!/usr/bin/env python3
"""Local MPS byte gate for the complete Lorentz/Drude ``update_P`` slot.

The oracle is :meth:`PolarizationState.update` itself: MEEP's isotropic update on
its four float32 constants, with the drive MEEP's per-chunk ``w = f_w ? f_w : f``
(:meth:`Fields.drive_field` on the one-rank region plan the build installs). Each
product is emitted to JSONL, flushed, and fsynced before the next product starts.
The result is bound to the exact host planner, gate, and specialized Metal source
bytes.

The mutation legs (KERNELS.md §7.1, §7.2): the previous package form -- the three
folded coefficients and its three-term grouping -- is planted and must be caught,
as are the other groupings and dropped terms; under an absorber the drive select is
removed both ways, inverted and keyed on the wrong lattice, and the bound record is
moved by one row. Every PML case splits each driven component between the two
drive forms, so both sides of the select are stepped.
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

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import record_index  # noqa: E402
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

from gate_chunk_records import layer_record  # noqa: E402

#: The one shipped real arithmetic line (no absorber, scalar sigma), the needle the
#: grouping mutations replace.
SHIPPED_LINE = ("p_out[idx] = gamma1inv * (((two_minus_od * p) - (gamma1 * q)) + "
                "(omega0dtsqr * (s * w)));")
#: The shipped drive select, the needle the select mutations replace.
SELECT_LINE = "w = dw ? fw[idx] : drive[idx];"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def words(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return array.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def _planted(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise RuntimeError(f"mutation needle {needle!r} is not unique in the source")
    return source.replace(needle, replacement, 1)


def mutation_sources() -> Dict[str, str]:
    """Arithmetic mutants of the no-absorber real body, each must-catch.

    ``previous_folded_three_term_form`` is the inverted mutation (KERNELS.md §7.1):
    the package's previous spelling, three coefficients folded from MEEP's four
    (``c_now = gamma1inv*two_minus_od``, ``c_prev = -(gamma1inv*gamma1)``,
    ``c_drive = gamma1inv*omega0dtsqr``) and its left-associated three-term sum.
    """
    base = family.ade_source("float32", False)
    return {
        "previous_folded_three_term_form": _planted(base, SHIPPED_LINE, (
            "float c_now = gamma1inv * two_minus_od; "
            "float c_prev = -(gamma1inv * gamma1); "
            "float c_drive = gamma1inv * omega0dtsqr; "
            "p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));")),
        "drop_previous_history": _planted(base, "(gamma1 * q)", "(0.0f * q)"),
        "drive_product_reassociated": _planted(
            base, "(omega0dtsqr * (s * w))", "((omega0dtsqr * s) * w)"),
        "outer_scale_distributed": _planted(base, SHIPPED_LINE, (
            "p_out[idx] = (gamma1inv * ((two_minus_od * p) - (gamma1 * q))) + "
            "(gamma1inv * (omega0dtsqr * (s * w)));")),
        "drop_sigma": _planted(base, "(s * w)", "(1.0f * w)"),
    }


def complex_mutation_sources() -> Dict[str, str]:
    """Grouping mutants of the complex (two real passes) body, each must-catch."""
    base = family.ade_source("complex64", True)
    return {
        "complex_sum_regrouped": _planted(
            base, "p_out[idx] = gamma1inv * ((a - b) + d);",
            "p_out[idx] = gamma1inv * (a - (b - d));"),
        "complex_drive_product_reassociated": _planted(
            base, "float2 d = omega0dtsqr * sw;", "float2 d = (omega0dtsqr * s) * w;"),
    }


def select_mutation_sources(axis: int) -> Dict[str, str]:
    """Drive-select mutants of the absorber body on one component's axis."""
    base = family.ade_source("float32", True, select_axis=axis)
    swapped = base.split("// chunk bounds: begin", 1)
    lattice = swapped[0] + "// chunk bounds: begin" + swapped[1].replace(
        "cbh_", "cbi_").replace("in_h_", "in_i_")
    return {
        "drive_select_removed_reads_f_w": _planted(base, SELECT_LINE, "w = fw[idx];"),
        "drive_select_removed_reads_stored_e": _planted(
            base, SELECT_LINE, "w = drive[idx];"),
        "drive_select_inverted": _planted(base, SELECT_LINE,
                                          "w = dw ? drive[idx] : fw[idx];"),
        "drive_select_on_the_integer_lattice": lattice,
    }


#: The record mutations (§7.2's off-by-one row): the kernel binds the true record
#: transformed, the oracle steps the true division.
def _record_moved(word: int, delta: int):
    def mutate(record: np.ndarray, axis: int) -> np.ndarray:
        moved = np.array(record, copy=True)
        moved[record_index(True, axis, word)] += delta
        return moved
    return mutate


RECORD_MUTATIONS = {
    "record_low_run_end_plus_one": _record_moved(1, +1),
    "record_high_run_start_minus_one": _record_moved(2, -1),
}


def source_provenance() -> Dict[str, Any]:
    sources = {
        f"{dtype}/sigma-{kind}/select-{axis}": sha256_bytes(
            family.ade_source(dtype, kind == "volume", select_axis=axis).encode("utf-8"))
        for dtype in ("float32", "complex64")
        for kind in ("scalar", "volume")
        for axis in (None, 0, 1, 2)
    }
    mutants = dict(mutation_sources())
    mutants.update(complex_mutation_sources())
    for axis in range(3):
        mutants.update({f"{name}/axis-{axis}": source
                        for name, source in select_mutation_sources(axis).items()})
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "launch": sha256_file(Path(metal_launch.__file__).resolve()),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": sources,
        "mutation_metal_sources": {
            name: sha256_bytes(source.encode("utf-8"))
            for name, source in mutants.items()
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

    if active_pml:
        # THE REGION PLAN A DRIVER INSTALLS WITH THE LAYER: the one-rank division of
        # this grid and absorber, which Fields.drive_field (the oracle's drive) reads.
        stepping._region_plan(fields, pml)
    for component in ("Ex", "Ey", "Ez"):
        # Both drive arrays are seeded, on disjoint ranges, so a select that reads
        # the wrong one on any cell moves the result.
        names = (("f_w_" + component, (-0.5, 0.5)), (component, (0.65, 0.95))) \
            if active_pml else ((component, (-0.5, 0.5)),)
        for name, (low, high) in names:
            array = getattr(fields, name)
            array.real[...] = rng.uniform(low, high, shape).astype(np.float32)
            if np.iscomplexobj(array):
                array.imag[...] = rng.uniform(low, high, shape).astype(np.float32)
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
        drive = getattr(fields, component)   # no absorber: the drive is the stored E
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
    plan = family.plan_metal_ade_update_p(
        actual, pml, residency, chunk_bounds=layer_record(actual.grid, pml))
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
    plan = family.plan_metal_ade_update_p(
        actual, pml, residency, chunk_bounds=layer_record(actual.grid, pml))
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
    forms = drive_forms(actual, pml)
    both = all(min(count) > 0 for count in forms.values()) if forms else True
    return {
        "passed": (diff == 0 and moved > 0 and plan.launches == expected_launches
                   and both),
        "differing_words": diff,
        "moved_words": moved,
        "launches": plan.launches,
        "expected_launches": expected_launches,
        "variants": list(plan.variants),
        "drive_forms": forms,
    }


def drive_forms(fields: Fields, pml: PML) -> Dict[str, Any]:
    """Per driven component under an absorber: (f_w cells, stored-E cells) the select reads.

    The form-coverage rule (DESIGN.md D13): a PML case must step both drive forms.
    """
    if not (pml is not None and pml.is_active):
        return {}
    plan = stepping._region_plan(fields, pml)
    driven = {component for state in fields.polarizations for component in state.driven()}
    out: Dict[str, Any] = {}
    for component in sorted(driven):
        mask = np.broadcast_to(plan.mask("constitutive", component, np),
                               fields.grid.shape)
        inside = int(np.count_nonzero(mask))
        out[component] = [inside, int(mask.size) - inside]
    return out


def _mutated_run(case: Dict[str, Any], seed: int, cycles: int,
                 functions: Dict[Any, Any] | None = None,
                 record_mutation=None) -> Dict[str, Any]:
    """Step the oracle and a plan built with ``functions`` (or a moved record)."""
    reference, _ = build(seed=seed, **case)
    actual, pml = build(seed=seed, **case)
    residency = Residency()
    record = layer_record(actual.grid, pml)
    if record_mutation is not None:
        record = record_mutation(record)
    plan = family.plan_metal_ade_update_p(
        actual, pml, residency, functions=functions, chunk_bounds=record)
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
            "differing_words": diff, "launches": plan.launches}


#: The mutation cases: real without an absorber (arithmetic), complex without one
#: (the per-part grouping), and real under an absorber with all three components
#: driven and split between the two drive forms (the select and the record).
MUTATION_CASE = {"shape": (17, 11, 9), "active_pml": False,
                 "complex_storage": False, "kind": LORENTZIAN,
                 "volume_sigma": False, "states": 1, "counts": (1, 1, 1)}
COMPLEX_MUTATION_CASE = {"shape": (17, 11, 9), "active_pml": False,
                         "complex_storage": True, "kind": DRUDE,
                         "volume_sigma": True, "states": 1, "counts": (1, 1, 1)}
SELECT_MUTATION_CASE = {"shape": (17, 11, 9), "active_pml": True,
                        "complex_storage": False, "kind": DRUDE,
                        "volume_sigma": True, "states": 1, "counts": (1, 1, 1)}


def run_mutation(name: str, source: str, cycles: int) -> Dict[str, Any]:
    mutant = compile_source(source).ade_update_p_step
    result = _mutated_run(MUTATION_CASE, 9917, cycles,
                          functions={("off", False, "float32", None): mutant})
    return {**result, "mutation": name}


def run_complex_mutation(name: str, source: str, cycles: int) -> Dict[str, Any]:
    mutant = compile_source(source).ade_update_p_step
    result = _mutated_run(COMPLEX_MUTATION_CASE, 9919, cycles,
                          functions={("off", True, "complex64", None): mutant})
    return {**result, "mutation": name}


def run_select_mutation(name: str, cycles: int) -> Dict[str, Any]:
    functions = {("off", True, "float32", axis):
                 compile_source(select_mutation_sources(axis)[name]).ade_update_p_step
                 for axis in range(3)}
    result = _mutated_run(SELECT_MUTATION_CASE, 9121, cycles, functions=functions)
    return {**result, "mutation": name}


def run_record_mutation(name: str, cycles: int) -> Dict[str, Any]:
    mutate = RECORD_MUTATIONS[name]

    def every_axis(record):
        for axis in range(3):
            record = mutate(record, axis)
        return record

    result = _mutated_run(SELECT_MUTATION_CASE, 9123, cycles, record_mutation=every_axis)
    return {**result, "mutation": name}


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
    complex_mutations = complex_mutation_sources()
    select_names = tuple(select_mutation_sources(0))
    record_names = tuple(RECORD_MUTATIONS)
    total = (len(products) + len(mutations) + len(complex_mutations)
             + len(select_names) + len(record_names) + 2)
    records = []
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8") as handle:
        index = 0
        for case in products:
            index += 1
            result = run_product(case, args.cycles, 7300 + index)
            record = {"index": index, "total": total, "leg": "product",
                      **case, **result}
            emit(handle, record)
            records.append(record)
        runs = ([(name, lambda name=name, source=source:
                  run_mutation(name, source, args.cycles))
                 for name, source in mutations.items()]
                + [(name, lambda name=name, source=source:
                    run_complex_mutation(name, source, args.cycles))
                   for name, source in complex_mutations.items()]
                + [(name, lambda name=name: run_select_mutation(name, args.cycles))
                   for name in select_names]
                + [(name, lambda name=name: run_record_mutation(name, args.cycles))
                   for name in record_names])
        for name, run in runs:
            index += 1
            result = run()
            record = {"index": index, "total": total, "leg": "mutation",
                      "label": name, "expected": "caught", **result}
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
        "mutation_records": (len(mutations) + len(complex_mutations)
                             + len(select_names) + len(record_names)),
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
