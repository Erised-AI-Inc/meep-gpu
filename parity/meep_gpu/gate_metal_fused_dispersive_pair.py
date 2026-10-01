#!/usr/bin/env python3
"""Native MPS byte gate for the FIRST fused Metal kernel: ``step_D`` -> ``update_E``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. Every other Metal family gate
seeds a state, launches ONE kernel and compares words. This product spans THREE
driver passes — ``step_D``, ``zero_metal_D`` and ``update_E`` — so a per-sub-step
comparison could not see it at all: the whole claim is about the SEAM between
them. The comparison is therefore per COMPLETE DRIVER STEP over a stated budget,
against an array-path oracle running the identical live pass list, with the first
divergent step reported rather than a final pass/fail.

THE FOUR THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION, because the
   oracle is the array path this walk would otherwise take. So every case asserts
   the exact LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``,
   and every case asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** ELEVEN mutations are ARMED — nine in the shader source, two
   in the HOST binding that no shader edit can reach — and each must be CAUGHT. Leg
   ``disarm`` reruns the identical case with the shipped functions and requires
   zero, so a mutation reported as caught cannot be a harness that diverges anyway.
3. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no
   lever, so byte-identity is claimed subject to a CHECKED subnormal-free
   precondition; every case censuses the oracle's own state and reports it.
4. **An unmeasured platform assumption.** The family is per component rather than
   one dispatch because an all-three-component fused signature needs 35 buffer
   bindings and the ceiling is 31. Leg ``binding_ceiling`` COMPILES that signature
   and requires the failure, so the shape of the product rests on a measurement.

LEGS
  0  binding_ceiling   the 35-binding signature must FAIL to compile
  1  product           complete steps, per-step byte compare, launch counters, movement
  2  separate_control  the two ALREADY CERTIFIED products stepping the same seam as
                       separate dispatches, on identical state: three-way byte
                       agreement plus the dispatch counts the fusion removes
  3  mutation          eleven armed defects, each of which MUST diverge
  4  disarm            the same harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_value_classes as values  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    fused_dispersive_pair as family, launch as metal_launch, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

#: The budget every case runs. Twelve, matching the whole-step arbiter's, and for
#: its reason: the classes this gate exists for COMPOUND. ``fu_D`` and ``f_w_E``
#: are state carried between steps, and a defect that needs three steps to reach
#: the low bits is exactly the kind a four-step budget reports as green. The
#: comparison is per COMPLETE STEP, so passing here means identical at all twelve.
STEPS = 12

#: Which array-path function each live pass is. Same table the whole-step arbiter
#: keeps, restricted to the passes an unfolded, source-free run executes.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every statically named volume a complete step of this configuration touches.
#: Polarization arrays are added at runtime by :func:`state_of`, because
#: ``update_P`` ROTATES which physical allocation holds ``P`` and no static string
#: table can name them without losing the semantic role.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name, cell size, boundary declaration, poles, susceptibility kind.
#: ``metallic_y`` and ``metallic_all`` are NOT decoration: they are the only rows
#: on which the inline ``zero_metal_D`` and the curl's ownership mask do any work
#: at all, so the two mutations that arm them would be vacuous without one.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int, str], ...] = (
    ("periodic_two_pole_lorentz", (1.2, 1.0, 0.9), "periodic", 2, LORENTZIAN),
    ("periodic_two_pole_drude", (1.2, 1.0, 0.9), "periodic", 2, DRUDE),
    ("metallic_y_two_pole", (1.2, 1.0, 0.9),
     ("periodic", "metallic", "periodic"), 2, LORENTZIAN),
    ("metallic_all_three_pole", (1.1, 1.0, 0.9), "metallic", 3, LORENTZIAN),
    ("periodic_one_pole_control", (1.2, 1.0, 0.9), "periodic", 1, LORENTZIAN),
    ("narrow_z_two_pole", (1.3, 1.1, 0.4), "periodic", 2, LORENTZIAN),
)

#: The case the mutations are armed on. Metallic on every axis, so the ownership
#: mask, the inline wall clear and the ghost rule are all LIVE — a mutation leg run
#: on a periodic-only row would report the two boundary defects as uncaught for the
#: uninteresting reason that neither line is emitted there.
MUTATION_CASE = "metallic_all_three_pole"


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def build(cell: Sequence[float], boundaries: Any, poles: int, kind: str,
          seed: int, value_class: str = values.UNIFORM,
          scale: float = 1.0) -> Tuple[Any, Fields, PML]:
    """One seeded engine. Called twice per case, identically, for the two sides."""
    grid = Grid(resolution=10.0, cell_size=tuple(cell), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    for order in range(poles):
        # Every pole drives all three components, so no component takes the
        # zero-pole specialisation by accident and every axis's chain is live.
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order, kind),
            {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
             "Ez": 0.19 + 0.06 * order},
            grid, np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid,
              thickness=tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                              for axis in range(3)))
    if value_class == values.PM_ZERO_LATTICE:
        # THE POLARIZATIONS TAKE THE OPPOSED LATTICE. D = P = P_prev = 0 is a
        # FIXED POINT of this seam (the ADE chain's own gate says so in capitals),
        # so P_prev is seeded with the NEGATION of P's lattice: the recurrence
        # then differences two zeros of opposite sign, and the sign is what this
        # class exists to compare.
        lattice = values.pm_zero_lattice(grid.shape)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = lattice
                state.P_prev[component][...] = -lattice
        return grid, fields, pml
    if value_class != values.UNIFORM:
        raise ValueError(f"unknown value class {value_class!r}")
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = values.scaled_normal(rng, grid.shape, 0.37, scale)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = values.scaled_normal(
                rng, grid.shape, 0.21, scale)
            state.P_prev[component][...] = values.scaled_normal(
                rng, grid.shape, 0.19, scale)
    return grid, fields, pml


def state_of(fields: Fields) -> Dict[str, Any]:
    """Every semantically live state array, with the ADE roles named not aliased."""
    state = {name: getattr(fields, name) for name in STATE_NAMES
             if getattr(fields, name, None) is not None}
    for index, polarization in enumerate(tuple(fields.polarizations)):
        for component in tuple(polarization.driven()):
            state[f"P[{index}].{component}"] = polarization.P[component]
            state[f"P_prev[{index}].{component}"] = polarization.P_prev[component]
        state[f"scratch[{index}]"] = polarization._scratch
    return state


def frozen(fields: Fields) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Fields, right: Fields) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def live_passes(fields: Fields, pml: PML) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is.

    ``None`` is a refusal rather than an empty tuple: a gate that stepped nothing
    would compare a no-op with a no-op and pass.
    """
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def array_step(fields: Fields, pml: PML, live: Sequence[str]) -> None:
    for name in live:
        ARRAY_PATH[name](fields, pml)


def metal_step(fields: Fields, pml: PML, dispatch: Mapping[str, Any],
               owned: Sequence[str], residency: Residency,
               live: Sequence[str]) -> None:
    """One complete step, on the device where a plan owns the pass.

    A pass no plan owns runs on the array path and is bracketed with an explicit
    ``sync_out`` / ``sync_in``. That bracket is the residency clause made
    operational: without it the next device launch would read the mirror's
    pre-``step_B`` bytes for H, which is smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because a fused plan owns three passes
    and dispatches at one of them. Deriving the skip set from the dispatch keys
    would silently leave ``zero_metal_D`` running on the host on top of the copy
    the kernel already carried.
    """
    skip = set(owned)
    for name in live:
        if name in skip:
            plan = dispatch.get(name)
            if plan is not None:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def leg_subnormal_ladder(seed: int, budget: int = 4) -> Dict[str, Any]:
    """THE EMPTY CENSUS, TURNED INTO A MEASUREMENT.

    Every product row in this file reports ``reference_subnormals: 0`` and
    ``subnormal_free: true`` under policy ``flush``. Read alone that is a SILENCE:
    it cannot distinguish "the flush policy was exercised and changed nothing"
    from "the band was never entered, so the policy was never asked anything".
    This leg is the difference — not a second byte comparison, which MPS cannot
    give, but the precondition FIRING.

    THE ORACLE ALONE runs here, seeded at each rung of
    :data:`metal_value_classes.PRECONDITION_SCALES` and stepped through the same
    live pass list, with the same census the product rows report taken after
    every step. Nothing is compared against the device and the artifact says so:
    ``banded_rows_were_byte_compared`` is False. MPS flushes float32 subnormals
    natively with no lever and the NumPy oracle keeps, so a subnormal in the
    reference state IS a divergence and the band is a REFUSAL rather than a
    class; what this family claims is byte identity under a CHECKED subnormal-free
    precondition, and this leg is that check being shown to work.
    """
    _label, cell, boundaries, poles, kind = next(
        row for row in CASES if row[0] == MUTATION_CASE)

    def make(scale: float) -> Tuple[Any, Any]:
        _, fields, pml = build(cell, boundaries, poles, kind, seed,
                               values.UNIFORM, scale)
        return fields, pml

    def step(engine: Tuple[Any, Any]) -> None:
        fields, pml = engine
        array_step(fields, pml, live_passes(fields, pml))

    def census(engine: Tuple[Any, Any]) -> int:
        return sum(subnormal.census(value)
                   for value in state_of(engine[0]).values())

    return values.census_ladder(make, step, census, budget, log)


def run_case(label: str, cell: Sequence[float], boundaries: Any, poles: int,
             kind: str, seed: int, steps: int,
             functions: Optional[Mapping[Tuple[str, int], Any]] = None,
             patch: Optional[Callable[[Any, PML, Residency], None]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    _, reference, reference_pml = build(cell, boundaries, poles, kind, seed,
                                        value_class)
    _, actual, actual_pml = build(cell, boundaries, poles, kind, seed,
                                  value_class)

    drift = compare(reference, actual)
    assert not drift, f"{label}: the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_fused_dispersive_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_fused_dispersive_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": "the fused pair was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(actual, actual_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    # THE FLOOR, AND WHICH ONE DEPENDS ON THE VALUE CLASS. The uniform class must
    # move every compared array or the comparison is a no-op agreeing with a
    # no-op. The +-0 lattice answers to the SIGN floor instead: the REFERENCE
    # OUTPUT must carry BOTH bit patterns, which is what makes a uint32 compare of
    # a zero state discriminating rather than decorative. It is not exempt from a
    # floor, it has a different one.
    signs = values.zero_sign_census(state_of(reference).values())
    if value_class == values.PM_ZERO_LATTICE:
        floor_ok = bool(signs["positive_zero_words"] and signs["negative_zero_words"])
    else:
        floor_ok = not still
    return {
        "passed": bool(identical and clean and launches_ok and floor_ok),
        "value_class": value_class,
        "floor_met": floor_ok,
        "reference_zero_sign_census": signs,
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "reference_subnormals": per_step[-1]["reference_subnormals"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.codes), "zero_metal": list(plan.zero_metal),
        "poles": [entry.pole_count for entry in plan.entries],
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(label: str, cell: Sequence[float], boundaries: Any,
                         poles: int, kind: str, seed: int, steps: int,
                         ) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path, the two ALREADY CERTIFIED Metal
    products stepping the same seam as separate dispatches with ``zero_metal_D``
    left on the host between them, and the fused product. All three must agree
    word for word at every complete step, and the DISPATCH COUNTS are recorded on
    both sides — which is the only place the difference between the compositions
    shows up at all, since a correct fusion is byte-neutral by construction.

    The separate side is not a strawman: it is exactly what ``plan_step`` composes
    today for this configuration, which is why disagreement here would be a defect
    in the fused product rather than in an invented comparator.
    """
    from meep_gpu.metal_kernels import dispersive_update_e as pointwise

    _, reference, reference_pml = build(cell, boundaries, poles, kind, seed)
    _, separate, separate_pml = build(cell, boundaries, poles, kind, seed)
    _, fused, fused_pml = build(cell, boundaries, poles, kind, seed)

    separate_residency = Residency()
    curl = metal_launch.plan_pml_curl(separate, separate_pml, "step_D",
                                      separate_residency)
    electric = pointwise.plan_metal_dispersive_e(separate, separate_pml,
                                                 separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_fused_dispersive_pair(
        fused, fused_pml, sources=(), residency=fused_residency)
    if curl is None or electric is None or plan is None:
        return {"passed": False,
                "reason": "a declared product was refused",
                "curl": curl is not None, "pointwise": electric is not None,
                "fused": plan is not None}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(separate, separate_pml, {"step_D": curl, "update_E": electric},
                   ("step_D", "update_E"), separate_residency, live)
        metal_step(fused, fused_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, fused_residency, live)
        separate_residency.sync_out()
        fused_residency.sync_out()
        per_step.append({
            "step": step,
            "separate_vs_array": sum(compare(reference, separate).values()),
            "fused_vs_array": sum(compare(reference, fused).values()),
            "fused_vs_separate": sum(compare(separate, fused).values()),
        })
        if any(value for key, value in per_step[-1].items() if key != "step"):
            break

    # The host pass each composition leaves on the array path INSIDE the seam.
    # The fused side carries `zero_metal_D` in the kernel; the separate side does
    # not, so on a walled run it pays a host round trip the fused side does not.
    seam_host_passes = [name for name in live
                        if name in ("fill_D", "zero_metal_D",
                                    "fill_folded_far_ghosts_D")]
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and electric.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + electric.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + electric.launches,
        "fused_launches": plan.launches,
        "seam_host_passes_for_separate": seam_host_passes,
        "seam_host_passes_for_fused": [],
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _case(label: str) -> Tuple[str, Tuple[float, float, float], Any, int, str]:
    return next(row for row in CASES if row[0] == label)


def _shipped_functions(codes: Sequence[int], poles: int,
                       zero_metal: Sequence[bool]) -> Dict[Tuple[str, int], Any]:
    return {(shaders.CONTRACT_OFF, axis): family.compile_fused_dispersive_pair(
        codes, axis, poles, zero_metal) for axis in range(3)}


def shader_mutations(codes: Sequence[int], poles: int,
                     zero_metal: Sequence[bool]) -> Dict[str, Dict[Tuple[str, int], Any]]:
    """Nine source defects, each a plausible transcription slip.

    Every one is armed on the X component only and the other two axes keep the
    shipped bytes, which is what makes a caught mutation evidence about the
    MUTATED LINE rather than about the harness.
    """
    base = family.fused_dispersive_pair_source(codes, 0, poles, zero_metal)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl before the split-field recurrence.
        "seam_takes_pre_recurrence_curl":
            base.replace("float source = v0;", "float source = curl0;", 1),
        # THE SEAM'S ORDER: let the E half consume D BEFORE the wall clear, which
        # is the array path's order inverted. Only the fused kernel can have it.
        "zero_metal_after_the_e_half":
            base.replace("    v0 = at_y ? 0.0f : v0;\n", "", 1)
                .replace("    v0 = at_z ? 0.0f : v0;\n", "", 1)
                .replace("    e_out[idx] = value;",
                         "    e_out[idx] = value;\n"
                         "    v0 = at_y ? 0.0f : v0;\n"
                         "    v0 = at_z ? 0.0f : v0;\n"
                         "    f0[ii] = v0;", 1),
        "zero_metal_dropped":
            base.replace("    v0 = at_y ? 0.0f : v0;\n", "", 1)
                .replace("    v0 = at_z ? 0.0f : v0;\n", "", 1),
        "ownership_mask_dropped":
            base.replace("    curl0 = at_y ? 0.0f : curl0;\n", "", 1)
                .replace("    curl0 = at_z ? 0.0f : curl0;\n", "", 1),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            base.replace("dtdx * ((c_y - c) + (b - b_z))",
                         "dtdx * (c_y - c + b - b_z)", 1),
        "second_pole_dropped":
            base.replace("    source = source - q1[idx];\n", "", 1),
        "fw_store_dropped":
            base.replace("    fw[idx] = src;", "    // stale fw", 1),
        "fu_store_dropped":
            base.replace("    u0[ii] = n0;", "    // stale fu_Dx", 1),
        "pml_accumulations_reversed":
            base.replace("    value = value + kps[i] * src;\n"
                         "    value = value - kms[i] * prev;",
                         "    value = value - kms[i] * src;\n"
                         "    value = value + kps[i] * prev;", 1),
    }
    mutants: Dict[str, Dict[Tuple[str, int], Any]] = {}
    for name, source in edits.items():
        if source == base:
            raise AssertionError(
                f"mutation {name!r} did not change the source; a no-op edit would "
                f"report the shipped kernel as an uncaught defect")
        table = _shipped_functions(codes, poles, zero_metal)
        table[(shaders.CONTRACT_OFF, 0)] = compile_source(
            source).fused_dispersive_pair_component
        mutants[name] = table
    return mutants


def swap_e_lattice(plan: Any, pml: PML, residency: Residency) -> None:
    """HOST DEFECT: bind the INTEGER-lattice constitutive coefficients to the E half.

    ``update_E`` takes ``kps_a_h``/``kms_a_h`` and ``update_H`` takes ``kps_a``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from, so this is a silent half-cell error in the absorber
    profile — converged, smooth and wrong — and no shader mutation can reach it.
    """
    entries = []
    for entry in plan.entries:
        axis = ("x", "y", "z")[entry.axis]
        kps, kms = getattr(pml, f"kps_{axis}"), getattr(pml, f"kms_{axis}")
        plan._tensors[id(kps)] = residency.mirror(
            f"mutation:kps_{axis}", kps, constant=True)
        plan._tensors[id(kms)] = residency.mirror(
            f"mutation:kms_{axis}", kms, constant=True)
        entries.append(dataclasses.replace(entry, kps=kps, kms=kms))
    plan.entries = tuple(entries)


def swap_curl_lattice(plan: Any, pml: PML, residency: Residency) -> None:
    """HOST DEFECT: hand the D curl the HALF-INTEGER split-field coefficients.

    ``step_D`` reads integer positions and ``step_B`` half-integer ones
    (launch.py:110-113). Getting it backwards is a half-cell error, not a crash.
    """
    swapped = []
    for axis in "xyz":
        for stem in ("kms", "sinv"):
            host = getattr(pml, f"{stem}_{axis}_h")
            plan._tensors[id(host)] = residency.mirror(
                f"mutation:{stem}_{axis}_h", host, constant=True)
            swapped.append(host)
    plan._curl_coefficients = tuple(swapped)


HOST_MUTATIONS: Dict[str, Callable[[Any, PML, Residency], None]] = {
    "e_half_takes_the_integer_lattice": swap_e_lattice,
    "curl_takes_the_half_integer_lattice": swap_curl_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """The 35-binding all-component signature MUST fail to compile.

    This is what makes "per component" a measurement rather than a preference. A
    host on which it compiled would mean the one-launch Triton shape is buildable
    here and this family is the wrong shape — which is a finding, so the leg fails
    loudly rather than being skipped.
    """
    source = family.refuted_all_component_source()
    bindings = family.ALL_COMPONENT_BINDINGS
    try:
        compile_source(source)
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return {"passed": ("out of bounds" in message and "buffer" in message),
                "bindings": bindings,
                "ceiling": 31,
                "compiled": False,
                "error": message.splitlines()[0] if message else "",
                "per_component_bindings": 29}
    return {"passed": False, "bindings": bindings, "compiled": True,
            "error": "the 35-binding signature COMPILED; the per-component shape "
                     "of this family rests on a ceiling this host does not have"}


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    label, cell, boundaries, poles, kind = _case(MUTATION_CASE)
    _, probe_fields, probe_pml = build(cell, boundaries, poles, kind, 1)
    codes = tuple(1 if k == "metallic" else 0
                  for k in stepping._boundary_kinds(probe_fields.grid, probe_pml))
    from meep_gpu.metal_kernels.coverage import zero_metal_axes
    walls = zero_metal_axes(probe_fields.grid)
    assert any(walls), (
        f"{MUTATION_CASE} has no walled axis; the two boundary mutations would be "
        f"vacuous")
    mutants = shader_mutations(codes, poles, walls)

    controls = ("periodic_two_pole_lorentz", "metallic_all_three_pole")
    total = (1 + len(CASES) + len(CASES) + 1 + len(controls) + len(mutants)
             + len(HOST_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "all_three_components_in_one_dispatch",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        for offset, (name, cell, boundaries, poles, kind) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps, "kind": kind,
                   **run_case(name, cell, boundaries, poles, kind,
                              41000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        # THE SECOND VALUE CLASS, on every product row, and then the band.
        for offset, (name, cell, boundaries, poles, kind) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps, "kind": kind,
                   **run_case(name, cell, boundaries, poles, kind,
                              43000 + offset, args.steps,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "subnormal_band_is_refused_and_the_census_fires",
                     **leg_subnormal_ladder(45000)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            _n, cell, boundaries, poles, kind = _case(name)
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(name, cell, boundaries, poles, kind,
                                          47000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        label, cell, boundaries, poles, kind = _case(MUTATION_CASE)
        for name, functions in mutants.items():
            index += 1
            result = run_case(MUTATION_CASE, cell, boundaries, poles, kind,
                              52000 + index, args.steps, functions=functions)
            caught = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "caught": caught,
                   "passed": bool(caught and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            index += 1
            result = run_case(MUTATION_CASE, cell, boundaries, poles, kind,
                              52000 + index, args.steps, patch=patch)
            caught = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "host_defect": True, "caught": caught,
                   "passed": bool(caught and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above
        # is a harness that diverges on its own.
        index += 1
        result = run_case(MUTATION_CASE, cell, boundaries, poles, kind,
                          52000 + index, args.steps)
        row = {"index": index, "total": total, "leg": "disarm",
               "label": "shipped_bytes_on_the_mutation_case",
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches"),
               "arrays_that_never_moved": result.get("arrays_that_never_moved"),
               "passed": bool(result.get("passed"))}
        rows.append(row)
        emit(handle, row)

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {"product": len(CASES), "separate_controls": len(controls),
                   "shader_mutations": len(mutants),
                   "host_mutations": len(HOST_MUTATIONS)},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
