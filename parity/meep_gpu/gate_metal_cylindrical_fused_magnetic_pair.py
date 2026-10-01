#!/usr/bin/env python3
"""Native MPS byte gate for the CYLINDRICAL COMPLEX fused magnetic Metal kernel:
Dcyl ``step_B`` -> ``update_H``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans THREE driver
passes — ``step_B``, ``zero_metal_B`` and ``update_H`` — so a per-sub-step
comparison could not see it at all: the whole claim is about the SEAM between them.
The comparison is therefore per COMPLETE DRIVER STEP over a stated budget, against
an array-path oracle running the identical live pass list, with the first divergent
step reported rather than a final pass/fail.

THE EIGHT THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION, because the
   oracle is the array path this walk would otherwise take. So every case asserts
   the exact LAUNCH COUNT, a nonzero ``runs``, the exact PREFIX SYNC count (this
   family's pre-pass, which a plan that skipped it would silently share with a
   stale mirror), and that every compared array MOVED from its seeded value.
2. **A hollow pass.** Shader mutations and host-binding mutations are ARMED, and
   each must be CAUGHT. Leg ``disarm`` reruns the identical case with the shipped
   bytes and no host patch and requires zero, so a mutation reported as caught
   cannot be a harness that diverges anyway.
3. **A DEAD-BRANCH mutation.** Every needle is checked against the enclosing guard
   of the case it is scored on, and the guards this family carries are compile-time
   substitutions rather than runtime branches: ``M_ONE`` and ``M_MANY`` emit
   DIFFERENT BODIES, and the wall clear emits a line only on a metallic z. So the
   mutation set is split by case — :data:`M_ONE_CASE` for the axis-increment
   defects, :data:`M_MANY_CASE` for the near-axis-hold defects,
   :data:`WALL_CASE` for the wall clear — and leg ``needle_reachability`` asserts,
   for every needle, that the text it edits IS PRESENT in the source the case it is
   scored on compiles and ABSENT where it should be.
4. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no
   lever, so byte-identity is claimed subject to a CHECKED subnormal-free
   precondition; every case censuses the oracle's own state and reports it.
5. **An unmeasured platform assumption.** This family binds THIRTY-ONE of THIRTY-ONE
   bindings. Leg ``binding_ceiling`` compiles the 38-binding separate-scalar
   signature AND a 32-binding signature and requires BOTH failures, then compiles
   and launches the shipped 31 and reads every ``Params`` field back off the
   device. The 32-binding refutation is what makes "zero headroom" a measurement.
6. **A transcription that drifted.** Both halves are LIFTED from the certified
   emitters rather than retyped, and leg ``transcription`` MEASURES that: the
   certified cylindrical curl body must appear in the fused source VERBATIM on both
   sides of the spliced wall clear, and the constitutive half must differ from the
   certified ``update_H`` body in EXACTLY the three seam lines.
7. **A seam that is not really a seam.** Leg ``byte_neutral_control`` replaces the
   three register reads with reloads of the very words the curl half just stored.
   That mutant must NOT diverge — it is the same value by construction — which is
   what makes "the fusion removes a round trip and changes no arithmetic" a
   measurement instead of a claim. It is the one armed edit required to be
   UNCAUGHT, and it is scored on that.
8. **A refusal that is really an omission.** Leg ``refusal`` requires: an undeclared
   source list refused; a real magnetic ``VolumeSource`` refused BY NAME with the
   driver line cited; a real ELECTRIC ``VolumeSource`` NOT refused (the polarity —
   and on this family the electric case is every corpus row); ``m = 0`` refused by
   name; a Cartesian grid refused; and a real-storage Dcyl run refused.

WHAT THIS GATE DOES NOT CLAIM. Nothing about throughput. The fusion removes ONE
DISPATCH and, on a walled run, one host wall pass; it does NOT remove the radial
prefix pre-pass, which stays on the host because its float32 summation order defines
the answer. Leg ``separate_control`` records the dispatch and in-seam host-pass
counts on both compositions, which is the only place a byte-neutral fusion's
difference shows up at all.

LEGS
  0  expansion            the arm is bound from THIS family's seven-pattern artifact
  1  binding_ceiling      38 separate and 32 packed must FAIL; the shipped 31 must
                          COMPILE, LAUNCH and read every Params field back
  2  needle_reachability  every mutation needle is present in the source of the case
                          it is scored on, and the dead ones are absent
  3  transcription        both halves are the certified emitters' own bytes,
                          differing only in the three seam lines
  4  signed_zero          the +-0 class at one launch, with its vacuity floor
  5  product              complete steps, per-step byte compare, launch + prefix-sync
                          counters, movement, subnormal census
  6  separate_control     the two ALREADY CERTIFIED products stepping the same seam
                          as separate dispatches with the wall clear on the host
                          between them: three-way byte agreement plus the counts
  7  refusal              the source seam, the m split, the geometry split
  8  byte_neutral_control the register-vs-reload edit must NOT diverge
  9  mutation             armed defects, each of which MUST diverge
 10  disarm               the same harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
"""

from __future__ import annotations

import argparse
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

import metal_composition_matrix as matrix  # noqa: E402
import metal_value_classes as values  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    cylindrical_complex, cylindrical_fused_magnetic_pair as family,
    launch as metal_launch, shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

#: The budget every case runs. Twelve, matching every other fused pair's and the
#: whole-step arbiter's, and for their reason: the classes this gate exists for
#: COMPOUND. ``fu_B`` and ``f_w_H`` are state carried between steps, and a
#: coefficient index off by one axis needs several steps to reach the low bits of
#: the interior. The comparison is per COMPLETE STEP.
STEPS = 12

#: Which array-path function each live pass is.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every stored volume a complete step can touch. The ``fu_*`` PML auxiliaries and
#: the ``f_w_*`` constitutive workspaces are STATE: a kernel right for one launch
#: and wrong forever after diverges only once they accumulate. The D/E half is in
#: this list even though this family does not touch it, because a fused pair that
#: corrupted B would reach E through ``step_D`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> ``metal_composition_matrix.cylindrical`` keywords.
#:
#: WHAT THE ROWS CARRY BETWEEN THEM, and every axis here is a COMPILE-TIME arm of
#: the shipped emitter rather than a variation of taste:
#:
#:   |m| CLASS      ``M_ONE`` emits an axis-row increment and no near-axis hold;
#:                  ``M_MANY`` emits the hold over ``zrows`` rows and NO increment.
#:                  They are DIFFERENT BODIES, which is why the mutation set is
#:                  split by case;
#:   z TERMINATION  a metallic z serves an exact 0.0 past the wall, emits the
#:                  ownership mask's z clauses AND is the only axis that can emit a
#:                  wall clear; a periodic z wraps and emits none of the three;
#:   zrows          ``accurate_fields_near_cylorigin`` selects 1 instead of |m| at
#:                  the same m — a RUNTIME uniform, so it varies the Params record
#:                  rather than the source, and the accurate row is what puts that
#:                  field of the struct under test.
#:   m = 0          the THIRD body (2026-09-04): no i*m/r block, no increment, and
#:                  ``Bx[r = 0] = 0`` on the stored register. Its three rows are the
#:                  corpus row's own z termination, the other one, and a
#:                  non-power-of-two Courant so nothing passes because every scale
#:                  factor happened to be exact.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m1_z_metallic", dict(m=1, z_kind="metallic")),
    ("m_minus1_z_metallic", dict(m=-1, z_kind="metallic")),
    ("m1_z_periodic", dict(m=1, z_kind="periodic")),
    ("m3_z_metallic", dict(m=3, z_kind="metallic")),
    ("m2_z_periodic", dict(m=2, z_kind="periodic")),
    ("m3_accurate_z_metallic",
     dict(m=3, z_kind="metallic", accurate=True, courant=0.25)),
    ("m0_z_metallic", dict(m=0, z_kind="metallic")),
    ("m0_z_periodic", dict(m=0, z_kind="periodic")),
    ("m0_odd_courant_z_metallic",
     dict(m=0, z_kind="metallic", courant=0.2718281828)),
)

#: The case the |m| = 1 mutations are armed on. z METALLIC, so the ownership mask's
#: z clauses and the wall clear are BOTH emitted alongside the axis increment.
M_ONE_CASE = "m1_z_metallic"

#: The case the m = 0 mutations are armed on — the ``M_ZERO`` body, which emits the
#: ``Bx[r = 0]`` clear and neither the increment nor the hold nor the coupling.
M_ZERO_CASE = "m0_z_metallic"

#: The case the |m| >= 2 mutations are armed on — the near-axis hold, which the
#: ``M_ONE`` body does not emit at all.
M_MANY_CASE = "m3_z_metallic"

#: The case the wall-clear mutations are armed on. Must be z METALLIC: on a
#: periodic z the shipped emitter writes "no walled axis clears a B component" and a
#: wall needle would edit a line that does not exist.
WALL_CASE = "m1_z_metallic"

#: The row leg ``refusal`` builds its source and geometry questions on.
REFUSAL_CASE: Dict[str, Any] = dict(m=1, z_kind="metallic")

#: Bound by ``leg_expansion`` before any other leg runs.
EXPANSION: Optional[str] = None
PROBE_ARTIFACT: Optional[str] = None


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

def build(keywords: Mapping[str, Any], seed: int,
          value_class: str = values.UNIFORM, scale: float = 1.0,
          ) -> Tuple[Any, Any]:
    """One seeded engine. Called twice (or three times) per case, identically.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the two halves' own gates use, so the fixture this certifies is the
    fixture they were certified on. Only the field state is reseeded here, and
    identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.
    """
    fields, pml = matrix.cylindrical(**dict(keywords))
    if value_class == values.PM_ZERO_LATTICE:
        # The COMPLEX lattice: real and imaginary planes carry OPPOSITE signs of
        # zero at every cell, which is the class this family's complex helpers
        # exist for (see metal_value_classes.pm_zero_lattice_complex).
        lattice = values.pm_zero_lattice_complex(fields.grid.shape)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice
        return fields, pml
    if value_class != values.UNIFORM:
        raise ValueError(f"unknown value class {value_class!r}")
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = ((rng.standard_normal(fields.grid.shape)
                           + 1j * rng.standard_normal(fields.grid.shape))
                          * 0.37 * scale).astype(np.complex64)
    return fields, pml


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is.

    ``None`` is a refusal rather than an empty tuple: a gate that stepped nothing
    would compare a no-op with a no-op and pass.
    """
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def array_step(fields: Any, pml: Any, live: Sequence[str]) -> None:
    for name in live:
        ARRAY_PATH[name](fields, pml)


def metal_step(fields: Any, pml: Any, dispatch: Mapping[str, Any],
               owned: Sequence[str], residency: Residency,
               live: Sequence[str]) -> None:
    """One complete step, on the device where a plan owns the pass.

    A pass no plan owns runs on the array path and is bracketed with an explicit
    ``sync_out`` / ``sync_in``. That bracket is the residency clause made
    operational: without it the next device launch would read the mirror's
    pre-``step_B`` bytes, which is smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns THREE passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``zero_metal_B`` running on the host on top of the clear the
    kernel already carried — which is IDEMPOTENT and would hide a dropped carry.
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

def leg_subnormal_ladder(keywords: Mapping[str, Any], seed: int,
                         budget: int = 4) -> Dict[str, Any]:
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
    def make(scale: float) -> Tuple[Any, Any]:
        return build(keywords, seed, values.UNIFORM, scale)

    def step(engine: Tuple[Any, Any]) -> None:
        fields, pml = engine
        array_step(fields, pml, live_passes(fields, pml))

    def census(engine: Tuple[Any, Any]) -> int:
        return sum(subnormal.census(value)
                   for value in state_of(engine[0]).values())

    return values.census_ladder(make, step, census, budget, log)


def run_case(keywords: Mapping[str, Any], seed: int, steps: int,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(keywords, seed, value_class)
    actual, actual_pml = build(keywords, seed, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_cylindrical_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.cylindrical_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False,
                "reason": "the cylindrical fused magnetic pair was refused",
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
    # THE PREFIX PRE-PASS IS PART OF THE CONTRACT, not diagnostics: a plan that
    # skipped it would step every launch after the first from a stale prefix, which
    # is smooth and wrong. One refresh per run, asserted rather than trusted.
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.prefix_syncs == len(per_step))
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
        "prefix_syncs": plan.prefix_syncs,
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "bcz": plan.bcz, "m_arm": plan.m_arm, "zero_rows": plan.zero_rows,
        "zero_metal": list(plan.zero_metal),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         ) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path, the two ALREADY CERTIFIED Metal products
    stepping the same seam as separate dispatches with the wall clear left on the
    HOST between them, and the fused product. All three must agree word for word at
    every complete step, and the DISPATCH COUNTS, the PREFIX SYNCS and the surviving
    in-seam host passes are recorded on both sides — which is the only place the
    difference between the compositions shows up at all, since a correct fusion is
    byte-neutral by construction.

    The separate side is not a strawman: those two plans are exactly what
    ``plan_step`` composes today for this configuration.

    NOTE WHAT THE COUNTS SHOW AND WHAT THEY DO NOT. Both sides pay ONE prefix
    pre-pass per step; the fusion does not remove it and this leg records it on both
    sides so no reader can mistake the dispatch saving for a scan saving.
    """
    reference, reference_pml = build(keywords, seed)
    separate, separate_pml = build(keywords, seed)
    fused, fused_pml = build(keywords, seed)

    separate_residency = Residency()
    curl = cylindrical_complex.plan_cylindrical_complex_pml_curl(
        separate, separate_pml, "step_B", separate_residency)
    magnetic = cylindrical_complex.plan_cylindrical_complex_constitutive(
        separate, separate_pml, "H", separate_residency)
    fused_residency = Residency()
    plan = family.plan_cylindrical_fused_magnetic_pair(
        fused, fused_pml, sources=(), residency=fused_residency)
    if curl is None or magnetic is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "constitutive": magnetic is not None,
                "fused": plan is not None}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(separate, separate_pml, {"step_B": curl, "update_H": magnetic},
                   ("step_B", "update_H"), separate_residency, live)
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

    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and magnetic.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches": curl.launches + magnetic.launches,
        "fused_dispatches": plan.launches,
        "separate_prefix_syncs": curl.prefix_syncs,
        "fused_prefix_syncs": plan.prefix_syncs,
        "dispatches_removed_per_step": (
            (curl.launches + magnetic.launches - plan.launches) // max(len(per_step), 1)),
        "host_passes_in_seam_on_the_separate_side": (
            ["zero_metal_B"] if "zero_metal_B" in live else []),
        "host_passes_in_seam_on_the_fused_side": [],
        "note": "the prefix pre-pass is paid on BOTH sides once per step; the fusion "
                "removes one DISPATCH and, on a walled run, one in-seam host pass. "
                "No throughput claim is made from these counts.",
    }


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[int, int, Tuple[bool, ...]]:
    """The (bcz, m_arm, walls) triple the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds

    kinds = _boundary_kinds(fields.grid, pml)
    bcz = templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC
    return (bcz, cylindrical_complex.m_class(int(fields.grid.m)),
            zero_metal_axes(fields.grid))


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure mode a mutation leg cannot see from
    its own result.
    """
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


#: Every needle, as DATA, keyed by mutation name -> (case, old, new).
#:
#: SPELLED AS DATA SO THE REACHABILITY LEG AND THE MUTATION LEG CANNOT DRIFT ABOUT
#: WHICH CASE A DEFECT IS SCORED ON. That drift is the dead-branch class this
#: project has already paid for: ``M_ONE`` and ``M_MANY`` compile DIFFERENT BODIES
#: and a metallic-z line does not exist on a periodic-z source, so a needle armed on
#: the wrong case would edit nothing and report UNCAUGHT while measuring nothing.
def _edits(expansion: str) -> Dict[str, Tuple[str, str, str]]:
    other = "NAIVE" if expansion == "FMA_V1" else "FMA_V1"
    zero = templates.COMPLEX_ZERO
    return {
        # ---- THE SEAM ITSELF ------------------------------------------------
        # Take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            (M_ONE_CASE, "float2 src0 = v0;", "float2 src0 = curl0;"),
        # Bx's constitutive source is v0, not v1.
        "seam_takes_the_wrong_component":
            (M_ONE_CASE, "float2 src1 = v1;", "float2 src1 = v0;"),
        # ---- THE CYLINDRICAL CURL HALF --------------------------------------
        # The i*m/r partner registers: target 0 takes `c` (Ez), target 2 takes `a`
        # (Ex). Swapping them is the single most plausible slip in this term.
        "imr_partners_swapped":
            (M_ONE_CASE, "float2 m0 = c_mul(q0, c);", "float2 m0 = c_mul(q0, a);"),
        # The i*m/r coefficient is on the LEFT (S:724 `factor * partner_values`).
        # For a FULL complex product the orientation is load-bearing.
        "imr_operands_swapped":
            (M_ONE_CASE, "float2 m2 = c_mul(q2, a);", "float2 m2 = c_mul(a, q2);"),
        # The row is indexed by the RADIAL index i, not by the lane or by z.
        "imr_row_indexed_by_z":
            (M_ONE_CASE, "float2 q0 = c0[i];", "float2 q0 = c0[k];"),
        # `curl - m` carries the array path's `curl + (-(c*g))`. Flipping the sign
        # is the classic transcription slip on an additive coupling term.
        "imr_sign_flipped":
            (M_ONE_CASE, "curl0 = curl0 - m0;", "curl0 = curl0 + m0;"),
        # THE PREFIX IS Bz's WHOLE CURL on the B side (S:343-347): ONE subtract and
        # ONE multiply. Falling back to the four-operand grouping is a different
        # float32 number per plane and is exactly what the prefix exists to avoid.
        "prefix_replaced_by_the_four_operand_grouping":
            (M_ONE_CASE, "curl2 = c_mul_coefficient_left(dtdx, pu - pd);",
             "curl2 = c_mul_coefficient_left(dtdx, ((b_x - b) + (a - a_y)));"),
        # The extended prefix's row stride is nyz CELLS. Reading the same row twice
        # is a zero difference everywhere, which is smooth and wrong.
        "prefix_row_stride_dropped":
            (M_ONE_CASE, "float2 pu = pfx[ii + nyz];", "float2 pu = pfx[ii];"),
        # ---- THE |m| = 1 AXIS INCREMENT (M_ONE body only) -------------------
        # NOTE: `axis_increment_accumulates` is NOT armed here. It is byte-visible
        # only on the +-0 LATTICE and leg `signed_zero` owns it, where the census
        # proves the row is not vacuous. Arming it on this uniformly seeded case
        # would report it UNCAUGHT for a reason that has nothing to do with the
        # kernel — see VALUE_CLASS_FINDING.
        # The negation is real: `-(A - B)` is NOT `B - A` at signed zeros, and the
        # whole increment is negated once.
        "axis_increment_not_negated":
            (M_ONE_CASE, "curl0 = at_x ? -inc : curl0;",
             "curl0 = at_x ? inc : curl0;"),
        # Ez[r+1] is the FIRST OFF-AXIS row (S:642's take(Ez, 1, axis=0)). Reading
        # the axis row instead is a plausible off-by-one and is silently in bounds.
        "axis_increment_reads_the_axis_row":
            (M_ONE_CASE, "float2 e1 = g2[nyz + j * nzi + k];",
             "float2 e1 = g2[j * nzi + k];"),
        # The increment sits on target 0 (Bx) on the B side, target 1 (Dy) on D.
        "axis_increment_lands_on_the_wrong_target":
            (M_ONE_CASE, "curl0 = at_x ? -inc : curl0;",
             "curl1 = at_x ? -inc : curl1;"),
        # ---- THE |m| >= 2 NEAR-AXIS HOLD (M_MANY body only) -----------------
        # The hold covers all three components AND their fu (post-#3164). Dropping
        # the auxiliaries is the 1.29-era component set.
        "axis_hold_skips_the_auxiliaries":
            (M_MANY_CASE, "        n0 = near ? float2(0.0f, 0.0f) : n0;\n", ""),
        # `zrows` is a THRESHOLD on the row index, not a count of one.
        "axis_hold_covers_only_row_zero":
            (M_MANY_CASE, "bool near = (i < int(zrows));", "bool near = (i == 0);"),
        # ---- THE m = 0 AXIS RULE (M_ZERO body only, 2026-09-04) ------------
        # stepping._cylindrical_axis_zero_B (:661-662): Bx[r=0] = 0, the FIELD only,
        # after the recurrence. Dropped, moved to the wrong component, and
        # over-carried onto the auxiliary — each a transcription slip a reader could
        # make from the D side's rule or from the |m| >= 2 hold.
        "m_zero_axis_clear_dropped":
            (M_ZERO_CASE, "    v0 = at_x ? float2(0.0f, 0.0f) : v0;",
             "    // MUTANT: Bx[r=0] kept"),
        "m_zero_axis_clear_on_the_wrong_component":
            (M_ZERO_CASE, "    v0 = at_x ? float2(0.0f, 0.0f) : v0;",
             "    v1 = at_x ? float2(0.0f, 0.0f) : v1;"),
        "m_zero_axis_clear_also_masks_the_auxiliary":
            (M_ZERO_CASE, "    v0 = at_x ? float2(0.0f, 0.0f) : v0;",
             "    v0 = at_x ? float2(0.0f, 0.0f) : v0;\n"
             "    n0 = at_x ? float2(0.0f, 0.0f) : n0;"),
        # ---- THE WALL CLEAR (metallic z only) -------------------------------
        "zero_metal_dropped":
            (WALL_CASE, f"    v2 = at_z ? {zero} : v2;\n", ""),
        # THE WALL TABLE IS THE DIAGONAL FOR B: Bz clears on a z wall. Clearing Bx
        # there is the D-side (off-diagonal) table, the most likely porting slip.
        "zero_metal_uses_the_d_side_table":
            (WALL_CASE, f"    v2 = at_z ? {zero} : v2;",
             f"    v0 = at_z ? {zero} : v0;"),
        # THE CLEAR IS A *COMPLEX* ZERO — both planes (S:1896, :1902).
        "zero_metal_clears_only_the_real_plane":
            (WALL_CASE, f"    v2 = at_z ? {zero} : v2;",
             "    v2 = at_z ? float2(0.0f, v2.y) : v2;"),
        # `zero_metal_B` passes B_COMPONENTS only (S:2203); masking fu_B as well is
        # a plausible over-carry, and it is NOT what the array path does.
        "zero_metal_also_masks_the_auxiliary":
            (WALL_CASE, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
             f"    u0[ii] = n0; u1[ii] = n1; u2[ii] = at_z ? {zero} : n2;"),
        # ---- THE CERTIFIED COMPLEX BASE -------------------------------------
        # THE ARM. Swapping the whole helper block to the other licensable arm is
        # the defect the expansion probe exists to prevent. It is reachable here
        # through the i*m/r product and the |m| = 1 scalar — both of which have a
        # ZERO real word, where the two arms are DEGENERATE — so this row is
        # expected to be a PREDICTED NULL rather than a catch, and it is scored
        # that way with the reason recorded. See PREDICTED_NULL_REASON.
        "expansion_arm_swapped":
            (M_ONE_CASE, templates.complex_helpers(expansion),
             templates.complex_helpers(other)),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        #
        # ARMED ON t1 AND ONLY t1, AND THAT IS A MEASUREMENT RATHER THAN A CHOICE.
        # `t0` and `t2` each lead with a PHI SELF-DIFFERENCE — `c_y` is the phi wrap
        # of `c` on a one-cell axis and is the SAME element, so `(c_y - c)` is an
        # exact +0.0 and both associations reduce to `b - b_z`. `t2` is null twice
        # over: on the B side `curl2` is unconditionally overwritten by the prefix
        # block, so the whole statement is dead. Measured on this host 2026-08-20:
        # t0 flattened 0 words, t2 flattened 0 words, t1 flattened 528 words at
        # step 1. The two nulls are carried below as predicted nulls with that
        # measurement, because deleting them would hide the reason.
        "curl_parens_flattened":
            (M_ONE_CASE, "float2 t1 = ((a_z - a) + (c - c_x));",
             "float2 t1 = (a_z - a + c - c_x);"),
        "curl_parens_flattened_on_the_invariant_term":
            (M_ONE_CASE, "float2 t0 = ((c_y - c) + (b - b_z));",
             "float2 t0 = (c_y - c + b - b_z);"),
        # step_B is FORWARD. step_D's negated strides are a different product.
        "curl_direction_reversed":
            (M_ONE_CASE, "int si = i + 1, sj = j + 1, sk = k + 1;",
             "int si = i - 1, sj = j - 1, sk = k - 1;"),
        # THE RECURRENCE PAIRS are vec.hpp's cycle_direction: target 0 takes (y, z).
        "recurrence_axis_pair_swapped":
            (M_ONE_CASE,
             "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);",
             "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_z) - curl0, si_z);"),
        "fu_store_dropped":
            (M_ONE_CASE, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
             "    u1[ii] = n1; u2[ii] = n2;"),
        "flux_store_dropped":
            (M_ONE_CASE, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
             "    f1[ii] = v1; f2[ii] = v2;"),
        "fw_store_dropped":
            (M_ONE_CASE, "    w0[ii] = src0;", "    // stale f_w_Hx"),
        # The dsigw index is the component's OWN axis (S:226-227), not the curl's
        # cycle. Moving it one axis over is a silent half-cell error.
        "constitutive_coefficient_index_moved":
            (M_ONE_CASE, "float kp_0 = kp0[i], km_0 = km0[i];",
             "float kp_0 = kp0[j], km_0 = km0[j];"),
        "constitutive_accumulations_reversed":
            (M_ONE_CASE,
             "    a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"
             "    a0 = a0 - c_mul_coefficient_left(km_0, prev0);",
             "    a0 = a0 - c_mul_coefficient_left(km_0, src0);\n"
             "    a0 = a0 + c_mul_coefficient_left(kp_0, prev0);"),
        # `prev` is read BEFORE `fw` is written (S:2083-2085). Reversed, this is
        # wrong only where kms != 0 -- INSIDE THE PML ONLY -- and looks like a
        # slightly worse absorber, not like a bug.
        "constitutive_prev_read_after_write":
            (M_ONE_CASE,
             "    float2 prev0 = w0[ii];\n"
             "    // THE SEAM: the register step_B just wrote, not a reload of Bx.\n"
             "    float2 src0 = v0;\n"
             "    w0[ii] = src0;",
             "    // THE SEAM: the register step_B just wrote, not a reload of Bx.\n"
             "    float2 src0 = v0;\n"
             "    w0[ii] = src0;\n"
             "    float2 prev0 = w0[ii];"),
    }


#: The rows scored on NOT diverging, with the measured reason each is null for.
#:
#: THE ARM SWAP IS A PREDICTED NULL ON THIS FAMILY, and that is a MEASURED property
#: of the cylindrical tranche rather than a convenience. FMA_V1 and NAIVE differ
#: only where a complex product's coefficient has a NONZERO real part: with
#: ``c_re = ±0.0`` the product ``c_re * z_re`` is exact, so
#: ``fma(c_re, z_re, -(c_im*z_im))`` and ``(c_re*z_re) - (c_im*z_im)`` are the same
#: single-rounding operation. This kernel's only general complex products are the
#: two i*m/r rows (real word exactly ``+0.0`` — the ``(-1j) * X`` spelling launders
#: the sign) and the |m| = 1 scalar ``1j*(m*dtdx)`` (real word ``±0.0``). Every
#: other multiply takes a REAL coefficient, where the arms coincide by construction.
#: THIS FAMILY HAS NO BLOCH ROTATION — it refuses a ``k_point`` by name — so unlike
#: the complex twin there is no phased row on which the arm becomes visible.
#:
#: So the row is armed, run, and scored on being NULL. Reporting it as caught would
#: be false; deleting it would hide the fact that the arm binding is INERT here,
#: which a reader of ``leg_expansion`` needs to know.
PREDICTED_NULL: Dict[str, str] = {
    "expansion_arm_swapped":
        "the two arms differ only where a complex product's coefficient has a "
        "NONZERO real word, and every general complex product in this kernel takes "
        "a coefficient whose real word is a zero (the i*m/r rows are exactly +0.0, "
        "the |m| = 1 scalar is ±0.0) — measured by the cylindrical tranche's own "
        "probe, which reports AMBIGUOUS_BOTH for both orientations. This family "
        "compiles no Bloch rotation, so there is no phased row on which the arm is "
        "discriminating. The binding is inert HERE and is still bound from a "
        "measured artifact, because a later widening could make it live.",
    "imr_operands_swapped":
        "the i*m/r coefficient's REAL WORD IS EXACTLY +0.0 on every entry of every "
        "row this family binds (measured on this host 2026-08-20: 16/16 words are "
        "0x00000000 at m = +-1 and m = 3, on both targets, because "
        "stepping:713-718 spells the numerator (-1j) * X and the cross-term "
        "subtraction launders the sign for both signs of X). With c_re = +-0.0 the "
        "product c_re * z_re is exact, so c_mul(q, z) and c_mul(z, q) are the same "
        "single-rounding operation on both planes. THE ORIENTATION IS STILL WHAT "
        "THE ARRAY PATH COMPUTES and is kept for that reason, pinned by a "
        "source-text assertion in meep_gpu/test_metal_cylindrical_fused_magnetic_"
        "pair.py; if a future change ever gives a coefficient row a nonzero real "
        "word, this row must be re-armed as a catch.",
    "curl_parens_flattened_on_the_invariant_term":
        "t0 leads with the PHI SELF-DIFFERENCE (c_y - c). phi has n = 1, so the "
        "wrap returns the SAME element and the difference is an exact +0.0; both "
        "associations then reduce to `b - b_z` and no word can differ. Measured on "
        "this host 2026-08-20: 0 differing words over 12 complete steps, against "
        "528 for the identical edit on t1, which carries no invariant-axis operand "
        "and is armed as a catch. The grouping is kept because it is what the "
        "array path computes and is pinned by a source-text assertion.",
}


#: THE VALUE-CLASS FINDING, measured 2026-08-20 on both backends.
VALUE_CLASS_FINDING = (
    "the increment REPLACES row 0 rather than accumulating, and after the "
    "ownership mask that row is exactly +0.0 — so `+0.0 + x` and `x` differ ONLY "
    "at x = -0.0. WHICH VALUE CLASS REACHES THAT IS THE WHOLE QUESTION. A random "
    "seed reaches it with probability zero. A ZERO-INIT state cannot manufacture "
    "it on this seam either: Bx's split-field dsig axis is PHI, whose kms is 1.0 "
    "and never negative, so the `fu * kms` minuend from an all-+0.0 state is "
    "+0.0. A +-0 LATTICE DOES: it puts -0.0 words into `fu` itself and "
    "`-0.0 * 1.0` keeps the sign. MEASURED: NULL under random seeding and under "
    "zero-init with 2- and 3-cell absorbers, CAUGHT on the lattice on both "
    "backends. The first cut of this gate swept zero-init only and reported the "
    "defect UNCAUGHT; the pair of rows is what corrects it.")


#: THE ARM SWAPS: the WHOLE body of another m class compiled for a case, which is the
#: m-class split armed rather than argued. ``(case, arm compiled instead)``. The
#: m = 0 body at |m| = 1 lacks the coupling and the increment; the |m| = 1 body at
#: m = 0 REPLACES Bx's axis row with an increment the array path never forms (its
#: i*m/r rows are exactly zero there, so the coupling alone would be a signed-zero
#: null — the increment is what makes the swap visible on a uniform seed).
ARM_SWAPS: Dict[str, Tuple[str, int]] = {
    "m_zero_case_compiled_with_the_m_one_body": (M_ZERO_CASE, cylindrical_complex.M_ONE),
    "m_one_case_compiled_with_the_m_zero_body": (M_ONE_CASE, cylindrical_complex.M_ZERO),
}


def shader_mutations(specialisations: Mapping[str, Tuple[int, int, Sequence[bool]]],
                     expansion: str) -> Dict[str, Dict[str, Any]]:
    """Compile every armed defect against the source of the case it is scored on."""
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, (case, old, new) in _edits(expansion).items():
        bcz, arm, walls = specialisations[case]
        base = family.cylindrical_fused_magnetic_pair_source(
            bcz, arm, walls, expansion)
        mutants[name] = {
            "case": case,
            "functions": {shaders.CONTRACT_OFF:
                          compile_source(needle(base, old, new)
                                         ).cylindrical_fused_magnetic_pair_step},
        }
    for name, (case, other_arm) in ARM_SWAPS.items():
        bcz, arm, walls = specialisations[case]
        assert other_arm != arm, (name, arm)
        swapped = family.cylindrical_fused_magnetic_pair_source(
            bcz, other_arm, walls, expansion)
        mutants[name] = {
            "case": case,
            "functions": {shaders.CONTRACT_OFF:
                          compile_source(swapped).cylindrical_fused_magnetic_pair_step},
        }
    return mutants


def byte_neutral_source(bcz: int, arm: int, walls: Sequence[bool],
                        expansion: str) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and
    ``v0`` hold the same complex64 word pair, and a float2 stored to a
    ``device float2*`` and reloaded is bit-identical to the register. This edit is
    therefore the fusion's central claim written as a program, and leg
    ``byte_neutral_control`` requires it NOT to diverge.
    """
    source = family.cylindrical_fused_magnetic_pair_source(bcz, arm, walls, expansion)
    for target in range(3):
        source = needle(source, f"float2 src{target} = v{target};",
                        f"float2 src{target} = f{target}[ii];")
    return source


#: Where each binding group starts in the plan's argument tuple. Spelled once, so
#: the host mutations and the kernel signature cannot drift apart. ASSERTED against
#: PACKED_BINDINGS below, so a signature change fails here rather than rebinding a
#: buffer the mutation did not mean to touch.
CURL_COEFFICIENT_SLOTS = tuple(range(12, 18))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(24, 30))
PARAMS_SLOT = 30
IMR_ROW_SLOTS = (10, 11)
assert PARAMS_SLOT + 1 == family.PACKED_BINDINGS, (PARAMS_SLOT, family.PACKED_BINDINGS)


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the Dcyl B curl the INTEGER split-field coefficients.

    ``step_B`` reads HALF-INTEGER positions and ``step_D`` integer ones
    (``SUB_STEPS['step_B']['suffix'] == '_h'``). The kernel cannot tell: it is a
    half-cell error in the absorber profile, not a crash.
    """
    _rebind(plan, CURL_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


def swap_h_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the HALF-INTEGER constitutive coefficients to the H half.

    ``update_H`` takes ``kps_a``/``kms_a`` and ``update_E`` takes ``kps_a_h``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from, so no shader mutation can reach this.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_imr_rows(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind target 2's i*m/r row to target 0 and vice versa.

    The two rows differ by the target's OWN radial Yee shift and by the SIGN of the
    term (``IMR_TERMS``: Bx <- Ez at +1, Bz <- Ex at -1), and the kernel takes two
    pointers it cannot distinguish. No shader mutation can reach this: the swap is
    entirely in what the host bound.
    """
    _rebind(plan, IMR_ROW_SLOTS,
            [plan._args[IMR_ROW_SLOTS[1]], plan._args[IMR_ROW_SLOTS[0]]])


def naive_params_tensor(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: the NAIVE end-to-end Params record — the alignment trap, armed.

    THE MOST IMPORTANT HOST MUTATION IN THIS FILE, because it is the one a future
    editor is most likely to write by accident. Metal aligns ``float2`` to 8 bytes;
    a NumPy record whose fields are laid end to end puts ``inc_b`` one word early
    once the scalars come first. Shipped, ``inc_b`` is FIRST and the natural offsets
    coincide, so this mutant reproduces the trap by moving it LAST.

    The result is not a crash and not noise: ``inc_b`` reads a well-formed complex
    number built from its neighbours' words. This mutation is what turns "the layout
    was measured" from a docstring into a gate row.
    """
    import torch  # noqa: PLC0415

    naive = np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                      ("n_elem", "<u4"), ("zrows", "<u4"), ("dtdx", "<f4"),
                      ("minus_dtdx", "<f4"), ("axis_coef", "<f4"),
                      ("inc_b", "<f4", 2)])
    record = np.zeros(1, dtype=naive)
    nx, ny, nz = plan.shape
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["zrows"] = plan.zero_rows
    record["dtdx"] = np.float32(plan.dtdx)
    minus_dtdx, inc_b = cylindrical_complex.axis_increment_scalars(
        _M_OF_CASE[M_ONE_CASE], plan.dtdx)
    record["minus_dtdx"] = np.float32(minus_dtdx)
    record["axis_coef"] = np.float32(cylindrical_complex.axis_coefficient(plan.dtdx))
    record["inc_b"] = (np.float32(inc_b[0]), np.float32(inc_b[1]))
    words_ = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    pad = (family.PARAMS_ITEMSIZE - words_.nbytes) // 4
    if pad > 0:
        words_ = np.concatenate([words_, np.zeros(pad, dtype=np.int32)])
    _rebind(plan, (PARAMS_SLOT,),
            (torch.from_numpy(words_).to(torch.device(residency.device)),))


#: m per case name, so the params mutation can rebuild the scalars the shipped plan
#: computed without re-deriving them from the plan (which does not hold m).
_M_OF_CASE: Dict[str, int] = {name: int(kw.get("m", 1)) for name, kw in CASES}

HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
    "imr_rows_swapped_between_targets": swap_imr_rows,
    "params_record_uses_the_naive_layout": naive_params_tensor,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_expansion() -> Dict[str, Any]:
    """Which arm was bound, from which artifact, and the three refusals.

    THE ARM IS NOT A FREE CHOICE. A complex product has two licensable
    transcription arms and which one the REFERENCE takes is a measured platform
    fact. This family binds through the CYLINDRICAL family's probe, which classifies
    SEVEN orientations — ``complex_fields``' base five plus the i*m/r row and the
    |m| = 1 scalar — because this kernel performs all seven. Reading the plain
    complex family's five-pattern artifact would licence a kernel from a record that
    never measured two of its calls.

    Four questions:

    1. an artifact must be present and must name exactly one arm;
    2. a MISSING probe must refuse — never a default arm;
    3. an AMBIGUOUS probe must ALSO refuse;
    4. a record carrying only the BASE FIVE patterns must refuse, which is what
       makes the separate environment variable load-bearing rather than decorative.

    AND THE CONSEQUENCE THIS LEG RECORDS RATHER THAN LEAVES IMPLICIT: on this family
    the arm is INERT. Every general complex product here has a coefficient whose
    real word is a zero, so the two arms coincide bit for bit — see
    :data:`PREDICTED_NULL`. The binding is still made from a measurement, because a
    widening (a z Bloch phase, an E-side ``inv_eps``) would make it live.
    """
    global EXPANSION, PROBE_ARTIFACT
    configured = os.environ.get(cylindrical_complex.PROBE_PATH_ENVIRONMENT)
    record = cylindrical_complex.load_expansion_probe()
    arm = cylindrical_complex.expansion_from_probe(record)
    EXPANSION, PROBE_ARTIFACT = arm, configured

    missing_refused = cylindrical_complex.expansion_from_probe(None) is None
    ambiguous = {"backend": cylindrical_complex.PROBE_BACKEND,
                 "patterns": {name: "AMBIGUOUS_BOTH"
                              for name in cylindrical_complex.CYLINDRICAL_PROBE_PATTERNS}}
    ambiguous_refused = cylindrical_complex.expansion_from_probe(ambiguous) is None
    base_only = {"backend": cylindrical_complex.PROBE_BACKEND,
                 "patterns": {name: value
                              for name, value in ((record or {}).get("patterns")
                                                  or {}).items()
                              if name not in (cylindrical_complex.IMR_ROW_PROBE_PATTERN,
                                              cylindrical_complex.AXIS_SCALAR_PROBE_PATTERN)}}
    base_only_refused = cylindrical_complex.expansion_from_probe(base_only) is None

    digest = (hashlib.sha256(Path(configured).read_bytes()).hexdigest()
              if configured and Path(configured).is_file() else None)
    return {
        "passed": bool(arm in templates.EXPANSIONS and missing_refused
                       and ambiguous_refused and base_only_refused),
        "arm": arm,
        "probe_artifact": configured,
        "probe_sha256": digest,
        "probe_backend": (record or {}).get("backend"),
        "probe_patterns": (record or {}).get("patterns"),
        "required_patterns": list(cylindrical_complex.CYLINDRICAL_PROBE_PATTERNS),
        "host_numpy": np.__version__,
        "missing_probe_refused": missing_refused,
        "ambiguous_probe_refused": ambiguous_refused,
        "base_five_only_probe_refused": base_only_refused,
        "note": "the arm is INERT on this family: every general complex product "
                "here takes a coefficient whose real word is a zero, and there is "
                "no Bloch rotation. The arm-swap mutation is therefore a PREDICTED "
                "NULL and is scored as one.",
    }


#: The absorber depths leg ``signed_zero`` sweeps, and the one that is VACUOUS.
#:
#: Measured on this host 2026-08-20 on the Dcyl fixture: a 2-cell absorber gives
#: ``min kms_z = -1.396`` (integer lattice) / ``-0.348`` (half-integer), a 3-cell one
#: ``-0.597`` / ``-0.109``, and a 5-cell one ``+0.042`` / ``+0.224`` — no negative
#: coefficient anywhere. So the FIVE-CELL ROW CARRIES NO SIGNED-ZERO PRODUCER AT ALL
#: and its census is 0: a mutation reported null there would be null for a reason
#: that has nothing to do with the mutation. It is kept as the VACUITY CONTROL, and
#: the leg FAILS if a depth declared non-vacuous censuses zero or if the depth
#: declared vacuous censuses anything.
SIGNED_ZERO_DEPTHS: Tuple[Tuple[int, bool], ...] = ((2, True), (3, True), (5, False))


def _thin_absorber(keywords: Mapping[str, Any], depth: int,
                   value_class: str, seed: int) -> Tuple[Any, Any]:
    """The Dcyl fixture with a chosen absorber depth and a chosen VALUE CLASS.

    Built from the SHARED matrix builder so the grid and the material are the ones
    every other leg uses, with only the absorber replaced — a second grid builder
    here would be a second fixture, and the comparison would stop being about the
    kernel.

    ``zero_init`` is every word ``+0.0``. ``signed_zero_lattice`` is every word an
    exact zero with a RANDOM SIGN, and the two are NOT the same class: MEASURED on
    the GPU host 2026-08-20, the ``axis_increment_accumulates`` needle is NULL under
    zero-init on this seam and CAUGHT on the lattice, because the lattice puts
    ``-0.0`` words into ``fu`` itself and ``-0.0 * 1.0`` keeps the sign where an
    all-``+0.0`` state cannot manufacture one (Bx's split-field dsig axis is phi,
    whose ``kms`` is 1.0 and never negative). A gate that swept zero-init alone
    would have reported that defect uncaught.
    """
    from meep_gpu.pml import PML  # noqa: PLC0415

    fields, _pml = matrix.cylindrical(**dict(keywords))
    # THE DRAW IS INDEPENDENT OF THE ABSORBER DEPTH, deliberately: the depth axis
    # of this leg is about which coefficients go negative, and a lattice that also
    # changed with it would confound the two. Seeded from a DIGEST, not hash(),
    # so a failing draw can be replayed in another process.
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(
            f"{value_class}|{seed}".encode()).digest()[:4], "big"))
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if value_class == "zero_init":
            array[...] = 0
        elif value_class == "signed_zero_lattice":
            shape = fields.grid.shape
            signs = 1 - 2 * rng.integers(0, 2, size=shape).astype(np.float32)
            imag = 1 - 2 * rng.integers(0, 2, size=shape).astype(np.float32)
            array[...] = ((np.float32(0.0) * signs)
                          + 1j * (np.float32(0.0) * imag)).astype(np.complex64)
        else:
            raise ValueError(f"unknown value class {value_class!r}")
    return fields, PML(grid=fields.grid, thickness={"x": (0, depth), "z": depth})


def _zero_init_launch(keywords: Mapping[str, Any], depth: int,
                      value_class: str = "zero_init", draw: int = 1,
                      functions: Optional[Mapping[str, Any]] = None,
                      ) -> Dict[str, int]:
    """ONE fused launch from a zero state, against ONE array-path B->H seam.

    Per-launch rather than per-complete-step, and that is the second half of the
    cylindrical tranche's finding: the divergence this class produces is TRANSIENT —
    a ``-0.0`` that survives one sub-step is laundered back to ``+0.0`` by the next
    ``fu *= kms`` sign flip — so an end-of-budget compare reports 0 on a state that
    genuinely diverged at step 1.
    """
    reference, reference_pml = _thin_absorber(keywords, depth, value_class, draw)
    actual, actual_pml = _thin_absorber(keywords, depth, value_class, draw)
    residency = Residency()
    plan = family.plan_cylindrical_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    assert plan is not None, family.cylindrical_fused_magnetic_pair_coverage(
        actual, actual_pml, (), residency).reasons
    before = frozen(actual)
    residency.sync_in()
    stepping.step_B(reference, reference_pml)
    stepping.zero_metal_B(reference)
    stepping.update_H(reference, reference_pml)
    plan.run()
    residency.sync_out()
    reference_state = state_of(reference)
    return {
        "differing_words": sum(compare(reference, actual).values()),
        "negative_zero_census": int(sum(
            np.count_nonzero(words(value) == 0x80000000)
            for value in reference_state.values())),
        "moved_words": int(sum(differing(before[name], value)
                               for name, value in state_of(actual).items())),
        "launches": plan.launches,
    }


def leg_signed_zero() -> Dict[str, Any]:
    """The +-0 class, measured at the launch where it is live — and its vacuity floor.

    THE MOST IMPORTANT LEG FOR THE THREE PREDICTED NULLS IN THIS FILE, because it is
    the one that decides whether a null is a fact about the kernel or a fact about
    the seeding. Random seeds are provably blind to the signed-zero class: the
    reachable producer is a NEGATIVE ``kms = kappa - sigma`` times a quiet ``+0.0``
    word, which needs a THIN absorber, and an exact ``-0.0`` difference has
    probability zero under a continuous seed.

    So this leg runs a zero-initialised state against thin absorbers, ASSERTS THE
    CENSUS IS NON-ZERO IN RUN (a row whose census is zero measures nothing, whatever
    its differing-word count says), requires the shipped kernel to be identical
    there, and then runs the ``axis_increment_accumulates`` mutant on the same
    configuration and RECORDS its result. The mutant is expected to be NULL — Bx's
    split-field dsig axis is phi, whose ``kms`` is 1.0 and never negative — and that
    expectation is what the leg measures rather than assumes.

    THE FIVE-CELL ROW IS THE VACUITY CONTROL and is required to census ZERO. Without
    it, a future absorber-depth change could quietly turn every row of this leg
    vacuous and the leg would keep passing.
    """
    rows: List[Dict[str, Any]] = []
    base = family.cylindrical_fused_magnetic_pair_source(
        templates.METALLIC, cylindrical_complex.M_ONE, (False, False, True),
        EXPANSION or "FMA_V1")
    # THE TWO NEEDLES WHOSE VISIBILITY IS A QUESTION ABOUT THE VALUE CLASS. Both
    # differ from the shipped spelling ONLY in the sign of a zero, so both are
    # invisible on a uniform seed by construction; whether they are visible at all
    # is what this leg measures rather than asserts.
    needles = {
        "axis_increment_accumulates":
            ("curl0 = at_x ? -inc : curl0;",
             "curl0 = at_x ? (curl0 - inc) : curl0;"),
        "imr_operands_swapped":
            ("float2 m2 = c_mul(q2, a);", "float2 m2 = c_mul(a, q2);"),
    }
    compiled = {name: {shaders.CONTRACT_OFF:
                       compile_source(needle(base, old, new)
                                      ).cylindrical_fused_magnetic_pair_step}
                for name, (old, new) in needles.items()}
    ok = True
    # THREE DRAWS OF THE LATTICE, because the catch is a SINGLE CELL — the axis row
    # whose seeded sign exposes the difference — and one draw is a statement about
    # that draw. Recorded per draw so the reader sees the spread rather than a
    # summary that hides it.
    for value_class, draw in (("zero_init", 1), ("signed_zero_lattice", 1),
                              ("signed_zero_lattice", 2),
                              ("signed_zero_lattice", 3)):
        for depth, expect_live in SIGNED_ZERO_DEPTHS:
            shipped = _zero_init_launch(REFUSAL_CASE, depth, value_class, draw)
            live = (shipped["negative_zero_census"] > 0
                    and shipped["moved_words"] > 0)
            expected = expect_live or value_class == "signed_zero_lattice"
            row = {
                "value_class": value_class,
                "draw": draw,
                "absorber_cells": depth,
                "declared_non_vacuous": expected,
                "negative_zero_census": shipped["negative_zero_census"],
                "moved_words": shipped["moved_words"],
                "shipped_differing_words": shipped["differing_words"],
                "non_vacuous": live,
                "vacuity_matches_declaration": live == expected,
                "shipped_identical": shipped["differing_words"] == 0,
            }
            for name, functions in compiled.items():
                row[f"{name}_differing_words"] = _zero_init_launch(
                    REFUSAL_CASE, depth, value_class, draw,
                    functions=functions)["differing_words"]
            rows.append(row)
            ok = ok and row["vacuity_matches_declaration"] and row["shipped_identical"]
    lattice = [row for row in rows if row["value_class"] == "signed_zero_lattice"]
    zeroed = [row for row in rows if row["value_class"] == "zero_init"]
    # AT LEAST ONE NON-VACUOUS LATTICE ROW, not all of them. The catch is a single
    # cell and whether a given draw exposes it is a property of that draw; demanding
    # every row would make this leg fail on a seed rather than on the kernel.
    caught_on_lattice = any(row["axis_increment_accumulates_differing_words"]
                            for row in lattice if row["non_vacuous"])
    null_on_zero_init = not any(row["axis_increment_accumulates_differing_words"]
                                for row in zeroed)
    return {
        # THE PASS CONDITION IS THE PAIR, and that is the finding. The needle must
        # be CAUGHT on the +-0 lattice (where `fu` itself carries -0.0 words) and
        # NULL under zero-init (where an all-+0.0 state cannot manufacture one on
        # this seam). A gate that swept zero-init alone reported it uncaught and
        # would have called the kernel certified — measured on the GPU host 2026-08-20
        # by the Triton twin, and re-measured here.
        "passed": bool(ok and caught_on_lattice and null_on_zero_init),
        "rows": rows,
        "accumulate_mutant_caught_on_the_signed_zero_lattice": caught_on_lattice,
        "accumulate_mutant_by_row": {
            f"{entry['value_class']}/draw{entry['draw']}@{entry['absorber_cells']}":
                entry["axis_increment_accumulates_differing_words"]
            for entry in rows},
        "accumulate_mutant_null_under_zero_init": null_on_zero_init,
        "imr_operands_swapped_by_row": {
            f"{entry['value_class']}/draw{entry['draw']}@{entry['absorber_cells']}":
                entry["imr_operands_swapped_differing_words"] for entry in rows},
        "reason": VALUE_CLASS_FINDING,
    }


def leg_binding_ceiling() -> Dict[str, Any]:
    """Both refuted signatures must FAIL; the shipped 31 must COMPILE and LAUNCH.

    This family sits AT the ceiling, and the two refutations license different
    claims:

    * 38 bindings with the eight non-pointer arguments bound separately, the way the
      certified 26-binding cylindrical curl binds them — which licenses "the packed
      struct is FORCED by the twelve pointers the fusion brings", not by style;
    * 32 bindings, one more pointer than the shipped signature — which licenses
      "ZERO HEADROOM" as a measurement rather than as a sentence. The failure is the
      front end's own: ``[[buffer(31)]]`` is out of bounds on this toolchain.

    A signature that compiled but bound the struct wrongly would produce a smooth,
    plausible, wrong field, so the shipped one is not merely compiled — it is
    LAUNCHED and every ``Params`` field is read back off the device.
    """
    separate_failed, separate_error = False, None
    try:
        compile_source(family.refuted_separate_scalar_source())
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        separate_failed, separate_error = True, str(exc)[:400]

    over_failed, over_error = False, None
    try:
        compile_source(family.refuted_thirty_second_binding())
    except Exception as exc:  # noqa: BLE001
        over_failed, over_error = True, str(exc)[:400]

    # The shipped signature, compiled AND launched, with every Params field read
    # back through a probe kernel that writes each one into a float2 volume.
    import torch  # noqa: PLC0415

    readback = family._TEMPLATE.replace("__BODY__", """
    if (idx >= 1u) { return; }
    f0[0] = float2(float(nx), float(ny));
    f0[1] = float2(float(nz), float(n_elem));
    f0[2] = float2(float(zrows), dtdx);
    f0[3] = float2(minus_dtdx, axis_coef);
    f0[4] = inc_b;
""")
    readback = templates.substitute(readback, {
        "__CONTRACT__": shaders.contraction_pragma(shaders.CONTRACT_OFF),
        "__HELPERS__": templates.complex_helpers(EXPANSION or "FMA_V1"),
    })
    function = compile_source(readback).cylindrical_fused_magnetic_pair_step
    out = torch.zeros(8, dtype=torch.complex64, device="mps")
    others = [torch.zeros(8, dtype=torch.complex64, device="mps") for _ in range(14)]
    reals = [torch.zeros(8, dtype=torch.float32, device="mps") for _ in range(12)]
    record = np.zeros(1, dtype=family.params_record_dtype())
    record["nx"], record["ny"], record["nz"] = 5, 6, 7
    record["n_elem"] = 210
    record["zrows"] = 3
    record["dtdx"] = np.float32(0.37)
    record["minus_dtdx"] = np.float32(-0.37)
    record["axis_coef"] = np.float32(1.48)
    record["inc_b"] = (np.float32(-0.0), np.float32(-0.37))
    params = torch.from_numpy(
        np.frombuffer(record.tobytes(), dtype=np.int32).copy()).to("mps")
    # Argument order is the signature's: 9 complex volumes, prefix, 2 imr rows,
    # 6 real curl coefficients, 6 complex constitutive volumes, 6 real ones, Params.
    function(out, *others[:8], *others[8:11], *reals[:6], *others[11:14],
             *[torch.zeros(8, dtype=torch.complex64, device="mps") for _ in range(3)],
             *reals[6:12], params, threads=1)
    torch.mps.synchronize()
    got = out.cpu().numpy()
    fields_ok = {
        "nx": float(got[0].real) == 5.0, "ny": float(got[0].imag) == 6.0,
        "nz": float(got[1].real) == 7.0, "n_elem": float(got[1].imag) == 210.0,
        "zrows": float(got[2].real) == 3.0,
        "dtdx": np.float32(got[2].imag) == np.float32(0.37),
        "minus_dtdx": np.float32(got[3].real) == np.float32(-0.37),
        "axis_coef": np.float32(got[3].imag) == np.float32(1.48),
        "inc_b_re_is_negative_zero": np.signbit(np.float32(got[4].real)),
        "inc_b_im": np.float32(got[4].imag) == np.float32(-0.37),
    }
    return {
        "passed": bool(separate_failed and over_failed and all(fields_ok.values())),
        "packed_bindings": family.PACKED_BINDINGS,
        "platform_ceiling": family.MAX_BUFFER_BINDINGS,
        "headroom": family.MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS,
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "separate_scalar_refused": separate_failed,
        "separate_scalar_error": separate_error,
        "over_ceiling_bindings": family.OVER_CEILING_BINDINGS,
        "over_ceiling_refused": over_failed,
        "over_ceiling_error": over_error,
        "params_fields_read_back": {k: bool(v) for k, v in fields_ok.items()},
        "params_itemsize": family.PARAMS_ITEMSIZE,
    }


def leg_needle_reachability(
        specialisations: Mapping[str, Tuple[int, int, Sequence[bool]]],
        expansion: str) -> Dict[str, Any]:
    """Every needle must be PRESENT in the source of the case it is scored on.

    THE DEAD-BRANCH CHECK, and it is the reason this leg exists at all. A mutation
    can rewrite REAL lines the scored grid never reaches, report UNCAUGHT, and
    measure nothing. On this family the guards are COMPILE-TIME: ``M_ONE`` and
    ``M_MANY`` emit different bodies and a metallic-z line is absent from a
    periodic-z source. So each needle is checked against the source its own case
    compiles, and the two cross-checks below are checked to be ABSENT where the
    guard says they must be — a needle that were present everywhere would mean the
    specialisation had stopped specialising.
    """
    edits = _edits(expansion)
    present: Dict[str, bool] = {}
    for name, (case, old, _new) in edits.items():
        bcz, arm, walls = specialisations[case]
        source = family.cylindrical_fused_magnetic_pair_source(
            bcz, arm, walls, expansion)
        present[name] = old in source

    # THE CROSS-CHECKS. Each asks whether a needle armed on one arm is ABSENT from
    # the other, which is what makes the split real rather than cosmetic.
    m_one_bcz, m_one_arm, m_one_walls = specialisations[M_ONE_CASE]
    m_many_bcz, m_many_arm, m_many_walls = specialisations[M_MANY_CASE]
    m_zero_bcz, m_zero_arm, m_zero_walls = specialisations[M_ZERO_CASE]
    m_one_source = family.cylindrical_fused_magnetic_pair_source(
        m_one_bcz, m_one_arm, m_one_walls, expansion)
    m_many_source = family.cylindrical_fused_magnetic_pair_source(
        m_many_bcz, m_many_arm, m_many_walls, expansion)
    m_zero_source = family.cylindrical_fused_magnetic_pair_source(
        m_zero_bcz, m_zero_arm, m_zero_walls, expansion)
    periodic_source = family.cylindrical_fused_magnetic_pair_source(
        templates.PERIODIC, m_one_arm, (False, False, False), expansion)
    zero = templates.COMPLEX_ZERO
    absent = {
        "axis_increment_absent_from_the_m_many_body":
            "curl0 = at_x ? -inc : curl0;" not in m_many_source,
        "near_axis_hold_absent_from_the_m_one_body":
            "bool near = (i < int(zrows));" not in m_one_source,
        # THE m = 0 BODY (2026-09-04): no coupling, no increment, no hold — and its
        # own clear is absent from the two |m| >= 1 bodies.
        "imr_coupling_absent_from_the_m_zero_body":
            "float2 q0 = c0[i];" not in m_zero_source
            and "c_mul(q0, c)" not in m_zero_source,
        "axis_increment_absent_from_the_m_zero_body":
            "curl0 = at_x ? -inc : curl0;" not in m_zero_source,
        "near_axis_hold_absent_from_the_m_zero_body":
            "bool near = (i < int(zrows));" not in m_zero_source,
        "m_zero_axis_clear_absent_from_the_m_one_body":
            f"v0 = at_x ? {zero} : v0;" not in m_one_source,
        "m_zero_axis_clear_absent_from_the_m_many_body":
            f"v0 = at_x ? {zero} : v0;" not in m_many_source,
        "wall_clear_absent_from_a_periodic_z_source":
            f"v2 = at_z ? {zero} : v2;" not in periodic_source,
        "r_wall_clear_never_emitted":
            f"v0 = at_x ? {zero} : v0;" not in m_one_source,
        "phi_wall_clear_never_emitted":
            f"v1 = at_y ? {zero} : v1;" not in m_one_source,
    }
    return {
        "passed": bool(all(present.values()) and all(absent.values())),
        "needles_present": present,
        "needles_absent_where_the_guard_says_they_must_be": absent,
        "needle_count": len(present),
        "note": "M_ZERO, M_ONE and M_MANY compile DIFFERENT BODIES and a metallic-z "
                "line does not exist on a periodic-z source; a needle armed on the "
                "wrong case would edit nothing and report UNCAUGHT while measuring "
                "nothing. The wall table on a Dcyl grid can only ever name z, which "
                "is why the r and phi clears are asserted ABSENT rather than armed. "
                "The m = 0 body's `v0 = at_x ? 0 : v0` is the AXIS rule, not a wall "
                "clear: it is emitted on every z termination.",
    }


def leg_transcription(bcz: int, arm: int, walls: Sequence[bool],
                      expansion: str) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes.

    The module claims its arithmetic is lifted rather than retyped. That is a
    property of the construction and therefore checkable, so it is checked:

    * the certified Dcyl ``step_B`` curl body must appear in the fused source
      VERBATIM on both sides of the spliced wall clear;
    * the constitutive half must differ from the certified ``update_H`` body in
      EXACTLY the three ``float2 srcN =`` lines. Any other differing line means a
      rename or a re-spelling reached the arithmetic.
    """
    from meep_gpu.metal_kernels import complex_fused_magnetic_pair as twin

    source = family.cylindrical_fused_magnetic_pair_source(bcz, arm, walls, expansion)
    curl = family.certified_cylindrical_curl_body(bcz, arm, expansion)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    certified = twin.certified_constitutive_body(expansion).splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    while spliced and (spliced[0].endswith("--") or spliced[0].lstrip().startswith("//")):
        spliced.pop(0)
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float2 src{t} = g{t}[ii];", f"    float2 src{t} = v{t};")
                for t in range(3)]
    lengths_match = len(certified) == len(spliced)
    helpers_present = templates.complex_helpers(expansion) in source
    return {
        "passed": bool(curl_head_present and curl_tail_present and lengths_match
                       and changed == expected and helpers_present),
        "curl_body_head_verbatim": curl_head_present,
        "curl_body_tail_verbatim": curl_tail_present,
        "complex_helpers_verbatim": helpers_present,
        "certified_constitutive_lines": len(certified),
        "spliced_constitutive_lines": len(spliced),
        "constitutive_lines_changed": [list(pair) for pair in changed],
        "constitutive_lines_expected": [list(pair) for pair in expected],
        "constitutive_half_lifted_from": "complex_fused_magnetic_pair."
                                         "certified_constitutive_body",
    }


def _volume_source(fields: Any, component: str) -> Any:
    """A REAL engine source on ``component``, so ``field_type`` is the engine's.

    A stub with a hand-set ``field_type`` would let this leg pass while the engine
    classified the same component the other way; ``sources.VolumeSource`` resolves
    the slot itself through ``_field_type_for`` (sources.py:222-229).
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0))


def leg_refusal() -> Dict[str, Any]:
    """The source seam, the m split and the geometry split, each measured by name.

    THE SOURCE CLAUSE COSTS THIS FAMILY ZERO CORPUS ROWS, which is exactly why it
    has to be tested: a clause that never fires on the measured corpus is a clause
    nothing has exercised, and a Dcyl script that drove a magnetic current would be
    admitted by both halves and stepped wrong through this weld.

    Six questions:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real MAGNETIC ``VolumeSource`` must be refused, naming the driver line;
    3. a real ELECTRIC ``VolumeSource`` must NOT be refused — on this family that is
       every one of the sixteen corpus rows;
    4. a REAL-STORAGE ``m = 0`` run must be refused BY NAME on STORAGE (the
       real-storage product's row), and a COMPLEX-STORAGE ``m = 0`` run must be
       ADMITTED and its plan must build — the ``M_ZERO`` arm, 2026-09-04, and the
       corpus row ``dipole_in_vacuum_cyl_off_axis.py``; before that round this leg
       pinned the m = 0 refusal, which is what left that row on the array path;
    5. a CARTESIAN grid must be refused;
    6. a REAL-STORAGE Dcyl run at |m| = 1 must be refused.
    """
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), residency)
    magnetic_named = [r for r in magnetic_coverage.reasons
                      if "is magnetic" in r and "driver.py:3283-3284" in r]
    magnetic_plan = family.plan_cylindrical_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), residency=residency)

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (electric,), residency)

    m0_fields, m0_pml = matrix.cylindrical(m=0, complex_storage=False)
    m0_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        m0_fields, m0_pml, (), Residency())
    m0_named = [r for r in m0_coverage.reasons
                if "force_complex_fields is not set" in r]

    m0c_fields, m0c_pml = matrix.cylindrical(m=0, complex_storage=True)
    m0c_residency = Residency()
    m0c_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        m0c_fields, m0c_pml, (), m0c_residency)
    m0c_plan = family.plan_cylindrical_fused_magnetic_pair(
        m0c_fields, m0c_pml, sources=(), residency=m0c_residency)
    # ...and the REAL family refuses the same run on the same clause, the other way.
    from meep_gpu.metal_kernels import cylindrical_real  # noqa: PLC0415

    m0c_real = cylindrical_real.cylindrical_real_curl_coverage(
        m0c_fields, m0c_pml, "step_B", Residency())
    m0c_real_named = [r for r in m0c_real.reasons if "force_complex_fields=True" in r]

    cart_fields, cart_pml = matrix.cart()
    cart_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        cart_fields, cart_pml, (), Residency())
    cart_named = [r for r in cart_coverage.reasons if "not cylindrical" in r]

    real_fields, real_pml = matrix.cylindrical(m=1, complex_storage=False)
    real_coverage = family.cylindrical_fused_magnetic_pair_coverage(
        real_fields, real_pml, (), Residency())
    real_named = [r for r in real_coverage.reasons
                  if "force_complex_fields" in r or "real" in r.lower()]

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and not magnetic_coverage.covered and magnetic_named
                       and magnetic_plan is None
                       and electric_coverage.covered
                       and not m0_coverage.covered and m0_named
                       and m0c_coverage.covered and m0c_plan is not None
                       and m0c_plan.m_arm == cylindrical_complex.M_ZERO
                       and not m0c_real.covered and m0c_real_named
                       and not cart_coverage.covered and cart_named
                       and not real_coverage.covered and real_named),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_refused": not magnetic_coverage.covered,
        "magnetic_plan_is_none": magnetic_plan is None,
        "magnetic_named": magnetic_named,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "m_zero_real_storage_refused": not m0_coverage.covered,
        "m_zero_real_storage_named": m0_named[:2],
        "m_zero_complex_storage_admitted": m0c_coverage.covered,
        "m_zero_complex_storage_reasons": list(m0c_coverage.reasons),
        "m_zero_complex_plan_built": m0c_plan is not None,
        "m_zero_complex_plan_arm": None if m0c_plan is None else m0c_plan.m_arm,
        "m_zero_complex_refused_by_the_real_family": not m0c_real.covered,
        "m_zero_complex_refused_by_the_real_family_named": m0c_real_named[:2],
        "cartesian_refused": not cart_coverage.covered,
        "cartesian_named": cart_named[:2],
        "real_storage_refused": not real_coverage.covered,
        "real_storage_named": real_named[:2],
        "note": "the source clause costs ZERO corpus rows on this family (all "
                "sixteen cylindrical rows declare electric sources only), which is "
                "why it is tested here rather than left to the corpus.",
    }


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

    cases = dict(CASES)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    # THE EXPANSION LEG RUNS FIRST AND EVERY LATER LEG READS ITS ARM. A leg that ran
    # before it would bind a guess, which is the one thing this family may not do.
    expansion_row = leg_expansion()
    if EXPANSION is None:
        raise SystemExit(
            "no complex-multiply expansion arm could be bound from a measured "
            f"artifact (${cylindrical_complex.PROBE_PATH_ENVIRONMENT}); which arm "
            f"the {cylindrical_complex.PROBE_BACKEND} reference takes on this "
            "family's SEVEN orientations is a platform fact this gate refuses to "
            "guess")

    specialisations: Dict[str, Tuple[int, int, Tuple[bool, ...]]] = {}
    for name in (M_ONE_CASE, M_MANY_CASE, M_ZERO_CASE, WALL_CASE):
        probe_fields, probe_pml = build(cases[name], 1)
        specialisations[name] = _specialisation(probe_fields, probe_pml)
    assert specialisations[M_ONE_CASE][1] == cylindrical_complex.M_ONE, (
        f"{M_ONE_CASE} does not compile the M_ONE arm; the axis-increment "
        f"mutations would be armed on lines the shipped kernel does not emit")
    assert specialisations[M_MANY_CASE][1] == cylindrical_complex.M_MANY, (
        f"{M_MANY_CASE} does not compile the M_MANY arm; the near-axis-hold "
        f"mutations would be armed on lines the shipped kernel does not emit")
    assert specialisations[M_ZERO_CASE][1] == cylindrical_complex.M_ZERO, (
        f"{M_ZERO_CASE} does not compile the M_ZERO arm; the m = 0 axis-clear "
        f"mutations would be armed on lines the shipped kernel does not emit")
    assert specialisations[WALL_CASE][2][2], (
        f"{WALL_CASE} does not wall z; the wall mutations would be armed on lines "
        f"the shipped kernel does not emit")

    mutants = shader_mutations(specialisations, EXPANSION)
    bcz, arm, walls = specialisations[M_ONE_CASE]
    neutral = {shaders.CONTRACT_OFF:
               compile_source(byte_neutral_source(bcz, arm, walls, EXPANSION)
                              ).cylindrical_fused_magnetic_pair_step}

    controls = ("m1_z_metallic", "m3_z_metallic", "m1_z_periodic", "m0_z_metallic")
    total = (1 + 1 + 1 + 1 + 1 + len(CASES) + len(CASES) + 1 + len(controls)
             + 1 + 1 + len(mutants) + len(HOST_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "expansion",
                     "label": "the_arm_is_bound_from_a_seven_pattern_artifact",
                     **expansion_row})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "thirty_one_of_thirty_one_with_zero_headroom",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "needle_reachability",
                     "label": "every_needle_is_in_the_body_it_is_scored_on",
                     **leg_needle_reachability(specialisations, EXPANSION)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_bytes",
                     **leg_transcription(bcz, arm, walls, EXPANSION)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "signed_zero",
                     "label": "the_pm_zero_class_at_the_launch_where_it_is_live",
                     **leg_signed_zero()})
        emit(handle, rows[-1])

        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 61000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        # THE SECOND VALUE CLASS, on every product row, and then the band.
        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps,
                   **run_case(keywords, 63000 + offset, args.steps,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "subnormal_band_is_refused_and_the_census_fires",
                     **leg_subnormal_ladder(dict(CASES)[WALL_CASE], 65000)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(cases[name], 67000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "the_source_seam_the_m_split_and_the_geometry_split",
                     **leg_refusal()})
        emit(handle, rows[-1])

        # THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT. Scored on NOT diverging.
        index += 1
        result = run_case(cases[M_ONE_CASE], 71000, args.steps, functions=neutral)
        row = {"index": index, "total": total, "leg": "byte_neutral_control",
               "label": "register_replaced_by_a_reload_of_the_same_word",
               "diverged": not result.get("bit_identical", False),
               "passed": bool(result.get("passed")),
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches")}
        rows.append(row)
        emit(handle, row)

        for name, entry in mutants.items():
            index += 1
            result = run_case(cases[entry["case"]], 72000 + index, args.steps,
                              functions=entry["functions"])
            diverged = not result.get("bit_identical", False)
            predicted = name in PREDICTED_NULL
            row = {"index": index, "total": total,
                   "leg": "predicted_null" if predicted else "mutation",
                   "label": name, "case": entry["case"],
                   "caught": diverged, "diverged": diverged,
                   "passed": bool((not diverged if predicted else diverged)
                                  and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            if predicted:
                row["reason"] = PREDICTED_NULL[name]
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            index += 1
            result = run_case(cases[M_ONE_CASE], 72000 + index, args.steps,
                              patch=patch)
            caught = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "case": M_ONE_CASE, "host_defect": True,
                   "caught": caught,
                   "passed": bool(caught and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above is
        # a harness that diverges on its own.
        index += 1
        result = run_case(cases[M_ONE_CASE], 72000 + index, args.steps)
        row = {"index": index, "total": total, "leg": "disarm",
               "label": "shipped_bytes_on_the_mutation_case",
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches"),
               "prefix_syncs": result.get("prefix_syncs"),
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
                   "shader_mutations": len(mutants) - len(PREDICTED_NULL),
                   "predicted_null_mutations": len(PREDICTED_NULL),
                   "host_mutations": len(HOST_MUTATIONS),
                   "byte_neutral_controls": 1},
        "predicted_null_reasons": PREDICTED_NULL,
        "expansion_arm": EXPANSION,
        "expansion_probe": PROBE_ARTIFACT,
        "environment": ENVIRONMENT,
        "mutation_cases": {"m_one": M_ONE_CASE, "m_many": M_MANY_CASE,
                           "m_zero": M_ZERO_CASE, "wall": WALL_CASE},
        "arm_swaps": {name: {"case": case, "compiled_arm": arm}
                      for name, (case, arm) in ARM_SWAPS.items()},
        "mutation_specialisations": {
            name: {"bcz": spec[0], "m_arm": spec[1], "zero_metal": list(spec[2])}
            for name, spec in specialisations.items()},
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "denominator": 186,
            "cylindrical_complex_curl_at_step_B": 16,
            "cylindrical_complex_constitutive_at_update_H": 16,
            "admit_both_halves": 16,
            "and_declare_no_magnetic_source": 16,
            "note": "the magnetic-source clause costs ZERO here: every one of the "
                    "sixteen cylindrical rows declares electric sources only "
                    "(fourteen ('D',), two ('D','D')). 16 is an UPPER BOUND because "
                    "the census records source FIELD TYPES but not whether a row's "
                    "grid would also satisfy the residency and wall-readability "
                    "clauses this predicate adds. The 194-row census of 2026-09-03 "
                    "(metal_coverage_2026-09-03_complete) carries one more Dcyl row, "
                    "examples:dipole_in_vacuum_cyl_off_axis.py (m = 0, complex64, "
                    "PML, electric sources only), which the M_ZERO arm of "
                    "2026-09-04 reaches; that row's census verdict is re-cut after "
                    "this gate, not asserted here.",
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "kernel_source_sha256": {
            label: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for label, source in family.enumerate_sources(EXPANSION).items()
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
