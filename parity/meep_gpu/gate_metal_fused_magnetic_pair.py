#!/usr/bin/env python3
"""Native MPS byte gate for the fused magnetic Metal kernel: ``step_B`` -> ``update_H``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans THREE driver
passes — ``step_B``, ``zero_metal_B`` and ``update_H`` — so a per-sub-step
comparison could not see it at all: the whole claim is about the SEAM between
them. The comparison is therefore per COMPLETE DRIVER STEP over a stated budget,
against an array-path oracle running the identical live pass list, with the first
divergent step reported rather than a final pass/fail.

THE SEVEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION, because the
   oracle is the array path this walk would otherwise take. So every case asserts
   the exact LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``,
   and every case asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** FOURTEEN shader mutations and TWO host-binding mutations are
   ARMED, and each must be CAUGHT. Leg ``disarm`` reruns the identical case with
   the shipped bytes and no host patch and requires zero, so a mutation reported as
   caught cannot be a harness that diverges anyway.
3. **A vacuous claim, and an unexercised policy.** On MPS the float32 subnormal
   flush is native and has no lever, so byte-identity is claimed subject to a
   CHECKED subnormal-free precondition; every case censuses the oracle's own state
   and reports it. That census reading zero on every row is a SILENCE unless the
   detector has been shown to fire, so leg ``value_class`` walks the oracle down
   the exponent range and requires it to fire in the band and stay clean above it,
   and it carries the SECOND value class — a +-0 lattice, byte-compared on every
   product row, scored on the sign floor. The band itself is a REFUSAL with the
   numbers behind it (``metal_value_classes``), not an absence.
4. **An unmeasured platform assumption.** This family is ONE dispatch for all three
   components because the five scalars are packed into a ``constant Params&``. Leg
   ``binding_ceiling`` compiles the 32-binding separate-scalar signature and
   requires the FAILURE, then compiles AND LAUNCHES the packed 28-binding one and
   requires every struct field to read back correctly.
5. **A transcription that drifted.** Both halves are LIFTED from the certified
   emitters rather than retyped, and leg ``transcription`` MEASURES that: the
   certified curl body must appear in the fused source verbatim on both sides of
   the spliced wall clear, and the constitutive half must differ from the certified
   H body in EXACTLY the three seam lines and nothing else.
6. **A seam that is not really a seam.** Leg ``byte_neutral_control`` replaces the
   three register reads with reloads of the very words the curl half just stored.
   That mutant must NOT diverge — it is the same value by construction — which is
   what makes "the fusion removes a round trip and changes no arithmetic" a
   measurement instead of a claim. It is the one armed edit in this file required
   to be UNCAUGHT, and it is scored on that.
7. **A refusal that is really an omission.** Leg ``refusal`` requires: an
   undeclared source list refused; a real ELECTRIC ``VolumeSource`` NOT refused (the
   polarity, which is the whole reason this seam is worth more than the D-side one);
   a folded grid refused naming both B-side fills; and a magnetic source whose
   deposit index the engine never resolved refused BY NAME.
8. **AN ADMISSION NOTHING MEASURES.** The magnetic source slot USED to cap this
   family on the measured corpus. It no longer does:
   ``fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR`` is True, so
   ``launch._install_fused_pair`` brackets the launch with
   ``deposit_repair.LeadingRepairPlan`` / ``TrailingRepairPlan`` and the predicate
   ADMITS every row whose magnetic source lands in this seam. An admitted row
   nobody measured is strictly worse than a refused one, so leg ``deposit`` walks
   complete driver steps WITH the injection performed in the seam and byte-compares
   them, on every case; and leg ``deposit_null_control`` builds the same shipped
   plan for an EMPTY source list -- the ``NoopPlan`` the flag's False branch would
   leave in the trailing slot -- injects anyway, and requires DIVERGENCE. A green
   ``deposit`` row without that control would be consistent with a deposit too small
   to see. Leg ``refusal`` additionally requires the admission to arrive WITH the
   wiring, and requires the flag not to travel by delegation to
   ``nonlinear_fused_magnetic_pair``, which holds no absorb row and would therefore
   run unbracketed.

LEGS
  0  binding_ceiling      32 separate bindings must FAIL; 27 pointers + one packed
                          Params& must COMPILE, LAUNCH and read back
  1  transcription        both halves must be the certified emitters' own bytes,
                          differing only in the three seam lines
  2  product              complete steps, per-step byte compare, launch counters,
                          movement, subnormal census
  2b value_class          the +-0 lattice on every product row, scored on the SIGN
                          floor; then the census ladder, which must FIRE in the
                          subnormal band and stay CLEAN above it. The band is a
                          named refusal, never a comparison
  2c synchronize         the MAGNETIC HALF-STEP -- flux_in_box / field_energy_in_box
                          run step's B/H half a second time and then undo it, and
                          this product owns every pass that half-step runs. Byte
                          compared after the half-step AND after the restore, with
                          two armed controls that must diverge: a window that also
                          advanced D (the update_H/step_D weld's shape, which the
                          restore cannot reach), and a backup missing f_w_H
  3  separate_control     the two ALREADY CERTIFIED products stepping the same seam
                          as separate dispatches with the wall clear on the host
                          between them: three-way byte agreement plus the dispatch
                          and host-pass counts the fusion removes
  3b deposit              complete steps with a magnetic source INSIDE the seam:
                          save, launch, inject, host wall clear, repair -- byte
                          compared per step, with the repaired point count nonzero
  3c deposit_null_control the same launch with the trailing repair replaced by the
                          NoopPlan: scored on DIVERGING
  4  refusal              the source seam, the fold and the wiring, by name
  5  byte_neutral_control the register-vs-reload edit must NOT diverge
  6  mutation             sixteen armed defects, each of which MUST diverge
  7  disarm               the same harness, shipped bytes, must not diverge

Rule 7: one flushed line per case, every row appended and fsynced as it lands.
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

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    fused_magnetic_pair as family, launch as metal_launch,
    nonlinear_fused_magnetic_pair as nonlinear, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the two D-side fused pairs' and
#: the whole-step arbiter's, and for their reason: the classes this gate exists
#: for COMPOUND. ``fu_B`` and ``f_w_H`` are state carried between steps, and a
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
#: corrupted B would reach E through ``step_D`` on the very next step, and a
#: comparison blind to that would be reporting on half the engine.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> ``metal_composition_matrix.cart`` keywords.
#:
#: WHAT THE ROWS CARRY BETWEEN THEM:
#:
#:   WALLS          ``zero_metal_B`` emits a line per metallic axis, and for B the
#:                  table is the DIAGONAL — Bx on x, By on y, Bz on z. The rows
#:                  span no wall, one wall, and all three, so the wall mutations
#:                  are structurally present where they are armed and the
#:                  all-periodic row proves the family is not wall-dependent;
#:   GHOST RULE     a metallic axis serves an exact 0.0 past the wall and a
#:                  periodic axis wraps, so the boundary triple selects a different
#:                  compiled kernel;
#:   DIMENSIONALITY the 2-D row is the corpus's most common shape and the 3-D rows
#:                  are where a wall plane is hundreds of words rather than tens.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("all_periodic_3d", dict()),
    ("wall_x_3d", dict(boundaries={"x": "metallic"})),
    ("wall_z_3d", dict(boundaries={"z": "metallic"})),
    ("wall_xz_3d", dict(boundaries={"x": "metallic", "z": "metallic"})),
    ("wall_xyz_3d",
     dict(boundaries={"x": "metallic", "y": "metallic", "z": "metallic"})),
    ("wall_xy_thin_pml_3d",
     dict(boundaries={"x": "metallic", "y": "metallic"}, pml=1)),
)

#: The case the mutations are armed on. ALL THREE axes walled, so every line the
#: shipped kernel can emit is present: all three wall-clear rows, the full
#: ownership mask and the metallic ghost rule on every axis. A mutation leg run on
#: an all-periodic row would report the wall defects as uncaught for the
#: uninteresting reason that no line is emitted there.
MUTATION_CASE = "wall_xyz_3d"

#: The row leg ``refusal`` builds its source and fold questions on.
REFUSAL_CASE: Dict[str, Any] = dict(boundaries={"x": "metallic"})


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
          value_class: str = values.UNIFORM, scale: float = 1.0
          ) -> Tuple[Any, Any]:
    """One seeded engine. Called twice (or three times) per case, identically.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the two halves' own gates use, so the fixture this certifies is
    the fixture they were certified on. Only the field state is reseeded here, and
    identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.

    ``value_class`` selects WHICH float32 values the state is seeded with, and it
    is the axis this gate did not carry until now. ``uniform`` is the physical
    band. ``pm_zero_lattice`` is a checkerboard of +0.0 and -0.0 over every stored
    volume, with the material left NORMAL so the sign propagates through the
    epsilon multiply — the second class, byte-compared, and scored on the SIGN
    floor rather than on movement. ``scale`` walks the uniform class down the
    exponent range and is used by the census ladder alone; NO scaled row is byte
    compared (:func:`leg_subnormal_ladder` says why, with the numbers).
    """
    fields, pml = matrix.cart(**dict(keywords))
    if value_class == values.PM_ZERO_LATTICE:
        lattice = values.pm_zero_lattice(fields.grid.shape)
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
            array[...] = values.scaled_normal(rng, fields.grid.shape, 0.37, scale)
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
    plan = family.plan_metal_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": "the fused magnetic pair was refused",
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
    # no-op. The +-0 lattice cannot answer to that floor honestly — a stored
    # volume the seam only ever writes zeros into holds the same word before and
    # after and has still been compared — so it answers to the SIGN floor instead:
    # the REFERENCE OUTPUT must carry BOTH bit patterns, which is what makes a
    # uint32 compare of a zero state discriminating rather than decorative. It is
    # not exempt from a floor, it has a different one.
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
        "boundary_codes": list(plan.codes),
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
    every complete step, and the DISPATCH COUNTS and the surviving in-seam host
    passes are recorded on both sides — which is the only place the difference
    between the compositions shows up at all, since a correct fusion is byte-neutral
    by construction.

    The separate side is not a strawman: those two plans are exactly what
    ``plan_step`` composes today for this configuration.
    """
    reference, reference_pml = build(keywords, seed)
    separate, separate_pml = build(keywords, seed)
    fused, fused_pml = build(keywords, seed)

    separate_residency = Residency()
    curl = metal_launch.plan_pml_curl(separate, separate_pml, "step_B",
                                      separate_residency)
    magnetic = metal_launch.plan_constitutive(separate, separate_pml, "H",
                                              separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_fused_magnetic_pair(
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

    # The host passes each composition leaves on the array path INSIDE the seam.
    # The fused side carries the wall clear in the kernel; the separate side does
    # not, so on every walled run it pays a host round trip the fused side does not.
    seam = ("zero_metal_B",)
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and magnetic.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + magnetic.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + magnetic.launches,
        "fused_launches": plan.launches,
        "seam_host_passes_for_separate": [n for n in live if n in seam],
        "seam_host_passes_for_fused": [],
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The magnetic half-step: the second consult site, never driven on this backend
# ---------------------------------------------------------------------------

#: ``stepping`` function name -> the pass name this gate and the composer use.
#: Written down so :func:`sync_order` can read the driver's own half-step and come
#: back in THIS file's vocabulary, rather than either side guessing the other's.
_SYNC_PASS_NAMES: Dict[str, str] = {
    "step_B": "step_B",
    "fill_symmetry_bc_B": "fill_B",
    "zero_metal_B": "zero_metal_B",
    "fill_folded_far_ghosts_B": "fill_folded_far_ghosts_B",
    "update_H": "update_H",
}


def sync_order() -> Tuple[str, ...]:
    """The passes ``FdtdDriver.synchronize_magnetic_fields`` runs, READ OFF ITS SOURCE.

    The same discipline the product legs take for a complete step: a gate that
    carried its own model of the half-step would certify the kernel against the
    model. Sorted by source position, so the ORDER is the driver's too.
    """
    import ast  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(FdtdDriver.synchronize_magnetic_fields)))
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            mapped = _SYNC_PASS_NAMES.get(node.func.id)
            if mapped is not None:
                seen.append(((node.lineno, node.col_offset), mapped))
    return tuple(name for _, name in sorted(seen))


def _sync_backup(fields: Any) -> Dict[str, np.ndarray]:
    """The driver's own backup list, taken from the driver's own class attributes."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    names = FdtdDriver._SYNC_FIELDS + FdtdDriver._SYNC_AUXILIARY
    return {name: np.array(getattr(fields, name), copy=True)
            for name in names if getattr(fields, name, None) is not None}


def _sync_average(fields: Any, backup: Mapping[str, np.ndarray]) -> None:
    """``average_with_backup`` — only ``_SYNC_FIELDS``, exactly as the driver does it."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    for name in FdtdDriver._SYNC_FIELDS:
        array = getattr(fields, name, None)
        if array is not None and name in backup:
            array *= 0.5
            array += 0.5 * backup[name]


def _sync_restore(fields: Any, backup: Mapping[str, np.ndarray]) -> None:
    for name, saved in backup.items():
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = saved


def leg_synchronize(keywords: Mapping[str, Any], seed: int, steps: int
                    ) -> Dict[str, Any]:
    """THE SECOND CONSULT SITE. Nothing on this backend had ever driven it.

    ``flux_in_box`` and ``field_energy_in_box`` run ``step``'s magnetic half a
    second time and then UNDO it, and this product owns EVERY pass that half-step
    runs — so the half-step is one launch of this kernel and no host pass at all.
    That is a composition no leg here had ever exercised: every other leg drives
    ``step``.

    The claim is the file's usual one, on this site: after the half-step, and again
    after the restore, the fused engine is byte-identical to the array engine over
    every stored volume, as uint32 words.

    TWO ARMED CONTROLS, because a comparison that has not been shown to fail is
    worth nothing here either:

    * ``advanced_D_inside_the_window`` — the half-step is run with ``step_D``
      appended, which is what an ``update_H``/``step_D`` weld installed at
      ``update_H`` would do at this site. ``D`` and ``fu_D`` are in NEITHER of the
      driver's backup lists, so the restore cannot reach them and the run must
      diverge. This is the hazard that ``fastpath.SYNC_PASS_OWNERS`` exists for,
      executed on the device rather than argued;
    * ``restore_without_f_w_H`` — the constitutive workspace dropped from the
      backup, which must also diverge, and which is what says the comparison sees
      the auxiliaries and not only the primaries.
    """
    order = sync_order()
    reference, reference_pml = build(keywords, seed)
    actual, actual_pml = build(keywords, seed)
    residency = Residency()
    plan = family.plan_metal_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency)
    if plan is None:
        return {"passed": False, "reason": "the fused magnetic pair was refused"}
    owned = tuple(plan.replaces_sub_steps)
    # WHAT THE PRODUCT OWNS OF THE HALF-STEP IS A MEASUREMENT, NOT A PRECONDITION.
    # This family declares ``step_B``, ``zero_metal_B`` and ``update_H`` — three of
    # the five passes the half-step runs — so the two mirror fills stay on the host
    # INSIDE the window, each bracketed by the residency mirror exactly as
    # :func:`metal_step` brackets them in a complete step. A leg that demanded the
    # product own all five would refuse the composition that actually ships.
    carried = tuple(name for name in order if name in owned)
    on_the_host = tuple(name for name in order if name not in owned)
    live = live_passes(actual, actual_pml)
    residency.sync_in()
    for _ in range(steps):
        array_step(reference, reference_pml, live)
        metal_step(actual, actual_pml, {family.SLOT: plan}, owned, residency, live)
    residency.sync_out()
    stepped_drift = compare(reference, actual)

    launches_before = plan.launches
    reference_backup = _sync_backup(reference)
    actual_backup = _sync_backup(actual)
    for name in order:
        ARRAY_PATH[name](reference, reference_pml)
    _sync_average(reference, reference_backup)
    # THE HALF-STEP THROUGH THE SAME WALK A COMPLETE STEP TAKES: the plan where it
    # owns the pass, the array path with the residency bracket where it does not.
    metal_step(actual, actual_pml, {family.SLOT: plan}, owned, residency, order)
    residency.sync_out()
    _sync_average(actual, actual_backup)
    synchronized = compare(reference, actual)
    # The half-step has to have DONE something, or two no-ops agree.
    moved = sum(differing(reference_backup[name], getattr(reference, name))
                for name in reference_backup)

    _sync_restore(reference, reference_backup)
    _sync_restore(actual, actual_backup)
    restored = compare(reference, actual)
    round_trip = {name: differing(actual_backup[name], getattr(actual, name))
                  for name in actual_backup}
    round_trip = {name: count for name, count in round_trip.items() if count}

    # --- the armed controls --------------------------------------------------
    controls: Dict[str, Any] = {}
    spoiled, spoiled_pml = build(keywords, seed)
    for _ in range(steps):
        array_step(spoiled, spoiled_pml, live)
    backup = _sync_backup(spoiled)
    for name in order:
        ARRAY_PATH[name](spoiled, spoiled_pml)
    ARRAY_PATH["step_D"](spoiled, spoiled_pml)   # the H->D weld's second half
    _sync_average(spoiled, backup)
    _sync_restore(spoiled, backup)
    controls["advanced_D_inside_the_window"] = compare(reference, spoiled)

    thin, thin_pml = build(keywords, seed)
    for _ in range(steps):
        array_step(thin, thin_pml, live)
    partial = {name: value for name, value in _sync_backup(thin).items()
               if not name.startswith("f_w_H")}
    for name in order:
        ARRAY_PATH[name](thin, thin_pml)
    _sync_average(thin, partial)
    _sync_restore(thin, partial)
    controls["restore_without_f_w_H"] = compare(reference, thin)

    residency.sync_in()
    passed = (bool(carried) and not stepped_drift and not synchronized and not restored
              and not round_trip and moved > 0
              and plan.launches == launches_before + plan.launches_per_run
              and all(controls.values()))
    return {
        "passed": bool(passed),
        "steps_before_the_half_step": steps,
        "sync_order": list(order),
        "passes_the_product_carried_in_launch": list(carried),
        "passes_left_on_the_host_inside_the_window": list(on_the_host),
        "replaces": list(owned),
        "drift_after_the_steps": stepped_drift,
        "differing_after_the_half_step": synchronized,
        "differing_after_the_restore": restored,
        "arrays_the_restore_did_not_return": round_trip,
        "words_the_half_step_moved": int(moved),
        "launches_for_the_half_step": int(plan.launches - launches_before),
        "launches_per_run": plan.launches_per_run,
        "controls_that_must_diverge": {
            name: {"arrays": len(difference),
                   "words": int(sum(difference.values())),
                   "diverged": bool(difference)}
            for name, difference in controls.items()},
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
    """The (codes, walls) pair the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds

    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(fields.grid, pml))
    return codes, zero_metal_axes(fields.grid)


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure mode a mutation leg cannot see
    from its own result.
    """
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


def shader_mutations(codes: Sequence[int],
                     walls: Sequence[bool]) -> Dict[str, Dict[str, Any]]:
    """Fourteen source defects, each a plausible transcription slip.

    Every needle is anchored on text that only the mutated line carries, and every
    one is verified present before it is applied — an absent needle raises here
    rather than silently arming nothing.

    THE MUTATION CASE'S GEOMETRY, which is what makes each of these reachable: all
    three axes metallic, so all three wall-clear rows and the full ownership mask
    are emitted and the metallic ghost rule is live on every axis.
    """
    base = family.fused_magnetic_pair_source(codes, walls)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float src0 = v0;", "float src0 = curl0;"),
        # THE SEAM, WRONG COMPONENT. Bx's constitutive source is v0, not v1.
        "seam_takes_the_wrong_component":
            needle(base, "float src1 = v1;", "float src1 = v0;"),
        # THE WALL CLEAR, DROPPED on the x wall.
        "zero_metal_dropped":
            needle(base, "    v0 = at_x ? 0.0f : v0;\n", ""),
        # THE WALL TABLE IS THE DIAGONAL FOR B, the off-diagonal for D. Reusing the
        # D-side table clears Bx on the y wall instead of the x wall, which is the
        # single most likely slip in porting this family from its D-side sibling.
        "zero_metal_uses_the_d_side_table":
            needle(base, "    v0 = at_x ? 0.0f : v0;",
                   "    v0 = at_y ? 0.0f : v0;"),
        # THE AUXILIARY IS NOT CLEARED. `zero_metal_B` passes B_COMPONENTS only
        # (stepping.py:2250); masking fu_B as well is a plausible over-carry.
        "zero_metal_also_masks_the_auxiliary":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u0[ii] = at_x ? 0.0f : n0; u1[ii] = n1; u2[ii] = n2;"),
        # THE OWNERSHIP MASK, dropped. A different pass from the wall clear, on the
        # same component and the same axis — which is exactly why both are armed.
        "ownership_mask_dropped":
            needle(base, "    curl0 = at_x ? 0.0f : curl0;\n", ""),
        # STEP_B IS FORWARD. step_D's negated strides are a different product.
        "curl_direction_reversed":
            needle(base, "int si = i + 1, sj = j + 1, sk = k + 1;",
                   "int si = i - 1, sj = j - 1, sk = k - 1;"),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        # THE RECURRENCE PAIRS are vec.hpp's cycle_direction: target 0 takes (y, z).
        "recurrence_axis_pair_swapped":
            needle(base, "float n0 = ((p0 * km_y) - curl0) * si_y;",
                   "float n0 = ((p0 * km_z) - curl0) * si_z;"),
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "flux_store_dropped":
            needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f1[ii] = v1; f2[ii] = v2;"),
        "fw_store_dropped":
            needle(base, "    w0[ii] = src0;", "    // stale f_w_Hx"),
        # The dsigw index is the component's OWN axis (stepping.py:227-228), not
        # the curl's cycle. Moving it one axis over is a silent half-cell error.
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = km0[i];",
                   "float kp_0 = kp0[j], km_0 = km0[j];"),
        "constitutive_accumulations_reversed":
            needle(base, "    a0 = a0 + kp_0 * src0;\n"
                         "    a0 = a0 - km_0 * prev0;",
                   "    a0 = a0 - km_0 * src0;\n"
                   "    a0 = a0 + kp_0 * prev0;"),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        mutants[name] = {shaders.CONTRACT_OFF:
                         compile_source(source).fused_magnetic_pair_step}
    return mutants


def byte_neutral_source(codes: Sequence[int], walls: Sequence[bool]) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and
    ``v0`` hold the same float32 word, and a float32 stored to a ``device float*``
    and reloaded is bit-identical to the register. This edit is therefore the
    fusion's central claim written as a program, and leg
    ``byte_neutral_control`` requires it NOT to diverge.
    """
    source = family.fused_magnetic_pair_source(codes, walls)
    for target in range(3):
        source = needle(source, f"float src{target} = v{target};",
                        f"float src{target} = f{target}[ii];")
    return source


#: Where each binding group starts in the plan's argument tuple. Spelled once, so
#: the two host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(15, 21))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(21, 27))


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

    ``step_B`` reads HALF-INTEGER positions and ``step_D`` integer ones
    (launch.py:109-124, ``SUB_STEPS['step_B']['suffix'] == '_h'``). This is the
    OPPOSITE polarity to the D/E fused pair's equivalent mutation, and the kernel
    cannot tell: it is a half-cell error in the absorber profile, not a crash.
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


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """32 separate bindings must FAIL; the packed 28 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. A signature that compiled but bound the struct wrongly would
    produce a smooth, plausible, wrong field, so the launch half is measured here
    and not inherited from another family's docstring.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS, "ceiling": 31}
    try:
        compile_source(family.refuted_separate_scalar_source())
        row.update(separate_compiled=True, separate_error="", passed=False,
                   note="the 32-binding signature COMPILED; this family's shape "
                        "rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(27))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };", "",
        "kernel void packed_probe(", pointers,
        "    constant Params&    prm     [[buffer(27)]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;", "    b5[idx] = b26[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(27)]
    buffers[26] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx = (3, 4, 5), 0.125
    # The real packer, not a hand-built record: the field the kernel reads is then
    # the field the plan builder writes. n_elem is the product of the shape (60),
    # deliberately not the eight-element probe buffers, so the guard is exercised.
    params = family._params_tensor(shape, dtdx, "mps")
    function(*buffers, params)
    torch.mps.synchronize()
    read = [buffers[index].cpu().numpy() for index in range(6)]
    expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx)), 8.0)
    fields_ok = all(bool(np.all(read[index] == expected[index]))
                    for index in range(6))
    row.update(packed_launched=True,
               packed_fields_read_back=[float(value[0]) for value in read],
               packed_fields_expected=[float(value) for value in expected],
               packed_fields_correct=fields_ok,
               passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok))
    return row


def leg_subnormal_ladder(keywords: Mapping[str, Any], seed: int, budget: int = 4,
                         ) -> Dict[str, Any]:
    """THE EMPTY CENSUS, TURNED INTO A MEASUREMENT.

    Every product row in this file reports ``reference_subnormals: 0`` and
    ``subnormal_free: true`` under policy ``flush``. Read alone that is a SILENCE:
    it cannot distinguish "the flush policy was exercised and changed nothing"
    from "the band was never entered, so the policy was never asked anything". A
    verifier named exactly that, and this leg is the answer — not a second byte
    comparison, which MPS cannot give, but the precondition FIRING.

    WHAT RUNS. The oracle alone, seeded at each rung of
    :data:`metal_value_classes.PRECONDITION_SCALES`, stepped through the same live
    pass list, with the same census the product rows report taken after every
    step. Nothing is compared against the device here and the artifact says so:
    ``banded_rows_were_byte_compared`` is False.

    WHY THE BANDED RUNGS ARE NOT COMPARED, measured rather than asserted. On this
    host, on ``wall_xyz_3d``, the shipped kernel against the array path over four
    complete steps: identical at 1e0, 1e-25 and 1e-30 with an empty census, and at
    1e-34 the census fires with 27 words and the bytes diverge — 1036 words at
    step 1 growing to 30688 by step 4. Three decades of headroom, then a CLIFF.
    MPS flushes natively with no lever and NumPy keeps, so a subnormal in the
    reference state IS a divergence and no kernel can be right about it. The claim
    this family makes is therefore byte identity under a CHECKED subnormal-free
    precondition, and this leg is the check being shown to work.

    THE FAILURE MODES IT SCORES SEPARATELY. A census that never fired would be a
    detector never demonstrated to work; a census that fired on every rung would
    be one that cannot discriminate, and would condemn the physical band too. Both
    fail here, and ``detector_is_discriminating`` says which.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in values.PRECONDITION_SCALES:
        reference, reference_pml = build(keywords, seed, values.UNIFORM, scale)
        live = live_passes(reference, reference_pml)
        per_step: List[int] = []
        for _ in range(budget):
            array_step(reference, reference_pml, live)
            per_step.append(int(sum(subnormal.census(value)
                                    for value in state_of(reference).values())))
        fired = [index for index, count in enumerate(per_step) if count]
        agrees = (expect_clean is None
                  or (expect_clean and not fired)
                  or (not expect_clean and bool(fired)))
        rows.append({"scale": label, "factor": scale,
                     "subnormal_words_per_step": per_step,
                     "census_fired": bool(fired),
                     "first_step": fired[0] + 1 if fired else None,
                     "expected_clean": expect_clean, "agrees": agrees})
        log(f"    ladder {label} factor={scale:g} fired={bool(fired)} "
            f"census={per_step} expected_clean={expect_clean}")
    return values.precondition_verdict(rows)


#: The shader defects a +-0 state CANNOT see, and why each one is invisible there.
#: Every array in that class holds a zero at every step, so a defect is observable
#: only if it changes a SIGN or moves a zero to a cell that held the other sign. A
#: defect that only ever scales, reorders or re-indexes a MAGNITUDE multiplies zero
#: by a different number and gets zero. These four are declared here so the leg
#: below can require the miss set to be EXACTLY them: a fifth miss would mean the
#: class went blind to something it should see, and a miss disappearing would mean
#: this table drifted from the kernel.
LATTICE_BLIND_MUTATIONS: Tuple[str, ...] = (
    # Masking cell 0 to zero, on a state that is already zero there.
    "ownership_mask_dropped",
    # A different neighbour's zero is still a zero; the sign rides on the
    # coefficients, which this edit does not touch.
    "curl_direction_reversed",
    # float32 reassociation of a sum of zeros: (0+0)+(0+0) == 0+0+0+0 exactly.
    "curl_parens_flattened",
    # A different absorber entry multiplying zero.
    "constitutive_coefficient_index_moved",
)


def leg_value_class_mutation_score(keywords: Mapping[str, Any], seed: int,
                                   steps: int,
                                   mutants: Mapping[str, Any]) -> Dict[str, Any]:
    """WHAT THE SECOND VALUE CLASS IS WORTH, scored rather than asserted.

    A value class added to answer a verifier and never shown to discriminate is
    decoration. So the SAME armed shader defects the mutation leg scores on the
    physical band are re-run under the +-0 lattice, and the outcome is required to
    be the one :data:`LATTICE_BLIND_MUTATIONS` predicts: caught everywhere except
    the four defects that can only change a MAGNITUDE, which a state of zeros
    multiplies away.

    Measured on this host, ``wall_xyz_3d``, four steps: 10 of 14 caught. The class
    is therefore a real second comparison — it independently executes the seam, the
    wall clear, both stores and the recurrence — and its blind spots are named
    rather than discovered later.
    """
    rows: List[Dict[str, Any]] = []
    for offset, (name, functions) in enumerate(sorted(mutants.items())):
        result = run_case(keywords, seed + offset, steps, functions=functions,
                          value_class=values.PM_ZERO_LATTICE)
        caught = not result.get("bit_identical", False)
        rows.append({"label": name, "caught": caught,
                     "differing_words": result.get("differing_words"),
                     "launches": result.get("launches"),
                     "predicted_blind": name in LATTICE_BLIND_MUTATIONS})
        log(f"    lattice mutation {name} caught={caught} "
            f"diff={result.get('differing_words')}")
    missed = sorted(row["label"] for row in rows if not row["caught"])
    launched = all(row["launches"] for row in rows)
    return {
        "passed": bool(rows and launched
                       and missed == sorted(LATTICE_BLIND_MUTATIONS)),
        "value_class": values.PM_ZERO_LATTICE,
        "caught": sum(1 for row in rows if row["caught"]),
        "armed": len(rows),
        "missed": missed,
        "predicted_blind": sorted(LATTICE_BLIND_MUTATIONS),
        "every_mutant_launched": launched,
        "rows": rows,
    }


def leg_transcription(codes: Sequence[int], walls: Sequence[bool]) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes.

    The module claims its arithmetic is lifted rather than retyped. That is a
    property of the construction and therefore checkable, so it is checked:

    * the certified ``step_B`` curl body must appear in the fused source VERBATIM on
      both sides of the spliced wall clear;
    * the constitutive half must differ from the certified ``update_H`` body in
      EXACTLY the three ``float srcN =`` lines. Any other differing line means a
      rename or a re-spelling reached the arithmetic.
    """
    source = family.fused_magnetic_pair_source(codes, walls)
    curl = family.certified_curl_body(codes)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    certified = family.certified_constitutive_body().splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    # The spliced half is preceded by the marker's own line remainder; drop the
    # leading fragment so the two lists start at the same statement.
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float src{t} = g{t}[ii];", f"    float src{t} = v{t};")
                for t in range(3)]
    lengths_match = len(certified) == len(spliced)
    return {
        "passed": bool(curl_head_present and curl_tail_present and lengths_match
                       and changed == expected),
        "curl_body_head_verbatim": curl_head_present,
        "curl_body_tail_verbatim": curl_tail_present,
        "certified_constitutive_lines": len(certified),
        "spliced_constitutive_lines": len(spliced),
        "constitutive_lines_changed": [list(pair) for pair in changed],
        "constitutive_lines_expected": [list(pair) for pair in expected],
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


# ---------------------------------------------------------------------------
# The deposit: an in-seam magnetic source CARRIED across the fused launch
# ---------------------------------------------------------------------------
#
# THIS IS THE LEG THE FLAG IS WORTH. `fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR` is
# now True, so the predicate ADMITS rows it refused before -- every row whose magnetic
# source lands between `step_B` and `update_H` (driver.py:3283-3284). An admitted row
# that nothing measures is strictly worse than a refused one: the refusal at least
# computed the right numbers on the array path.
#
# WHY THIS WALK IS NOT `metal_step`, and the difference is the whole mechanism:
#
#   * the INJECTION is performed, between the fused launch and the repair, because
#     that is where the driver performs it;
#   * `zero_metal_B` runs ON THE HOST after the injection even though the kernel
#     carries it inline, because driver.py:3295 runs it UNCONDITIONALLY rather than
#     behind a `dispatch` consult. That is what lets the repair read a final B and
#     own no wall handling of its own (`deposit_repair`, "WHERE IT BELONGS IN THE
#     STEP"). A deposit that lands on a walled cell is recomputed from the zero that
#     clear left, which is what the array path computes too;
#   * the two slots are run as the two separate consults they are: `LeadingRepairPlan`
#     at `step_B` (save, then launch) and `TrailingRepairPlan` at `update_H` (apply).
#
# AND THE NULL CONTROL IS THE POINT. `repair=False` builds the SHIPPED plan for an
# empty source list -- which is the bare pair plus the `NoopPlan` the False branch of
# this flag would leave in the trailing slot -- and then injects anyway. That is
# exactly "a product that passes True without those wrappers", constructed out of
# shipped code rather than a hand-written mutant, and it MUST diverge. Without it a
# green `repair=True` row would be consistent with a deposit too small to see.


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    """``driver.py:3282``'s withdraw pass, on both sides identically."""
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


#: The extent of the EXTENDED deposit row, in the composition matrix's own cell units.
#: A point source deposits into four cells; this one deposits into the whole grid, so
#: the repair's per-point loop is exercised at thousands of points rather than at four
#: and a repair that handled only its first point would show.
EXTENDED_SIZE: Tuple[float, float, float] = (2.0, 2.1, 1.2)


def _magnetic_source(fields: Any, component: str = "Hy",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Any:
    """One REAL engine magnetic source, built against THIS engine's own grid.

    A source shared between the two engines would inject through one grid's index
    arrays into the other's arrays; they are equal here, and relying on that is how a
    fixture starts certifying its own coincidences.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=size,
                        envelope=ContinuousEnvelope(frequency=1.0))


def deposits_on_a_cleared_cell(fields: Any, source: Any) -> int:
    """How many of this source's deposit points ``zero_metal_B`` also zeroes.

    THE ONE ASYMMETRY BETWEEN THE TWO PATHS, MEASURED RATHER THAN ARGUED. The fused
    kernel carries ``zero_metal_B`` INLINE and therefore clears before the injection;
    the driver clears AFTER it (driver.py:3295). The two can only differ where a
    deposit lands on a cell the clear zeroes, and both paths would then read the same
    final zero -- but "would" is a claim, so the count is measured per row and
    reported.

    MEASURED ZERO ON EVERY ROW HERE, and structurally so: on an all-walled grid a
    whole-cell ``VolumeSource`` on Hx, Hy or Hz resolves 4788 / 4800 / 4620 points and
    NONE of them is on a plane ``zero_metal_B`` writes -- the cleared plane is the
    wall itself, which is not an owned cell for a component half-integer in that
    direction. Reported anyway: a zero this leg never looked at would be a silence.
    """
    index = deposit_repair._deposit_index(source)
    if index is None:
        return 0
    probe = {name: np.array(getattr(fields, name), copy=True)
             for name in ("Bx", "By", "Bz")}
    try:
        for name in ("Bx", "By", "Bz"):
            getattr(fields, name)[...] = 1.0
        stepping.zero_metal_B(fields)
        component = "B" + str(source.component)[-1]
        return int(np.count_nonzero(getattr(fields, component)[index] == 0.0))
    finally:
        for name, saved in probe.items():
            getattr(fields, name)[...] = saved


def run_deposit_case(keywords: Mapping[str, Any], seed: int, steps: int,
                     repair: bool = True, component: str = "Hy",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     ) -> Dict[str, Any]:
    """Complete driver steps with a magnetic source IN the seam, byte compared."""
    reference, reference_pml = build(keywords, seed)
    actual, actual_pml = build(keywords, seed)
    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    reference_sources = (_magnetic_source(reference, component, size),)
    actual_sources = (_magnetic_source(actual, component, size),)
    on_cleared = deposits_on_a_cleared_cell(actual, actual_sources[0])
    in_seam = deposit_repair.in_seam_sources(actual_sources, "B")
    assert len(in_seam) == 1, (
        "the source this leg builds is not in the B seam; the walk would inject "
        "nothing and compare a no-op with a no-op")
    assert deposit_repair._deposit_index(actual_sources[0]) is not None, (
        "the source publishes no deposit index; there would be nothing to repair")

    residency = Residency()
    # THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    # `plan_metal_fused_magnetic_pair` directly would test the kernel and skip the
    # wiring, and the wiring is what this leg exists for.
    composed = metal_launch.plan_step(actual, actual_pml, residency=residency,
                                      sources=(actual_sources if repair else ()),
                                      fuse=True)
    leading = composed.plans.get("step_B")
    trailing = composed.plans.get("update_H")
    installed = (type(leading).__name__, type(trailing).__name__)
    # WHAT EACH SLOT MUST HOLD. The bracketed shape is named exactly; the null
    # control's leading slot is the plan builder's own class and is asserted only to
    # NOT be the repair wrapper, because naming it would pin this leg to a class name
    # the builder is free to change.
    if repair:
        installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
    else:
        installed_ok = (installed[0] != "LeadingRepairPlan"
                        and installed[1] == "NoopPlan")
    if leading is None or trailing is None:
        return {"passed": False, "reason": "the composer did not fuse the seam",
                "refusals": [r for key, value in composed.reasons.items()
                             if key.startswith("fused_pair") for r in value]}
    inner = getattr(leading, "absorbed_by", leading)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    assert live.index("step_B") == 0 and "update_H" in live, live
    before = frozen(actual)
    dt = float(actual.grid.dt)

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
            if name == "step_B":
                for source in reference_sources:
                    source.inject(reference, when)

        _withdraw(actual_sources, actual)
        for name in live:
            if name == "step_B":
                residency.sync_in()
                leading.run()
                residency.sync_out()
                for source in actual_sources:
                    source.inject(actual, when)
            elif name == "update_H":
                trailing.run()
            else:
                ARRAY_PATH[name](actual, actual_pml)

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
    repairs = int(getattr(leading, "repairs", 0))
    launches_ok = (inner.runs == len(per_step)
                   and inner.launches == inner.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and clean and launches_ok and not still
                       and installed_ok
                       and (repairs > 0 if repair else repairs == 0)),
        "repair_wired": repair,
        "installed_plans": list(installed),
        "installed_as_expected": installed_ok,
        "deposit_points_repaired": repairs,
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_that_never_moved": still,
        "source_component": component,
        "source_size": list(size),
        "deposit_points_on_a_cleared_cell": on_cleared,
        "source_field_type": str(actual_sources[0].field_type),
        "deposit_points": int(np.asarray(
            deposit_repair._deposit_index(actual_sources[0])[0]).size),
        "runs": inner.runs, "launches": inner.launches,
        "launches_per_run": inner.launches_per_run,
        "live_passes": list(live),
        "selected": {slot: composed.selected.get(slot)
                     for slot in ("step_B", "update_H")},
    }


def leg_deposit_null_control(keywords: Mapping[str, Any], seed: int, steps: int,
                             ) -> Dict[str, Any]:
    """The unbracketed launch MUST diverge. Scored on diverging, not on agreeing."""
    result = run_deposit_case(keywords, seed, steps, repair=False)
    diverged = not result.get("bit_identical", False)
    return {
        "passed": bool(diverged and result.get("launches")),
        "diverged": diverged,
        "installed_plans": result.get("installed_plans"),
        "first_divergence": result.get("first_divergence"),
        "differing_words": result.get("differing_words"),
        "differing_arrays": result.get("differing_arrays"),
        "deposit_points_repaired": result.get("deposit_points_repaired"),
        "launches": result.get("launches"),
    }


def leg_refusal() -> Dict[str, Any]:
    """The source seam and the fold, each measured by name.

    THE MOST IMPORTANT REFUSAL LEG IN THIS FILE, AND CLAUSE 2 HAS INVERTED. The
    magnetic source slot used to CAP this family on the measured corpus; it is now
    CARRIED, because ``fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR`` is True and
    ``launch._install_fused_pair`` brackets the launch. So this leg no longer asks
    that a magnetic source be refused -- leg ``deposit`` measures the numbers that
    admission is worth -- and asks instead the four questions the flip does NOT
    license:

    1. an UNDECLARED source list must still be refused (ignorance is not an empty
       set, and no repair can carry a deposit nobody declared);
    2. a real MAGNETIC ``VolumeSource`` must now be ADMITTED, and the admission must
       come with the WIRING: ``launch.plan_step(..., fuse=True)`` must put a
       ``LeadingRepairPlan`` in ``step_B`` and a ``TrailingRepairPlan`` in
       ``update_H``. Admission without those two wrappers is the exact defect
       ``deposit_repair`` exists to prevent, so the two are asserted together;
    2b. a magnetic source that does NOT publish the index it writes must still be
       refused BY NAME -- the flip carries deposits this module can reconstruct, not
       every deposit -- and it must be refused with the family's PLAN too, not only
       its predicate;
    3. a real ELECTRIC ``VolumeSource`` is still not this seam's business;
    4. a FOLDED grid must still be refused, naming both B-side fills.

    5. THE FLAG DOES NOT TRAVEL BY DELEGATION.
       :mod:`~meep_gpu.metal_kernels.nonlinear_fused_magnetic_pair` borrows this
       family's whole predicate and holds NO row in ``launch.FUSED_PAIR_ARMS``, so
       nothing would bracket its launch. It must still refuse the same magnetic
       source this family now admits, with the driver line cited. A borrowed
       predicate that borrowed the wiring claim would be a product reporting success
       against a pre-injection field, reached without anyone editing a constant.
    """
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.metal_fused_magnetic_pair_coverage(fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = family.metal_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), residency)
    magnetic_plan = family.plan_metal_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), residency=residency)

    # THE WIRING, ASKED OF THE SHIPPED COMPOSER. A separate Residency, because the
    # one above already carries this family's mirrors and plan_step registers its
    # own for every slot it fills.
    wired_fields, wired_pml = build(REFUSAL_CASE, 91001)
    wired_source = _volume_source(wired_fields, "Hy")
    composed = metal_launch.plan_step(wired_fields, wired_pml, residency=Residency(),
                                      sources=(wired_source,), fuse=True)
    installed = [type(composed.plans.get(slot)).__name__
                 for slot in ("step_B", "update_H")]
    wiring_ok = installed == ["LeadingRepairPlan", "TrailingRepairPlan"]

    # 2b. THE DEPOSIT THIS MODULE CANNOT SAVE. A source whose index the engine never
    # resolved publishes no `_point_ix`, and `deposit_repair.save` would have nothing
    # to capture. Built by BLANKING the real source's resolved index rather than by
    # stubbing a fake source, so `field_type` stays the engine's own answer.
    indexless = _volume_source(fields, "Hy")
    indexless._point_ix = indexless._point_iy = indexless._point_iz = None
    indexless_coverage = family.metal_fused_magnetic_pair_coverage(
        fields, pml, (indexless,), residency)
    indexless_named = [r for r in indexless_coverage.reasons
                       if "does not publish the index it writes" in r]
    indexless_plan = family.plan_metal_fused_magnetic_pair(
        fields, pml, sources=(indexless,), residency=residency)

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.metal_fused_magnetic_pair_coverage(
        fields, pml, (electric,), residency)

    folded_fields, folded_pml = matrix.folded(boundaries={"y": "metallic"}, depth=1.2)
    folded_coverage = family.metal_fused_magnetic_pair_coverage(
        folded_fields, folded_pml, (), Residency())
    folded_named = [r for r in folded_coverage.reasons
                    if "fill_symmetry_bc_B" in r and "fill_folded_far_ghosts_B" in r]

    # 5. The borrower states its own claim, and it is False. ON A NONLINEAR
    # FIXTURE, because that arm's first clause requires an INSTALLED chi2/chi3: on
    # the linear grid above it refuses for the absence of a nonlinearity and the
    # delegated seam clause is never reached, so the row would report a refusal that
    # says nothing about the flag. `matrix.nonlinear(cart())` is the same builder the
    # composition matrix's `nonlinear_real` row uses.
    nl_fields, nl_pml = matrix.nonlinear(matrix.cart())
    nl_magnetic = _volume_source(nl_fields, "Hy")
    nl_residency = Residency()
    borrowed = nonlinear.metal_nonlinear_fused_magnetic_pair_coverage(
        nl_fields, nl_pml, (nl_magnetic,), nl_residency)
    borrowed_named = [r for r in borrowed.reasons
                      if "is magnetic" in r and "driver.py:3283-3284" in r]
    # AND THE ROW IT WOULD OTHERWISE HOLD. Without the source the same arm must
    # ADMIT, or the refusal above is about the fixture rather than about the seam.
    borrowed_clean = nonlinear.metal_nonlinear_fused_magnetic_pair_coverage(
        nl_fields, nl_pml, (), nl_residency)

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and magnetic_coverage.covered
                       and magnetic_plan is not None
                       and wiring_ok
                       and not indexless_coverage.covered and indexless_named
                       and indexless_plan is None
                       and electric_coverage.covered
                       and not folded_coverage.covered and folded_named
                       and not borrowed.covered and borrowed_named
                       and borrowed_clean.covered
                       and not nonlinear.CARRIES_DEPOSIT_REPAIR
                       and family.CARRIES_DEPOSIT_REPAIR),
        "family_carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_admitted": magnetic_coverage.covered,
        "magnetic_reasons": list(magnetic_coverage.reasons),
        "magnetic_plan_built": magnetic_plan is not None,
        "repair_plans_installed": installed,
        "repair_wiring_present": wiring_ok,
        "indexless_source_refused": not indexless_coverage.covered,
        "indexless_named": indexless_named,
        "indexless_plan_is_none": indexless_plan is None,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "folded_refused": not folded_coverage.covered,
        "folded_named": folded_named,
        "borrower_carries_deposit_repair": nonlinear.CARRIES_DEPOSIT_REPAIR,
        "borrower_refuses_the_same_source": not borrowed.covered,
        "borrower_named": borrowed_named,
        "borrower_admits_the_same_row_without_the_source": borrowed_clean.covered,
        "borrower_clean_reasons": list(borrowed_clean.reasons),
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
    probe_fields, probe_pml = build(cases[MUTATION_CASE], 1)
    codes, walls = _specialisation(probe_fields, probe_pml)
    assert all(walls), (
        f"{MUTATION_CASE} does not wall every axis; the wall mutations would be "
        f"armed on lines the shipped kernel does not emit")
    mutants = shader_mutations(codes, walls)
    neutral = {shaders.CONTRACT_OFF:
               compile_source(byte_neutral_source(codes, walls)
                              ).fused_magnetic_pair_step}

    controls = ("wall_xz_3d", "all_periodic_3d")
    # The +-0 lattice is run on the SAME rows the uniform class is, not on a
    # reduced set: the wall clear, the ghost rule and the ownership mask each
    # write a sign, and a class carried on one row would leave the others
    # single-classed.
    lattice_cases = tuple(name for name, _ in CASES)
    # THE DEPOSIT ROWS RUN ON EVERY CASE, not on a representative one. The wall
    # clear is the pass whose ORDER relative to the injection differs between the
    # fused kernel (inline, before) and the driver (host, after), so a walled row is
    # where a repair that mishandled it would show -- and the all-periodic row is
    # where it could not hide behind a clear that zeroed the difference away.
    deposit_cases = tuple(name for name, _ in CASES)
    deposit_null_cases = (MUTATION_CASE, "all_periodic_3d")
    # THE HALF-STEP RUNS ON EVERY CASE, not on a representative one: this product
    # performs the wall clear IN-LAUNCH, and the half-step is the one site where
    # the driver runs no host pass on top of it — so a walled row and an
    # all-periodic row are different compositions of exactly the pass in question.
    sync_cases = tuple(name for name, _ in CASES)
    #: (label, case, size). The EXTENDED row deposits into the whole grid on the
    #: all-walled case, which is where a deposit could touch a cleared cell if any
    #: deposit ever could -- `deposits_on_a_cleared_cell` reports whether one did.
    deposit_extended = ((f"{MUTATION_CASE}:extended_volume_deposit",
                         MUTATION_CASE, EXTENDED_SIZE),)
    total = (1 + 1 + len(CASES) + len(lattice_cases) + 1 + 1 + len(controls)
             + len(deposit_cases) + len(deposit_extended) + len(deposit_null_cases)
             + len(sync_cases)
             + 1 + 1 + len(mutants) + len(HOST_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "twenty_seven_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_bytes",
                     **leg_transcription(codes, walls)})
        emit(handle, rows[-1])

        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 61000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, name in enumerate(lattice_cases):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps,
                   **run_case(cases[name], 64000 + offset, args.steps,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": f"{values.SUBNORMAL_BAND}_is_refused_and_the_"
                              f"census_fires",
                     **leg_subnormal_ladder(cases[MUTATION_CASE], 66000)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "what_the_pm_zero_lattice_catches",
                     **leg_value_class_mutation_score(
                         cases[MUTATION_CASE], 68000, args.steps, mutants)})
        emit(handle, rows[-1])

        for offset, name in enumerate(sync_cases):
            index += 1
            row = {"index": index, "total": total, "leg": "synchronize",
                   "label": f"{name}:the_magnetic_half_step",
                   "steps": args.steps,
                   **leg_synchronize(cases[name], 71000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, name in enumerate(controls):
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(cases[name], 67000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, name in enumerate(deposit_cases):
            index += 1
            row = {"index": index, "total": total, "leg": "deposit",
                   "label": f"{name}:magnetic_source_in_the_seam",
                   "steps": args.steps,
                   **run_deposit_case(cases[name], 69000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, (label, name, size) in enumerate(deposit_extended):
            index += 1
            row = {"index": index, "total": total, "leg": "deposit",
                   "label": label, "steps": args.steps,
                   **run_deposit_case(cases[name], 69500 + offset, args.steps,
                                      size=size)}
            rows.append(row)
            emit(handle, row)

        # THE NULL CONTROL, SCORED ON DIVERGING. Same shipped plan builder, empty
        # source list, injection performed anyway: the `NoopPlan` the False branch
        # of the flag would leave in the trailing slot, measured.
        for offset, name in enumerate(deposit_null_cases):
            index += 1
            row = {"index": index, "total": total, "leg": "deposit_null_control",
                   "label": f"{name}:unbracketed_launch_must_diverge",
                   "steps": args.steps,
                   **leg_deposit_null_control(cases[name], 70000 + offset,
                                              args.steps)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "the_source_seam_and_the_fold", **leg_refusal()})
        emit(handle, rows[-1])

        # THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT. Scored on NOT diverging.
        index += 1
        result = run_case(cases[MUTATION_CASE], 71000, args.steps, functions=neutral)
        row = {"index": index, "total": total, "leg": "byte_neutral_control",
               "label": "register_replaced_by_a_reload_of_the_same_word",
               "diverged": not result.get("bit_identical", False),
               "passed": bool(result.get("passed")),
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches")}
        rows.append(row)
        emit(handle, row)

        for name, functions in mutants.items():
            index += 1
            result = run_case(cases[MUTATION_CASE], 72000 + index, args.steps,
                              functions=functions)
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
            result = run_case(cases[MUTATION_CASE], 72000 + index, args.steps,
                              patch=patch)
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
        result = run_case(cases[MUTATION_CASE], 72000 + index, args.steps)
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
                   "deposit_rows": len(deposit_cases) + len(deposit_extended),
                   "deposit_null_controls": len(deposit_null_cases),
                   "shader_mutations": len(mutants),
                   "host_mutations": len(HOST_MUTATIONS),
                   "byte_neutral_controls": 1,
                   "synchronize_rows": len(sync_cases),
                   "pm_zero_lattice_rows": len(lattice_cases),
                   "subnormal_ladder_rungs": len(values.PRECONDITION_SCALES)},
        "value_classes": {
            "compared": list(values.VALUE_CLASSES),
            "refused": [values.SUBNORMAL_BAND],
            "refusal_reason": values.band_refusal_note(),
        },
        "mutation_case": MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes), "zero_metal": list(walls)},
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_recut_2026-08-16",
            "admit_both_halves": 46,
            "and_declare_no_magnetic_source": 24,
            "note": "the magnetic-source clause costs 22 of the 46; 24 is an upper "
                    "bound because the census records source FIELD TYPES but not "
                    "whether a row's grid would also satisfy the residency and "
                    "wall-readability clauses this predicate adds",
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
