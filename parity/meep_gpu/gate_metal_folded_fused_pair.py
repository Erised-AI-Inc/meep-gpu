#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED fused Metal kernel: ``step_D`` -> ``update_E``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans FIVE driver
passes — ``step_D``, ``fill_symmetry_bc_D``, ``zero_metal_D``,
``fill_folded_far_ghosts_D`` and ``update_E`` — so a per-sub-step comparison could
not see it at all: the whole claim is about the SEAM between them, and about mirror
fills carried INSIDE one dispatch whose source cells other threads compute. The
comparison is therefore per COMPLETE DRIVER STEP over a stated budget, against an
array-path oracle running the identical live pass list, with the first divergent
step reported rather than a final pass/fail.

THE SCOPE THIS FILE OWNS, AND THE ONE IT DOES NOT. Every PRODUCT row here is
MIRROR_METALLIC — the termination this family shipped on, where no far ghost exists
and the near carry is the whole of the fill. The MIRROR_PERIODIC termination and the
``fill_folded_far_ghosts_D`` carry that closed it on 2026-08-21 are certified by
``gate_metal_folded_far_carry_d.py``, which carries the periodic product rows, the
ownership law, the reflect-row parity, the moved far coefficient and the far
mutations. Splitting them keeps each file's product rows about one termination; legs
``refusal`` and ``deposit`` are where the two meet, and both build PERIODIC fixtures
deliberately: the refusal leg because that is where the retired clause has to be shown
still retired, and the deposit leg because the FAR ghost is one of the two fills whose
images the 2026-08-28 repair has to carry, and a deposit case that never ran on a
periodic fold would measure only the near one.

THE FIVE THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION, because the
   oracle is the array path this walk would otherwise take. So every case asserts
   the exact LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``,
   and every case asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** THIRTEEN shader mutations and TWO host-binding mutations are
   ARMED, and each must be CAUGHT. Leg ``disarm`` reruns the identical case with
   the shipped bytes and no host patch and requires zero, so a mutation reported as
   caught cannot be a harness that diverges anyway.
3. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no
   lever, so byte-identity is claimed subject to a CHECKED subnormal-free
   precondition; every case censuses the oracle's own state and reports it.
4. **An unmeasured platform assumption.** This family is ONE dispatch for all three
   components because the eight scalars are packed into a ``constant Params&``. Leg
   ``binding_ceiling`` compiles the 38-binding separate-scalar signature and
   requires the FAILURE, then compiles AND LAUNCHES the packed 31-binding one and
   requires every struct field to read back correctly. Both halves are measured;
   neither is inherited from the first fused pair's docstring, which records the
   compile and never launched it.
5. **A refusal that is really an omission.** Leg ``refusal`` builds a folded
   PERIODIC grid — where ``fill_folded_far_ghosts_D`` is live inside the seam — and
   measures the source clause in BOTH directions: a REAL electric deposit placed on a
   row a fill images is ADMITTED (the 2026-08-28 closure), a source that publishes no
   deposit index is refused BY NAME, an undeclared source list is a refusal rather
   than an assumed empty one, and the same real deposit is refused again with
   ``CARRIES_DEPOSIT_REPAIR`` held False — so the admission is the flag's doing and
   not a clause that quietly went away. It also requires the folded PERIODIC
   configuration itself to be ADMITTED and the far pass DECLARED, which is the half
   that says the clause retired on 2026-08-21 has not quietly come back.
6. **A flag without a measurement.** ``CARRIES_DEPOSIT_REPAIR`` became True on
   2026-08-28, and leg ``deposit`` is what that claim rests on here: the kernel steps
   a seam with a REAL source in it, in the DRIVER'S order (the injection and the three
   fill/clear passes on the host between the two consults, because the driver runs
   them unconditionally), byte-compared per complete step against the array path. Each
   case is placed on a row a fill READS FROM — near, far, and a two-fold corner — and
   carries TWO controls that must diverge: no repair at all, and the repair cut back
   to the deposit index, which is the shape this module shipped before the closure and
   the one that isolates what the closure buys.

LEGS
  0  binding_ceiling   38 separate bindings must FAIL; 30 pointers + one packed
                       Params& must COMPILE, LAUNCH and read back
  1  product           complete steps, per-step byte compare, launch counters,
                       movement, subnormal census
  2  separate_control  the two ALREADY CERTIFIED folded products stepping the same
                       seam as separate dispatches with the fill and the wall clear
                       on the host between them: three-way byte agreement plus the
                       dispatch and host-pass counts the fusion removes
  3  deposit           a REAL in-seam deposit on an imaged row, stepped on the
                       device in the driver's order and byte-compared; two controls
                       (no repair; repair without the image closure) must diverge
  4  refusal           the deposit is carried and what cannot be carried is refused
                       by name, in both directions on the flag; the retired fold
                       clause is still retired and the far pass still declared
  5  mutation          fifteen armed defects, each of which MUST diverge
  6  disarm            the same harness, shipped bytes, must not diverge

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
    folded_fused_pair as family, launch as metal_launch, shaders, subnormal,
    symmetry,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the whole-step arbiter's and the
#: first fused pair's, and for its reason: the classes this gate exists for
#: COMPOUND. ``fu_D`` and ``f_w_E`` are state carried between steps, and a ghost
#: plane imaged one row over is a defect that needs several steps to reach the low
#: bits of the interior. The comparison is per COMPLETE STEP.
STEPS = 12

#: Which array-path function each live pass is. The two FILL passes are in this
#: table and are not in the first fused pair's, which is the whole difference
#: between an unfolded seam and a folded one.
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

#: Every stored volume a folded complete step can touch. The ``fu_*`` PML
#: auxiliaries and the ``f_w_*`` constitutive workspaces are STATE: a kernel right
#: for one launch and wrong forever after diverges only once they accumulate.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> ``metal_composition_matrix.folded`` keywords.
#:
#: EVERY ROW IS MIRROR_METALLIC. That is now a SCOPE SPLIT rather than a refusal:
#: the periodic termination is carried by the kernel as of 2026-08-21 and is
#: certified by ``gate_metal_folded_far_carry_d.py``. What the rows carry between
#: them:
#:
#:   PHASE          the parity is a SOURCE specialisation on this backend, so an
#:                  odd plane is a DIFFERENT COMPILED KERNEL and not a scalar;
#:   WALL           ``zero_metal_D`` only emits a line when some axis is metallic
#:                  and NOT mirrored, so the rows without a ``z`` wall would report
#:                  the two wall mutations as structurally absent;
#:   FOLDED AXES    one folded axis exercises no composition at all; two make a
#:                  corner unowned on both planes carry the PRODUCT of the two
#:                  parities; three is the maximum the engine builds, and no single
#:                  component ever sees more than two of them;
#:   DIMENSIONALITY the 2-D row is the corpus's most common folded shape, and the
#:                  3-D rows are where a ghost plane is 192 words rather than 16.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("y_metallic_even_3d", dict(boundaries={"y": "metallic"}, depth=1.2)),
    ("y_metallic_odd_3d", dict(phase=-1, boundaries={"y": "metallic"}, depth=1.2)),
    ("y_metallic_even_2d", dict(boundaries={"y": "metallic"})),
    ("y_metallic_wall_z_3d",
     dict(boundaries={"y": "metallic", "z": "metallic"}, depth=1.2)),
    ("xy_mixed_wall_z_3d",
     dict(axis="XY", phase=(1, -1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2)),
    ("xy_mixed_odd_3d",
     dict(axis="XY", phase=(-1, 1),
          boundaries={"x": "metallic", "y": "metallic"}, depth=1.2)),
    ("xyz_all_folded_3d",
     dict(axis="XYZ", phase=(1, -1, 1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=2.0)),
)

#: The case the mutations are armed on. Two folded axes with MIXED parities and a
#: live ``z`` wall, so every line the shipped kernel can emit is present: both
#: parity spellings, the composite corner, the wall clear on the owned cell AND on
#: the imaged one, and the full six-row ownership mask. A mutation leg run on a
#: single-fold row would report the composition defects as uncaught for the
#: uninteresting reason that no line is emitted there.
MUTATION_CASE = "xy_mixed_wall_z_3d"

#: The row leg ``refusal`` builds: a folded PERIODIC axis, where
#: ``fill_folded_far_ghosts_D`` is live inside the seam AND — since the far carry of
#: 2026-08-21 — carried by the kernel rather than refused.
REFUSAL_CASE: Dict[str, Any] = dict(depth=1.2)


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
    same builder the folded family's own gate uses, so the fixture this certifies
    is the fixture that family was certified on. Only the field state is reseeded
    here, and identically on every side — the epsilon volume must stay bit-equal or
    the comparison measures the material rather than the kernel.
    """
    fields, pml = matrix.folded(**dict(keywords))
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
    pre-``step_B`` bytes for H, which is smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns FOUR passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``fill_symmetry_bc_D`` and ``zero_metal_D`` running on the host
    on top of the copies the kernel already carried — which, for the fill, is
    IDEMPOTENT and would hide a dropped carry.
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
    plan = family.plan_metal_folded_fused_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_folded_fused_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": "the folded fused pair was refused",
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
        "boundary_codes": list(plan.codes), "phases": list(plan.phases),
        "zero_metal": list(plan.zero_metal),
        "carried_ghost_axes": [list(axes) for axes in plan.carried_axes],
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         ) -> Dict[str, Any]:
    """The SEPARATE certified folded products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path, the two ALREADY CERTIFIED folded Metal
    products stepping the same seam as separate dispatches with the mirror fill and
    the wall clear left on the HOST between them, and the fused product. All three
    must agree word for word at every complete step, and the DISPATCH COUNTS and
    the surviving in-seam host passes are recorded on both sides — which is the
    only place the difference between the compositions shows up at all, since a
    correct fusion is byte-neutral by construction.

    The separate side is not a strawman: those two plans are exactly what
    ``plan_step`` composes today for this configuration.
    """
    reference, reference_pml = build(keywords, seed)
    separate, separate_pml = build(keywords, seed)
    fused, fused_pml = build(keywords, seed)

    separate_residency = Residency()
    curl = symmetry.plan_folded_pml_curl(separate, separate_pml, "step_D",
                                         separate_residency)
    electric = symmetry.plan_folded_constitutive(separate, separate_pml, "E",
                                                 separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_folded_fused_pair(
        fused, fused_pml, sources=(), residency=fused_residency)
    if curl is None or electric is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "constitutive": electric is not None,
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

    # The host passes each composition leaves on the array path INSIDE the seam.
    # The fused side carries the mirror fill and the wall clear in the kernel; the
    # separate side does not, so on every folded run it pays a host round trip the
    # fused side does not.
    seam = ("fill_D", "zero_metal_D", "fill_folded_far_ghosts_D")
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
        "seam_host_passes_for_separate": [n for n in live if n in seam],
        "seam_host_passes_for_fused": [],
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...],
                                                    Tuple[int, ...],
                                                    Tuple[bool, ...]]:
    """The (codes, phases, walls) triple the shipped plan compiles from."""
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    assert codes is not None
    phases = tuple(int(fields.grid.mirror_phase(axis) or 0) for axis in range(3))
    return tuple(int(code) for code in codes), phases, zero_metal_axes(fields.grid)


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


def shader_mutations(codes: Sequence[int], phases: Sequence[int],
                     walls: Sequence[bool]) -> Dict[str, Dict[str, Any]]:
    """Thirteen source defects, each a plausible transcription slip.

    Every needle is anchored on text that only the mutated line carries, and every
    one is verified present before it is applied — an absent needle raises here
    rather than silently arming nothing.

    THE MUTATION CASE'S GEOMETRY, which is what makes each of these reachable:
    x and y folded with phases (+1, -1), z a live metallic wall. Component 0 (Dx)
    images its ghost on y with parity -1; component 1 (Dy) images on x with parity
    +1; component 2 (Dz) images on BOTH, so it carries the composite corner whose
    parity is the product.
    """
    base = family.folded_fused_pair_source(codes, phases, walls)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float osrc = v0 * ie0[ii];",
                   "float osrc = curl0 * ie0[ii];"),
        # THE FILL, DROPPED. The ghost cell then keeps last step's displacement
        # and last step's E — which is what a fused pair that ignored
        # `fill_symmetry_bc_D` would produce on every folded grid.
        "near_fill_dropped":
            needle(base, "        if ((j == 2)) {", "        if (false) {"),
        # THE PARITY, DROPPED on the odd plane. `-v0` is component 0's ghost and
        # is the only `-v0` in the source.
        "fold_parity_dropped":
            needle(base, "float gj_v = -v0;", "float gj_v = v0;"),
        # THE COMPOSITE PARITY IS THE PRODUCT of the two planes' parities, not
        # either one. Anchored on the doubly shifted index, which only the corner
        # block carries.
        "composite_parity_is_not_the_product":
            needle(base, "int gij_i = ii - 2 * nyz - 2 * nzi;\n"
                         "            float gij_v = -v2;",
                   "int gij_i = ii - 2 * nyz - 2 * nzi;\n"
                   "            float gij_v = v2;"),
        # THE SOURCE ROW. MEEP's `little_owned_corner0` puts the image at stored
        # cell 2, never cell 1 (stepping._mirror_source, MIRROR_SOURCE_INDEX).
        "ghost_images_the_wrong_source_row":
            needle(base, "int gj_i = ii - 2 * nzi;", "int gj_i = ii - 1 * nzi;"),
        # THE ORDER. The driver fills BEFORE it clears the wall, so a cleared
        # ghost is `+0.0`; clearing first and negating after leaves `-0.0`, which
        # is a byte difference and nothing else.
        "ghost_clear_before_the_parity":
            needle(base, "float gj_v = -v0;\n"
                         "            gj_v = at_z ? 0.0f : gj_v;",
                   "float gj_v = v0;\n"
                   "            gj_v = at_z ? 0.0f : gj_v;\n"
                   "            gj_v = -gj_v;"),
        # THE WALL CLEAR ON THE IMAGED CELL. `zero_metal_D` runs after the fill and
        # reaches the ghost plane's intersection with the wall.
        "ghost_wall_clear_dropped":
            needle(base, "            gj_v = at_z ? 0.0f : gj_v;\n", ""),
        # THE WALL CLEAR ON THE OWNED CELL.
        "zero_metal_dropped":
            needle(base, "        v0 = at_z ? 0.0f : v0;\n", ""),
        "ownership_mask_dropped":
            needle(base, "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "fw_store_dropped":
            needle(base, "        w0[ii] = osrc;", "        // stale f_w_Ex"),
        "constitutive_accumulations_reversed":
            needle(base, "        oacc = oacc + kp_0 * osrc;\n"
                         "        oacc = oacc - km_0 * oprev;",
                   "        oacc = oacc - km_0 * osrc;\n"
                   "        oacc = oacc + kp_0 * oprev;"),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        mutants[name] = {shaders.CONTRACT_OFF:
                         compile_source(source).folded_fused_pair_step}
    return mutants


#: Where each binding group starts in the plan's argument tuple. Spelled once, so
#: the two host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(18, 24))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(24, 30))


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_e_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the INTEGER-lattice constitutive coefficients to the E half.

    ``update_E`` takes ``kps_a_h``/``kms_a_h`` and ``update_H`` takes ``kps_a``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from, so this is a silent half-cell error in the absorber
    profile — converged, smooth and wrong — and no shader mutation can reach it.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the D curl the HALF-INTEGER split-field coefficients.

    ``step_D`` reads integer positions and ``step_B`` half-integer ones
    (launch.py:110-113, SUB_STEPS' ``suffix``). Getting it backwards is a half-cell
    error, not a crash.
    """
    _rebind(plan, CURL_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "e_half_takes_the_integer_lattice": swap_e_lattice,
    "curl_takes_the_half_integer_lattice": swap_curl_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """35 separate bindings must FAIL; the packed 31 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. The first fused pair's docstring records the COMPILE half of
    the packing result and never launched it; a signature that compiled but bound
    the struct wrongly would produce a smooth, plausible, wrong field, so the launch
    half is measured here.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {"separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
                           "packed_bindings": family.PACKED_BINDINGS, "ceiling": 31}
    try:
        compile_source(family.refuted_separate_scalar_source())
        row.update(separate_compiled=True, separate_error="",
                   passed=False,
                   note="the 38-binding signature COMPILED; this family's shape "
                        "rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    # The packed signature, compiled AND launched, with every field read back.
    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(30))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };", "",
        "kernel void packed_probe(", pointers,
        "    constant Params&    prm     [[buffer(30)]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;", "    b5[idx] = b29[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(30)]
    buffers[29] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx = (3, 4, 5), 0.125
    # THE RECORD IS EIGHT SCALARS since the far carry of 2026-08-21; the three
    # reflect rows ride in it rather than as bindings. This probe binds the real
    # packer, so a signature that stopped matching the plan builder's record fails
    # here rather than silently reading a shifted field.
    params = family._params_tensor(shape, dtdx, (-1, -1, -1), "mps")
    # n_elem in the record is the product of the shape, which is 60 here and NOT
    # the eight-element probe buffers; the guard is exercised by binding the real
    # packer rather than a hand-built record, so the field the kernel reads is the
    # field the plan builder writes.
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


#: The two DEPOSIT cases' fixtures and the component each drives. THE COMPONENT IS THE
#: LEVER, not the grid: which fill images a deposit is decided by the DEPOSITED
#: component's Yee shift on the folded axis, so ``Ez`` (shift 0 on Y) exercises the NEAR
#: fill and ``Ey`` (shift 1) the FAR ghost, on the SAME grid. A sweep that varied only
#: the fixture would run the near carry three times and never touch the far one.
#:
#: The XY row is the two-fold corner: with both folds live a deposit sitting on a source
#: row of BOTH carries three images, the composite one holding the product of two
#: parities, which is the case a single-axis closure would half-solve.
DEPOSIT_CASES: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("periodic_y_near_Ez", dict(depth=1.2), "Ez"),
    ("periodic_y_far_Ey", dict(depth=1.2), "Ey"),
    ("metallic_y_near_Ez", dict(boundaries={"y": "metallic"}, depth=1.2), "Ez"),
    ("xy_periodic_corner_Ey", dict(axis="XY", phase=(1, -1), depth=1.2), "Ey"),
)

#: The array the seam's constitutive half reads for each electric component -- the array
#: the two fills actually write, and so the one whose Yee shift decides the images.
DEPOSIT_TARGET = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}


def deposit_source(fields: Any, component: str) -> Any:
    """A real point source PLACED on every row a post-injection fill reads from.

    NOT SEARCHED FOR BY DIVERGENCE, and not spelled either: ``fill_image_rules`` names
    the row each live fill images on each axis (``deposit_repair``), and this walks that
    axis until the source's own deposit index lands on it. A hand-written coordinate
    would silently stop being on the imaged row the moment an extent or a Yee table
    moved, and the case would pass while measuring the ordinary carry.
    """
    grid = fields.grid
    centre = [0.0, 0.0, 0.0]
    for axis, row, _destination in deposit_repair.fill_image_rules(
            fields, DEPOSIT_TARGET[component]):
        origin, spacing = float(grid.axis_origin(axis)), float(grid.dx)
        for step in range(2 * int(grid.shape[axis]) + 2):
            trial = list(centre)
            trial[axis] = origin + 0.5 * step * spacing
            try:
                candidate = GaussianPulsedSource(
                    grid=grid, component=component, center=tuple(trial),
                    size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2, amplitude=1.0)
            except ValueError:
                continue  # a parity the plane refuses; not this axis's answer
            if getattr(candidate, "_point_ix", None) is None:
                continue
            index = (candidate._point_ix, candidate._point_iy,
                     candidate._point_iz)[axis]
            if row in {int(value) for value in np.atleast_1d(index)}:
                centre[axis] = trial[axis]
                break
    return GaussianPulsedSource(grid=grid, component=component, center=tuple(centre),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


def _deposit_cells(source: Any) -> List[Tuple[int, int, int]]:
    return sorted({tuple(int(value) for value in cell) for cell in np.stack(
        [np.atleast_1d(source._point_ix), np.atleast_1d(source._point_iy),
         np.atleast_1d(source._point_iz)], axis=1)})


def _image_cells(fields: Any, source: Any, component: str) -> List[Tuple[int, int, int]]:
    """The cells the SHIPPED closure adds to this deposit, as index triples."""
    cells = deposit_repair.repair_cells(
        fields, DEPOSIT_TARGET[component],
        (source._point_ix, source._point_iy, source._point_iz))
    whole = {tuple(int(value) for value in cell)
             for cell in np.stack([np.atleast_1d(column) for column in cells], axis=1)}
    return sorted(whole - set(_deposit_cells(source)))


def run_deposit_case(keywords: Mapping[str, Any], seed: int, steps: int,
                     component: str, repair: bool = True,
                     carry_images: bool = True) -> Dict[str, Any]:
    """THE IN-SEAM DEPOSIT, on the device, in the DRIVER'S OWN ORDER.

    Every other product leg in this file steps a seam with NO source in it, because
    until 2026-08-28 this family refused every one. It now declares
    ``CARRIES_DEPOSIT_REPAIR``, and a flag is not a measurement: this leg runs the
    kernel with a real deposit inside its seam and byte-compares the whole state
    against the array path, per complete step.

    THE WALK IS THE DRIVER'S, NOT ``metal_step``'S. ``metal_step`` skips every pass in
    ``REPLACES`` because the kernel carried them; that models a CLEAN seam. The driver
    runs the injection and the three fill/clear passes UNCONDITIONALLY between its two
    consults (driver.py:3301-3311), so on a seam carrying a deposit the kernel's inline
    fills happen BEFORE the injection and the host's happen after -- which is the whole
    reason the repair has to carry the fills' images and not just the deposit index.
    Only ``step_D`` (the leading slot) and ``update_E`` (the repair) are taken from the
    plan here; everything else runs on the host exactly as the driver runs it.

    TWO CONTROLS, both required to diverge, and they fail differently:
    ``repair=False`` leaves the deposit itself wrong, and ``carry_images=False`` cuts
    ``repair_cells`` back to the deposit index -- the shape this module shipped before
    the closure -- so what is left wrong is exactly the image set. The second is the one
    that says the closure, and not something else, is what makes this leg pass.
    """
    original = deposit_repair.repair_cells
    probe, _probe_pml = build(keywords, seed)
    placed = deposit_source(probe, component)
    centre = tuple(float(value) for value in placed.center)
    images = _image_cells(probe, placed, component)
    deposits = _deposit_cells(placed)
    if not carry_images:
        deposit_repair.repair_cells = lambda fields, target, index: index
    try:
        reference, reference_pml = build(keywords, seed)
        actual, actual_pml = build(keywords, seed)
        drift = compare(reference, actual)
        assert not drift, f"the two builds are not identical: {drift}"

        def source_on(fields: Any) -> Any:
            return GaussianPulsedSource(grid=fields.grid, component=component,
                                        center=centre, size=(0.0, 0.0, 0.0),
                                        frequency=1.0, fwidth=0.2, amplitude=1.0)

        reference_source, actual_source = source_on(reference), source_on(actual)
        residency = Residency()
        plan = family.plan_metal_folded_fused_pair(
            actual, actual_pml, sources=(actual_source,), residency=residency)
        if plan is None:
            reasons = family.metal_folded_fused_pair_coverage(
                actual, actual_pml, (actual_source,), residency).reasons
            return {"passed": False, "reason": "the deposit was refused",
                    "refusals": list(reasons)}
        leading = deposit_repair.LeadingRepairPlan(
            plan, actual, actual_pml, (actual_source,), "D")
        trailing = deposit_repair.TrailingRepairPlan(
            "update_E", leading, actual, actual_pml)
        live = live_passes(actual, actual_pml)
        assert live == live_passes(reference, reference_pml)
        before = frozen(actual)
        dt = float(actual.grid.dt)
        residency.sync_in()

        per_step: List[Dict[str, Any]] = []
        for step in range(1, steps + 1):
            when = (step - 1) * dt + 0.5 * dt
            for name in live:
                ARRAY_PATH[name](reference, reference_pml)
                if name == "step_D":
                    reference_source.inject(reference, when)
            for name in live:
                if name == family.SLOT:
                    residency.sync_in()
                    leading.run()
                    residency.sync_out()
                    actual_source.inject(actual, when)
                elif name == "update_E":
                    if repair:
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
        launches_ok = (plan.runs == len(per_step)
                       and plan.launches == plan.launches_per_run * len(per_step))
        return {
            # A CONTROL PASSES BY DIVERGING. Reading `bit_identical` for both
            # directions off one field would make "the control worked" and "the
            # product worked" the same sentence.
            "passed": bool(identical and clean and launches_ok and not still
                           if (repair and carry_images) else not identical),
            "repair": repair, "carries_images": carry_images,
            "component": component,
            "deposit_cells": deposits, "image_cells": images,
            "n_image_cells": len(images),
            "repairs_reported": leading.repairs,
            "bit_identical": identical, "subnormal_free": clean,
            "first_divergence": next((row["step"] for row in per_step
                                      if row["differing_words"]), None),
            "per_step": per_step,
            "differing_words": per_step[-1]["differing_words"],
            "differing_arrays": per_step[-1]["differing_arrays"],
            "arrays_compared": len(before),
            "arrays_that_never_moved": still,
            "runs": plan.runs, "launches": plan.launches,
            "launches_per_run": plan.launches_per_run,
            "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
            "source_centre": list(centre),
        }
    finally:
        deposit_repair.repair_cells = original


def leg_deposit(name: str, keywords: Mapping[str, Any], seed: int, steps: int,
                component: str) -> Dict[str, Any]:
    """One deposit case and its two controls, reported as one row.

    NON-VACUITY FIRST: the placed deposit must actually HAVE images, or the two legs
    below are the ordinary point repair wearing a folded label and the controls would
    have nothing to miss.
    """
    carried = run_deposit_case(keywords, seed, steps, component)
    point_only = run_deposit_case(keywords, seed, steps, component,
                                  carry_images=False)
    unrepaired = run_deposit_case(keywords, seed, steps, component, repair=False)
    imaged = int(carried.get("n_image_cells") or 0)
    return {
        "passed": bool(imaged and carried.get("passed")
                       and point_only.get("passed") and unrepaired.get("passed")),
        "case": name, "component": component,
        "deposit_has_images": bool(imaged),
        "n_image_cells": imaged,
        "image_cells": carried.get("image_cells"),
        "deposit_cells": carried.get("deposit_cells"),
        "carried": carried,
        "point_only_control": point_only,
        "unrepaired_control": unrepaired,
    }


def leg_refusal() -> Dict[str, Any]:
    """The clause that caps this family must still refuse BY NAME.

    RE-AIMED 2026-08-21. This leg used to require a folded PERIODIC grid to be
    REFUSED; that clause was RETIRED when ``fill_folded_far_ghosts_D`` was carried,
    and the retirement's own evidence — the periodic product rows, the ownership
    law, the top-plane obligation and the far mutations — lives in
    ``gate_metal_folded_far_carry_d.py``. What is left here is the clause that
    actually caps the family at 4 of the 53 rows admitting both halves: the
    ELECTRIC source injected INSIDE this seam (driver.py:3294-3299), and the
    undeclared source list that must never be read as an empty one.

    The far pass is still asserted LIVE on the periodic fixture, because the
    retirement is only meaningful about a pass that actually runs; and the
    configuration is asserted ADMITTED, so a clause that quietly came back would
    fail here rather than in the other gate alone.
    """

    class Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()
    coverage = family.metal_folded_fused_pair_coverage(fields, pml, (), residency)
    plan = family.plan_metal_folded_fused_pair(fields, pml, sources=(),
                                               residency=residency)
    live = live_passes(fields, pml)
    # THE ADMISSION. A REAL electric deposit, placed on a row a fill images, is now
    # covered -- this is the clause the 2026-08-28 closure retired, and it is asserted
    # on the shipped flag rather than on a remembered string.
    deposit = deposit_source(fields, "Ez")
    carried = family.metal_folded_fused_pair_coverage(
        fields, pml, (deposit,), residency)
    images = _image_cells(fields, deposit, "Ez")
    # THE REFUSAL THAT SURVIVES IT. `Source` declares a field type and nothing else, so
    # it names no cells for the repair to save; a closure cannot carry what a source
    # will not name, and that is refused BY NAME rather than admitted on the flag.
    electric = family.metal_folded_fused_pair_coverage(
        fields, pml, (Source("D"),), residency)
    undeclared = family.metal_folded_fused_pair_coverage(
        fields, pml, None, residency)
    named = [reason for reason in electric.reasons
             if "does not publish the index it writes" in reason]
    unknown = [reason for reason in undeclared.reasons
               if "was not declared" in reason]
    # AND THE OTHER DIRECTION ON THE FLAG ITSELF, so the admission above cannot be a
    # clause that quietly went away: held False, the seam refusal comes straight back.
    shipped = family.CARRIES_DEPOSIT_REPAIR
    try:
        family.CARRIES_DEPOSIT_REPAIR = False
        held_false = family.metal_folded_fused_pair_coverage(
            fields, pml, (deposit,), residency)
    finally:
        family.CARRIES_DEPOSIT_REPAIR = shipped
    seam_named = [reason for reason in held_false.reasons
                  if "is electric" in reason
                  and "BETWEEN step_D and update_E" in reason]
    return {
        "passed": bool(coverage.covered and plan is not None
                       and "fill_folded_far_ghosts_D" in live
                       and "fill_folded_far_ghosts_D" in plan.replaces_sub_steps
                       and shipped is True and images and carried.covered
                       and named and not electric.covered
                       and unknown and not undeclared.covered
                       and seam_named and not held_false.covered),
        "folded_periodic_admitted": coverage.covered,
        "folded_periodic_refusals": list(coverage.reasons),
        "far_pass_is_live": "fill_folded_far_ghosts_D" in live,
        "far_pass_is_declared": (plan is not None
                                 and "fill_folded_far_ghosts_D"
                                 in plan.replaces_sub_steps),
        "carries_deposit_repair": bool(shipped),
        "real_deposit_admitted": carried.covered,
        "real_deposit_refusals_if_any": list(carried.reasons),
        "real_deposit_image_cells": images,
        "source_without_an_index_refused": named,
        "undeclared_source_refused": unknown,
        "refused_again_with_the_flag_held_false": seam_named,
        "live_passes": list(live),
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
    codes, phases, walls = _specialisation(probe_fields, probe_pml)
    assert any(walls), (
        f"{MUTATION_CASE} has no walled axis; the two wall mutations would be vacuous")
    assert sum(1 for code in codes
               if code in symmetry.MIRROR_CODES) >= 2, (
        f"{MUTATION_CASE} folds fewer than two axes; the composite corner mutation "
        f"would be vacuous")
    mutants = shader_mutations(codes, phases, walls)

    controls = ("y_metallic_wall_z_3d", "xy_mixed_wall_z_3d")
    total = (1 + len(CASES) + len(CASES) + 1 + len(controls)
             + len(DEPOSIT_CASES) + 1
             + len(mutants) + len(HOST_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "thirty_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 41000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        # THE SECOND VALUE CLASS, on every product row.
        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps,
                   **run_case(keywords, 43000 + offset, args.steps,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "subnormal_band_is_refused_and_the_census_fires",
                     **leg_subnormal_ladder(dict(CASES)[MUTATION_CASE], 45000)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(cases[name], 47000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, (name, keywords, component) in enumerate(DEPOSIT_CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "deposit",
                   "label": name, "steps": args.steps,
                   **leg_deposit(name, keywords, 49000 + offset, args.steps,
                                 component)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "the_deposit_is_carried_and_what_it_cannot_carry_is_named",
                     **leg_refusal()})
        emit(handle, rows[-1])

        for name, functions in mutants.items():
            index += 1
            result = run_case(cases[MUTATION_CASE], 52000 + index, args.steps,
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
            result = run_case(cases[MUTATION_CASE], 52000 + index, args.steps,
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
        result = run_case(cases[MUTATION_CASE], 52000 + index, args.steps)
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
                   "deposit_cases": len(DEPOSIT_CASES),
                   "shader_mutations": len(mutants),
                   "host_mutations": len(HOST_MUTATIONS)},
        "mutation_case": MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes), "phases": list(phases),
                                    "zero_metal": list(walls)},
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
