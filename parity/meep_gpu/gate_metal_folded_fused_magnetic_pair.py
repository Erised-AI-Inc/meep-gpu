#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED fused magnetic kernel: ``step_B`` -> ``update_H``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans FOUR driver
passes — ``step_B``, ``fill_symmetry_bc_B``, ``zero_metal_B`` and ``update_H`` — so a
per-sub-step comparison could not see it at all: the whole claim is about the SEAM
between them, and about a mirror fill carried INSIDE one dispatch whose source cell
a different thread computes. The comparison is therefore per COMPLETE DRIVER STEP
over a stated budget, against an array-path oracle running the identical live pass
list, with the first divergent step reported rather than a final pass/fail.

THE SIX THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION, because the
   oracle is the array path this walk would otherwise take. So every case asserts
   the exact LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``, and
   every case asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** FIFTEEN shader mutations and TWO host-binding mutations are
   ARMED, and each must be CAUGHT. Leg ``disarm`` reruns the identical case with the
   shipped bytes and no host patch and requires zero, so a mutation reported as
   caught cannot be a harness that diverges anyway.
3. **A LINE NOTHING CAN SEE.** This family's one novel line is the imaged ghost's
   coefficient pair, read at stored index 0 rather than reused from the source
   thread at index 2 — the property :mod:`.folded_fused_pair` gets for free and this
   one does not. On a FOLDED axis the mirror plane carries no absorber
   (``stepping._require_consistent_pml``: high face only), so at the shared
   fixture's 2-cell layer ``kps[0] == kps[2]`` EXACTLY and the mutation that reuses
   the wrong entry is UNOBSERVABLE. Leg ``moved_coefficient`` censuses the
   coefficients, PREDICTS observability from them, runs the mutation on a case where
   the prediction is False and one where it is True, and requires ``caught ==
   observable`` on both. A mutation leg that quietly reported this defect as
   uncaught would be the one failure a mutation leg cannot see from its own result.
4. **A construction claimed rather than measured.** Both halves of this kernel are
   LIFTED from the certified emitters' own output rather than retyped. Leg
   ``transcription`` re-runs those emitters and requires the lifted curl head to be
   a verbatim prefix of ``symmetry.folded_curl_source``'s body and this family's
   parameterised constitutive statements to reproduce
   ``shaders.constitutive_source('H')`` statement for statement.
5. **An unmeasured platform assumption.** This family is ONE dispatch for all three
   components because the five scalars are packed into a ``constant Params&``. Leg
   ``binding_ceiling`` compiles the 32-binding separate-scalar signature and requires
   the FAILURE, then compiles AND LAUNCHES the packed one and requires every struct
   field to read back correctly.
6. **A refusal that is really an omission.** Leg ``carry`` builds a folded PERIODIC
   grid — where ``fill_folded_far_ghosts_B`` is live inside the seam — and measures
   the source clause in BOTH directions: a REAL magnetic deposit placed on a row a
   fill images is ADMITTED (the 2026-08-28 closure), a source that publishes no
   deposit index is refused BY NAME, an undeclared source list is a refusal rather
   than an assumed empty one, and the same real deposit is refused again with
   ``CARRIES_DEPOSIT_REPAIR`` held False — so the admission is the flag's doing and
   not a clause that quietly went away.
7. **A flag without a measurement.** ``CARRIES_DEPOSIT_REPAIR`` became True on
   2026-08-28 and leg ``deposit`` is what that claim rests on: the kernel steps a seam
   with a REAL magnetic source in it, in the DRIVER'S order (the injection and the
   three fill/clear passes on the host between the two consults, because the driver
   runs them unconditionally), byte-compared per complete step against the array path.
   Each case is placed on a row a fill READS FROM — near, far, and a two-fold corner —
   and carries TWO controls that must diverge: no repair at all, and the repair cut
   back to the deposit index, the shape this module shipped before the closure.

LEGS
  0  binding_ceiling   32 separate bindings must FAIL; 27 pointers + one packed
                       Params& must COMPILE, LAUNCH and read back
  1  transcription     both halves are the certified emitters' own text
  2  product           complete steps, per-step byte compare, launch counters,
                       movement, subnormal census
  3  separate_control  the two ALREADY CERTIFIED folded products stepping the same
                       seam as separate dispatches with the fill and the wall clear
                       on the host between them: three-way byte agreement plus the
                       dispatch and host-pass counts the fusion removes
  4  deposit           a REAL in-seam magnetic deposit on an imaged row, stepped on
                       the device in the driver's order and byte-compared; two
                       controls (no repair; repair without the image closure) must
                       diverge
  5  carry             the far fill is carried; the deposit is admitted and what
                       cannot be carried is refused by name, both directions on the
                       flag; the reflect-row clauses still refuse their stubs
  6  moved_coefficient predicted observability vs measured catch, on two cases
  7  mutation          seventeen armed defects, each of which MUST diverge
  8  disarm            the same harness, shipped bytes, must not diverge

Rule 7: one flushed line per case, every row appended and fsynced as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
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
    folded_fused_magnetic_pair as family, launch as metal_launch, shaders,
    subnormal, symmetry,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the whole-step arbiter's and the
#: other three fused pairs', and for their reason: the classes this gate exists for
#: COMPOUND. ``fu_B`` and ``f_w_H`` are state carried between steps, and a ghost
#: plane imaged one row over is a defect that needs several steps to reach the low
#: bits of the interior. The comparison is per COMPLETE STEP.
STEPS = 12

#: Which array-path function each live pass is. The two FILL passes are in this
#: table and are not in the unfolded fused magnetic pair's, which is the whole
#: difference between an unfolded seam and a folded one.
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

#: Every stored volume a folded complete step can touch. The ``fu_*`` PML auxiliaries
#: and the ``f_w_*`` constitutive workspaces are STATE: a kernel right for one launch
#: and wrong forever after diverges only once they accumulate.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> (``metal_composition_matrix.folded`` keywords, explicit PML faces or None).
#:
#: BOTH FOLD TERMINATIONS RUN HERE, which is the change of 2026-08-20: the
#: ``metallic`` rows are the ones this family shipped on, and the ``periodic`` rows
#: are the ones the far carry adds. They are kept SIDE BY SIDE rather than replaced,
#: because a mutation caught on one termination is not caught on the other — the
#: top-plane mask and the whole far carry are emitted only on MIRROR_PERIODIC, and
#: the near carry's ``i != 0`` ownership is the only one a MIRROR_METALLIC row has.
#:
#: What the rows carry between them:
#:
#:   PHASE          the parity is a SOURCE specialisation on this backend, so an odd
#:                  plane is a DIFFERENT COMPILED KERNEL and not a scalar;
#:   WALL           ``zero_metal_B`` only emits a line when some axis is metallic and
#:                  NOT mirrored, so the rows without a ``z`` wall would report the
#:                  two wall mutations as structurally absent;
#:   FOLDED AXES    the NEAR fill reaches a B component on its OWN axis alone and the
#:                  FAR fill on the two that are not, so the two are COMPLEMENTARY:
#:                  one folded periodic axis gives two components a far ghost and one
#:                  a near ghost, two give a component BOTH plus a composite corner
#:                  on the third, and the three-axis row is where one source thread
#:                  owns seven ghost cells. ``carried_ghost_axes`` and
#:                  ``carried_far_ghost_axes`` in every product row are where that is
#:                  visible rather than asserted;
#:   FULL-COUNT     ``_far_reflect_rows`` is ``n_full - stored + 2``, which is
#:   PARITY         ``stored - 2`` at an even full count and ``stored - 3`` at an odd
#:                  one. At ``extent=2.0`` the WRONG fixed ``n - 2`` happens to be
#:                  right, so every periodic row is carried at BOTH parities and leg
#:                  ``reflect_row`` is what turns that into a measurement;
#:   DIMENSIONALITY the 2-D row is the corpus's most common folded shape, and the 3-D
#:                  rows are where a ghost plane is 132 words rather than 11;
#:   ABSORBER DEPTH ``xy_deep_pml_3d`` is the ONLY row on which this family's moved
#:                  NEAR coefficient index is observable at all (leg
#:                  ``moved_coefficient`` measures exactly that), because a folded
#:                  axis carries no absorber at the mirror plane and the shared
#:                  matrix's 2-cell layer never reaches stored cell 2. Its PML is
#:                  built here rather than by the matrix for that one reason.
CASES: Tuple[Tuple[str, Dict[str, Any], Optional[Any]], ...] = (
    ("y_metallic_even_3d", dict(boundaries={"y": "metallic"}, depth=1.2), None),
    ("y_metallic_odd_3d",
     dict(phase=-1, boundaries={"y": "metallic"}, depth=1.2), None),
    ("y_metallic_even_2d", dict(boundaries={"y": "metallic"}), None),
    ("y_metallic_wall_z_3d",
     dict(boundaries={"y": "metallic", "z": "metallic"}, depth=1.2), None),
    ("xy_mixed_wall_z_3d",
     dict(axis="XY", phase=(1, -1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2), None),
    ("xy_mixed_odd_3d",
     dict(axis="XY", phase=(-1, 1),
          boundaries={"x": "metallic", "y": "metallic"}, depth=1.2), None),
    ("xyz_all_folded_3d",
     dict(axis="XYZ", phase=(1, -1, 1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=2.0), None),
    ("xy_deep_pml_3d",
     dict(axis="XY", phase=(1, -1), extent=0.8,
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2), ((0, 4), (0, 4), (2, 2))),
    # --- the folded PERIODIC rows: fill_folded_far_ghosts_B is LIVE on every one
    ("y_periodic_even_3d", dict(depth=1.2), None),
    ("y_periodic_odd_3d", dict(extent=2.1, depth=1.2), None),
    ("y_periodic_odd_2d", dict(phase=-1, extent=2.1), None),
    ("y_periodic_wall_z_3d",
     dict(boundaries={"z": "metallic"}, depth=1.2), None),
    ("xy_periodic_even_wall_z_3d",
     dict(axis="XY", phase=(1, -1), boundaries={"z": "metallic"}, depth=1.2),
     None),
    ("xy_periodic_odd_wall_z_3d",
     dict(axis="XY", phase=(1, -1), extent=2.1, boundaries={"z": "metallic"},
          depth=1.2), None),
    # A fold of EACH termination on the same grid: x images a far ghost and y does
    # not, so the two code paths are exercised against each other in one kernel.
    ("x_periodic_y_metallic_3d",
     dict(axis="XY", phase=(-1, 1), boundaries={"y": "metallic"}, depth=1.2),
     None),
    # Three folded periodic axes: ONE source thread owns SEVEN ghost cells for one
    # component (two single far planes, three pairs, one triple), which is the
    # largest composition carried_destinations can produce on this family.
    ("xyz_periodic_odd_3d",
     dict(axis="XYZ", phase=(-1, 1, -1), extent=2.1, depth=2.1), None),
    # The periodic twin of xy_deep_pml_3d, and the ONLY row on which the COMPOSITE
    # near/far ghost's coefficient index is observable: a 4-cell high-face layer on
    # a 6-cell folded axis is the first that reaches stored cell 2.
    ("xy_periodic_deep_pml_3d",
     dict(axis="XY", phase=(1, -1), extent=0.8, boundaries={"z": "metallic"},
          depth=1.2), ((0, 4), (0, 4), (2, 2))),
)

#: The case the general mutations are armed on. TWO FOLDED PERIODIC AXES with MIXED
#: parities, an ODD full count and a live ``z`` wall, so every line the shipped
#: kernel can emit is present at once: both parity spellings, a near carry, two far
#: carries, a composite near/far ghost, a composite far/far corner, the top-plane
#: mask, a reflect row that is NOT ``stored - 2``, and the wall clear on the owned
#: cell. ``main`` asserts each of those is present rather than trusting the
#: keywords, because a mutation armed on a line the case never emits reports the
#: defect as uncaught while measuring nothing.
MUTATION_CASE = "xy_periodic_odd_wall_z_3d"

#: THE THIRD CASE THE GENERAL MUTATIONS ARE ARMED ON, and the one this round adds.
#: THREE folded PERIODIC axes, so ONE source thread owns SEVEN ghost cells for each
#: component — two single far planes, the near plane, three pairs and one TRIPLE —
#: which is the largest composition ``carried_destinations`` can produce on this
#: family and the geometry the whole far carry was restructured for.
#:
#: WHY IT NEEDED ITS OWN ROW. This case was byte-verified by the product leg from
#: the day the far carry landed and was never mutation-scored: the general set is
#: armed on :data:`MUTATION_CASE`, whose geometry floor is ``max(owned) >= 3``. A
#: path byte-verified but with no planted defect it is known to catch is weaker
#: evidence than the rest of this file, and the difference is not academic — the
#: three-clause ownership carve-out and the three-factor parity product are lines
#: NO two-fold grid emits, so nothing armed on ``MUTATION_CASE`` can reach them.
#: :func:`triple_shader_edits` arms them, and ``main`` asserts ``max(owned) >= 7``
#: before scoring rather than trusting the keywords.
TRIPLE_MUTATION_CASE = "xyz_periodic_odd_3d"

#: The MIRROR_METALLIC case the general mutations are ALSO armed on. The termination
#: changes which lines exist, so a mutation set scored on one alone would leave the
#: other's kernel unmeasured — the shipped shape before this round.
METALLIC_MUTATION_CASE = "xy_mixed_wall_z_3d"

#: The case leg ``moved_coefficient`` uses for its OBSERVABLE half. Same fold, same
#: parities, same wall — only the absorber reaches stored cell 2.
DEEP_CASE = "xy_deep_pml_3d"

#: The folded PERIODIC twin of :data:`DEEP_CASE`, and the observable half of the
#: COMPOSITE ghost's coefficient prediction. Its fold is periodic, so it carries the
#: near/far composite the metallic deep case cannot.
DEEP_PERIODIC_CASE = "xy_periodic_deep_pml_3d"

#: Leg ``reflect_row``'s pair. Same fold, same parities, same wall; only the FULL
#: COUNT'S PARITY differs, which is the whole of what ``n_full - stored + 2`` says
#: and a fixed ``n - 2`` does not.
REFLECT_ODD_CASE = "xy_periodic_odd_wall_z_3d"
REFLECT_EVEN_CASE = "xy_periodic_even_wall_z_3d"

#: The row leg ``carry`` builds: a folded PERIODIC axis, where
#: ``fill_folded_far_ghosts_B`` is live inside the seam.
FAR_CARRY_CASE: Dict[str, Any] = dict(depth=1.2)


class _MagneticSource:
    """The smallest thing the source clause can read: a declared field type.

    Not an engine source. The predicate asks ``field_type`` and nothing else
    (driver.py:3283-3284 is what makes that the question), so a stub is the honest
    fixture — building a real ``GaussianSource`` here would test the source class.
    """

    field_type = "B"


class _ElectricSource:
    """The polarity that must NOT disqualify this pair — the whole matrix asymmetry."""

    field_type = "D"


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
          pml_faces: Optional[Any] = None,
          value_class: str = values.UNIFORM, scale: float = 1.0,
          ) -> Tuple[Any, Any]:
    """One seeded engine. Called twice (or three times) per case, identically.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the folded family's own gate uses, so the fixture this certifies is
    the fixture that family was certified on. Only the field state is reseeded here,
    and identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.

    ``pml_faces`` REPLACES the matrix's absorber with an explicitly requested one.
    Exactly one case uses it and the reason is measured rather than aesthetic: the
    matrix gives a folded axis a 2-cell high-face layer, and on such a layer
    ``kps[0] == kps[2]`` to the bit, which makes this family's moved coefficient
    index unobservable. The grid, the material, the fold and the wall are the
    matrix's; only the layer depth is this gate's.
    """
    fields, pml = matrix.folded(**dict(keywords))
    if pml_faces is not None:
        pml = PML(grid=fields.grid, thickness=pml_faces)
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
    operational: without it the next device launch would read the mirror's stale
    bytes, which is smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns FOUR passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``fill_symmetry_bc_B`` and ``zero_metal_B`` running on the host on
    top of the copies the kernel already carried — which, for the fill, is IDEMPOTENT
    and would hide a dropped carry.
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
             pml_faces: Optional[Any] = None,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(keywords, seed, pml_faces, value_class)
    actual, actual_pml = build(keywords, seed, pml_faces, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_folded_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_folded_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": "the folded fused magnetic pair was refused",
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
    # move every compared array. The +-0 lattice answers to the SIGN floor instead:
    # the REFERENCE OUTPUT must carry BOTH bit patterns, which is what makes a
    # uint32 compare of a zero state discriminating rather than decorative. It is
    # not exempt from a floor, it has a different one — and on this family the sign
    # is the whole content of the mirror fill, so the class is a real second
    # comparison of every parity the carry composes.
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
        "carried_far_ghost_axes": [list(axes) for axes in plan.carried_far_axes],
        # How many ghost cells ONE source thread owns, per component. 1 on a
        # metallic fold, up to 7 on a three-axis periodic one; recorded so a reader
        # can see which rows exercise the composition at all rather than inferring
        # it from the keywords.
        "ghost_cells_per_source_thread": [
            len(family.carried_destinations(near, far))
            for near, far in zip(plan.carried_axes, plan.carried_far_axes)],
        "reflect_rows": list(plan.reflect),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         pml_faces: Optional[Any] = None) -> Dict[str, Any]:
    """The SEPARATE certified folded products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path, the two ALREADY CERTIFIED folded Metal
    products stepping the same seam as separate dispatches with the mirror fill and
    the wall clear left on the HOST between them, and the fused product. All three
    must agree word for word at every complete step, and the DISPATCH COUNTS and the
    surviving in-seam host passes are recorded on both sides — which is the only
    place the difference between the compositions shows up at all, since a correct
    fusion is byte-neutral by construction.

    The separate side is not a strawman: those two plans are exactly what
    ``plan_step`` composes today for this configuration.
    """
    reference, reference_pml = build(keywords, seed, pml_faces)
    separate, separate_pml = build(keywords, seed, pml_faces)
    fused, fused_pml = build(keywords, seed, pml_faces)

    separate_residency = Residency()
    curl = symmetry.plan_folded_pml_curl(separate, separate_pml, "step_B",
                                         separate_residency)
    magnetic = symmetry.plan_folded_constitutive(separate, separate_pml, "H",
                                                 separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_folded_fused_magnetic_pair(
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

    # The host passes each composition leaves on the array path INSIDE the seam. The
    # fused side carries the mirror fill and the wall clear in the kernel; the
    # separate side does not, so on every folded run it pays a host round trip the
    # fused side does not.
    seam = ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B")
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


def needle(source: str, old: str, new: str, occurrences: int = 1) -> str:
    """Replace EXACTLY ``occurrences`` matches and REFUSE anything else.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure mode a mutation leg cannot see from
    its own result. AN AMBIGUOUS NEEDLE IS THE SAME FAILURE WEARING A DISGUISE, and
    it became reachable the moment the far carry started emitting several ghost
    blocks per component: a needle that matched two of them would edit an arbitrary
    one and the row's label would name a line the run did not touch. So the match
    COUNT is asserted, not just its presence.
    """
    found = source.count(old)
    if found != occurrences:
        raise AssertionError(
            f"mutation needle matches {found} times, expected {occurrences}: "
            f"{old!r}")
    mutated = source.replace(old, new, occurrences)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


def shader_edit_anchors(periodic: bool) -> Dict[str, Tuple[str, str]]:
    """The source defects as ``name -> (old, new)``, each anchored ONCE.

    SPLIT FROM :func:`shader_edits` SO ONE TABLE SERVES TWO CASES. The
    three-fold case (:data:`TRIPLE_MUTATION_CASE`) carries most of these rows
    and structurally cannot carry five of them, and a second hand-written copy
    of the anchors for that case would be a transcription that drifts. The
    anchors live here; :func:`shader_edits` applies them and
    :func:`carried_shader_edits` applies the subset, both through
    :func:`needle`, so an anchor that stops matching still RAISES rather than
    quietly arming nothing.

    ``periodic`` says which fold termination the base source was emitted for, and it
    changes the SET rather than only the anchors: the top-plane mask, the far carry,
    the composite near/far ghost and the far/far corner are lines a MIRROR_METALLIC
    kernel does not contain at all, so arming them there would raise in
    :func:`needle` rather than measure anything. Every row below is scored on a case
    whose geometry emits it — ``main`` asserts that geometry rather than assuming it,
    which is the check that keeps a mutation from reporting UNCAUGHT because the
    line it edited is in a branch the case never enters.

    THE MUTATION CASE'S GEOMETRY, and it is why each of these is reachable. Two
    folded axes with phases (+1, -1) and z a live metallic wall:

      metallic termination  component 0 (Bx) images a NEAR ghost on x with parity
                            +1; component 1 (By) a near ghost on y with parity -1;
                            component 2 (Bz) images nothing and is the one the wall
                            clear reaches;
      periodic termination  every one of those, PLUS component 0's far ghost on y
                            (+1) and their composite; component 1's far ghost on x
                            (-1) and its composite; and component 2's two far ghosts
                            (x: -1, y: +1) and the corner carrying their product.
    """
    edits = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            ("float o0_src = v0;", "float o0_src = curl0;"),
        # THE NEAR FILL, DROPPED. The ghost cell then keeps last step's displacement
        # and last step's H — what a fused pair that ignored `fill_symmetry_bc_B`
        # would produce on every folded grid.
        "near_fill_dropped":
            ("        if (i == 2) {", "        if (false) {"),
        # THE PARITY, DROPPED on the odd plane. `-v1` is component 1's near ghost.
        "near_fold_parity_dropped":
            ("float g1_n_v = -v1;", "float g1_n_v = v1;"),
        # THE SOURCE ROW. MEEP's `little_owned_corner0` puts the image at stored cell
        # 2, never cell 1 (stepping._mirror_source, MIRROR_SOURCE_INDEX).
        "near_ghost_images_the_wrong_source_row":
            ("int g1_n_i = ii - 2 * nzi;", "int g1_n_i = ii - 1 * nzi;"),
        # THE CELL-0 MASK, on the B DIAGONAL (`at_x` for component 0, not the D
        # side's off-diagonal pair).
        "ownership_mask_dropped":
            ("    curl0 = at_x ? 0.0f : curl0;\n", ""),
        # THE WALL CLEAR on the owned cell.
        "zero_metal_dropped":
            ("v2 = at_z ? 0.0f : v2;\n", ""),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            ("dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        # THE DIRECTION. `step_B` shifts UP and `step_D` shifts DOWN
        # (SUB_STEPS['step_B']['backward'] == 0); this is the D curl in the B seam.
        "curl_shift_is_backward":
            ("int si = i + 1, sj = j + 1, sk = k + 1;",
                   "int si = i - 1, sj = j - 1, sk = k - 1;"),
        "fu_store_dropped":
            ("    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "fw_store_dropped":
            ("w0[ii] = o0_src;", "// stale f_w_Hx"),
        "constitutive_accumulations_reversed":
            ("o0_acc = o0_acc + kp_0 * o0_src;\n"
             "        o0_acc = o0_acc - km_0 * o0_prev;",
             "o0_acc = o0_acc - km_0 * o0_src;\n"
             "        o0_acc = o0_acc + kp_0 * o0_prev;"),
        # THE NEAR GHOST'S OWN CONSTITUTIVE UPDATE, in three separable pieces: the
        # workspace copy, the H accumulation and the flux store. `update_H` reads the
        # ghost cell the fill just wrote, so all three are live work.
        "near_ghost_fw_store_dropped":
            ("w0[g0_n_i] = g0_n_src;", "// stale ghost f_w_Hx"),
        "near_ghost_constitutive_dropped":
            ("            h0[g0_n_i] = g0_n_acc;\n", ""),
        "near_ghost_flux_store_dropped":
            ("            f0[g0_n_i] = g0_n_v;\n", ""),
    }
    if not periodic:
        # A MIRROR_METALLIC fold gives component 0 exactly one destination plane, so
        # its ownership guard is the near one alone.
        edits["destination_thread_keeps_its_displacement"] = (
            "    if (!(i == 0)) {", "    if (true) {")
        return edits

    edits.update({
        # THE OWNERSHIP CARVE-OUT, DROPPED, on a component that is a destination on
        # BOTH a near and a far plane. The destination thread then forms `v` from a
        # word the source thread writes and stores its own displacement over the
        # imaged ghost — the race the restructure exists to remove.
        "destination_thread_keeps_its_displacement":
            ("    if (!(i == 0 || last_y)) {", "    if (true) {"),
        # THE FAR-ONLY CARVE-OUT: component 2 is a destination on two far planes and
        # on no near one, which is the guard shape the near carry alone never makes.
        "far_destination_thread_keeps_its_displacement":
            ("    if (!(last_x || last_y)) {", "    if (true) {"),
        # THE TOP-PLANE MASK. `_mask_non_owned_cells`' shift-1 arm; emitted only on
        # a folded PERIODIC axis, and the block the shipped family used to retire by
        # refusing that termination outright.
        "top_plane_mask_dropped":
            ("    curl0 = last_y ? 0.0f : curl0;\n", ""),
        # THE FAR CARRY'S THREE WRITES, separately.
        "far_ghost_flux_store_dropped":
            ("            f0[g0_j_i] = g0_j_v;\n", ""),
        "far_ghost_fw_store_dropped":
            ("w0[g0_j_i] = g0_j_src;", "// stale far ghost f_w_Hx"),
        "far_ghost_constitutive_dropped":
            ("            h0[g0_j_i] = g0_j_acc;\n", ""),
        # THE FAR PARITY. `mirror_parity` is -phase for a shift-1 component, so
        # component 2's x ghost is -1 where the plane declared +1 — the sign a carry
        # that reused the NEAR fill's `+phase` rule would get backwards.
        "far_fold_parity_dropped":
            ("float g2_i_v = -v2;", "float g2_i_v = v2;"),
        # THE COMPOSITE NEAR/FAR GHOST: the cell at stored 0 on the near axis AND at
        # the top plane on the far one. A carry that wrote each fill's plane and
        # stopped would leave this cell holding only the far image of an unfilled
        # row.
        "near_far_composite_dropped":
            ("            f0[g0_jn_i] = g0_jn_v;\n", ""),
        # THE FAR/FAR CORNER, and the PRODUCT of two parities it carries. Neither
        # exists on this family until the far carry does: a B component is a NEAR
        # destination on one axis only.
        "far_corner_dropped":
            ("            f2[g2_ij_i] = g2_ij_v;\n", ""),
        "far_corner_takes_one_parity":
            ("float g2_ij_v = -v2;", "float g2_ij_v = v2;"),
    })
    return edits


def shader_edits(base: str, periodic: bool) -> Dict[str, str]:
    """Apply every anchor for this fold termination to ``base``."""
    return {name: needle(base, old, new)
            for name, (old, new) in shader_edit_anchors(periodic).items()}


#: General edits :func:`shader_edits` arms on the two-fold mutation case that the
#: THREE-FOLD case cannot carry, each with the structural reason. These are scored
#: NULL CONFIRMED on the triple case rather than left out silently: an edit missing
#: from a mutation table and an edit that could not be armed look identical in an
#: artifact, and only one of them is a measurement.
#:
#: The parity spellings are the interesting pair. A near or far ghost's sign is
#: baked into the SOURCE, so ``-v1`` on one grid is ``v1`` on another with the
#: opposite declared phase; arming the two-fold case's spelling here would raise in
#: :func:`needle` rather than measure anything, and the triple set carries its own
#: parity edits on the lines this case does emit.
TRIPLE_STRUCTURALLY_ABSENT: Dict[str, str] = {
    "near_fold_parity_dropped":
        "component 1's near ghost is +v1 at this case's phases, not -v1; the "
        "triple set arms triple_composite_takes_two_of_three_parities and "
        "far_corner_takes_one_parity on the parity lines this case does emit",
    "far_fold_parity_dropped":
        "component 2's x far ghost is +v2 at this case's phases, not -v2",
    "zero_metal_dropped":
        "no axis is walled: all three are folded, and stepping._zero_metal "
        "excludes a folded axis by construction, so the kernel emits no wall "
        "line at all (`no walled axis clears this target`)",
    "destination_thread_keeps_its_displacement":
        "the two-clause carve-out `!(i == 0 || last_y)` does not exist here; the "
        "three-fold geometry emits `!(i == 0 || last_y || last_z)`, and the "
        "triple set arms BOTH the total drop and the exact two-clause form",
    "far_destination_thread_keeps_its_displacement":
        "`!(last_x || last_y)` is the FAR-ONLY guard a component with no near "
        "ghost gets; every component here has one, so no such guard is emitted",
}


def triple_shader_edits(base: str) -> Dict[str, str]:
    """The defects that exist ONLY where one source thread owns SEVEN ghost cells.

    WHY THIS SET EXISTS. ``xyz_periodic_odd_3d`` is byte-verified by the product
    leg and was never mutation-scored: the general set is armed on
    :data:`MUTATION_CASE`, whose geometry floor is ``max(owned) >= 3``. A path with
    no planted defect it is known to catch is weaker evidence than the rest of this
    file, and the seventh destination is not a bigger version of the third — it is
    a different line with a different closed form.

    WHAT IS STRUCTURALLY NEW AT SEVEN, and every row below is one of these:

      THE TRIPLE COMPOSITE   the cell at the top plane of BOTH far axes AND at
                             stored 0 on the near one. Its parity is the PRODUCT OF
                             THREE factors, which no two-fold grid can form, and
                             its index carries three displacement terms;
      THE THREE-CLAUSE       a component is a destination on a near plane and TWO
      CARVE-OUT              far planes, so its ownership guard has three clauses.
                             The two-clause form is exactly what a kernel
                             generalised from the two-fold case would emit, and it
                             is armed here as its own row;
      THE SECOND MASK        a component with two far axes gets TWO top-plane mask
                             lines. The two-fold case emits one per component, so a
                             dropped second mask is invisible there.

    Anchored ONCE each, and :func:`needle` refuses anything else — a needle that
    matched two of the seven ghost blocks would edit an arbitrary one and the row's
    label would name a line the run did not touch.
    """
    return {
        # --- THE SEVENTH DESTINATION'S THREE WRITES, separately. `update_H` reads
        # the cell the fill wrote, so all three are live work at the triple corner.
        "triple_composite_flux_store_dropped":
            needle(base, "            f0[g0_jkn_i] = g0_jkn_v;\n", ""),
        "triple_composite_fw_store_dropped":
            needle(base, "w0[g0_jkn_i] = g0_jkn_src;",
                   "// stale triple-composite ghost f_w_Hx"),
        "triple_composite_constitutive_dropped":
            needle(base, "            h0[g0_jkn_i] = g0_jkn_acc;\n", ""),
        # --- THE PRODUCT OF THREE PARITIES. `carried_destinations` composes one
        # factor per far axis at the top plane and one for the near axis, so this
        # cell carries phase_y_far * phase_z_far * phase_x_near. Flipping the sign
        # is what dropping ANY ONE of the three factors looks like, and there is no
        # grid with fewer than three folded axes on which it is observable.
        "triple_composite_takes_two_of_three_parities":
            needle(base, "float g0_jkn_v = v0;", "float g0_jkn_v = -v0;"),
        # --- THE THREE-TERM INDEX, with the near displacement dropped. The cell is
        # then the far/far CORNER, which another block already writes: the triple
        # ghost goes unwritten and the corner is written twice with different
        # values.
        "triple_composite_drops_the_near_displacement":
            needle(base,
                   "int g0_jkn_i = ii + ((nyi - 1) - reflect_y) * nzi "
                   "+ ((nzi - 1) - reflect_z) * 1 - 2 * nyz;",
                   "int g0_jkn_i = ii + ((nyi - 1) - reflect_y) * nzi "
                   "+ ((nzi - 1) - reflect_z) * 1;"),
        # --- THE TRIPLE GUARD, with the near clause dropped. The triple ghost is
        # then written by every thread along the far/far corner line rather than by
        # the one at stored 2 on the near axis: the race the ownership rule exists
        # to remove, planted at the one destination only this case has.
        "triple_composite_guard_drops_the_near_clause":
            needle(base, "if (j == reflect_y && k == reflect_z && i == 2) {",
                   "if (j == reflect_y && k == reflect_z) {"),
        # --- THE THREE-CLAUSE OWNERSHIP CARVE-OUT, dropped outright.
        "three_clause_carve_out_dropped":
            needle(base, "    if (!(i == 0 || last_y || last_z)) {",
                   "    if (true) {"),
        # --- THE SAME CARVE-OUT, REDUCED TO THE TWO-FOLD CASE'S EXACT FORM. This
        # is the defect a kernel generalised from two folded axes would carry, and
        # the reason this case had to be scored: on MUTATION_CASE the two-clause
        # guard IS the shipped text, so nothing there can catch it.
        "three_clause_carve_out_reduced_to_the_two_fold_form":
            needle(base, "    if (!(i == 0 || last_y || last_z)) {",
                   "    if (!(i == 0 || last_y)) {"),
        # --- THE SECOND TOP-PLANE MASK. `_mask_non_owned_cells` emits one line per
        # (target, folded periodic axis whose Yee shift is 1), so a component with
        # two far axes gets two. Dropping the first is the general set's row; this
        # drops the SECOND, which does not exist below three folded axes.
        "second_top_plane_mask_dropped":
            needle(base, "    curl0 = last_z ? 0.0f : curl0;\n", ""),
        # NOTE ON WHAT IS NOT HERE. ``far_corner_takes_one_parity`` and the rest of
        # the far/far corner rows are in the GENERAL table and still anchor on this
        # source, so :func:`carried_shader_edits` arms them and ``main`` asserts the
        # two sets are disjoint. Repeating one here would silently overwrite the
        # other and inflate the armed count.
    }


def carried_shader_edits(base: str) -> Dict[str, str]:
    """The general periodic anchors, minus the rows this case cannot carry.

    THE SAME TABLE THE TWO-FOLD CASE USES, applied to the three-fold source, so
    the seven-destination kernel is scored on the whole armed set and not only on
    its own new lines. The exemptions come from :data:`TRIPLE_STRUCTURALLY_ABSENT`
    and are cross-checked against the table in both directions: an exemption naming
    an edit that no longer exists is a drifted excuse, and an anchor that stops
    matching still raises inside :func:`needle` rather than arming nothing.
    """
    anchors = shader_edit_anchors(True)
    missing = set(TRIPLE_STRUCTURALLY_ABSENT) - set(anchors)
    if missing:
        raise AssertionError(
            f"TRIPLE_STRUCTURALLY_ABSENT names edits shader_edit_anchors no "
            f"longer emits: {sorted(missing)}; the exemption table has drifted "
            f"from the mutation set and would be excusing rows that do not exist")
    return {name: needle(base, old, new)
            for name, (old, new) in anchors.items()
            if name not in TRIPLE_STRUCTURALLY_ABSENT}


#: THE ONE MUTATION WHOSE CATCH IS CONDITIONAL, and the condition is measured.
#: Reusing the source thread's coefficient pair for the imaged NEAR ghost is exactly
#: the shortcut :mod:`.folded_fused_pair` is entitled to and this family is not. It
#: is observable only where the two entries differ; leg ``moved_coefficient``
#: predicts that from the coefficient vectors and requires the prediction to hold.
#: TWO ghost cells reload it, so TWO edits are armed: the pure near ghost, and the
#: COMPOSITE near/far ghost, whose near half still moves the indexed axis to 0. The
#: composite one is the far carry's own row and its observable case had to be built
#: (:data:`DEEP_PERIODIC_CASE`) — the metallic deep case has no composite at all.
MOVED_COEFFICIENT_EDITS: Dict[str, Tuple[Tuple[str, str], Tuple[str, ...]]] = {
    "near_ghost_reuses_the_source_coefficient": (
        ("float g0_n_kp = kp0[0], g0_n_km = km0[0];",
         "float g0_n_kp = kp_0, g0_n_km = km_0;"),
        ("MUTATION_CASE", "DEEP_CASE")),
    "composite_ghost_reuses_the_source_coefficient": (
        ("float g0_jn_kp = kp0[0], g0_jn_km = km0[0];",
         "float g0_jn_kp = kp_0, g0_jn_km = km_0;"),
        ("MUTATION_CASE", "DEEP_PERIODIC_CASE")),
}

#: THE MUTATION THAT MEASURES A LINE THIS FAMILY DELIBERATELY DOES NOT EMIT.
#: A far ghost keeps its source thread's constitutive pair because it moves along an
#: axis that is NOT the one ``update_H`` indexes on. That is a claim about
#: ``fields.IYEE_SHIFTS`` and ``stepping.H_CONSTITUTIVE_TERMS``, not an observation,
#: so the opposite — reloading at stored index 0, which is what the NEAR ghost
#: correctly does — is armed and must be CAUGHT. Uncaught would mean the two
#: coefficient entries are the same word on this fixture and the claim is untested.
FAR_COEFFICIENT_EDIT = ("g0_j_acc = g0_j_acc + kp_0 * g0_j_src;",
                        "g0_j_acc = g0_j_acc + kp0[0] * g0_j_src;")

#: THE DOCUMENTED TRAP IN ``_far_reflect_rows``, ARMED. Its docstring says in
#: capitals that the image row is NOT always ``n - 2``: both full-count parities put
#: the ghost one half-cell above ``big_corner``, but ``big_corner`` IS the second
#: mirror at an even count and one half-cell above it at an odd one, so the row is
#: ``stored - 2`` even and ``stored - 3`` odd. This edit hard-codes the even answer.
#: It is a NO-OP on an even-count grid and a whole cell wrong on an odd one, which is
#: what leg ``reflect_row`` predicts from the grid and then measures.
REFLECT_ROW_EDIT = ("int g0_j_i = ii + ((nyi - 1) - reflect_y) * nzi;",
                    "int g0_j_i = ii + 1 * nzi;")


def compile_edits(edits: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    return {name: {shaders.CONTRACT_OFF:
                   compile_source(source).folded_fused_magnetic_pair_step}
            for name, source in edits.items()}


#: Where each binding group starts in the plan's argument tuple. Spelled once, so the
#: two host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(15, 21))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(21, 27))


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_h_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the HALF-INTEGER constitutive coefficients to the H half.

    ``update_H`` takes ``kps_a``/``kms_a`` and ``update_E`` takes ``kps_a_h``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from, so this is a silent half-cell error in the absorber
    profile — converged, smooth and wrong — and no shader mutation can reach it.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

    ``step_B`` reads half-integer positions and ``step_D`` integer ones
    (SUB_STEPS' ``suffix``). Getting it backwards is a half-cell error, not a crash.
    """
    _rebind(plan, CURL_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
    "curl_takes_the_integer_lattice": swap_curl_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_subnormal_ladder(keywords: Mapping[str, Any], seed: int,
                         pml_faces: Optional[Any] = None, budget: int = 4,
                         ) -> Dict[str, Any]:
    """THE EMPTY CENSUS, TURNED INTO A MEASUREMENT.

    Every product row in this file reports ``reference_subnormals: 0`` and
    ``subnormal_free: true`` under policy ``flush``. Read alone that is a SILENCE:
    it cannot distinguish "the flush policy was exercised and changed nothing" from
    "the band was never entered, so the policy was never asked anything". This leg
    is the difference — not a second byte comparison, which MPS cannot give, but
    the precondition FIRING.

    THE ORACLE ALONE runs here, seeded at each rung of
    :data:`metal_value_classes.PRECONDITION_SCALES` and stepped through the same
    live pass list, with the same census the product rows report taken after every
    step. Nothing is compared against the device and the artifact says so:
    ``banded_rows_were_byte_compared`` is False.

    WHY THE BANDED RUNGS ARE NOT COMPARED, measured rather than asserted. On this
    host the shipped kernel and the array path are byte-identical at 1e0, 1e-25 and
    1e-30 with an empty census, and at 1e-34 the census fires and the bytes diverge
    — three decades of headroom, then a CLIFF. MPS flushes float32 subnormals
    natively with no lever and NumPy keeps, so a subnormal in the reference state
    IS a divergence and no kernel can be right about it. What this family claims is
    byte identity under a CHECKED subnormal-free precondition, and this leg is that
    check being shown to work.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in values.PRECONDITION_SCALES:
        reference, reference_pml = build(keywords, seed, pml_faces,
                                         values.UNIFORM, scale)
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


def legal_specialisations() -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...],
                                           Tuple[bool, ...]], ...]:
    """Every ``(codes, phases, walls)`` triple a REAL GRID can hand this family.

    THE LEGALITY RULE, READ OFF THE PREDICATES RATHER THAN LISTED. Three clauses,
    each of which is a function in the package and not a convention here:

    * ``phases[a]`` is ``grid.mirror_phase(a) or 0``, so it is +1 or -1 exactly on
      a folded axis and 0 on every other — the same expression
      :func:`_specialisation` evaluates, and the emitter refuses a folded axis
      whose phase is 0 by name;
    * ``walls[a]`` is ``zero_metal_axes``' own predicate, ``has_metallic and
      is_metallic(a) and not is_mirrored(a)`` (stepping.py:2284-2286). A
      MIRROR_METALLIC axis is metallic AND mirrored, so it is never walled, and a
      METALLIC axis is metallic, unmirrored and makes ``has_metallic`` true by
      itself — so ``walls[a]`` is DERIVED, exactly ``codes[a] == METALLIC``. It is
      not a free coordinate, and the emitter refuses folded-and-walled by name;
    * at least one axis is folded, or this product does not exist and the grid
      belongs to :mod:`.fused_magnetic_pair`.

    That leaves six options per axis — PERIODIC, METALLIC, and MIRROR_METALLIC and
    MIRROR_PERIODIC at each of two parities — minus the unfolded corner, so 6**3 -
    2**3 = 208. The number is COMPUTED here rather than written down, so a code
    value added to :mod:`.symmetry` widens this sweep instead of silently leaving
    a configuration unmeasured.
    """
    per_axis: List[Tuple[int, int, bool]] = []
    for code in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
                 symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC):
        wall = code == symmetry.CODE_METALLIC
        if code in symmetry.MIRROR_CODES:
            per_axis.extend((code, phase, wall) for phase in (1, -1))
        else:
            per_axis.append((code, 0, wall))
    out: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[bool, ...]]] = []
    for triple in itertools.product(per_axis, repeat=3):
        codes = tuple(item[0] for item in triple)
        if not any(code in symmetry.MIRROR_CODES for code in codes):
            continue
        out.append((codes, tuple(item[1] for item in triple),
                    tuple(item[2] for item in triple)))
    return tuple(out)


#: The binding shape the platform ceiling forces, as ``(count, highest index)``.
#: Read off the emitted text of every legal configuration, never assumed.
EXPECTED_BINDING_SHAPE = (28, 27)


def leg_binding_ceiling() -> Dict[str, Any]:
    """32 separate bindings must FAIL; the packed 28 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. The signature is :mod:`.fused_magnetic_pair`'s exactly — the
    fold adds no argument — so the REFUTED source is imported from that module rather
    than re-spelled, and re-measured here on this tree.

    THE SWEEP, AND WHY IT IS THE WHOLE SPACE. This leg used to compile and launch
    ONE configuration and read back nine packed fields, and then the artifact said
    "28 bindings" as though that had been established for the family. It had been
    established for one kernel. A verifier said so, and the fix is the strong form:
    :func:`legal_specialisations` enumerates every ``(codes, phases, walls)`` triple
    a real grid can produce, and each one's SHIPPED source is emitted, its
    ``[[buffer(n)]]`` declarations counted, and the kernel COMPILED. The compile is
    the part that matters — a signature over the ceiling is a compile failure on
    this toolchain, which is exactly what the 32-binding half measures — so the
    claim "28 across the family" is now 208 compiled kernels rather than one.

    THE LAUNCH HALF is widened too, over records rather than configurations: the
    packer is the same function for every specialisation, so what varies there is
    its INPUT. ``reflect`` carries a ``-1`` sentinel on an unfolded axis and a real
    row on a folded one, and a signed field read back through an unsigned struct
    member would return 4294967295 — a wrong reflect row is a plane of wrong
    values, not a crash — so the sentinel is one of the records.
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

    # The packed signature, compiled AND launched, with every field read back. 27
    # pointers plus one packed Params& at buffer 27 is the shipped shape.
    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(family.PACKED_BINDINGS - 1))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        "                int rx; int ry; int rz; };", "",
        "kernel void packed_probe(", pointers,
        f"    constant Params&    prm     [[buffer({family.PACKED_BINDINGS - 1})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;",
        "    b5[idx] = float(prm.rx);", "    b6[idx] = float(prm.ry);",
        "    b7[idx] = float(prm.rz);",
        f"    b8[idx] = b{family.PACKED_BINDINGS - 2}[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(family.PACKED_BINDINGS - 1)]
    buffers[-1] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    # EVERY RECORD comes from the PLAN BUILDER's own packer, so the field the kernel
    # reads is the field the plan writes rather than a hand-built twin. The records
    # span what the packer's inputs actually do: a mixed reflect triple with the
    # unfolded sentinel, all three axes folded, all three unfolded, and the largest
    # extent this family's own cases reach.
    launch_records: Tuple[Tuple[str, Tuple[int, int, int], float,
                                Tuple[int, int, int]], ...] = (
        ("mixed_with_unfolded_sentinel", (3, 4, 5), 0.125, (10, -1, 4)),
        ("all_three_folded", (7, 6, 5), 0.35, (5, 4, 3)),
        ("no_folded_axis", (4, 4, 4), 0.5, (-1, -1, -1)),
        ("odd_full_count_rows", (13, 13, 13), 0.35, (10, 10, 10)),
    )
    launch_rows: List[Dict[str, Any]] = []
    for label, shape, dtdx, reflect in launch_records:
        params = family._params_tensor(shape, dtdx, reflect, "mps")
        function(*buffers, params)
        torch.mps.synchronize()
        read = [buffers[index].cpu().numpy() for index in range(9)]
        expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                    float(shape[0] * shape[1] * shape[2]),
                    float(np.float32(dtdx)), float(reflect[0]),
                    float(reflect[1]), float(reflect[2]), 8.0)
        correct = all(bool(np.all(read[index] == expected[index]))
                      for index in range(9))
        launch_rows.append({
            "label": label, "shape": list(shape), "dtdx": dtdx,
            "reflect": list(reflect),
            "packed_fields_read_back": [float(value[0]) for value in read],
            "packed_fields_expected": [float(value) for value in expected],
            "packed_fields_correct": correct})
        log(f"    binding_ceiling launch {label} nine_fields_correct={correct}")
    fields_ok = all(entry["packed_fields_correct"] for entry in launch_rows)

    # THE SWEEP. Every legal configuration's SHIPPED source, emitted, its bindings
    # counted off the text, and the kernel compiled. One failure anywhere is the
    # family's shape claim failing, so the first is reported by name rather than
    # only counted.
    shapes: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []
    specialisations = legal_specialisations()
    for codes, phases, walls in specialisations:
        try:
            emitted = family.folded_fused_magnetic_pair_source(codes, phases, walls)
        except Exception as exc:  # noqa: BLE001 - a refusal here is a legality bug
            failures.append({"codes": list(codes), "phases": list(phases),
                             "walls": list(walls), "stage": "emit",
                             "error": str(exc).splitlines()[0]})
            continue
        bindings = [int(match) for match
                    in re.findall(r"\[\[buffer\((\d+)\)\]\]", emitted)]
        observed = (len(bindings), max(bindings) if bindings else -1)
        shapes[str(list(observed))] = shapes.get(str(list(observed)), 0) + 1
        if observed != EXPECTED_BINDING_SHAPE:
            failures.append({"codes": list(codes), "phases": list(phases),
                             "walls": list(walls), "stage": "count",
                             "binding_shape": list(observed)})
            continue
        try:
            compile_source(emitted)
        except Exception as exc:  # noqa: BLE001 - a compile failure IS the ceiling
            failures.append({"codes": list(codes), "phases": list(phases),
                             "walls": list(walls), "stage": "compile",
                             "error": str(exc).splitlines()[0]})
    log(f"    binding_ceiling swept {len(specialisations)} legal configurations, "
        f"shapes={shapes} failures={len(failures)}")
    row.update(
        packed_launched=True,
        packed_launch_records=launch_rows,
        packed_fields_correct=fields_ok,
        configurations_swept=len(specialisations),
        binding_shapes_observed=shapes,
        expected_binding_shape=list(EXPECTED_BINDING_SHAPE),
        sweep_failures=failures[:8],
        sweep_failure_count=len(failures),
        every_legal_configuration_compiles=not failures,
        passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok
                    and specialisations and not failures))
    return row


def leg_transcription(codes: Sequence[int]) -> Dict[str, Any]:
    """Both halves must be the certified emitters' OWN TEXT, not a similar one.

    A construction is a hypothesis until something compares the strings. Two
    comparisons:

    * the lifted curl head must be a VERBATIM PREFIX of
      ``symmetry.folded_curl_source(codes, backward=False)``'s body — so the ghost
      gather, the curl grouping and the B-diagonal cell-0 mask in the fused kernel
      are character-for-character the certified folded curl's;
    * this family's parameterised constitutive statements, rendered with the
      certified spellings, must equal ``shaders.constitutive_source('H')``'s own
      statements. That is the check that keeps a template which is re-emitted at two
      flat indices from drifting away from the body it stands for.
    """
    rows: Dict[str, Any] = {"modes": {}}
    ok = True
    for mode in shaders.CONTRACT_MODES:
        head = family.certified_curl_head(codes, mode)
        certified = symmetry.folded_curl_source(codes, family.BACKWARD, mode)
        body = certified.split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
        head_ok = body.startswith(head) and bool(head.strip())
        transcription = family.constitutive_transcription(mode)
        statements = family.certified_curl_statements(codes, mode)
        recurrence_ok = all(
            all(line in body for line in triple)
            for triple in statements["recurrence"])
        rows["modes"][mode] = {
            "curl_head_is_a_verbatim_prefix": head_ok,
            "curl_head_lines": len(head.splitlines()),
            "recurrence_statements_are_lifted": recurrence_ok,
            "constitutive_identical": transcription["identical"],
            "constitutive_statements_per_component":
                [len(block) for block in transcription["certified"]],
        }
        ok = ok and head_ok and recurrence_ok and transcription["identical"]
    rows["passed"] = bool(ok)
    return rows


class _StubGrid:
    """The smallest grid :func:`family._far_carry_reasons` can read, and it LIES.

    Not an engine grid: the clauses under test are arithmetic on ``shape``,
    ``stored_cells``, ``owned_cells`` and ``shape_full``, and building a real
    ``Grid`` that violates them is not possible — ``Grid`` derives the stored extent
    from the fold, so a bad reflect row cannot be requested through it. The refusals
    still have to be measured, because on this family a violated row is an
    OUT-OF-RANGE WRITE issued by a thread that owns neither cell, so a stub that
    answers the four questions is the honest fixture.
    """

    cylindrical = False

    def __init__(self, shape: Sequence[int], full: Sequence[int],
                 mirrored: Sequence[bool]) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.shape_full = tuple(int(n) for n in full)
        self._mirrored = tuple(bool(v) for v in mirrored)

    def is_mirrored(self, axis: int) -> bool:
        return self._mirrored[axis]

    def is_metallic(self, axis: int) -> bool:
        return False

    def is_axis(self, axis: int) -> bool:
        return False

    def has_symmetry(self) -> bool:
        return any(self._mirrored)

    def mirror_phase(self, axis: int) -> Optional[int]:
        return 1 if self._mirrored[axis] else None

    def stored_cells(self, axis: int) -> int:
        return self.shape[axis]

    def owned_cells(self, axis: int) -> int:
        # One less on every mirrored axis, which is what makes _stored_past_owned
        # True there — the definition of a folded PERIODIC axis.
        return self.shape[axis] - (1 if self._mirrored[axis] else 0)


#: The deposit cases' fixtures and the component each drives. THE COMPONENT IS THE
#: LEVER: which fill images a deposit is decided by the DEPOSITED component's Yee shift
#: on the folded axis, and this family's shifts are the D family's SWAPPED -- on a Y
#: fold ``Bx``/``By`` carry shift 0 and take the NEAR fill while ``Bz`` carries shift 1
#: and takes the FAR ghost. A sweep that varied only the grid would run one of the two
#: carries three times.
DEPOSIT_CASES: Tuple[Tuple[str, Dict[str, Any], Optional[Any], str], ...] = (
    ("periodic_y_near_Hx", dict(depth=1.2), None, "Hx"),
    ("periodic_y_far_Hz", dict(depth=1.2), None, "Hz"),
    ("metallic_y_near_Hy", dict(boundaries={"y": "metallic"}, depth=1.2), None, "Hy"),
    ("xy_periodic_corner_Hx", dict(axis="XY", phase=(1, -1), depth=1.2), None, "Hx"),
)

#: The array this seam's constitutive half reads for each magnetic component -- the
#: array the two fills write, whose Yee shift decides the images.
DEPOSIT_TARGET = {"Hx": "Bx", "Hy": "By", "Hz": "Bz"}


def magnetic_source(grid: Any, component: str, centre: Sequence[float]) -> Any:
    """A magnetic point source. ``GaussianPulsedSource`` refuses an H component by
    name (sources.py:205), so the magnetic seam can only be driven through
    ``VolumeSource`` -- the same mismatch that made this seam's first package-level
    cases pass vacuously."""
    return VolumeSource(grid=grid, component=component, center=tuple(centre),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def deposit_source(fields: Any, component: str) -> Any:
    """A real point source PLACED on every row a post-injection fill reads from.

    NOT SEARCHED FOR BY DIVERGENCE, and not spelled: ``deposit_repair.fill_image_rules``
    names the row each live fill images on each axis, and this walks that axis until
    the source's own deposit index lands on it. A hand-written coordinate would stop
    being on the imaged row the moment an extent or a Yee table moved, and the case
    would pass while measuring the ordinary carry.
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
                candidate = magnetic_source(grid, component, trial)
            except ValueError:
                continue  # a parity the plane refuses; not this axis's answer
            if getattr(candidate, "_point_ix", None) is None:
                continue
            index = (candidate._point_ix, candidate._point_iy,
                     candidate._point_iz)[axis]
            if row in {int(value) for value in np.atleast_1d(index)}:
                centre[axis] = trial[axis]
                break
    return magnetic_source(grid, component, centre)


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
                     component: str, faces: Optional[Any] = None,
                     repair: bool = True,
                     carry_images: bool = True) -> Dict[str, Any]:
    """THE IN-SEAM DEPOSIT, on the device, in the DRIVER'S OWN ORDER.

    Every other product leg in this file steps a seam with NO source in it, because
    until 2026-08-28 this family refused every one. It now declares
    ``CARRIES_DEPOSIT_REPAIR``, and a flag is not a measurement.

    THE WALK IS THE DRIVER'S, NOT ``metal_step``'S. ``metal_step`` skips every pass in
    ``REPLACES`` because the kernel carried them, which models a CLEAN seam. The driver
    runs the magnetic injection and the three fill/clear passes UNCONDITIONALLY between
    its two consults (driver.py:3292-3299), so on a seam carrying a deposit the
    kernel's inline fills happen BEFORE the injection and the host's after -- which is
    why the repair must carry the fills' images and not just the deposit index. And a
    missed B image does not stay in the magnetic half: ``update_H``'s ghost feeds
    ``step_D``'s curl, so the electric arrays diverge in the SAME step, which is what
    the unrepaired control shows.

    TWO CONTROLS, both required to diverge and failing differently: ``repair=False``
    leaves the deposit itself wrong, and ``carry_images=False`` cuts ``repair_cells``
    back to the deposit index -- the shape this module shipped before the closure -- so
    what is left wrong is exactly the image set.
    """
    original = deposit_repair.repair_cells
    probe, _probe_pml = build(keywords, seed, faces)
    placed = deposit_source(probe, component)
    centre = tuple(float(value) for value in placed.center)
    images = _image_cells(probe, placed, component)
    deposits = _deposit_cells(placed)
    if not carry_images:
        deposit_repair.repair_cells = lambda fields, target, index: index
    try:
        reference, reference_pml = build(keywords, seed, faces)
        actual, actual_pml = build(keywords, seed, faces)
        drift = compare(reference, actual)
        assert not drift, f"the two builds are not identical: {drift}"
        reference_source = magnetic_source(reference.grid, component, centre)
        actual_source = magnetic_source(actual.grid, component, centre)
        residency = Residency()
        plan = family.plan_metal_folded_fused_magnetic_pair(
            actual, actual_pml, sources=(actual_source,), residency=residency)
        if plan is None:
            reasons = family.metal_folded_fused_magnetic_pair_coverage(
                actual, actual_pml, (actual_source,), residency).reasons
            return {"passed": False, "reason": "the deposit was refused",
                    "refusals": list(reasons)}
        leading = deposit_repair.LeadingRepairPlan(
            plan, actual, actual_pml, (actual_source,), "B")
        trailing = deposit_repair.TrailingRepairPlan(
            "update_H", leading, actual, actual_pml)
        live = live_passes(actual, actual_pml)
        assert live == live_passes(reference, reference_pml)
        before = frozen(actual)
        dt = float(actual.grid.dt)
        residency.sync_in()

        per_step: List[Dict[str, Any]] = []
        for step in range(1, steps + 1):
            when = (step - 1) * dt
            for name in live:
                ARRAY_PATH[name](reference, reference_pml)
                if name == "step_B":
                    reference_source.inject(reference, when)
            for name in live:
                if name == family.SLOT:
                    residency.sync_in()
                    leading.run()
                    residency.sync_out()
                    actual_source.inject(actual, when)
                elif name == "update_H":
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
            # A CONTROL PASSES BY DIVERGING; reading one field for both directions
            # would make "the control worked" and "the product worked" one sentence.
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


def leg_deposit(name: str, keywords: Mapping[str, Any], faces: Optional[Any],
                seed: int, steps: int, component: str) -> Dict[str, Any]:
    """One deposit case and its two controls, reported as one row.

    NON-VACUITY FIRST: the placed deposit must actually HAVE images, or the legs below
    are the ordinary point repair wearing a folded label and the controls have nothing
    to miss.
    """
    carried = run_deposit_case(keywords, seed, steps, component, faces)
    point_only = run_deposit_case(keywords, seed, steps, component, faces,
                                  carry_images=False)
    unrepaired = run_deposit_case(keywords, seed, steps, component, faces,
                                  repair=False)
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


def leg_carry() -> Dict[str, Any]:
    """THE FLIP. The pass this family used to refuse BY NAME is now CARRIED.

    Until 2026-08-20 this leg asserted the opposite: it built a folded PERIODIC grid,
    where ``fill_folded_far_ghosts_B`` is live inside the seam (driver.py:3287), and
    required the predicate to REFUSE it. The refusal is gone because the kernel now
    images that plane itself, so the same fixture is used to measure the flip:

    * the composer must still say the pass is LIVE on that grid — otherwise the carry
      would be about a pass that never runs and the whole leg would be decoration;
    * the predicate must now COVER it and the plan must be built;
    * ``REPLACES`` must NAME the pass, so the walk's ``owned`` set skips it on the
      host and a dropped carry cannot hide behind an idempotent host re-run;
    * the plan must report a far ghost on the components the Yee table says, and a
      reflect row that is ``stepping._far_reflect_rows``' own answer.

    THE REFUSALS THAT REMAIN are measured beside it, because a product that stopped
    refusing everything would pass this leg by accident: a MAGNETIC source is
    injected inside the seam and an ELECTRIC one is not, an UNDECLARED source set is
    a refusal rather than an assumed empty one, and the reflect-row clauses the carry
    rests on refuse a grid that cannot support the ownership move.
    """
    far_fields, far_pml = build(FAR_CARRY_CASE, 91000)
    residency = Residency()
    carried = family.metal_folded_fused_magnetic_pair_coverage(
        far_fields, far_pml, (), residency)
    plan = family.plan_metal_folded_fused_magnetic_pair(
        far_fields, far_pml, sources=(), residency=residency)
    live = live_passes(far_fields, far_pml)
    codes, _ = symmetry.folded_axis_kinds(far_fields.grid, far_pml)
    rows = stepping._far_reflect_rows(far_fields.grid)
    expected_far = [list(family.far_fill_axes(codes, target)) for target in range(3)]

    keywords, faces = cases_table()[MUTATION_CASE]
    fields, pml = build(keywords, 91001, faces)
    # THE ADMISSION, since 2026-08-28: a REAL magnetic deposit placed on a row a fill
    # images is COVERED, because the repair now restores the images with the point.
    deposit = deposit_source(fields, "Hx")
    carried_deposit = family.metal_folded_fused_magnetic_pair_coverage(
        fields, pml, (deposit,), Residency())
    deposit_images = _image_cells(fields, deposit, "Hx")
    # THE REFUSAL THAT SURVIVES IT. `_MagneticSource` declares a field type and nothing
    # else, so it names no cells for the repair to save; a closure cannot carry what a
    # source will not name, and that is refused BY NAME rather than admitted on a flag.
    magnetic = family.metal_folded_fused_magnetic_pair_coverage(
        fields, pml, (_MagneticSource(),), Residency())
    magnetic_named = [reason for reason in magnetic.reasons
                      if "does not publish the index it writes" in reason]
    electric = family.metal_folded_fused_magnetic_pair_coverage(
        fields, pml, (_ElectricSource(),), Residency())
    undeclared = family.metal_folded_fused_magnetic_pair_coverage(
        fields, pml, None, Residency())
    undeclared_named = [reason for reason in undeclared.reasons
                        if "was not declared" in reason]
    # AND THE OTHER DIRECTION ON THE FLAG, so the admission cannot be a clause that
    # quietly went away: held False, the polarity refusal comes straight back.
    shipped = family.CARRIES_DEPOSIT_REPAIR
    try:
        family.CARRIES_DEPOSIT_REPAIR = False
        held_false = family.metal_folded_fused_magnetic_pair_coverage(
            fields, pml, (deposit,), Residency())
    finally:
        family.CARRIES_DEPOSIT_REPAIR = shipped
    polarity_named = [reason for reason in held_false.reasons
                      if "is magnetic" in reason and "driver.py:3283" in reason]

    # The reflect-row clauses, on stubs that violate one requirement each.
    periodic_codes = (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                      symmetry.CODE_PERIODIC)
    stubs: Dict[str, Any] = {}
    # (a) reflect row past the top plane: n_full - stored + 2 == stored - 1.
    stubs["reflect_row_is_the_plane_it_writes"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 13, 8), (False, True, False)),
            periodic_codes),
        "wanted": "outside [0,"}
    # (b) reflect row 0, the plane the NEAR fill writes.
    stubs["reflect_row_is_the_near_fills_destination"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 6, 8), (False, True, False)),
            periodic_codes),
        "wanted": "reflect row is 0"}
    # (c) the code and stepping._stored_past_owned disagree: the kernel would mask
    #     the top plane and image the far ghost on a different axis set than the
    #     array path. Declared PERIODIC, grid says the axis is not folded at all.
    stubs["code_and_stored_past_owned_disagree"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 14, 8), (False, False, False)),
            periodic_codes),
        "wanted": "disagree"}
    for entry in stubs.values():
        entry["named"] = [reason for reason in entry["reasons"]
                          if entry["wanted"] in reason]

    carried_ok = bool(
        carried.covered and plan is not None
        and "fill_folded_far_ghosts_B" in live
        and "fill_folded_far_ghosts_B" in family.REPLACES
        and plan is not None
        and [list(axes) for axes in plan.carried_far_axes] == expected_far
        and any(expected_far)
        and list(plan.reflect) == [None if r is None else int(r) for r in rows])
    return {
        "passed": bool(carried_ok
                       and all(entry["named"] for entry in stubs.values())
                       and shipped is True and deposit_images
                       and carried_deposit.covered
                       and not magnetic.covered and magnetic_named
                       and electric.covered
                       and not undeclared.covered and undeclared_named
                       and polarity_named and not held_false.covered
                       and family.plan_metal_folded_fused_magnetic_pair(
                           fields, pml, None, Residency()) is None),
        "carries_deposit_repair": bool(shipped),
        "real_deposit_admitted": carried_deposit.covered,
        "real_deposit_refusals_if_any": list(carried_deposit.reasons),
        "real_deposit_image_cells": deposit_images,
        "refused_again_with_the_flag_held_false": polarity_named,
        "far_fill_covered": carried.covered,
        "far_fill_plan_built": plan is not None,
        "far_fill_refusals_if_any": list(carried.reasons),
        "far_pass_is_live": "fill_folded_far_ghosts_B" in live,
        "far_pass_is_in_replaces": "fill_folded_far_ghosts_B" in family.REPLACES,
        "replaces": list(family.REPLACES),
        "carried_far_ghost_axes": ([list(a) for a in plan.carried_far_axes]
                                   if plan is not None else None),
        "carried_far_ghost_axes_expected": expected_far,
        "reflect_rows": (list(plan.reflect) if plan is not None else None),
        "reflect_rows_expected": [None if r is None else int(r) for r in rows],
        "boundary_codes": list(codes) if codes is not None else None,
        "stub_grid_refusals": {name: {"named": entry["named"],
                                      "all": entry["reasons"]}
                               for name, entry in stubs.items()},
        "magnetic_source_covered": magnetic.covered,
        "magnetic_named_refusals": magnetic_named,
        "electric_source_covered": electric.covered,
        "undeclared_sources_covered": undeclared.covered,
        "undeclared_named_refusals": undeclared_named,
        "live_passes": list(live),
    }


def leg_reflect_row(steps: int) -> Dict[str, Any]:
    """PREDICT from the full count, then measure: is ``n - 2`` wrong or not?

    ``stepping._far_reflect_rows``' docstring says in capitals that the image row is
    NOT always ``n - 2``, and that writing the fixed row "reflects about the window
    top instead of about the mirror, which is a whole cell wrong on every odd-count
    run". This leg is that sentence turned into a number. Two cases identical but for
    the cell extent — an even full count where ``stored - 2`` IS the reflect row and
    an odd one where it is ``stored - 3`` — with the prediction read off the grid
    (``last - reflect``) before the mutation runs, and agreement required.

    A gate carrying only the even extent would have armed this defect, watched it do
    nothing, and recorded a pass.
    """
    cases = cases_table()
    rows: Dict[str, Any] = {}
    ok = True
    for label in (REFLECT_EVEN_CASE, REFLECT_ODD_CASE):
        keywords, faces = cases[label]
        fields, pml = build(keywords, 94000, faces)
        codes, phases, walls = _specialisation(fields, pml)
        reflect = stepping._far_reflect_rows(fields.grid)
        gap = int(fields.grid.shape[1] - 1 - int(reflect[1]))
        source = family.folded_fused_magnetic_pair_source(codes, phases, walls)
        functions = compile_edits({label: needle(source, *REFLECT_ROW_EDIT)})[label]
        result = run_case(keywords, 94100 + len(rows), steps, faces,
                          functions=functions)
        caught = not result.get("bit_identical", False)
        observable = gap != 1
        agree = bool(caught) == bool(observable)
        ok = ok and agree and bool(result.get("launches"))
        rows[label] = {
            "shape": [int(n) for n in fields.grid.shape],
            "full_count": [int(n) for n in fields.grid.shape_full],
            "reflect_row": [None if r is None else int(r) for r in reflect],
            "top_minus_reflect": gap,
            "fixed_n_minus_two_is_wrong_here": observable,
            "caught": caught, "prediction_holds": agree,
            "first_divergence": result.get("first_divergence"),
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches")}
    return {"passed": bool(ok), "cases": rows,
            "note": "_far_reflect_rows is n_full - stored + 2: stored - 2 at an "
                    "even full count and stored - 3 at an odd one, so the fixed "
                    "n - 2 is a no-op on the first and a whole cell wrong on the "
                    "second"}


def leg_far_coefficient(steps: int) -> Dict[str, Any]:
    """The far ghost keeps its source thread's coefficient pair. MEASURED.

    The near ghost reloads ``kps``/``kms`` at stored index 0 because it moves the
    component along the axis ``update_H`` indexes on. The far ghost does NOT reload,
    because it moves along an axis that is not that one. This leg arms the reload
    and requires it CAUGHT: if the two entries were the same word on this fixture,
    the claim would be untested and the row would say so through an UNCAUGHT.

    The census beside it is what makes an uncaught row readable rather than a
    mystery — it reports whether ``kps_x`` varies at all along the indexed axis.
    """
    cases = cases_table()
    keywords, faces = cases[MUTATION_CASE]
    fields, pml = build(keywords, 95000, faces)
    codes, phases, walls = _specialisation(fields, pml)
    source = family.folded_fused_magnetic_pair_source(codes, phases, walls)
    functions = compile_edits({"far": needle(source, *FAR_COEFFICIENT_EDIT)})["far"]
    result = run_case(keywords, 95100, steps, faces, functions=functions)
    caught = not result.get("bit_identical", False)
    vector = np.asarray(pml.kps_x).ravel()
    varies = bool(np.any(vector != vector[0]))
    return {
        "passed": bool(caught and varies and result.get("launches")),
        "caught": caught,
        "indexed_axis_coefficient_varies": varies,
        "kps_x_first_and_last": [float(vector[0]), float(vector[-1])],
        "first_divergence": result.get("first_divergence"),
        "differing_words": result.get("differing_words"),
        "launches": result.get("launches"),
        "note": "a far ghost moves along an axis that is NOT the one update_H "
                "indexes on, so it takes its source thread's pair unchanged; "
                "reloading at stored index 0 is the near ghost's rule and is wrong "
                "here",
    }


def cases_table() -> Dict[str, Tuple[Dict[str, Any], Optional[Any]]]:
    return {name: (keywords, faces) for name, keywords, faces in CASES}


def coefficient_census(pml: Any, codes: Sequence[int]) -> Dict[str, Any]:
    """Does the moved index CHANGE the coefficient, per folded axis?

    The imaged ghost sits at stored index 0 of the folded axis and its source at
    ``NEAR_SOURCE_INDEX``. This family reads ``kps``/``kms`` at 0; reusing the
    source's entry is only a DIFFERENT NUMBER where the two entries differ, and on a
    folded axis they usually do not — the mirror plane carries no absorber
    (``stepping._require_consistent_pml`` admits the high face only), so the layer
    has to be deep enough to reach stored cell 2 before the distinction exists.
    """
    out: Dict[str, Any] = {}
    for axis, name in enumerate("xyz"):
        if int(codes[axis]) not in symmetry.MIRROR_CODES:
            continue
        entry: Dict[str, Any] = {}
        for stem in ("kps", "kms"):
            vector = np.asarray(getattr(pml, f"{stem}_{name}")).ravel()
            entry[stem] = [float(vector[0]),
                           float(vector[family.NEAR_SOURCE_INDEX])]
        entry["differs"] = bool(entry["kps"][0] != entry["kps"][1]
                                or entry["kms"][0] != entry["kms"][1])
        out[name] = entry
    return out


def leg_moved_coefficient(cases: Mapping[str, Tuple[Dict[str, Any], Optional[Any]]],
                          steps: int) -> Dict[str, Any]:
    """PREDICT observability from the coefficients, then measure the catch.

    THE LEG THIS FAMILY EXISTS TO HAVE. Its lines that :mod:`.folded_fused_pair`
    does not carry are the ghost's own coefficient pair, and a mutation leg that
    simply armed the reuse would have reported it UNCAUGHT on every ordinary fixture
    and left a reader to guess whether the line is wrong or the fixture is blind. So
    the prediction is made first, from the absorber vectors, and the requirement is
    agreement: caught exactly where the two coefficient entries differ, not caught
    where they are the same word.

    TWO EDITS RUN HERE. The pure NEAR ghost's reload is the one this family shipped
    with. The COMPOSITE near/far ghost's is the far carry's own, and it landed here
    rather than in the unconditional set for a MEASURED reason: armed on the
    mutation case it is bit-identical over twelve complete steps, because a folded
    axis carries no absorber at the mirror plane and the shared matrix's 2-cell
    high-face layer never reaches stored cell 2. :data:`DEEP_PERIODIC_CASE` is the
    row built to make it observable.

    Both edits touch COMPONENT 0's ghost, so the prediction is read off the x axis;
    the census reports every folded axis so a reader can see the whole picture.
    """
    resolved = {"MUTATION_CASE": MUTATION_CASE, "DEEP_CASE": DEEP_CASE,
                "DEEP_PERIODIC_CASE": DEEP_PERIODIC_CASE}
    out: Dict[str, Any] = {"edits": {}}
    ok = True
    seed = 93000
    for edit_name, (edit, case_names) in MOVED_COEFFICIENT_EDITS.items():
        rows: Dict[str, Any] = {}
        for name in case_names:
            label = resolved[name]
            keywords, faces = cases[label]
            fields, pml = build(keywords, seed, faces)
            codes, phases, walls = _specialisation(fields, pml)
            census = coefficient_census(pml, codes)
            source = family.folded_fused_magnetic_pair_source(codes, phases, walls)
            functions = compile_edits({label: needle(source, *edit)})[label]
            observable = bool(census.get("x", {}).get("differs", False))
            seed += 1
            result = run_case(keywords, seed, steps, faces, functions=functions)
            caught = not result.get("bit_identical", False)
            agree = bool(caught) == observable
            ok = ok and agree and bool(result.get("launches"))
            rows[label] = {
                "census": census,
                "shape": [int(n) for n in fields.grid.shape],
                "pml_faces": [list(face) for face in pml.thickness_by_face],
                "observable": observable, "caught": caught,
                "prediction_holds": agree,
                "first_divergence": result.get("first_divergence"),
                "differing_words": result.get("differing_words"),
                "differing_arrays": result.get("differing_arrays"),
                "launches": result.get("launches")}
        out["edits"][edit_name] = rows
    out["passed"] = bool(ok)
    out["note"] = ("an imaged ghost that moves the INDEXED axis reads kps/kms at "
                   "stored index 0; reusing the source thread's entry at index 2 is "
                   "a different number only where the folded axis's absorber "
                   "reaches stored cell 2. A far-only ghost does not move that axis "
                   "at all and is measured by leg far_coefficient instead")
    return out


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def _assert_mutation_geometry(label: str, codes, phases, walls, periodic: bool,
                              require_wall: bool = True, min_owned: int = 3,
                              ) -> Dict[str, Any]:
    """Every line the mutation set edits must be a line THIS case emits.

    THE DEAD-BRANCH TRAP, closed before the run rather than after. A mutation armed
    on a guard the scored case never enters reports UNCAUGHT while measuring
    nothing, so the geometry each row needs is asserted here — from the plan's own
    derivation of near and far axes, not from the keywords that built the grid.

    ``min_owned`` is the FLOOR ON THE COMPOSITION, and raising it is the whole
    point of :data:`TRIPLE_MUTATION_CASE`. Three is what two folded periodic axes
    give (a near plane, a far plane and their composite); SEVEN is what three give,
    and the triple composite at the top of that lattice is a destination no lower
    floor can reach. ``require_wall`` is False for that case because all three of
    its axes are folded and ``stepping._zero_metal`` excludes a folded axis by
    construction, so there is no axis left to wall — the wall edits are exempted by
    name in :data:`TRIPLE_STRUCTURALLY_ABSENT` instead of being armed on text the
    kernel does not emit.
    """
    near = [family.near_fill_axes(codes, target) for target in range(3)]
    far = [family.far_fill_axes(codes, target) for target in range(3)]
    owned = [len(family.carried_destinations(n, f)) for n, f in zip(near, far)]
    folded = [axis for axis, code in enumerate(codes)
              if code in symmetry.MIRROR_CODES]
    if require_wall:
        assert any(walls), (
            f"{label} has no walled axis; the wall mutation is vacuous")
    else:
        assert not any(walls), (
            f"{label} is declared wall-free but reports walls {walls}; the wall "
            f"edits are exempted for this case and would then go unscored")
    assert len(folded) >= 2, (
        f"{label} folds fewer than two axes; both parity spellings would not be "
        f"present and the parity mutations would be vacuous")
    assert set(phases[axis] for axis in folded) == {1, -1}, (
        f"{label} does not carry BOTH parities: {phases}")
    assert any(near), f"{label} emits no near carry"
    if periodic:
        assert any(far), f"{label} emits no far carry"
        assert any(n and f for n, f in zip(near, far)), (
            f"{label} has no component with BOTH a near and a far ghost, so the "
            f"composite mutation would be armed on absent text")
        assert any(len(f) >= 2 for f in far), (
            f"{label} has no component with two far axes, so the corner mutation "
            f"and its parity PRODUCT would be armed on absent text")
        assert max(owned) >= min_owned, (
            f"{label} owns at most {max(owned)} ghost cells per source thread, "
            f"below this case's floor of {min_owned}")
        if min_owned >= 7:
            # The TRIPLE destination itself, read off the same closed form the
            # kernel emits from: a ghost at the top plane of TWO far axes AND at
            # stored 0 on the near one. Without it the seven-destination edits
            # would be armed on text no component carries.
            triples = [[subset for subset, carries_near
                        in family.carried_destinations(n, f)
                        if carries_near and len(subset) >= 2]
                       for n, f in zip(near, far)]
            assert all(triples), (
                f"{label} has a component with no far/far/near TRIPLE composite: "
                f"{triples}; the seventh destination's edits would be armed on "
                f"absent text")
    else:
        assert not any(far), (
            f"{label} is declared MIRROR_METALLIC but emits a far carry: {far}")
    return {"case": label, "codes": list(codes), "phases": list(phases),
            "zero_metal": list(walls), "near_axes": [list(a) for a in near],
            "far_axes": [list(a) for a in far],
            "ghost_cells_per_source_thread": owned}


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

    cases = cases_table()
    geometry: Dict[str, Any] = {}
    mutants: Dict[str, Dict[str, Dict[str, Any]]] = {}
    bases: Dict[str, Tuple[Dict[str, Any], Optional[Any]]] = {}
    transcription_codes = None
    # (label, fold termination is periodic, a wall is required, ghost-cell floor).
    # THE FLOOR IS THE THIRD ROW'S WHOLE REASON: 3 is what two folded periodic axes
    # compose and 7 is what three do, and the seventh destination is a line no
    # lower floor emits.
    mutation_cases = ((MUTATION_CASE, True, True, 3),
                      (METALLIC_MUTATION_CASE, False, True, 3),
                      (TRIPLE_MUTATION_CASE, True, False, 7))
    for label, periodic, require_wall, min_owned in mutation_cases:
        keywords, faces = cases[label]
        fields, pml = build(keywords, 1, faces)
        codes, phases, walls = _specialisation(fields, pml)
        geometry[label] = _assert_mutation_geometry(
            label, codes, phases, walls, periodic, require_wall, min_owned)
        base = family.folded_fused_magnetic_pair_source(codes, phases, walls)
        if label == TRIPLE_MUTATION_CASE:
            # The general anchors this geometry still carries, PLUS the ten that
            # only exist at seven destinations. Overlapping names would silently
            # drop one of the two, so the disjointness is asserted rather than
            # assumed.
            carried = carried_shader_edits(base)
            triple = triple_shader_edits(base)
            clash = sorted(set(carried) & set(triple))
            assert not clash, (
                f"the carried and seven-destination edit sets share names {clash}; "
                f"one would overwrite the other and the artifact would report a "
                f"count higher than the number of defects actually armed")
            mutants[label] = compile_edits({**carried, **triple})
        else:
            mutants[label] = compile_edits(shader_edits(base, periodic))
        bases[label] = (keywords, faces)
        if label == MUTATION_CASE:
            transcription_codes = codes
    assert transcription_codes is not None

    mutation_keywords, mutation_faces = bases[MUTATION_CASE]
    controls = ("y_metallic_wall_z_3d", "xy_mixed_wall_z_3d",
                "y_periodic_even_3d", "xy_periodic_odd_wall_z_3d")
    total = (1 + 1 + len(CASES) + len(CASES) + 1 + len(controls)
             + len(DEPOSIT_CASES) + 1 + 1 + 1 + 1
             + sum(len(rows) for rows in mutants.values())
             + len(TRIPLE_STRUCTURALLY_ABSENT)
             + len(HOST_MUTATIONS) + len(mutants))
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
                     "label": "both_halves_are_the_certified_text",
                     **leg_transcription(transcription_codes)})
        emit(handle, rows[-1])

        for offset, (name, keywords, faces) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 41000 + offset, args.steps, faces)}
            rows.append(row)
            emit(handle, row)

        # THE SECOND VALUE CLASS, on every product row. A +-0 lattice reaches the
        # mirror fill and the far carry the same way the physical band does, and
        # the sign it carries IS the parity these seven destinations compose — so
        # a class of zeros is not a weak case here, it is the one that isolates
        # the parity product from the magnitude entirely.
        for offset, (name, keywords, faces) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps,
                   **run_case(keywords, 44000 + offset, args.steps, faces,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "subnormal_band_is_refused_and_the_census_fires",
                     **leg_subnormal_ladder(mutation_keywords, 46000,
                                            mutation_faces)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            keywords, faces = cases[name]
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(keywords, 47000 + offset, args.steps,
                                          faces)}
            rows.append(row)
            emit(handle, row)

        for offset, (name, keywords, faces, component) in enumerate(DEPOSIT_CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "deposit",
                   "label": name, "steps": args.steps,
                   **leg_deposit(name, keywords, faces, 49000 + offset, args.steps,
                                 component)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "carry",
                     "label": "far_fill_carried_and_the_refusals_that_remain",
                     **leg_carry()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "moved_coefficient",
                     "label": "predicted_observability_vs_measured_catch",
                     **leg_moved_coefficient(cases, args.steps)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "reflect_row",
                     "label": "fixed_n_minus_two_vs_the_measured_reflect_row",
                     **leg_reflect_row(args.steps)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "far_coefficient",
                     "label": "the_far_ghost_keeps_its_source_threads_pair",
                     **leg_far_coefficient(args.steps)})
        emit(handle, rows[-1])

        for case_label, compiled in mutants.items():
            keywords, faces = bases[case_label]
            for name, functions in compiled.items():
                index += 1
                result = run_case(keywords, 52000 + index, args.steps, faces,
                                  functions=functions)
                caught = not result.get("bit_identical", False)
                row = {"index": index, "total": total, "leg": "mutation",
                       "label": name, "case": case_label, "caught": caught,
                       "passed": bool(caught and result.get("launches")),
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "differing_arrays": result.get("differing_arrays"),
                       "launches": result.get("launches")}
                rows.append(row)
                emit(handle, row)

        # THE EDITS THE THREE-FOLD CASE CANNOT ARM, scored NULL CONFIRMED with the
        # structural reason and the anchor's own absence from the emitted text.
        # An edit missing from a table and an edit that could not be armed read
        # identically in an artifact, and only one of them is a measurement — so
        # each one asserts here that the line really is absent rather than that
        # someone believed it was.
        triple_keywords, triple_faces = bases[TRIPLE_MUTATION_CASE]
        triple_fields, triple_pml = build(triple_keywords, 1, triple_faces)
        triple_codes, triple_phases, triple_walls = _specialisation(
            triple_fields, triple_pml)
        triple_base = family.folded_fused_magnetic_pair_source(
            triple_codes, triple_phases, triple_walls)
        triple_anchors = shader_edit_anchors(True)
        for name, reason in sorted(TRIPLE_STRUCTURALLY_ABSENT.items()):
            index += 1
            anchor = triple_anchors[name][0]
            occurrences = triple_base.count(anchor)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "case": TRIPLE_MUTATION_CASE,
                   "outcome": "NULL CONFIRMED", "armed": False,
                   "anchor_occurrences_in_emitted_source": occurrences,
                   "reason": reason, "passed": occurrences == 0}
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            index += 1
            result = run_case(mutation_keywords, 52000 + index, args.steps,
                              mutation_faces, patch=patch)
            caught = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "case": MUTATION_CASE, "host_defect": True,
                   "caught": caught,
                   "passed": bool(caught and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        # THE DISARM CHECK, ONCE PER MUTATION CASE. The identical harness, the
        # identical case, the SHIPPED bytes and no host patch. A nonzero here would
        # mean every "caught" above is a harness that diverges on its own.
        for case_label in mutants:
            index += 1
            keywords, faces = bases[case_label]
            result = run_case(keywords, 52000 + index, args.steps, faces)
            row = {"index": index, "total": total, "leg": "disarm",
                   "label": f"shipped_bytes_on_{case_label}",
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
                   "shader_mutations": sum(len(r) for r in mutants.values()),
                   "shader_mutations_per_case": {k: len(v)
                                                 for k, v in mutants.items()},
                   "host_mutations": len(HOST_MUTATIONS),
                   "seven_destination_mutations": len(
                       triple_shader_edits(triple_base)),
                   "null_confirmed_on_the_triple_case": len(
                       TRIPLE_STRUCTURALLY_ABSENT),
                   "pm_zero_lattice_rows": len(CASES),
                   "subnormal_ladder_rungs": len(values.PRECONDITION_SCALES)},
        "value_classes": {
            "compared": list(values.VALUE_CLASSES),
            "refused": [values.SUBNORMAL_BAND],
            "refusal_reason": values.band_refusal_note(),
        },
        "mutation_case": MUTATION_CASE,
        "metallic_mutation_case": METALLIC_MUTATION_CASE,
        "triple_composite_mutation_case": TRIPLE_MUTATION_CASE,
        "deep_absorber_case": DEEP_CASE,
        "mutation_specialisation": geometry,
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
