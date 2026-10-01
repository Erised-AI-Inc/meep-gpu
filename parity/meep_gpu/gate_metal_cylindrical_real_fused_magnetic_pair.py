"""Byte gate for the Metal Dcyl m = 0 fused magnetic pair: ``step_B`` into ``update_H``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.cylindrical_real_fused_magnetic_pair.metal_cylindrical_real_fused_magnetic_pair_coverage`
admits, the device radial scan followed by ONE dispatch of
``cyl_real_fused_magnetic_pair_step`` leaves the engine in a state that is
BIT-IDENTICAL, PER COMPLETE STEP, to

  * the CuPy-free NumPy array path (``stepping.step_B`` / ``zero_metal_B`` /
    ``update_H``, plus the rest of the composer's live pass set), and
  * the two SEPARATELY CERTIFIED Metal products it replaces —
    ``cylindrical_real.plan_cylindrical_real_curl`` on ``step_B`` and
    ``cylindrical_real.plan_cylindrical_real_constitutive`` on ``H``, with
    ``zero_metal_B`` left on the HOST between them,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere.

THE LEGS

  1  binding_ceiling      34 separate bindings must FAIL; the packed 29 must
                          COMPILE, LAUNCH and read every struct field back
  2  transcription        both halves must be the certified emitters' own bytes
  3  product              every case, bit-identical per complete step
  4  separate_control     the fused product beside the two certified ones, from
                          one seed, with the dispatch counts on both sides
  5  inert_passes         the two driver passes REPLACES omits, EXECUTED on an
                          admitted grid, must move not one word — with
                          ``zero_metal_B`` as the control that must move something
  6  byte_neutral_control the register-vs-reload edit must NOT diverge
  7  mutation             armed defects, each of which MUST diverge
  8  refusal              the source seam and the fold, each measured by name
  9  disarm               the same harness, shipped bytes, must not diverge

WHAT THE GATE REFUSES TO INFER

* **A no-op agreeing with a no-op is trivially identical.** Every compared volume
  must MOVE during a case, and the fixture seeds the ``fu_*`` and ``f_w_*``
  auxiliaries as well as the twelve field volumes: zero init is a fixed point of the
  constitutive sub-step and a near-fixed-point of the split-field recurrence.
* **"The two omitted passes are inert" is a measurement**, not a reading of another
  module's guard — leg 5 runs them.
* **The mutation case's geometry must emit the lines the mutations rewrite.** The
  wall-clear mutation is armed on a z-WALLED case, because a z-periodic Dcyl grid
  emits no wall-clear line at all and the defect would report uncaught while
  measuring nothing.

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

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
    cylindrical_real as cyl,
    cylindrical_real_fused_magnetic_pair as family,
    launch as metal_launch,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the sibling fused pairs' and for
#: their reason: the classes this gate exists for COMPOUND. ``fu_B`` and ``f_w_H``
#: are state carried between steps, and a coefficient index off by one axis needs
#: several steps to reach the low bits of the interior.
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

#: Every stored volume a complete step can touch. The D/E half is in this list even
#: though this family does not touch it, because a fused pair that corrupted B would
#: reach E through ``step_D`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> the grid this case builds.
#:
#: WHAT THE ROWS CARRY BETWEEN THEM:
#:
#:   THE z DECLARATION   a metallic z emits the ``Bz`` wall-clear row and puts
#:                       ``zero_metal_B`` in the composer's live set; a PERIODIC z
#:                       emits no clear at all and drops the pass. Both are admitted
#:                       on this backend (unlike the Triton twin, which pins the
#:                       Dcyl declaration), so both are walked.
#:   THE COURANT NUMBER  three of the four are NOT powers of two, which is the family
#:                       of non-representable multiplicands the association defects
#:                       need in order to be visible at all.
#:   THE SHAPE           square, tall-and-thin and short-and-wide, because the radial
#:                       scan's column count and its row count trade off, and because
#:                       an index defect that reads a length-nr column at the z index
#:                       is in bounds on one and not on the others.
#:   matrix_row          one case is built by the SHARED composition matrix, so the
#:                       fixture this certifies is the fixture the two halves' own
#:                       gates were certified on rather than a grid that resembles it.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("square_metallic", dict(shape=(20, 1, 20), courant=0.5, z_kind="metallic")),
    ("square_metallic_odd_courant",
     dict(shape=(20, 1, 20), courant=0.3141592653589793, z_kind="metallic")),
    ("tall_metallic_odd_courant",
     dict(shape=(32, 1, 13), courant=0.4142135623730951, z_kind="metallic")),
    ("wide_metallic_odd_courant",
     dict(shape=(13, 1, 32), courant=0.37, z_kind="metallic")),
    ("square_periodic", dict(shape=(20, 1, 20), courant=0.5, z_kind="periodic")),
    ("composition_matrix_row", dict(matrix_row=True)),
)

#: The case the mutations are armed on. SQUARE and z-WALLED, and both halves of that
#: are load-bearing:
#:
#: * z WALLED, so the ``Bz`` wall-clear row is emitted at all — on a z-periodic case
#:   ``zero_metal_mask`` writes a comment and the wall defects would report uncaught
#:   for the uninteresting reason that no line exists;
#: * SQUARE (nr == nz), so the coefficient-index defect that reads a length-nr column
#:   at the z index is IN BOUNDS and measures a wrong VALUE. On ``wide_metallic`` it
#:   would read past the end, which is a memory fault reported as an inert defect.
MUTATION_CASE = "square_metallic_odd_courant"

#: The row leg ``refusal`` builds its source and fold questions on.
REFUSAL_CASE = "square_metallic"


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

def build(label: str, seed: int, value_class: str = values.UNIFORM,
          scale: float = 1.0) -> Tuple[Any, Any]:
    """One seeded Dcyl m = 0 engine. Called twice (or three times) per case.

    THE SEED COVERS ``fu_*`` AND ``f_w_*`` AS WELL AS THE TWELVE FIELD VOLUMES, and
    that is not cosmetic. Zero init is a FIXED POINT of the constitutive sub-step and
    a near-fixed-point of the split-field recurrence — a no-op agreeing with a no-op
    is trivially identical — so the auxiliaries carry physical-band values from step
    zero and the vacuity floor below has something to measure.

    IT ALSO DECIDES WHETHER THE CURL'S OWNERSHIP MASK IS OBSERVABLE. Measured on the
    Triton twin's first device run: with the ELECTRIC constitutive history left at
    zero, ``Ep`` is exactly ``+0.0`` on both planes the mask touches — the m = 0 axis
    rule sets ``Dp[0] = 0`` and ``zero_metal_D`` clears ``Dy`` at z = 0 — so the mask
    writes a value already present and a mutation dropping it cannot fail. Seeding
    every ``f_w_E*`` is what makes it observable, and it is done here for that reason
    rather than for symmetry.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    options = dict(CASES)[label]
    if options.get("matrix_row"):
        fields, pml = matrix.cylindrical(m=0, complex_storage=False,
                                         z_kind="metallic")
        shape = tuple(int(n) for n in fields.grid.shape)
    else:
        shape = tuple(int(n) for n in options["shape"])
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), 0.0, float(shape[2])),
                    cylindrical=True, m=0,
                    boundaries={"z": options["z_kind"]},
                    courant=float(options["courant"]), xp=np)
        assert tuple(grid.shape) == shape, (tuple(grid.shape), shape)
        fields = Fields(grid=grid, force_complex_fields=False)
        matrix._epsilon(fields)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness={"x": (0, max(2, shape[0] // 4)),
                                        "z": max(2, shape[2] // 4)})

    if value_class == values.PM_ZERO_LATTICE:
        lattice = values.pm_zero_lattice(shape)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice.astype(array.dtype)
        return fields, pml
    if value_class != values.UNIFORM:
        raise ValueError(f"unknown value class {value_class!r}")
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.uniform(-1.0, 1.0, size=shape)
                          * scale).astype(array.dtype)
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

def leg_subnormal_ladder(label: str, seed: int,
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
        return build(label, seed, values.UNIFORM, scale)

    def step(engine: Tuple[Any, Any]) -> None:
        fields, pml = engine
        array_step(fields, pml, live_passes(fields, pml))

    def census(engine: Tuple[Any, Any]) -> int:
        return sum(subnormal.census(value)
                   for value in state_of(engine[0]).values())

    return values.census_ladder(make, step, census, budget, log)


def run_case(label: str, seed: int, steps: int,
             fused_function: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(label, seed, value_class)
    actual, actual_pml = build(label, seed, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_cylindrical_real_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency,
        fused_function=fused_function)
    if plan is None:
        reasons = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
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
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.prefix_launches == plan.fused_launches == len(per_step))
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
        "prefix_launches": plan.prefix_launches,
        "fused_launches": plan.fused_launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.bc),
        "zero_metal": list(plan.zero_metal),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(label: str, seed: int, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path, the two ALREADY CERTIFIED Metal products
    stepping the same seam with the wall clear left on the HOST between them, and the
    fused product. All three must agree word for word at every complete step, and the
    DISPATCH COUNTS are recorded on both sides — which is where the difference
    between the compositions shows up at all, since a correct fusion is byte-neutral
    by construction.

    THE SEPARATE SIDE IS THREE DISPATCHES AND THE FUSED SIDE IS TWO. Both compute the
    same radial scan; what the fusion removes is the constitutive dispatch and the
    ``B`` round trip in front of it.
    """
    reference, reference_pml = build(label, seed)
    separate, separate_pml = build(label, seed)
    fused, fused_pml = build(label, seed)

    separate_residency = Residency()
    curl = cyl.plan_cylindrical_real_curl(separate, separate_pml, "step_B",
                                          separate_residency)
    magnetic = cyl.plan_cylindrical_real_constitutive(separate, separate_pml, "H",
                                                      separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_cylindrical_real_fused_magnetic_pair(
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
    # fused side carries the wall clear in the kernel; the separate side does not, so
    # on a walled run it pays a host round trip the fused side does not.
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
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
    """The (codes, walls) pair the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    return (cyl.boundary_codes(_boundary_kinds(fields.grid, pml)),
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


def shader_mutations(codes: Sequence[int],
                     walls: Sequence[bool]) -> Dict[str, Dict[str, Any]]:
    """Armed source defects, each a plausible transcription slip.

    Every needle is anchored on text only the mutated line carries, and every one is
    verified present before it is applied — an absent needle raises here rather than
    silently arming nothing.

    THE MUTATION CASE'S GEOMETRY, which is what makes each of these reachable: z
    walled (so the ``Bz`` clear row is emitted), r pinned METALLIC (so the axis ghost
    and the r ownership mask are live), and square (so the coefficient-index defect
    reads in bounds).
    """
    base = family.cylindrical_real_fused_magnetic_pair_source(codes, walls)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float src0 = v0;", "float src0 = curl0;"),
        # THE SEAM, WRONG COMPONENT. Bx's constitutive source is v0, not v1.
        "seam_takes_the_wrong_component":
            needle(base, "float src1 = v1;", "float src1 = v0;"),
        # THE WALL CLEAR, DROPPED. z is the walled axis on the mutation case, and Bz
        # is the component whose z Yee shift is 0.
        "zero_metal_dropped":
            needle(base, "    v2 = at_z ? 0.0f : v2;\n", ""),
        # THE WALL TABLE IS THE DIAGONAL FOR B. Clearing v0 on the z wall instead of
        # v2 is the D-side table's shape and the single most likely porting slip.
        "zero_metal_uses_the_d_side_table":
            needle(base, "    v2 = at_z ? 0.0f : v2;",
                   "    v0 = at_z ? 0.0f : v0;"),
        # THE WALL CLEAR AFTER THE CONSTITUTIVE READ. B ends up right and H does not,
        # which is the seam's ORDER rather than the clear's existence.
        "zero_metal_after_the_constitutive_read":
            needle(base, "    v2 = at_z ? 0.0f : v2;\n"
                         "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f0[ii] = v0; f1[ii] = v1; f2[ii] = at_z ? 0.0f : v2;"),
        # THE m = 0 AXIS RULE, dropped (stepping._cylindrical_axis_zero_B:661-662).
        "axis_rule_dropped":
            needle(base, "    v0 = at_x ? 0.0f : v0;\n", ""),
        # Bz's CURL IS THE PREFIX FORWARD DIFFERENCE, not the generic four-operand
        # stencil (stepping.py:371-375).
        "bz_curl_is_the_generic_one":
            needle(base, "float curl2 = dtdx * (pfx_up - pfx_here);",
                   "float curl2 = dtdx * ((b_x - b) + (a - a_y));"),
        # ONE SUBTRACT, ONE MULTIPLY. Distributing dtdx is a different float32
        # number wherever dtdx is not a power of two — which the mutation case's
        # courant guarantees.
        "bz_prefix_grouping_distributed":
            needle(base, "float curl2 = dtdx * (pfx_up - pfx_here);",
                   "float curl2 = (dtdx * pfx_up) - (dtdx * pfx_here);"),
        # THE PREFIX ROW OFFSET is one radial row at the same (phi, z) stride.
        "prefix_row_offset_dropped":
            needle(base, "float pfx_up   = pfx[ii + nyz];",
                   "float pfx_up   = pfx[ii + 1];"),
        # THE OWNERSHIP MASK, dropped on the r axis — a different pass from both the
        # wall clear and the axis rule, on the same row.
        "ownership_mask_dropped":
            needle(base, "    curl0 = at_x ? 0.0f : curl0;\n", ""),
        # shaders rule 2: flattening these parens is a different float32 number —
        # BUT ONLY WHERE BOTH DIFFERENCES ARE LIVE. Target 1's pair is (z, r) and
        # both neighbours are real, so this one is a genuine rounding change.
        "curl_parens_flattened_on_a_live_pair":
            needle(base, "dtdx * ((a_z - a) + (c - c_x))",
                   "dtdx * (a_z - a + c - c_x)"),
        # THE SAME EDIT ON TARGET 0, WHOSE FIRST PAIR IS THE INVARIANT AXIS. Declared
        # a NULL and confirmed as one — see the expectation table below.
        "curl_parens_flattened_on_the_invariant_pair":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        # THE INVARIANT-AXIS TERM is computed, not elided; computing it WRONG is
        # loud, where eliding it differs only on a signed zero.
        "phi_term_sign_flipped":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * ((c_y + c) + (b - b_z))"),
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
        # The dsigw index is the component's OWN axis (stepping.py:227-228). Moving
        # it to the z index is in bounds on a SQUARE case and is a wrong value.
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = km0[i];",
                   "float kp_0 = kp0[k], km_0 = km0[k];"),
        "constitutive_accumulations_reversed":
            needle(base, "    a0 = a0 + kp_0 * src0;\n"
                         "    a0 = a0 - km_0 * prev0;",
                   "    a0 = a0 - km_0 * src0;\n"
                   "    a0 = a0 + kp_0 * prev0;"),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        mutants[name] = {
            shaders.CONTRACT_OFF:
                compile_source(source).cyl_real_fused_magnetic_pair_step}
    return mutants


#: Mutation id -> what it is DECLARED to do. Anything not named here must be CAUGHT.
#:
#: A DECLARED NULL MUST BE CONFIRMED, not merely permitted: a null that IS caught
#: means the reasoning behind it is wrong, and that has to fail too.
#:
#: ``curl_parens_flattened_on_the_invariant_pair`` is the one entry, and it is a
#: MEASUREMENT rather than a concession. phi has n = 1 on a Dcyl grid, so the rolled
#: operand IS the original and ``(c_y - c)`` is an exact ``+0.0``; the flattened form
#: is then ``((+0.0) + b) - b_z``, which equals ``+0.0 + (b - b_z)`` for every
#: float32 except where ``(b - b_z)`` is ``-0.0``. Measured on this gate's first run:
#: 0 differing words over the whole sweep. Its SIBLING on target 1 — whose pair is
#: (z, r), both live — is armed as must-catch, and that is what stops this null from
#: reading as "the parenthesisation does not matter here".
MUTATION_EXPECTATION: Dict[str, str] = {
    "curl_parens_flattened_on_the_invariant_pair": "null",
}


def mutation_must_be_caught(name: str) -> bool:
    return MUTATION_EXPECTATION.get(name, "caught") == "caught"


def byte_neutral_source(codes: Sequence[int], walls: Sequence[bool]) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and ``v0``
    hold the same float32 word, and a float32 stored to a ``device float*`` and
    reloaded is bit-identical to the register. This edit is therefore the fusion's
    central claim written as a program, and leg ``byte_neutral_control`` requires it
    NOT to diverge.
    """
    source = family.cylindrical_real_fused_magnetic_pair_source(codes, walls)
    for target in range(3):
        source = needle(source, f"float src{target} = v{target};",
                        f"float src{target} = f{target}[ii];")
    return source


#: Where each binding group starts in the plan's fused argument tuple. Spelled once,
#: so the two host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(10, 16))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(22, 28))


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._fused_args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._fused_args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

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


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """34 separate bindings must FAIL; the packed 29 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. A signature that compiled but bound the struct wrongly would
    produce a smooth, plausible, wrong field, so the launch half is measured here and
    not inherited from another family's docstring.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS, "ceiling": 31}
    try:
        compile_source(family.refuted_separate_scalar_source())
        row.update(separate_compiled=True, separate_error="", passed=False,
                   note="the 34-binding signature COMPILED; this family's shape "
                        "rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(28))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;"
        " float axis_coef; };", "",
        "kernel void packed_probe(", pointers,
        "    constant Params&    prm     [[buffer(28)]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;", "    b5[idx] = prm.axis_coef;",
        "    b6[idx] = b27[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(28)]
    buffers[27] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx, axis_coef = (3, 4, 5), 0.125, 0.5
    # The real packer, not a hand-built record: the field the kernel reads is then
    # the field the plan builder writes. n_elem is the product of the shape (60),
    # deliberately not the eight-element probe buffers, so the guard is exercised.
    params = family._params_tensor(shape, dtdx, axis_coef, "mps")
    function(*buffers, params)
    torch.mps.synchronize()
    read = [buffers[index].cpu().numpy() for index in range(7)]
    expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx)),
                float(np.float32(axis_coef)), 8.0)
    fields_ok = all(bool(np.all(read[index] == expected[index]))
                    for index in range(7))
    row.update(packed_launched=True,
               packed_fields_read_back=[float(value[0]) for value in read],
               packed_fields_expected=[float(value) for value in expected],
               packed_fields_correct=fields_ok,
               passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok))
    return row


def leg_transcription(codes: Sequence[int], walls: Sequence[bool]) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes.

    The module claims its arithmetic is lifted rather than retyped. That is a
    property of the construction and therefore checkable, so it is checked:

    * the certified cylindrical ``step_B`` curl body must appear in the fused source
      VERBATIM on both sides of the spliced wall clear;
    * the constitutive half must differ from the certified ``update_H`` body in
      EXACTLY the three ``float srcN =`` lines. Any other differing line means a
      rename or a re-spelling reached the arithmetic.
    """
    source = family.cylindrical_real_fused_magnetic_pair_source(codes, walls)
    curl = family.certified_cyl_curl_body(codes)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    certified = family.certified_constitutive_body().splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
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


def leg_inert_passes() -> Dict[str, Any]:
    """The two driver passes :data:`REPLACES` omits, EXECUTED and measured.

    The product names three passes where the seam has five. The other two are omitted
    on the claim that they cannot execute on a grid this predicate admits — a reading
    of another module's guard until somebody runs them. So this leg runs them: a
    seeded state on every admitted case, both passes applied, every stored volume
    compared as uint32.

    NON-VACUITY: the CONTROL is ``zero_metal_B``, which is NOT inert on a walled case
    and must move something, so an inert pass cannot be confused with a frozen state.
    On a z-PERIODIC case there is no wall at all and the control is skipped by name
    rather than silently passing.
    """
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for offset, (label, _options) in enumerate(CASES):
        fields, pml = build(label, 81000 + offset)
        verdict = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
            fields, pml, (), Residency())
        if not verdict.covered:
            findings.append(f"{label}: refused, so this leg measured nothing: "
                            f"{list(verdict.reasons)}")
            continue
        before = frozen(fields)
        constant = sorted(name for name, value in before.items()
                          if len(np.unique(words(value))) <= 1)
        if constant:
            findings.append(f"{label}: {constant} are constant before the passes "
                            f"run; this leg would report inertness about nothing")
        for pass_name in family.INERT_PASSES:
            getattr(stepping, pass_name)(fields)
        after = state_of(fields)
        moved = sorted(name for name in before
                       if differing(before[name], after[name]))
        middle = frozen(fields)
        stepping.zero_metal_B(fields)
        control = state_of(fields)
        control_moved = sorted(name for name in middle
                               if differing(middle[name], control[name]))
        walls = zero_metal_axes(fields.grid)
        rows.append({"case": label, "shape": list(fields.grid.shape),
                     "walls": list(walls),
                     "moved_by_the_inert_passes": moved,
                     "moved_by_zero_metal_B": control_moved})
        if moved:
            findings.append(
                f"{label}: {sorted(family.INERT_PASSES)} moved {moved} — they are "
                f"NOT inert on this grid and REPLACES may not omit them")
        if any(walls) and not control_moved:
            findings.append(
                f"{label}: zero_metal_B moved nothing on a WALLED grid, so this leg "
                f"cannot tell an inert pass from a frozen state")
    return {"passed": not findings, "rows": rows, "findings": findings,
            "inert_passes": sorted(family.INERT_PASSES)}


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
    """The source seam, the azimuthal order and the Cartesian grid, each by name.

    Four questions, and the third is the one that separates this seam from the D-side
    one rather than merely restating it:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real MAGNETIC ``VolumeSource`` must be refused, naming the driver line;
    3. a real ELECTRIC ``VolumeSource`` must NOT be refused — it is injected in the
       D/E half, outside this seam, and it is what all three corpus rows carry;
    4. an |m| >= 1 Dcyl grid and a CARTESIAN grid must both be refused by name.
    """
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    magnetic = _volume_source(fields, "Hz")
    magnetic_coverage = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), residency)
    magnetic_named = [r for r in magnetic_coverage.reasons
                      if "is magnetic" in r and "driver.py:3283-3284" in r]
    magnetic_plan = family.plan_metal_cylindrical_real_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), residency=residency)

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (electric,), residency)

    m_fields, m_pml = matrix.cylindrical(m=1, complex_storage=True)
    m_coverage = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        m_fields, m_pml, (), Residency())
    m_named = [r for r in m_coverage.reasons if "grid.m = 1" in r]

    cart_fields, cart_pml = matrix.cart()
    cart_coverage = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        cart_fields, cart_pml, (), Residency())
    cart_named = [r for r in cart_coverage.reasons if "cylindrical" in r]

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and not magnetic_coverage.covered and magnetic_named
                       and magnetic_plan is None
                       and electric_coverage.covered
                       and not m_coverage.covered and m_named
                       and not cart_coverage.covered and cart_named),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_refused": not magnetic_coverage.covered,
        "magnetic_plan_is_none": magnetic_plan is None,
        "magnetic_named": magnetic_named,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "m_is_one_refused": not m_coverage.covered,
        "m_named": m_named[:2],
        "cartesian_refused": not cart_coverage.covered,
        "cartesian_named": cart_named[:2],
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

    probe_fields, probe_pml = build(MUTATION_CASE, 1)
    codes, walls = _specialisation(probe_fields, probe_pml)
    assert walls[2], (
        f"{MUTATION_CASE} does not wall z; the wall mutations would be armed on "
        f"lines the shipped kernel does not emit")
    assert probe_fields.grid.shape[0] == probe_fields.grid.shape[2], (
        f"{MUTATION_CASE} is not square; the coefficient-index mutation would read "
        f"a length-nr column past its end, which is a memory fault reported as an "
        f"inert defect")
    mutants = shader_mutations(codes, walls)
    neutral = {shaders.CONTRACT_OFF:
               compile_source(byte_neutral_source(codes, walls)
                              ).cyl_real_fused_magnetic_pair_step}

    controls = ("square_metallic", "square_periodic")
    total = (1 + 1 + len(CASES) + len(CASES) + 1 + len(controls) + 1 + 1
             + len(mutants) + len(HOST_MUTATIONS) + 1 + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "twenty_eight_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_bytes",
                     **leg_transcription(codes, walls)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "inert_passes",
                     "label": "the_two_passes_REPLACES_omits", **leg_inert_passes()})
        emit(handle, rows[-1])

        for offset, (name, _keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(name, 61000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        # THE SECOND VALUE CLASS, on every product row, and then the band.
        for offset, (name, _keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "value_class",
                   "label": f"{name}:{values.PM_ZERO_LATTICE}",
                   "steps": args.steps,
                   **run_case(name, 63000 + offset, args.steps,
                              value_class=values.PM_ZERO_LATTICE)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_class",
                     "label": "subnormal_band_is_refused_and_the_census_fires",
                     **leg_subnormal_ladder(MUTATION_CASE, 65000)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(name, 67000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "the_source_seam_the_order_and_the_geometry",
                     **leg_refusal()})
        emit(handle, rows[-1])

        # THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT. Scored on NOT diverging.
        index += 1
        result = run_case(MUTATION_CASE, 71000, args.steps, fused_function=neutral)
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
            result = run_case(MUTATION_CASE, 72000 + index, args.steps,
                              fused_function=functions)
            caught = not result.get("bit_identical", False)
            wanted = mutation_must_be_caught(name)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "caught": caught,
                   "expectation": "caught" if wanted else "null",
                   "passed": bool(caught is wanted and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            index += 1
            result = run_case(MUTATION_CASE, 72000 + index, args.steps, patch=patch)
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
        # bytes and no host patch. A nonzero here would mean every "caught" above is
        # a harness that diverges on its own.
        index += 1
        result = run_case(MUTATION_CASE, 72000 + index, args.steps)
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
                   "host_mutations": len(HOST_MUTATIONS),
                   "byte_neutral_controls": 1},
        "mutation_case": MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes), "zero_metal": list(walls)},
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "admit_both_halves": 3,
            "and_declare_no_magnetic_source": 3,
            "note": "the cell has NO attrition: 3 -> 3 -> 3. The Triton census "
                    "funnels to the same three rows through its own independently "
                    "written predicates. 3 is an upper bound because the census "
                    "records source FIELD TYPES but not whether a row's grid would "
                    "also satisfy the residency and wall-readability clauses this "
                    "predicate adds.",
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
