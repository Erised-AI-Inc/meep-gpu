"""Byte gate for the Metal Dcyl m = 0 fused ELECTRIC pair: ``step_D`` into ``update_E``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.cylindrical_real_fused_electric_pair.metal_cylindrical_real_fused_electric_pair_coverage`
admits, the device radial scan followed by ONE dispatch of
``cyl_real_fused_electric_pair_step`` leaves the engine in a state that is
BIT-IDENTICAL, PER COMPLETE STEP, to

  * the CuPy-free NumPy array path (``stepping.step_D`` / ``zero_metal_D`` /
    ``stepping.update_E``, plus the rest of the composer's live pass set), and
  * the two SEPARATELY CERTIFIED Metal products it replaces —
    ``cylindrical_real.plan_cylindrical_real_curl`` on ``step_D`` and
    ``cylindrical_real.plan_cylindrical_real_constitutive`` on ``E``, with
    ``zero_metal_D`` left on the HOST between them,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere.

WHAT IS NEW ON THIS PRODUCT, and it is the reason this gate carries two legs its
magnetic twin does not: **the fused signature does not fit unless an ARRAY is
packed.** Every shipped Metal pair buys its slots back by packing SCALARS into one
``constant Params&``; this cell is one POINTER over the ceiling with that already
done. So the curl half's six read-only PML coefficient vectors ride in ONE buffer
with six element offsets in the same struct, and two legs measure that:

  * ``binding_ceiling`` compiles THREE signatures and locates the ceiling by
    bisection, so 31 is a number this host returned rather than one this file cites;
  * ``pack_identity`` measures the packed buffer against the six separate vectors,
    bytes and reads, and the OFFSET mutations below are what make a wrong offset a
    caught defect rather than a plausible absorber.

THE LEGS

  1  binding_ceiling      the UNPACKED 32-binding signature must FAIL, the packed
                          27 must COMPILE, LAUNCH and read every struct field back
                          — including the six offsets — and the ceiling itself is
                          bisected on this host
  2  pack_identity        the packed buffer's bytes ARE the six vectors end to end,
                          and a kernel reading through the pack returns the same
                          words as one reading six separate pointers
  3  transcription        both halves must be the certified emitters' own bytes
  4  inert_passes         the two driver passes REPLACES omits, EXECUTED on an
                          admitted grid, must move not one word — with
                          ``zero_metal_D`` as the control that must move something
  5  product              every case, bit-identical per complete step
  6  value_class          the same, on the +-0 lattice, plus the subnormal ladder
  7  separate_control     the fused product beside the two certified ones, from one
                          seed, with the dispatch counts on both sides
  8  deposit              a REAL electric ``VolumeSource`` inside the seam, carried
                          through the shipped ``LeadingRepairPlan``/
                          ``TrailingRepairPlan`` bracket the composer installs
  9  deposit_null_control the same launch UNBRACKETED, which MUST diverge
 10  byte_neutral_control the register-vs-reload edit must NOT diverge
 11  mutation             armed defects, each of which must diverge — shader edits,
                          host lattice swaps, and PACK OFFSET corruptions
 12  refusal              the source seam, the order and the geometry, by name
 13  disarm               the same harness, shipped bytes, must not diverge

WHAT THE GATE REFUSES TO INFER

* **A no-op agreeing with a no-op is trivially identical.** Every compared volume
  must MOVE during a case, and the fixture seeds the ``fu_*`` and ``f_w_*``
  auxiliaries as well as the twelve field volumes.
* **"The two omitted passes are inert" is a measurement**, not a reading of another
  module's guard — leg 4 runs them.
* **"The pack is byte-neutral" is a measurement**, not an argument from
  concatenation — leg 2 runs both shapes.
* **A DECLARED NULL MUST BE CONFIRMED.** Two of the eleven offset corruptions are
  declared null and the reason is measured: on a Dcyl grid the phi axis carries no
  PML, so ``kms_y[0]`` and ``sinv_y[0]`` are both exactly ``1.0f`` and swapping the
  two offsets moves no word. Their siblings — the same two offsets moved the OTHER
  way, into a neighbouring vector — are armed as must-catch, which is what stops the
  null reading as "offsets do not matter here".

Rule 7: one flushed line per case, every row appended and fsynced as it lands.
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

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    coefficient_pack,
    cylindrical_real as cyl,
    cylindrical_real_fused_electric_pair as family,
    cylindrical_real_fused_magnetic_pair as sibling,
    launch as metal_launch,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the sibling fused pairs' and for
#: their reason: the classes this gate exists for COMPOUND. ``fu_D`` and ``f_w_E``
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

#: Every stored volume a complete step can touch. The B/H half is in this list even
#: though this family does not touch it, because a fused pair that corrupted D would
#: reach B through ``step_B`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> the grid this case builds. THE ROWS ARE THE MAGNETIC TWIN'S, deliberately:
#: the two products are the same two halves on the two seams, so a divergence between
#: the gates' verdicts is then a fact about the seam and not about the fixtures.
#:
#:   THE z DECLARATION   a metallic z emits the ``Dx``/``Dy`` wall-clear rows and puts
#:                       ``zero_metal_D`` in the composer's live set; a PERIODIC z
#:                       emits no clear at all and drops the pass. Both are admitted
#:                       on this backend, so both are walked.
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
#: * z WALLED, so the ``Dx``/``Dy`` wall-clear rows are emitted at all — on a
#:   z-periodic case ``zero_metal_mask`` writes a comment and the wall defects would
#:   report uncaught for the uninteresting reason that no line exists;
#: * SQUARE (nr == nz), so the coefficient-index defect that reads a length-nr column
#:   at the z index is IN BOUNDS and measures a wrong VALUE. On ``wide_metallic`` it
#:   would read past the end, which is a memory fault reported as an inert defect.
MUTATION_CASE = "square_metallic_odd_courant"

#: The row legs ``refusal`` and ``deposit`` build their questions on.
REFUSAL_CASE = "square_metallic"
DEPOSIT_CASE = "square_metallic"


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
    """The composer's OWN live set, never a second model of what a step is."""
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

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns THREE passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``zero_metal_D`` running on the host on top of the clear the
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

def leg_subnormal_ladder(label: str, seed: int, budget: int = 4) -> Dict[str, Any]:
    """THE EMPTY CENSUS, TURNED INTO A MEASUREMENT.

    Every product row reports ``reference_subnormals: 0`` under policy ``flush``.
    Read alone that is a SILENCE: it cannot distinguish "the band was exercised and
    the policy changed nothing" from "the band was never entered". The ladder scales
    the seeded state down until the ARRAY PATH's own output enters the subnormal
    band, and requires the census to FIRE there.
    """
    rows: List[Dict[str, Any]] = []
    scale = 1.0
    for rung in range(budget):
        scale *= 1e-9
        fields, pml = build(label, seed + rung, scale=scale)
        live = live_passes(fields, pml)
        array_step(fields, pml, live)
        census = sum(subnormal.census(value) for value in state_of(fields).values())
        rows.append({"rung": rung, "scale": scale, "reference_subnormals": census})
        log(f"    ladder rung {rung}: scale={scale:.3e} subnormals={census}")
        if census:
            break
    return {"passed": bool(rows and rows[-1]["reference_subnormals"] > 0),
            "rows": rows, "note": values.band_refusal_note(),
            "policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")}


def run_case(label: str, seed: int, steps: int,
             fused_function: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             params: Optional[Callable[[Any], Any]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(label, seed, value_class)
    actual, actual_pml = build(label, seed, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    record = None
    if params is not None:
        # The HOST mutation seam. Building the shipped plan first is what lets a leg
        # read the layout the packer produced and hand back a struct that disagrees
        # with it — the defect is in the OFFSETS, so it cannot be reached from the
        # kernel source at all.
        probe = family.plan_metal_cylindrical_real_fused_electric_pair(
            actual, actual_pml, sources=(), residency=Residency())
        assert probe is not None, "the params mutation has no shipped plan to read"
        record = params(probe)
    plan = family.plan_metal_cylindrical_real_fused_electric_pair(
        actual, actual_pml, sources=(), residency=residency,
        fused_function=fused_function, params=record)
    if plan is None:
        reasons = family.metal_cylindrical_real_fused_electric_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False,
                "reason": "the cylindrical fused electric pair was refused",
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
        "pack_layout": plan.pack_layout.describe(),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(label: str, seed: int, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME, and on this
    product it also makes "the pack is byte-neutral" one: the separate side binds the
    six PML coefficient vectors as six pointers through the CERTIFIED curl kernel,
    the fused side reads the same six through one packed buffer, and the two must
    agree word for word at every complete step.

    THE SEPARATE SIDE IS THREE DISPATCHES AND THE FUSED SIDE IS TWO. Both compute the
    same radial scan; what the fusion removes is the constitutive dispatch and the
    ``D`` round trip in front of it.
    """
    reference, reference_pml = build(label, seed)
    separate, separate_pml = build(label, seed)
    fused, fused_pml = build(label, seed)

    separate_residency = Residency()
    curl = cyl.plan_cylindrical_real_curl(separate, separate_pml, "step_D",
                                          separate_residency)
    electric = cyl.plan_cylindrical_real_constitutive(separate, separate_pml, "E",
                                                      separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_cylindrical_real_fused_electric_pair(
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

    seam = ("zero_metal_D",)
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
        "separate_binds_six_coefficient_pointers": True,
        "fused_binds_one_packed_coefficient_buffer": True,
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The deposit: a real electric source INSIDE the seam
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str = "Ez") -> Any:
    """A REAL engine source, so ``field_type`` is the engine's own answer.

    A stub with a hand-set ``field_type`` would let this leg pass while the engine
    classified the same component the other way; ``sources.VolumeSource`` resolves the
    slot itself.

    ``amplitude=1j`` IS A MEASUREMENT, NOT A DECORATION, and it is the difference
    between this leg measuring the repair and measuring nothing. A real-storage run
    injects ``real(amp * dt * current(t))`` (sources.py, ``_inject_points``), and
    ``ContinuousEnvelope``'s current carries the CW source's ``1/(-i*omega)`` factor —
    so with a REAL amplitude the current is PURELY IMAGINARY at every t and the real
    part deposited is ~1e-16. Measured on this fixture before the amplitude was
    changed: the injection moved 0 of 400 words, the unbracketed null control agreed
    with the array path to the bit, and the leg reported PASS while exercising no
    deposit at all. ``1j`` cancels that factor and the deposit becomes the O(dt)
    real current MEEP writes. THE FLOOR BELOW is what stops this from ever being a
    silence again: the injection must MOVE WORDS on both sides or the leg fails.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.5, 0.0, 0.5), size=(0.0, 0.0, 0.0),
                        amplitude=1j,
                        envelope=ContinuousEnvelope(frequency=1.0))


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    """``driver.py``'s withdraw pass, on both sides identically."""
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def run_deposit_case(label: str, seed: int, steps: int,
                     repair: bool = True) -> Dict[str, Any]:
    """Complete driver steps with an ELECTRIC source in the seam, byte compared.

    THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    ``plan_metal_cylindrical_real_fused_electric_pair`` directly would test the
    kernel and skip the wiring, and the wiring — ``launch._install_fused_pair``
    reaching this family through its new ``FUSED_PAIR_ARMS`` row — is what
    :data:`family.CARRIES_DEPOSIT_REPAIR` claims.
    """
    reference, reference_pml = build(label, seed)
    actual, actual_pml = build(label, seed)
    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    reference_sources = (_volume_source(reference),)
    actual_sources = (_volume_source(actual),)
    in_seam = deposit_repair.in_seam_sources(actual_sources, "D")
    assert len(in_seam) == 1, (
        "the source this leg builds is not in the D seam; the walk would inject "
        "nothing and compare a no-op with a no-op")
    assert deposit_repair._deposit_index(actual_sources[0]) is not None, (
        "the source publishes no deposit index; there would be nothing to repair")

    residency = Residency()
    composed = metal_launch.plan_step(actual, actual_pml, residency=residency,
                                      sources=(actual_sources if repair else ()),
                                      fuse=True)
    leading = composed.plans.get("step_D")
    trailing = composed.plans.get("update_E")
    installed = (type(leading).__name__, type(trailing).__name__)
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
    assert "step_D" in live and "update_E" in live, live
    assert not actual.grid.has_symmetry(), (
        "this fixture is folded; fill_symmetry_bc_D would be live and this family "
        "declares neither fill in REPLACES")

    before = frozen(actual)
    dt = float(actual.grid.dt)

    injected = 0
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
            if name == "step_D":
                for source in reference_sources:
                    source.inject(reference, when)

        _withdraw(actual_sources, actual)
        for name in live:
            if name == "step_D":
                residency.sync_in()
                leading.run()
                residency.sync_out()
                # THE FLOOR: the injection must MOVE WORDS. A source whose real
                # deposit rounds away would make this leg a no-op agreeing with a
                # no-op — measured once, and the reason `_volume_source` carries a
                # complex amplitude.
                pre = np.array(getattr(actual, "Dz"), copy=True)
                for source in actual_sources:
                    source.inject(actual, when)
                injected += differing(pre, getattr(actual, "Dz"))
            elif name == "update_E":
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
                       and installed_ok and injected > 0
                       and (repairs > 0 if repair else repairs == 0)),
        "words_the_injection_moved": injected,
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
        "source_field_type": str(actual_sources[0].field_type),
        "deposit_points": int(np.asarray(
            deposit_repair._deposit_index(actual_sources[0])[0]).size),
        "runs": inner.runs, "launches": inner.launches,
        "launches_per_run": inner.launches_per_run,
        "live_passes": list(live),
        "selected": {slot: composed.selected.get(slot)
                     for slot in ("step_D", "update_E")},
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
    """Replace ``count`` occurrences and REFUSE a no-op edit."""
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

    THE FIRST SIX ARE THE PACK'S OWN, and they exist because this product's prologue
    is new code rather than lifted text: the six aliases are the one place a wrong
    edit reads a valid float from the wrong vector, which is a plausible absorber and
    not a crash.
    """
    base = family.cylindrical_real_fused_electric_pair_source(codes, walls)
    edits: Dict[str, str] = {
        # --- THE PACK PROLOGUE ------------------------------------------------
        # THE ALIAS TAKES THE WRONG OFFSET, in the source rather than in the struct.
        # Reads sinvx where kmx belongs: a valid float from a real vector.
        "pack_alias_kmx_reads_sinvx":
            needle(base, "device const float* kmx = cpml + prm.off_kmx;",
                   "device const float* kmx = cpml + prm.off_sinvx;"),
        # THE AXIS PAIR SWAPPED IN THE PROLOGUE. x and z are both PML axes here, so
        # this is a real absorber transposition rather than a null.
        "pack_alias_x_and_z_swapped":
            needle(base, "device const float* kmx = cpml + prm.off_kmx;\n"
                         "    device const float* sinvx = cpml + prm.off_sinvx;",
                   "device const float* kmx = cpml + prm.off_kmz;\n"
                   "    device const float* sinvx = cpml + prm.off_sinvz;"),
        # THE OFFSET DROPPED ENTIRELY — the alias reads the pack's base, which is
        # kmx, so every axis reads the radial coefficients.
        "pack_alias_sinvz_loses_its_offset":
            needle(base, "device const float* sinvz = cpml + prm.off_sinvz;",
                   "device const float* sinvz = cpml;"),
        # --- THE SEAM ---------------------------------------------------------
        # Take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float src0 = v0 * ie0[ii];",
                   "float src0 = curl0 * ie0[ii];"),
        # THE SEAM, WRONG COMPONENT. Dx's constitutive source is v0, not v1.
        "seam_takes_the_wrong_component":
            needle(base, "float src1 = v1 * ie1[ii];",
                   "float src1 = v0 * ie1[ii];"),
        # THE INVERSE PERMITTIVITY DROPPED. update_E is (D * inv_eps), not D.
        "seam_drops_the_inverse_permittivity":
            needle(base, "float src2 = v2 * ie2[ii];", "float src2 = v2;"),
        # --- THE WALL CLEAR ---------------------------------------------------
        # z is the walled axis on the mutation case; the D table clears the two
        # TANGENTIAL components there, Dx and Dy.
        "zero_metal_dropped":
            needle(base, "    v0 = at_z ? 0.0f : v0;\n", ""),
        # THE WALL TABLE IS THE OFF-DIAGONAL FOR D. Clearing v2 on the z wall is the
        # B-side table's shape and the single most likely porting slip.
        "zero_metal_uses_the_b_side_table":
            needle(base, "    v0 = at_z ? 0.0f : v0;",
                   "    v2 = at_z ? 0.0f : v2;"),
        # THE WALL CLEAR MOVED AFTER THE CONSTITUTIVE READ. D ends up right and E
        # does not, which is the seam's ORDER rather than the clear's existence. TWO
        # edits, and both are needed: re-applying the clear at the store while the
        # shipped one still runs in front of it is BYTE-NEUTRAL (measured: 0 words),
        # because the register is already zero there. So the pre-store clear is
        # REMOVED and re-applied at the store, which is what "after" means.
        "zero_metal_after_the_constitutive_read":
            needle(needle(base, "    v0 = at_z ? 0.0f : v0;\n", ""),
                   "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f0[ii] = at_z ? 0.0f : v0; f1[ii] = v1; f2[ii] = v2;"),
        # --- THE m = 0 AXIS TAIL ----------------------------------------------
        # stepping._cylindrical_axis_zero_D:586-587, the POST-ADD.
        "axis_increment_dropped":
            needle(base, "    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;\n", ""),
        "axis_increment_is_folded_into_the_curl":
            needle(base, "    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;",
                   "    curl2 = at_x ? (curl2 + (axis_coef * g1[ii])) : curl2;"),
        "axis_dp_zero_dropped":
            needle(base, "    v1 = at_x ? 0.0f : v1;\n", ""),
        # --- Dz's PREFIX CURL --------------------------------------------------
        # stepping.step_D:425-426 — the BACKWARD difference of the prefix inside the
        # four-operand grouping, not Bz's two-operand forward difference.
        "dz_curl_is_the_generic_one":
            needle(base, "float curl2 = dtdx * ((pb_x - pb) + (a - a_y));",
                   "float curl2 = dtdx * ((b_x - b) + (a - a_y));"),
        "dz_prefix_difference_is_forward":
            needle(base, "float curl2 = dtdx * ((pb_x - pb) + (a - a_y));",
                   "float curl2 = dtdx * ((pb - pb_x) + (a - a_y));"),
        # --- THE OWNERSHIP MASK AND THE CURL ----------------------------------
        # THE OWNERSHIP MASK ON THE r AXIS, both components, and the pair is the
        # point. `cylindrical_real`'s docstring MEASURED that dropping the r mask on
        # Dz changes nothing (0/8): that curl's radial operand pair is
        # (prefix[0], the metallic zero ghost) and prefix row 0 is an exact +0.0 by
        # construction, while its phi pair is a self-difference on the one-cell
        # invariant axis -- so the row is already +0.0 before the mask runs. On Dy
        # the same edit diverges (11-40 words per case). Both are armed; the Dz one
        # is DECLARED a null below and the Dy one must be caught, which is what
        # stops the null reading as "the mask does not matter here".
        "ownership_mask_dropped_on_r_for_dz":
            needle(base, "    curl2 = at_x ? 0.0f : curl2;\n", ""),
        "ownership_mask_dropped_on_r_for_dy":
            needle(base, "    curl1 = at_x ? 0.0f : curl1;\n", ""),
        # shaders rule 2: flattening these parens is a different float32 number —
        # BUT ONLY WHERE BOTH DIFFERENCES ARE LIVE. Target 1's pair is (z, r).
        "curl_parens_flattened_on_a_live_pair":
            needle(base, "dtdx * ((a_z - a) + (c - c_x))",
                   "dtdx * (a_z - a + c - c_x)"),
        # THE SAME EDIT ON TARGET 0, whose first pair is the invariant phi axis.
        # Declared a NULL — see the expectation table.
        "curl_parens_flattened_on_the_invariant_pair":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        "recurrence_axis_pair_swapped":
            needle(base, "float n0 = ((p0 * km_y) - curl0) * si_y;",
                   "float n0 = ((p0 * km_z) - curl0) * si_z;"),
        # --- THE STORES -------------------------------------------------------
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "flux_store_dropped":
            needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f1[ii] = v1; f2[ii] = v2;"),
        "fw_store_dropped":
            needle(base, "    w0[ii] = src0;", "    // stale f_w_Ex"),
        # --- THE CONSTITUTIVE HALF --------------------------------------------
        # The dsigw index is the component's OWN axis. Moving it to the z index is
        # in bounds on a SQUARE case and is a wrong value.
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
                compile_source(source).cyl_real_fused_electric_pair_step}
    return mutants


#: Mutation id -> what it is DECLARED to do. Anything not named here must be CAUGHT.
#:
#: A DECLARED NULL MUST BE CONFIRMED, not merely permitted: a null that IS caught
#: means the reasoning behind it is wrong, and that has to fail too.
#:
#: THE THREE ENTRIES, and each has a live sibling beside it in the armed set:
#:
#: * ``curl_parens_flattened_on_the_invariant_pair`` — phi has n = 1 on a Dcyl grid,
#:   so the rolled operand IS the original and ``(c_y - c)`` is an exact ``+0.0``; the
#:   flattened form then differs only where ``(b - b_z)`` is ``-0.0``. Its sibling on
#:   target 1, whose pair is (z, r) and both live, is armed as must-catch.
#: * the two PHI OFFSET swaps — MEASURED on this host before they were armed:
#:   ``kms_y`` and ``sinv_y`` are each ONE element on a Dcyl grid and both hold
#:   exactly ``1.0f`` (0x3F800000), because the phi axis carries no PML at all. So
#:   pointing ``kmy`` at ``sinvy[0]`` (and ``sinvy`` back at ``kmy[0]``) reads the
#:   same word. Their siblings — the SAME two offsets moved the other way, into the
#:   neighbouring vector — are armed as must-catch and diverge, which is what stops
#:   these nulls reading as "the offsets do not matter".
#: * ``ownership_mask_dropped_on_r_for_dz`` — the r mask on Dz is redundant, because
#:   that curl's radial pair is (prefix[0], the metallic zero ghost) and prefix row 0
#:   is an exact ``+0.0``. Measured 0/8 by ``cylindrical_real``'s own probe and
#:   re-measured here; its sibling on Dy is armed as must-catch.
MUTATION_EXPECTATION: Dict[str, str] = {
    "curl_parens_flattened_on_the_invariant_pair": "null",
    "ownership_mask_dropped_on_r_for_dz": "null",
    "offset_kmy_plus_one_reads_sinvy": "null",
    "offset_sinvy_minus_one_reads_kmy": "null",
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
    source = family.cylindrical_real_fused_electric_pair_source(codes, walls)
    for target in range(3):
        source = needle(source, f"float src{target} = v{target} * ie{target}[ii];",
                        f"float src{target} = f{target}[ii] * ie{target}[ii];")
    return source


#: Where each binding group starts in the plan's fused argument tuple. Spelled once,
#: so the host mutations and the kernel signature cannot drift apart.
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(20, 26))
PACK_SLOT = 19


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._fused_args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._fused_args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: pack the HALF-INTEGER split-field coefficients for the D curl.

    ``step_D`` reads INTEGER positions and ``step_B`` half-integer ones
    (``SUB_STEPS['step_D']['suffix'] == ''``). The kernel cannot tell: it is a
    half-cell error in the absorber profile, not a crash. Reached by rebuilding the
    PACK from the other lattice, which is where this family's version of the defect
    now lives — the six pointers became one, so the six-way swap became a one-way one.
    """
    tensor, _layout = coefficient_pack.packed_mirror(
        residency, "mutation:pack_half_integer", np, family.PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kms", "sinv")])
    _rebind(plan, (PACK_SLOT,), (tensor,))


def swap_e_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the INTEGER constitutive coefficients to the E half.

    ``update_E`` takes ``kps_a_h``/``kms_a_h`` and ``update_H`` the integer ones
    (stepping.py:1015 vs :948). The kernel takes six pointers and never asks which
    lattice they came from, so no shader mutation can reach this.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_pack_takes_the_half_integer_lattice": swap_curl_lattice,
    "e_half_takes_the_integer_lattice": swap_e_lattice,
}


def _offset_mutation(index: int, delta: int) -> Callable[[Any], Any]:
    """A ``params`` record whose ``index``-th pack offset is off by ``delta``.

    THE HAZARD THE PACK INTRODUCES, ARMED. Six vectors in one allocation are told
    apart by six uints and nothing else, so an offset that is wrong by one element
    reads a valid float from a real vector — the exact shape of a defect that
    produces a plausible absorber rather than a fault. Built from the SHIPPED plan's
    own layout, so a leg cannot arm an offset the packer never produced.
    """
    def make(plan: Any) -> Any:
        offsets = list(plan.pack_layout.offsets)
        moved = offsets[index] + delta
        if moved < 0:
            raise AssertionError(
                f"offset {index} is already 0; a negative offset is a fault rather "
                f"than a wrong value and this leg would measure the platform")
        offsets[index] = moved
        return family._params_tensor(plan.shape, plan.dtdx, plan.axis_coef,
                                     offsets, plan.residency.device)
    return make


#: id -> (pack index, delta). ELEVEN, which is every one-element move of every offset
#: except the one that would take ``off_kmx`` negative — a negative offset reads
#: before the allocation, which is a fault rather than a wrong value, and a leg that
#: armed it would be measuring the platform's bounds behaviour.
OFFSET_MUTATIONS: Dict[str, Tuple[int, int]] = {
    "offset_kmx_plus_one_reads_into_kmx": (0, 1),
    "offset_sinvx_plus_one": (1, 1),
    "offset_sinvx_minus_one": (1, -1),
    "offset_kmy_plus_one_reads_sinvy": (2, 1),
    "offset_kmy_minus_one_reads_sinvx": (2, -1),
    "offset_sinvy_plus_one_reads_kmz": (3, 1),
    "offset_sinvy_minus_one_reads_kmy": (3, -1),
    "offset_kmz_plus_one": (4, 1),
    "offset_kmz_minus_one": (4, -1),
    "offset_sinvz_plus_one": (5, 1),
    "offset_sinvz_minus_one": (5, -1),
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def _sweep_source(pointers: int) -> str:
    """A kernel with ``pointers`` buffers plus one packed ``Params&``.

    The body touches every buffer, because a body the compiler could drop would let
    dead-code elimination decide where the ceiling is.
    """
    lines = [f"    device float* p{n} [[buffer({n})]]," for n in range(pointers)]
    touch = " + ".join(f"p{n}[0]" for n in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n_elem; float dtdx; };", "",
        "kernel void ceiling_sweep(", *lines,
        f"    constant Params& prm [[buffer({pointers})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float touch = {touch};",
        *(f"    p{n}[idx] = p{n}[idx] + touch * prm.dtdx;" for n in range(pointers)),
        "}", ""))


def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, (message.splitlines()[0] if message else "")


def leg_binding_ceiling() -> Dict[str, Any]:
    """Three signatures compiled, and the ceiling BISECTED on this host.

    THIS IS THE LEG THE PACK EXISTS FOR, and it measures four things rather than
    asserting one:

    1. the SEPARATE-SCALAR shape (37 bindings) FAILS — the shape scalar packing
       already exists to avoid;
    2. the UNPACKED-POINTER shape (31 pointers + one ``Params&`` = 32 bindings)
       FAILS. This is the fused pair built the way every other D/E product on this
       backend is built, and it is over by EXACTLY ONE binding. It is the measurement
       behind the board's ``UNFUSABLE ON METAL`` verdict for this cell, re-taken here
       rather than cited;
    3. the SHIPPED packed shape (27 bindings) COMPILES, LAUNCHES, and reads every
       struct field back — including the six pack offsets, from the REAL packer;
    4. THE CEILING ITSELF, by bisection: N pointers plus one ``Params&`` for N around
       the limit, so the largest N that compiles and the smallest that does not are
       both numbers this host returned. ``MAX_BUFFER_BINDINGS`` is then checked
       against the measurement instead of standing in for it.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "declared_max_buffer_bindings": MAX_BUFFER_BINDINGS,
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "unpacked_pointer_bindings": family.UNPACKED_POINTER_BINDINGS,
        "unpacked_pointers": family.UNPACKED_POINTERS,
        "packed_bindings": family.PACKED_BINDINGS,
        "packed_pointers": family.PACKED_POINTERS,
    }

    separate_ok, separate_error = _compiles(family.refuted_separate_scalar_source())
    row["separate_scalar_compiled"] = separate_ok
    row["separate_scalar_error"] = separate_error
    row["separate_refused_for_the_right_reason"] = (
        not separate_ok and "out of bounds" in separate_error
        and "buffer" in separate_error)

    unpacked_ok, unpacked_error = _compiles(family.refuted_unpacked_pointer_source())
    row["unpacked_pointer_compiled"] = unpacked_ok
    row["unpacked_pointer_error"] = unpacked_error
    row["unpacked_refused_for_the_right_reason"] = (
        not unpacked_ok and "out of bounds" in unpacked_error
        and "buffer" in unpacked_error)

    # THE BISECTION. Both directions, and the pair of them is what pins the ceiling:
    # the largest signature that compiles and the smallest that does not.
    sweep: Dict[str, bool] = {}
    for pointers in range(MAX_BUFFER_BINDINGS - 3, MAX_BUFFER_BINDINGS + 2):
        ok, _error = _compiles(_sweep_source(pointers))
        sweep[f"{pointers}_pointers_{pointers + 1}_bindings"] = ok
    row["ceiling_sweep"] = sweep
    largest = max((int(key.split("_")[0]) for key, ok in sweep.items() if ok),
                  default=-1)
    smallest = min((int(key.split("_")[0]) for key, ok in sweep.items() if not ok),
                   default=-1)
    row["largest_pointer_count_that_compiles_with_one_struct"] = largest
    row["smallest_pointer_count_refused_with_one_struct"] = smallest
    ceiling_ok = (largest + 1 == MAX_BUFFER_BINDINGS
                  and smallest == largest + 1)
    row["ceiling_measured_equals_declared"] = ceiling_ok
    # AND THE ARITHMETIC THE MODULE ARGUES FROM: the cell is over by exactly one.
    row["over_the_ceiling_by"] = family.UNPACKED_POINTERS - largest
    row["pack_saves_pointers"] = family.UNPACKED_POINTERS - family.PACKED_POINTERS

    shipped_ok, shipped_error = _compiles(
        family.cylindrical_real_fused_electric_pair_source((1, 0, 1),
                                                           (False, False, True)))
    row["shipped_compiled"] = shipped_ok
    row["shipped_error"] = shipped_error

    # THE STRUCT MUST BE READ CORRECTLY, not merely bound. A signature that compiled
    # but decoded the record wrongly would produce a smooth, plausible, wrong field,
    # so every field is written out to its own buffer and read back — the six
    # OFFSETS included, because they are what the pack rests on.
    names = list(family.PACKED_VECTORS)
    probe_fields = ["nx", "ny", "nz", "n_elem", "dtdx", "axis_coef"] + [
        f"off_{name}" for name in names]
    pointers = "\n".join(f"    device float*    b{n:<6}[[buffer({n})]],"
                         for n in range(len(probe_fields)))
    struct = ("struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;"
              " float axis_coef;\n"
              + coefficient_pack.params_fields(names) + "\n};")
    reads = "\n".join(f"    b{n}[idx] = float(prm.{name});"
                      for n, name in enumerate(probe_fields))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "", struct, "",
        "kernel void packed_probe(", pointers,
        f"    constant Params& prm [[buffer({len(probe_fields)})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }", reads, "}", ""))
    probe_ok, probe_error = _compiles(source)
    if not probe_ok:
        row.update(packed_compiled=False, packed_error=probe_error, passed=False)
        return row
    function = compile_source(source).packed_probe
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in probe_fields]
    shape, dtdx, axis_coef = (3, 4, 5), 0.125, 0.5
    offsets = tuple(range(11, 11 + len(names)))
    # The REAL packer, not a hand-built record: the field the kernel reads is then
    # the field the plan builder writes. n_elem is the product of the shape (60),
    # deliberately not the eight-element probe buffers, so the guard is exercised.
    params = family._params_tensor(shape, dtdx, axis_coef, offsets, "mps")
    function(*buffers, params)
    torch.mps.synchronize()
    read = [float(buffer.cpu().numpy()[0]) for buffer in buffers]
    expected = [float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]),
                float(np.float32(dtdx)), float(np.float32(axis_coef))] + [
        float(offset) for offset in offsets]
    fields_ok = read == expected
    row.update(packed_launched=True,
               packed_fields=probe_fields,
               packed_fields_read_back=read,
               packed_fields_expected=expected,
               packed_fields_correct=fields_ok,
               passed=bool(row["separate_refused_for_the_right_reason"]
                           and row["unpacked_refused_for_the_right_reason"]
                           and shipped_ok and ceiling_ok and fields_ok
                           and row["over_the_ceiling_by"] == 1))
    return row


def leg_pack_identity(label: str, seed: int) -> Dict[str, Any]:
    """The packed buffer IS the six separate vectors, in bytes and in reads.

    TWO MEASUREMENTS, because either alone would be a silence:

    * **the BYTES.** The pack's host array, sliced at the layout's offsets, must be
      uint32-identical to each vector ``Residency.mirror`` would have flattened on its
      own. That is the claim ``build_pack`` makes by construction, checked rather than
      trusted.
    * **the READS.** A kernel that binds six pointers and one that binds the pack and
      re-creates the six names through the prologue must write the same words, for
      every cell of every axis. That is the claim the PROLOGUE makes, and no amount of
      byte equality on the host establishes it — the offsets are applied on the
      device.
    """
    import torch  # noqa: PLC0415

    fields, pml = build(label, seed)
    residency = Residency()
    vectors = [np.ascontiguousarray(getattr(pml, f"{stem}_{axis}"),
                                    dtype=np.float32).reshape(-1)
               for axis in "xyz" for stem in ("kms", "sinv")]
    tensor, layout = coefficient_pack.packed_mirror(
        residency, "pack_identity", np, family.PACKED_VECTORS, vectors)
    host = residency.host("pack_identity")
    slices = {name: np.asarray(host)[offset:offset + length]
              for name, offset, length in zip(layout.names, layout.offsets,
                                              layout.lengths)}
    byte_diff = {name: differing(slices[name], vector)
                 for name, vector in zip(layout.names, vectors)}
    lengths_ok = layout.total == sum(v.size for v in vectors)

    # THE READS. Two kernels, one grid, one comparison.
    n = max(v.size for v in vectors)
    separate_src = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "kernel void separate(",
        "    device float* out [[buffer(0)]],",
        *(f"    device const float* {name} [[buffer({i + 1})]],"
          for i, name in enumerate(family.PACKED_VECTORS)),
        f"    constant uint& n [[buffer({len(family.PACKED_VECTORS) + 1})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= n) { return; }",
        "    float acc = 0.0f;",
        *(f"    acc = acc * 3.0f + {name}[idx % {v.size}u];"
          for name, v in zip(family.PACKED_VECTORS, vectors)),
        "    out[idx] = acc;", "}", ""))
    packed_src = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n;\n"
        + coefficient_pack.params_fields(family.PACKED_VECTORS) + "\n};", "",
        "kernel void packed(",
        "    device float* out [[buffer(0)]],",
        "    device const float* cpml [[buffer(1)]],",
        "    constant Params& prm [[buffer(2)]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n) { return; }",
        coefficient_pack.prologue("cpml", family.PACKED_VECTORS),
        "    float acc = 0.0f;",
        *(f"    acc = acc * 3.0f + {name}[idx % {v.size}u];"
          for name, v in zip(family.PACKED_VECTORS, vectors)),
        "    out[idx] = acc;", "}", ""))
    separate = compile_source(separate_src).separate
    packed = compile_source(packed_src).packed
    out_a = torch.zeros(n, dtype=torch.float32, device="mps")
    out_b = torch.zeros(n, dtype=torch.float32, device="mps")
    mirrors = [residency.mirror(f"identity:{name}", vector, constant=True)
               for name, vector in zip(family.PACKED_VECTORS, vectors)]
    count = torch.from_numpy(np.array([n], dtype=np.int32)).to("mps")
    record = np.zeros(1, dtype=np.dtype(
        [("n", "<u4")] + coefficient_pack.record_dtype_fields(family.PACKED_VECTORS)))
    record[0] = (n,) + tuple(layout.offsets)
    prm = torch.from_numpy(
        np.frombuffer(record.tobytes(), dtype=np.int32).copy()).to("mps")
    separate(out_a, *mirrors, count)
    packed(out_b, tensor, prm)
    torch.mps.synchronize()
    read_diff = differing(out_a.cpu().numpy(), out_b.cpu().numpy())
    moved = int(np.count_nonzero(out_a.cpu().numpy() != 0.0))
    return {
        "passed": bool(not any(byte_diff.values()) and lengths_ok
                       and read_diff == 0 and moved > 0),
        "layout": layout.describe(),
        "bytes_differing_per_vector": byte_diff,
        "packed_total_elements": layout.total,
        "separate_total_elements": int(sum(v.size for v in vectors)),
        "lengths_agree": lengths_ok,
        "read_differing_words": read_diff,
        "cells_compared": n,
        "nonzero_outputs": moved,
    }


def leg_transcription(codes: Sequence[int], walls: Sequence[bool]) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes.

    The module claims its arithmetic is lifted rather than retyped. That is a
    property of the construction and therefore checkable, so it is checked:

    * the certified cylindrical ``step_D`` curl body must appear in the fused source
      VERBATIM on both sides of the spliced wall clear;
    * the constitutive half must differ from the certified ``update_E`` body in
      EXACTLY the three ``float srcN =`` lines.

    AND THE PACK MAY NOT HAVE REACHED THE ARITHMETIC. The prologue re-creates the six
    names, so the lifted text is unchanged — which means the fused source must
    contain no ``prm.off_`` outside the prologue block, and the six aliases must be
    declared before the first line of the lifted curl body.
    """
    source = family.cylindrical_real_fused_electric_pair_source(codes, walls)
    curl = family.certified_cyl_curl_body(codes)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    certified = family.certified_constitutive_body().splitlines()
    marker = "    // --- update_E (stepping.update_E"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float src{t} = g{t}[ii] * ie{t}[ii];",
                 f"    float src{t} = v{t} * ie{t}[ii];") for t in range(3)]
    lengths_match = len(certified) == len(spliced)

    prologue = coefficient_pack.prologue("cpml", family.PACKED_VECTORS)
    prologue_present = prologue in source
    offsets_outside = [line for line in source.splitlines()
                       if "prm.off_" in line and line not in prologue.splitlines()]
    # AGAINST THE WHOLE LIFTED HEAD, not against one of its lines: a single line of
    # the certified body can also occur inside this family's own template preamble,
    # and `str.index` would then find the template's copy and report the prologue as
    # late. The head is unique by construction — the assertion above just checked it
    # appears verbatim.
    prologue_first = (source.index(prologue) < source.index(head)
                      if prologue_present and head in source else False)
    return {
        "passed": bool(curl_head_present and curl_tail_present and lengths_match
                       and changed == expected and prologue_present
                       and not offsets_outside and prologue_first),
        "curl_body_head_verbatim": curl_head_present,
        "curl_body_tail_verbatim": curl_tail_present,
        "certified_constitutive_lines": len(certified),
        "spliced_constitutive_lines": len(spliced),
        "constitutive_lines_changed": [list(pair) for pair in changed],
        "constitutive_lines_expected": [list(pair) for pair in expected],
        "pack_prologue_verbatim": prologue_present,
        "pack_prologue_precedes_the_lifted_body": prologue_first,
        "offset_reads_outside_the_prologue": offsets_outside,
    }


def leg_inert_passes() -> Dict[str, Any]:
    """The two driver passes :data:`REPLACES` omits, EXECUTED and measured.

    NON-VACUITY: the CONTROL is ``zero_metal_D``, which is NOT inert on a walled case
    and must move something, so an inert pass cannot be confused with a frozen state.
    On a z-PERIODIC case there is no wall at all and the control is skipped by name.
    """
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for offset, (label, _options) in enumerate(CASES):
        fields, pml = build(label, 81000 + offset)
        verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
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
        stepping.zero_metal_D(fields)
        control = state_of(fields)
        control_moved = sorted(name for name in middle
                               if differing(middle[name], control[name]))
        walls = zero_metal_axes(fields.grid)
        rows.append({"case": label, "shape": list(fields.grid.shape),
                     "walls": list(walls),
                     "moved_by_the_inert_passes": moved,
                     "moved_by_zero_metal_D": control_moved})
        if moved:
            findings.append(
                f"{label}: {sorted(family.INERT_PASSES)} moved {moved} — they are "
                f"NOT inert on this grid and REPLACES may not omit them")
        if any(walls) and not control_moved:
            findings.append(
                f"{label}: zero_metal_D moved nothing on a WALLED grid, so this leg "
                f"cannot tell an inert pass from a frozen state")
    return {"passed": not findings, "rows": rows, "findings": findings,
            "inert_passes": sorted(family.INERT_PASSES)}


def leg_refusal() -> Dict[str, Any]:
    """The source seam, the order and the geometry, each measured by name.

    SEVEN QUESTIONS, and the second is the one this family exists for:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real ELECTRIC ``VolumeSource`` must be ADMITTED, and the admission must come
       WITH THE WIRING: ``plan_step(..., fuse=True)`` must put a ``LeadingRepairPlan``
       in ``step_D`` and a ``TrailingRepairPlan`` in ``update_E``;
    3. an electric source that does NOT publish the index it writes must still be
       refused BY NAME, and by the PLAN as well as the predicate;
    4. a real MAGNETIC ``VolumeSource`` is not this seam's business and must
       contribute no refusal — and its MAGNETIC TWIN must refuse the same source,
       which is the pairing that proves neither family copied the other's clause;
    5. an |m| >= 1 Dcyl grid and a CARTESIAN grid must both be refused by name;
    6. the ABSORB ROW must exist and name the two arms this predicate is built out of;
    7. the family must declare :data:`CARRIES_DEPOSIT_REPAIR`.
    """
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (electric,), residency)
    electric_plan = family.plan_metal_cylindrical_real_fused_electric_pair(
        fields, pml, sources=(electric,), residency=residency)

    wired_fields, wired_pml = build(REFUSAL_CASE, 91001)
    wired_source = _volume_source(wired_fields, "Ez")
    composed = metal_launch.plan_step(wired_fields, wired_pml, residency=Residency(),
                                      sources=(wired_source,), fuse=True)
    installed = [type(composed.plans.get(slot)).__name__
                 for slot in ("step_D", "update_E")]
    wiring_ok = installed == ["LeadingRepairPlan", "TrailingRepairPlan"]

    indexless = _volume_source(fields, "Ez")
    indexless._point_ix = indexless._point_iy = indexless._point_iz = None
    indexless_coverage = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (indexless,), residency)
    indexless_named = [r for r in indexless_coverage.reasons
                       if "does not publish the index it writes" in r]
    indexless_plan = family.plan_metal_cylindrical_real_fused_electric_pair(
        fields, pml, sources=(indexless,), residency=residency)

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (magnetic,), residency)
    # THE MIRROR IMAGE, asked of the magnetic twin rather than asserted about it.
    sibling_magnetic = sibling.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), Residency())
    sibling_electric = sibling.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (electric,), Residency())

    m_fields, m_pml = matrix.cylindrical(m=1, complex_storage=True)
    m_coverage = family.metal_cylindrical_real_fused_electric_pair_coverage(
        m_fields, m_pml, (), Residency())
    m_named = [r for r in m_coverage.reasons if "grid.m = 1" in r]

    cart_fields, cart_pml = matrix.cart()
    cart_coverage = family.metal_cylindrical_real_fused_electric_pair_coverage(
        cart_fields, cart_pml, (), Residency())
    cart_named = [r for r in cart_coverage.reasons if "cylindrical" in r]

    absorb_row = metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY)
    seam_row = metal_launch.FUSED_PAIR_SEAMS.get(family.SLOT)

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and electric_coverage.covered and electric_plan is not None
                       and wiring_ok
                       and not indexless_coverage.covered and indexless_named
                       and indexless_plan is None
                       and magnetic_coverage.covered
                       and not any("source 0" in r
                                   for r in magnetic_coverage.reasons)
                       and not sibling_magnetic.covered
                       and sibling_electric.covered
                       and not m_coverage.covered and m_named
                       and not cart_coverage.covered and cart_named
                       and absorb_row == ("cylindrical m=0", "cylindrical m=0")
                       and seam_row == ("update_E", "D")
                       and family.CARRIES_DEPOSIT_REPAIR),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_plan_built": electric_plan is not None,
        "installed_plans": installed,
        "wiring_installs_the_bracket": wiring_ok,
        "indexless_refused": not indexless_coverage.covered,
        "indexless_named": indexless_named,
        "indexless_plan_is_none": indexless_plan is None,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_admitted_here": magnetic_coverage.covered,
        "magnetic_source_refused_by_the_twin": not sibling_magnetic.covered,
        "electric_source_admitted_by_the_twin": sibling_electric.covered,
        "m_is_one_refused": not m_coverage.covered,
        "m_named": m_named[:2],
        "cartesian_refused": not cart_coverage.covered,
        "cartesian_named": cart_named[:2],
        "absorb_row": list(absorb_row) if absorb_row else None,
        "seam_row": list(seam_row) if seam_row else None,
        "family_carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
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
                              ).cyl_real_fused_electric_pair_step}

    controls = ("square_metallic", "square_periodic")
    # binding_ceiling, pack_identity, transcription, inert_passes, the two case
    # sweeps, the subnormal ladder, the separate controls, deposit,
    # deposit_null_control, refusal, byte_neutral_control, every mutation, disarm.
    total = (4 + 2 * len(CASES) + 1 + len(controls) + 2 + 1 + 1
             + len(mutants) + len(HOST_MUTATIONS) + len(OFFSET_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "twenty_six_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "pack_identity",
                     "label": "one_buffer_reads_as_six_vectors",
                     **leg_pack_identity(MUTATION_CASE, 51000)})
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
        rows.append({"index": index, "total": total, "leg": "deposit",
                     "label": "a_real_electric_source_inside_the_seam",
                     "steps": args.steps,
                     **run_deposit_case(DEPOSIT_CASE, 69000, args.steps)})
        emit(handle, rows[-1])

        # THE NULL CONTROL. The SAME launch with the trailing repair replaced by the
        # NoopPlan the False branch would leave there. Scored on DIVERGING.
        index += 1
        result = run_deposit_case(DEPOSIT_CASE, 69001, args.steps, repair=False)
        diverged = not result.get("bit_identical", False)
        row = {"index": index, "total": total, "leg": "deposit_null_control",
               "label": "the_unbracketed_launch_must_diverge",
               "diverged": diverged,
               "words_the_injection_moved": result.get("words_the_injection_moved"),
               "passed": bool(diverged and result.get("launches")
                              and result.get("words_the_injection_moved")),
               "installed_plans": result.get("installed_plans"),
               "first_divergence": result.get("first_divergence"),
               "differing_words": result.get("differing_words"),
               "differing_arrays": result.get("differing_arrays"),
               "deposit_points_repaired": result.get("deposit_points_repaired"),
               "launches": result.get("launches")}
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

        for name, (slot, delta) in OFFSET_MUTATIONS.items():
            index += 1
            result = run_case(MUTATION_CASE, 72000 + index, args.steps,
                              params=_offset_mutation(slot, delta))
            caught = not result.get("bit_identical", False)
            wanted = mutation_must_be_caught(name)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "offset_defect": True,
                   "pack_index": slot, "pack_vector": family.PACKED_VECTORS[slot],
                   "delta": delta, "caught": caught,
                   "expectation": "caught" if wanted else "null",
                   "passed": bool(caught is wanted and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no patch. A nonzero here would mean every "caught" above is a
        # harness that diverges on its own.
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
                   "offset_mutations": len(OFFSET_MUTATIONS),
                   "byte_neutral_controls": 1, "deposit_null_controls": 1},
        "mutation_case": MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes), "zero_metal": list(walls)},
        "packed_vectors": list(family.PACKED_VECTORS),
        "binding_counts": {
            "separate_scalars": family.SEPARATE_SCALAR_BINDINGS,
            "unpacked_pointers_plus_struct": family.UNPACKED_POINTER_BINDINGS,
            "packed": family.PACKED_BINDINGS,
            "ceiling": MAX_BUFFER_BINDINGS,
        },
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "admit_both_halves": 3,
            "and_carrying_an_electric_deposit": 3,
            "note": "the same three rows the MAGNETIC twin serves, on the other "
                    "seam. All three declare an ELECTRIC source, so with "
                    "CARRIES_DEPOSIT_REPAIR at False this product would admit ZERO. "
                    "3 is an upper bound because the census records source FIELD "
                    "TYPES but not whether a row's grid would also satisfy the "
                    "residency and wall-readability clauses this predicate adds.",
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "pack": hashlib.sha256(
                Path(coefficient_pack.__file__).read_bytes()).hexdigest(),
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
