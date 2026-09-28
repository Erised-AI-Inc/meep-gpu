#!/usr/bin/env python3
"""Native MPS byte gate for the COMPLEX fused magnetic Metal kernel: ``step_B`` -> ``update_H``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans THREE driver
passes — ``step_B``, ``zero_metal_B`` and ``update_H`` — so a per-sub-step comparison
could not see it at all: the whole claim is about the SEAM between them. The
comparison is therefore per COMPLETE DRIVER STEP over a stated budget, against an
array-path oracle running the identical live pass list, with the first divergent step
reported rather than a final pass/fail.

THE NINE THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that was
   never launched is byte-identical to the oracle BY CONSTRUCTION, because the oracle
   is the array path this walk would otherwise take. So every case asserts the exact
   LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``, and every case
   asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** Shader mutations and host-binding mutations are ARMED, and each
   must be CAUGHT. Leg ``disarm`` reruns the identical case with the shipped bytes and
   no host patch and requires zero, so a mutation reported as caught cannot be a
   harness that diverges anyway.
3. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no lever,
   so byte-identity is claimed subject to a CHECKED subnormal-free precondition; every
   case censuses the oracle's own state and reports it. The signed-zero seeding is
   censused too, because a complex kernel's zero cross terms are only under test if
   signed zeros are actually present.
4. **An unmeasured platform assumption — TWICE OVER, because this family sits at the
   binding ceiling from two directions.** Leg ``binding_ceiling`` compiles the
   35-binding separate-scalar/phase signature AND the 53-binding re/im-split signature
   and requires BOTH to fail, then compiles and LAUNCHES the shipped 28-binding one.
   That is what makes "float2 volumes and a packed struct are FORCED" a measurement
   rather than a preference.
5. **A struct that binds wrongly.** Leg ``params_layout`` reads every ``Params`` field
   back off the device and compares it to what the host wrote. This is not ceremony:
   the phases are ``float2`` and Metal aligns them to 8 bytes, so the natural 44-byte
   NumPy record puts every phase ONE WORD EARLY — a well-formed, plausible complex
   number rather than garbage. That record is carried as a MUST-CATCH mutation.
6. **A transcription that drifted.** Both halves are LIFTED from the certified COMPLEX
   emitters rather than retyped, and leg ``transcription`` MEASURES that: the certified
   curl body must appear in the fused source verbatim on both sides of the spliced wall
   clear, and the constitutive half must differ from the certified H body in EXACTLY
   the three seam lines and nothing else.
7. **An arm that was chosen rather than measured.** Leg ``expansion`` records which
   complex-multiply arm was bound, from which artifact, and requires that a MISSING and
   an AMBIGUOUS probe both refuse. It also states the consequence the phased cases
   exist for: on an unphased row the two arms coincide bit for bit, so a gate that ran
   only ``k = 0`` would certify the arm binding VACUOUSLY.
8. **A seam that is not really a seam.** Leg ``byte_neutral_control`` replaces the three
   register reads with reloads of the very words the curl half just stored. That mutant
   must NOT diverge — it is the same value by construction — which is what makes "the
   fusion removes a round trip and changes no arithmetic" a measurement instead of a
   claim. It is the one armed edit in this file required to be UNCAUGHT.
9. **A refusal that is really an omission.** Leg ``refusal`` measures the magnetic
   source slot — this family's binding clause on the measured corpus — plus the fold
   and the real-storage domain split, each by name.

A MEASURED LIMIT OF THIS GATE, STATED UP FRONT RATHER THAN DISCOVERED. Two refuted
spellings — folding the literal zero cross terms, and spelling negation ``0.0f - x`` —
are genuinely different arithmetic on this toolchain and genuinely change the bits this
kernel computes, but the difference they produce is a SIGN OF ZERO and the certified
recurrence's next operation is an addition. Measured 2026-08-19: the folded inner
product differs at 282/5040 cells on the real seeded field, and after one ``- curl0``
with a generic nonzero curl 0/5040 survive. A complete-step byte compare therefore
CANNOT see either defect. They are carried as PREDICTED NULLS with that reason, and the
defect itself is put under test by leg ``zero_spellings`` at the point of use, where it
is observable. Neither deleting them nor reporting them as caught would be honest.

LEGS
  0  expansion            the arm, the artifact that bound it, and the three refusals
  1  binding_ceiling      35 separate and 53 split-plane bindings must BOTH fail; the
                          shipped 27 pointers + one packed Params& must COMPILE and LAUNCH
  2  params_layout        every Params field must read back off the device; the naive
                          44-byte record must be CAUGHT
  3  zero_spellings       the two signed-zero spellings measured at the point of use,
                          and the annihilation that hides them from a walk
  4  transcription        both halves must be the certified complex emitters' own bytes
  5  product              complete steps, per-step byte compare, launch counters,
                          movement, subnormal and signed-zero census
  6  separate_control     the two ALREADY CERTIFIED complex products stepping the same
                          seam as separate dispatches: three-way byte agreement
  7  refusal              the source seam, the fold and the storage split, by name
  8  byte_neutral_control the register-vs-reload edit must NOT diverge
  9  mutation             armed defects, each of which MUST diverge
 10  predicted_null       the two signed-zero spellings, required to be NULL, with the
                          measured reason on the row
 11  disarm               the same harness, shipped bytes, must not diverge

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

#: The probe paths and the subnormal policy, resolved ONCE, the way every other
#: complex-family gate resolves them. ``setdefault`` throughout, so an explicit export
#: from ``recut_metal_gates.sh`` still wins and a standalone run is not a different
#: measurement from a campaign run. Recorded in the artifact: which probe bound the arm
#: is part of the result, not part of the setup.
ENVIRONMENT = matrix.prepare_environment()  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields, complex_fused_magnetic_pair as family, launch as metal_launch,
    shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs. Twelve, matching the real twin's and the whole-step
#: arbiter's, and for their reason: the classes this gate exists for COMPOUND.
#: ``fu_B`` and ``f_w_H`` are state carried between steps, and a coefficient index off
#: by one axis needs several steps to reach the low bits of the interior. The
#: comparison is per COMPLETE STEP.
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

#: Every stored volume a complete step can touch. The ``fu_*`` PML auxiliaries and the
#: ``f_w_*`` constitutive workspaces are STATE: a kernel right for one launch and wrong
#: forever after diverges only once they accumulate. The D/E half is in this list even
#: though this family does not touch it, because a fused pair that corrupted B would
#: reach E through ``step_D`` on the very next step, and a comparison blind to that
#: would be reporting on half the engine.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> ``metal_composition_matrix.cart`` keywords.
#:
#: WHAT THE ROWS CARRY BETWEEN THEM, and the phase axis is the one that matters here:
#:
#:   k = 0          the REDUCTION. No phase object is built, the kernel emits no
#:                  rotation at all, and the fused pair must be byte-identical to the
#:                  plain complex composition. It is also the row on which the
#:                  expansion arm is INERT, which is exactly why it cannot be the only
#:                  row;
#:   k = 0.3        a generic interior point — the phase is neither real nor a root of
#:                  unity, so both planes of the rotation are live and the
#:                  FMA_V1/NAIVE arms actually separate;
#:   k = 0.5        the BRILLOUIN EDGE, where ``grid.bloch_phase`` is EXACTLY -1+0j.
#:                  The imaginary plane is an exact zero there, which is the one phase
#:                  value at which a plane-wise collapse would be invisible on random
#:                  data — a required case, not a sampled one;
#:   THREE AXES     ``ph111`` is a distinct compiled specialisation and the y branch in
#:                  particular is easy to emit and never launch;
#:   WALLS          ``zero_metal_B`` emits a line per metallic axis, and for B the
#:                  table is the DIAGONAL — Bx on x, By on y, Bz on z. The rows span no
#:                  wall, one wall and all three;
#:   PML ON A PHASED AXIS is ADMITTED by the array path (settled by measurement,
#:                  S:2322-2337), so a gate that only ran unabsorbed phased axes would
#:                  not be testing what ships. Every periodic row here is absorbed.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("periodic_k0", dict(complex_storage=True)),
    ("periodic_kx", dict(complex_storage=True, k_point=(0.3, 0.0, 0.0))),
    ("periodic_edge_x", dict(complex_storage=True, k_point=(0.5, 0.0, 0.0))),
    ("periodic_ky", dict(complex_storage=True, k_point=(0.0, 0.4, 0.0))),
    ("periodic_kxyz", dict(complex_storage=True, k_point=(0.3, -0.4, 0.2))),
    ("wall_x_kz", dict(complex_storage=True, k_point=(0.0, 0.0, 0.4),
                       boundaries={"x": "metallic"})),
    ("wall_xyz_k0", dict(complex_storage=True,
                         boundaries={"x": "metallic", "y": "metallic",
                                     "z": "metallic"})),
)

#: The case the mutations are armed on. ALL THREE axes phased and none walled, so every
#: rotation line the shipped kernel can emit is present on a phase value that is
#: neither real nor a root of unity — which is what makes the ARM and the OPERAND ORDER
#: mutations reachable. The wall mutations are armed separately on the walled case,
#: because a wall row emits no line here.
MUTATION_CASE = "periodic_kxyz"

#: The case the WALL mutations are armed on. All three axes metallic, so all three
#: wall-clear rows and the full ownership mask are emitted. A wall mutation armed on
#: :data:`MUTATION_CASE` would report as uncaught for the uninteresting reason that no
#: line is emitted there.
WALL_MUTATION_CASE = "wall_xyz_k0"

#: The row leg ``refusal`` builds its source and fold questions on.
REFUSAL_CASE: Dict[str, Any] = dict(complex_storage=True,
                                    boundaries={"x": "metallic"})

#: Bound once by leg ``expansion`` and read by every later leg. Deliberately NOT
#: defaulted: a leg that ran before the probe would bind a guess.
EXPANSION: Optional[str] = None
PROBE_ARTIFACT: Optional[str] = None


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose.

    A complex64 array's word view is its two planes interleaved, which is exactly what
    the kernel writes, so this is the same comparison the real gate makes with no
    complex-specific relaxation anywhere.
    """
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a complex128
    scalar, so the product is a full complex multiply whose imaginary part comes out
    ``-0.0`` regardless of what ``imag`` held. That defect made the first three
    expansion probes classify NumPy as matching NO arm. Every complex construction in
    this file goes through the word view, for that reason.
    """
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def build(keywords: Mapping[str, Any], seed: int,
          value_class: str = values.UNIFORM, scale: float = 1.0,
          ) -> Tuple[Any, Any]:
    """One seeded COMPLEX engine. Called twice (or three times) per case, identically.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the two halves' own gates use, so the fixture this certifies is the
    fixture they were certified on. Only the field state is reseeded here, and
    identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.

    SIGNED ZEROS ARE SEEDED INTO BOTH PLANES. MEEP keeps float32 subnormals on arm64,
    so the signed-zero class is constructible on the HOST here, and it is the class the
    complex helpers' literal zero cross terms exist for: folding ``z_im * 0.0f`` to a
    literal misses 12/128 words on the exhaustive table and 0/16,384 on random data.
    Random seeding alone would arm those mutations at nothing.
    """
    fields, pml = matrix.cart(**dict(keywords))
    if value_class == values.PM_ZERO_LATTICE:
        # THE PURE class, beside the physical band's SEEDED signed zeros above.
        # The uniform class already carries some -0.0 words on purpose; this one
        # is every word a zero, with the two planes OPPOSED, so the parity and the
        # cross terms are exercised with no magnitude present at all.
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
        if array is None:
            continue
        real = values.scaled_normal(rng, fields.grid.shape, 0.37, scale)
        imag = values.scaled_normal(rng, fields.grid.shape, 0.29, scale)
        real.reshape(-1)[::17] = np.float32(-0.0)
        real.reshape(-1)[7::23] = np.float32(0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        imag.reshape(-1)[11::29] = np.float32(0.0)
        array[...] = complex_from_planes(real, imag)
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

    ``None`` is a refusal rather than an empty tuple: a gate that stepped nothing would
    compare a no-op with a no-op and pass.
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
    ``sync_out`` / ``sync_in``. That bracket is the residency clause made operational:
    without it the next device launch would read the mirror's pre-``step_B`` bytes,
    which is smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns THREE passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``zero_metal_B`` running on the host on top of the clear the kernel
    already carried — which is IDEMPOTENT and would hide a dropped carry.
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
    plan = family.plan_metal_complex_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_complex_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False,
                "reason": "the complex fused magnetic pair was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    # THE SIGNED-ZERO FLOOR. A census of zero would mean the seeding never built the
    # class the literal zero cross terms exist for, and every zero-term mutation would
    # be armed at nothing.
    zeros = subnormal.signed_zero_census(
        np.concatenate([words(value).view(np.float32) for value in before.values()]))
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
    seeded_zeros = int(zeros.get("negative_zero", 0)) + int(zeros.get("positive_zero", 0))
    # THE FLOOR, AND WHICH ONE DEPENDS ON THE VALUE CLASS. The uniform class must
    # move every compared array AND carry the seeded signed zeros the literal-zero
    # cross-term mutations are armed at. The +-0 lattice is every word a zero, so
    # it answers to the SIGN floor on the REFERENCE OUTPUT instead: both bit
    # patterns must survive the step, which is what makes a uint32 compare of a
    # zero state discriminating rather than decorative.
    signs = values.zero_sign_census(state_of(reference).values())
    if value_class == values.PM_ZERO_LATTICE:
        floor_ok = bool(signs["positive_zero_words"] and signs["negative_zero_words"])
    else:
        floor_ok = bool(not still and seeded_zeros > 0)
    return {
        "passed": bool(identical and clean and launches_ok and floor_ok),
        "value_class": value_class,
        "floor_met": floor_ok,
        "reference_zero_sign_census": signs,
        "bit_identical": identical,
        "subnormal_free": clean,
        "seeded_signed_zeros": dict(zeros),
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
        "phased": list(plan.phased),
        "phase_values": [list(pair) for pair in plan.phase_values],
        "zero_metal": list(plan.zero_metal),
        "expansion": plan.expansion,
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         ) -> Dict[str, Any]:
    """The SEPARATE certified complex products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three engines
    from one seed: the array path, the two ALREADY CERTIFIED complex Metal products
    stepping the same seam as separate dispatches with the wall clear left on the HOST
    between them, and the fused product. All three must agree word for word at every
    complete step, and the DISPATCH COUNTS and the surviving in-seam host passes are
    recorded on both sides — which is the only place the difference between the
    compositions shows up at all, since a correct fusion is byte-neutral by
    construction.

    The separate side is not a strawman: those two plans are exactly what ``plan_step``
    composes today for this configuration.
    """
    reference, reference_pml = build(keywords, seed)
    separate, separate_pml = build(keywords, seed)
    fused, fused_pml = build(keywords, seed)

    separate_residency = Residency()
    curl = complex_fields.plan_complex_pml_curl(
        separate, separate_pml, "step_B", separate_residency)
    magnetic = complex_fields.plan_complex_constitutive(
        separate, separate_pml, "H", separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_complex_fused_magnetic_pair(
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
    # on every walled run it pays a host round trip the fused side does not.
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

def _specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...],
                                                    Tuple[bool, ...]]:
    """The (codes, phased, walls) triple the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds

    kinds = _boundary_kinds(fields.grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    flags, _values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(fields.grid, kinds), backward=False)
    return codes, tuple(flags), zero_metal_axes(fields.grid)


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


def shader_mutations(codes: Sequence[int], phased: Sequence[int],
                     walls: Sequence[bool], expansion: str) -> Dict[str, Dict[str, Any]]:
    """Source defects, each a plausible transcription slip.

    Every needle is anchored on text that only the mutated line carries, and every one
    is verified present before it is applied — an absent needle raises here rather than
    silently arming nothing.

    THE MUTATION CASE'S GEOMETRY, which is what makes each of these reachable: all
    three axes periodic AND phased at a generic k, so all three rotation blocks are
    emitted on a phase that is neither real nor a root of unity. The wall defects are
    armed separately, on the walled case.
    """
    base = family.complex_fused_magnetic_pair_source(codes, phased, walls, expansion)
    other = "NAIVE" if expansion == "FMA_V1" else "FMA_V1"
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float2 src0 = v0;", "float2 src0 = curl0;"),
        # THE SEAM, WRONG COMPONENT. Bx's constitutive source is v0, not v1.
        "seam_takes_the_wrong_component":
            needle(base, "float2 src1 = v1;", "float2 src1 = v0;"),
        # THE ARM. Swapping the whole helper block to the other licensable arm is the
        # defect the expansion probe exists to prevent. It is reachable ONLY through
        # the Bloch rotation: every other multiply here has a real coefficient and the
        # two arms coincide bit for bit, which is why this is armed on a PHASED case.
        "expansion_arm_swapped":
            needle(base, templates.complex_helpers(expansion),
                   templates.complex_helpers(other)),
        # THE ROTATION'S OPERAND ORDER. For a FULL complex product the orientation IS
        # load-bearing: the imaginary part fuses `z_re*p_im` one way and `p_re*z_im`
        # the other, and they round differently.
        "phase_multiply_operands_swapped":
            needle(base, "c_mul(b_x, px)", "c_mul(px, b_x)"),
        # THE PLANE-WISE COLLAPSE. `{re*c, im*c}` is the headline risk of every complex
        # transcription: it misses 24/128 words on the exhaustive signed-zero table and
        # NOTHING on random data.
        "phase_collapsed_to_plane_wise":
            needle(base, "c_mul(a_y, py)",
                   "float2(a_y.x * py.x, a_y.y * py.y)"),
        # THE WRAPPED LANE. An up-shift wraps at i == n-1 because it read slot n.
        # Rotating the WHOLE volume instead of the wrapped plane is smooth and wrong.
        "phase_applied_to_every_lane":
            needle(base, "bool wx = (i == nxi - 1);", "bool wx = true;"),
        # STEP_B IS FORWARD. step_D's negated strides are a different product.
        "curl_direction_reversed":
            needle(base, "int si = i + 1, sj = j + 1, sk = k + 1;",
                   "int si = i - 1, sj = j - 1, sk = k - 1;"),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            needle(base, "float2 t0 = ((c_y - c) + (b - b_z));",
                   "float2 t0 = (c_y - c + b - b_z);"),
        # THE RECURRENCE PAIRS are vec.hpp's cycle_direction: target 0 takes (y, z).
        "recurrence_axis_pair_swapped":
            needle(base,
                   "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);",
                   "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_z) - curl0, si_z);"),
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "flux_store_dropped":
            needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f1[ii] = v1; f2[ii] = v2;"),
        "fw_store_dropped":
            needle(base, "    w0[ii] = src0;", "    // stale f_w_Hx"),
        # The dsigw index is the component's OWN axis (stepping.py:227-228), not the
        # curl's cycle. Moving it one axis over is a silent half-cell error.
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = km0[i];",
                   "float kp_0 = kp0[j], km_0 = km0[j];"),
        "constitutive_accumulations_reversed":
            needle(base, "    a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"
                         "    a0 = a0 - c_mul_coefficient_left(km_0, prev0);",
                   "    a0 = a0 - c_mul_coefficient_left(km_0, src0);\n"
                   "    a0 = a0 + c_mul_coefficient_left(kp_0, prev0);"),
        # `prev` is read BEFORE `fw` is written (S:2083-2085). Reversed, this is wrong
        # only where kms != 0 -- INSIDE THE PML ONLY -- and looks like a slightly worse
        # absorber, not like a bug.
        "constitutive_prev_read_after_write":
            needle(base, "    float2 prev0 = w0[ii];\n"
                         "    // THE SEAM: the register step_B just wrote, not a "
                         "reload of Bx.\n"
                         "    float2 src0 = v0;\n"
                         "    w0[ii] = src0;",
                   "    // THE SEAM: the register step_B just wrote, not a reload of "
                   "Bx.\n"
                   "    float2 src0 = v0;\n"
                   "    w0[ii] = src0;\n"
                   "    float2 prev0 = w0[ii];"),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        mutants[name] = {shaders.CONTRACT_OFF:
                         compile_source(source).complex_fused_magnetic_pair_step}
    return mutants


#: The two SIGNED-ZERO SPELLINGS, and why they are not in :func:`shader_mutations`.
#:
#: MEASURED 2026-08-19, and this is a finding rather than a convenience. Both spellings
#: are genuinely wrong and both genuinely change the bits this kernel computes —
#: ``leg_zero_spellings`` measures 384/2048 and 144/2048 differing words at the point of
#: use on the exhaustive signed-zero table. But the difference they produce is a SIGN OF
#: ZERO, and the certified recurrence's very next operation is an addition:
#: ``c_mul_field_left(p0, km_y) - curl0``. On the real seeded field the folded inner
#: product differs at 282/5040 cells and, after that one subtraction with a generic
#: nonzero curl, 0/5040 survive; with an exact complex-zero curl 14/5040 survive.
#:
#: SO THESE TWO DEFECTS ARE STRUCTURALLY INVISIBLE TO A COMPLETE-STEP BYTE COMPARE, and
#: a mutation leg that armed them there would report them as uncaught for a reason that
#: has nothing to do with the kernel being right. They are carried HERE, scored on being
#: NULL with that measured reason recorded, and the defect itself is put under test by
#: ``leg_zero_spellings`` at the level where it is observable. Deleting them would hide
#: the gap; reporting them as caught would be false.
PREDICTED_NULL_REASON = (
    "the spelling changes only the SIGN OF A ZERO, and the recurrence's next addition "
    "with a nonzero operand destroys it before any word is stored (measured: 282/5040 "
    "cells differ at the inner product, 0/5040 after `- curl`). The defect is real and "
    "is measured by leg zero_spellings at the point of use; it cannot reach a stored "
    "word through a complete-step walk.")


def predicted_null_mutations(codes: Sequence[int], phased: Sequence[int],
                             walls: Sequence[bool],
                             expansion: str) -> Dict[str, Dict[str, Any]]:
    """The two signed-zero spellings, armed and expected to be NULL.

    See :data:`PREDICTED_NULL_REASON`. Scored on NOT diverging: if one of these ever
    DOES diverge, the model of why is wrong and the row should be read, not silenced.
    """
    base = family.complex_fused_magnetic_pair_source(codes, phased, walls, expansion)
    edits: Dict[str, str] = {
        # THE ZERO CROSS TERMS MUST BE LITERAL. Folding them misses 12/128 words on the
        # exhaustive signed-zero table in the certified complex family's own gate.
        "zero_cross_term_folded":
            needle(base, "return float2(fma(z.x, c,    -(z.y * 0.0f)),\n"
                         "                  fma(z.x, 0.0f,  (z.y * c)));",
                   "return float2(z.x * c, z.y * c);"),
        # NEGATION IS `-x` ON METAL. `0.0f - x` misses 36/512 on the exhaustive table.
        # The Triton track's `(a*b) * -1.0` workaround does NOT reproduce here.
        "negation_spelled_as_zero_minus":
            needle(base, "fma(z.x, p.x, -(z.y * p.y))",
                   "fma(z.x, p.x, 0.0f - (z.y * p.y))"),
    }
    return {name: {shaders.CONTRACT_OFF:
                   compile_source(source).complex_fused_magnetic_pair_step}
            for name, source in edits.items()}


def wall_mutations(codes: Sequence[int], phased: Sequence[int],
                   walls: Sequence[bool], expansion: str) -> Dict[str, Dict[str, Any]]:
    """The wall-clear and ownership-mask defects, armed on the WALLED case.

    Separated from :func:`shader_mutations` because these lines are emitted only where
    an axis is metallic. Armed on the phased case they would report as uncaught for the
    uninteresting reason that the shipped kernel emits nothing to mutate.
    """
    base = family.complex_fused_magnetic_pair_source(codes, phased, walls, expansion)
    zero = templates.COMPLEX_ZERO
    edits: Dict[str, str] = {
        # THE WALL CLEAR, DROPPED on the x wall.
        "zero_metal_dropped":
            needle(base, f"    v0 = at_x ? {zero} : v0;\n", ""),
        # THE WALL TABLE IS THE DIAGONAL FOR B, the off-diagonal for D. Reusing the
        # D-side table clears Bx on the y wall instead of the x wall, which is the
        # single most likely slip in porting this family from its D-side sibling.
        "zero_metal_uses_the_d_side_table":
            needle(base, f"    v0 = at_x ? {zero} : v0;",
                   f"    v0 = at_y ? {zero} : v0;"),
        # THE CLEAR IS A *COMPLEX* ZERO — both planes (S:1896, :1902). Clearing only
        # the real plane is the plane-wise slip in its wall-clear disguise.
        "zero_metal_clears_only_the_real_plane":
            needle(base, f"    v0 = at_x ? {zero} : v0;",
                   "    v0 = at_x ? float2(0.0f, v0.y) : v0;"),
        # THE AUXILIARY IS NOT CLEARED. `zero_metal_B` passes B_COMPONENTS only
        # (stepping.py:2250); masking fu_B as well is a plausible over-carry.
        "zero_metal_also_masks_the_auxiliary":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   f"    u0[ii] = at_x ? {zero} : n0; u1[ii] = n1; u2[ii] = n2;"),
        # THE OWNERSHIP MASK, dropped. A different pass from the wall clear, on the
        # same component and the same axis — which is exactly why both are armed.
        "ownership_mask_dropped":
            needle(base, f"    curl0 = at_x ? {zero} : curl0;\n", ""),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        mutants[name] = {shaders.CONTRACT_OFF:
                         compile_source(source).complex_fused_magnetic_pair_step}
    return mutants


def byte_neutral_source(codes: Sequence[int], phased: Sequence[int],
                        walls: Sequence[bool], expansion: str) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and ``v0``
    hold the same complex64 word pair, and a float2 stored to a ``device float2*`` and
    reloaded is bit-identical to the register. This edit is therefore the fusion's
    central claim written as a program, and leg ``byte_neutral_control`` requires it
    NOT to diverge.
    """
    source = family.complex_fused_magnetic_pair_source(codes, phased, walls, expansion)
    for target in range(3):
        source = needle(source, f"float2 src{target} = v{target};",
                        f"float2 src{target} = f{target}[ii];")
    return source


#: Where each binding group starts in the plan's argument tuple. Spelled once, so the
#: host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(15, 21))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(21, 27))
PARAMS_SLOT = 27


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the complex B curl the INTEGER split-field coefficients.

    ``step_B`` reads HALF-INTEGER positions and ``step_D`` integer ones
    (``complex_fields.SUB_STEPS['step_B']['suffix'] == '_h'``). This is the OPPOSITE
    polarity to the D/E fused pair's equivalent mutation, and the kernel cannot tell:
    it is a half-cell error in the absorber profile, not a crash.
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


def naive_params_tensor(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: the NAIVE 44-byte Params record — the alignment trap, armed.

    THE MOST IMPORTANT HOST MUTATION IN THIS FILE, because it is the one a future
    editor is most likely to write by accident. Metal aligns ``float2`` to 8 bytes; a
    NumPy record whose fields are laid end to end is 44 bytes where Metal's struct is
    48. Shipped, the phases are FIRST and the natural offsets coincide, so this mutant
    reproduces the trap by moving them LAST — which is the ordering the real twin's
    Params uses and therefore the ordering a reader would copy.

    The result is not a crash and not noise: every phase reads its neighbour's
    component, which is a well-formed complex number. This mutation is what turns
    "the layout was measured" from a docstring into a gate row.
    """
    import torch  # noqa: PLC0415

    naive = np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
                      ("dtdx", "<f4"), ("px", "<f4", 2), ("py", "<f4", 2),
                      ("pz", "<f4", 2)])
    record = np.zeros(1, dtype=naive)
    nx, ny, nz = plan.shape
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(plan.dtdx)
    for name, (real, imag) in zip(("px", "py", "pz"), plan.phase_values):
        record[name] = (np.float32(real), np.float32(imag))
    words_ = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    pad = (family.PARAMS_ITEMSIZE - words_.nbytes) // 4
    if pad > 0:
        words_ = np.concatenate([words_, np.zeros(pad, dtype=np.int32)])
    _rebind(plan, (PARAMS_SLOT,),
            (torch.from_numpy(words_).to(torch.device(residency.device)),))


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
    "params_record_uses_the_naive_44_byte_layout": naive_params_tensor,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """Both refuted signatures must FAIL; the shipped 28 must COMPILE and LAUNCH.

    This family sits against the ceiling from TWO directions and each is measured
    separately, because they license different design claims:

    * 53 bindings with the complex volumes as separate re/im planes — that is over the
      ceiling on the FIELD POINTERS ALONE, which is what licenses "float2 is FORCED";
    * 35 bindings with float2 volumes but the scalars and phases bound separately, the
      way the certified 23-binding complex curl binds them — which is what licenses
      "the packed struct is FORCED by this pair's fifteen volumes", not by style.

    A signature that compiled but bound the struct wrongly would produce a smooth,
    plausible, wrong field, so the launch half is measured here and not inherited from
    another family's docstring.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "split_plane_bindings": family.SPLIT_PLANE_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS, "ceiling": 31}

    for label, builder in (("separate", family.refuted_separate_scalar_source),
                           ("split_plane", family.split_plane_pair_signature)):
        try:
            compile_source(builder())
            row.update({f"{label}_compiled": True, f"{label}_error": "",
                        "passed": False,
                        "note": f"the {label} signature COMPILED; this family's shape "
                                f"rests on a ceiling this host does not have"})
            return row
        except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
            message = str(exc)
            row[f"{label}_compiled"] = False
            row[f"{label}_error"] = message.splitlines()[0] if message else ""
            row[f"{label}_refused_for_the_right_reason"] = (
                "out of bounds" in message and "buffer" in message)

    # The SHIPPED signature, launched. Fifteen float2 volumes, twelve float coefficient
    # vectors and the packed struct — the real thing, not a stand-in, so what reads back
    # is what the plan builder writes.
    shape, dtdx = (3, 4, 5), 0.125
    phases = ((-0.5, 0.8660254), (-1.0, 0.0), (0.309017, -0.95105654))
    source = family.complex_fused_magnetic_pair_source(
        (0, 0, 0), (1, 1, 1), (False, False, False), EXPANSION or "FMA_V1")
    try:
        compile_source(source).complex_fused_magnetic_pair_step
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True

    # A minimal readback kernel on the SHIPPED struct declaration, so what is measured
    # is the layout the shipped kernel parses rather than a hand-copied twin.
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    probe = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params {" + struct + "};", "",
        "kernel void params_probe(device float* out [[buffer(0)]],",
        "                         constant Params& prm [[buffer(1)]],",
        "                         uint idx [[thread_position_in_grid]])",
        "{", "    if (idx != 0) { return; }",
        "    out[0]=float(prm.nx); out[1]=float(prm.ny); out[2]=float(prm.nz);",
        "    out[3]=float(prm.n_elem); out[4]=prm.dtdx;",
        "    out[5]=prm.px.x; out[6]=prm.px.y; out[7]=prm.py.x;",
        "    out[8]=prm.py.y; out[9]=prm.pz.x; out[10]=prm.pz.y;", "}", ""))
    function = compile_source(probe).params_probe
    out = torch.zeros(11, dtype=torch.float32, device="mps")
    params = family._params_tensor(shape, dtdx, phases, "mps")
    function(out, params, threads=1)
    torch.mps.synchronize()
    read = [float(v) for v in out.cpu().numpy()]
    expected = [float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx))]
    for real, imag in phases:
        expected.extend((float(np.float32(real)), float(np.float32(imag))))
    fields_ok = all(np.float32(a) == np.float32(b) for a, b in zip(read, expected))
    row.update(packed_launched=True, packed_fields_read_back=read,
               packed_fields_expected=expected, packed_fields_correct=bool(fields_ok),
               params_itemsize=family.PARAMS_ITEMSIZE,
               passed=bool(row["separate_refused_for_the_right_reason"]
                           and row["split_plane_refused_for_the_right_reason"]
                           and fields_ok))
    return row


def leg_params_layout() -> Dict[str, Any]:
    """The struct layout, both orders, measured — and the naive record must FAIL.

    THIS LEG EXISTS BECAUSE THE FAILURE IS SILENT. A ``float2`` member forces 8-byte
    alignment; with the scalars first the struct needs internal padding and the natural
    NumPy record is 44 bytes against Metal's 48, so every phase lands one word early
    and reads as its neighbour's component. That is a converged, smooth, WRONG field.

    Three measurements, and the third is the one that turns the shipped field ORDER
    from taste into evidence:

    1. the shipped record (phases first, explicit offsets, itemsize 48) round-trips;
    2. the SCALARS-FIRST order with a naive record does NOT — reported with the exact
       words read back, so the shift is visible rather than asserted;
    3. the PHASES-FIRST order with the SAME naive record DOES round-trip, which is why
       the shipped struct puts them first: the ordering survives the mistake.
    """
    import torch  # noqa: PLC0415

    template = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "__STRUCT__", "",
        "kernel void readback(device float* out [[buffer(0)]],",
        "                     constant Params& prm [[buffer(1)]],",
        "                     uint idx [[thread_position_in_grid]])",
        "{", "    if (idx != 0) { return; }",
        "    out[0]=float(prm.nx); out[1]=float(prm.ny); out[2]=float(prm.nz);",
        "    out[3]=float(prm.n_elem); out[4]=prm.dtdx;",
        "    out[5]=prm.px.x; out[6]=prm.px.y; out[7]=prm.py.x;",
        "    out[8]=prm.py.y; out[9]=prm.pz.x; out[10]=prm.pz.y;", "}", ""))
    scalars_first = ("struct Params { uint nx; uint ny; uint nz; uint n_elem; "
                     "float dtdx; float2 px; float2 py; float2 pz; };")
    phases_first = ("struct Params { float2 px; float2 py; float2 pz; uint nx; "
                    "uint ny; uint nz; uint n_elem; float dtdx; };")
    naive_scalars_first = np.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
         ("dtdx", "<f4"), ("px", "<f4", 2), ("py", "<f4", 2), ("pz", "<f4", 2)])
    naive_phases_first = np.dtype(
        [("px", "<f4", 2), ("py", "<f4", 2), ("pz", "<f4", 2), ("nx", "<u4"),
         ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"), ("dtdx", "<f4")])

    shape, dtdx = (7, 11, 13), 0.3517
    phases = ((-0.5, 0.8660254), (-1.0, 0.0), (0.309017, -0.95105654))
    names = ["nx", "ny", "nz", "n_elem", "dtdx",
             "px.x", "px.y", "py.x", "py.y", "pz.x", "pz.y"]
    expected = [float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx))]
    for real, imag in phases:
        expected.extend((float(np.float32(real)), float(np.float32(imag))))

    def measure(struct: str, dtype: Any) -> Tuple[List[float], List[str]]:
        record = np.zeros(1, dtype=dtype)
        record["nx"], record["ny"], record["nz"] = shape
        record["n_elem"] = shape[0] * shape[1] * shape[2]
        record["dtdx"] = np.float32(dtdx)
        for name, (real, imag) in zip(("px", "py", "pz"), phases):
            record[name] = (np.float32(real), np.float32(imag))
        buffer = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
        pad = (family.PARAMS_ITEMSIZE - buffer.nbytes) // 4
        if pad > 0:
            buffer = np.concatenate([buffer, np.zeros(pad, dtype=np.int32)])
        params = torch.from_numpy(buffer).to(torch.device("mps"))
        out = torch.zeros(11, dtype=torch.float32, device="mps")
        compile_source(template.replace("__STRUCT__", struct)).readback(
            out, params, threads=1)
        torch.mps.synchronize()
        read = [float(v) for v in out.cpu().numpy()]
        wrong = [f"{n}: device {r!r} host {e!r}"
                 for n, r, e in zip(names, read, expected)
                 if np.float32(r) != np.float32(e)]
        return read, wrong

    shipped_read, shipped_wrong = measure(phases_first, family.params_record_dtype())
    scalars_read, scalars_wrong = measure(scalars_first, naive_scalars_first)
    phases_read, phases_wrong = measure(phases_first, naive_phases_first)

    return {
        "passed": bool(not shipped_wrong and scalars_wrong and not phases_wrong),
        "shipped_itemsize": family.params_record_dtype().itemsize,
        "naive_itemsize": naive_scalars_first.itemsize,
        "shipped_round_trips": not shipped_wrong,
        "shipped_read_back": shipped_read,
        "expected": expected,
        "scalars_first_naive_corrupted": bool(scalars_wrong),
        "scalars_first_naive_mismatches": scalars_wrong,
        "phases_first_naive_round_trips": not phases_wrong,
        "note": "the shipped struct puts the three float2 members FIRST because that "
                "order needs no internal padding, so the natural NumPy record's "
                "offsets coincide with Metal's; with the scalars first the same "
                "record shifts every phase one word early and reads as a plausible "
                "complex number rather than as garbage",
    }


def leg_zero_spellings() -> Dict[str, Any]:
    """The two signed-zero spellings, measured AT THE POINT OF USE — and why they die.

    THIS LEG EXISTS BECAUSE A COMPLETE-STEP BYTE COMPARE CANNOT SEE THESE DEFECTS, and
    that fact is itself measured here rather than asserted. Two halves:

    * **the defect is real.** On an exhaustive signed-zero table the shipped helpers and
      the two refuted spellings disagree, at the point of use, by a counted number of
      words. Folding ``z_im * 0.0f`` to a literal and spelling negation ``0.0f - x`` are
      therefore genuinely different arithmetic on this toolchain, not stylistic
      variants — which is what licenses the literal zeros and the ``-x`` in the shipped
      helpers;
    * **and it cannot reach a stored word.** The difference is a SIGN OF ZERO, and the
      certified recurrence's next operation is ``- curl0``. Run on the REAL seeded
      field, the folded inner product differs at a counted number of cells and, after
      that one subtraction with a generic nonzero curl, none survive. The control is the
      exact-complex-zero curl, where some do — which is what shows the annihilation is
      the addition's doing rather than the mutation failing to apply.

    Both numbers go in the artifact, because "the mutation is a predicted null" is a
    claim about the recurrence and has to be readable as one.
    """
    import torch  # noqa: PLC0415

    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        templates.complex_helpers(EXPANSION or "FMA_V1"), "",
        "static inline float2 folded_field_left(float2 z, float c) {",
        "    return float2(z.x * c, z.y * c);", "}",
        "static inline float2 negation_zero_minus(float2 z, float2 p) {",
        "    return float2(fma(z.x, p.x, 0.0f - (z.y * p.y)),",
        "                  fma(z.x, p.y,  (z.y * p.x)));", "}",
        "kernel void spellings(device float2* a [[buffer(0)]],",
        "                      device float2* b [[buffer(1)]],",
        "                      device float2* c2 [[buffer(2)]],",
        "                      device float2* d [[buffer(3)]],",
        "                      device float2* inner_ship [[buffer(4)]],",
        "                      device float2* inner_fold [[buffer(5)]],",
        "                      device float2* final_ship [[buffer(6)]],",
        "                      device float2* final_fold [[buffer(7)]],",
        "                      device const float2* z [[buffer(8)]],",
        "                      device const float* cc [[buffer(9)]],",
        "                      device const float2* p [[buffer(10)]],",
        "                      device const float2* curl [[buffer(11)]],",
        "                      constant float& km [[buffer(12)]],",
        "                      constant float& si [[buffer(13)]],",
        "                      constant uint& n [[buffer(14)]],",
        "                      uint idx [[thread_position_in_grid]])",
        "{", "    if (idx >= n) { return; }",
        "    a[idx]  = c_mul_field_left(z[idx], cc[idx]);",
        "    b[idx]  = folded_field_left(z[idx], cc[idx]);",
        "    c2[idx] = c_mul(z[idx], p[idx]);",
        "    d[idx]  = negation_zero_minus(z[idx], p[idx]);",
        "    // ... and the certified recurrence's next two operations, which is where",
        "    // a sign of zero either survives or dies (stepping._apply_pml_update:1905).",
        "    float2 s = c_mul_field_left(z[idx], km);",
        "    float2 f = folded_field_left(z[idx], km);",
        "    inner_ship[idx] = s;  inner_fold[idx] = f;",
        "    final_ship[idx] = c_mul_field_left(s - curl[idx], si);",
        "    final_fold[idx] = c_mul_field_left(f - curl[idx], si);", "}", ""))
    function = compile_source(source).spellings
    device = torch.device("mps")

    def launch(z: np.ndarray, coefficient: np.ndarray, phase: np.ndarray,
               curl: np.ndarray) -> List[np.ndarray]:
        count = z.shape[0]
        outs = [torch.zeros((count, 2), dtype=torch.float32, device=device)
                for _ in range(8)]
        function(*outs,
                 torch.from_numpy(z).to(device),
                 torch.from_numpy(coefficient).to(device),
                 torch.from_numpy(phase).to(device),
                 torch.from_numpy(curl).to(device),
                 0.83, 0.61, count, threads=count)
        torch.mps.synchronize()
        return [o.cpu().numpy().view(np.uint32) for o in outs]

    # HALF ONE: the exhaustive signed-zero table. Both zeros, both signs of a normal,
    # and the two seeding magnitudes this gate actually uses, crossed over every operand.
    import itertools  # noqa: PLC0415
    values = [np.float32(v) for v in (-0.0, 0.0, -1.5, 1.5, -0.37, 0.29)]
    table = list(itertools.product(values, repeat=5))
    count = len(table)
    z = np.empty((count, 2), np.float32)
    phase = np.empty((count, 2), np.float32)
    coefficient = np.empty(count, np.float32)
    for row, (zr, zi, cc, pr, pi) in enumerate(table):
        z[row] = (zr, zi)
        phase[row] = (pr, pi)
        coefficient[row] = cc
    a, b, c2, d, *_ = launch(z, coefficient, phase, np.zeros((count, 2), np.float32))
    fold_words = int(np.count_nonzero(a != b))
    negation_words = int(np.count_nonzero(c2 != d))

    # HALF TWO: the annihilation, on the REAL seeded field.
    fields, _pml = build(dict(CASES)[MUTATION_CASE], 61004)
    live = np.ascontiguousarray(
        np.asarray(fields.fu_Bx).reshape(-1)).view(np.float32).reshape(-1, 2).copy()
    cells = live.shape[0]
    rng = np.random.default_rng(7)
    generic = np.empty((cells, 2), np.float32)
    generic[:, 0] = rng.standard_normal(cells) * 0.21
    generic[:, 1] = rng.standard_normal(cells) * 0.17
    ones = np.ones(cells, np.float32)
    zeros_phase = np.zeros((cells, 2), np.float32)
    *_, inner_s, inner_f, final_s, final_f = launch(live, ones, zeros_phase, generic)
    inner_cells = int((inner_s != inner_f).any(axis=1).sum())
    survive_generic = int((final_s != final_f).any(axis=1).sum())
    *_, _is, _if, zs, zf = launch(live, ones, zeros_phase,
                                  np.zeros((cells, 2), np.float32))
    survive_zero_curl = int((zs != zf).any(axis=1).sum())

    return {
        "passed": bool(fold_words > 0 and negation_words > 0 and inner_cells > 0
                       and survive_generic == 0 and survive_zero_curl > 0),
        "table_rows": count,
        "table_words": int(a.size),
        "folded_zero_cross_terms_differing_words": fold_words,
        "negation_zero_minus_differing_words": negation_words,
        "field_cells": cells,
        "inner_product_differing_cells": inner_cells,
        "surviving_after_subtracting_a_generic_curl": survive_generic,
        "surviving_after_subtracting_an_exact_zero_curl": survive_zero_curl,
        "note": "the two refuted spellings ARE different arithmetic at the point of "
                "use, and the difference is a sign of zero that the recurrence's next "
                "addition destroys. That is why their complete-step mutations are "
                "carried as predicted nulls rather than as caught defects.",
    }


def leg_transcription(codes: Sequence[int], phased: Sequence[int],
                      walls: Sequence[bool], expansion: str) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED complex emitters' own bytes.

    The module claims its arithmetic is lifted rather than retyped. That is a property
    of the construction and therefore checkable, so it is checked:

    * the certified complex ``step_B`` curl body must appear in the fused source
      VERBATIM on both sides of the spliced wall clear;
    * the constitutive half must differ from the certified ``update_H`` body in EXACTLY
      the three ``float2 srcN =`` lines. Any other differing line means a rename or a
      re-spelling reached the arithmetic.
    """
    source = family.complex_fused_magnetic_pair_source(codes, phased, walls, expansion)
    curl = family.certified_curl_body(codes, phased, expansion)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    certified = family.certified_constitutive_body(expansion).splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float2 src{t} = g{t}[ii];", f"    float2 src{t} = v{t};")
                for t in range(3)]
    lengths_match = len(certified) == len(spliced)
    # The helpers must be the certified block, unedited: the arm is what they encode.
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
    }


def leg_expansion() -> Dict[str, Any]:
    """Which arm was bound, from which artifact, and the two refusals.

    THE ARM IS NOT A FREE CHOICE. A complex product has two licensable transcription
    arms and which one the REFERENCE takes is a measured platform fact. This family
    does not carry a probe of its own: its arithmetic is spliced from
    :mod:`.complex_fields`' emitters, so it binds through that module's probe and
    inherits its refusals. A second probe would be a second answer to one question.

    Three questions:

    1. an artifact must be present and must name exactly one arm;
    2. a MISSING probe must refuse — never a default arm;
    3. an AMBIGUOUS probe (every pattern ``AMBIGUOUS_BOTH``) must ALSO refuse, because
       a platform on which nothing discriminates has not been measured.

    AND THE CONSEQUENCE THIS LEG RECORDS RATHER THAN LEAVES IMPLICIT: on an UNPHASED
    row the two arms produce identical bytes, because every multiply outside the Bloch
    rotation takes a real coefficient whose imaginary operand is an exact ``+0.0``. So
    the arm binding is only under test on the phased cases, and the arm-swap mutation
    is armed on one.
    """
    global EXPANSION, PROBE_ARTIFACT
    configured = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
    record = complex_fields.load_expansion_probe()
    arm = complex_fields.expansion_from_probe(record)
    EXPANSION, PROBE_ARTIFACT = arm, configured

    missing_refused = complex_fields.expansion_from_probe(None) is None
    ambiguous = {"backend": complex_fields.PROBE_BACKEND,
                 "patterns": {name: complex_fields.AMBIGUOUS_BOTH
                              for name in complex_fields.PROBE_PATTERNS}}
    ambiguous_refused = complex_fields.expansion_from_probe(ambiguous) is None
    wrong_backend = dict(ambiguous, backend="cupy")
    backend_refused = complex_fields.expansion_from_probe(wrong_backend) is None

    digest = (hashlib.sha256(Path(configured).read_bytes()).hexdigest()
              if configured and Path(configured).is_file() else None)
    return {
        "passed": bool(arm in templates.EXPANSIONS and missing_refused
                       and ambiguous_refused and backend_refused),
        "arm": arm,
        "probe_artifact": configured,
        "probe_sha256": digest,
        "probe_backend": (record or {}).get("backend"),
        "probe_measured": (record or {}).get("measured"),
        "probe_numpy": (record or {}).get("numpy"),
        "host_numpy": np.__version__,
        "probe_patterns": (record or {}).get("patterns"),
        "missing_probe_refused": missing_refused,
        "ambiguous_probe_refused": ambiguous_refused,
        "wrong_backend_refused": backend_refused,
        "note": "the arm is inert on an unphased row: every multiply outside the "
                "Bloch rotation has a real coefficient, so both arms coincide bit "
                "for bit. The phased cases are what put it under test.",
    }


def _volume_source(fields: Any, component: str) -> Any:
    """A REAL engine source on ``component``, so ``field_type`` is the engine's.

    A stub with a hand-set ``field_type`` would let this leg pass while the engine
    classified the same component the other way; ``sources.VolumeSource`` resolves the
    slot itself through ``_field_type_for`` (sources.py:222-229).
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0))


class _SourceWithNoDepositIndex:
    """A magnetic source that never resolved the cells it writes.

    ``deposit_repair.save`` reads ``_point_ix``/``_point_iy``/``_point_iz``
    (deposit_repair.py:79-85); a source that publishes none cannot be saved and
    restored across the launch, and the carried clause must still refuse it BY NAME.
    Spelled as a stub rather than built from the engine BECAUSE the engine resolves
    the index eagerly — there is no real source in this position, which is exactly why
    the clause is a fail-closed one.
    """

    field_type = "B"


def leg_refusal() -> Dict[str, Any]:
    """The source seam, the fold and the storage split, each measured by name.

    THE MOST IMPORTANT LEG IN THIS FILE, and on 2026-08-28 the source-seam question
    turned over. It used to be "is a magnetic source REFUSED"; the family now declares
    :data:`family.CARRIES_DEPOSIT_REPAIR` and holds a row in
    ``metal_kernels.launch.FUSED_PAIR_ARMS``, so the question is "is it CARRIED, and is
    what the repair cannot reconstruct still refused". BOTH DIRECTIONS ARE MEASURED
    HERE, because a leg that only checked the new answer could not tell a carried seam
    from a deleted clause. Seven questions:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real MAGNETIC ``VolumeSource`` must now be ADMITTED, and its plan must BUILD —
       the launch is bracketed by ``deposit_repair``'s two plans at
       ``launch._install_fused_pair``, and the arithmetic that licenses it is
       ``test_deposit_repair.COMPLEX_CASES`` (byte-identical to the driver's order on
       complex storage, null control diverging);
    2b. with ``CARRIES_DEPOSIT_REPAIR`` HELD FALSE, the pre-flip prose must come back
       verbatim, naming the driver line. This is the record of what this clause said
       before the flip and the reason the flip is a change rather than a deletion;
    2c. a magnetic source that publishes NO deposit index must STILL be refused by
       name — the carried clause is not a blanket admission;
    3. a real ELECTRIC ``VolumeSource`` must NOT be refused — it is injected in the D/E
       half, outside this seam, and treating it as disqualifying would throw away every
       row this family exists to serve;
    4. a FOLDED grid must be refused, naming both B-side fills. UNCHANGED BY THE FLIP
       AND LOAD-BEARING BECAUSE OF IT: a folded seam fills mirror ghosts AFTER the
       injection and a point repair never visits a deposit's mirror image, so this
       clause is what keeps the carried seam inside what the repair can actually do;
    5. a REAL-STORAGE run must be refused — this product binds float2 volumes, and the
       domain split against the shipped real kernels runs both ways.
    """
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.metal_complex_fused_magnetic_pair_coverage(
        fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = family.metal_complex_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), residency)
    magnetic_plan = family.plan_metal_complex_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), residency=residency)
    deposit_index = deposit_repair._deposit_index(magnetic)
    deposit_points = (0 if deposit_index is None
                      else int(np.asarray(deposit_index[0]).size))

    # 2b. THE PRE-FLIP ANSWER, from the same code path with the flag held down.
    shipped_flag = family.CARRIES_DEPOSIT_REPAIR
    try:
        family.CARRIES_DEPOSIT_REPAIR = False
        pre_flip = family.metal_complex_fused_magnetic_pair_coverage(
            fields, pml, (magnetic,), residency)
    finally:
        family.CARRIES_DEPOSIT_REPAIR = shipped_flag
    pre_flip_named = [r for r in pre_flip.reasons
                      if "is magnetic" in r and "driver.py:3283-3284" in r]

    # 2c. What the repair cannot carry is still refused, with the flag shipped True.
    indexless_coverage = family.metal_complex_fused_magnetic_pair_coverage(
        fields, pml, (_SourceWithNoDepositIndex(),), residency)
    indexless_named = [r for r in indexless_coverage.reasons
                       if "does not publish the index it writes" in r]

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.metal_complex_fused_magnetic_pair_coverage(
        fields, pml, (electric,), residency)

    folded_fields, folded_pml = matrix.folded(boundaries={"y": "metallic"}, depth=1.2,
                                              complex_storage=True)
    folded_coverage = family.metal_complex_fused_magnetic_pair_coverage(
        folded_fields, folded_pml, (), Residency())
    folded_named = [r for r in folded_coverage.reasons
                    if "fill_symmetry_bc_B" in r and "fill_folded_far_ghosts_B" in r]

    real_fields, real_pml = matrix.cart()
    real_coverage = family.metal_complex_fused_magnetic_pair_coverage(
        real_fields, real_pml, (), Residency())
    real_named = [r for r in real_coverage.reasons if "storage is real float32" in r]

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and shipped_flag is True
                       and magnetic_coverage.covered
                       and magnetic_plan is not None
                       and deposit_points > 0
                       and not pre_flip.covered and pre_flip_named
                       and not indexless_coverage.covered and indexless_named
                       and electric_coverage.covered
                       and not folded_coverage.covered and folded_named
                       and not real_coverage.covered and real_named),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "carries_deposit_repair": shipped_flag,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_carried": magnetic_coverage.covered,
        "magnetic_source_reasons": list(magnetic_coverage.reasons),
        "magnetic_plan_built": magnetic_plan is not None,
        "magnetic_deposit_points": deposit_points,
        "pre_flip_refused": not pre_flip.covered,
        "pre_flip_named": pre_flip_named,
        "indexless_source_refused": not indexless_coverage.covered,
        "indexless_named": indexless_named,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "folded_refused": not folded_coverage.covered,
        "folded_named": folded_named,
        "real_storage_refused": not real_coverage.covered,
        "real_storage_named": real_named,
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
            f"artifact (${complex_fields.PROBE_PATH_ENVIRONMENT}); which arm the "
            f"{complex_fields.PROBE_BACKEND} reference takes is a platform fact and "
            "this gate refuses to guess it")

    probe_fields, probe_pml = build(cases[MUTATION_CASE], 1)
    codes, phased, walls = _specialisation(probe_fields, probe_pml)
    assert all(phased), (
        f"{MUTATION_CASE} does not phase every axis; the rotation mutations would be "
        f"armed on lines the shipped kernel does not emit")
    wall_fields, wall_pml = build(cases[WALL_MUTATION_CASE], 2)
    wall_codes, wall_phased, wall_walls = _specialisation(wall_fields, wall_pml)
    assert all(wall_walls), (
        f"{WALL_MUTATION_CASE} does not wall every axis; the wall mutations would be "
        f"armed on lines the shipped kernel does not emit")

    mutants = shader_mutations(codes, phased, walls, EXPANSION)
    wall_mutants = wall_mutations(wall_codes, wall_phased, wall_walls, EXPANSION)
    null_mutants = predicted_null_mutations(codes, phased, walls, EXPANSION)
    neutral = {shaders.CONTRACT_OFF:
               compile_source(byte_neutral_source(codes, phased, walls, EXPANSION)
                              ).complex_fused_magnetic_pair_step}

    controls = ("periodic_kx", "periodic_k0", "wall_x_kz")
    total = (1 + 1 + 1 + 1 + 1 + len(CASES) + len(CASES) + 1 + len(controls)
             + 1 + 1 + len(mutants) + len(wall_mutants) + len(null_mutants)
             + len(HOST_MUTATIONS) + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "expansion",
                     "label": "the_arm_is_bound_from_a_measured_artifact",
                     **expansion_row})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "float2_and_the_packed_struct_are_both_forced",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "params_layout",
                     "label": "every_params_field_reads_back_off_the_device",
                     **leg_params_layout()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "zero_spellings",
                     "label": "the_signed_zero_spellings_and_why_a_walk_cannot_see_them",
                     **leg_zero_spellings()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_complex_bytes",
                     **leg_transcription(codes, phased, walls, EXPANSION)})
        emit(handle, rows[-1])

        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 61000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

        # THE PURE +-0 CLASS beside the physical band's seeded signed zeros, and
        # then the band itself, refused with its census shown to fire.
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
                     **leg_subnormal_ladder(dict(CASES)[MUTATION_CASE], 65000)})
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
                     "label": "the_source_seam_the_fold_and_the_storage_split",
                     **leg_refusal()})
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

        for case_name, group in ((MUTATION_CASE, mutants),
                                 (WALL_MUTATION_CASE, wall_mutants)):
            for name, functions in group.items():
                index += 1
                result = run_case(cases[case_name], 72000 + index, args.steps,
                                  functions=functions)
                caught = not result.get("bit_identical", False)
                row = {"index": index, "total": total, "leg": "mutation",
                       "label": name, "case": case_name, "caught": caught,
                       "passed": bool(caught and result.get("launches")),
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "differing_arrays": result.get("differing_arrays"),
                       "launches": result.get("launches")}
                rows.append(row)
                emit(handle, row)

        # THE PREDICTED NULLS. Scored on NOT diverging, with the measured reason
        # carried on the row. A row here that DID diverge would mean the annihilation
        # model is wrong — which is a result to read, not a failure to silence.
        for name, functions in null_mutants.items():
            index += 1
            result = run_case(cases[MUTATION_CASE], 73000 + index, args.steps,
                              functions=functions)
            diverged = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "predicted_null",
                   "label": name, "case": MUTATION_CASE, "diverged": diverged,
                   "passed": bool(not diverged and result.get("launches")),
                   "differing_words": result.get("differing_words"),
                   "launches": result.get("launches"),
                   "reason": PREDICTED_NULL_REASON}
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            # The params-layout defect only bites where a phase is actually read, so
            # it is armed on the phased case like the rest.
            index += 1
            result = run_case(cases[MUTATION_CASE], 72000 + index, args.steps,
                              patch=patch)
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

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above is a
        # harness that diverges on its own.
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
                   "shader_mutations": len(mutants),
                   "wall_mutations": len(wall_mutants),
                   "predicted_null_mutations": len(null_mutants),
                   "host_mutations": len(HOST_MUTATIONS),
                   "byte_neutral_controls": 1},
        "predicted_null_reason": PREDICTED_NULL_REASON,
        "expansion_arm": EXPANSION,
        "expansion_probe": PROBE_ARTIFACT,
        "environment": ENVIRONMENT,
        "mutation_case": MUTATION_CASE,
        "wall_mutation_case": WALL_MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes), "phased": list(phased),
                                    "zero_metal": list(walls)},
        "wall_mutation_specialisation": {"codes": list(wall_codes),
                                         "phased": list(wall_phased),
                                         "zero_metal": list(wall_walls)},
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "denominator": 186,
            "complex_curl_at_step_B": 16,
            "complex_constitutive_at_update_H": 16,
            "admit_both_halves": 16,
            "and_declare_no_magnetic_source": 12,
            "note": "the magnetic-source clause costs 4 of the 16 (three rows declare "
                    "('D','B') and one declares ('B',)); 12 is an UPPER BOUND because "
                    "the census records source FIELD TYPES but not whether a row's "
                    "grid would also satisfy the residency and wall-readability "
                    "clauses this predicate adds. The fusion matrix's '16 rows' is "
                    "the SEAM's count; a fused product's count is the seam's minus "
                    "the source clause.",
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
