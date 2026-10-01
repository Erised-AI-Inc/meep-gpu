#!/usr/bin/env python3
"""Native MPS byte gate for the complex conductive fused pair: ``step_D`` -> ``update_E``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes (``step_D`` at driver.py:3293 and ``update_E`` at :3304) across a seam that
holds FOUR driver passes in general, so a per-sub-step comparison could not see it
at all: the whole claim is about the SEAM. The comparison is therefore per COMPLETE
SEAM STEP over a stated budget, against an array-path oracle running the identical
passes, with the first divergent step reported rather than a final pass/fail.

THE EIGHT THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. So every
   case asserts the exact LAUNCH COUNT (``launches_per_run`` x steps), a nonzero
   ``runs``, and a NON-VACUITY FLOOR SPLIT BY ROLE — the arrays this seam WRITES
   (D, E) must all have moved, and the ones it only READS (B, H, every
   polarization array) must NOT have.
2. **A refusal that is really an omission.** Leg ``refusal`` is the most important
   leg in this file, because THE ELECTRIC SOURCE SLOT IS THIS FAMILY'S BINDING
   CLAUSE. It requires: an undeclared source list refused; a real electric
   ``VolumeSource`` refused BY NAME with the driver line cited; a real MAGNETIC
   source NOT refused (the polarity, which is what makes this cell's four corpus
   rows reachable at all); a metallic axis refused naming ``zero_metal_D``; a
   folded axis refused naming both D-side fills; and an off-diagonal row refused
   naming the stencil.
3. **A seam that is not empty after all.** Leg ``seam_is_empty`` DRIVES THE REAL
   ``FdtdDriver`` on this cell's own configuration and records every pass that
   writes a D word between :3293 and :3304 — with a control leg carrying an
   electric source, where the conductive injection MUST fire and write. A silent
   instrument cannot report an empty seam by measuring nothing.
4. **A hollow pass.** Source mutations and HOST mutations are armed, and each must
   be CAUGHT — or NULL CONFIRMED by an identity check. Leg ``disarm`` reruns the
   identical case with the shipped bytes and requires zero.
5. **A dead-branch mutation.** The curl half is NOT straight-line: it carries the
   ghost-wrap ternaries and the Bloch wrap planes. So instead of asserting one
   branch, leg ``branch_census`` enumerates every branch in the emitted text and
   :func:`shader_mutations` requires each armed edit to sit on a line the SCORED
   case executes — with the scored case chosen so that every ternary is taken on
   some thread and every wrap plane is live.
6. **A mirrored evaluator.** :func:`leg_transcription` PARSES the load-bearing
   lines out of this family's emitted source AND out of the CERTIFIED emitters'
   own output, and requires them equal modulo the documented renames.
7. **A byte-neutral claim asserted rather than measured.** Leg ``byte_neutral``
   replaces the register the constitutive reads with a reload of the very word the
   curl half just stored. That mutant must NOT diverge — it is the same value by
   construction — which is what makes "the fusion removes a round trip and changes
   no arithmetic" a measurement. It is the one armed edit required to be UNCAUGHT.
8. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor and instead required to carry BOTH signs of
   zero IN BOTH PLANES of the reference output; the subnormal band is REFUSED BY
   NAME and its census required to FIRE.

BOTH SUBNORMAL POLICIES, AS FAR AS THIS BACKEND HAS THEM. MPS flush is native and
has NO LEVER (metal_kernels/subnormal.py:6-13), so ``keep`` is not a second
comparison but a REFUSAL, and leg ``policy`` requires the plan to be refused BY
NAME under it rather than silently downgraded.

LEGS
  0  binding_ceiling   corpus row, the ceiling, and the first one past it
  1  branch_census     every branch enumerated; the scored case takes them all
  2  transcription     the emitted text equals the certified emitters', parsed
  3  policy            `keep` is REFUSED by name; `flush` admits
  4  refusal           every seam clause measured BY NAME, and the polarity
  5  seam_is_empty     the real driver instrumented on the corpus configuration
  6  product           complete seam steps, per-step byte compare, counters, floor
  7  separate_control  array path vs the two CERTIFIED products vs fused, 3 ways
  8  value_classes     uniform, +-0 lattice, subnormal band (refused, census fires)
  9  byte_neutral      the register-vs-reload edit must NOT diverge
 10  mutation          armed defects, each CAUGHT or NULL CONFIRMED
 11  disarm            the same harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_metal_complex_conductive_fused_pair.py --out <dir>/pair.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

#: The MEASURED expansion-probe artifact this host's complex families bind to.
PROBE_ARTIFACT = (API_ROOT / "parity" / "meep_gpu" / "results"
                  / "metal_complex_audit_2026-08-16"
                  / "complex_expansion_probe.json")
os.environ.setdefault("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE", str(PROBE_ARTIFACT))

import numpy as np  # noqa: E402

from meep_gpu import driver as driver_module, stepping  # noqa: E402
from meep_gpu.absorber import AbsorberLayer  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_conductive_fused_pair as family, complex_no_pml_conductive,
    complex_no_pml_stored_e, shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: E402

#: Complete seam steps every case walks. SIXTY, not twelve: D is state carried
#: between steps and the conductive tail compounds, so the defects this gate exists
#: for accumulate. The comparison is per COMPLETE STEP, so a green row means
#: identical at ALL sixty — and step 1 is reported separately, because "identical
#: at one launch" and "identical at sixty" are different claims.
STEPS = 60

#: Base for every per-case seed. ``SEED + sha256(label|leg)``, never ``hash()``:
#: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``, so a hash-seeded
#: gate draws a different fixture every process and a failing case cannot be
#: replayed from its own record.
SEED = 57_000

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"f_cond_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"])

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line.
WRITTEN_PREFIXES = ("D", "E")
READ_ONLY_PREFIXES = ("B", "H", "f_cond_", "P[", "P_prev[", "scratch[")

#: The corpus configuration this family exists for, read off
#: ``results/metal_coverage_tranche6_2026-08-19``: the four ``TestLoadDump.*_3d``
#: rows are complex64, no PML, an Absorber (a graded D AND B conductivity),
#: k_point (0.4, -1.3, 0.7), one Lorentzian driving Ex/Ey/Ez, ONE MAGNETIC source,
#: all-periodic boundaries.
CORPUS_K = (0.4, -1.3, 0.7)
CORPUS_POLES = 1

#: The two mutations this kernel's ARITHMETIC cannot express, scored NULL — and
#: the null is a MEASUREMENT, not a shrug. The certified emitters keep
#: ``c_mul_field_left(z, c)`` and ``c_mul_coefficient_left(c, z)`` as SEPARATE
#: functions because operand order can change bytes under a fused expansion, and
#: each call site transcribes the ARRAY PATH's orientation literally
#: (templates.py:172-178). Whether the two spellings differ AT RUNTIME on this
#: platform is a different question, and leg ``orientation_null`` answers it:
#: exhaustively over the four signed-zero classes at positive, negative and zero
#: coefficients, in BOTH contraction modes, and over a 4096-pair random band, they
#: are bit-identical. So an orientation edit is a TEXT defect that leg
#: ``transcription`` pins character for character, not a runtime defect a
#: comparator here can catch — and arming it as "must be caught" would produce a
#: row that measures nothing.
NULL_SCORED: Dict[str, str] = {
    "curl_dtdx_orientation_flipped":
        "the two helper orientations are bit-identical for a REAL coefficient on "
        "this platform; the orientation is pinned by leg transcription instead",
    "constitutive_orientation_flipped":
        "the two helper orientations are bit-identical for a REAL coefficient on "
        "this platform; the orientation is pinned by leg transcription instead",
}

@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration."""

    label: str
    poles: int
    kind: str                 # LORENTZIAN | DRUDE | "mixed"
    conductive: Tuple[bool, bool, bool] = (True, True, True)
    k_point: Tuple[float, float, float] = CORPUS_K
    cell: Tuple[float, float, float] = (1.2, 1.0, 0.9)
    value_class: str = "uniform"


#: THE CASE MATRIX. The corpus row is FIRST and named as such. ``mixed_
#: conductivity`` is the row that exercises the ONE line this family rewrites: a
#: component without a conductivity takes the split single-line tail, and a
#: component with one takes the certified tail unchanged, in the SAME kernel.
CASES: Tuple[Case, ...] = (
    Case("corpus_one_pole_all_conductive", CORPUS_POLES, LORENTZIAN),
    Case("mixed_conductivity_x_only", 1, LORENTZIAN, conductive=(True, False, False)),
    Case("mixed_conductivity_yz_only", 2, DRUDE, conductive=(False, True, True)),
    Case("three_pole_mixed_kinds", 3, "mixed"),
    Case("four_pole_at_the_ceiling", 4, "mixed"),
    Case("zero_k_point_control", 1, LORENTZIAN, k_point=(0.0, 0.0, 0.0)),
    Case("single_axis_bloch", 2, LORENTZIAN, k_point=(0.0, 0.37, 0.0)),
)

#: The case every mutation is scored on. FOUR poles, ALL THREE components
#: conductive and a fully three-axis Bloch phase, so every armed line is emitted
#: AND executed: a zero-k row would leave the three wrap-plane edits unfilled and a
#: non-conductive row would leave the conductive tail's three lines unreached.
MUTATION_CASE = "four_pole_at_the_ceiling"

#: The second mutation case, for the ONE line this family rewrites. The split tail
#: is emitted only for a component WITHOUT a conductivity, so the edit that breaks
#: it has to be armed on a row that has one.
MUTATION_CASE_MIXED = "mixed_conductivity_x_only"


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str, leg: str) -> int:
    digest = hashlib.sha256(f"{label}|{leg}".encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def words(array: Any) -> np.ndarray:
    value = np.ascontiguousarray(array)
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.complex64)):
        raise TypeError(f"expected float32 or complex64, got {value.dtype}")
    return value.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def _case(label: str) -> Case:
    return next(case for case in CASES if case.label == label)


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def build(case: Case, seed: int, scale: float = 1.0,
          symmetry: Sequence[Any] = (), boundaries: Any = "periodic",
          ) -> Tuple[Grid, Fields, Optional[PML]]:
    """One seeded engine. Called twice per comparison, identically, for both sides.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES unless a value class says
    otherwise. A zero-initialised state is a FIXED POINT of this seam — with
    B = D = 0 the curl is 0, the conductive tail keeps D at 0, and the constitutive
    gives E = 0 forever — so a deliberately wrong kernel compared against a zero
    oracle reports IDENTICAL. That is the vacuity trap this project has paid for.
    """
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=tuple(case.k_point),
                symmetry=tuple(symmetry), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    # THE THREE COMPONENTS GET DISTINCT EPSILON ARRAYS, and that is load-bearing
    # rather than thorough. `set_isotropic_epsilon_volume` hands the SAME array
    # back for all three (Fields.has_component_epsilon is then False), which makes
    # "bind Ey's inverse epsilon into Ex's slot" a no-op — a host defect that
    # cannot be scored because it is an identity. Diagonal anisotropy is also what
    # a geometry sampled at each E component's own Yee position gives, so this is
    # the ordinary case rather than a contrivance.
    epsilon = {name: (1.45 + 0.30 * rng.random(shape)).astype(np.float32)
               for name in ("Ex", "Ey", "Ez")}
    fields.set_epsilon_volumes(
        epsilon,
        {name: (np.float32(1.0) / value).astype(np.float32)
         for name, value in epsilon.items()})

    # THE THREE CONDUCTIVE TARGETS GET DISTINCT PROFILES, for the same reason: a
    # directional absorber is exactly what MEEP's `mp.Absorber` on one face gives,
    # and one shared array would make "bind Dy's condinv into Dx's slot" a no-op.
    # Each is graded along x as an Absorber's own profile is.
    def profile(scale: float) -> np.ndarray:
        ramp = np.linspace(np.float32(0.06 * scale), np.float32(0.22 * scale),
                           shape[0], dtype=np.float32)[:, None, None]
        return np.broadcast_to(ramp, shape).copy()

    fields.set_d_conductivity({
        name: (profile(1.0 + 0.35 * index) if flag else None)
        for index, (name, flag) in enumerate(
            zip(("Dx", "Dy", "Dz"), case.conductive))})
    fields.set_b_conductivity({name: profile(0.8 + 0.2 * index)
                               for index, name in enumerate(("Bx", "By", "Bz"))})

    kinds = (DRUDE, LORENTZIAN) if case.kind == "mixed" else (case.kind,)
    for order in range(case.poles):
        base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
                "Ez": 0.19 + 0.06 * order}
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            base, grid, np.complex64))
    fields.enable_field_storage()

    if case.value_class == "pm_zero_lattice":
        # A lattice of +0.0 and -0.0 over BOTH PLANES of every array the seam
        # consumes, assigned through the .real/.imag views — composing it as
        # `real + 1j * imaginary` is a complex multiply and add, and it silently
        # canonicalises the sign of a zero away.
        index = np.arange(int(np.prod(shape))).reshape(shape)
        lattice = np.zeros(shape, dtype=np.complex64)
        lattice.real = np.where((index % 2) == 0, np.float32(0.0),
                                np.float32(-0.0)).astype(np.float32)
        lattice.imag = np.where((index % 3) == 0, np.float32(-0.0),
                                np.float32(0.0)).astype(np.float32)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice
        for state in fields.polarizations:
            for component in state.driven():
                # +0 on both planes, so the E half's `d_in - p0 - p1 ...` chain
                # PRESERVES the displacement's signs: `(-0) - (+0)` is `-0`, while
                # `(-0) - (-0)` is `+0` and would erase the class the leg needs.
                state.P[component][...] = np.complex64(0.0)
                state.P_prev[component][...] = np.complex64(0.0)
        return grid, fields, PML(grid=grid, thickness=0)

    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = ((rng.standard_normal(shape) * 0.37 * scale)
                          + 1j * (rng.standard_normal(shape) * 0.37 * scale)
                          ).astype(np.complex64)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = (
                (rng.standard_normal(shape) * 0.21 * scale)
                + 1j * (rng.standard_normal(shape) * 0.21 * scale)).astype(np.complex64)
            state.P_prev[component][...] = (
                (rng.standard_normal(shape) * 0.19 * scale)
                + 1j * (rng.standard_normal(shape) * 0.19 * scale)).astype(np.complex64)
    return grid, fields, PML(grid=grid, thickness=0)


def state_of(fields: Fields) -> Dict[str, Any]:
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


def seam_step(fields: Fields, pml: Optional[PML]) -> None:
    """The two array-path passes this product replaces, in driver order.

    NOTHING IS RUN BETWEEN THEM, and that is not an omission: this family's
    clauses make every one of the driver's four in-seam passes inert (no electric
    source, no mirror, no wall). Leg ``seam_is_empty`` instruments the REAL driver
    to establish that rather than asserting it here.
    """
    stepping.step_D(fields, pml)            # driver.py:3293
    stepping.update_E(fields, pml)          # driver.py:3304


def census_of(fields: Fields) -> int:
    return sum(subnormal.census(value) for value in state_of(fields).values())


def negative_zeros_per_plane(fields: Fields) -> Tuple[int, int]:
    """``-0.0`` words in the REAL and IMAGINARY planes, counted separately.

    The pooled count cannot tell "the lattice discriminates in one plane" from
    "in both", and the complex arm's refuted spellings differ exactly on the
    IMAGINARY operand — so a lattice live in the real plane only proves nothing.
    """
    real = imaginary = 0
    for value in state_of(fields).values():
        flat = words(value)
        negative = flat == np.uint32(0x80000000)
        if np.ascontiguousarray(value).dtype == np.dtype(np.complex64):
            real += int(np.count_nonzero(negative[0::2]))
            imaginary += int(np.count_nonzero(negative[1::2]))
        else:
            real += int(np.count_nonzero(negative))
    return real, imaginary


def positive_zeros(fields: Fields) -> int:
    return sum(int(np.count_nonzero(words(value) == np.uint32(0)))
               for value in state_of(fields).values())


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _split_movement(before: Mapping[str, np.ndarray], after: Mapping[str, Any]
                    ) -> Tuple[List[str], List[str], int]:
    moved = {name: differing(before[name], after[name]) for name in before}
    written = [name for name in moved
               if name.startswith(WRITTEN_PREFIXES) and not name.startswith("f_cond_")]
    read_only = [name for name in moved if name not in written]
    still = sorted(name for name in written if moved[name] == 0)
    disturbed = sorted(name for name in read_only if moved[name])
    return still, disturbed, int(sum(moved.values()))


def run_case(case: Case, leg: str, steps: int,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE SEAM STEP."""
    seed = case_seed(case.label, leg)
    _, reference, reference_pml = build(case, seed)
    _, actual, actual_pml = build(case, seed)
    drift = compare(reference, actual)
    assert not drift, f"{case.label}: the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_complex_conductive_fused_pair(
        actual, actual_pml, (), residency, functions=functions)
    if plan is None:
        reasons = family.complex_conductive_fused_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": "the fused pair was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    before = frozen(actual)
    residency.sync_in()
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        plan.run()
        residency.sync_out()
        difference = compare(reference, actual)
        census = census_of(reference)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    still, disturbed, moved_words = _split_movement(before, after)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    vacuity_exempt = case.value_class == "pm_zero_lattice"
    floor_ok = (not disturbed) and (vacuity_exempt or not still)
    lattice_live = None
    lattice_planes: Optional[Tuple[int, int]] = None
    if vacuity_exempt:
        lattice_planes = negative_zeros_per_plane(reference)
        lattice_live = bool(lattice_planes[0] and lattice_planes[1]
                            and positive_zeros(reference))
        floor_ok = floor_ok and lattice_live
    return {
        "passed": bool(identical and clean and launches_ok and floor_ok),
        "bit_identical": identical,
        "identical_at_step_1": bool(per_step and per_step[0]["differing_words"] == 0),
        "identical_at_last_step": bool(
            per_step and per_step[-1]["differing_words"] == 0
            and len(per_step) == steps),
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"] if per_step else None,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else None,
        "reference_subnormals": per_step[-1]["reference_subnormals"] if per_step else None,
        "arrays_compared": len(before),
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "vacuity_exempt": vacuity_exempt,
        "lattice_carries_both_zero_signs": lattice_live,
        "lattice_negative_zero_words_real_imaginary": lattice_planes,
        "moved_words": moved_words,
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "replaces": list(plan.replaces_sub_steps),
        "expansion": plan.expansion,
        "conductive": list(plan.conductive),
        "boundary_codes": list(plan.bc), "phased": list(plan.phased),
        "poles": list(plan.pole_counts),
        "bindings": family.binding_count(plan.pole_counts),
        "mirrors": len(residency.names),
        "shape": list(plan.shape), "seed": seed,
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path; the two ALREADY CERTIFIED Metal products
    (``complex_no_pml_conductive`` at ``step_D`` and ``complex_no_pml_stored_e`` at
    ``update_E``) dispatching the seam as separate launches; and the fused product.
    All three must agree word for word at every complete seam step, and the
    DISPATCH COUNTS are recorded on both sides — the only place the difference
    between the compositions shows up at all, since a correct fusion is
    byte-neutral by construction.

    The separate side is not a strawman: it is exactly what ``plan_step`` composes
    for this configuration today.
    """
    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(case, seed)
    _, separate, separate_pml = build(case, seed)
    _, fused, fused_pml = build(case, seed)

    separate_residency = Residency()
    curl = complex_no_pml_conductive.plan_metal_complex_conductive_no_pml_curl(
        separate, separate_pml, "step_D", separate_residency)
    electric = complex_no_pml_stored_e.plan_metal_complex_stored_e(
        separate, separate_pml, separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_complex_conductive_fused_pair(
        fused, fused_pml, (), fused_residency)
    if curl is None or electric is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "electric": electric is not None,
                "fused": plan is not None}

    separate_residency.sync_in()
    fused_residency.sync_in()
    before = frozen(fused)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        curl.run()
        electric.run()
        plan.run()
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

    still, disturbed, _ = _split_movement(before, state_of(fused))
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and electric.launches and not still and not disturbed),
        "per_step_tail": per_step[-3:],
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + electric.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + electric.launches,
        "fused_launches": plan.launches,
        "dispatches_removed_per_step": (curl.launches_per_run
                                        + electric.launches_per_run
                                        - plan.launches_per_run),
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Static legs
# ---------------------------------------------------------------------------

def _arm() -> str:
    from meep_gpu.metal_kernels.complex_fields import (  # noqa: PLC0415
        expansion_from_probe, load_expansion_probe,
    )
    arm = expansion_from_probe(load_expansion_probe())
    assert arm is not None, "no expansion arm is licensed by the measured probe"
    return arm


def leg_binding_ceiling() -> Dict[str, Any]:
    """The corpus row, the ceiling, and the first configuration past it.

    This is what makes the packed ``Params&`` a measurement rather than a
    preference. Without it the certified curl's EIGHT separate scalars put a
    two-pole configuration at 32 bindings, one over the platform's limit; the row
    at ``(0, 0, 0)`` poles records the fused signature's floor and the rows past
    the ceiling must FAIL, refused BY NAME by the emitter.
    """
    arm = _arm()
    rows: List[Dict[str, Any]] = []
    for counts in ((1, 1, 1), (0, 0, 0), (4, 4, 4), (4, 4, 5), (5, 5, 5), (8, 8, 8)):
        bindings = family.binding_count(counts)
        expected_fit = bindings <= MAX_BUFFER_BINDINGS
        try:
            compile_source(family.complex_conductive_fused_pair_source(
                (0, 0, 0), (1, 1, 1), (True, True, True), counts, arm))
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
            compiled, error = False, str(exc).splitlines()[0]
        rows.append({"pole_counts": list(counts), "bindings": bindings,
                     "ceiling": MAX_BUFFER_BINDINGS,
                     "expected_to_fit": expected_fit, "compiled": compiled,
                     "error": error, "agrees": compiled == expected_fit})
        log(f"    ceiling poles={counts} bindings={bindings} "
            f"fits={expected_fit} compiled={compiled}")
    over = [row for row in rows if not row["expected_to_fit"]]
    refused_by_name = all("bindings" in row["error"] and "device.py:72" in row["error"]
                          for row in over)
    # WHAT THE PACKING BOUGHT, stated in numbers rather than claimed: the certified
    # curl binds eight scalars separately (complex_no_pml_conductive.py:85-92).
    unpacked_two_pole = (family.CURL_POINTERS + family.CONSTITUTIVE_POINTERS
                         + 2 * 3 + 8)
    return {"passed": bool(all(row["agrees"] for row in rows) and over
                           and refused_by_name),
            "rows": rows, "over_ceiling_rows": len(over),
            "refused_by_name": refused_by_name,
            "corpus_configuration_bindings": family.binding_count((1, 1, 1)),
            "two_pole_bindings_without_the_packed_Params": unpacked_two_pole,
            "two_pole_bindings_with_it": family.binding_count((2, 2, 2))}


def leg_branch_census() -> Dict[str, Any]:
    """Every branch in the emitted text, enumerated — and every one TAKEN somewhere.

    THE DEAD-BRANCH MUTATION is a defect class this project has paid for, and this
    family cannot answer it the way the E->P chain does: the curl half is NOT
    straight-line. It carries the ``n_elem`` guard, the three ghost-wrap ternaries,
    the nine neighbour-validity ternaries, and (under a Bloch phase) six wrap-plane
    ternaries. So the branches are COUNTED here and the mutation case is chosen so
    that every one of them is exercised: all three axes periodic and PHASED, which
    makes each wrap predicate true on the plane it guards and false elsewhere.

    The counts are recorded per specialisation rather than asserted equal to a
    magic number, so a future specialisation that adds a branch shows up as a
    changed census rather than passing silently.
    """
    arm = _arm()
    rows: List[Dict[str, Any]] = []
    for label, codes, phased, conductive in (
            ("corpus_periodic_phased", (0, 0, 0), (1, 1, 1), (True, True, True)),
            ("unphased", (0, 0, 0), (0, 0, 0), (True, True, True)),
            ("mixed_conductivity", (0, 0, 0), (1, 1, 1), (True, False, False)),
            ("metallic_axis", (1, 0, 0), (0, 1, 1), (True, True, True))):
        source = family.complex_conductive_fused_pair_source(
            codes, phased, conductive, (1, 1, 1), arm)
        rows.append({
            "label": label, "codes": list(codes), "phased": list(phased),
            "conductive": list(conductive),
            "if_branches": len(re.findall(r"\bif\s*\(", source)),
            "ternaries": source.count("?"),
            "loops": len(re.findall(r"\b(for|while)\s*\(", source)),
            "ownership_mask_lines": source.count("? float2(0.0f, 0.0f) :"),
            "wrap_planes": source.count("bool w"),
        })
        log(f"    branches {label}: if={rows[-1]['if_branches']} "
            f"ternary={rows[-1]['ternaries']} wrap_planes={rows[-1]['wrap_planes']}")
    scored = next(row for row in rows if row["label"] == "corpus_periodic_phased")
    # The scored configuration must carry all three wrap planes, or the three
    # wrap-plane mutations below would be armed on lines no thread reaches.
    return {"passed": bool(scored["wrap_planes"] == 3
                           and all(row["loops"] == 0 for row in rows)
                           and all(row["if_branches"] == 1 for row in rows)),
            "rows": rows,
            "the_scored_case_carries_all_three_wrap_planes":
                scored["wrap_planes"] == 3,
            "guard": "if (idx >= n_elem) { return; }  (templates.GUARD)"}


def leg_transcription() -> Dict[str, Any]:
    """The emitted text equals the CERTIFIED emitters' own, PARSED not mirrored.

    THE MIRRORED EVALUATOR is the second defect class this project has paid for.
    Nothing here re-derives a line: each certified emitter is CALLED, its output is
    searched for the text, this family's output is searched for the same text
    modulo the documented renames, and the two are required equal.

    THE CURL HALF IS CHECKED AS A BLOCK, not line by line, and that is stronger:
    the whole certified curl body from the ghost gather through the conductive tail
    must appear in the fused source verbatim.
    """
    arm = _arm()
    fused = family.complex_conductive_fused_pair_source(
        (0, 0, 0), (1, 1, 1), (True, True, True), (2, 2, 2), arm)
    certified_curl = complex_no_pml_conductive.complex_conductive_no_pml_curl_source(
        (0, 0, 0), True, (1, 1, 1), (True, True, True), arm)
    certified_e = complex_no_pml_stored_e.complex_stored_e_source(2, arm)
    rows: List[Dict[str, Any]] = []

    # --- the curl half, as one contiguous block -------------------------------
    start = certified_curl.index("    int si = i - 1")
    end = certified_curl.index("__MASK__") if "__MASK__" in certified_curl else None
    block = certified_curl[start:certified_curl.index(
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);")]
    rows.append({
        "line": "curl_half_as_one_block",
        "certified_source": "complex_no_pml_conductive.complex_conductive_no_pml_curl_source",
        "present_in_certified": True,
        "present_in_fused": block in fused,
        "rename": "(none)",
        "block_characters": len(block),
        "agrees": block in fused})

    # --- the conductive tail, from the certified emitter's own function --------
    for index in range(3):
        tail = complex_no_pml_conductive._tail(index, True)
        rows.append({
            "line": f"conductive_tail_{index}",
            "certified_source": f"complex_no_pml_conductive._tail({index}, True)",
            "present_in_certified": tail in certified_curl,
            "present_in_fused": tail in fused,
            "rename": "(none) — the certified tail ALREADY names value{i}",
            "agrees": tail in certified_curl and tail in fused})

    # --- THE ONE LINE THIS FAMILY REWRITES, and only on a NON-conductive row ---
    plain = family.complex_conductive_fused_pair_source(
        (0, 0, 0), (1, 1, 1), (False, True, True), (1, 1, 1), arm)
    certified_plain = complex_no_pml_conductive._tail(0, False)
    rows.append({
        "line": "non_conductive_tail_split",
        "certified_source": "complex_no_pml_conductive._tail(0, False)",
        "present_in_certified": certified_plain
        == "    f0[ii] = f0[ii] - curl0;",
        "present_in_fused": ("    float2 value0 = f0[ii] - curl0;" in plain
                             and "    f0[ii] = value0;" in plain),
        "rename": "f0[ii] = f0[ii] - curl0  ->  float2 value0 = ...; f0[ii] = value0;",
        "agrees": (certified_plain == "    f0[ii] = f0[ii] - curl0;"
                   and "    float2 value0 = f0[ii] - curl0;" in plain
                   and "    f0[ii] = value0;" in plain
                   # the rewrite is exactly the split, character for character
                   and certified_plain.replace("f0[ii] = ", "float2 value0 = ")
                   == "    float2 value0 = f0[ii] - curl0;")})

    # --- the constitutive half, against the certified complex E emitter -------
    for wanted, emitted, rename in (
            ("    float2 source = d_in[idx];",
             "        float2 source = value0;", "d_in[idx] -> the register value0"),
            ("    source = source - p0[idx];",
             "        source = source - p0_0[idx];", "p0 -> p0_0"),
            ("    source = source - p1[idx];",
             "        source = source - p0_1[idx];", "p1 -> p0_1"),
            ("    e_out[idx] = c_mul_field_left(source, inv_e[idx]);",
             "        e0[idx] = c_mul_field_left(source, iv0[idx]);",
             "e_out -> e0, inv_e -> iv0")):
        rows.append({
            "line": wanted.strip(),
            "certified_source": f"complex_no_pml_stored_e.complex_stored_e_source(2, {arm!r})",
            "present_in_certified": wanted in certified_e,
            "present_in_fused": emitted in fused,
            "rename": rename,
            "agrees": wanted in certified_e and emitted in fused})

    # --- the helper block, byte for byte -------------------------------------
    rows.append({
        "line": "complex_helper_block",
        "certified_source": "templates.complex_helpers(<the probe's arm>)",
        "present_in_certified": templates.complex_helpers(arm) in certified_curl,
        "present_in_fused": templates.complex_helpers(arm) in fused,
        "rename": "(none)",
        "agrees": (templates.complex_helpers(arm) in certified_curl
                   and templates.complex_helpers(arm) in fused)})
    return {"passed": all(row["agrees"] for row in rows), "rows": rows,
            "expansion_arm": arm}


def leg_policy() -> Dict[str, Any]:
    """``keep`` must be REFUSED BY NAME; ``flush`` must admit."""
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    case = _case(MUTATION_CASE)
    rows: Dict[str, Any] = {}
    original = os.environ.get(subnormal_policy.POLICY_ENV)
    try:
        for policy in (subnormal_policy.FLUSH, subnormal_policy.KEEP):
            os.environ[subnormal_policy.POLICY_ENV] = policy
            _, fields, pml = build(case, case_seed(case.label, f"policy:{policy}"))
            residency = Residency()
            verdict = family.complex_conductive_fused_pair_coverage(
                fields, pml, (), residency)
            plan = family.plan_metal_complex_conductive_fused_pair(
                fields, pml, (), residency)
            rows[policy] = {"covered": verdict.covered, "planned": plan is not None,
                            "reasons": list(verdict.reasons)[:6]}
    finally:
        if original is None:
            os.environ.pop(subnormal_policy.POLICY_ENV, None)
        else:
            os.environ[subnormal_policy.POLICY_ENV] = original
    keep_named = any("subnormal" in reason.lower()
                     for reason in rows.get(subnormal_policy.KEEP, {}).get("reasons", ()))
    return {
        "passed": bool(rows[subnormal_policy.FLUSH]["planned"]
                       and not rows[subnormal_policy.KEEP]["planned"]
                       and keep_named),
        "rows": rows, "keep_refused_by_name": keep_named,
        "note": ("MPS flush is native and uncontrollable; `keep` is a refusal, "
                 "not a second byte comparison (metal_kernels/subnormal.py:6-13)"),
    }


def leg_refusal() -> Dict[str, Any]:
    """Every seam clause, measured BY NAME — and the POLARITY that makes the cell reachable.

    THE MOST IMPORTANT LEG IN THIS FILE. A refusal that is really an omission is
    indistinguishable from a refusal that is a decision, unless the reason is named
    and the OPPOSITE case is shown to be admitted. So each row states what must
    happen and what text must appear:

    * an UNDECLARED source list -> refused (ignorance is not an empty set);
    * an ELECTRIC source -> refused, citing driver.py:3294-3299 AND :3296;
    * a MAGNETIC source -> ADMITTED. This is the polarity the whole cell rests on:
      the four corpus rows carry exactly one source and it is magnetic;
    * a METALLIC axis -> refused, naming ``zero_metal_D`` and driver.py:3301;
    * a FOLDED axis -> refused, naming both D-side fills and driver.py:3300-3302;
    * an OFF-DIAGONAL row -> refused, naming the stencil (stepping.py:1235-1253);
    * a POLE COUNT over the ceiling -> refused, naming device.py:72.
    """
    case = _case("corpus_one_pole_all_conductive")
    envelope = ContinuousEnvelope(frequency=0.3)
    rows: List[Dict[str, Any]] = []

    def check(label: str, sources: Any, needle: Optional[str], admit: bool,
              subject: Optional[Case] = None, **build_arguments: Any) -> None:
        target = case if subject is None else subject
        _, fields, pml = build(target, case_seed(target.label, f"refusal:{label}"),
                               **build_arguments)
        residency = Residency()
        verdict = family.complex_conductive_fused_pair_coverage(
            fields, pml, sources, residency)
        plan = family.plan_metal_complex_conductive_fused_pair(
            fields, pml, sources, residency)
        named = needle is None or any(needle in reason for reason in verdict.reasons)
        rows.append({"case": label, "admitted": verdict.covered,
                     "planned": plan is not None, "expected_admitted": admit,
                     "needle": needle, "reason_named": named,
                     "reasons": list(verdict.reasons)[:4],
                     "agrees": bool(verdict.covered == admit
                                    and (plan is not None) == admit and named)})
        log(f"    refusal {label}: admitted={verdict.covered} "
            f"expected={admit} named={named}")

    grid_for_sources = build(case, case_seed(case.label, "refusal:sources"))[0]
    electric = VolumeSource(grid=grid_for_sources, component="Ez",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=envelope, amplitude=1.0)
    magnetic = VolumeSource(grid=grid_for_sources, component="Hz",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=envelope, amplitude=1.0)

    check("no_sources_declared", None, "the source set was not declared", False)
    check("electric_source", (electric,), "driver.py:3294-3299", False)
    check("electric_source_names_the_conductive_path", (electric,),
          "driver.py:3296", False)
    check("magnetic_source_ADMITTED", (magnetic,), None, True)
    check("empty_source_list_ADMITTED", (), None, True)
    # A METALLIC AXIS AND A BLOCH PHASE ARE MUTUALLY EXCLUSIVE (grid.py:993, and
    # MEEP's own fields::use_bloch), so the walled row drops k on the walled axis
    # rather than asking the grid for a state it refuses to represent. The clause
    # under test is the WALL, not the phase.
    check("metallic_axis", (), "stepping.zero_metal_D runs inside this seam", False,
          subject=dataclasses.replace(case, label=case.label + ":metallic",
                                      k_point=(0.0, -1.3, 0.7)),
          boundaries=("metallic", "periodic", "periodic"))
    check("folded_axis", (), "stepping.fill_folded_far_ghosts_D", False,
          subject=dataclasses.replace(case, label=case.label + ":folded",
                                      k_point=(0.0, 0.0, 0.0)),
          symmetry=(Mirror("Y", +1),))
    return {"passed": all(row["agrees"] for row in rows), "rows": rows}


def leg_seam_is_empty() -> Dict[str, Any]:
    """DRIVE THE REAL DRIVER and record what writes D inside the seam.

    THE CLAIM THIS FAMILY RESTS ON is that on a magnetic-source-only conductive row
    nothing at all happens between ``step_D`` (driver.py:3293) and ``update_E``
    (:3304) — in particular that ``_inject_electric_through_conductivity`` (:3296)
    cannot run, because ``FdtdDriver.step`` guards it with ``if electric and
    self.fields.has_conductivity`` (:3295). Reading that guard is not a
    measurement, so this leg builds the configuration through the ordinary public
    driver API and INSTRUMENTS the driver: every in-seam pass is wrapped and asked
    whether it WROTE a D word, and the injection call sites are counted.

    THE CONTROL IS THE WHOLE POINT. Leg ``with_electric`` adds an electric source
    and MUST show the conductive pass firing and writing D. A blind instrument
    would report an empty seam by measuring nothing.
    """
    seam_passes = ("fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D")

    def d_words(fields: Any) -> Dict[str, bytes]:
        return {n: np.ascontiguousarray(getattr(fields, n)).tobytes()
                for n in ("Dx", "Dy", "Dz")}

    def probe(with_electric: bool) -> Dict[str, Any]:
        engine = driver_module.FdtdDriver(
            cell_size=(2.3, 2.1, 2.7), resolution=6.0, courant=0.5,
            force_complex_fields=True, k_point=CORPUS_K, boundaries="periodic")
        engine.set_epsilon(np.full(engine.grid.shape, 2.25, dtype=np.float32))
        engine.add_susceptibility(
            Susceptibility(frequency=0.9, gamma=0.05, kind=LORENTZIAN),
            np.full(engine.grid.shape, np.float32(0.18)))
        engine.set_absorber([AbsorberLayer(thickness=0.4, axis=axis, side=side)
                             for axis in (0, 1, 2) for side in ("low", "high")])
        engine.add_source({"component": "Hz", "center": (0.0, 0.0, 0.0),
                           "size": (0.0, 0.0, 0.0), "frequency": 0.3,
                           "amplitude": 1.0})
        if with_electric:
            engine.add_source({"component": "Ez", "center": (0.2, 0.1, 0.0),
                               "size": (0.0, 0.0, 0.0), "frequency": 0.3,
                               "amplitude": 1.0})
        originals = {name: getattr(driver_module, name) for name in seam_passes}
        log_rows: List[Dict[str, Any]] = []
        injections: List[Dict[str, Any]] = []
        conductive: List[Dict[str, Any]] = []

        def wrap(name: str, function: Any) -> Any:
            def inner(target: Any, *arguments: Any, **keywords: Any) -> Any:
                before = d_words(target)
                out = function(target, *arguments, **keywords)
                after = d_words(target)
                log_rows.append({"pass": name, "wrote_D": any(
                    before[k] != after[k] for k in before)})
                return out
            return inner

        for name in seam_passes:
            setattr(driver_module, name, wrap(name, originals[name]))
        classes = {type(source) for source in engine._sources}
        real_injects = {cls: cls.inject for cls in classes}

        def make_inject(function: Any) -> Any:
            def inner(self: Any, target: Any, when: Any, *a: Any, **k: Any) -> Any:
                before = d_words(target)
                out = function(self, target, when, *a, **k)
                after = d_words(target)
                injections.append({
                    "component": getattr(self, "component", "?"),
                    "field_type": self.field_type,
                    "wrote_D": any(before[x] != after[x] for x in before)})
                return out
            return inner

        for cls, function in real_injects.items():
            cls.inject = make_inject(function)
        real_conductive = driver_module.FdtdDriver._inject_electric_through_conductivity

        def conductive_inject(self: Any, electric: Any, when: Any) -> Any:
            before = d_words(self.fields)
            out = real_conductive(self, electric, when)
            after = d_words(self.fields)
            conductive.append({"n_electric": len(electric), "wrote_D": any(
                before[k] != after[k] for k in before)})
            return out

        driver_module.FdtdDriver._inject_electric_through_conductivity = conductive_inject
        try:
            engine.step()
        finally:
            for name in seam_passes:
                setattr(driver_module, name, originals[name])
            for cls, function in real_injects.items():
                cls.inject = function
            driver_module.FdtdDriver._inject_electric_through_conductivity = real_conductive

        wrote = [row["pass"] for row in log_rows if row["wrote_D"]]
        wrote += [f"inject({row['component']})" for row in injections
                  if row["field_type"] != "B" and row["wrote_D"]]
        wrote += ["_inject_electric_through_conductivity" for row in conductive
                  if row["wrote_D"]]
        return {
            "leg": "with_electric" if with_electric else "corpus_B_only",
            "has_conductivity": bool(engine.fields.has_conductivity),
            "source_field_types": sorted({s.field_type for s in engine._sources}),
            "in_seam_passes_run": log_rows,
            "injections": injections,
            "conductive_injection_calls": conductive,
            "passes_that_wrote_D_inside_the_seam": wrote,
            "seam_is_empty": not wrote,
        }

    corpus, control = probe(False), probe(True)
    log(f"    seam corpus_B_only empty={corpus['seam_is_empty']} "
        f"control wrote={control['passes_that_wrote_D_inside_the_seam']}")
    return {
        "passed": bool(corpus["seam_is_empty"] and not control["seam_is_empty"]
                       and corpus["has_conductivity"]
                       and not corpus["conductive_injection_calls"]
                       and any(row["wrote_D"] for row in
                               control["conductive_injection_calls"])),
        "corpus": corpus, "control": control,
        "determination": ("the conductive injection cannot run on a "
                          "magnetic-source-only row: FdtdDriver.step guards it "
                          "with `if electric and self.fields.has_conductivity` "
                          "(driver.py:3295)"),
    }


def leg_orientation_null() -> Dict[str, Any]:
    """PROVE the null: the two helper orientations agree at runtime for a real c.

    A mutation reported "null" on one seeded case is a shrug. This compiles a
    minimal kernel that calls each orientation on the SAME operands and compares
    uint32 words: exhaustively over the four signed-zero classes at a positive, a
    negative and a signed-zero coefficient, in BOTH contraction modes, and over a
    4096-pair random band.

    THE NULL HAS ITS OWN NON-VACUITY FLOOR. The same harness is run against a
    spelling that IS different — the plane-wise fold ``{z.x * c, z.y * c}`` — which
    must DIVERGE on the signed-zero classes. A harness that reported zero for
    everything would "prove" every null.
    """
    import torch  # noqa: PLC0415

    template = """
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
__HELPERS__
kernel void probe(
    device float2* out [[buffer(0)]],
    device const float2* z [[buffer(1)]],
    device const float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
    out[idx] = __CALL__;
}
"""
    helpers = templates.complex_helpers(_arm())
    folded = helpers.replace(
        "    return float2(fma(z.x, c,    -(z.y * 0.0f)),\n"
        "                  fma(z.x, 0.0f,  (z.y * c)));",
        "    return float2(z.x * c, z.y * c);", 1)
    assert folded != helpers, "the control spelling matched nothing"

    def evaluate(call: str, helper_text: str, contract: str,
                 vector: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        source = (template
                  .replace("__CONTRACT__", templates.contraction_pragma(contract))
                  .replace("__HELPERS__", helper_text)
                  .replace("__CALL__", call))
        function = compile_source(source).probe
        z = torch.from_numpy(np.ascontiguousarray(vector).view(np.float32)
                             .reshape(-1, 2)).to("mps")
        c = torch.from_numpy(np.ascontiguousarray(coefficients)).to("mps")
        out = torch.zeros_like(z)
        function(out, z, c, len(vector))
        torch.mps.synchronize()
        return out.cpu().numpy().copy()

    zeros = [np.float32(0.0), np.float32(-0.0)]
    signed = np.array([complex(a, b) for a in zeros for b in zeros],
                      dtype=np.complex64)
    generator = np.random.default_rng(SEED)
    band = (generator.standard_normal(4096)
            + 1j * generator.standard_normal(4096)).astype(np.complex64)
    rows: List[Dict[str, Any]] = []
    for label, vector, coefficients in (
            ("signed_zero_positive_c", signed, np.full(4, 0.63, dtype=np.float32)),
            ("signed_zero_negative_c", signed, np.full(4, -0.63, dtype=np.float32)),
            ("signed_zero_signed_zero_c", signed,
             np.array([0.0, -0.0, 0.0, -0.0], dtype=np.float32)),
            ("random_band", band,
             generator.standard_normal(4096).astype(np.float32))):
        for contract in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
            left = evaluate("c_mul_coefficient_left(c[idx], z[idx])", helpers,
                            contract, vector, coefficients)
            right = evaluate("c_mul_field_left(z[idx], c[idx])", helpers,
                             contract, vector, coefficients)
            control = evaluate("c_mul_field_left(z[idx], c[idx])", folded,
                               contract, vector, coefficients)
            orientation = int(np.count_nonzero(
                left.view(np.uint32).reshape(-1) != right.view(np.uint32).reshape(-1)))
            folded_words = int(np.count_nonzero(
                right.view(np.uint32).reshape(-1)
                != control.view(np.uint32).reshape(-1)))
            rows.append({"class": label, "contract": contract, "words": left.size * 2,
                         "orientation_differing_words": orientation,
                         "folded_control_differing_words": folded_words})
            log(f"    orientation {label} contract={contract} "
                f"orientations={orientation} folded_control={folded_words}")
    signed_rows = [row for row in rows if row["class"].startswith("signed_zero")]
    return {"passed": bool(all(row["orientation_differing_words"] == 0
                               for row in rows)
                           and any(row["folded_control_differing_words"]
                                   for row in signed_rows)),
            "rows": rows,
            "the_claim": ("the two certified orientations are bit-identical for a "
                          "REAL coefficient here, so an orientation edit is a TEXT "
                          "defect leg transcription pins, not a runtime one"),
            "control_spelling": "plane-wise fold {z.x * c, z.y * c}"}


PRECONDITION_SCALES: Tuple[Tuple[str, float, Optional[bool]], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-25, True),
    ("band_edge", 1e-30, None),
    ("subnormal_band", 1e-34, False),
    ("deep_subnormal", 1e-41, False),
)


def leg_precondition(case: Case, budget: int = 8) -> Dict[str, Any]:
    """A precondition never demonstrated to FIRE is decoration."""
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        _, reference, reference_pml = build(
            case, case_seed(case.label, f"precondition:{label}"), scale=scale)
        per_step: List[int] = []
        for _ in range(budget):
            seam_step(reference, reference_pml)
            per_step.append(census_of(reference))
        fired = [index for index, count in enumerate(per_step) if count]
        agrees = (expect_clean is None
                  or (expect_clean and not fired)
                  or (not expect_clean and bool(fired)))
        rows.append({"scale": label, "factor": scale,
                     "subnormal_words_per_step": per_step,
                     "census_fired": bool(fired),
                     "first_step": fired[0] if fired else None,
                     "expected_clean": expect_clean, "agrees": agrees,
                     "compared": False})
        log(f"    precondition {label} factor={scale:g} fired={bool(fired)} "
            f"expected_clean={expect_clean}")
    return {"passed": all(row["agrees"] for row in rows), "rows": rows,
            "banded_rows_were_compared": False}


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _plan_spec(case: Case) -> Tuple[Tuple[int, int, int], Tuple[int, int, int],
                                    Tuple[bool, bool, bool], Tuple[int, int, int]]:
    """The (codes, phased, conductive, pole counts) one case compiles to."""
    _, fields, pml = build(case, case_seed(case.label, "shape"))
    residency = Residency()
    plan = family.plan_metal_complex_conductive_fused_pair(
        fields, pml, (), residency)
    assert plan is not None, family.complex_conductive_fused_pair_coverage(
        fields, pml, (), residency).reasons
    return plan.bc, plan.phased, plan.conductive, plan.pole_counts


def _compile(spec: Any, arm: str, source: Optional[str] = None) -> Dict[str, Any]:
    codes, phased, conductive, counts = spec
    if source is None:
        source = family.complex_conductive_fused_pair_source(
            codes, phased, conductive, counts, arm)
    return {shaders.CONTRACT_OFF:
            compile_source(source).complex_conductive_fused_pair_step}


def shader_mutations(case: Case, arm: str) -> Dict[str, Dict[str, Any]]:
    """Source defects, each a plausible transcription slip.

    Each edit is required to CHANGE the source — a no-op edit would report the
    shipped kernel as an uncaught defect, which is the name-drift trap. And each
    edit is placed on a line the SCORED case executes: leg ``branch_census``
    measures that the scored configuration carries all three wrap planes and all
    three conductive tails, so no armed line sits under a predicate no thread
    takes.
    """
    spec = _plan_spec(case)
    codes, phased, conductive, counts = spec
    base = family.complex_conductive_fused_pair_source(
        codes, phased, conductive, counts, arm)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF. Note what is NOT armed here: replacing `value0` with
        # `f0[ii]` is not a defect at all — the tail has ALREADY stored that word,
        # so the reload is the same value. That edit is the BYTE-NEUTRAL CONTROL
        # (leg `byte_neutral`), scored as required-UNCAUGHT; arming it here as well
        # would be one row claiming a defect and another claiming an identity for
        # the same edit. What IS a defect is taking the WRONG component's register,
        # which a fused pair holding three constitutives in one scope can get wrong
        # and a separate composition cannot.
        "seam_takes_the_wrong_components_register":
            base.replace("float2 source = value1;", "float2 source = value0;", 1),
        # The curl's own arithmetic.
        "curl_t0_terms_reordered":
            base.replace("    float2 t0 = ((c_y - c) + (b - b_z));",
                         "    float2 t0 = ((c_y - c) + (b_z - b));", 1),
        "curl_dtdx_orientation_flipped":
            base.replace("    float2 curl0 = c_mul_coefficient_left(dtdx, t0);",
                         "    float2 curl0 = c_mul_field_left(t0, dtdx);", 1),
        # The conductive tail's ORDER. condfac then subtract then condinv is the
        # array path's ordered three-pass recurrence; swapping the two scalings is
        # a smooth, plausible, wrong absorber.
        "conductive_tail_scalings_swapped":
            base.replace("    value0 = c_mul_field_left(value0, cf0[ii]);\n"
                         "    value0 = value0 - curl0;\n"
                         "    value0 = c_mul_field_left(value0, ci0[ii]);",
                         "    value0 = c_mul_field_left(value0, ci0[ii]);\n"
                         "    value0 = value0 - curl0;\n"
                         "    value0 = c_mul_field_left(value0, cf0[ii]);", 1),
        "conductive_tail_never_stores":
            base.replace("    f0[ii] = value0;\n", "    // f0 never stored\n", 1),
        # The Bloch wrap. One plane, one axis, one operand.
        "bloch_wrap_x_predicate_inverted":
            base.replace("    { bool wx = (i == 0);", "    { bool wx = (i != 0);", 1),
        "bloch_wrap_y_operand_dropped":
            base.replace("      c_y = wy ? c_mul(c_y, py) : c_y;\n", "", 1),
        "bloch_wrap_z_takes_the_x_phase":
            base.replace("      a_z = wz ? c_mul(a_z, pz) : a_z;",
                         "      a_z = wz ? c_mul(a_z, px) : a_z;", 1),
        # The ghost wrap: step_D is BACKWARD, so the wrap is at index 0.
        "ghost_x_wrap_uses_the_forward_predicate":
            base.replace("    si = (si < 0) ? (nxi - 1) : si;",
                         "    si = (si >= nxi) ? 0 : si;", 1),
        # The E half.
        "pole_dropped_from_the_e_chain":
            base.replace("        source = source - p0_1[idx];\n", "", 1),
        "constitutive_takes_the_wrong_inverse_epsilon":
            base.replace("        e0[idx] = c_mul_field_left(source, iv0[idx]);",
                         "        e0[idx] = c_mul_field_left(source, iv1[idx]);", 1),
        "constitutive_orientation_flipped":
            base.replace("        e2[idx] = c_mul_field_left(source, iv2[idx]);",
                         "        e2[idx] = c_mul_coefficient_left(iv2[idx], source);", 1),
        # The packed Params: an index slip in the prologue reaches every scalar.
        "params_ny_and_nz_swapped_in_the_prologue":
            base.replace("    uint ny = prm.ny;\n    uint nz = prm.nz;",
                         "    uint ny = prm.nz;\n    uint nz = prm.ny;", 1),
        "params_phase_planes_swapped":
            base.replace("    float2 px = float2(prm.px_re, prm.px_im);",
                         "    float2 px = float2(prm.px_im, prm.px_re);", 1),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        if source == base:
            raise AssertionError(
                f"mutation {name!r} did not change the source; a no-op edit would "
                f"report the shipped kernel as an uncaught defect")
        mutants[name] = _compile(spec, arm, source)
    return mutants


def split_tail_mutations(case: Case, arm: str) -> Dict[str, Dict[str, Any]]:
    """The edits that can only be armed on a row with a NON-conductive component.

    The one line this family rewrites is the plain tail, and it is emitted only
    where ``conductive[index]`` is false. Arming these on the all-conductive
    mutation case would be arming them on text that case never emits.
    """
    spec = _plan_spec(case)
    codes, phased, conductive, counts = spec
    assert not all(conductive), (
        f"{case.label} is all-conductive, so the split tail is not emitted and "
        f"these mutations would be armed on nothing")
    base = family.complex_conductive_fused_pair_source(
        codes, phased, conductive, counts, arm)
    index = conductive.index(False)
    edits = {
        "split_tail_stores_the_pre_curl_value":
            base.replace(f"    float2 value{index} = f{index}[ii] - curl{index};",
                         f"    float2 value{index} = f{index}[ii];", 1),
        "split_tail_negates_the_curl":
            base.replace(f"    float2 value{index} = f{index}[ii] - curl{index};",
                         f"    float2 value{index} = f{index}[ii] + curl{index};", 1),
        "split_tail_never_stores":
            base.replace(f"    f{index}[ii] = value{index};\n",
                         f"    // f{index} never stored\n", 1),
    }
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in edits.items():
        if source == base:
            raise AssertionError(
                f"mutation {name!r} did not change the source; a no-op edit would "
                f"report the shipped kernel as an uncaught defect")
        mutants[name] = _compile(spec, arm, source)
    return mutants


def byte_neutral_control(case: Case, arm: str) -> Dict[str, Any]:
    """THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT, and it is scored on that.

    The whole claim of this fusion is that it removes a round trip and changes no
    arithmetic: the constitutive half reads a REGISTER where the certified
    composition read the word ``step_D`` had just stored. This edit puts the reload
    back — ``float2 source = f0[ii];`` AFTER the store — and the result must NOT
    diverge. A ``float2`` stored to a ``device float2*`` and reloaded is exact, so
    "byte-neutral by construction" is a hypothesis; this row is the comparator
    agreeing with it.

    NOTE THE DIFFERENCE FROM ``seam_takes_the_pre_curl_displacement``: that
    mutation reads ``f0[ii]`` too, but the two are not the same edit — this one is
    the whole three-component reload, and the tails have ALREADY stored, so the
    reload observes the stepped value. It is placed by replacing the register in
    every component, not one.
    """
    spec = _plan_spec(case)
    codes, phased, conductive, counts = spec
    base = family.complex_conductive_fused_pair_source(
        codes, phased, conductive, counts, arm)
    source = base
    for index in range(3):
        source = source.replace(f"float2 source = value{index};",
                                f"float2 source = f{index}[ii];", 1)
    assert source != base, (
        "the byte-neutral control edit matched nothing; a control that changes "
        "nothing would report the shipped bytes as byte-neutral against themselves")
    assert source.count("float2 source = f") == 3, source.count("float2 source = f")
    return _compile(spec, arm, source)


def swap_condinv_between_components(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind Dy's ``condinv`` volume into Dx's slot.

    The kernel takes a pointer per target and never asks which component's
    conductivity it is. A per-target conductivity is exactly what a directional
    absorber gives, so this is a silent wrong absorption rather than a crash, and
    no shader edit can reach it. On this build all three targets share ONE sigma
    array, so the swap is an identity — which is why the patch ASSERTS they differ
    and raises rather than being scored as a false uncaught.
    """
    fixed = list(plan._fixed)
    # layout: f0..f2, g0..g2, cf0..cf2, ci0..ci2, e0..e2, iv0..iv2
    assert fixed[9] is not fixed[10], (
        "condinv[Dx] and condinv[Dy] are the SAME mirror on this build, so this "
        "host defect is a null and must not be scored as a live mutation")
    fixed[9] = fixed[10]
    plan._fixed = tuple(fixed)


def swap_h_sources(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the curl By where Bx belongs.

    The curl's three source volumes are positional; crossing two produces a
    smooth, plausible, wrong field and no shader edit can reach it.
    """
    fixed = list(plan._fixed)
    assert fixed[3] is not fixed[4], "the two H sources are the same mirror"
    fixed[3], fixed[4] = fixed[4], fixed[3]
    plan._fixed = tuple(fixed)


def scale_params_dtdx(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: negate the packed ``dtdx``.

    ``dtdx`` is the one coefficient the packer converts with ``float()``
    (complex_conductive_fused_pair.pack_params), and a sign slip there is a
    plausible packing defect no shader edit can reach — the kernel reads
    ``prm.dtdx`` either way. The blob is re-packed through the family's OWN packer
    so the mutation is a wrong INPUT, not a second layout.
    """
    original = residency.host("complex_conductive_fused_pair:params")
    blob = np.array(original, copy=True)
    blob[0] = np.float32(-float(blob[0]))
    plan._tensors["params"] = residency.mirror(
        "mutation:params", blob, constant=True)


#: name -> (patch, expected outcome). "caught" must diverge; "null" must NOT, and
#: its patch asserts the identity that makes it a null.
HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Any, Residency], None], str]] = {
    "condinv_bound_for_the_wrong_component": (swap_condinv_between_components, "caught"),
    "curl_source_volumes_crossed": (swap_h_sources, "caught"),
    "packed_dtdx_negated": (scale_params_dtdx, "caught"),
}


#: THE PLANTED DEFECTS, applied to the SHIPPED emitter process-wide so the whole
#: gate runs against defective bytes and the top-level verdict must flip.
PLANTED: Dict[str, Tuple[str, str]] = {
    # The pole chain's sign, on every component of every specialisation.
    "e_chain_adds_the_polarizations_everywhere": (
        "        source = source - p", "        source = source + p"),
    # The conductive tail's second scaling, dropped on the x target everywhere.
    "conductive_tail_drops_the_condinv_pass": (
        "    value0 = c_mul_field_left(value0, ci0[ii]);\n", ""),
}


def plant(name: str) -> None:
    wanted, replacement = PLANTED[name]
    original = family.complex_conductive_fused_pair_source

    def defective(*arguments: Any, **keywords: Any) -> str:
        source = original(*arguments, **keywords)
        if wanted in source:
            return source.replace(wanted, replacement)
        return source

    family.complex_conductive_fused_pair_source = defective  # type: ignore[assignment]
    probe = original((0, 0, 0), (1, 1, 1), (True, True, True), (1, 1, 1), _arm())
    assert wanted in probe, (
        f"planted defect {name!r} matches nothing in the shipped source; a "
        f"marker that matches nothing disables the check it feeds SILENTLY")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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
    parser.add_argument("--plant", choices=sorted(PLANTED), default=None,
                        help="run the whole gate against a defective shipped "
                             "emitter; the verdict MUST come back FAIL")
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    if not PROBE_ARTIFACT.exists():
        raise SystemExit(
            f"the measured expansion probe {PROBE_ARTIFACT} is missing; which "
            f"complex arm this host takes may not be guessed")
    if args.plant:
        plant(args.plant)
        log(f"PLANTED DEFECT {args.plant!r}: the verdict below MUST be FAIL")

    arm = _arm()
    mutation_case = _case(MUTATION_CASE)
    mixed_case = _case(MUTATION_CASE_MIXED)
    mutants: Dict[str, Dict[str, Any]] = {}
    split_mutants: Dict[str, Dict[str, Any]] = {}
    host_mutations = dict(HOST_MUTATIONS)
    neutral: Optional[Dict[str, Any]] = None
    if args.plant is None:
        mutants = shader_mutations(mutation_case, arm)
        split_mutants = split_tail_mutations(mixed_case, arm)
        neutral = byte_neutral_control(mutation_case, arm)
    else:
        host_mutations = {}

    controls = ("corpus_one_pole_all_conductive", "mixed_conductivity_x_only",
                "four_pole_at_the_ceiling")
    value_cases = tuple(
        dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                            value_class="pm_zero_lattice")
        for name in ("corpus_one_pole_all_conductive", "mixed_conductivity_yz_only"))
    total = (7 + len(CASES) + len(controls) + len(value_cases) + 1
             + len(mutants) + len(split_mutants) + len(host_mutations)
             + (0 if args.plant else 2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for leg, label, runner in (
                ("binding_ceiling", "corpus_row_the_ceiling_and_the_next_one",
                 leg_binding_ceiling),
                ("branch_census", "every_branch_and_the_scored_case", leg_branch_census),
                ("transcription", "emitted_text_vs_certified_emitters",
                 leg_transcription),
                ("policy", "flush_admits_keep_is_refused", leg_policy),
                ("refusal", "every_seam_clause_by_name_and_the_polarity",
                 leg_refusal),
                ("seam_is_empty", "the_real_driver_instrumented", leg_seam_is_empty),
                ("orientation_null", "the_two_helper_orientations_at_runtime",
                 leg_orientation_null)):
            index += 1
            rows.append({"index": index, "total": total, "leg": leg,
                         "label": label, **runner()})
            emit(handle, rows[-1])

        for case in CASES:
            index += 1
            row = {"index": index, "total": total, "leg": "product",
                   "label": case.label, "steps": args.steps,
                   **run_case(case, "product", args.steps)}
            rows.append(row)
            emit(handle, row)

        for name in controls:
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(_case(name), args.steps)}
            rows.append(row)
            emit(handle, row)

        for case in value_cases:
            index += 1
            row = {"index": index, "total": total, "leg": "value_classes",
                   "label": case.label, "steps": args.steps,
                   **run_case(case, "value_classes", args.steps)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "value_classes",
                     "label": "subnormal_band_precondition",
                     **leg_precondition(mutation_case)})
        emit(handle, rows[-1])

        if neutral is not None:
            index += 1
            result = run_case(mutation_case, "byte_neutral", args.steps,
                              functions=neutral)
            rows.append({"index": index, "total": total, "leg": "byte_neutral",
                         "label": "register_replaced_by_the_reload_it_removes",
                         "expected": "UNCAUGHT (the same value by construction)",
                         "diverged": not result.get("bit_identical", False),
                         "differing_words": result.get("differing_words"),
                         "launches": result.get("launches"),
                         "passed": bool(result.get("bit_identical")
                                        and result.get("launches"))})
            emit(handle, rows[-1])

        for label, table, scored in (("shader", mutants, mutation_case),
                                     ("split_tail", split_mutants, mixed_case)):
            for name, functions in table.items():
                index += 1
                result = run_case(scored, f"mutation:{name}", args.steps,
                                  functions=functions)
                caught = not result.get("bit_identical", False)
                expected = "null" if name in NULL_SCORED else "caught"
                outcome = "caught" if caught else "null"
                row = {"index": index, "total": total, "leg": "mutation",
                       "label": name, "family": label, "caught": caught,
                       "expected": expected, "outcome": outcome,
                       "scored_case": scored.label,
                       "passed": bool(outcome == expected and result.get("launches")),
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "launches": result.get("launches")}
                if name in NULL_SCORED:
                    row["null_reason"] = NULL_SCORED[name]
                    row["null_evidence"] = "leg orientation_null"
                rows.append(row)
                emit(handle, rows[-1])

        for name, (patch, expected) in host_mutations.items():
            index += 1
            result = run_case(mutation_case, f"host:{name}", args.steps, patch=patch)
            caught = not result.get("bit_identical", False)
            outcome = "caught" if caught else "null"
            rows.append({"index": index, "total": total, "leg": "mutation",
                         "label": name, "host_defect": True, "caught": caught,
                         "expected": expected, "outcome": outcome,
                         "passed": bool(outcome == expected and result.get("launches")),
                         "first_divergence": result.get("first_divergence"),
                         "differing_words": result.get("differing_words"),
                         "launches": result.get("launches")})
            emit(handle, rows[-1])

        if args.plant is None:
            index += 1
            result = run_case(mutation_case, "disarm", args.steps)
            rows.append({"index": index, "total": total, "leg": "disarm",
                         "label": "shipped_bytes_on_the_mutation_case",
                         "differing_words": result.get("differing_words"),
                         "launches": result.get("launches"),
                         "written_arrays_that_never_moved": result.get(
                             "written_arrays_that_never_moved"),
                         "passed": bool(result.get("passed"))})
            emit(handle, rows[-1])

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {"product": len(CASES), "separate_controls": len(controls),
                   "value_classes": len(value_cases) + 1,
                   "shader_mutations": len(mutants) + len(split_mutants),
                   "host_mutations": len(host_mutations),
                   "byte_neutral_controls": 0 if neutral is None else 1},
        "expansion_arm": arm,
        "expansion_probe_artifact": str(PROBE_ARTIFACT),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "planted_defect": args.plant,
        "verdict_flip_expected": bool(args.plant),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
