#!/usr/bin/env python3
"""Native MPS byte gate for the COMPLEX fused E->P chain: ``update_E`` -> ``update_P``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes (``update_E`` at driver.py:3304 and ``update_P`` at :3306) with nothing
between them, so a per-sub-step comparison could not see it at all: the whole claim
is about the SEAM. The comparison is therefore per COMPLETE SEAM STEP over a stated
budget, against an array-path oracle running the identical two passes, with the
first divergent step reported rather than a final pass/fail.

THE SEVEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. So every
   case asserts the exact LAUNCH COUNT (``launches_per_run`` x steps), a nonzero
   ``runs``, an ``alias_checks`` count equal to the launches, and a NON-VACUITY
   FLOOR SPLIT BY ROLE — the arrays this seam WRITES (E, every polarization array)
   must all have moved, and the arrays it only READS (B, D, H) must NOT have.
2. **A hollow pass.** Source mutations and HOST mutations are armed, and each must
   be CAUGHT — or, where an edit is provably a no-op on the scored configuration,
   NULL CONFIRMED by an identity check rather than reported as uncaught. Leg
   ``disarm`` reruns the identical case with the shipped bytes and requires zero.
3. **A dead-branch mutation.** A mutation that rewrites a line the scored grid
   never reaches measures nothing. This kernel has EXACTLY ONE branch — the
   ``n_elem`` guard — and :func:`leg_single_guard` asserts that from the emitted
   text, so every other emitted line is reached by every in-range thread and no
   mutation can be dead.
4. **A mirrored evaluator.** :func:`leg_transcription` PARSES the load-bearing
   arithmetic lines out of this family's emitted source AND out of the CERTIFIED
   emitters' own output, and requires them equal modulo the documented pointer
   renames. It does not re-implement either.
5. **An inherited licence.** ``probe_metal_ade_rotation_seam`` established the
   per-component interleave and the alias-free orbit for the two REAL diagonal E
   bodies and SKIPPED this cell by name (probe_metal_ade_rotation_seam.py:579-584).
   So leg ``separate_control`` re-establishes it here on complex storage: the array
   path, the two ALREADY CERTIFIED complex products in the DRIVER's order, and the
   fused product, three ways, word for word, at every one of sixty steps.
6. **An unmeasured arm.** Both halves multiply complex numbers and the two
   certified halves take their expansion arm from DIFFERENT places — the E half
   from the measured probe, the ADE half baked in at ade_update_p.py:105. Leg
   ``arm_binding`` measures that they agree TODAY and that a disagreeing probe is
   REFUSED BY NAME, so the agreement is a checked precondition rather than a
   coincidence the product silently rests on.
7. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor (a zero state is a fixed point of this seam)
   and is instead required to carry BOTH signs of zero in the reference output, so
   a uint32 compare is actually discriminating something; the subnormal band is
   REFUSED BY NAME and its census required to FIRE, which is what makes the empty
   censuses next door a measurement rather than a silence.

BOTH SUBNORMAL POLICIES, AS FAR AS THIS BACKEND HAS THEM. MPS flush is native and
has NO LEVER (metal_kernels/subnormal.py:6-13), so ``keep`` is not a second
comparison but a REFUSAL, and leg ``policy`` requires the plan to be refused BY
NAME under it rather than silently downgraded.

THE EXPANSION PROBE IS THE MEASURED ARTIFACT, NOT AN INLINE DICT. This gate reads
``results/metal_complex_audit_2026-08-16/complex_expansion_probe.json`` — the same
record ``recut_metal_gates.sh`` exports — so the claim rests on a measurement of
this host rather than on a literal typed into a gate. A synthesized DISAGREEING
record is used in exactly one place, leg ``arm_binding``, to show the refusal fires.

LEGS
  0  binding_ceiling   worst in-ceiling signature COMPILES; the next one FAILS
  1  single_guard      the emitted kernel has exactly one branch
  2  transcription     the arithmetic lines equal the certified emitters', parsed
  3  arm_binding       probe arm == ADE's baked arm; a disagreeing probe is refused
  4  policy            `keep` is REFUSED by name; `flush` admits
  5  product           complete seam steps, per-step byte compare, counters, floor
  6  separate_control  array path vs the two CERTIFIED complex products vs fused
  7  value_classes     uniform, +-0 lattice, subnormal band (refused, census fires)
  8  byte_neutral      the register-vs-reload edit must NOT diverge
  9  mutation          armed defects, each CAUGHT or NULL CONFIRMED
 10  disarm            the same harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_metal_complex_fused_ade_chain.py --out <dir>/chain.json
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
#: Exported before any meep_gpu import so ``load_expansion_probe`` finds it, which
#: is how ``recut_metal_gates.sh`` arms every complex gate.
PROBE_ARTIFACT = (API_ROOT / "parity" / "meep_gpu" / "results"
                  / "metal_complex_audit_2026-08-16"
                  / "complex_expansion_probe.json")
os.environ.setdefault("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE", str(PROBE_ARTIFACT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, complex_fused_ade_chain as family, complex_no_pml_stored_e,
    shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.complex_fields import load_expansion_probe  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

#: Complete seam steps every case walks. SIXTY, not twelve: the whole ADE history
#: is state carried between steps and the rotation's own period is only ``d + 1``,
#: so the defects this gate exists for COMPOUND. The comparison is per COMPLETE
#: STEP, so a green row means identical at ALL sixty — and step 1 is reported
#: separately, because "identical at one launch" and "identical at sixty" are
#: different claims and both are made here.
STEPS = 60

#: Base for every per-case seed. The seed itself is ``SEED + sha256(label|leg)``,
#: never ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``,
#: so a hash-seeded gate draws a different fixture every process and a failing
#: case cannot be replayed from its own record.
SEED = 91_000

STATE_NAMES: Tuple[str, ...] = tuple(
    f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz")

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line: demanding movement from all of them fails a correct
#: run, demanding it from none passes a run that launched nothing.
WRITTEN_PREFIXES = ("E", "P[", "P_prev[", "scratch[")
READ_ONLY_PREFIXES = ("B", "D", "H")

#: The four signed-zero classes a complex operand plane pair can take. Used by the
#: ``sign_class_product`` value class to enumerate the COMPLETE operand product,
#: which is what turns a null mutation into a measurement rather than a shrug.
SIGN_CLASSES: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0))

#: The corpus configuration this family exists for, read off
#: ``results/metal_coverage_tranche6_2026-08-19``: the four
#: ``TestLoadDump.*_3d`` rows are complex64, no PML, Bloch k = (0.4, -1.3, 0.7),
#: ONE Lorentzian driving Ex/Ey/Ez with a SCALAR sigma.
CORPUS_K = (0.4, -1.3, 0.7)
CORPUS_POLES = 1


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration."""

    label: str
    poles: int
    kind: str                # LORENTZIAN | DRUDE | "mixed"
    sigma_volume: bool
    k_point: Tuple[float, float, float] = CORPUS_K
    cell: Tuple[float, float, float] = (1.2, 1.0, 0.9)
    value_class: str = "uniform"


#: THE CASE MATRIX. The corpus row is FIRST and named as such; the rest widen the
#: pole count and the sigma kind up to the binding ceiling, so the ceiling leg's
#: arithmetic is exercised by a running case and not only by a compile.
CASES: Tuple[Case, ...] = (
    Case("corpus_one_pole_scalar_sigma", CORPUS_POLES, LORENTZIAN, False),
    Case("one_pole_volume_sigma", 1, DRUDE, True),
    Case("three_pole_mixed_volume_sigma", 3, "mixed", True),
    Case("five_pole_mixed_scalar_sigma", 5, "mixed", False),
    Case("six_pole_volume_sigma_at_the_ceiling", 6, "mixed", True),
    Case("zero_k_point_control", 2, LORENTZIAN, True, k_point=(0.0, 0.0, 0.0)),
)

#: The case every mutation is scored on. SIX poles with VOLUME sigmas, so every
#: armed line is emitted: a one-pole row would make the "second pole" edits vacuous
#: and a scalar-sigma row would leave the ``s1[idx]`` edit unfilled.
MUTATION_CASE = "six_pole_volume_sigma_at_the_ceiling"


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str, leg: str) -> int:
    """A replayable per-case seed: ``SEED + sha256(label|leg)``, never ``hash()``."""
    digest = hashlib.sha256(f"{label}|{leg}".encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose — and the only
    comparison that can tell ``-0.0`` from ``+0.0``, which the lattice class needs."""
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

def build(case: Case, seed: int, scale: float = 1.0
          ) -> Tuple[Grid, Fields, Optional[PML]]:
    """One seeded engine. Called twice per comparison, identically, for both sides.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES unless a value class says
    otherwise, and that is the first thing this gate establishes rather than an
    afterthought: a zero-initialised state is a FIXED POINT of this seam — with
    D = P = P_prev = 0 the constitutive gives E = 0 and the recurrence gives
    P = 0 forever — so a deliberately wrong kernel compared against a zero oracle
    reports IDENTICAL.
    """
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell), boundaries="periodic",
                dimensions=3, courant=0.35, k_point=tuple(case.k_point), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = (1.45 + 0.30 * rng.random(shape)).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    kinds = (DRUDE, LORENTZIAN) if case.kind == "mixed" else (case.kind,)
    for order in range(case.poles):
        base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
                "Ez": 0.19 + 0.06 * order}
        if case.sigma_volume:
            sigma: Any = {name: (value * (0.7 + 0.6 * rng.random(shape))
                                 ).astype(np.float32)
                          for name, value in base.items()}
        else:
            sigma = base
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            sigma, grid, np.complex64))
    fields.enable_field_storage()

    if case.value_class == "pm_zero_lattice":
        # A lattice of +0.0 and -0.0 over both PLANES of the DISPLACEMENT, with the
        # inverse epsilon left NORMAL so the sign actually propagates through a
        # multiply. Exempt from the movement floor by construction; the floor it
        # answers to instead is "the reference output carries BOTH signs".
        #
        # THE POLARIZATIONS ARE SEEDED +0 ON BOTH PLANES, AND THAT IS LOAD-BEARING
        # rather than tidy. The E half's chain is `source = d_in - p0 - p1 ...`,
        # and IEEE subtraction of equal signed zeros gives +0: seeding P with the
        # SAME lattice wipes the sign out of `source` on the first pole and leaves
        # every downstream multiply looking at +0, which is how an earlier revision
        # of this leg reported three live arm-spelling defects as UNCAUGHT while
        # measuring nothing. `(-0) - (+0)` is `-0`, so the lattice survives the
        # chain at any pole count.
        # THE PLANES ARE ASSIGNED THROUGH THE .real/.imag VIEWS, not composed with
        # `real + 1j * imaginary`. That composition is a complex MULTIPLY and an
        # ADD, and it silently loses the sign of a zero: measured on this host, it
        # turned every intended -0.0 imaginary word into +0.0 and left the lattice
        # discriminating in one plane only, which is how an earlier revision of
        # this leg reported the two folded-cross-term defects as UNCAUGHT. The
        # imaginary plane is the discriminating one (probe: the two spellings
        # differ exactly when z.y is -0.0), so losing it lost the whole leg.
        count = int(np.prod(shape))
        index = np.arange(count).reshape(shape)
        lattice = np.zeros(shape, dtype=np.complex64)
        lattice.real = np.where((index % 2) == 0,
                                np.float32(0.0), np.float32(-0.0)).astype(np.float32)
        lattice.imag = np.where((index % 3) == 0,
                                np.float32(-0.0), np.float32(0.0)).astype(np.float32)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = np.complex64(0.0)
                state.P_prev[component][...] = np.complex64(0.0)
        return grid, fields, PML(grid=grid, thickness=0)

    if case.value_class == "sign_class_product":
        # THE COMPLETE PRODUCT OF SIGNED-ZERO OPERAND CLASSES, and it exists to
        # make a NULL a measurement. D, P and P_prev each independently cycle the
        # four (+-0, +-0) sign patterns on a stride of 16, 4 and 1, so all 64
        # combinations occur — and every ADE arm therefore sees every combination
        # of its p, q and drive signs somewhere on the grid. A defect that this
        # class cannot expose is not one the seam's operands can express.
        index = np.arange(int(np.prod(shape))).reshape(shape)

        def pattern(stride: int) -> np.ndarray:
            value = np.zeros(shape, dtype=np.complex64)
            real = np.zeros(shape, dtype=np.float32)
            imaginary = np.zeros(shape, dtype=np.float32)
            for slot, (r, i) in enumerate(SIGN_CLASSES):
                mask = (index // stride) % len(SIGN_CLASSES) == slot
                real[mask] = np.float32(r)
                imaginary[mask] = np.float32(i)
            value.real = real          # planes assigned directly; see above
            value.imag = imaginary
            return value
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = pattern(16)
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = pattern(4)
                state.P_prev[component][...] = pattern(1)
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
    """Every semantically live array, with the ADE roles NAMED rather than aliased.

    ``update_P`` ROTATES which physical allocation holds ``P``, so a static string
    table could not name them without losing the semantic role — which is the one
    thing the comparison is about.
    """
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
    """The two array-path passes this product replaces, in driver order."""
    stepping.update_E(fields, pml)          # driver.py:3304
    stepping.update_P(fields, pml)          # driver.py:3306


def census_of(fields: Fields) -> int:
    return sum(subnormal.census(value) for value in state_of(fields).values())


def negative_zeros(fields: Fields) -> int:
    """How many ``-0.0`` words the state carries. ``0x80000000`` exactly."""
    return sum(int(np.count_nonzero(words(value) == np.uint32(0x80000000)))
               for value in state_of(fields).values())


def positive_zeros(fields: Fields) -> int:
    return sum(int(np.count_nonzero(words(value) == np.uint32(0)))
               for value in state_of(fields).values())


def negative_zeros_per_plane(fields: Fields) -> Tuple[int, int]:
    """``-0.0`` words in the REAL and IMAGINARY planes, counted separately.

    NOT DECORATION. The two folded cross-term spellings this gate arms differ from
    the certified ones exactly when the IMAGINARY operand is ``-0.0`` (measured on
    this device, both orientations), so a lattice that carries negative zeros only
    in the real plane discriminates neither — and the pooled count cannot tell the
    two situations apart. An earlier revision of this leg had exactly that lattice
    and reported both defects UNCAUGHT.
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


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _split_movement(before: Mapping[str, np.ndarray], after: Mapping[str, Any]
                    ) -> Tuple[List[str], List[str], int]:
    moved = {name: differing(before[name], after[name]) for name in before}
    written = [name for name in moved if name.startswith(WRITTEN_PREFIXES)]
    read_only = [name for name in moved if name.startswith(READ_ONLY_PREFIXES)]
    assert set(written) | set(read_only) == set(moved), sorted(
        set(moved) - set(written) - set(read_only))
    still = sorted(name for name in written if moved[name] == 0)
    disturbed = sorted(name for name in read_only if moved[name])
    return still, disturbed, int(sum(moved.values()))


def run_case(case: Case, leg: str, steps: int,
             functions: Optional[Mapping[Tuple[str, int], Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE SEAM STEP."""
    seed = case_seed(case.label, leg)
    _, reference, reference_pml = build(case, seed)
    _, actual, actual_pml = build(case, seed)
    drift = compare(reference, actual)
    assert not drift, f"{case.label}: the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_complex_fused_ade_chain(
        actual, actual_pml, residency, functions=functions)
    if plan is None:
        reasons = family.complex_fused_ade_chain_coverage(
            actual, actual_pml, residency).reasons
        return {"passed": False, "reason": "the fused chain was refused",
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
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.alias_checks == plan.launches)
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
        "reference_negative_zero_words": negative_zeros(reference),
        "moved_words": moved_words,
        "runs": plan.runs, "launches": plan.launches,
        "alias_checks": plan.alias_checks,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "replaces": list(plan.replaces_sub_steps),
        "expansion": plan.expansion,
        "poles": [entry.pole_count for entry in plan.entries],
        "volume_sigmas": [sum(1 for flag in entry.sigma_is_volume if flag)
                          for entry in plan.entries],
        "bindings": [family.binding_count(
            entry.pole_count, sum(1 for flag in entry.sigma_is_volume if flag))
            for entry in plan.entries],
        "mirrors": len(residency.names),
        "shape": list(plan.shape), "seed": seed,
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified complex products beside the fused one, on identical state.

    THIS IS THE LEG THAT RE-ESTABLISHES THE INHERITED LICENCE. The orbit walk in
    ``probe_metal_ade_rotation_seam`` skipped complex storage by name, so the claim
    that a per-component ``update_E(c)`` -> ``update_P(*, c)`` interleave is the
    DRIVER's order (all of ``update_E``, then all of ``update_P``) is not inherited
    here — it is measured. Three engines from one seed: the array path; the two
    ALREADY CERTIFIED complex Metal products (``complex_no_pml_stored_e`` for
    ``update_E`` and ``ade_update_p`` for ``update_P``) dispatching the seam as
    separate launches IN DRIVER ORDER; and the fused product. All three must agree
    word for word at every complete seam step.

    The separate side is not a strawman: it is exactly what ``plan_step`` composes
    for this configuration today, which is why disagreement here would be a defect
    in the fused product rather than in an invented comparator.
    """
    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(case, seed)
    _, separate, separate_pml = build(case, seed)
    _, fused, fused_pml = build(case, seed)

    separate_residency = Residency()
    electric = complex_no_pml_stored_e.plan_metal_complex_stored_e(
        separate, separate_pml, separate_residency)
    polarization = ade_update_p.plan_metal_ade_update_p(
        separate, separate_pml, separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_complex_fused_ade_chain(
        fused, fused_pml, fused_residency)
    if electric is None or polarization is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "electric": electric is not None,
                "polarization": polarization is not None,
                "fused": plan is not None}

    separate_residency.sync_in()
    fused_residency.sync_in()
    before = frozen(fused)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        electric.run()          # ALL of update_E ...
        polarization.run()      # ... THEN all of update_P: the driver's order.
        plan.run()              # the per-component interleave.
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
        "passed": bool(identical and plan.launches and electric.launches
                       and polarization.launches and not still and not disturbed),
        "the_claim": ("the per-component interleave is the driver's order on "
                      "COMPLEX storage; probe_metal_ade_rotation_seam.py:579-584 "
                      "skipped this cell, so it is measured here"),
        "per_step_tail": per_step[-3:],
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (electric.launches_per_run
                                         + polarization.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": electric.launches + polarization.launches,
        "fused_launches": plan.launches,
        "dispatches_removed_per_step": (electric.launches_per_run
                                        + polarization.launches_per_run
                                        - plan.launches_per_run),
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Static legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """The worst in-ceiling signature COMPILES; the first one over it FAILS.

    This is what makes the packed ``Params&`` a measurement rather than a
    preference. The corpus row carries ONE pole with a scalar sigma (7 bindings);
    the rows here walk out to the ceiling and one step past it.
    """
    rows: List[Dict[str, Any]] = []
    for poles, volumes in ((1, 0), (1, 1), (6, 6), (7, 7), (8, 8), (8, 0), (9, 0)):
        bindings = family.binding_count(poles, volumes)
        flags = tuple(index < volumes for index in range(poles))
        expected_fit = bindings <= MAX_BUFFER_BINDINGS
        try:
            compile_source(family.complex_fused_ade_chain_source(
                poles, flags, family.ade_expansion_arm()))
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
            compiled, error = False, str(exc).splitlines()[0]
        rows.append({"poles": poles, "volume_sigmas": volumes,
                     "bindings": bindings, "ceiling": MAX_BUFFER_BINDINGS,
                     "expected_to_fit": expected_fit, "compiled": compiled,
                     "error": error, "agrees": compiled == expected_fit})
        log(f"    ceiling poles={poles} volumes={volumes} bindings={bindings} "
            f"fits={expected_fit} compiled={compiled}")
    over = [row for row in rows if not row["expected_to_fit"]]
    refused_by_name = all("bindings" in row["error"] and "device.py:72" in row["error"]
                          for row in over)
    return {"passed": bool(all(row["agrees"] for row in rows) and over
                           and refused_by_name),
            "rows": rows, "over_ceiling_rows": len(over),
            "refused_by_name": refused_by_name,
            "corpus_configuration_bindings": family.binding_count(CORPUS_POLES, 0)}


def leg_single_guard() -> Dict[str, Any]:
    """The emitted kernel has EXACTLY ONE branch, so no mutation can be dead.

    THE DEAD-BRANCH MUTATION is a defect class this project has already paid for:
    an edit can rewrite real lines the scored grid never reaches, because they sit
    under a guard that case does not enter, and then report UNCAUGHT while
    measuring nothing. This family cannot have one — its whole body is straight
    line after the ``n_elem`` guard — and that is asserted from the emitted TEXT
    for every shipped specialisation rather than argued from the source layout.

    The complex helper block carries no branch either, which is what lets the
    count be exactly one rather than "one plus whatever the helpers add"; the
    helper text is measured separately here rather than assumed.

    WHICH HELPERS THE BODY CALLS IS ALSO MEASURED, and it is what stops the OTHER
    half of the dead-code trap. The helper block is emitted VERBATIM from
    :func:`.templates.complex_helpers` — the certified emitters' own text, all
    three orientations — but this kernel invokes only two of them: there is no
    complex-by-complex multiply anywhere in an E->P chain, so ``c_mul``'s body is
    emitted and never reached. An arm mutation planted there would report UNCAUGHT
    while measuring nothing, so :data:`LATTICE_SCORED` arms only the called ones
    and this row records the call census that licenses that choice.
    """
    arm = family.ade_expansion_arm()
    helpers = templates.complex_helpers(arm)
    rows: List[Dict[str, Any]] = []
    for poles in (0, 1, 3, 6):
        flags = tuple((index % 2) == 0 for index in range(poles))
        source = family.complex_fused_ade_chain_source(poles, flags, arm)
        branches = len(re.findall(r"\bif\s*\(", source))
        ternaries = source.count("?")
        loops = len(re.findall(r"\b(for|while)\s*\(", source))
        rows.append({"poles": poles, "if_branches": branches,
                     "ternaries": ternaries, "loops": loops,
                     "straight_line": branches == 1 and not ternaries and not loops})
    helper_branchless = (not re.findall(r"\bif\s*\(", helpers)
                         and "?" not in helpers
                         and not re.findall(r"\b(for|while)\s*\(", helpers))

    # The call census, taken on the BODY (everything after the helper block) so a
    # helper's own definition does not count as a call to itself.
    source = family.complex_fused_ade_chain_source(2, (True, False), arm)
    body = source.split("kernel void", 1)[1]
    called = {name: (f"{name}(" in body) for name in
              ("c_mul", "c_mul_field_left", "c_mul_coefficient_left")}
    armed_helpers = {"c_mul_field_left", "c_mul_coefficient_left"}
    census_agrees = (not called["c_mul"] and called["c_mul_field_left"]
                     and called["c_mul_coefficient_left"])
    return {"passed": bool(all(row["straight_line"] for row in rows)
                           and helper_branchless and census_agrees),
            "rows": rows, "helper_block_is_branchless": helper_branchless,
            "helpers_the_body_calls": called,
            "helpers_arm_mutations_are_planted_in": sorted(armed_helpers),
            "call_census_agrees_with_the_arming": census_agrees,
            "expansion_arm": arm,
            "guard": "if (idx >= n_elem) { return; }  (templates.GUARD)"}


def leg_transcription() -> Dict[str, Any]:
    """The arithmetic lines equal the CERTIFIED emitters' own, PARSED not mirrored.

    THE MIRRORED EVALUATOR is the second defect class this project has paid for:
    a test that re-implements a kernel's assembly mirrors a defect instead of
    executing it. So nothing here re-derives a line. Each certified emitter is
    CALLED, its output is searched for the line, this family's output is searched
    for the same line modulo the documented pointer renames, and the two are
    required equal.
    """
    arm = family.ade_expansion_arm()
    fused = family.complex_fused_ade_chain_source(2, (True, True), arm)
    rows: List[Dict[str, Any]] = []

    # --- the ADE recurrence, against ade_update_p.ade_source's complex64 body --
    certified = ade_update_p.ade_source("complex64", True)
    for wanted, emitted, rename in (
            ("    float2 a = c_mul_field_left(p, c_now);",
             "        float2 a = c_mul_field_left(p, c_now);", "(indent only)"),
            ("    float2 b = c_mul_coefficient_left(c_prev, q);",
             "        float2 b = c_mul_coefficient_left(c_prev, q);", "(indent only)"),
            ("    float2 sw = c_mul_coefficient_left(s, w);",
             "        float2 sw = c_mul_coefficient_left(s, w);", "(indent only)"),
            ("    float2 d = c_mul_coefficient_left(c_drive, sw);",
             "        float2 d = c_mul_coefficient_left(c_drive, sw);", "(indent only)"),
            ("    p_out[idx] = (a + b) + d;",
             "        o0[idx] = (a + b) + d;", "p_out -> o0"),
            ("    float2 p = p_now[idx];",
             "        float2 p = p0[idx];", "p_now -> p0 (the SHARED pointer)"),
            ("    float2 q = p_prev[idx];",
             "        float2 q = q0[idx];", "p_prev -> q0")):
        rows.append({
            "line": wanted.strip(),
            "certified_source": "ade_update_p.ade_source('complex64', True)",
            "present_in_certified": wanted in certified,
            "present_in_fused": emitted in fused,
            "rename": rename,
            "agrees": wanted in certified and emitted in fused})

    # --- the sigma read, on the VOLUME arm ------------------------------------
    rows.append({
        "line": "float s = sigma[idx];",
        "certified_source": "ade_update_p.ade_source('complex64', True)",
        "present_in_certified": "    float s = sigma[idx];" in certified,
        "present_in_fused": "        float s = s0[idx];" in fused,
        "rename": "sigma -> s0",
        "agrees": ("    float s = sigma[idx];" in certified
                   and "        float s = s0[idx];" in fused)})

    # --- the pole chain and the constitutive product, against the E emitter ----
    certified_e = complex_no_pml_stored_e.complex_stored_e_source(2, arm)
    for wanted in ("    float2 source = d_in[idx];",
                   "    source = source - p0[idx];",
                   "    source = source - p1[idx];"):
        rows.append({
            "line": wanted.strip(),
            "certified_source": f"complex_no_pml_stored_e.complex_stored_e_source(2, {arm!r})",
            "present_in_certified": wanted in certified_e,
            "present_in_fused": wanted in fused,
            "rename": "(none)",
            "agrees": wanted in certified_e and wanted in fused})

    # --- THE ONE LINE THIS FAMILY REWRITES, stated rather than hidden ---------
    split_source = "    e_out[idx] = c_mul_field_left(source, inv_e[idx]);"
    rows.append({
        "line": "constitutive_split",
        "certified_source": f"complex_no_pml_stored_e.complex_stored_e_source(2, {arm!r})",
        "present_in_certified": split_source in certified_e,
        "present_in_fused": ("    float2 e = c_mul_field_left(source, inv_e[idx]);"
                             in fused and "    e_out[idx] = e;" in fused),
        "rename": ("e_out[idx] = c_mul_field_left(source, inv_e[idx])  ->  "
                   "float2 e = ...; e_out[idx] = e;"),
        "agrees": (split_source in certified_e
                   and "    float2 e = c_mul_field_left(source, inv_e[idx]);" in fused
                   and "    e_out[idx] = e;" in fused
                   # and the rewrite is exactly the split, character for character
                   and split_source.replace("e_out[idx] = ", "float2 e = ")
                   == "    float2 e = c_mul_field_left(source, inv_e[idx]);")})

    # --- the seam substitution, against the ADE half's own drive read ----------
    rows.append({
        "line": "seam_drive_is_the_register",
        "certified_source": "ade_update_p.ade_source('complex64', True)",
        "present_in_certified": "    float2 w = drive[idx];" in certified,
        "present_in_fused": "        float2 w = e;" in fused,
        "rename": "drive[idx] -> e (the register update_E just wrote)",
        "agrees": ("    float2 w = drive[idx];" in certified
                   and "        float2 w = e;" in fused)})

    # --- the helper block, byte for byte with the certified ADE half's --------
    rows.append({
        "line": "complex_helper_block",
        "certified_source": "ade_update_p.ade_source('complex64', True)",
        "present_in_certified": templates.complex_helpers(arm) in certified,
        "present_in_fused": templates.complex_helpers(arm) in fused,
        "rename": "(none)",
        "agrees": (templates.complex_helpers(arm) in certified
                   and templates.complex_helpers(arm) in fused)})
    return {"passed": all(row["agrees"] for row in rows), "rows": rows}


def leg_arm_binding() -> Dict[str, Any]:
    """The probe-licensed complex arm and the ADE half's baked arm must AGREE.

    ONE FUSED SOURCE CARRIES ONE SET OF HELPERS under one set of names, so a
    platform whose probe licensed an arm the certified ADE half does not bake in
    is unspellable here. On this host they agree — and this leg makes that a
    CHECKED PRECONDITION rather than a coincidence the product silently rests on:

    * the arm is read from the MEASURED artifact this gate exported, not an inline
      literal, and the artifact's own path and digest are recorded;
    * the ADE half's arm is PARSED out of its emitted source, so the answer moves
      if ade_update_p.py:105 does;
    * a synthesized DISAGREEING probe must be REFUSED BY NAME. Without that row
      the clause is a comment.
    """
    from meep_gpu.metal_kernels import templates as _templates  # noqa: PLC0415

    artifact = load_expansion_probe()
    baked = family.ade_expansion_arm()
    licensed = None
    if isinstance(artifact, dict):
        from meep_gpu.metal_kernels.complex_fields import (  # noqa: PLC0415
            expansion_from_probe,
        )
        licensed = expansion_from_probe(artifact)

    other = next(arm for arm in _templates.EXPANSION_ARMS if arm != baked)
    disagreeing = {
        "backend": "numpy",
        "measured": "SYNTHESIZED FOR THIS LEG — not a measurement of any host",
        "patterns": {name: other for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left")},
    }
    case = _case("corpus_one_pole_scalar_sigma")
    _, fields, pml = build(case, case_seed(case.label, "arm_binding"))
    residency = Residency()
    verdict = family.complex_fused_ade_chain_coverage(
        fields, pml, residency, probe=disagreeing)
    plan = family.plan_metal_complex_fused_ade_chain(
        fields, pml, residency, probe=disagreeing)
    named = any("cannot carry two arms" in reason for reason in verdict.reasons)

    admitted = family.complex_fused_ade_chain_coverage(
        fields, pml, Residency(), probe=artifact)
    return {
        "passed": bool(licensed is not None and licensed == baked
                       and not verdict.covered and plan is None and named
                       and admitted.covered),
        "probe_artifact": str(PROBE_ARTIFACT),
        "probe_artifact_sha256": (
            hashlib.sha256(PROBE_ARTIFACT.read_bytes()).hexdigest()
            if PROBE_ARTIFACT.exists() else None),
        "arm_the_probe_licenses": licensed,
        "arm_the_certified_ADE_half_bakes_in": baked,
        "they_agree": licensed == baked,
        "measured_artifact_admits": admitted.covered,
        "disagreeing_probe_arm": other,
        "disagreeing_probe_covered": verdict.covered,
        "disagreeing_probe_planned": plan is not None,
        "disagreeing_probe_refused_by_name": named,
        "refusal_reasons": list(verdict.reasons)[:4],
    }


def leg_policy() -> Dict[str, Any]:
    """``keep`` must be REFUSED BY NAME; ``flush`` must admit.

    MPS flush is native and has NO LEVER (metal_kernels/subnormal.py:6-13), so
    "both policies" cannot mean two comparisons here. It means: the policy this
    executor cannot honour is a refusal at PLAN TIME with the reason named, not a
    silent downgrade that would make every byte claim above rest on an
    unadvertised flush.
    """
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    case = _case(MUTATION_CASE)
    rows: Dict[str, Any] = {}
    original = os.environ.get(subnormal_policy.POLICY_ENV)
    try:
        for policy in (subnormal_policy.FLUSH, subnormal_policy.KEEP):
            os.environ[subnormal_policy.POLICY_ENV] = policy
            _, fields, pml = build(case, case_seed(case.label, f"policy:{policy}"))
            residency = Residency()
            verdict = family.complex_fused_ade_chain_coverage(fields, pml, residency)
            plan = family.plan_metal_complex_fused_ade_chain(fields, pml, residency)
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
        "rows": rows,
        "keep_refused_by_name": keep_named,
        "note": ("MPS flush is native and uncontrollable; `keep` is a refusal, "
                 "not a second byte comparison (metal_kernels/subnormal.py:6-13)"),
    }


def leg_sign_class_null(shape: Sequence[Tuple[int, int, Tuple[bool, ...]]],
                        arm: str, steps: int = 4) -> Dict[str, Any]:
    """PROVE the null: the folded coefficient-left spelling is unobservable HERE.

    A mutation reported "null" on one seeded case is a shrug. This leg enumerates
    the COMPLETE product of signed-zero operand classes — D, P and P_prev each
    cycling all four ``(+-0, +-0)`` patterns independently, so every one of the 64
    combinations occurs on the grid — at three pole counts, over four complete seam
    steps, and requires ZERO differing words between the shipped and folded
    spellings.

    THE NULL HAS ITS OWN NON-VACUITY FLOOR. A class product that did not actually
    contain all four sign patterns in each of the three roles would prove nothing,
    so the patterns present are counted and all four required in each role. And the
    same enumeration is run against a mutation that IS observable
    (``complex_field_left_zero_cross_term_folded``), which must DIVERGE — a
    harness that reported zero for everything would "prove" every null.
    """
    folded_coefficient = (
        "    return float2(fma(c, z.x, -(0.0f * z.y)),\n"
        "                  fma(c, z.y,  (0.0f * z.x)));",
        "    return float2(c * z.x, c * z.y);")
    folded_field = (
        "    return float2(fma(z.x, c,    -(z.y * 0.0f)),\n"
        "                  fma(z.x, 0.0f,  (z.y * c)));",
        "    return float2(z.x * c, z.y * c);")

    def walk(case: Case, edit: Tuple[str, str]) -> int:
        seed = case_seed(case.label, "sign_class_null")
        _, shipped, shipped_pml = build(case, seed)
        _, mutated, mutated_pml = build(case, seed)
        assert not compare(shipped, mutated)
        shipped_residency, mutated_residency = Residency(), Residency()
        base = family.plan_metal_complex_fused_ade_chain(
            shipped, shipped_pml, shipped_residency)
        assert base is not None
        functions: Dict[Tuple[str, int], Any] = {}
        for entry in base.entries:
            source = family.complex_fused_ade_chain_source(
                entry.pole_count, entry.sigma_is_volume, arm)
            replaced = source.replace(edit[0], edit[1], 1)
            assert replaced != source, "the enumerated edit matched nothing"
            functions[(shaders.CONTRACT_OFF, entry.axis)] = compile_source(
                replaced).complex_fused_ade_chain_component
        other = family.plan_metal_complex_fused_ade_chain(
            mutated, mutated_pml, mutated_residency, functions=functions)
        assert other is not None
        shipped_residency.sync_in()
        mutated_residency.sync_in()
        total = 0
        for _ in range(steps):
            base.run()
            other.run()
            shipped_residency.sync_out()
            mutated_residency.sync_out()
            total += sum(compare(shipped, mutated).values())
        return total

    rows: List[Dict[str, Any]] = []
    for poles in (1, 2, 6):
        case = Case(f"sign_class_{poles}_pole", poles, LORENTZIAN, True,
                    value_class="sign_class_product")
        null_words = walk(case, folded_coefficient)
        control_words = walk(case, folded_field)
        rows.append({"poles": poles,
                     "coefficient_left_fold_differing_words": null_words,
                     "field_left_fold_differing_words": control_words,
                     "null_holds": null_words == 0,
                     "control_diverged": control_words > 0})
        log(f"    sign_class poles={poles} coefficient_fold={null_words} "
            f"field_fold(control)={control_words}")

    # The non-vacuity floor on the class product itself.
    case = Case("sign_class_floor", 2, LORENTZIAN, True,
                value_class="sign_class_product")
    _, fields, _pml = build(case, case_seed(case.label, "sign_class_null"))

    def classes_present(array: Any) -> int:
        flat = words(array)
        pairs = set(zip(flat[0::2].tolist(), flat[1::2].tolist()))
        wanted = {(np.float32(r).view(np.uint32).item(),
                   np.float32(i).view(np.uint32).item())
                  for r, i in SIGN_CLASSES}
        return len(wanted & pairs)

    present = {
        "D": classes_present(fields.Dx),
        "P": classes_present(fields.polarizations[0].P["Ex"]),
        "P_prev": classes_present(fields.polarizations[0].P_prev["Ex"]),
    }
    complete = all(count == len(SIGN_CLASSES) for count in present.values())
    return {"passed": bool(complete and all(row["null_holds"] for row in rows)
                           and all(row["control_diverged"] for row in rows)),
            "rows": rows,
            "sign_classes_present_per_role": present,
            "class_product_is_complete": complete,
            "combinations_enumerated": len(SIGN_CLASSES) ** 3,
            "steps": steps}


#: Scale, and what the per-step census MUST do at it: ``True`` stay clean,
#: ``False`` fire. The clean rows are what make the firing rows mean something —
#: a detector that fired on everything would refuse the physical band too and
#: certify nothing. The banded rows are REFUSED BY NAME, never compared.
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

def _shipped_functions(shape: Sequence[Tuple[int, int, Tuple[bool, ...]]],
                       arm: str) -> Dict[Tuple[str, int], Any]:
    return {(shaders.CONTRACT_OFF, axis): family.compile_complex_fused_ade_chain(
        poles, flags, arm) for axis, poles, flags in shape}


def _mutate(base: str, shape: Sequence[Tuple[int, int, Tuple[bool, ...]]],
            arm: str, source: str) -> Dict[Tuple[str, int], Any]:
    table = _shipped_functions(shape, arm)
    table[(shaders.CONTRACT_OFF, 0)] = compile_source(
        source).complex_fused_ade_chain_component
    return table


#: A mutation whose defect is only visible on a value class the uniform band does
#: not contain is scored on THAT class, and the choice is recorded per mutation.
#: The three complex-arm edits below are exactly that case and the reason is
#: MEASURED, not guessed: the certified helper block's own comment records that
#: folding a zero cross term misses 12/128 words and 24/128, and spelling negation
#: ``0.0f - x`` misses 36/512, on EXHAUSTIVE signed-zero tables — while "RANDOM
#: DATA DISCRIMINATES NONE OF THEM (0/16,384 for all three)"
#: (templates.py:42-56, re-measured by gate_metal_complex leg `zero_cross_terms`).
#: Scoring them on the uniform band and reporting UNCAUGHT would be a hollow row:
#: the line IS reached, the inputs simply cannot tell the two spellings apart.
LATTICE_SCORED: Tuple[str, ...] = (
    "complex_field_left_zero_cross_term_folded",
    "complex_negation_spelled_zero_minus_x",
)

#: A mutation this seam's OPERANDS CANNOT EXPRESS, scored NULL — and the null is a
#: MEASUREMENT, not a shrug. ``c_mul_coefficient_left`` is called three times in
#: the ADE arm and its folded spelling differs from the certified one in exactly
#: one place: a lone ``-0.0`` where the certified form writes ``+0.0``. Every one
#: of those three results then flows into the certified body's own
#: ``o[idx] = (a + b) + d`` — and IEEE addition of a lone negative zero to anything
#: yields the other operand's sign, so the difference is absorbed before it can be
#: stored. Leg ``sign_class_null`` proves that over the COMPLETE product of
#: signed-zero operand classes (D x P x P_prev = 64 combinations, at 1, 2 and 6
#: poles, over four complete seam steps) rather than over a sampled lattice.
#:
#: THIS IS NOT A HOLE IN THE ARM'S CERTIFICATION. The spelling is certified where
#: it IS observable — ``gate_metal_complex``'s exhaustive ``zero_cross_terms``
#: tables, on the helper itself. What is recorded here is narrower and true: no
#: input this fused seam can be given distinguishes the two spellings through it.
NULL_SCORED: Dict[str, str] = {
    "complex_coefficient_left_zero_cross_term_folded":
        "every difference it can produce is a lone -0.0 that the certified "
        "`(a + b) + d` addition absorbs before the store (ade_update_p.py:114)",
}


def shader_mutations(shape: Sequence[Tuple[int, int, Tuple[bool, ...]]], arm: str
                     ) -> Dict[str, Dict[Tuple[str, int], Any]]:
    """Source defects, each a plausible transcription slip.

    ARMED ON THE X COMPONENT ONLY for the per-pole edits, with the other two axes
    keeping the shipped bytes: that is what makes a caught mutation evidence about
    the MUTATED LINE rather than about the harness. (The three helper-block edits
    are necessarily whole-kernel: the helpers are file scope.) Each edit is
    required to CHANGE the source — a no-op edit would report the shipped kernel as
    an uncaught defect, which is the name-drift trap.

    EVERY EDIT SITS ON A LINE THE SCORED CASE EXECUTES. There is exactly one
    branch in this kernel (leg ``single_guard`` measures that), so no armed line
    can sit under a guard the scored grid does not enter. Whether the scored
    VALUES can discriminate it is a separate question, answered per mutation by
    :data:`LATTICE_SCORED`.
    """
    axis, poles, flags = next(row for row in shape if row[0] == 0)
    base = family.complex_fused_ade_chain_source(poles, flags, arm)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: feed update_P the pre-inverse-epsilon displacement.
        "seam_takes_the_pre_constitutive_source":
            base.replace("float2 w = e;", "float2 w = source;"),
        # The complex association. ade_update_p's parens are the complex answer.
        "ade_parens_flattened":
            base.replace("        o0[idx] = (a + b) + d;",
                         "        o0[idx] = a + (b + d);", 1),
        "ade_p_and_q_swapped":
            base.replace("        float2 p = p0[idx];\n        float2 q = q0[idx];",
                         "        float2 p = q0[idx];\n        float2 q = p0[idx];", 1),
        "ade_c_now_and_c_prev_swapped":
            base.replace("        float c_now = prm.c_now[0];\n"
                         "        float c_prev = prm.c_prev[0];",
                         "        float c_now = prm.c_prev[0];\n"
                         "        float c_prev = prm.c_now[0];", 1),
        "ade_arm_one_never_stores":
            base.replace("        o1[idx] = (a + b) + d;\n",
                         "        // arm 1 never stores\n", 1),
        "second_pole_dropped_from_the_e_chain":
            base.replace("    source = source - p1[idx];\n", "", 1),
        "sigma_index_off_by_one":
            base.replace("        float s = s1[idx];", "        float s = s0[idx];", 1),
        "e_store_dropped":
            base.replace("    e_out[idx] = e;", "    // stale E", 1),
        # --- THE COMPLEX ARM, three refuted spellings the certified helpers name
        #     as measured misses. Rewriting the helper block reaches every
        #     multiply in the body, so these are the arithmetic's own mutations.
        #     EACH ONE IS ARMED ON A HELPER THIS KERNEL ACTUALLY CALLS — see
        #     leg_single_guard's `helpers_the_body_calls`: `c_mul` is emitted (the
        #     helper block is the certified emitters' own text) but NEVER INVOKED
        #     here, so an edit to its body would be a dead-code mutation reporting
        #     UNCAUGHT while measuring nothing.
        "complex_field_left_zero_cross_term_folded":
            base.replace("    return float2(fma(z.x, c,    -(z.y * 0.0f)),\n"
                         "                  fma(z.x, 0.0f,  (z.y * c)));",
                         "    return float2(z.x * c, z.y * c);", 1),
        "complex_negation_spelled_zero_minus_x":
            base.replace("fma(z.x, c,    -(z.y * 0.0f))",
                         "fma(z.x, c,    0.0f - (z.y * 0.0f))", 1),
        "complex_coefficient_left_zero_cross_term_folded":
            base.replace("    return float2(fma(c, z.x, -(0.0f * z.y)),\n"
                         "                  fma(c, z.y,  (0.0f * z.x)));",
                         "    return float2(c * z.x, c * z.y);", 1),
        # The real/imaginary planes crossed in the constitutive product.
        "constitutive_planes_swapped":
            base.replace("    float2 e = c_mul_field_left(source, inv_e[idx]);",
                         "    float2 e = c_mul_field_left(source, inv_e[idx]).yx;", 1),
    }
    mutants: Dict[str, Dict[Tuple[str, int], Any]] = {}
    for name, source in edits.items():
        if source == base:
            raise AssertionError(
                f"mutation {name!r} did not change the source; a no-op edit would "
                f"report the shipped kernel as an uncaught defect")
        mutants[name] = _mutate(base, shape, arm, source)
    return mutants


def byte_neutral_control(shape: Sequence[Tuple[int, int, Tuple[bool, ...]]], arm: str
                         ) -> Dict[Tuple[str, int], Any]:
    """THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT, and it is scored on that.

    The whole claim of this fusion is that it removes a round trip and changes no
    arithmetic: the ADE half reads a REGISTER where the certified composition read
    the word ``update_E`` had just stored. This edit puts the reload back — ``float2
    w = e_out[idx];`` — and the result must NOT diverge. A float32 pair stored to a
    ``device float2*`` and reloaded is exact, so "byte-neutral by construction" is
    a hypothesis; this row is the comparator agreeing with it.
    """
    axis, poles, flags = next(row for row in shape if row[0] == 0)
    base = family.complex_fused_ade_chain_source(poles, flags, arm)
    source = base.replace("float2 w = e;", "float2 w = e_out[idx];")
    assert source != base, (
        "the byte-neutral control edit matched nothing; a control that changes "
        "nothing would report the shipped bytes as byte-neutral against themselves")
    return _mutate(base, shape, arm, source)


def swap_sigma_between_components(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind Ey's sigma volumes to Ex's arms.

    The kernel takes a pointer per pole and never asks which component's sigma it
    is. A per-component sigma is exactly what an anisotropic material grid gives,
    so this is a silent wrong coupling rather than a crash, and no shader edit can
    reach it.
    """
    entries = list(plan.entries)
    x = next(index for index, e in enumerate(entries) if e.component == "Ex")
    y = next(index for index, e in enumerate(entries) if e.component == "Ey")
    assert entries[x].sigmas and entries[y].sigmas, (
        "the scored case carries no VOLUME sigma, so this host defect is unarmed")
    assert entries[x].sigmas[0] is not entries[y].sigmas[0], (
        "the two components hold the SAME sigma array, so this mutation is a null")
    entries[x] = dataclasses.replace(entries[x], sigmas=entries[y].sigmas)
    plan.entries = tuple(entries)


def swap_params_coefficients(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: pack ``c_prev`` and ``c_drive`` into each other's slots.

    The recurrence constants come from the susceptibility and the timestep
    (dispersion.py:644). Swapping two of them is a plausible packing slip that no
    shader edit can reach — the kernel reads ``prm.c_prev[k]`` either way.
    """
    entries = list(plan.entries)
    for index, entry in enumerate(entries):
        span = max(entry.pole_count, 1)
        blob = np.array(entry.params, copy=True)
        previous = blob[span:2 * span].copy()
        blob[span:2 * span] = blob[2 * span:3 * span]
        blob[2 * span:3 * span] = previous
        name = f"mutation:params:{entry.component}"
        plan._tensors[id(blob)] = residency.mirror(name, blob, constant=True)
        entries[index] = dataclasses.replace(entry, params=blob)
    plan.entries = tuple(entries)


def reverse_pole_order(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: reverse Ex's pole order, mispairing both halves' arms.

    This is the defect the predicate's pole-agreement clause exists to refuse:
    the E chain would subtract the poles in a different order (a different complex
    sum) AND pole k's displacement would be paired with pole j's recurrence.
    """
    entries = list(plan.entries)
    index = next(i for i, e in enumerate(entries) if e.component == "Ex")
    entry = entries[index]
    assert entry.pole_count > 1, "a single-pole entry cannot be reversed"
    entries[index] = dataclasses.replace(entry, states=tuple(reversed(entry.states)))
    plan.entries = tuple(entries)


def swap_inverse_epsilon(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT — expected NULL on this configuration, and CONFIRMED as one.

    ``set_isotropic_epsilon_volume`` hands the SAME array back for all three
    components, so binding Ey's inverse epsilon to Ex's launch binds the identical
    object. Scoring this as "uncaught" would be a false defect; it is scored NULL
    and the identity that makes it one is ASSERTED here rather than assumed.
    """
    entries = list(plan.entries)
    x = next(i for i, e in enumerate(entries) if e.component == "Ex")
    y = next(i for i, e in enumerate(entries) if e.component == "Ey")
    assert entries[x].inverse_epsilon is entries[y].inverse_epsilon, (
        "the two components hold DIFFERENT inverse-epsilon arrays, so this "
        "mutation is live and must not be scored NULL")
    entries[x] = dataclasses.replace(
        entries[x], inverse_epsilon=entries[y].inverse_epsilon)
    plan.entries = tuple(entries)


#: name -> (patch, expected outcome). "caught" must diverge; "null" must NOT, and
#: its patch asserts the identity that makes it a null.
HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Any, Residency], None], str]] = {
    "sigma_bound_for_the_wrong_component": (swap_sigma_between_components, "caught"),
    "params_c_prev_and_c_drive_swapped": (swap_params_coefficients, "caught"),
    "pole_order_reversed_on_Ex": (reverse_pole_order, "caught"),
    "inverse_epsilon_bound_for_the_wrong_component": (swap_inverse_epsilon, "null"),
}


#: THE PLANTED DEFECTS. A mutation leg proves that ONE case diverges when ONE
#: kernel is wrong; it does not prove the gate's RELEASE VERDICT can fail. These
#: edits are applied to the family's SHIPPED emitter for the whole process, on
#: EVERY component, so the complete gate — product, separate control, value
#: classes, disarm — runs against defective bytes and the top-level verdict must
#: flip to FAIL. Run with ``--plant <name>`` and require a nonzero exit.
PLANTED: Dict[str, Tuple[str, str]] = {
    "ade_parens_reassociated_everywhere": (
        "[idx] = (a + b) + d;", "[idx] = a + (b + d);"),
    "seam_reads_the_stale_pole_instead_of_the_drive": (
        "float2 w = e;", "float2 w = p;"),
}


def plant(name: str) -> None:
    """Replace the family's shipped emitter with a defective one, process-wide."""
    wanted, replacement = PLANTED[name]
    original = family.complex_fused_ade_chain_source

    def defective(*arguments: Any, **keywords: Any) -> str:
        source = original(*arguments, **keywords)
        if wanted in source:
            return source.replace(wanted, replacement)
        return source

    family.complex_fused_ade_chain_source = defective  # type: ignore[assignment]
    probe = original(2, (True, True), family.ade_expansion_arm())
    assert wanted in probe, (
        f"planted defect {name!r} matches nothing in the shipped source; a "
        f"marker that matches nothing disables the check it feeds SILENTLY")


def plan_shape_of(case: Case) -> Tuple[Tuple[int, int, Tuple[bool, ...]], ...]:
    """The (axis, poles, sigma flags) triple each component compiles to."""
    _, fields, pml = build(case, case_seed(case.label, "shape"))
    residency = Residency()
    plan = family.plan_metal_complex_fused_ade_chain(fields, pml, residency)
    assert plan is not None, family.complex_fused_ade_chain_coverage(
        fields, pml, residency).reasons
    return tuple((entry.axis, entry.pole_count, entry.sigma_is_volume)
                 for entry in plan.entries)


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

    arm = family.ade_expansion_arm()
    mutation_case = _case(MUTATION_CASE)
    #: The same configuration seeded with the signed-zero lattice, for the three
    #: arm-spelling edits the uniform band provably cannot discriminate.
    lattice_case = dataclasses.replace(
        mutation_case, label=f"{MUTATION_CASE}:pm_zero_lattice",
        value_class="pm_zero_lattice")
    #: And with the COMPLETE signed-zero operand product, for the one edit this
    #: seam's operands provably cannot express (see NULL_SCORED).
    sign_class_case = dataclasses.replace(
        mutation_case, label=f"{MUTATION_CASE}:sign_class_product",
        value_class="sign_class_product")
    # UNDER A PLANTED DEFECT THE MUTATION LEGS ARE MEANINGLESS and are skipped by
    # name rather than run: they would compare defective bytes against defective
    # bytes, and one armed edit IS the planted one, which the name-drift guard
    # correctly refuses as a no-op. What must flip is the verdict of the legs that
    # speak about the SHIPPED bytes — transcription, product, separate control and
    # the value classes — and those all run.
    mutants: Dict[str, Dict[Tuple[str, int], Any]] = {}
    host_mutations = dict(HOST_MUTATIONS)
    neutral: Optional[Dict[Tuple[str, int], Any]] = None
    if args.plant is None:
        shape = plan_shape_of(mutation_case)
        mutants = shader_mutations(shape, arm)
        neutral = byte_neutral_control(shape, arm)
    else:
        host_mutations = {}

    controls = ("corpus_one_pole_scalar_sigma", "three_pole_mixed_volume_sigma",
                "six_pole_volume_sigma_at_the_ceiling")
    value_cases = tuple(
        dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                            value_class="pm_zero_lattice")
        for name in ("corpus_one_pole_scalar_sigma", "three_pole_mixed_volume_sigma"))
    total = (5 + len(CASES) + len(controls) + len(value_cases) + 1
             + len(mutants) + len(host_mutations)
             + (0 if args.plant else 3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for leg, label, runner in (
                ("binding_ceiling", "corpus_row_the_ceiling_and_the_next_one",
                 leg_binding_ceiling),
                ("single_guard", "every_shipped_specialisation", leg_single_guard),
                ("transcription", "arithmetic_lines_vs_certified_emitters",
                 leg_transcription),
                ("arm_binding", "probe_arm_vs_the_ADE_halfs_baked_arm",
                 leg_arm_binding),
                ("policy", "flush_admits_keep_is_refused", leg_policy)):
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

        if args.plant is None:
            index += 1
            rows.append({"index": index, "total": total, "leg": "sign_class_null",
                         "label": "the_complete_signed_zero_operand_product",
                         **leg_sign_class_null(plan_shape_of(mutation_case), arm)})
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

        for name, functions in mutants.items():
            index += 1
            lattice = name in LATTICE_SCORED
            null = name in NULL_SCORED
            scored = (sign_class_case if null
                      else lattice_case if lattice else mutation_case)
            result = run_case(scored, f"mutation:{name}", args.steps,
                              functions=functions)
            caught = not result.get("bit_identical", False)
            expected = "null" if null else "caught"
            outcome = "caught" if caught else "null"
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "caught": caught, "expected": expected,
                   "outcome": outcome,
                   "scored_on_value_class": scored.value_class,
                   "scored_case": scored.label,
                   "passed": bool(outcome == expected and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "launches": result.get("launches")}
            if null:
                row["null_reason"] = NULL_SCORED[name]
                row["null_evidence"] = ("leg sign_class_null, which enumerates the "
                                        "COMPLETE signed-zero operand product")
            if lattice:
                # REPORTED, NOT GATED. The uniform band is expected not to
                # discriminate these three (templates.py:42-56); recording what it
                # actually did keeps that a measurement rather than a citation.
                uniform = run_case(mutation_case, f"mutation:{name}:uniform",
                                   args.steps, functions=functions)
                row["also_scored_on_uniform_band"] = {
                    "caught": not uniform.get("bit_identical", False),
                    "differing_words": uniform.get("differing_words"),
                    "note": ("random data discriminates none of the three arm "
                             "spellings; the signed-zero lattice is the class "
                             "that does (templates.py:42-56)")}
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

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above
        # is a harness that diverges on its own.
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
                   "shader_mutations": len(mutants),
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
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
