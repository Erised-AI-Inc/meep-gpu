#!/usr/bin/env python3
"""Native MPS byte gate for the fused E->P chain: ``update_E`` welded to ``update_P``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes (``update_E`` at driver.py:3304 and ``update_P`` at :3306) with nothing
between them, so a per-sub-step comparison could not see it at all: the whole
claim is about the SEAM. The comparison is therefore per COMPLETE SEAM STEP over a
stated budget, against an array-path oracle running the identical two passes, with
the first divergent step reported rather than a final pass/fail.

THE SIX THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. So every
   case asserts the exact LAUNCH COUNT (``launches_per_run`` x steps), a nonzero
   ``runs``, and a NON-VACUITY FLOOR SPLIT BY ROLE — the arrays this seam WRITES
   (E, f_w_E, every polarization array) must all have moved, and the arrays it
   only READS (B, D, H, fu_*) must NOT have.
2. **A hollow pass.** Source mutations and HOST mutations are armed, and each must
   be CAUGHT — or, where an edit is provably a no-op on the scored configuration,
   NULL CONFIRMED by an identity check rather than reported as uncaught. Leg
   ``disarm`` reruns the identical case with the shipped bytes and requires zero.
3. **A dead-branch mutation.** A mutation that rewrites a line the scored grid
   never reaches measures nothing. This kernel has EXACTLY ONE branch — the
   ``n_elem`` guard — and :func:`leg_single_guard` asserts that from the emitted
   text, so every other emitted line is reached by every in-range thread and no
   mutation can be dead.
4. **A mirrored evaluator.** :func:`leg_transcription` PARSES the three
   load-bearing arithmetic lines out of this family's emitted source AND out of
   the CERTIFIED emitters' own output, and requires them equal modulo the
   documented pointer renames. It does not re-implement either.
5. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor (a zero state is a fixed point of this seam)
   and is instead required to carry BOTH signs of zero in the reference output, so
   a uint32 compare is actually discriminating something; ``subnormal_band`` is
   REFUSED BY NAME and its census required to FIRE, which is what makes the empty
   censuses next door a measurement rather than a silence.
6. **An unmeasured platform assumption.** The signature packs every scalar into one
   ``constant Params&`` to fit the corpus's six-pole rows. Leg ``binding_ceiling``
   compiles the worst IN-ceiling signature and requires success, then compiles the
   first OVER-ceiling one and requires the failure.

BOTH SUBNORMAL POLICIES, AS FAR AS THIS BACKEND HAS THEM. MPS flush is native and
has NO LEVER (metal_kernels/subnormal.py:6-13), so ``keep`` is not a second
comparison but a REFUSAL, and leg ``policy`` requires the plan to be refused BY
NAME under it rather than silently downgraded.

LEGS
  0  binding_ceiling   worst in-ceiling signature COMPILES; the next one FAILS
  1  single_guard      the emitted kernel has exactly one branch
  2  transcription     the arithmetic lines equal the certified emitters', parsed
  3  policy            `keep` is REFUSED by name; `flush` admits
  4  product           complete seam steps, per-step byte compare, counters, floor
  5  separate_control  array path vs the two CERTIFIED products vs fused, 3 ways
  6  value_classes     uniform, +-0 lattice, subnormal band (refused, census fires)
  7  mutation          armed defects, each CAUGHT or NULL CONFIRMED
  8  disarm            the same harness, shipped bytes, must not diverge

Rule 7: one flushed line per case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_metal_fused_ade_chain.py --out <dir>/fused_ade_chain.json
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

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, dispersive_update_e, folded_dispersive_update_e,
    fused_ade_chain as family, no_pml_stored_e, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

#: Complete seam steps every case walks. SIXTY, not twelve: ``f_w_E`` and the
#: whole ADE history are state carried between steps, and the rotation's own
#: period is only ``d + 1``, so the defects this gate exists for COMPOUND. The
#: comparison is per COMPLETE STEP, so a green row means identical at ALL sixty
#: — and step 1 is reported separately, because "identical at one launch" and
#: "identical at sixty" are different claims and both are made here.
STEPS = 60

#: Base for every per-case seed. The seed itself is ``SEED + sha256(label)``,
#: never ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``,
#: so a hash-seeded gate draws a different fixture every process and a failing
#: case cannot be replayed.
SEED = 73_000

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line: demanding movement from all of them fails a correct
#: run, demanding it from none passes a run that launched nothing.
WRITTEN_PREFIXES = ("E", "f_w_E", "P[", "P_prev[", "scratch[")
READ_ONLY_PREFIXES = ("B", "D", "H", "fu_", "f_w_H")


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration, and the arm it drives."""

    label: str
    arm: str                 # "no_pml" | "dispersive"
    poles: int
    kind: str                # LORENTZIAN | DRUDE | "mixed"
    sigma_volume: bool
    folded: bool = False
    cell: Tuple[float, float, float] = (1.2, 1.0, 0.9)
    boundaries: Any = "periodic"
    value_class: str = "uniform"


#: THE CASE MATRIX. The pole counts are the corpus's own, read off
#: ``results/metal_coverage_tranche6_2026-08-19``: the no-PML rows carry 5 and 2
#: (``absorber-1d.py``, ``material-dispersion.py``), the folded dispersive rows 5
#: (``TestLoadDump.*_2d``) and the unfolded dispersive rows SIX
#: (``stochastic_emitter*.py``) — the worst on the board and the one the binding
#: arithmetic is sized for.
CASES: Tuple[Case, ...] = (
    Case("no_pml_five_pole_volume_sigma", "no_pml", 5, "mixed", True),
    Case("no_pml_two_pole_scalar_sigma", "no_pml", 2, LORENTZIAN, False),
    Case("no_pml_one_pole_drude", "no_pml", 1, DRUDE, True),
    Case("pml_six_pole_volume_sigma", "dispersive", 6, "mixed", True),
    Case("pml_five_pole_scalar_sigma", "dispersive", 5, "mixed", False),
    Case("pml_one_pole_lorentz", "dispersive", 1, LORENTZIAN, True),
    Case("folded_pml_five_pole_volume_sigma", "dispersive", 5, "mixed", True,
         folded=True, cell=(1.3, 1.6, 0.9)),
    Case("folded_pml_two_pole_scalar_sigma", "dispersive", 2, DRUDE, False,
         folded=True, cell=(1.3, 1.6, 0.9)),
)

#: The case every mutation is scored on. Six poles with VOLUME sigmas, so every
#: armed line is emitted: a two-pole row would make the "second pole" edits
#: vacuous and a scalar-sigma row would make the ``s1[idx]`` edit unfilled.
MUTATION_CASE = "pml_six_pole_volume_sigma"

#: The second mutation case, for the arm whose seam line is the REWRITTEN one.
#: ``no_pml`` is the only arm where the drive register did not already exist in
#: the certified body, so the seam mutation has to be armed there too.
MUTATION_CASE_NO_PML = "no_pml_five_pole_volume_sigma"


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str, leg: str) -> int:
    """A replayable per-case seed: ``SEED + sha256(label|leg)``, never ``hash()``."""
    digest = hashlib.sha256(f"{label}|{leg}".encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose — and the only
    comparison that can tell ``-0.0`` from ``+0.0``, which the lattice class needs."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


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

def build(case: Case, seed: int, scale: float = 1.0) -> Tuple[Grid, Fields, Optional[PML]]:
    """One seeded engine. Called twice per comparison, identically, for both sides.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES unless a value class says
    otherwise, and that is the first thing this gate establishes rather than an
    afterthought: a zero-initialised state is a FIXED POINT of this seam — with
    D = P = P_prev = 0 the constitutive gives E = 0 and the recurrence gives
    P = 0 forever — so a deliberately wrong kernel compared against a zero oracle
    reports IDENTICAL. That is the exact vacuity half of an earlier gate's cases
    could not fail on.
    """
    symmetry = (Mirror("Y", +1),) if case.folded else ()
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell),
                boundaries=case.boundaries, dimensions=3, courant=0.35,
                k_point=(0.0, 0.0, 0.0), symmetry=symmetry, xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
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
            sigma, grid, np.float32))
    pml: Optional[PML] = None
    if case.arm == "dispersive":
        fields.enable_pml_storage()
        folded_axes = {1} if case.folded else set()
        thickness = []
        for index in range(3):
            if shape[index] < 6:
                thickness.append((0, 0))
            elif index in folded_axes:
                thickness.append((0, 2))
            else:
                thickness.append((2, 2))
        pml = PML(grid=grid, thickness=tuple(thickness))
    else:
        fields.enable_field_storage()

    if case.value_class == "pm_zero_lattice":
        # A lattice of +0.0 and -0.0 over every array the seam consumes, with the
        # inverse epsilon left NORMAL so the sign actually propagates through a
        # multiply. Exempt from the movement floor by construction; the floor it
        # answers to instead is "the reference output carries BOTH signs".
        lattice = np.where(
            ((np.arange(int(np.prod(shape))).reshape(shape) % 2) == 0),
            np.float32(0.0), np.float32(-0.0)).astype(np.float32)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = lattice
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = lattice
                state.P_prev[component][...] = -lattice
        return grid, fields, pml

    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.standard_normal(shape) * 0.37 * scale).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = (
                rng.standard_normal(shape) * 0.21 * scale).astype(np.float32)
            state.P_prev[component][...] = (
                rng.standard_normal(shape) * 0.19 * scale).astype(np.float32)
    return grid, fields, pml


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
    plan = family.plan_metal_fused_ade_chain(
        actual, actual_pml, case.arm, case.folded, residency, functions=functions)
    if plan is None:
        reasons = family.fused_ade_chain_coverage(
            actual, actual_pml, case.arm, case.folded, residency).reasons
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
    if vacuity_exempt:
        lattice_live = bool(negative_zeros(reference) and positive_zeros(reference))
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
        "reference_negative_zero_words": negative_zeros(reference),
        "moved_words": moved_words,
        "runs": plan.runs, "launches": plan.launches,
        "alias_checks": plan.alias_checks,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "replaces": list(plan.replaces_sub_steps),
        "arm": plan.arm, "folded": plan.folded,
        "poles": [entry.pole_count for entry in plan.entries],
        "volume_sigmas": [sum(1 for flag in entry.sigma_is_volume if flag)
                          for entry in plan.entries],
        "bindings": [family.binding_count(
            entry.arm, entry.pole_count,
            sum(1 for flag in entry.sigma_is_volume if flag))
            for entry in plan.entries],
        "mirrors": len(residency.names),
        "shape": list(plan.shape), "seed": seed,
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path; the two ALREADY CERTIFIED Metal
    products (``no_pml_stored_e`` / ``dispersive_update_e`` for ``update_E`` and
    ``ade_update_p`` for ``update_P``) dispatching the seam as separate launches;
    and the fused product. All three must agree word for word at every complete
    seam step, and the DISPATCH COUNTS are recorded on both sides — the only place
    the difference between the compositions shows up at all, since a correct
    fusion is byte-neutral by construction.

    The separate side is not a strawman: it is exactly what ``plan_step`` composes
    for this configuration today, which is why disagreement here would be a defect
    in the fused product rather than in an invented comparator.
    """
    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(case, seed)
    _, separate, separate_pml = build(case, seed)
    _, fused, fused_pml = build(case, seed)

    separate_residency = Residency()
    # THE FOLDED ROW TAKES THE FOLDED FAMILY, not the ordinary one. Both dispatch
    # the SAME binding builder (folded_dispersive_update_e.py:5-8); what differs is
    # the predicate that establishes the stored extent, and the ordinary one
    # refuses a live mirror by name. Picking it here would have made the folded
    # control a refusal rather than a comparison.
    if case.arm == "dispersive" and case.folded:
        electric = folded_dispersive_update_e.plan_folded_dispersive_e(
            separate, separate_pml, separate_residency)
    elif case.arm == "dispersive":
        electric = dispersive_update_e.plan_metal_dispersive_e(
            separate, separate_pml, separate_residency)
    else:
        electric = no_pml_stored_e.plan_metal_stored_e(
            separate, separate_pml, separate_residency)
    polarization = ade_update_p.plan_metal_ade_update_p(
        separate, separate_pml, separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_fused_ade_chain(
        fused, fused_pml, case.arm, case.folded, fused_residency)
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
        electric.run()
        polarization.run()
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
        "passed": bool(identical and plan.launches and electric.launches
                       and polarization.launches and not still and not disturbed),
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
    """The corpus's worst signature COMPILES; the first one over the ceiling FAILS.

    This is what makes the packed ``Params&`` a measurement rather than a
    preference. A host on which the worst corpus row did NOT fit would mean the
    six-pole ``stochastic_emitter`` rows are unreachable and this family's claimed
    coverage is wrong — so the leg fails loudly rather than being skipped.
    """
    rows: List[Dict[str, Any]] = []
    for arm, poles, volumes in (("dispersive", 6, 6), ("dispersive", 6, 0),
                                ("no_pml", 5, 5), ("no_pml", 8, 8)):
        bindings = family.binding_count(arm, poles, volumes)
        flags = tuple(index < volumes for index in range(poles))
        expected_fit = bindings <= MAX_BUFFER_BINDINGS
        try:
            source = family.fused_ade_chain_source(arm, 0, poles, flags)
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
            compiled, error = False, str(exc).splitlines()[0]
        rows.append({"arm": arm, "poles": poles, "volume_sigmas": volumes,
                     "bindings": bindings, "ceiling": MAX_BUFFER_BINDINGS,
                     "expected_to_fit": expected_fit, "compiled": compiled,
                     "error": error, "agrees": compiled == expected_fit})
        log(f"    ceiling {arm} poles={poles} volumes={volumes} "
            f"bindings={bindings} fits={expected_fit} compiled={compiled}")
    # The emitter must REFUSE an over-ceiling signature by name rather than let it
    # reach the compiler, so the predicate and the emitter agree about the limit.
    over = [row for row in rows if not row["expected_to_fit"]]
    refused_by_name = all(
        "bindings" in row["error"] and "device.py:72" in row["error"]
        for row in over)
    return {"passed": bool(all(row["agrees"] for row in rows) and over
                           and refused_by_name),
            "rows": rows, "over_ceiling_rows": len(over),
            "refused_by_name": refused_by_name}


def leg_single_guard() -> Dict[str, Any]:
    """The emitted kernel has EXACTLY ONE branch, so no mutation can be dead.

    THE DEAD-BRANCH MUTATION is a defect class this project has already paid for:
    an edit can rewrite real lines the scored grid never reaches, because they sit
    under a guard that case does not enter, and then report UNCAUGHT while
    measuring nothing. This family cannot have one — its whole body is straight
    line after the ``n_elem`` guard — and that is asserted from the emitted TEXT
    for every shipped specialisation rather than argued from the source layout.
    """
    rows: List[Dict[str, Any]] = []
    for arm, poles in (("no_pml", 0), ("no_pml", 5), ("dispersive", 1),
                       ("dispersive", 6)):
        for axis in (0, 1, 2):
            flags = tuple((index % 2) == 0 for index in range(poles))
            source = family.fused_ade_chain_source(arm, axis, poles, flags)
            branches = len(re.findall(r"\bif\s*\(", source))
            ternaries = source.count("?")
            loops = len(re.findall(r"\b(for|while)\s*\(", source))
            rows.append({"arm": arm, "axis": axis, "poles": poles,
                         "if_branches": branches, "ternaries": ternaries,
                         "loops": loops,
                         "straight_line": branches == 1 and not ternaries and not loops})
    return {"passed": all(row["straight_line"] for row in rows),
            "rows": rows,
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
    rows: List[Dict[str, Any]] = []

    # --- the ADE recurrence, against ade_update_p.ade_source's float32 body ----
    certified = ade_update_p.ade_source("float32", True)
    wanted = "p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));"
    fused = family.fused_ade_chain_source("no_pml", 0, 2, (True, True))
    emitted = "o0[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));"
    rows.append({
        "line": "ade_recurrence",
        "certified_source": "ade_update_p.ade_source('float32', True)",
        "present_in_certified": wanted in certified,
        "present_in_fused": emitted in fused,
        "rename": "p_out -> o0",
        "agrees": (wanted in certified and emitted in fused
                   and wanted.replace("p_out", "o0") == emitted)})

    # --- the pole chain, against no_pml_stored_e.stored_e_source ---------------
    certified = no_pml_stored_e.stored_e_source(2)
    wanted = "source = source - p1[idx];"
    rows.append({
        "line": "pole_subtraction",
        "certified_source": "no_pml_stored_e.stored_e_source(2)",
        "present_in_certified": wanted in certified,
        "present_in_fused": wanted in fused,
        "rename": "(none)",
        "agrees": wanted in certified and wanted in fused})

    # --- the constitutive product, against the same certified body -------------
    # The ONE line this family rewrites, and the rewrite is stated rather than
    # hidden: the certified body stores the product unnamed, the fused body names
    # it so update_P can read a register.
    rows.append({
        "line": "no_pml_constitutive_split",
        "certified_source": "no_pml_stored_e.stored_e_source(2)",
        "present_in_certified": "e_out[idx] = source * inv_e[idx];" in certified,
        "present_in_fused": ("float e = source * inv_e[idx];" in fused
                             and "e_out[idx] = e;" in fused),
        "rename": "e_out[idx] = source * inv_e[idx]  ->  float e = ...; e_out[idx] = e;",
        "agrees": ("e_out[idx] = source * inv_e[idx];" in certified
                   and "float e = source * inv_e[idx];" in fused
                   and "e_out[idx] = e;" in fused)})

    # --- the PML half, against dispersive_update_e.dispersive_e_source ---------
    certified = dispersive_update_e.dispersive_e_source(2, 1)
    fused_pml = family.fused_ade_chain_source("dispersive", 1, 2, (True, True))
    for wanted in ("    float src = source * inv_e[idx];",
                   "    fw[idx] = src;",
                   "    value = value + kps[j] * src;",
                   "    value = value - kms[j] * prev;"):
        rows.append({
            "line": wanted.strip(),
            "certified_source": "dispersive_update_e.dispersive_e_source(2, 1)",
            "present_in_certified": wanted in certified,
            "present_in_fused": wanted in fused_pml,
            "rename": "(none)",
            "agrees": wanted in certified and wanted in fused_pml})

    # --- the seam itself, on the arm where it needed no rewrite at all ---------
    rows.append({
        "line": "dispersive_seam_is_a_pure_addition",
        "certified_source": "dispersive_update_e.dispersive_e_source(2, 1)",
        "present_in_certified": "float src = source * inv_e[idx];" in certified,
        "present_in_fused": "float w = src;" in fused_pml,
        "rename": "drive[idx] -> src (the certified body's own register)",
        "agrees": ("float src = source * inv_e[idx];" in certified
                   and "float w = src;" in fused_pml
                   and "float w = drive[idx];" in ade_update_p.ade_source(
                       "float32", True))})
    return {"passed": all(row["agrees"] for row in rows), "rows": rows}


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
            verdict = family.fused_ade_chain_coverage(
                fields, pml, case.arm, case.folded, residency)
            plan = family.plan_metal_fused_ade_chain(
                fields, pml, case.arm, case.folded, residency)
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

def _shipped_functions(case: Case, plan_shape: Sequence[Tuple[int, int, Tuple[bool, ...]]]
                       ) -> Dict[Tuple[str, int], Any]:
    return {(shaders.CONTRACT_OFF, axis): family.compile_fused_ade_chain(
        case.arm, axis, poles, flags) for axis, poles, flags in plan_shape}


def shader_mutations(case: Case, plan_shape: Sequence[Tuple[int, int, Tuple[bool, ...]]]
                     ) -> Dict[str, Dict[Tuple[str, int], Any]]:
    """Source defects, each a plausible transcription slip.

    ARMED ON THE X COMPONENT ONLY, with the other two axes keeping the shipped
    bytes: that is what makes a caught mutation evidence about the MUTATED LINE
    rather than about the harness. Each edit is required to CHANGE the source —
    a no-op edit would report the shipped kernel as an uncaught defect, which is
    the name-drift trap.
    """
    axis, poles, flags = next(row for row in plan_shape if row[0] == 0)
    base = family.fused_ade_chain_source(case.arm, axis, poles, flags)
    edits: Dict[str, str] = {
        # THE SEAM ITSELF: feed update_P the pre-inverse-epsilon displacement.
        "seam_takes_the_pre_constitutive_source":
            base.replace("float w = src;", "float w = source;")
                .replace("float w = e;", "float w = source;"),
        # The ADE association. ade_update_p's parens are the float32 answer.
        "ade_parens_flattened":
            base.replace(
                "o0[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));",
                "o0[idx] = p * c_now + c_prev * q + c_drive * s * w;", 1),
        "ade_p_and_q_swapped":
            base.replace("        float p = p0[idx];\n        float q = q0[idx];",
                         "        float p = q0[idx];\n        float q = p0[idx];", 1),
        "ade_c_now_and_c_prev_swapped":
            base.replace("        float c_now = prm.c_now[0];\n"
                         "        float c_prev = prm.c_prev[0];",
                         "        float c_now = prm.c_prev[0];\n"
                         "        float c_prev = prm.c_now[0];", 1),
        "ade_arm_one_never_stores":
            base.replace(
                "        o1[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));\n",
                "        // arm 1 never stores\n", 1),
        "second_pole_dropped_from_the_e_chain":
            base.replace("    source = source - p1[idx];\n", "", 1),
        "sigma_index_off_by_one":
            base.replace("        float s = s1[idx];", "        float s = s0[idx];", 1),
    }
    if case.arm == "dispersive":
        edits.update({
            "pml_accumulations_reversed":
                base.replace("    value = value + kps[i] * src;\n"
                             "    value = value - kms[i] * prev;",
                             "    value = value - kms[i] * src;\n"
                             "    value = value + kps[i] * prev;", 1),
            "fw_store_dropped":
                base.replace("    fw[idx] = src;", "    // stale fw", 1),
        })
    else:
        edits["e_store_dropped"] = base.replace(
            "    e_out[idx] = e;", "    // stale E", 1)

    mutants: Dict[str, Dict[Tuple[str, int], Any]] = {}
    for name, source in edits.items():
        if source == base:
            raise AssertionError(
                f"mutation {name!r} did not change the source; a no-op edit would "
                f"report the shipped kernel as an uncaught defect")
        table = _shipped_functions(case, plan_shape)
        table[(shaders.CONTRACT_OFF, 0)] = compile_source(
            source).fused_ade_chain_component
        mutants[name] = table
    return mutants


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
        prev = blob[span:2 * span].copy()
        blob[span:2 * span] = blob[2 * span:3 * span]
        blob[2 * span:3 * span] = prev
        name = f"mutation:params:{entry.component}"
        plan._tensors[id(blob)] = residency.mirror(name, blob, constant=True)
        entries[index] = dataclasses.replace(entry, params=blob)
    plan.entries = tuple(entries)


def reverse_pole_order(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: reverse Ex's pole order, mispairing both halves' arms.

    This is the defect the predicate's pole-agreement clause exists to refuse:
    the E chain would subtract the poles in a different order (a different float32
    sum) AND pole k's displacement would be paired with pole j's recurrence.
    """
    entries = list(plan.entries)
    index = next(i for i, e in enumerate(entries) if e.component == "Ex")
    entry = entries[index]
    entries[index] = dataclasses.replace(entry, states=tuple(reversed(entry.states)))
    plan.entries = tuple(entries)


def swap_inverse_epsilon(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT — expected NULL on this configuration, and CONFIRMED as one.

    ``set_isotropic_epsilon_volume`` hands the SAME array back for all three
    components (device.py:170-173 allows exactly that aliasing), so binding Ey's
    inverse epsilon to Ex's launch binds the identical object. Scoring this as
    "uncaught" would be a false defect; it is scored NULL and the identity that
    makes it one is asserted here rather than assumed.
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
#: EVERY axis and every arm, so the complete gate — product, separate control,
#: value classes, disarm — runs against defective bytes and the top-level verdict
#: must flip to FAIL. Run with ``--plant <name>`` and require a nonzero exit.
PLANTED: Dict[str, Tuple[str, str]] = {
    "ade_parens_flattened_everywhere": (
        "o0[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));",
        "o0[idx] = p * c_now + c_prev * q + c_drive * s * w;"),
    "seam_reads_the_stale_pole_instead_of_the_drive": (
        "float w = src;", "float w = p;"),
}


def plant(name: str) -> None:
    """Replace the family's shipped emitter with a defective one, process-wide."""
    wanted, replacement = PLANTED[name]
    original = family.fused_ade_chain_source

    def defective(*arguments: Any, **keywords: Any) -> str:
        source = original(*arguments, **keywords)
        if wanted in source:
            return source.replace(wanted, replacement)
        return source

    family.fused_ade_chain_source = defective  # type: ignore[assignment]
    probe = original("dispersive", 0, 2, (True, True))
    assert wanted in probe, (
        f"planted defect {name!r} matches nothing in the shipped source; a "
        f"marker that matches nothing disables the check it feeds SILENTLY")


def plan_shape_of(case: Case) -> Tuple[Tuple[int, int, Tuple[bool, ...]], ...]:
    """The (axis, poles, sigma flags) triple each component compiles to."""
    _, fields, pml = build(case, case_seed(case.label, "shape"))
    residency = Residency()
    plan = family.plan_metal_fused_ade_chain(
        fields, pml, case.arm, case.folded, residency)
    assert plan is not None, family.fused_ade_chain_coverage(
        fields, pml, case.arm, case.folded, residency).reasons
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
    if args.plant:
        plant(args.plant)
        log(f"PLANTED DEFECT {args.plant!r}: the verdict below MUST be FAIL")

    mutation_case = _case(MUTATION_CASE)
    mutation_case_no_pml = _case(MUTATION_CASE_NO_PML)
    # UNDER A PLANTED DEFECT THE MUTATION LEGS ARE MEANINGLESS and are skipped by
    # name rather than run: they would compare defective bytes against defective
    # bytes, and several armed edits ARE the planted one, which the name-drift
    # guard correctly refuses as a no-op. What must flip is the verdict of the
    # legs that speak about the SHIPPED bytes — transcription, product, separate
    # control and the value classes — and those all run.
    mutants: Dict[str, Dict[str, Any]] = {}
    host_mutations = dict(HOST_MUTATIONS)
    if args.plant is None:
        shapes = {case.label: plan_shape_of(case)
                  for case in (mutation_case, mutation_case_no_pml)}
        mutants = {
            MUTATION_CASE: shader_mutations(mutation_case, shapes[MUTATION_CASE]),
            MUTATION_CASE_NO_PML: shader_mutations(
                mutation_case_no_pml, shapes[MUTATION_CASE_NO_PML])}
    else:
        host_mutations = {}

    controls = ("no_pml_five_pole_volume_sigma", "pml_six_pole_volume_sigma",
                "folded_pml_five_pole_volume_sigma")
    value_cases = tuple(
        dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                            value_class="pm_zero_lattice")
        for name in ("no_pml_two_pole_scalar_sigma", "pml_five_pole_scalar_sigma"))
    total = (4 + len(CASES) + len(controls) + len(value_cases) + 1
             + sum(len(table) for table in mutants.values())
             + len(host_mutations) + (0 if args.plant else 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for leg, label, runner in (
                ("binding_ceiling", "worst_corpus_signature_and_the_next_one",
                 leg_binding_ceiling),
                ("single_guard", "every_shipped_specialisation", leg_single_guard),
                ("transcription", "arithmetic_lines_vs_certified_emitters",
                 leg_transcription),
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

        for label, table in mutants.items():
            case = _case(label)
            for name, functions in table.items():
                index += 1
                result = run_case(case, f"mutation:{name}", args.steps,
                                  functions=functions)
                caught = not result.get("bit_identical", False)
                rows.append({"index": index, "total": total, "leg": "mutation",
                             "label": f"{label}/{name}", "caught": caught,
                             "expected": "caught",
                             "passed": bool(caught and result.get("launches")),
                             "first_divergence": result.get("first_divergence"),
                             "differing_words": result.get("differing_words"),
                             "launches": result.get("launches")})
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
                   "shader_mutations": sum(len(t) for t in mutants.values()),
                   "host_mutations": len(host_mutations)},
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
