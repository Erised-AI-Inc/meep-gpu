#!/usr/bin/env python3
"""CUDA byte gate for the fused E->P chain: ``update_E`` welded to ``update_P``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes (``update_E`` at driver.py:3304 and ``update_P`` at :3306) with nothing
between them, so a per-sub-step comparison could not see it at all: the whole claim
is about the SEAM. The comparison is therefore per COMPLETE SEAM STEP over a stated
budget, against

  * the CuPy array path (``stepping.update_E`` then ``stepping.update_P``), and
  * the SEPARATELY CERTIFIED Triton products this one replaces — the E half
    (``no_pml_stored_e`` / ``dispersive_update_e``) and the ADE half
    (``no_pml_ade`` / ``launch.plan_ade_update_p``) dispatched as separate launches,

over every allocated volume, compared as uint32 WORDS. ``allclose`` appears
nowhere, and the FIRST divergent step is reported rather than a final pass/fail.

THE SEVEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. Every case
   asserts the exact LAUNCH COUNT through a proxy that owns the kernel object, a
   nonzero ``runs``, one alias check per launch, and a NON-VACUITY FLOOR SPLIT BY
   ROLE — the arrays this seam WRITES (E, f_w_E, every polarization volume) must
   all have moved, and the arrays it only READS (B, D, H, fu_*, the material) must
   NOT have.
2. **A hollow pass.** Source mutations and HOST mutations are armed, and each is
   scored CAUGHT or NULL CONFIRMED with its evidence; a mutant that compiled to
   PTX already present for the shipped kernel is reported PTX-IDENTICAL rather
   than as a passing null. Leg ``disarm`` reruns the identical case with the
   shipped bytes and requires zero.
3. **A dead mutation.** The E chain and the ADE arms are unrolled by ``NP``
   constexprs, so a rewrite of a slot the scored case does not carry is emitted
   nowhere and measures nothing. Every armed mutation declares the SLOT it edits
   and :func:`leg_no_dead_mutations` refuses any whose slot the mutation case does
   not reach.
4. **A mirrored evaluator.** :func:`leg_transcription` PARSES the arithmetic out of
   the certified kernels' own files and out of this one, and requires them equal
   modulo the documented renames. It re-implements nothing.
5. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor (a zero state is a fixed point of this seam)
   and must instead carry BOTH signs of zero in the reference output;
   ``subnormal_band`` is COMPARED rather than refused — Triton has both float32
   policies and CuPy's flush is strippable, so the band is where the two policies
   are visible — and its census must FIRE or the row is VACUOUS, not passed.
6. **An unmeasured shape.** The whole licence for this product is that one scratch
   per driven component makes the launch's write set disjoint from its read set at
   every position of the rotation's orbit. :func:`leg_shape` walks the orbit to
   closure THROUGH THE SHIPPED PLAN with the alias check armed, and two host
   mutations break the chain on purpose and require the plan to refuse.
7. **A verdict that cannot fail.** ``--plant`` rewrites the SHIPPED kernel source
   for the whole process and the release verdict must come back FAIL.

BOTH FLOAT32 SUBNORMAL POLICIES, ONE PER PROCESS. ``--subnormal-policy`` drives
host, CuPy and Triton before the first device compile, and the runner
(``run_triton_fused_ade_chain_direct.sh``) executes this gate twice with distinct
CuPy and Triton cache directories. One process per policy is not fastidiousness:
CuPy computes its kernel cache key ABOVE the seam where the ``-ftz=true`` strip
installs, so a cache shared between the two policies serves flushed binaries under
the keeping record's name.

Progress reporting: one flushed line per case, every row appended and fsynced as it lands, and
the artifact re-written (provenance-stamped) after every leg.

Usage::

    # laptop, no CUDA and no Triton — the legs that need no device
    PYTHONPATH=. python -u \\
        parity/meep_gpu/gate_triton_fused_ade_chain.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/gate_triton_fused_ade_chain.py \\
        --subnormal-policy keep --out <fresh-dir>/gate_keep.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Complete seam steps every product case walks. SIXTY, not one: ``f_w_E`` and the
#: whole ADE history are state carried between steps and the rotation's own period
#: is three, so the defects this gate exists for COMPOUND. The comparison is per
#: COMPLETE STEP, so a green row means identical at ALL sixty — and step 1 is
#: reported separately, because "identical at one launch" and "identical at sixty"
#: are different claims and both are made here.
STEPS = 60

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch and PTX evidence.
MUTATION_STEPS = 3

#: Base for every per-case seed. The seed itself is ``SEED + sha256(label|leg)``,
#: never ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``,
#: so a hash-seeded gate draws a different fixture every process and a failing case
#: cannot be replayed.
SEED = 84_000

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line: demanding movement from all of them fails a correct
#: run, and demanding it from none passes a run that launched nothing.
WRITTEN_PREFIXES = ("E", "f_w_E", "P[", "P_prev[", "scratch[")
READ_ONLY_PREFIXES = ("B", "D", "H", "fu_", "f_w_H", "eps", "inv_eps")

_TEMPORARY: List[str] = []

#: The float32 subnormal policy this process INSTALLED, set once in :func:`main`
#: before the first device compile. Two legs read it, and both would otherwise
#: assert a policy-dependent fact as if it were policy-free: whether the subnormal
#: band can exist at all is decided by the policy, not by the kernel.
_POLICY: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration, and the arm it drives."""

    label: str
    arm: str                 # "no_pml" | "dispersive"
    poles: int
    kind: str                # LORENTZIAN | DRUDE | "mixed"
    sigma_volume: bool
    cell: Tuple[float, float, float] = (1.2, 1.0, 0.9)
    boundaries: Any = "periodic"
    value_class: str = "uniform"
    scale: float = 1.0


#: THE CASE MATRIX. The pole counts are the corpus's own at this seam: the no-PML
#: rows carry 5 and 2 (``absorber-1d.py``, ``TestAbsorber``, ``material-dispersion
#: .py``) and the dispersive rows SIX (``stochastic_emitter*.py``) — the worst on
#: the board and the one the ADE arm count is sized for.
CASES: Tuple[Case, ...] = (
    Case("no_pml_five_pole_volume_sigma", "no_pml", 5, "mixed", True),
    Case("no_pml_two_pole_scalar_sigma", "no_pml", 2, "lorentzian", False),
    Case("no_pml_one_pole_drude", "no_pml", 1, "drude", True),
    Case("pml_six_pole_volume_sigma", "dispersive", 6, "mixed", True),
    Case("pml_five_pole_scalar_sigma", "dispersive", 5, "mixed", False),
    Case("pml_two_pole_metallic", "dispersive", 2, "lorentzian", True,
         boundaries="metallic"),
    Case("pml_one_pole_lorentz", "dispersive", 1, "lorentzian", True),
    # THE FOLDED ROWS. The four corpus rows this arm exists for are the
    # ``TestLoadDump.*_2d`` set at FIVE poles; the two-pole row beside it is what
    # keeps a "second pole" edit meaningful on a folded grid too.
    Case("folded_pml_five_pole_volume_sigma", "folded", 5, "mixed", True,
         cell=(1.3, 1.6, 0.9)),
    Case("folded_pml_two_pole_scalar_sigma", "folded", 2, "drude", False,
         cell=(1.3, 1.6, 0.9)),
)

#: The case every mutation is scored on. Six poles with VOLUME sigmas, so every
#: armed line is emitted on every component: a two-pole row would make the "second
#: pole" edits dead and a scalar-sigma row would leave the ``sigma_a1`` edit
#: unfilled.
MUTATION_CASE = "pml_six_pole_volume_sigma"

#: The second mutation case, for the arm whose seam line is the REWRITTEN one.
#: ``no_pml`` is the only arm where the drive register did not already exist in the
#: certified body, so the seam edits are armed there too.
MUTATION_CASE_NO_PML = "no_pml_five_pole_volume_sigma"


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str, leg: str) -> int:
    """A replayable per-case seed: ``SEED + sha256(label|leg)``, never ``hash()``."""
    digest = hashlib.sha256(f"{label}|{leg}".encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The bytes under certification, curated. See ``gate_provenance``'s docstring
    for why this is kept apart from ``imported_source_sha256``."""
    names = (
        "meep_gpu/driver.py", "meep_gpu/stepping.py", "meep_gpu/dispersion.py",
        "meep_gpu/fields.py", "meep_gpu/subnormal_policy.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/triton_kernels/no_pml_stored_e.py",
        "meep_gpu/triton_kernels/no_pml_ade.py",
        "meep_gpu/triton_kernels/dispersive_update_e.py",
        "meep_gpu/triton_kernels/fused_ade_chain.py",
        "meep_gpu/test_triton_fused_ade_chain.py",
        "parity/meep_gpu/gate_provenance.py",
        "parity/meep_gpu/gate_triton_fused_ade_chain.py",
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically.

    THE STAMP IS HERE AND NOT AT THE CALL SITES. ``main`` calls this after every
    leg, and a stamp bolted onto the last call would leave every partial artifact —
    including the one an aborted run leaves behind, which is the artifact a failure
    is read from — unattributable. The POLICY stamp is RE-READ rather than carried:
    taken once at install time it records every executor counter at zero, because
    nothing had compiled yet.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception as exc:  # noqa: BLE001
            payload["subnormal_policy_reread_error"] = repr(exc)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def build(xp, case: Case, seed: int, scale: Optional[float] = None):
    """One seeded engine. Called once per route, identically, from one seed.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES unless a value class says
    otherwise, and that is the first thing this gate establishes rather than an
    afterthought: a zero-initialised state is a FIXED POINT of this seam — with
    D = P = P_prev = 0 the constitutive gives E = 0 and the recurrence gives P = 0
    forever — so a deliberately wrong kernel compared against a zero oracle reports
    IDENTICAL.
    """
    from meep_gpu.dispersion import (  # noqa: PLC0415
        DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
    )
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    amplitude = case.scale if scale is None else scale
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell),
                boundaries=case.boundaries, dimensions=3, courant=0.35,
                k_point=(0.0, 0.0, 0.0),
                symmetry=(Mirror("Y", +1),) if case.arm == "folded" else (),
                xp=xp)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)
    epsilon = (1.45 + 0.30 * rng.random(shape)).astype(np.float32)
    # ONE ALIASED PAIR for all three components (fields.py:1321-1325), which is
    # what makes the ``inverse_epsilon_bound_for_the_wrong_component`` host
    # mutation a NULL rather than an uncaught defect — asserted there, not assumed.
    fields.set_isotropic_epsilon_volume(
        xp.asarray(np.ascontiguousarray(epsilon)),
        xp.asarray(np.ascontiguousarray((np.float32(1.0) / epsilon).astype(np.float32))))

    kinds = ((DRUDE, LORENTZIAN) if case.kind == "mixed"
             else ((DRUDE,) if case.kind == "drude" else (LORENTZIAN,)))
    for order in range(case.poles):
        base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
                "Ez": 0.19 + 0.06 * order}
        if case.sigma_volume:
            sigma: Any = {
                name: xp.asarray(np.ascontiguousarray(
                    (value * (0.7 + 0.6 * rng.random(shape))).astype(np.float32)))
                for name, value in base.items()}
        else:
            sigma = base
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            sigma, grid, np.float32))

    pml = None
    if case.arm == "no_pml":
        fields.enable_field_storage()
    else:
        fields.enable_pml_storage()
        # A MIRRORED axis takes the HIGH FACE ONLY: cell 0 lies on the mirror
        # plane, which is a boundary condition and not an absorber (pml.py:408
        # refuses the low face by name).
        thickness = tuple(
            (0, 0) if shape[index] < 6
            else ((0, 2) if grid.is_mirrored(index) else (2, 2))
            for index in range(3))
        pml = PML(grid=grid, thickness=thickness)

    if case.value_class == "pm_zero_lattice":
        # A LATTICE of +0.0 and -0.0 over every array the seam consumes, with the
        # inverse epsilon left NORMAL so the sign propagates through a multiply.
        # Exempt from the movement floor by construction; the floor it answers to
        # instead is "the reference output carries BOTH signs".
        flat = np.arange(int(np.prod(shape))).reshape(shape)
        lattice = np.ascontiguousarray(
            np.where((flat % 2) == 0, np.float32(0.0), np.float32(-0.0))
            .astype(np.float32))
        alternate = np.ascontiguousarray(
            np.where(((flat // 2) % 2) == 0, np.float32(-0.0), np.float32(0.0))
            .astype(np.float32))
        for name in STATE_NAMES:
            if getattr(fields, name, None) is not None:
                getattr(fields, name)[...] = xp.asarray(lattice)
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = xp.asarray(lattice)
                state.P_prev[component][...] = xp.asarray(alternate)
        return grid, fields, pml

    for name in STATE_NAMES:
        if getattr(fields, name, None) is not None:
            getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.37 * amplitude).astype(np.float32)))
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.21 * amplitude).astype(np.float32)))
            state.P_prev[component][...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.19 * amplitude).astype(np.float32)))
    return grid, fields, pml


def state_of(fields: Any, plan: Any = None) -> Dict[str, Any]:
    """Every semantically live array, with the ADE roles NAMED rather than aliased.

    ``update_P`` ROTATES which physical allocation holds ``P``, so a static string
    table could not name them without losing the semantic role — which is the one
    thing the comparison is about.

    THE SCRATCH IS THE ONE ASYMMETRY AND IT IS NAMED, NOT DROPPED. The reference
    keeps ONE retired buffer per susceptibility and this product keeps one per
    driven component (see the module's shape note), so only the LAST driven
    component's is comparable: the reference overwrote the others with the next
    component's result. That one IS compared; the rest are dead state the next
    launch overwrites before reading, and pretending otherwise would be comparing
    the harness.
    """
    state = {name: getattr(fields, name) for name in STATE_NAMES
             if getattr(fields, name, None) is not None}
    for index, polarization in enumerate(tuple(fields.polarizations)):
        driven = tuple(polarization.driven())
        for component in driven:
            state[f"P[{index}].{component}"] = polarization.P[component]
            state[f"P_prev[{index}].{component}"] = polarization.P_prev[component]
        if driven:
            state[f"scratch[{index}]"] = (
                plan._scratch[(index, driven[-1])] if plan is not None
                else polarization._scratch)
    for name in ("eps", "inv_eps"):
        volumes = getattr(fields, name, None)
        if isinstance(volumes, dict):
            for component in sorted(volumes):
                state[f"{name}[{component}]"] = volumes[component]
    return state


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose — and the only
    comparison that can tell ``-0.0`` from ``+0.0``, which the lattice class needs."""
    host = array if isinstance(array, np.ndarray) else array.get()
    return np.ascontiguousarray(host).view(np.uint32).ravel()


def snapshot(fields: Any, plan: Any = None) -> Dict[str, np.ndarray]:
    return {name: words(value) for name, value in state_of(fields, plan).items()}


def differing(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape:
        return max(left.size, right.size)
    return int(np.count_nonzero(left != right))


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]
            ) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: count for name in sorted(left)
            if (count := differing(left[name], right[name]))}


def seam_step(fields: Any, pml: Any) -> None:
    """The two array-path passes this product replaces, in driver order."""
    from meep_gpu import stepping  # noqa: PLC0415

    stepping.update_E(fields, pml)          # driver.py:3304
    stepping.update_P(fields, pml)          # driver.py:3306


def subnormal_words(state: Dict[str, np.ndarray]) -> int:
    """float32 words in the subnormal band: exponent zero, mantissa nonzero."""
    total = 0
    for value in state.values():
        exponent = (value >> np.uint32(23)) & np.uint32(0xFF)
        mantissa = value & np.uint32(0x7FFFFF)
        total += int(np.count_nonzero((exponent == 0) & (mantissa != 0)))
    return total


def zero_words(state: Dict[str, np.ndarray], negative: bool) -> int:
    needle = np.uint32(0x80000000) if negative else np.uint32(0)
    return sum(int(np.count_nonzero(value == needle)) for value in state.values())


def _split_movement(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]
                    ) -> Tuple[List[str], List[str], int]:
    moved = {name: differing(before[name], after[name]) for name in before}
    written = [name for name in moved if name.startswith(WRITTEN_PREFIXES)]
    read_only = [name for name in moved if name.startswith(READ_ONLY_PREFIXES)]
    assert set(written) | set(read_only) == set(moved), sorted(
        set(moved) - set(written) - set(read_only))
    still = sorted(name for name in written if moved[name] == 0)
    disturbed = sorted(name for name in read_only if moved[name])
    return still, disturbed, int(sum(moved.values()))


# ---------------------------------------------------------------------------
# Launch counting
# ---------------------------------------------------------------------------

class CountingKernel:
    """Owns the JIT kernel and counts every launch the plan makes through it.

    This is the proof-of-execution instrument. It is NOT a wrapper around the plan:
    the plan calls ``kernel[grid](...)`` and nothing else, so a plan that silently
    did not launch shows up as a count of zero rather than as a passing byte
    comparison.
    """

    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*arguments, **keywords):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*arguments, **keywords)

        return run


def kernel_ptx(jit: Any) -> List[str]:
    """Every compiled specialization's PTX, for the stale-binary tripwire."""
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            assembly = getattr(compiled, "asm", None)
            if assembly and "ptx" in assembly:
                out.append(assembly["ptx"])
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(xp, case: Case, leg: str, steps: int, kernel: Any = None,
             patch: Optional[Callable[[Any], Dict[str, Any]]] = None
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE SEAM STEP."""
    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    seed = case_seed(case.label, leg)
    _, reference, reference_pml = build(xp, case, seed)
    _, actual, actual_pml = build(xp, case, seed)

    counter = CountingKernel(kernel if kernel is not None
                             else family.fused_ade_chain_kernel())
    plan = family.plan_fused_ade_chain(actual, actual_pml, case.arm,
                                       num_warps=1, kernel=counter)
    if plan is None:
        return {"passed": False, "reason": "the fused chain was refused",
                "refusals": list(family.fused_ade_chain_coverage(
                    actual, actual_pml, case.arm).reasons)}
    note: Dict[str, Any] = {}
    if patch is not None:
        note = patch(plan) or {}

    drift = compare(snapshot(reference), snapshot(actual, plan))
    assert not drift, f"{case.label}: the two builds are not identical: {drift}"

    before = snapshot(actual, plan)
    per_step: List[Dict[str, Any]] = []
    census_trail: List[int] = []
    error = None
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        try:
            plan.run()
        except Exception as exc:  # noqa: BLE001 - a refusal IS a measurement here
            error = f"{type(exc).__name__}: {exc}"[:400]
            break
        difference = compare(snapshot(reference), snapshot(actual, plan))
        census = subnormal_words(snapshot(reference))
        census_trail.append(census)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": census})
        if difference:
            break

    after = snapshot(actual, plan)
    still, disturbed, moved_words = _split_movement(before, after)
    identical = (error is None and len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.alias_checks == plan.launches
                   and counter.calls == plan.launches)
    banded = case.value_class == "subnormal_band"
    vacuity_exempt = case.value_class == "pm_zero_lattice"
    floor_ok = (not disturbed) and (vacuity_exempt or not still)
    lattice_live = census_live = None
    if vacuity_exempt:
        reference_state = snapshot(reference)
        lattice_live = bool(zero_words(reference_state, True)
                            and zero_words(reference_state, False))
        floor_ok = floor_ok and lattice_live
    band_reach = None
    if banded:
        # THE BAND IS A CLASS ONLY WHERE THE POLICY LETS ONE EXIST, and this is
        # the three-way rule rather than a pass/fail — the same shape
        # ``gate_triton_cylindrical_complex.signed_zero_reach`` uses for the
        # signed-zero class, for the same reason: a row that COULD NOT have
        # reached the class must not be failed for a property of the
        # configuration, and a row that could have and did not must not be
        # passed.
        #
        # * under ``keep`` the census MUST fire, or the row measured the physical
        #   band under another name — that is VACUOUS;
        # * under ``flush`` it must NOT, because flushing the band to zero IS the
        #   policy. The row then measures something the keeping run cannot: that
        #   the fused launch and the array path agree word for word about a band
        #   the executor has removed from under both of them.
        peak = max(census_trail or [0])
        band_reach = ("live" if peak else
                      "flushed_by_policy" if _POLICY == "flush" else "VACUOUS")
        census_live = band_reach != "VACUOUS"
        floor_ok = floor_ok and census_live
    # A UNIFORM ROW WHOSE CENSUS FIRED is a banded row under another name: the
    # value classes are only classes while each one stays in its own band.
    clean = (case.value_class != "uniform") or (max(census_trail or [0]) == 0)
    return {
        "passed": bool(identical and launches_ok and floor_ok and clean),
        "bit_identical": identical,
        "error": error,
        "subnormal_free": clean,
        "identical_at_step_1": bool(per_step and per_step[0]["differing_words"] == 0),
        "identical_at_last_step": bool(
            per_step and per_step[-1]["differing_words"] == 0
            and len(per_step) == steps),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"] if per_step else None,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else None,
        "reference_subnormal_peak": max(census_trail or [0]),
        "value_class": case.value_class,
        "value_class_live": (lattice_live if vacuity_exempt else census_live),
        "subnormal_band_reach": band_reach,
        "subnormal_policy": _POLICY,
        "arrays_compared": len(before),
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "moved_words": moved_words,
        "runs": plan.runs, "launches": plan.launches,
        "kernel_calls": counter.calls, "alias_checks": plan.alias_checks,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "replaces": list(plan.replaces_sub_steps),
        "arm": plan.arm, "poles": list(plan.counts),
        "extra_scratch_volumes": plan.extra_scratch_volumes,
        "grid": [list(g) for g in counter.grids[:1]],
        "shape": list(plan.shape), "seed": seed, "patch": note,
    }


def run_separate_control(xp, case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path; the two ALREADY CERTIFIED Triton
    products (``no_pml_stored_e``/``dispersive_update_e`` on ``update_E`` and
    ``no_pml_ade``/``launch.plan_ade_update_p`` on ``update_P``) dispatching the
    seam as separate launches; and the fused product. All three must agree word for
    word at every complete seam step, and the LAUNCH COUNTS are recorded on both
    sides — the only place the difference between the compositions shows up at all,
    since a correct fusion is byte-neutral by construction.

    The separate side is not a strawman: those two products are what the corpus row
    is stepped by today, so disagreement here would be a defect in the fused
    product rather than in an invented comparator.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        dispersive_update_e, folded_dispersive_update_e,
        fused_ade_chain as family, launch as launch_module,
        no_pml_ade, no_pml_stored_e,
    )

    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(xp, case, seed)
    _, separate, separate_pml = build(xp, case, seed)
    _, fused, fused_pml = build(xp, case, seed)

    if case.arm == "folded":
        # THE FOLDED ROW TAKES THE FOLDED FAMILY, not the unfolded one. Both
        # dispatch the SAME kernel and the same plan class
        # (folded_dispersive_update_e.py:64-72); what differs is the predicate
        # that establishes the fold, and the unfolded one refuses a live mirror
        # by name. Picking it here would have made the folded control a refusal
        # rather than a comparison.
        electric = folded_dispersive_update_e.plan_folded_dispersive_constitutive(
            separate, separate_pml)
        polarizations = [launch_module.plan_ade_update_p(separate, state)
                         for state in separate.polarizations]
    elif case.arm == "dispersive":
        electric = dispersive_update_e.plan_dispersive_constitutive(
            separate, separate_pml)
        polarizations = [launch_module.plan_ade_update_p(separate, state)
                         for state in separate.polarizations]
    else:
        electric = no_pml_stored_e.plan_stored_e_constitutive(separate, separate_pml)
        polarizations = [no_pml_ade.plan_no_pml_ade_update_p(
            separate, separate_pml, state) for state in separate.polarizations]
    plan = family.plan_fused_ade_chain(fused, fused_pml, case.arm, num_warps=1)
    if electric is None or plan is None or any(p is None for p in polarizations):
        return {"passed": False, "reason": "a declared product was refused",
                "electric": electric is not None,
                "polarizations": [p is not None for p in polarizations],
                "fused": plan is not None}

    before = snapshot(fused, plan)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        electric.run()
        for entry in polarizations:
            entry.run(separate.drive_field)
        plan.run()
        row = {
            "step": step,
            "separate_vs_array": sum(compare(snapshot(reference),
                                             snapshot(separate)).values()),
            "fused_vs_array": sum(compare(snapshot(reference),
                                          snapshot(fused, plan)).values()),
            "fused_vs_separate": sum(compare(snapshot(separate),
                                             snapshot(fused, plan)).values()),
        }
        per_step.append(row)
        if any(value for key, value in row.items() if key != "step"):
            break

    still, disturbed, _ = _split_movement(before, snapshot(fused, plan))
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    separate_dispatches = 1 + len(polarizations) * len(
        tuple(separate.polarizations[0].driven()))
    return {
        "passed": bool(identical and plan.launches and not still and not disturbed),
        "per_step_tail": per_step[-3:],
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": separate_dispatches,
        "fused_dispatches_per_step": plan.launches_per_run,
        "dispatches_removed_per_step": separate_dispatches - plan.launches_per_run,
        "fused_launches": plan.launches,
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "extra_scratch_volumes": plan.extra_scratch_volumes,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Static legs — no device, no Triton
# ---------------------------------------------------------------------------

def _function_text(path: str, name: str) -> str:
    source = open(path, encoding="utf-8").read()
    node = next(child for child in ast.walk(ast.parse(source))
                if isinstance(child, ast.FunctionDef) and child.name == name)
    segment = ast.get_source_segment(source, node)
    assert segment, f"{name} has no source segment in {path}"
    return segment


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def leg_transcription() -> Dict[str, Any]:
    """The arithmetic lines equal the CERTIFIED kernels' own, PARSED not mirrored.

    THE MIRRORED EVALUATOR is a defect class this project has paid for: a check
    that re-implements a kernel's arithmetic mirrors a defect instead of executing
    it. So nothing here re-derives a line. Each certified kernel's own file is
    parsed, the line is searched for in its body AND in this family's, and the two
    are required equal modulo the documented renames.
    """
    kernels = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
    fused = _normalised(_function_text(
        os.path.join(kernels, "fused_ade_chain.py"), "fused_ade_chain_step"))
    stored = _normalised(_function_text(
        os.path.join(kernels, "no_pml_stored_e.py"), "stored_e_constitutive_step"))
    dispersive = _normalised(_function_text(
        os.path.join(kernels, "dispersive_update_e.py"),
        "constitutive_step_dispersive"))
    ade = _normalised(_function_text(os.path.join(kernels, "kernels.py"),
                                     "ade_update_p"))
    expression = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"
    rows: List[Dict[str, Any]] = []

    def row(line: str, certified: str, source: str, wanted: str,
            emitted: str, rename: str) -> None:
        rows.append({
            "line": line, "certified_source": source,
            "present_in_certified": wanted in certified,
            "present_in_fused": emitted in fused, "rename": rename,
            "agrees": (wanted in certified) and (emitted in fused)})

    for index, letter in enumerate(("a", "b", "c")):
        for slot in range(8):
            row(f"pole_subtraction[{letter}{slot}]", stored,
                "no_pml_stored_e.stored_e_constitutive_step",
                f"if NP{index} > {slot}:",
                f"s{index} = s{index} - p{letter}{slot}",
                f"the inline tl.load is NAMED p{letter}{slot} and subtracted")
        row(f"constitutive_product[{index}]", dispersive,
            "dispersive_update_e.constitutive_step_dispersive",
            f"src{index} = s{index} * tl.load(e{index} + idx, mask=live, other=0.0)",
            f"src{index} = s{index} * tl.load(e{index} + idx, mask=live, other=0.0)",
            "(none)")
        row(f"no_pml_store_split[{index}]", stored,
            "no_pml_stored_e.stored_e_constitutive_step",
            f"tl.store(f{index} + idx, s{index} * tl.load(e{index} + idx, "
            f"mask=live, other=0.0), mask=live)",
            f"tl.store(f{index} + idx, src{index}, mask=live)",
            "the stored product is NAMED src, then stored")
        for line in (f"prev{index} = tl.load(w{index} + idx, mask=live, other=0.0)",
                     f"tl.store(w{index} + idx, src{index}, mask=live)",
                     f"v{index} = v{index} + kp_{index} * src{index}",
                     f"v{index} = v{index} - km_{index} * prev{index}",
                     f"tl.store(f{index} + idx, v{index}, mask=live)"):
            row(f"pml_tail[{index}]: {line}", dispersive,
                "dispersive_update_e.constitutive_step_dispersive",
                line, line, "(none)")
        for slot in range(6):
            row(f"ade_recurrence[{letter}{slot}]", ade, "kernels.ade_update_p",
                f"tl.store(p_out + idx, {expression}, mask=live)",
                f"tl.store(p_out_{letter}{slot} + idx, {expression}, mask=live)",
                f"p_out -> p_out_{letter}{slot}")
            row(f"ade_history_load[{letter}{slot}]", ade, "kernels.ade_update_p",
                "q = tl.load(p_prev + idx, mask=live, other=0.0)",
                f"q = tl.load(p_prev_{letter}{slot} + idx, mask=live, other=0.0)",
                f"p_prev -> p_prev_{letter}{slot}")
            row(f"ade_sigma[{letter}{slot}]", ade, "kernels.ade_update_p",
                "s = tl.load(sigma + idx, mask=live, other=0.0)",
                f"s = tl.load(sigma_{letter}{slot} + idx, mask=live, other=0.0)",
                f"sigma -> sigma_{letter}{slot}")
            row(f"ade_pole_register[{letter}{slot}]", ade, "kernels.ade_update_p",
                "p = tl.load(p_now + idx, mask=live, other=0.0)",
                f"p = p{letter}{slot}",
                "THE FUSION: p_now[idx] -> the register the E chain loaded")
            row(f"ade_drive_register[{letter}{slot}]", ade, "kernels.ade_update_p",
                "w = tl.load(drive + idx, mask=live, other=0.0)",
                f"w = src{index}",
                "THE SEAM: drive[idx] -> the register update_E just wrote")
    # ...and neither eliminated load may reappear anywhere in the fused body.
    rows.append({"line": "the drive is never re-loaded", "rename": "(none)",
                 "certified_source": "kernels.ade_update_p",
                 "present_in_certified": "tl.load(drive" in ade,
                 "present_in_fused": "tl.load(drive" not in fused,
                 "agrees": ("tl.load(drive" in ade) and ("tl.load(drive" not in fused)})
    return {"passed": all(entry["agrees"] for entry in rows),
            "rows_checked": len(rows),
            "failed_rows": [entry for entry in rows if not entry["agrees"]]}


def leg_shape(xp) -> Dict[str, Any]:
    """The rotation orbit, walked to closure THROUGH THE SHIPPED PLAN.

    The shape probe's LEG 2 enumeration re-asked on real engine objects with the
    alias check armed, plus its LEG 3 cost re-measured by the plan that pays it. A
    product whose whole licence is a buffer-disjointness argument has to have that
    argument checked against the object that allocates the buffers, not against a
    model of it.
    """
    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        _, fields, pml = build(xp, case, case_seed(case.label, "shape"))
        plan = family.FusedAdeChainPlan(fields, pml, case.arm, 256, num_warps=1)
        driven = len(tuple(fields.polarizations[0].driven()))
        positions: List[Tuple[int, ...]] = []
        conflicts = 0
        for _turn in range(9):
            groups = plan._poles.arrays()
            chain = plan._resolve(groups)
            try:
                plan._check_aliasing(chain)
            except RuntimeError:
                conflicts += 1
            outputs = {id(row[2]) for entries in chain for row in entries}
            inputs = {id(row[3]) for entries in chain for row in entries}
            inputs |= {id(row[4]) for entries in chain for row in entries}
            conflicts += len(outputs & inputs)
            positions.append(tuple(id(state.P[name])
                                   for state in fields.polarizations
                                   for name in state.driven()))
            plan._rotate(chain)
        expected = case.poles * (driven - 1)
        rows.append({
            "case": case.label, "arm": case.arm, "poles": case.poles,
            "driven": driven, "positions_walked": len(positions),
            "conflicts": conflicts,
            "orbit_is_three": positions[0] == positions[3] == positions[6],
            "orbit_is_not_one": positions[0] != positions[1],
            "extra_scratch_volumes": plan.extra_scratch_volumes,
            "extra_expected_K_times_d_minus_1": expected,
            "alias_checks": plan.alias_checks,
            "agrees": (conflicts == 0
                       and plan.extra_scratch_volumes == expected
                       and plan.alias_checks == 9
                       and positions[0] == positions[3] == positions[6]
                       and positions[0] != positions[1])})
        log(f"    shape {case.label}: conflicts={conflicts} "
            f"extra={plan.extra_scratch_volumes}/{expected} orbit=3")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_plan_refusals(xp) -> Dict[str, Any]:
    """Break the chain on purpose; the plan must refuse rather than launch.

    A guard never demonstrated to fire is decoration, and these are the guards the
    shape rests on: no two arms may share an output, no output may be an array the
    launch reads, and the pole ORDER the E chain resolves must still be the one the
    ADE arms were paired against. Driven through the plan's own pre-launch
    resolution, so the leg runs on a host with no Triton too — the merge bar.
    """
    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    case = _case(MUTATION_CASE)
    rows: List[Dict[str, Any]] = []

    def fresh():
        _, fields, pml = build(xp, case, case_seed(case.label, "plan_refusals"))
        return family.FusedAdeChainPlan(fields, pml, case.arm, 256, num_warps=1)

    def refusal(defect: str, needle: str, action: Callable[[Any], Any]) -> None:
        plan = fresh()
        caught = ""
        try:
            action(plan)
        except RuntimeError as exc:
            caught = str(exc)
        rows.append({"defect": defect, "refused": bool(caught),
                     "reason": caught[:220], "expected_reason": needle,
                     "agrees": needle in caught})
        log(f"    plan refusal: {defect} -> refused={bool(caught)}")

    def shared_scratch(plan: Any) -> None:
        plan._scratch[(1, "Ex")] = plan._scratch[(0, "Ex")]
        plan._check_aliasing(plan._resolve(plan._poles.arrays()))

    def output_is_an_input(plan: Any) -> None:
        chain = plan._resolve(plan._poles.arrays())
        plan._check_aliasing(chain)                     # clean before the break
        broken = [list(entries) for entries in chain]
        row = list(chain[0][0])
        row[2] = row[3]
        broken[0][0] = tuple(row)
        plan._check_aliasing(broken)

    def order_reversed_under_the_arms(plan: Any) -> None:
        plan._order["Ex"] = tuple(reversed(plan._order["Ex"]))
        plan._resolve(plan._poles.arrays())

    def order_changed_under_the_live_binding(plan: Any) -> None:
        plan._poles.order["Ex"] = tuple(reversed(plan._poles.order["Ex"]))
        plan._resolve(plan._poles.arrays())

    refusal("two susceptibilities share one Ex scratch",
            "SAME output buffer", shared_scratch)
    refusal("an ADE output is also a P this launch reads",
            "also read by this launch", output_is_an_input)
    refusal("the Ex ADE arms are paired against a reversed pole order",
            "is not the array the subtraction slot resolved",
            order_reversed_under_the_arms)
    refusal("the live pole binding sees a different order than the plan was built for",
            "no longer valid", order_changed_under_the_live_binding)
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_refusals(xp) -> Dict[str, Any]:
    """Configurations the product must refuse, on the host's own arrays."""
    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    table = (
        ("target_dispersive", _case("pml_six_pole_volume_sigma"), "dispersive", True),
        ("target_no_pml", _case("no_pml_five_pole_volume_sigma"), "no_pml", True),
        ("target_folded", _case("folded_pml_five_pole_volume_sigma"), "folded", True),
        ("no_pml_arm_on_an_absorbing_run",
         _case("pml_six_pole_volume_sigma"), "no_pml", False),
        ("dispersive_arm_on_an_inert_layer",
         _case("no_pml_five_pole_volume_sigma"), "dispersive", False),
        ("dispersive_arm_on_a_folded_run",
         _case("folded_pml_five_pole_volume_sigma"), "dispersive", False),
        ("folded_arm_on_an_unfolded_run",
         _case("pml_six_pole_volume_sigma"), "folded", False),
        ("seven_poles",
         dataclasses.replace(_case("pml_six_pole_volume_sigma"),
                             label="seven_poles", poles=7), "dispersive", False),
        ("no_susceptibility",
         dataclasses.replace(_case("pml_six_pole_volume_sigma"),
                             label="no_poles", poles=0), "dispersive", False),
        ("unknown_arm", _case("pml_six_pole_volume_sigma"), "complex", False),
    )
    device = xp is not np
    for name, case, arm, expected in table:
        _, fields, pml = build(xp, case, case_seed(name, "refusals"))
        verdict = family.fused_ade_chain_coverage(fields, pml, arm)
        # ``covered_modulo_backend`` is the convention every predicate record on
        # this board is read through: on a host with no CuPy every verdict is False
        # for that reason ALONE, which is a fact about the host and not about the
        # configuration. On the device host the two columns are the same number and
        # ``planned`` is what proves it.
        residual = [reason for reason in verdict.reasons if "not cupy" not in reason]
        planned = family.plan_fused_ade_chain(fields, pml, arm) is not None
        agrees = (not residual) == expected
        if device:
            agrees = agrees and verdict.covered == expected == planned
        rows.append({"case": name, "arm": arm, "covered": verdict.covered,
                     "covered_modulo_backend": not residual, "planned": planned,
                     "expected": expected, "device": device, "agrees": agrees,
                     "reasons": list(verdict.reasons)[:4]})
        log(f"    refusal {name}: covered={verdict.covered} "
            f"modulo_backend={not residual} expected={expected}")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


#: Scale, and what the per-step census MUST do at it. The clean rows are what make
#: the firing rows mean something — a detector that fired on everything would refuse
#: the physical band too and certify nothing.
PRECONDITION_SCALES: Tuple[Tuple[str, float, Optional[bool]], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-25, True),
    ("band_edge", 1e-30, None),
    ("subnormal_band", 1e-34, False),
    ("deep_subnormal", 1e-41, False),
)


def leg_precondition(xp, case: Case, budget: int = 8) -> Dict[str, Any]:
    """A precondition never demonstrated to FIRE is decoration.

    WHAT THIS LEG MEANS DEPENDS ON THE INSTALLED POLICY, and both meanings are
    asserted rather than one being quietly assumed:

    * under ``keep`` it is the detector's own calibration — the census stays
      silent through the physical band and FIRES in the subnormal one, so a row
      elsewhere that reports "the band fired" is reporting something real;
    * under ``flush`` every row must come back CLEAN, at every scale, including
      the two that fire under ``keep``. That is not the detector failing: it is
      the policy working, measured end to end through the same array path the
      byte comparison runs against. A row that fired here would mean the
      installed ``flush`` policy did not reach the executor that produced it.
    """
    flushing = _POLICY == "flush"
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        if flushing:
            expect_clean = True
        _, reference, reference_pml = build(
            xp, case, case_seed(case.label, f"precondition:{label}"), scale=scale)
        per_step: List[int] = []
        for _ in range(budget):
            seam_step(reference, reference_pml)
            per_step.append(subnormal_words(snapshot(reference)))
        fired = [index for index, count in enumerate(per_step) if count]
        agrees = (expect_clean is None
                  or (expect_clean and not fired)
                  or (not expect_clean and bool(fired)))
        rows.append({"scale": label, "factor": scale,
                     "subnormal_words_per_step": per_step,
                     "census_fired": bool(fired),
                     "first_step": fired[0] if fired else None,
                     "expected_clean": expect_clean, "agrees": agrees,
                     "subnormal_policy": _POLICY})
        log(f"    precondition {label} factor={scale:g} fired={bool(fired)} "
            f"expected_clean={expect_clean}")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows,
            "subnormal_policy": _POLICY,
            "reading": ("under 'flush' EVERY scale must come back clean — the "
                        "policy removed the band; under 'keep' the two banded "
                        "scales must FIRE and the physical ones must not")}


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def _case(label: str) -> Case:
    return next(case for case in CASES if case.label == label)


def shipped_source(family) -> str:
    """The shipped kernel's own text, dedented, decorator included."""
    return textwrap.dedent(inspect.getsource(family.fused_ade_chain_step.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = "import triton\nimport triton.language as tl\n\n"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_ade_chain.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_ade_chain_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _lines_at(source: str, needle: str) -> Tuple[int, str, List[str]]:
    lines = source.split("\n")
    hits = [index for index, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        return -1, "", lines
    index = hits[0]
    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    return index, indent, lines


def _replace_span(source: str, needle: str, span: int,
                  replacement: Sequence[str]) -> Tuple[str, int]:
    """Replace the ``span`` lines starting at the UNIQUE line holding ``needle``."""
    index, indent, lines = _lines_at(source, needle)
    if index < 0:
        return source, 0
    lines[index:index + span] = [indent + line for line in replacement]
    return "\n".join(lines), 1


def _swap_two(source: str, first: str, second: str,
              replacement: Sequence[str]) -> Tuple[str, int]:
    index, indent, lines = _lines_at(source, first)
    if index < 0 or second not in lines[index + 1]:
        return source, 0
    lines[index:index + 2] = [indent + line for line in replacement]
    return "\n".join(lines), 1


#: ``(id, arms, slot, why, expectation, rewrite)``. ``slot`` is the pole slot the
#: edit lands in and :func:`leg_no_dead_mutations` refuses any the mutation case
#: does not reach — the constexpr chains mean an edit to an unlived slot is emitted
#: NOWHERE and measures nothing. ``expectation`` is what the design predicts; the
#: artifact records what was MEASURED, and a predicted null that turns out visible
#: (or the reverse) is reported either way.
def mutation_table() -> Tuple[Tuple[str, Tuple[str, ...], int, str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:

    def m1_drive_reloaded_from_f_w(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return (source.replace(
            needle, "w = tl.load(w0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m1n_drive_reloaded_from_stored_E(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return (source.replace(
            needle, "w = tl.load(f0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m2_seam_takes_the_pre_constitutive_source(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return source.replace(needle, "w = s0"), source.count(needle)

    def m3_pole_reloaded_rather_than_reused(source: str) -> Tuple[str, int]:
        needle = "p = pa0"
        return (source.replace(
            needle, "p = tl.load(a0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m4_ade_association_flattened(source: str) -> Tuple[str, int]:
        needle = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"
        return (source.replace(needle, "p * c_now + c_prev * q + c_drive * s * w"),
                source.count(needle))

    def m5_ade_p_and_q_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "p = pa0", "q = tl.load(p_prev_a0 + idx", (
            "p = tl.load(p_prev_a0 + idx, mask=live, other=0.0)", "q = pa0"))

    def m6_c_now_and_c_prev_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "c_now = cnow_a0", "c_prev = cprev_a0",
                         ("c_now = cprev_a0", "c_prev = cnow_a0"))

    def m7_second_pole_dropped_from_the_e_chain(source: str) -> Tuple[str, int]:
        return _replace_span(source, "s0 = s0 - pa1", 1, ("pa1 = pa1",))

    def m8_ade_arm_one_never_stores(source: str) -> Tuple[str, int]:
        return _replace_span(source, "tl.store(p_out_a1 + idx,", 3, ("pass",))

    def m9_sigma_index_off_by_one(source: str) -> Tuple[str, int]:
        needle = "s = tl.load(sigma_a1 + idx, mask=live, other=0.0)"
        return (source.replace(
            needle, "s = tl.load(sigma_a0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m10_commuted_multiply(source: str) -> Tuple[str, int]:
        needle = "(c_drive * (s * w))"
        return source.replace(needle, "(c_drive * (w * s))"), source.count(needle)

    def m11_pml_accumulations_reversed(source: str) -> Tuple[str, int]:
        return _swap_two(source, "v0 = v0 + kp_0 * src0", "v0 = v0 - km_0 * prev0",
                         ("v0 = v0 - km_0 * src0", "v0 = v0 + kp_0 * prev0"))

    def m12_fw_store_dropped(source: str) -> Tuple[str, int]:
        return _replace_span(source, "tl.store(w0 + idx, src0, mask=live)", 1,
                             ("pass",))

    def m13_e_store_dropped(source: str) -> Tuple[str, int]:
        return _replace_span(source, "tl.store(f0 + idx, src0, mask=live)", 1,
                             ("pass",))

    def m14_arm_reads_another_components_drive(source: str) -> Tuple[str, int]:
        # EZ'S ARMS, READING EX'S DRIVE. The direction matters: an EARLIER
        # component reading a LATER one's register is a NameError the compiler
        # refuses, which measures the emission order rather than the seam. This
        # direction compiles and is a wrong answer.
        needle = "w = src2"
        return source.replace(needle, "w = src0"), source.count(needle)

    # THE FOLDED ARM IS NOT MUTATED, and the absence is a measurement rather than
    # an omission: it compiles the SAME PML=1 specialisation the dispersive arm
    # does (folded is an ADMISSION, not a body — fused_ade_chain.ARMS), so every
    # edit below would re-measure bytes already scored on the dispersive rows. The
    # folded configuration is carried by the product, separate-control, value-class
    # and refusal legs instead, which is where its predicate can differ.
    both = ("dispersive", "no_pml")
    return (
        ("m1_drive_reloaded_from_f_w", ("dispersive",), 0,
         "the fusion itself, undone: read f_w back instead of keeping the register",
         "null", m1_drive_reloaded_from_f_w),
        ("m1n_drive_reloaded_from_stored_E", ("no_pml",), 0,
         "the same, on the arm whose drive is the stored E",
         "null", m1n_drive_reloaded_from_stored_E),
        ("m2_seam_takes_the_pre_constitutive_source", both, 0,
         "feed update_P the D-minus-P before the inverse epsilon multiply",
         "caught", m2_seam_takes_the_pre_constitutive_source),
        ("m3_pole_reloaded_rather_than_reused", both, 0,
         "the second elimination, undone: re-load P instead of reusing the register",
         "null", m3_pole_reloaded_rather_than_reused),
        ("m4_ade_association_flattened", both, 0,
         "ade_update_p's parentheses ARE the float32 answer",
         "caught", m4_ade_association_flattened),
        ("m5_ade_p_and_q_swapped", both, 0,
         "advance the recurrence from the history and the history from P",
         "caught", m5_ade_p_and_q_swapped),
        ("m6_c_now_and_c_prev_swapped", both, 0,
         "the recurrence constants, crossed",
         "caught", m6_c_now_and_c_prev_swapped),
        ("m7_second_pole_dropped_from_the_e_chain", both, 1,
         "one pole missing from D - sum P: a smooth, plausible, wrong field",
         "caught", m7_second_pole_dropped_from_the_e_chain),
        ("m8_ade_arm_one_never_stores", both, 1,
         "a frozen P on one pole of one component",
         "caught", m8_ade_arm_one_never_stores),
        ("m9_sigma_index_off_by_one", both, 1,
         "pole 1's recurrence driven through pole 0's sigma volume",
         "caught", m9_sigma_index_off_by_one),
        ("m10_commuted_multiply", both, 0,
         "float32 multiplication is bitwise commutative: a DECLARED null control",
         "null", m10_commuted_multiply),
        ("m11_pml_accumulations_reversed", ("dispersive",), 0,
         "the split-field accumulation with its two coefficients crossed",
         "caught", m11_pml_accumulations_reversed),
        ("m12_fw_store_dropped", ("dispersive",), 0,
         "a stale f_w: next step's PML history term is one step old",
         "caught", m12_fw_store_dropped),
        ("m13_e_store_dropped", ("no_pml",), 0,
         "a stale E: the drive the NEXT step reads never lands",
         "caught", m13_e_store_dropped),
        ("m14_arm_reads_another_components_drive", both, 0,
         "Ez's poles advanced from Ex's drive — the cross-component seam error",
         "caught", m14_arm_reads_another_components_drive),
    )


def leg_no_dead_mutations(xp) -> Dict[str, Any]:
    """Every armed edit must land in a slot the scored case actually carries.

    THE DEAD MUTATION is a defect class this project has paid for: an edit rewrites
    real lines the scored configuration never emits — here because the ``NP``
    constexpr chain unrolls only the live slots — and then reports UNCAUGHT while
    measuring nothing.
    """
    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    counts: Dict[str, Tuple[int, ...]] = {}
    for arm, label in (("dispersive", MUTATION_CASE), ("no_pml", MUTATION_CASE_NO_PML)):
        case = _case(label)
        _, fields, pml = build(xp, case, case_seed(case.label, "dead"))
        plan = family.FusedAdeChainPlan(fields, pml, case.arm, 256, num_warps=1)
        counts[arm] = tuple(plan.counts)
    rows = []
    for identifier, arms, slot, _why, _expectation, _rewrite in mutation_table():
        for arm in arms:
            live = slot < min(counts[arm])
            rows.append({"mutation": identifier, "arm": arm, "slot": slot,
                         "case_pole_counts": list(counts[arm]), "slot_is_live": live,
                         "agrees": live})
    return {"passed": all(entry["agrees"] for entry in rows),
            "pole_counts": {arm: list(value) for arm, value in counts.items()},
            "rows": rows}


def run_mutations(xp, family, pristine_ptx: Sequence[str],
                  steps: int) -> List[Dict[str, Any]]:
    source = shipped_source(family)
    rows: List[Dict[str, Any]] = []
    cases = {"dispersive": _case(MUTATION_CASE), "no_pml": _case(MUTATION_CASE_NO_PML)}
    for identifier, arms, slot, why, expectation, rewrite in mutation_table():
        for arm in arms:
            case = cases[arm]
            mutated, hits = rewrite(source)
            entry: Dict[str, Any] = {
                "mutation": identifier, "arm": arm, "slot": slot, "why": why,
                "expectation": expectation, "needle_hits": hits,
                "case": case.label, "steps_budget": steps}
            if hits == 0 or mutated == source:
                entry["status"] = "NEEDLE-MISSED"
                entry["passed"] = False
                entry["failure"] = ("the rewrite matched nothing; this mutation was "
                                    "never armed")
                rows.append(entry)
                log(f"  {identifier}/{arm}: NEEDLE-MISSED")
                continue
            kernel_name = f"mutant_{identifier}_{arm}"
            body = mutated.replace("def fused_ade_chain_step(",
                                   f"def {kernel_name}(")
            try:
                mutant = compile_mutant(body, kernel_name)
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "UNCOMPILABLE"
                entry["passed"] = False
                entry["failure"] = f"{type(exc).__name__}: {exc}"[:400]
                rows.append(entry)
                log(f"  {identifier}/{arm}: UNCOMPILABLE {exc}")
                continue
            result = run_case(xp, case, f"mutation:{identifier}", steps,
                              kernel=mutant)
            normalise = (lambda text: text.replace(kernel_name,
                                                   "fused_ade_chain_step"))
            mutant_ptx = [normalise(text) for text in kernel_ptx(mutant)]
            entry["launches"] = result.get("launches")
            entry["error"] = result.get("error")
            entry["ptx_specializations"] = len(mutant_ptx)
            entry["ptx_differs_from_shipped"] = bool(
                mutant_ptx and pristine_ptx
                and all(text not in set(pristine_ptx) for text in mutant_ptx))
            entry["first_divergence"] = result.get("first_divergence")
            entry["differing_words"] = result.get("differing_words")
            caught = result.get("first_divergence") is not None or bool(
                result.get("error"))
            if not entry["launches"] and not result.get("error"):
                entry["status"] = "DISARMED"
                entry["note"] = "the mutant kernel was never launched"
            elif caught:
                entry["status"] = "CAUGHT"
            elif not entry["ptx_differs_from_shipped"]:
                entry["status"] = "PTX-IDENTICAL"
                entry["note"] = (
                    "the mutant compiled to PTX already present for the shipped "
                    "kernel; the source change was canonicalized away and nothing "
                    "was tested by it")
            else:
                entry["status"] = "NULL CONFIRMED"
                entry["note"] = (
                    "launched, PTX differs, and the gate saw no byte difference: "
                    "recorded as a measured null with its evidence")
            entry["passed"] = bool(
                (expectation == "caught" and entry["status"] == "CAUGHT")
                or (expectation == "null"
                    and entry["status"] in ("NULL CONFIRMED", "PTX-IDENTICAL")))
            rows.append(entry)
            log(f"  {identifier}/{arm}: {entry['status']} "
                f"launches={entry['launches']} "
                f"ptx_differs={entry['ptx_differs_from_shipped']} "
                f"expected={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Armed HOST mutations — defects no kernel edit can reach
# ---------------------------------------------------------------------------

def swap_sigma_between_components(plan: Any) -> Dict[str, Any]:
    """Bind Ey's sigma volumes to Ex's arms.

    The kernel takes a pointer per arm and never asks which component's sigma it
    is. A per-component sigma is exactly what an anisotropic material grid gives,
    so this is a silent wrong coupling rather than a crash, and no kernel edit can
    reach it.
    """
    changed = 0
    for slot in range(len(plan._order["Ex"])):
        left, right = plan._sigma[("Ex", slot)], plan._sigma[("Ey", slot)]
        assert getattr(left, "array", None) is not getattr(right, "array", left), (
            "Ex and Ey hold the SAME sigma array, so this mutation is a null and "
            "must not be scored caught")
        plan._sigma[("Ex", slot)], plan._sigma[("Ey", slot)] = right, left
        changed += 1
    return {"slots_swapped": changed}


def reverse_ade_coefficients_on_Ex(plan: Any) -> Dict[str, Any]:
    """Pair pole k's displacement subtraction with pole j's recurrence.

    THE MISPAIRED ARM. The E chain still subtracts the poles in order; only the
    coefficients the ADE arms use are reversed, so every arm advances the right
    volume with the wrong susceptibility's constants — smooth, plausible and wrong.
    """
    slots = len(plan._order["Ex"])
    triples = [plan._coeff[("Ex", slot)] for slot in range(slots)]
    assert len(set(triples)) > 1, "the poles share one coefficient triple"
    for slot, triple in enumerate(reversed(triples)):
        plan._coeff[("Ex", slot)] = triple
    return {"slots_reversed": slots}


def swap_c_prev_and_c_drive(plan: Any) -> Dict[str, Any]:
    """The recurrence constants come from the susceptibility and dt
    (dispersion.py:644). Swapping two of them is a plausible packing slip that no
    kernel edit can reach — the body reads ``cprev_a0`` either way."""
    changed = 0
    for key, (c_now, c_prev, c_drive) in list(plan._coeff.items()):
        plan._coeff[key] = (c_now, c_drive, c_prev)
        changed += 1
    return {"triples_swapped": changed}


def swap_inverse_epsilon(plan: Any) -> Dict[str, Any]:
    """EXPECTED NULL on this configuration, and CONFIRMED as one.

    ``set_isotropic_epsilon_volume`` hands the SAME array back for all three
    components (fields.py:1321-1325), so binding Ey's inverse epsilon to Ex's slot
    binds the identical object. Scoring this "uncaught" would be a false defect; it
    is scored NULL and the identity that makes it one is asserted here rather than
    assumed.
    """
    first, second = plan._inv_eps[0], plan._inv_eps[1]
    assert first.array is second.array, (
        "the two components hold DIFFERENT inverse-epsilon arrays, so this "
        "mutation is live and must not be scored NULL")
    inverse = list(plan._inv_eps)
    inverse[0], inverse[1] = second, first
    plan._inv_eps = tuple(inverse)
    return {"aliased": True}


def lie_about_the_sigma_kind(plan: Any) -> Dict[str, Any]:
    """Tell the constexpr the volume sigma is a scalar.

    ``coverage.sigma_is_volume`` is the ONE place that decides ``SIGMA_IS_VOLUME``
    precisely so the clause and the constexpr cannot disagree; this is what
    disagreeing looks like from the launcher's side.
    """
    assert plan._sv["SV_a0"] == 1, "the scored case has no volume sigma on Ex slot 0"
    plan._sv["SV_a0"] = 0
    return {"flag": "SV_a0 forced to 0 while a pointer is bound"}


#: name -> (patch, expected outcome). "caught" must diverge (or raise); "null" must
#: NOT, and its patch asserts the identity that makes it a null.
HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any], Dict[str, Any]], str]] = {
    "sigma_bound_for_the_wrong_component": (swap_sigma_between_components, "caught"),
    "ade_coefficients_reversed_on_Ex": (reverse_ade_coefficients_on_Ex, "caught"),
    "c_prev_and_c_drive_swapped": (swap_c_prev_and_c_drive, "caught"),
    "sigma_constexpr_disagrees_with_the_binding": (lie_about_the_sigma_kind, "caught"),
    "inverse_epsilon_bound_for_the_wrong_component": (swap_inverse_epsilon, "null"),
}


# ---------------------------------------------------------------------------
# The planted defect — the verdict itself, shown to fail
# ---------------------------------------------------------------------------

#: A mutation leg proves that ONE case diverges when ONE kernel is wrong; it does
#: not prove the gate's RELEASE VERDICT can fail. These edits replace the family's
#: SHIPPED kernel for the whole process, so the complete gate — product, separate
#: control, value classes, disarm — runs against defective bytes and the top-level
#: verdict must flip to FAIL. Run with ``--plant <name>`` and require a nonzero exit.
PLANTED: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ade_association_flattened_everywhere": (
        ("((p * c_now) + (c_prev * q)) + (c_drive * (s * w))",
         "p * c_now + c_prev * q + c_drive * s * w"),),
    "seam_reads_the_stale_pole_instead_of_the_drive": (
        ("w = src0", "w = p"), ("w = src1", "w = p"), ("w = src2", "w = p")),
}


def plant(family, name: str) -> Dict[str, Any]:
    """Replace the family's shipped kernel with a defective one, process-wide."""
    source = shipped_source(family)
    mutated = source
    hits = {}
    for needle, replacement in PLANTED[name]:
        hits[needle] = mutated.count(needle)
        if not hits[needle]:
            raise SystemExit(
                f"planted defect {name!r} matches nothing at {needle!r}; a marker "
                f"that matches nothing disables the check it feeds SILENTLY")
        mutated = mutated.replace(needle, replacement)
    kernel_name = "planted_" + name
    body = mutated.replace("def fused_ade_chain_step(", f"def {kernel_name}(")
    family.fused_ade_chain_step = compile_mutant(body, kernel_name)
    return {"planted": name, "needle_hits": hits}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')} "
        f"elapsed={row.get('elapsed_seconds', '-')}")


def environment(xp) -> Dict[str, Any]:
    record: Dict[str, Any] = {"numpy": np.__version__}
    try:
        import triton  # noqa: PLC0415

        record["triton"] = getattr(triton, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        record["triton_error"] = repr(exc)
    if xp is not np:
        record["cupy"] = getattr(xp, "__version__", "?")
        try:
            device = xp.cuda.Device()
            record["device_id"] = int(device.id)
            record["device_name"] = xp.cuda.runtime.getDeviceProperties(
                device.id)["name"].decode()
            free, total = xp.cuda.runtime.memGetInfo()
            record["device_free_bytes"] = int(free)
            record["device_total_bytes"] = int(total)
        except Exception as exc:  # noqa: BLE001
            record["device_error"] = repr(exc)
    record["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    record["CUPY_CACHE_DIR"] = os.environ.get("CUPY_CACHE_DIR")
    record["TRITON_CACHE_DIR"] = os.environ.get("TRITON_CACHE_DIR")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--mutation-steps", type=int, default=MUTATION_STEPS)
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"),
                        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR "
                             "TO before the first device compile. One process per "
                             "policy: CuPy's cache key sits ABOVE the -ftz strip, "
                             "so a shared cache serves flushed binaries under the "
                             "keeping record's name.")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument("--plant", choices=sorted(PLANTED), default=None,
                        help="run the whole gate against a defective SHIPPED "
                             "kernel; the verdict below MUST come back FAIL")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    started = time.perf_counter()
    payload: Dict[str, Any] = {
        "gate": "triton_fused_ade_chain",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": args.steps, "mutation_steps": args.mutation_steps,
        "no_device": bool(args.no_device),
        "planted_defect": args.plant,
        "verdict_flip_expected": bool(args.plant),
        "source_sha256": source_hashes(),
        "rows": [],
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    jsonl = os.path.splitext(out)[0] + ".jsonl"

    xp: Any = np
    if not args.no_device:
        # CUPY FIXES ITS REDUCTION-ACCELERATOR LIST AT IMPORT, and
        # ``set_reduction_accelerators([])`` does not move where a reduction is
        # dispatched: measured on device (subnormal_policy.py:1112-1120), ``cp.sum``
        # over one 2^-135 among 4095 zeros returned the KEPT word with the list
        # reported empty, because CUB is a binary built with CuPy and the dispatch
        # reads a list fixed at import. There is no in-process route to 'flush' for
        # reductions, so the variable is set BEFORE the import and recorded.
        if args.subnormal_policy == "flush":
            os.environ["CUPY_ACCELERATORS"] = ""
        if "cupy" in sys.modules:
            raise SystemExit(
                "cupy was imported before this gate could set CUPY_ACCELERATORS; "
                "the accelerator list is fixed at import and the requested policy "
                "would be a wish rather than a configuration")
        import cupy as cp  # noqa: PLC0415
        from meep_gpu import backends as _backends  # noqa: PLC0415
        from meep_gpu import subnormal_policy  # noqa: PLC0415

        xp = cp
        # THE HOST LEVER. ``MEEP``'s ``set_zero_subnormals`` is the ONLY exposure
        # of this process's FTZ/DAZ bits this package may use
        # (subnormal_policy.py:1911-1922 refuses without it), so a policy the host
        # is not already sitting on needs MEEP imported first. It is imported ONLY
        # when the measured host state disagrees with the request: importing it
        # initialises MPI as a side effect, and perturbing a configuration that
        # already satisfies the policy would change what the artifact describes.
        want_flush = args.subnormal_policy == "flush"
        before = bool(_backends.subnormals_flushed())
        lever: Dict[str, Any] = {"host_flushing_before": before,
                                 "meep_imported": False}
        if before != want_flush:
            import meep  # noqa: PLC0415, F401

            lever.update(meep_imported=True,
                         meep_version=getattr(meep, "__version__", "unknown"),
                         host_flushing_after_import=bool(
                             _backends.subnormals_flushed()))
        payload["host_subnormal_lever"] = lever
        payload["cupy_accelerators"] = os.environ.get("CUPY_ACCELERATORS")
        log(f"host subnormal lever: {json.dumps(lever, sort_keys=True)}")
        # STRICT, and BEFORE the first device compile: an artifact whose policy
        # field is a wish is worse than none. This also enforces the CUPY_CACHE_DIR
        # token rule (subnormal_policy.cupy_cache_reasons).
        subnormal_policy.install_subnormal_policy(
            args.subnormal_policy, cupy=cp, strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        payload["subnormal_policy_requested"] = args.subnormal_policy
        global _POLICY
        _POLICY = args.subnormal_policy
        log(f"subnormal policy installed: {args.subnormal_policy!r}")
    payload["environment"] = environment(xp)
    log(f"environment: {json.dumps(payload['environment'], sort_keys=True)}")

    from meep_gpu.triton_kernels import fused_ade_chain as family  # noqa: PLC0415

    if args.plant:
        payload["planted"] = plant(family, args.plant)
        log(f"PLANTED DEFECT {args.plant!r}: the verdict below MUST be FAIL")

    value_cases: Tuple[Case, ...] = ()
    controls: Tuple[str, ...] = ()
    if not args.no_device:
        value_cases = tuple(
            [dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                                 value_class="pm_zero_lattice")
             for name in ("no_pml_two_pole_scalar_sigma", "pml_five_pole_scalar_sigma")]
            + [dataclasses.replace(_case(name), label=f"{name}:subnormal_band",
                                   value_class="subnormal_band", scale=1e-34)
               for name in ("no_pml_five_pole_volume_sigma",
                            "pml_six_pole_volume_sigma",
                            "folded_pml_five_pole_volume_sigma")])
        controls = ("no_pml_five_pole_volume_sigma", "pml_six_pole_volume_sigma",
                    "pml_two_pole_metallic", "folded_pml_five_pole_volume_sigma")

    static_legs: List[Tuple[str, str, Callable[[], Dict[str, Any]]]] = [
        ("transcription", "arithmetic_lines_vs_the_certified_kernels",
         leg_transcription),
        ("shape", "the_rotation_orbit_walked_to_closure", lambda: leg_shape(xp)),
        ("shape", "the_plan_refuses_a_broken_chain", lambda: leg_plan_refusals(xp)),
        ("refusals", "the_predicate_table", lambda: leg_refusals(xp)),
    ]
    if not args.no_device and args.plant is None:
        static_legs.append(("mutation", "no_armed_edit_lands_in_a_dead_slot",
                            lambda: leg_no_dead_mutations(xp)))

    total = (len(static_legs) + (0 if args.no_device else
                                 len(CASES) + len(controls) + len(value_cases) + 1))
    if not args.no_device and args.plant is None:
        total += sum(len(arms) for _i, arms, _s, _w, _e, _r in mutation_table())
        total += len(HOST_MUTATIONS) + 1

    rows: List[Dict[str, Any]] = []
    index = 0
    with open(jsonl, "w", encoding="utf-8") as handle:
        for leg, label, runner in static_legs:
            index += 1
            clock = time.perf_counter()
            row = {"index": index, "total": total, "leg": leg, "label": label,
                   **runner()}
            row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
            rows.append(row)
            emit(handle, row)
            payload["rows"] = rows
            save(payload, out)

        if not args.no_device:
            for case in CASES:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "product",
                       "label": case.label, "steps": args.steps,
                       **run_case(xp, case, "product", args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            for name in controls:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "separate_control",
                       "label": name, "steps": args.steps,
                       **run_separate_control(xp, _case(name), args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            for case in value_cases:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "value_classes",
                       "label": case.label, "steps": args.steps,
                       **run_case(xp, case, "value_classes", args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            index += 1
            clock = time.perf_counter()
            row = {"index": index, "total": total, "leg": "value_classes",
                   "label": "the_subnormal_census_precondition",
                   **leg_precondition(xp, _case(MUTATION_CASE))}
            row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
            rows.append(row)
            emit(handle, row)
            payload["rows"] = rows
            save(payload, out)

        if not args.no_device and args.plant is None:
            pristine = kernel_ptx(family.fused_ade_chain_kernel())
            payload["shipped_ptx_specializations"] = len(pristine)
            for entry in run_mutations(xp, family, pristine, args.mutation_steps):
                index += 1
                rows.append({"index": index, "total": total, "leg": "mutation",
                             "label": f"{entry['mutation']}/{entry['arm']}", **entry})
                emit(handle, rows[-1])
                payload["rows"] = rows
                save(payload, out)

            for name, (patch, expected) in HOST_MUTATIONS.items():
                index += 1
                case = _case(MUTATION_CASE)
                try:
                    result = run_case(xp, case, f"host:{name}", args.mutation_steps,
                                      patch=patch)
                    refused = None
                except AssertionError as exc:
                    result = {}
                    refused = f"the patch refused to arm: {exc}"[:300]
                caught = bool(result.get("error")) or (
                    result.get("first_divergence") is not None)
                outcome = "caught" if caught else "NULL CONFIRMED"
                rows.append({
                    "index": index, "total": total, "leg": "mutation",
                    "label": name, "host_defect": True, "expected": expected,
                    "outcome": outcome, "unarmed": refused,
                    "passed": bool(refused is None
                                   and (outcome == "caught") == (expected == "caught")
                                   and (result.get("launches")
                                        or result.get("error"))),
                    "first_divergence": result.get("first_divergence"),
                    "differing_words": result.get("differing_words"),
                    "error": result.get("error"),
                    "patch": result.get("patch"),
                    "launches": result.get("launches")})
                emit(handle, rows[-1])
                payload["rows"] = rows
                save(payload, out)

            # THE DISARM CHECK. The identical harness, the identical case, the
            # SHIPPED bytes and no patch. A nonzero here would mean every "caught"
            # above is a harness that diverges on its own.
            index += 1
            result = run_case(xp, _case(MUTATION_CASE), "disarm",
                              args.mutation_steps)
            rows.append({"index": index, "total": total, "leg": "disarm",
                         "label": "shipped_bytes_on_the_mutation_case",
                         "differing_words": result.get("differing_words"),
                         "launches": result.get("launches"),
                         "written_arrays_that_never_moved": result.get(
                             "written_arrays_that_never_moved"),
                         "passed": bool(result.get("passed"))})
            emit(handle, rows[-1])
            payload["rows"] = rows
            save(payload, out)

    payload["rows"] = rows
    payload["counts"] = {
        "static": len(static_legs),
        "product": 0 if args.no_device else len(CASES),
        "separate_controls": len(controls),
        "value_classes": len(value_cases),
        "mutations": sum(1 for row in rows if row["leg"] == "mutation"),
    }
    payload["failed"] = [row["label"] for row in rows if not row.get("passed")]
    payload["verdict"] = "PASS" if not payload["failed"] else "FAIL"
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    payload["jsonl"] = jsonl
    save(payload, out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"VERDICT {payload['verdict']} in {payload['elapsed_seconds']:.2f}s; "
        f"failed={payload['failed']}; artifact {out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
