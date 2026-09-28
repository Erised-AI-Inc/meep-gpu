#!/usr/bin/env python3
"""CUDA byte gate for the COMPLEX fused E->P chain: complex ``update_E`` welded to
complex ``update_P``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes (``update_E`` at driver.py:3304 and ``update_P`` at :3306) with nothing
between them, so a per-sub-step comparison could not see it at all: the whole claim
is about the SEAM. The comparison is therefore per COMPLETE SEAM STEP over a stated
budget, against

  * the CuPy array path (``stepping.update_E`` then ``stepping.update_P``), and
  * the SEPARATELY CERTIFIED Triton products this one replaces —
    ``complex_no_pml_stored_e`` on ``update_E`` and ``complex_ade`` on ``update_P``
    dispatched as separate launches,

over every allocated volume, compared as uint32 WORDS. complex64 volumes are
compared through their float32 word view, which is the only comparison that can
tell ``-0.0`` from ``+0.0`` in either the real or the imaginary half.
``allclose`` appears nowhere, and the FIRST divergent step is reported rather than
a final pass/fail.

THE SEVEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. Every case
   asserts the exact LAUNCH COUNT through a proxy that owns the kernel object, a
   nonzero ``runs``, one alias check per launch, and a NON-VACUITY FLOOR SPLIT BY
   ROLE — the arrays this seam WRITES (E, every polarization volume) must all have
   moved, and the arrays it only READS (B, D, H, fu_*, the material) must NOT have.
2. **A hollow pass.** Source mutations and HOST mutations are armed, and each is
   scored CAUGHT or NULL CONFIRMED with its evidence; a mutant that compiled to
   PTX already present for the shipped kernel is reported PTX-IDENTICAL rather
   than as a passing null. Leg ``disarm`` reruns the identical case with the
   shipped bytes and requires zero.
3. **A dead mutation.** The E chain and the ADE arms are unrolled by ``NP``
   constexprs, so a rewrite of a slot the scored case does not carry is emitted
   nowhere and measures nothing. Every armed mutation declares the COMPONENT it
   edits and :func:`leg_no_dead_mutations` refuses any whose component the
   mutation case does not drive.
4. **A mirrored evaluator.** :func:`leg_transcription` PARSES the arithmetic out of
   the certified kernels' own files and out of this one, and requires them equal
   modulo the documented renames. It re-implements nothing.
5. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor (a zero state is a fixed point of this seam)
   and must instead carry BOTH signs of zero in the reference output —
   which for this family is the sharpest class it has, because
   ``_mul_field_left``'s zero cross terms are exactly what carries a word's sign
   through the complex multiply; ``subnormal_band`` is COMPARED rather than
   refused, and its census must FIRE or the row is VACUOUS, not passed.
6. **An unmeasured shape.** The whole licence for this product is that one scratch
   per driven component makes the launch's write set disjoint from its read set at
   every position of the rotation's orbit — an argument this family INHERITS from
   the real chain's probe, whose walk never looked at a dtype.
   :func:`leg_shape` RE-WALKS that orbit to closure THROUGH THE SHIPPED PLAN on
   complex64 buffers with the alias check armed, so the inheritance is measured
   rather than asserted, and two host mutations break the chain on purpose and
   require the plan to refuse.
7. **A verdict that cannot fail.** ``--plant`` rewrites the SHIPPED kernel source
   for the whole process and the release verdict must come back FAIL.

AND ONE MORE THIS FAMILY OWNS ALONE: **the expansion arm.** Two certified halves
ask the same probe record for different pattern sets and one fused source carries
ONE ``EXPANSION`` constexpr. :func:`leg_expansion` records what the probe on this
host licenses for each half, requires them equal, and requires the constexpr the
plan launched to BE that value — so a platform that split them would refuse here
rather than silently downgrade one half.

BOTH FLOAT32 SUBNORMAL POLICIES, ONE PER PROCESS. ``--subnormal-policy`` drives
host, CuPy and Triton before the first device compile, and the runner
(``run_triton_complex_fused_ade_chain_direct.sh``) executes this gate twice with
distinct CuPy and Triton cache directories. One process per policy is not
fastidiousness: CuPy computes its kernel cache key ABOVE the seam where the
``-ftz=true`` strip installs, so a cache shared between the two policies serves
flushed binaries under the keeping record's name.

Rule 7: one flushed line per case, every row appended and fsynced as it lands, and
the artifact re-written (provenance-stamped) after every leg.

Usage::

    # laptop, no CUDA and no Triton — the legs that need no device
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/gate_triton_complex_fused_ade_chain.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/gate_triton_complex_fused_ade_chain.py \\
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

#: Complete seam steps every product case walks. SIXTY, not one: the whole ADE
#: history is state carried between steps and the rotation's own period is three,
#: so the defects this gate exists for COMPOUND. The comparison is per COMPLETE
#: STEP, so a green row means identical at ALL sixty — and step 1 is reported
#: separately, because "identical at one launch" and "identical at sixty" are
#: different claims and both are made here.
STEPS = 60

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch and PTX evidence.
MUTATION_STEPS = 3

#: Base for every per-case seed. The seed itself is ``SEED + sha256(label|leg)``,
#: never ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``,
#: so a hash-seeded gate draws a different fixture every process and a failing case
#: cannot be replayed.
SEED = 91_000

COMPONENTS: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"])

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line: demanding movement from all of them fails a correct
#: run, and demanding it from none passes a run that launched nothing. There is no
#: ``f_w_*`` here and that is the family's own scope — this arm exists only where
#: the absorber is inert and ``drive_field`` hands back the stored E.
WRITTEN_PREFIXES = ("E", "P[", "P_prev[", "scratch[")
READ_ONLY_PREFIXES = ("B", "D", "H", "fu_", "f_w_", "eps", "inv_eps")

_TEMPORARY: List[str] = []

#: The float32 subnormal policy this process INSTALLED, set once in :func:`main`
#: before the first device compile.
_POLICY: Optional[str] = None

#: The expansion probe record this process resolved, recorded once so every leg
#: asks the SAME record. ``None`` means "ask the installed artifact", which is what
#: the shipped predicates do.
_PROBE: Any = None


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration. Every one of them is complex64 and absorber-inert —
    that is the whole cell this family serves."""

    label: str
    #: One driven-component tuple per susceptibility. ``CHAIN_MAX_POLES = 1``
    #: refuses two poles on ONE component, so a K > 1 case must give the states
    #: DISJOINT component sets.
    driven: Tuple[Tuple[str, ...], ...]
    kind: str                # LORENTZIAN | DRUDE | "mixed"
    sigma_volume: bool
    cell: Tuple[float, float, float] = (1.2, 1.0, 0.9)
    boundaries: Any = "periodic"
    k_point: Tuple[float, float, float] = (0.2, -0.1, 0.3)
    value_class: str = "uniform"
    scale: float = 1.0


#: THE CASE MATRIX. The corpus cell is measured, not invented:
#: ``results/predicate_coverage_2026-08-21_eop_chain/`` records the four
#: ``TestLoadDump.*_3d`` rows as ONE lorentzian susceptibility driving ALL THREE
#: components under a nonzero ``k_point`` with an inert absorber — which is
#: ``corpus_one_state_three_components``. The rest move exactly one axis of that
#: away so a divergence can be attributed: the pole partition, the sigma kind, the
#: susceptibility kind, the boundary, and the Bloch vector.
CASES: Tuple[Case, ...] = (
    Case("corpus_one_state_three_components", (COMPONENTS,), "lorentzian", False),
    Case("one_state_three_components_volume_sigma", (COMPONENTS,), "lorentzian", True),
    Case("one_state_three_components_drude", (COMPONENTS,), "drude", True),
    Case("one_state_two_components", (("Ex", "Ey"),), "lorentzian", True),
    Case("one_state_one_component", (("Ez",),), "drude", False),
    # K = 2 THROUGH DISJOINT COMPONENT SETS. Without this the ``K *`` in the
    # scratch-cost law is never exercised, and the "which state drives which
    # component" host mutation has nothing to swap.
    Case("two_states_disjoint_components", (("Ex", "Ey"), ("Ez",)), "mixed", True),
    Case("metallic_boundary", (COMPONENTS,), "lorentzian", True,
         boundaries="metallic", k_point=(0.0, 0.0, 0.0)),
    Case("zero_bloch_vector", (COMPONENTS,), "mixed", False,
         k_point=(0.0, 0.0, 0.0)),
)

#: The case every mutation is scored on. Three driven components with VOLUME
#: sigmas, so every armed line is emitted on every component: a one-component row
#: would make the Ey/Ez edits dead and a scalar-sigma row would leave the
#: ``sigma_a0`` volume branch unfilled.
MUTATION_CASE = "one_state_three_components_volume_sigma"


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
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/complex_no_pml_stored_e.py",
        "meep_gpu/triton_kernels/complex_ade.py",
        "meep_gpu/triton_kernels/complex_fused_ade_chain.py",
        "meep_gpu/test_triton_complex_fused_ade_chain.py",
        "parity/meep_gpu/gate_provenance.py",
        "parity/meep_gpu/gate_triton_complex_fused_ade_chain.py",
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically.

    THE STAMP IS HERE AND NOT AT THE CALL SITES. ``main`` calls this after every
    leg, and a stamp bolted onto the last call would leave every partial artifact —
    including the one an aborted run leaves behind, which is the artifact a failure
    is read from — unattributable.
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

    BOTH HALVES OF EVERY COMPLEX VOLUME ARE FILLED. A run that left every
    imaginary word at +0.0 would be a real-valued gate wearing a complex dtype: the
    two expansion arms differ precisely in how the cross terms are rounded, so a
    zero imaginary part hides the whole question this family's ``EXPANSION``
    constexpr answers.
    """
    from meep_gpu.dispersion import (  # noqa: PLC0415
        DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
    )
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    amplitude = case.scale if scale is None else scale
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell),
                boundaries=case.boundaries, dimensions=3, courant=0.35,
                k_point=tuple(case.k_point), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)
    epsilon = (1.45 + 0.30 * rng.random(shape)).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        xp.asarray(np.ascontiguousarray(epsilon)),
        xp.asarray(np.ascontiguousarray((np.float32(1.0) / epsilon).astype(np.float32))))

    kinds = ((DRUDE, LORENTZIAN) if case.kind == "mixed"
             else ((DRUDE,) if case.kind == "drude" else (LORENTZIAN,)))
    for order, driven in enumerate(case.driven):
        base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
                "Ez": 0.19 + 0.06 * order}
        sigma: Dict[str, Any] = {}
        for name, value in base.items():
            if name not in driven:
                sigma[name] = 0.0
            elif case.sigma_volume:
                sigma[name] = xp.asarray(np.ascontiguousarray(
                    (value * (0.7 + 0.6 * rng.random(shape))).astype(np.float32)))
            else:
                sigma[name] = value
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            sigma, grid, np.complex64))

    # THE ABSORBER IS INERT AND THE STORAGE IS FIELD STORAGE. That is the whole
    # cell: ``complex_ade``'s drive clause requires ``drive_field`` to hand back
    # the stored E, and ``complex_no_pml_stored_e`` requires no split-field path.
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=0)

    def fill(array, real, imag):
        array.real = xp.asarray(np.ascontiguousarray(real.astype(np.float32)))
        array.imag = xp.asarray(np.ascontiguousarray(imag.astype(np.float32)))

    if case.value_class == "pm_zero_lattice":
        # A LATTICE of +0.0 and -0.0 over every array the seam consumes, real and
        # imaginary halves lattice-shifted against each other so a cross term that
        # dropped a sign is visible. The inverse epsilon is left NORMAL so the sign
        # propagates through the multiply.
        flat = np.arange(int(np.prod(shape))).reshape(shape)
        lattice = np.where((flat % 2) == 0, np.float32(0.0), np.float32(-0.0))
        alternate = np.where(((flat // 2) % 2) == 0, np.float32(-0.0),
                             np.float32(0.0))
        for name in STATE_NAMES:
            if getattr(fields, name, None) is not None:
                fill(getattr(fields, name), lattice, alternate)
        for state in fields.polarizations:
            for component in state.driven():
                fill(state.P[component], lattice, alternate)
                fill(state.P_prev[component], alternate, lattice)
        return grid, fields, pml

    for name in STATE_NAMES:
        if getattr(fields, name, None) is not None:
            fill(getattr(fields, name),
                 rng.standard_normal(shape) * 0.37 * amplitude,
                 rng.standard_normal(shape) * 0.31 * amplitude)
    for state in fields.polarizations:
        for component in state.driven():
            fill(state.P[component],
                 rng.standard_normal(shape) * 0.21 * amplitude,
                 rng.standard_normal(shape) * 0.17 * amplitude)
            fill(state.P_prev[component],
                 rng.standard_normal(shape) * 0.19 * amplitude,
                 rng.standard_normal(shape) * 0.23 * amplitude)
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
    """One array as uint32 WORDS.

    A complex64 volume views as float32 word pairs with no copy, so this is the
    same comparison for both dtypes — and it is the only one that can tell
    ``-0.0`` from ``+0.0`` in either half.
    """
    host = array if isinstance(array, np.ndarray) else array.get()
    host = np.ascontiguousarray(host)
    if host.dtype == np.complex64:
        host = host.view(np.float32)
    return host.view(np.uint32).ravel()


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

    __slots__ = ("jit", "calls", "grids", "constexprs")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []
        self.constexprs: List[Dict[str, Any]] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*arguments, **keywords):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            self.constexprs.append({
                key: value for key, value in keywords.items()
                if key in ("NP0", "NP1", "NP2", "EXPANSION", "BLOCK",
                           "SV_a0", "SV_b0", "SV_c0")})
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
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
    )

    seed = case_seed(case.label, leg)
    _, reference, reference_pml = build(xp, case, seed)
    _, actual, actual_pml = build(xp, case, seed)

    counter = CountingKernel(kernel if kernel is not None
                             else family.complex_fused_ade_chain_kernel())
    plan = family.plan_complex_fused_ade_chain(
        actual, actual_pml, num_warps=1, kernel=counter, probe=_PROBE)
    if plan is None:
        return {"passed": False, "reason": "the complex fused chain was refused",
                "refusals": list(family.complex_fused_ade_chain_coverage(
                    actual, actual_pml, probe=_PROBE).reasons)}
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
        # THE BAND IS A CLASS ONLY WHERE THE POLICY LETS ONE EXIST, and this is a
        # three-way rule rather than a pass/fail: a row that COULD NOT have reached
        # the class must not be failed for a property of the configuration, and a
        # row that could have and did not must not be passed.
        peak = max(census_trail or [0])
        band_reach = ("live" if peak else
                      "flushed_by_policy" if _POLICY == "flush" else "VACUOUS")
        census_live = band_reach != "VACUOUS"
        floor_ok = floor_ok and census_live
    # A UNIFORM ROW WHOSE CENSUS FIRED is a banded row under another name: the
    # value classes are only classes while each one stays in its own band.
    clean = (case.value_class != "uniform") or (max(census_trail or [0]) == 0)
    # THE IMAGINARY HALF MUST BE LIVE, or this is a real gate in a complex dtype.
    imaginary_live = bool(np.count_nonzero(
        words(getattr(reference, "Ex"))[1::2]))
    return {
        "passed": bool(identical and launches_ok and floor_ok and clean
                       and (vacuity_exempt or imaginary_live)),
        "bit_identical": identical,
        "error": error,
        "subnormal_free": clean,
        "imaginary_half_live": imaginary_live,
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
        "expansion": plan.expansion,
        "constexprs": counter.constexprs[:1],
        "extra_scratch_volumes": plan.extra_scratch_volumes,
        "grid": [list(g) for g in counter.grids[:1]],
        "shape": list(plan.shape), "seed": seed, "patch": note,
    }


def run_separate_control(xp, case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path; the two ALREADY CERTIFIED Triton
    products (``complex_no_pml_stored_e`` on ``update_E`` and ``complex_ade`` on
    ``update_P``) dispatching the seam as separate launches; and the fused product.
    All three must agree word for word at every complete seam step, and the LAUNCH
    COUNTS are recorded on both sides — the only place the difference between the
    compositions shows up at all, since a correct fusion is byte-neutral by
    construction.

    The separate side is not a strawman: those two products are what the corpus row
    would be stepped by, so disagreement here would be a defect in the fused
    product rather than in an invented comparator.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_ade, complex_fused_ade_chain as family, complex_no_pml_stored_e,
    )

    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(xp, case, seed)
    _, separate, separate_pml = build(xp, case, seed)
    _, fused, fused_pml = build(xp, case, seed)

    electric = complex_no_pml_stored_e.plan_complex_stored_e(
        separate, separate_pml, probe=_PROBE)
    polarizations = [complex_ade.plan_complex_ade_update_p(
        separate, separate_pml, state, probe=_PROBE)
        for state in separate.polarizations]
    plan = family.plan_complex_fused_ade_chain(
        fused, fused_pml, num_warps=1, probe=_PROBE)
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
    separate_dispatches = 1 + sum(len(tuple(state.driven()))
                                  for state in separate.polarizations)
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
        os.path.join(kernels, "complex_fused_ade_chain.py"),
        "complex_fused_ade_chain_step"))
    subtract = _normalised(_function_text(
        os.path.join(kernels, "complex_no_pml_stored_e.py"),
        "_subtract_complex_poles"))
    stored = _normalised(_function_text(
        os.path.join(kernels, "complex_no_pml_stored_e.py"),
        "complex_stored_e_step"))
    ade = _normalised(_function_text(os.path.join(kernels, "complex_ade.py"),
                                     "complex_ade_update_p"))
    rows: List[Dict[str, Any]] = []

    def row(name: str, certified_body: str, certified_line: str,
            emitted: str, rename: str) -> None:
        rows.append({
            "line": name,
            "present_in_certified": certified_line in certified_body,
            "present_in_fused": emitted in fused,
            "rename": rename,
            "agrees": (certified_line in certified_body) and (emitted in fused)})

    for index, letter in enumerate("abc"):
        # THE SUBTRACTION CHAIN. Slots 1-7 keep the certified anonymous spelling;
        # slot 0's load is NAMED, which is the ONE documented edit.
        for slot in range(1, 8):
            row(f"subtract[{letter}{slot}].re", subtract,
                f"real = real - tl.load(p{slot} + word, mask=live, other=0.0)",
                f"s{index}_re = s{index}_re - tl.load({letter}{slot} + word, "
                f"mask=live, other=0.0)",
                f"real -> s{index}_re, p{slot} -> {letter}{slot}")
            row(f"subtract[{letter}{slot}].im", subtract,
                f"imag = imag - tl.load(p{slot} + word + 1, mask=live, other=0.0)",
                f"s{index}_im = s{index}_im - tl.load({letter}{slot} + word + 1, "
                f"mask=live, other=0.0)",
                f"imag -> s{index}_im, p{slot} -> {letter}{slot}")
        row(f"subtract[{letter}0].named", subtract,
            "real = real - tl.load(p0 + word, mask=live, other=0.0)",
            f"p{letter}0_re = tl.load({letter}0 + word, mask=live, other=0.0)",
            "the load is NAMED so the register is the ADE arm's p_now")
        # THE CONSTITUTIVE PRODUCT and its store, character for character.
        row(f"constitutive[{index}]", stored,
            f"o{index}_re, o{index}_im = _mul_field_left( s{index}_re, "
            f"s{index}_im, tl.load(e{index} + idx, mask=live, other=0.0), "
            f"EXPANSION)",
            f"o{index}_re, o{index}_im = _mul_field_left( s{index}_re, "
            f"s{index}_im, tl.load(e{index} + idx, mask=live, other=0.0), "
            f"EXPANSION)", "none")
        row(f"store_E[{index}].re", stored,
            f"tl.store(f{index} + word, o{index}_re, mask=live)",
            f"tl.store(f{index} + word, o{index}_re, mask=live)", "none")
        # THE RECURRENCE, every arithmetic line, under the per-component names.
        for line in (
            "a_re, a_im = _mul_field_left(p_re, p_im, c_now, EXPANSION)",
            "b_re, b_im = _mul_coefficient_left(c_prev, q_re, q_im, EXPANSION)",
            "out_re = a_re + b_re",
            "out_im = a_im + b_im",
            "sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)",
            "d_re, d_im = _mul_coefficient_left(c_drive, sw_re, sw_im, EXPANSION)",
            "out_re = out_re + d_re",
            "out_im = out_im + d_im",
        ):
            renamed = (line.replace("c_now", f"cnow_{letter}0")
                           .replace("c_prev", f"cprev_{letter}0")
                           .replace("c_drive", f"cdrive_{letter}0"))
            row(f"recurrence[{letter}] {line.split('=')[0].strip()}", ade, line,
                renamed, f"c_* -> c*_{letter}0")
        row(f"sigma_branch[{letter}].volume", ade,
            "s = tl.load(sigma + idx, mask=live, other=0.0)",
            f"s = tl.load(sigma_{letter}0 + idx, mask=live, other=0.0)",
            f"sigma -> sigma_{letter}0")
        # THE TWO ELIMINATIONS, which are the fusion itself.
        row(f"eliminated_drive[{letter}]", ade,
            "w_re = tl.load(drive + word, mask=live, other=0.0)",
            f"w_re = o{index}_re",
            "the load becomes the register the E half just stored")
        row(f"eliminated_pole[{letter}]", ade,
            "p_re = tl.load(p_now + word, mask=live, other=0.0)",
            f"p_re = p{letter}0_re",
            "the load becomes the register the subtraction chain just held")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_shape(xp) -> Dict[str, Any]:
    """The rotation orbit, walked to closure THROUGH THE SHIPPED PLAN.

    THE ARGUMENT IS INHERITED AND THE MEASUREMENT IS NOT.
    ``probe_triton_ade_rotation_shape``'s LEG 2 enumerated this orbit on buffer
    NAMES, so it transfers to complex64 unchanged — but a transferred argument is
    not a measurement, so it is re-walked here on real complex64 engine objects
    with the alias check armed, plus the scratch cost re-measured by the plan that
    pays it.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
    )

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        _, fields, pml = build(xp, case, case_seed(case.label, "shape"))
        expansion, refusals = family.resolve_chain_expansion(_PROBE)
        if expansion is None:
            rows.append({"case": case.label, "agrees": False,
                         "refusals": refusals})
            continue
        plan = family.ComplexFusedAdeChainPlan(
            fields, pml, 256, expansion, num_warps=1)
        driven = sum(len(tuple(state.driven())) for state in fields.polarizations)
        expected = sum(len(tuple(state.driven())) - 1
                       for state in fields.polarizations)
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
        rows.append({
            "case": case.label, "driven_component_slots": driven,
            "positions_walked": len(positions), "conflicts": conflicts,
            "orbit_is_three": positions[0] == positions[3] == positions[6],
            "orbit_is_not_one": positions[0] != positions[1],
            "extra_scratch_volumes": plan.extra_scratch_volumes,
            "extra_expected_sum_d_minus_1": expected,
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
    shape rests on: no two arms may share an output, and no output may be an array
    the launch reads.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
    )

    case = _case(MUTATION_CASE)
    expansion, _ = family.resolve_chain_expansion(_PROBE)
    rows: List[Dict[str, Any]] = []

    def refuses(name: str, break_it) -> None:
        _, fields, pml = build(xp, case, case_seed(case.label, "refusal:" + name))
        plan = family.ComplexFusedAdeChainPlan(
            fields, pml, 256, expansion or 1, num_warps=1)
        chain = plan._resolve(plan._poles.arrays())
        broken = break_it(plan, chain)
        try:
            plan._check_aliasing(broken)
            refused = False
        except RuntimeError:
            refused = True
        rows.append({"defect": name, "refused": refused, "agrees": refused})
        log(f"    plan refusal {name}: refused={refused}")

    refuses("two_arms_share_one_output",
            lambda plan, chain: [[(r[0], r[1], chain[0][0][2], r[3], r[4])
                                  for r in entries] for entries in chain])
    refuses("an_output_is_an_array_the_launch_reads",
            lambda plan, chain: [[(r[0], r[1], r[3], r[3], r[4])
                                  for r in entries] for entries in chain])
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_refusals(xp) -> Dict[str, Any]:
    """The predicate's table: every bound this family declares, asked and recorded."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
    )

    def ask(fields, pml) -> List[str]:
        return [reason for reason in family.complex_fused_ade_chain_coverage(
            fields, pml, probe=_PROBE).reasons if "not cupy" not in reason]

    rows: List[Dict[str, Any]] = []
    base = _case(MUTATION_CASE)
    _, fields, pml = build(xp, base, case_seed(base.label, "refusals"))
    admitted = ask(fields, pml)
    rows.append({"configuration": "the corpus cell", "expected": "ADMIT",
                 "reasons": admitted, "agrees": not admitted})

    def refuses(name: str, mutate, needle: str) -> None:
        _, f, p = build(xp, base, case_seed(base.label, "refuse:" + name))
        mutate(f, p)
        residual = ask(f, p)
        hit = any(needle in reason for reason in residual)
        rows.append({"configuration": name, "expected": "REFUSE",
                     "needle": needle, "hit": hit,
                     "reasons": residual[:6], "agrees": hit})
        log(f"    refusal {name}: {'BY NAME' if hit else 'MISSED'}")

    def add_a_second_pole(f, p):
        from meep_gpu.dispersion import (  # noqa: PLC0415
            LORENTZIAN, PolarizationState, Susceptibility,
        )
        f.polarizations.append(PolarizationState(
            Susceptibility(0.9, 0.05, LORENTZIAN),
            {name: 0.2 for name in COMPONENTS}, f.grid, np.complex64))

    refuses("a_second_pole_on_one_component", add_a_second_pole, "frozen P")
    refuses("no_susceptibility_at_all",
            lambda f, p: f.polarizations.clear(),
            "no susceptibility is registered")
    refuses("Fields_in_PML_storage_mode",
            lambda f, p: f.enable_pml_storage(), "PML storage mode")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_expansion() -> Dict[str, Any]:
    """The ONE fact this family must not inherit, measured on THIS host.

    The two certified halves ask the same probe record for different pattern sets
    and one fused source carries ONE constexpr. This records what each half's own
    resolver returned, requires them equal, and requires the shared resolver to
    hand back that value — so a platform that split them refuses rather than
    silently downgrading one half.
    """
    from meep_gpu.triton_kernels import complex_ade as ade  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
        complex_no_pml_stored_e as e_half,
    )

    electric = e_half._resolve_expansion(_PROBE)
    ade_arm = ade._resolve_complex_ade_expansion(_PROBE)
    shared, refusals = family.resolve_chain_expansion(_PROBE)
    superset = (set(ade.COMPLEX_ADE_PROBE_PATTERNS)
                > set(e_half.PROBE_PATTERNS))
    agrees = (electric is not None and electric == ade_arm == shared
              and not refusals and superset)
    log(f"    expansion: E half={electric} ADE half={ade_arm} shared={shared}")
    return {"passed": bool(agrees), "e_half_arm": electric,
            "ade_half_arm": ade_arm, "shared_arm": shared,
            "refusals": refusals,
            "ade_pattern_set_is_a_strict_superset": superset,
            "e_half_patterns": list(e_half.PROBE_PATTERNS),
            "ade_half_patterns": list(ade.COMPLEX_ADE_PROBE_PATTERNS)}


def leg_precondition(xp, case: Case, budget: int = 8) -> Dict[str, Any]:
    """Can the subnormal band exist here at all? Measured, then asserted.

    A ``subnormal_band`` row that never reaches the band measures the physical band
    under another name. This walks the ARRAY PATH alone at the banded amplitude and
    reports the census, so the value-class rows' verdicts rest on a measurement
    made without the product in the picture.
    """
    scaled = dataclasses.replace(case, value_class="subnormal_band", scale=1e-34)
    _, fields, pml = build(xp, scaled, case_seed(case.label, "precondition"),
                           scale=1e-34)
    trail: List[int] = []
    for _step in range(budget):
        seam_step(fields, pml)
        trail.append(subnormal_words(snapshot(fields)))
    peak = max(trail or [0])
    expected_live = _POLICY != "flush"
    return {"passed": bool((peak > 0) == expected_live),
            "subnormal_peak": peak, "trail": trail,
            "policy": _POLICY, "expected_live": expected_live}


# ---------------------------------------------------------------------------
# Armed KERNEL-SOURCE mutations
# ---------------------------------------------------------------------------

def _case(label: str) -> Case:
    return next(case for case in CASES if case.label == label)


def shipped_source(family) -> str:
    """The shipped kernel's own text, dedented, decorator included."""
    return textwrap.dedent(
        inspect.getsource(family.complex_fused_ade_chain_step.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = ("import triton\nimport triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import (\n"
              "    _mul_coefficient_left, _mul_field_left)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_complex_chain.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_complex_chain_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _swap_two(source: str, first: str, second: str,
              replacement: Sequence[str]) -> Tuple[str, int]:
    lines = source.split("\n")
    hits = [index for index, line in enumerate(lines) if first in line]
    if len(hits) != 1 or second not in lines[hits[0] + 1]:
        return source, 0
    index = hits[0]
    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    lines[index:index + 2] = [indent + line for line in replacement]
    return "\n".join(lines), 1


#: The value classes a mutation may be scored on, by name. ``uniform`` is the
#: physical band; ``subnormal_band`` is the same case at 1e-34, which is where the
#: expansion probe's own record says the two licensable arms separate (74 words on
#: ``c8_mul_f4_field_left`` under keep, measured candidate-against-candidate).
#: A mutation whose defect can only be visible in an underflowing product must be
#: scored THERE or its null is a statement about the case, not about the kernel.
MUTATION_VALUE_CLASSES: Dict[str, str] = {"uniform": "", "subnormal_band": "1e-34"}


#: ``(id, component, classes, why, expectation, rewrite)``. ``component`` is the E
#: component the edit lands on and :func:`leg_no_dead_mutations` refuses any the
#: mutation case does not drive — the ``NP`` constexprs mean an edit to an undriven
#: component is emitted NOWHERE and measures nothing. ``classes`` is the set of
#: value classes the edit is scored on; "caught anywhere" is caught, and a null is
#: only a null once every class it was scored on agrees.
def mutation_table() -> Tuple[Tuple[str, str, Tuple[str, ...], str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:

    def m1_drive_reloaded_from_stored_E(source: str) -> Tuple[str, int]:
        needle = "w_re = o0_re"
        return (source.replace(
            needle, "w_re = tl.load(f0 + word, mask=live, other=0.0)"),
            source.count(needle))

    def m2_seam_takes_the_pre_constitutive_source(source: str) -> Tuple[str, int]:
        needle = "w_re = o0_re"
        return source.replace(needle, "w_re = s0_re"), source.count(needle)

    def m3_pole_reloaded_rather_than_reused(source: str) -> Tuple[str, int]:
        needle = "p_re = pa0_re"
        return (source.replace(
            needle, "p_re = tl.load(a0 + word, mask=live, other=0.0)"),
            source.count(needle))

    def m4_drive_real_and_imaginary_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "w_re = o0_re", "w_im = o0_im",
                         ("w_re = o0_im", "w_im = o0_re"))

    def m5_pole_real_and_imaginary_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "p_re = pa0_re", "p_im = pa0_im",
                         ("p_re = pa0_im", "p_im = pa0_re"))

    def m6_recurrence_association_broken(source: str) -> Tuple[str, int]:
        # (a + b) + d  ->  a + (b + d). MEASURED CORRECTION 2026-08-21: the first
        # spelling of this edit COMMUTED the outer add (`d_re + out_re`) and the
        # gate reported NULL CONFIRMED on every scored case — correctly, because
        # IEEE-754 addition IS commutative and only associativity fails. A
        # mutation that cannot fail is not evidence, so the edit was replaced with
        # one that re-associates.
        needle = "out_re = out_re + d_re"
        return (source.replace(
            needle, "out_re = a_re + (b_re + d_re)"), source.count(needle))

    def m7_field_left_becomes_coefficient_left(source: str) -> Tuple[str, int]:
        needle = "a_re, a_im = _mul_field_left(p_re, p_im, cnow_a0, EXPANSION)"
        return (source.replace(
            needle,
            "a_re, a_im = _mul_coefficient_left(cnow_a0, p_re, p_im, EXPANSION)"),
            source.count(needle))

    def m8_c_now_and_c_prev_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(
            source,
            "a_re, a_im = _mul_field_left(p_re, p_im, cnow_a0, EXPANSION)",
            "b_re, b_im = _mul_coefficient_left(cprev_a0, q_re, q_im, EXPANSION)",
            ("a_re, a_im = _mul_field_left(p_re, p_im, cprev_a0, EXPANSION)",
             "b_re, b_im = _mul_coefficient_left(cnow_a0, q_re, q_im, EXPANSION)"))

    def m9_second_pole_slot_dropped(source: str) -> Tuple[str, int]:
        needle = ("            s0_re = s0_re - tl.load(a1 + word, mask=live, "
                  "other=0.0)\n")
        return source.replace(needle, ""), source.count(needle)

    def m10_imaginary_store_dropped(source: str) -> Tuple[str, int]:
        # MEASURED CORRECTION 2026-08-21: the first spelling anchored on the
        # line's INDENTATION and `shipped_source` is dedented, so the needle
        # matched nothing and the row read NEEDLE-MISSED. The gate refuses an
        # unarmed mutation rather than scoring it, which is why this was a
        # failure and not a silent pass.
        needle = "tl.store(f0 + word + 1, o0_im, mask=live)"
        return (source.replace(needle, "pass  # the imaginary store, dropped"),
                source.count(needle))

    def m11_expansion_pinned_to_naive(source: str) -> Tuple[str, int]:
        needle = "EXPANSION)"
        return source.replace(needle, "0)"), source.count(needle)

    def m12_ade_output_written_to_the_pole(source: str) -> Tuple[str, int]:
        needle = "tl.store(p_out_a0 + word, out_re, mask=live)"
        return (source.replace(needle, "tl.store(a0 + word, out_re, mask=live)"),
                source.count(needle))

    def m13_sigma_dropped_from_the_drive(source: str) -> Tuple[str, int]:
        needle = "sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)"
        return (source.replace(needle, "sw_re, sw_im = w_re, w_im"),
                source.count(needle))

    def m14_ey_arm_takes_ex_drive(source: str) -> Tuple[str, int]:
        needle = "w_re = o1_re"
        return source.replace(needle, "w_re = o0_re"), source.count(needle)

    def m15_ez_arm_takes_ey_pole(source: str) -> Tuple[str, int]:
        needle = "p_re = pc0_re"
        return source.replace(needle, "p_re = pb0_re"), source.count(needle)

    return (
        ("m1_drive_reloaded_from_stored_E", "Ex", ("uniform",),
         "the eliminated drive load, put back; the register IS what was stored",
         "null", m1_drive_reloaded_from_stored_E),
        ("m2_seam_takes_the_pre_constitutive_source", "Ex", ("uniform",),
         "the drive becomes D - sum P instead of (D - sum P) * inv_eps",
         "caught", m2_seam_takes_the_pre_constitutive_source),
        ("m3_pole_reloaded_rather_than_reused", "Ex", ("uniform",),
         "the eliminated pole load, put back; the register IS that volume",
         "null", m3_pole_reloaded_rather_than_reused),
        ("m4_drive_real_and_imaginary_swapped", "Ex", ("uniform",),
         "the complex drive's halves exchanged", "caught",
         m4_drive_real_and_imaginary_swapped),
        ("m5_pole_real_and_imaginary_swapped", "Ex", ("uniform",),
         "the complex pole's halves exchanged", "caught",
         m5_pole_real_and_imaginary_swapped),
        ("m6_recurrence_association_broken", "Ex", ("uniform",),
         "float addition is not associative in float32", "caught",
         m6_recurrence_association_broken),
        # SCORED ON BOTH CLASSES, and the reason is the probe's own record.
        # Every complex multiply in this kernel has a REAL operand, so the two
        # orientations and the two expansion arms can only separate where the
        # extra precision of the fused product changes a rounded word — the
        # underflow class, not the physical band. Scoring these two on `uniform`
        # alone would report a property of the CASE as a property of the kernel.
        ("m7_field_left_becomes_coefficient_left", "Ex",
         ("uniform", "subnormal_band"),
         "the operand ORIENTATION the probe licensed, reversed", "measure",
         m7_field_left_becomes_coefficient_left),
        ("m8_c_now_and_c_prev_swapped", "Ex", ("uniform",),
         "the recurrence's two history coefficients exchanged", "caught",
         m8_c_now_and_c_prev_swapped),
        ("m9_second_pole_slot_dropped", "Ex", ("uniform",),
         "a slot the mutation case does not carry — DEAD, and named as such so "
         "leg_no_dead_mutations can refuse it if it is ever scored",
         "dead", m9_second_pole_slot_dropped),
        ("m10_imaginary_store_dropped", "Ex", ("uniform",),
         "half of E never written", "caught", m10_imaginary_store_dropped),
        ("m11_expansion_pinned_to_naive", "Ex", ("uniform", "subnormal_band"),
         "the constexpr the probe licensed, replaced by the other arm", "measure",
         m11_expansion_pinned_to_naive),
        ("m12_ade_output_written_to_the_pole", "Ex", ("uniform",),
         "the recurrence writes its own input — the alias this shape exists to "
         "prevent, introduced INSIDE the kernel where the plan's check cannot see "
         "it", "caught", m12_ade_output_written_to_the_pole),
        ("m13_sigma_dropped_from_the_drive", "Ex", ("uniform",),
         "the susceptibility strength removed from the drive term", "caught",
         m13_sigma_dropped_from_the_drive),
        ("m14_ey_arm_takes_ex_drive", "Ey", ("uniform",),
         "one component's recurrence driven by another's field", "caught",
         m14_ey_arm_takes_ex_drive),
        ("m15_ez_arm_takes_ey_pole", "Ez", ("uniform",),
         "one component's recurrence advanced from another's history", "caught",
         m15_ez_arm_takes_ey_pole),
    )


def leg_no_dead_mutations(xp) -> Dict[str, Any]:
    """No SCORED mutation edits a component the mutation case does not drive.

    THE DEAD-BRANCH CLASS, refused rather than discovered: the ``NP`` constexprs
    unroll this kernel, so an edit landing on an undriven component is emitted
    NOWHERE and reports UNCAUGHT while measuring nothing. ``m9`` is declared
    ``dead`` on purpose and is the control for this leg — if it were ever scored
    as a null, this leg would have to fail.
    """
    case = _case(MUTATION_CASE)
    driven = {name for group in case.driven for name in group}
    rows = []
    for identifier, component, classes, _why, expectation, _rewrite in mutation_table():
        alive = component in driven
        # AND THE DECLARED VALUE CLASSES MUST BE REAL ONES. A typo there is not a
        # crash: ``run_case`` would score the row under the uniform rules while the
        # artifact reported it as banded, which is a row measuring one thing and
        # claiming another. :data:`MUTATION_VALUE_CLASSES` is the vocabulary and
        # this is what makes it a check rather than a comment.
        unknown = sorted(set(classes) - set(MUTATION_VALUE_CLASSES))
        rows.append({"mutation": identifier, "component": component,
                     "case_drives_it": alive, "expectation": expectation,
                     "value_classes": list(classes),
                     "unknown_value_classes": unknown,
                     "agrees": (alive or expectation == "dead") and not unknown})
    return {"passed": all(entry["agrees"] for entry in rows),
            "mutation_case": case.label,
            "known_value_classes": sorted(MUTATION_VALUE_CLASSES),
            "components_the_case_drives": sorted(driven), "rows": rows}


def run_mutations(xp, family, pristine_ptx: Sequence[str],
                  steps: int) -> List[Dict[str, Any]]:
    """Every armed edit, on every value class it declares.

    "CAUGHT ANYWHERE IS CAUGHT" and a null is only a null once every class the
    edit declares agrees. The ``measure`` expectation is neither: it says the
    design makes no prediction and the artifact records what the device did — used
    for the two edits whose visibility depends on a rounding class rather than on
    an arithmetic identity.
    """
    source = shipped_source(family)
    base = _case(MUTATION_CASE)
    rows: List[Dict[str, Any]] = []
    for identifier, component, classes, why, expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        entry: Dict[str, Any] = {
            "mutation": identifier, "component": component, "why": why,
            "expectation": expectation, "needle_hits": hits,
            "value_classes": list(classes), "case": base.label,
            "steps_budget": steps}
        if expectation == "dead":
            # DECLARED DEAD AND NOT RUN. Running it would produce an "UNCAUGHT"
            # that measured nothing, which is exactly the reading this project has
            # paid for once already.
            entry["status"] = "DECLARED DEAD — NOT SCORED"
            entry["passed"] = True
            rows.append(entry)
            log(f"  {identifier}: DECLARED DEAD (not scored)")
            continue
        if hits == 0 or mutated == source:
            entry["status"] = "NEEDLE-MISSED"
            entry["passed"] = False
            entry["failure"] = ("the rewrite matched nothing; this mutation was "
                                "never armed")
            rows.append(entry)
            log(f"  {identifier}: NEEDLE-MISSED")
            continue
        kernel_name = f"mutant_{identifier}"
        body = mutated.replace("def complex_fused_ade_chain_step(",
                               f"def {kernel_name}(")
        try:
            mutant = compile_mutant(body, kernel_name)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "UNCOMPILABLE"
            entry["passed"] = False
            entry["failure"] = f"{type(exc).__name__}: {exc}"[:400]
            rows.append(entry)
            log(f"  {identifier}: UNCOMPILABLE {exc}")
            continue

        per_class: List[Dict[str, Any]] = []
        for value_class in classes:
            case = (base if value_class == "uniform" else dataclasses.replace(
                base, label=f"{base.label}:{value_class}",
                value_class=value_class, scale=1e-34))
            result = run_case(xp, case, f"mutation:{identifier}:{value_class}",
                              steps, kernel=mutant)
            caught = result.get("first_divergence") is not None or bool(
                result.get("error"))
            per_class.append({
                "value_class": value_class,
                "launches": result.get("launches"),
                "error": result.get("error"),
                "first_divergence": result.get("first_divergence"),
                "differing_words": result.get("differing_words"),
                "reference_subnormal_peak": result.get(
                    "reference_subnormal_peak"),
                "caught": caught})
        normalise = (lambda text: text.replace(
            kernel_name, "complex_fused_ade_chain_step"))
        mutant_ptx = [normalise(text) for text in kernel_ptx(mutant)]
        entry["per_value_class"] = per_class
        entry["launches"] = sum(row["launches"] or 0 for row in per_class)
        entry["error"] = next((row["error"] for row in per_class if row["error"]),
                              None)
        entry["ptx_specializations"] = len(mutant_ptx)
        entry["ptx_differs_from_shipped"] = bool(
            mutant_ptx and pristine_ptx
            and all(text not in set(pristine_ptx) for text in mutant_ptx))
        entry["first_divergence"] = next(
            (row["first_divergence"] for row in per_class
             if row["first_divergence"] is not None), None)
        entry["differing_words"] = max(
            (row["differing_words"] or 0) for row in per_class)
        entry["caught_on"] = [row["value_class"] for row in per_class
                              if row["caught"]]
        caught = bool(entry["caught_on"])
        if not entry["launches"] and not entry["error"]:
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
                "launched, PTX differs, and the gate saw no byte difference on "
                f"any of {list(classes)}: recorded as a measured null with its "
                "evidence")
        entry["passed"] = bool(
            expectation == "measure"
            or (expectation == "caught" and entry["status"] == "CAUGHT")
            or (expectation == "null"
                and entry["status"] in ("NULL CONFIRMED", "PTX-IDENTICAL")))
        rows.append(entry)
        log(f"  {identifier}: {entry['status']} launches={entry['launches']} "
            f"classes={list(classes)} caught_on={entry['caught_on']} "
            f"ptx_differs={entry['ptx_differs_from_shipped']} "
            f"expected={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Armed HOST mutations — defects no kernel edit can reach
# ---------------------------------------------------------------------------

def swap_sigma_between_components(plan: Any) -> Dict[str, Any]:
    """Bind Ey's sigma volumes to Ex's arm.

    The kernel takes a pointer per arm and never asks which component's sigma it
    is. A per-component sigma is exactly what an anisotropic material grid gives,
    so this is a silent wrong coupling rather than a crash, and no kernel edit can
    reach it.
    """
    left, right = plan._sigma[("Ex", 0)], plan._sigma[("Ey", 0)]
    assert getattr(left, "array", None) is not getattr(right, "array", left), (
        "Ex and Ey hold the SAME sigma array, so this mutation is a null and must "
        "not be scored caught")
    plan._sigma[("Ex", 0)], plan._sigma[("Ey", 0)] = right, left
    return {"swapped": "sigma[Ex,0] <-> sigma[Ey,0]"}


def reverse_ade_coefficients_on_Ex(plan: Any) -> Dict[str, Any]:
    c_now, c_prev, c_drive = plan._coeff[("Ex", 0)]
    assert c_now != c_prev, "the coefficients are equal; this mutation is a null"
    plan._coeff[("Ex", 0)] = (c_prev, c_now, c_drive)
    return {"was": [c_now, c_prev, c_drive], "now": [c_prev, c_now, c_drive]}


def swap_inverse_epsilon(plan: Any) -> Dict[str, Any]:
    """Bind Ey's inverse epsilon to Ex's constitutive slot.

    THIS IS A NULL AND THE REASON IS ASSERTED RATHER THAN DISCOVERED. ``build``
    installs an ISOTROPIC material through ``set_isotropic_epsilon_volume``, which
    hands all three components ONE array (fields.py:1321-1325), so the swap
    exchanges two references to the same allocation. MEASURED 2026-08-21: scored
    as a predicted CATCH it came back NULL CONFIRMED, which was a fact about the
    fixture. The assertion below is what stops that reading recurring — if a
    future fixture gives the components distinct volumes, this patch stops being a
    null and the assertion fires instead of the row quietly changing meaning.
    """
    left, right = plan._inv_eps[0], plan._inv_eps[1]
    aliased = getattr(left, "array", None) is getattr(right, "array", None)
    assert aliased, (
        "the components hold DISTINCT inverse-epsilon volumes, so this patch is a "
        "real defect rather than the declared null; re-score it as 'caught'")
    order = list(plan._inv_eps)
    order[0], order[1] = right, left
    plan._inv_eps = tuple(order)
    return {"swapped": "inv_eps[Ex] <-> inv_eps[Ey]", "arrays_aliased": aliased,
            "why_null": "one isotropic volume serves all three components"}


def lie_about_the_sigma_kind(plan: Any) -> Dict[str, Any]:
    """Tell the kernel a VOLUME sigma is a scalar.

    The constexpr and the argument are decided by one function in the plan; getting
    them out of step reads a pointer as a float, which is a wrong answer and not a
    crash — the exact defect ``coverage.sigma_is_volume`` exists to prevent.
    """
    assert plan._sv["SV_a0"] == 1, "the mutation case must carry a VOLUME sigma"
    plan._sv["SV_a0"] = 0
    return {"SV_a0": "1 -> 0 with the pointer still bound"}


def share_one_scratch_across_components(plan: Any) -> Dict[str, Any]:
    """Undo the shape: point every component's scratch at the first one's.

    THIS IS THE DEFECT THE WHOLE SHAPE ARGUMENT EXISTS TO PREVENT — the reference's
    single shared ``_scratch``, which makes the Ey arm's output the Ex arm's
    ``p_prev`` input inside one launch. The plan's alias check must REFUSE it
    before any launch, so this mutation is expected to be caught as an ERROR
    rather than as a byte difference.
    """
    keys = sorted(plan._scratch)
    assert len(keys) > 1, "this case drives one component; nothing to share"
    first = plan._scratch[keys[0]]
    for key in keys[1:]:
        plan._scratch[key] = first
    return {"shared": [str(key) for key in keys]}


#: ``name -> (patch, expectation)``. Each is scored CAUGHT or NULL CONFIRMED and
#: the expectation is what the design predicts; a predicted null that turns out
#: visible (or the reverse) is reported either way.
HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any], Dict[str, Any]], str]] = {
    "sigma_bound_to_the_wrong_component": (swap_sigma_between_components, "caught"),
    "ade_coefficients_reversed_on_Ex": (reverse_ade_coefficients_on_Ex, "caught"),
    # NULL BY THE FIXTURE, asserted in the patch. See swap_inverse_epsilon.
    "inverse_epsilon_bound_for_the_wrong_component": (swap_inverse_epsilon,
                                                      "null"),
    "a_volume_sigma_declared_scalar": (lie_about_the_sigma_kind, "caught"),
    "one_scratch_shared_across_components": (share_one_scratch_across_components,
                                             "caught"),
}


#: THE VERDICT'S OWN TEST. Each entry replaces the SHIPPED kernel process-wide; the
#: verdict must flip to FAIL. Run with ``--plant <name>`` and require a nonzero exit.
PLANTED: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # MEASURED CORRECTION 2026-08-21. The first entry here re-associated nothing:
    # it COMMUTED the outer add, the planted run came back PASS, and the runner's
    # own "ABORT-CLASS RESULT: the planted defect PASSED" fired. A verdict test
    # that cannot fail is the defect it exists to detect, so the edit was replaced
    # with a genuine re-association — `(a + b) + d` -> `a + (b + d)`, which float32
    # addition does not preserve.
    "recurrence_association_broken_everywhere": (
        ("out_re = out_re + d_re", "out_re = a_re + (b_re + d_re)"),
        ("out_im = out_im + d_im", "out_im = a_im + (b_im + d_im)")),
    "seam_reads_the_stale_pole_instead_of_the_drive": (
        ("w_re = o0_re", "w_re = p_re"), ("w_re = o1_re", "w_re = p_re"),
        ("w_re = o2_re", "w_re = p_re")),
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
    body = mutated.replace("def complex_fused_ade_chain_step(",
                           f"def {kernel_name}(")
    family.complex_fused_ade_chain_step = compile_mutant(body, kernel_name)
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
    parser.add_argument("--probe", default=None,
                        help="path to the complex-multiply expansion probe "
                             "artifact; omitted means 'the installed record', "
                             "which is what the shipped predicates read")
    parser.add_argument("--plant", choices=sorted(PLANTED), default=None,
                        help="run the whole gate against a defective SHIPPED "
                             "kernel; the verdict below MUST come back FAIL")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    started = time.perf_counter()
    payload: Dict[str, Any] = {
        "gate": "triton_complex_fused_ade_chain",
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
        subnormal_policy.install_subnormal_policy(
            args.subnormal_policy, cupy=cp, strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        payload["subnormal_policy_requested"] = args.subnormal_policy
        global _POLICY
        _POLICY = args.subnormal_policy
        log(f"subnormal policy installed: {args.subnormal_policy!r}")
    payload["environment"] = environment(xp)
    log(f"environment: {json.dumps(payload['environment'], sort_keys=True)}")

    # THE PROBE IS RESOLVED ONCE and every leg asks the SAME record. A gate whose
    # legs each re-read the installed artifact could score one leg against a
    # different licence than the next.
    global _PROBE
    if args.probe:
        from meep_gpu.triton_kernels.complex_fields import (  # noqa: PLC0415
            load_expansion_probe,
        )

        _PROBE = load_expansion_probe(args.probe)
        payload["probe_path"] = args.probe
    payload["probe_supplied"] = _PROBE is not None

    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as family,
    )

    if args.plant:
        payload["planted"] = plant(family, args.plant)
        log(f"PLANTED DEFECT {args.plant!r}: the verdict below MUST be FAIL")

    value_cases: Tuple[Case, ...] = ()
    controls: Tuple[str, ...] = ()
    if not args.no_device:
        value_cases = tuple(
            [dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                                 value_class="pm_zero_lattice")
             for name in ("corpus_one_state_three_components",
                          "two_states_disjoint_components")]
            + [dataclasses.replace(_case(name), label=f"{name}:subnormal_band",
                                   value_class="subnormal_band", scale=1e-34)
               for name in ("one_state_three_components_volume_sigma",
                            "two_states_disjoint_components")])
        controls = ("corpus_one_state_three_components",
                    "one_state_three_components_volume_sigma",
                    "two_states_disjoint_components",
                    "metallic_boundary")

    static_legs: List[Tuple[str, str, Callable[[], Dict[str, Any]]]] = [
        ("transcription", "arithmetic_lines_vs_the_certified_kernels",
         leg_transcription),
    ]
    if not args.no_device:
        static_legs += [
            ("expansion", "one_arm_for_both_halves", leg_expansion),
            ("shape", "the_rotation_orbit_walked_to_closure", lambda: leg_shape(xp)),
            ("shape", "the_plan_refuses_a_broken_chain",
             lambda: leg_plan_refusals(xp)),
            ("refusals", "the_predicate_table", lambda: leg_refusals(xp)),
        ]
    if not args.no_device and args.plant is None:
        static_legs.append(("mutation", "no_scored_edit_lands_on_an_undriven_"
                            "component", lambda: leg_no_dead_mutations(xp)))

    total = (len(static_legs) + (0 if args.no_device else
                                 len(CASES) + len(controls) + len(value_cases) + 1))
    if not args.no_device and args.plant is None:
        total += len(mutation_table()) + len(HOST_MUTATIONS) + 1

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
            pristine = kernel_ptx(family.complex_fused_ade_chain_kernel())
            payload["shipped_ptx_specializations"] = len(pristine)
            for entry in run_mutations(xp, family, pristine, args.mutation_steps):
                index += 1
                rows.append({"index": index, "total": total, "leg": "mutation",
                             "label": entry["mutation"], **entry})
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
    payload["status"] = payload["verdict"]
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
