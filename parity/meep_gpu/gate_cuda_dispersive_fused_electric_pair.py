"""Byte-identity gate for the DISPERSIVE hand-CUDA fused pair on the electric seam.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/dispersive_fused_electric_pair.py``: ONE
kernel performing five driver passes in one launch, on a run carrying a
susceptibility::

    step_D (driver.py:3302) -> fill_symmetry_bc_D (:3309) -> zero_metal_D (:3310)
    -> fill_folded_far_ghosts_D (:3311) -> pole-aware update_E (:3313)

THE CELL IS THE LARGEST ANY BACKEND HAD LEFT UNOCCUPIED: ``D_to_E (cuda_curl/PML,
cuda_dispersive/dispersive)``, seven corpus rows, verdict POINTWISE-BUILDABLE, no
product on it. All seven carry an ELECTRIC DEPOSIT inside the seam, so the deposit
legs are not a side observation here -- without the bracket this product serves ZERO.

=============================================================================
THE ONE THING THIS GATE MEASURES THAT NO OTHER GATE ON ANY BACKEND DOES
=============================================================================

An imaged GHOST's ``D - sum_n P_n`` must be re-formed from the DESTINATION's pole
words. ``update_E`` is element-wise over the whole volume
(``Fields.displacement_minus_polarization``, fields.py:1096-1105), so the ghost cell
subtracts ITS OWN ``P``; the source thread that writes it holds a different cell's.
Both sibling backends refuse a folded grid on their dispersive pair -- Metal through
``dispersive_update_e``'s "a mirror plane is active" clause (it ships the fold as a
separate product), Triton through the same clause in its own
``dispersive_constitutive_coverage`` -- so neither has ever met the question. FOUR OF
THE SEVEN ROWS ARE FOLDED, so this product carries it, and
:data:`SOURCE_MUTATIONS`'s ``ghost_chain_reads_the_source_poles`` is what makes the
carry a measurement rather than a claim.

``parity/meep_gpu/probe_cuda_dispersive_electric_pair.py`` answered the same question
OFF DEVICE first, against the driver's own five passes over complete steps, with six
armed mutations. This gate answers it about the COMPILED kernel.

=============================================================================
WHY THE COMPARISON IS PER COMPLETE DRIVER STEP, WITH THE POLE VOLUMES IN IT
=============================================================================

Two engines are built from one seed and stepped side by side: the reference runs
``stepping``'s own pass sequence, the subject runs the fused launch plus the passes it
does not replace, and the two are compared as raw uint32 WORDS after EVERY step --
over the twenty-four stored field volumes AND over every ``P``/``P_prev`` buffer the
run carries. The pole volumes are in the comparison because ``update_P`` closes the
step from ``f_w_E``, so a wrong constitutive source reaches ``P`` on the same step and
``D`` on the next.

THE POLE VOLUMES ARE SEEDED NON-ZERO, and that is a precondition this gate asserts
rather than assumes. Zero is a fixed point of ``s = s - P``: a zero-seeded fixture
turns every pole mutation into a null and would report a green sweep that measured
nothing. ``operand_census`` records the live pole count per case and a case whose
poles are all zero FAILS.

=============================================================================
WHAT THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A step this gate invented.** The driver-order leg is the real twin's own: it
   reads ``FdtdDriver.step``'s source and requires ``REPLACES`` to be a subset of the
   pass list in that order.

2. **A silent fallback.** Bytes alone cannot prove the kernel ran -- an engine that
   never launched it is byte-identical to the oracle BY CONSTRUCTION. Every case
   asserts exact launch counts from TWO INDEPENDENT COUNTERS and requires them to
   agree, to equal the step budget, and to name only this kernel.

3. **A vacuous comparison.** Every stored volume must move on the uniform class; the
   read-only volumes (the PML vectors, the inverse-permittivity volumes AND the pole
   buffers, which this kernel reads and ``update_P`` writes) must be bit-unchanged
   ACROSS THE LAUNCH.

4. **An arity nobody swept.** The product sweep runs every arity in
   ``dispersive_kernels.POLE_COUNTS_SWEPT`` that the corpus drives, including
   ``(0, 0, 0)`` -- where the emitted constitutive lines are the non-dispersive
   family's and the whole product must reduce to its twin's arithmetic -- and the two
   corpus arities ``(5, 5, 5)`` and ``(6, 6, 6)``.

5. **A chain summed instead of subtracted.** float32 addition is not associative, so
   ``((D - P0) - P1)`` and ``D - (P0 + P1)`` are different numbers at two poles and
   above. Armed, and scored ONLY at arity >= 2 where a difference can exist.

6. **A ghost reading the wrong cell's poles, inverse epsilon or coefficient row.**
   Three separate needles, each scored only where a ghost is actually imaged.

7. **A predicate that admits what the launch cannot serve**, and its complement: the
   REAL twin must refuse every configuration this one admits and vice versa, because
   two admitters on ``step_D`` leave the seam UNFUSED.

8. **A deposit computed against a pre-injection field**, on a DISPERSIVE constitutive
   half -- which is what ``deposit_repair`` has never been gated against on this
   track.

=============================================================================
WHAT THIS GATE DOES NOT CLAIM
=============================================================================

No throughput claim. No dispatch claim. No verdict about a complex, cylindrical,
conductive, BFAST, special-kz, off-diagonal or nonlinear run -- every one is refused
BY NAME and none was swept. No verdict about the non-dispersive twin, which is a
different product with its own gate.

=============================================================================
RUNNING IT
=============================================================================

ONE verified-empty GPU, pinned by UUID. ``CUPY_ACCELERATORS`` must be EMPTY IN THE
ENVIRONMENT before CuPy imports, and the CuPy disk cache must be private per policy::

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
      python -u gate_cuda_dispersive_fused_electric_pair.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/native \\
      python -u gate_cuda_dispersive_fused_electric_pair.py \\
        --subnormal-policy flush --import-meep-for-host-policy \\
        --out $OUT/flush/gate.json

``--out`` is a FILE. One flushed line per case (the progress-reporting rule); the artifact is
rewritten atomically after every case, so an interrupted run keeps everything up to
the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE REAL TWIN'S GATE, IMPORTED FOR ITS SHARED MACHINERY. This product IS that
# product with one half exchanged, and its gate's driver-order reader, launch counter,
# pass runner and needle helper answer the same questions here. Importing them keeps
# ONE copy of each: a second transcription of the driver's pass list is a second place
# for this gate to walk an order the driver does not.
import gate_cuda_fused_electric_pair as base  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

run_pass = base.run_pass
array_step = base.array_step
needle = base.needle
words = base.words
differing = base.differing
MemoLaunchCounter = base.MemoLaunchCounter
DRIVER_ORDER = base.DRIVER_ORDER
IMMUTABLE_PML = base.IMMUTABLE_PML
GUARD_OPTIONS = base.GUARD_OPTIONS
VALUE_CLASSES = base.VALUE_CLASSES

SEED = 20260902

#: Complete DRIVER STEPS every product leg runs.
STEPS = 60

#: The five passes ONE launch performs -- read from the module, never transcribed.
FUSED_PASSES: Tuple[str, ...] = base.FUSED_PASSES

#: Every stored FIELD volume a complete step can touch. The pole buffers are added
#: per case by :func:`state_names`, because how many there are is the arity.
FIELD_NAMES: Tuple[str, ...] = base.STATE_NAMES

#: The families the SUBNORMAL BAND class is held to, with the pole volumes added: the
#: ADE recurrence writes ``P`` every step from ``f_w_E`` and a run whose ``P`` never
#: moved would be one where the chain this product exists for did nothing.
BAND_MUST_MOVE: Tuple[str, ...] = base.BAND_MUST_MOVE

MOVEMENT_FLOOR_NOTE = base.MOVEMENT_FLOOR_NOTE + (
    " The pole volumes are held to the same rule as the field ones on the uniform "
    "class: update_P writes every driven P from f_w_E on every step, so a P that "
    "never moved is a chain that never ran and the case would be measuring an "
    "engine with no dispersion in it.")

CELL: Tuple[float, float, float] = (9.0, 10.0, 11.0)

#: The sweep. THE FIRST TWO ROWS ARE THE CORPUS SHAPES this cell owns, read off the
#: stamped census: the three ``stochastic_emitter`` rows are all-periodic with an
#: active layer and six susceptibilities on every component, and the four
#: ``TestLoadDump.*_2d`` rows are metallic on x, mirror-folded on y, periodic on z
#: with five. The rest exist to arm mutations those two cannot, and to walk the arity
#: axis: every axis is walled somewhere, one row is walled nowhere, one row is folded
#: on two axes at mixed phase, half the rows are diagonally ANISOTROPIC (an isotropic
#: run binds one inverse-epsilon allocation three times and would hide a per-component
#: binding defect), and the arities span 0 to the corpus ceiling.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_all_periodic", "boundaries": ("periodic",) * 3, "pml": 2,
     "cell": CELL, "epsilon": "anisotropic", "arity": (6, 6, 6)},
    {"label": "corpus_wall_x_fold_y",
     "boundaries": ("metallic", "metallic", "periodic"), "pml": 2, "cell": CELL,
     "epsilon": "anisotropic", "symmetry": (("Y", 1),), "arity": (5, 5, 5)},
    # THE ARITY-ZERO REDUCTION IS NOT A SPEC HERE, AND THAT IS MEASURED RATHER THAN
    # OMITTED. At ``(0, 0, 0)`` this product's predicate REFUSES by name -- "no
    # susceptibility is registered ... the run belongs to
    # coverage.covers_real_pml_constitutive" -- which is the single boolean that keeps
    # the two electric products disjoint, so a spec at that arity could not be
    # launched at all. ``leg_refusal``'s
    # ``degenerate_arity_belongs_to_the_real_twin`` row is where it is recorded, and
    # ``mixed_arity_wall_z`` below is where the degenerate LINE is measured on a
    # device: its Ey carries no contributor, so Ey's emitted source is character for
    # character the non-dispersive family's inside a configuration this product does
    # admit.
    {"label": "mixed_arity_wall_z",
     "boundaries": ("periodic", "periodic", "metallic"), "pml": 2, "cell": CELL,
     "epsilon": "anisotropic", "arity": (2, 0, 3)},
    {"label": "wall_xyz", "boundaries": ("metallic",) * 3, "pml": 2, "cell": CELL,
     "epsilon": "isotropic", "arity": (2, 2, 2)},
    {"label": "wall_y_single_pole", "boundaries": ("periodic", "metallic", "periodic"),
     "pml": 2, "cell": CELL, "epsilon": "anisotropic", "arity": (1, 1, 1)},
    {"label": "fold_y_periodic", "boundaries": ("periodic",) * 3, "pml": 2,
     "cell": CELL, "epsilon": "anisotropic", "symmetry": (("Y", 1),),
     "arity": (2, 2, 2)},
    {"label": "fold_y_odd_wall_z",
     "boundaries": ("periodic", "periodic", "metallic"), "pml": 2, "cell": CELL,
     "epsilon": "isotropic", "symmetry": (("Y", -1),), "arity": (2, 2, 2)},
    {"label": "fold_xy_mixed_phase", "boundaries": ("periodic",) * 3, "pml": 2,
     "cell": CELL, "epsilon": "anisotropic",
     "symmetry": (("X", 1), ("Y", -1)), "arity": (3, 2, 1)},
    {"label": "fold_x_deep_pml", "boundaries": ("periodic",) * 3,
     "pml": ((0, 5), (2, 2), (2, 2)), "cell": CELL, "epsilon": "anisotropic",
     "symmetry": (("X", 1),), "arity": (2, 2, 2)},
)

#: The rows every mutation is scored on. ``fold_x_deep_pml`` is here for ONE leg and
#: it is the leg that matters: with the absorber five cells deep on the high x face,
#: the far ghost's coefficient row (stored ``nx - 1``) and the source thread's
#: (``reflect_x``) hold DIFFERENT absorber values, so
#: ``far_ghost_takes_the_source_coefficient_row`` is a defect that is present rather
#: than one the profile happens to hide.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "corpus_all_periodic", "corpus_wall_x_fold_y", "wall_xyz", "mixed_arity_wall_z",
    "fold_y_odd_wall_z", "fold_xy_mixed_phase", "wall_y_single_pole",
    "fold_x_deep_pml")
SEPARATE_CONTROL_LABELS: Tuple[str, ...] = ("corpus_all_periodic", "wall_xyz")
DEPOSIT_SPEC_LABELS: Tuple[str, ...] = ("corpus_all_periodic", "wall_xyz",
                                        "corpus_wall_x_fold_y")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def case_rng(label: str) -> np.random.Generator:
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big"))


def pole_names(arity: Sequence[int]) -> Tuple[str, ...]:
    """The pole volumes' record names at one arity, in the chain's own order."""
    return tuple(f"P_{component}_{index}"
                 for component, count in zip(("Ex", "Ey", "Ez"), arity)
                 for index in range(int(count)))


def state_names(arity: Sequence[int]) -> Tuple[str, ...]:
    return FIELD_NAMES + pole_names(arity)


def build(spec: Dict[str, Any], value_class: str, rng):
    """One seeded engine: ``(fields, grid, pml, host)``, with LIVE poles.

    The field half is the real twin's own ``build`` -- one copy of the grid, layer,
    permittivity and seeding rules -- and this adds the susceptibilities.

    THE POLE VOLUMES ARE SEEDED NON-ZERO IN BOTH CLASSES. Zero is a fixed point of
    ``s = s - P``: a zero-seeded fixture would make every pole mutation a null and the
    sweep would report a pass it never earned.
    """
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    fields, grid, pml, host = base.build(spec, value_class, rng)
    arity = tuple(int(v) for v in spec["arity"])
    shape = tuple(int(n) for n in grid.shape)
    kinds = ("lorentzian", "drude")
    for index in range(max(arity) if arity else 0):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        # A PER-COMPONENT SIGMA OF ZERO is the engine's own spelling of "this term
        # does not drive that component" (dispersion.py:645-647), which is MEEP's
        # trivial_sigma. A fixture that poked ``_driven`` would be measuring an object
        # the engine cannot build.
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(("Ex", "Ey", "Ez"))}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    pole_host: Dict[str, np.ndarray] = {}
    for name, array in zip(pole_names(arity), _pole_arrays(fields)):
        if value_class == "uniform":
            values = rng.uniform(-0.6, 0.6, size=shape).astype(np.float32)
        else:
            values = subnormal_band_hosts((name,), shape, rng)[name]
        array[...] = cp.asarray(np.ascontiguousarray(values))
        pole_host[name] = values
    # P_prev matters as much as P: the ADE recurrence is second order, so a zero
    # history would start the run in a state it never lived through.
    for state in fields.polarizations:
        for component, array in state.P_prev.items():
            array[...] = cp.asarray(np.ascontiguousarray(
                rng.uniform(-0.6, 0.6, size=shape).astype(np.float32)))
    host.update(pole_host)
    return fields, grid, pml, host


def _pole_arrays(fields) -> List[Any]:
    """``P`` in the chain's own order -- the launcher's own reading, not a second one."""
    from meep_gpu.cuda_kernels import dispersive_kernels  # noqa: PLC0415

    plan = dispersive_kernels.resolve_pole_plan(fields)
    return [array for component in ("Ex", "Ey", "Ez") for array in plan[component]]


def state_of(fields, arity) -> Dict[str, Any]:
    values = {name: getattr(fields, name) for name in FIELD_NAMES}
    values.update(zip(pole_names(arity), _pole_arrays(fields)))
    return values


def compare(left, right, arity) -> Dict[str, int]:
    a, b = state_of(left, arity), state_of(right, arity)
    return {name: differing(a[name], b[name])
            for name in a if differing(a[name], b[name])}


def frozen(fields, arity) -> Dict[str, np.ndarray]:
    return {name: words(array) for name, array in state_of(fields, arity).items()}


def read_only_snapshot(fields, pml, arity) -> Dict[str, np.ndarray]:
    """Every volume the LAUNCH may only read, as words.

    The PML vectors and the three inverse-permittivity volumes are the real twin's
    set; THE POLE BUFFERS JOIN IT HERE. This kernel binds them ``__restrict__`` and
    read-only, and ``update_P`` -- a later sub-step -- is the only writer. A kernel
    that wrote through one of those bindings is UB NVRTC does not diagnose, and the
    snapshot is taken and checked ACROSS THE LAUNCH, before ``update_P`` runs.
    """
    snapshot = {name: words(getattr(pml, name)) for name in IMMUTABLE_PML
                if getattr(pml, name, None) is not None}
    for component in ("Ex", "Ey", "Ez"):
        snapshot[f"inv_eps_{component}"] = words(fields.inverse_epsilon_for(component))
    for name, array in zip(pole_names(arity), _pole_arrays(fields)):
        snapshot[f"pole:{name}"] = words(array)
    return snapshot


def read_only_drift(before: Dict[str, np.ndarray], fields, pml, arity) -> List[str]:
    after = read_only_snapshot(fields, pml, arity)
    return sorted(name for name in before
                  if before[name].shape != after[name].shape
                  or bool(np.any(before[name] != after[name])))


def live_pole_words(host: Dict[str, np.ndarray], arity) -> int:
    """How many pole words are NONZERO in the seeded state.

    Asserted per case: zero is a fixed point of the subtraction chain, so a fixture
    whose poles are all zero cannot arm a single pole mutation and every verdict it
    produced would be about an engine with no dispersion in it.
    """
    return sum(int(np.count_nonzero(host[name])) for name in pole_names(arity))


# ---------------------------------------------------------------------------
# The product leg
# ---------------------------------------------------------------------------

def run_case(spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             tables_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             codes_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
             walls_patch: Optional[Callable[[Sequence[int], Any], Sequence[int]]] = None,
             fills_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             plan_patch: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP.

    ``kernel`` and the ``*_patch`` hooks are THE GATE'S DOORS. ``plan_patch`` is this
    family's own and is the one the real twin has no analogue for: the shipped
    launcher takes a pole plan precisely so a harness can hand it a REVERSED chain,
    which is the ordering defect that actually threatens this kernel.
    """
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import dispersive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|{value_class}{label_suffix}"
    arity = tuple(int(v) for v in spec["arity"])
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "symmetry": [list(pair) for pair in spec.get("symmetry", ())],
                            "epsilon": spec.get("epsilon", "isotropic"),
                            "arity": list(arity), "pml_cells": spec["pml"]}

    reference, ref_grid, ref_pml, host = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["operand_census"] = operand_census(host)
    case["live_pole_words"] = live_pole_words(host, arity)
    case["distinct_inverse_epsilon_allocations"] = len({
        int(actual.inverse_epsilon_for(c).data.ptr) for c in ("Ex", "Ey", "Ez")})
    if sum(arity) and not case["live_pole_words"]:
        case["passed"] = False
        case["why"] = ("the pole volumes are all zero, so every pole mutation is a "
                       "null and this case measures nothing about the chain")
        return case

    drift = compare(reference, actual, arity)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case

    covered, reason = family.covers_dispersive_fused_electric_pair(
        actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    twin_covered, twin_reason = real.covers_fused_electric_pair(actual, pml, grid, ())
    case["real_twin_predicate"] = {"covered": bool(twin_covered),
                                   "reason": twin_reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case
    if twin_covered:
        # TWO ADMITTERS ON ONE SEAM leave it UNFUSED naming both, so an overlap does
        # not add coverage -- it costs the 79 rows the real twin serves.
        case["passed"] = False
        case["why"] = ("the REAL twin admits this fixture too; install_fused_pairs "
                       "would leave the seam unfused naming both products")
        return case

    tables = real.fused_electric_pair_tables(pml)
    plan = dispersive_kernels.resolve_pole_plan(actual)
    if plan_patch is not None:
        plan = plan_patch(plan)
    poles = tuple(array for component in ("Ex", "Ey", "Ez") for array in plan[component])
    counts = dispersive_kernels.pole_counts_of(plan)
    case["distinct_allocations_checked"] = family.assert_disjoint_bindings(
        actual, tables, poles)
    case["compiled_arity"] = list(counts)
    if tables_patch is not None:
        tables = tables_patch(tables, pml)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    fills = real.fused_electric_pair_fills(grid)
    case["boundary_codes"] = [int(c) for c in codes]
    case["zero_metal_axes"] = [bool(w) for w in walls]
    case["fills"] = {key: list(fills[key]) for key in ("near", "reflect", "phase")}
    case["fills_live"] = bool(any(fills["near"])
                              or any(row >= 0 for row in fills["reflect"]))
    if codes_patch is not None:
        codes = codes_patch(codes)
    if walls_patch is not None:
        walls = walls_patch(walls, grid)
    if fills_patch is not None:
        fills = fills_patch(fills, grid)
    case["boundary_codes_used"] = [int(c) for c in codes]
    case["zero_metal_axes_used"] = [bool(w) for w in walls]
    case["fills_used"] = {key: list(fills[key])
                          for key in ("near", "reflect", "phase")}

    before = frozen(actual, arity)
    launched = 0
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    read_only_breaches: List[str] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            # THE POLE PLAN IS RE-RESOLVED EVERY STEP, because PolarizationState.update
            # rotates the buffers (dispersion.py:689-691) and a plan cached across
            # steps names last step's arrays -- stale in a way that still computes.
            live = dispersive_kernels.resolve_pole_plan(actual)
            if plan_patch is not None:
                live = plan_patch(live)
            step_poles = tuple(array for component in ("Ex", "Ey", "Ez")
                               for array in live[component])
            snapshot = read_only_snapshot(actual, pml, arity)
            for name in DRIVER_ORDER:
                if name == "step_D":
                    report = family.launch_dispersive_fused_electric_pair(
                        actual, tables, codes, walls, fills, case["dtdx"],
                        step_poles, dispersive_kernels.pole_counts_of(live), kernel)
                    launched += int(bool(report.get("launched")))
                    cp.cuda.runtime.deviceSynchronize()
                    # ACROSS THE LAUNCH, before update_P writes P: the pole buffers
                    # are read-only to THIS kernel and a write through one of them
                    # would otherwise be indistinguishable from the ADE's own store.
                    read_only_breaches.extend(
                        f"step {step}: {name}"
                        for name in read_only_drift(snapshot, actual, pml, arity))
                elif name not in FUSED_PASSES:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual, arity)
            census = operand_census({n: to_host(v)
                                     for n, v in state_of(reference, arity).items()})
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items())),
                             "reference_subnormals": census["subnormals"],
                             "reference_negative_zeros": census["negative_zeros"]})
            if difference:
                break

    after = state_of(actual, arity)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in state_names(arity)}
    still = sorted(name for name, count in moved.items() if count == 0)
    required = (state_names(arity) if value_class == "uniform"
                else BAND_MUST_MOVE + pole_names(arity))
    unmoved_required = sorted(name for name in required if moved.get(name, 0) == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    launch_ok = (launched == len(per_step)
                 and counter.named().get(family.KERNEL_NAME, 0)
                 == (len(per_step) if kernel is None else 0))
    case.update({
        "passed": bool(identical and not unmoved_required and not read_only_breaches
                       and launch_ok),
        "bit_identical": identical,
        "steps_run": len(per_step),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "arrays_compared": len(state_names(arity)),
        "pole_arrays_compared": len(pole_names(arity)),
        "arrays_that_never_moved": still,
        "movement_floor": {"class": value_class, "required": list(required),
                           "unmet": unmoved_required, "note": MOVEMENT_FLOOR_NOTE},
        "read_only_volumes_checked": sorted(read_only_snapshot(actual, pml, arity)),
        "read_only_volumes_that_moved": read_only_breaches,
        "launch_counts": {
            "launcher_reports": launched,
            "memo_proxy": counter.named(),
            "memo_proxy_total": counter.total,
            "kernel_supplied_by_gate": kernel is not None,
            "agree": bool(launch_ok),
            "expected_per_step": 1,
        },
        "seconds": time.time() - started,
    })
    return case


# ---------------------------------------------------------------------------
# The separate composition: what the fusion removes, measured
# ---------------------------------------------------------------------------

def leg_separate_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """THREE engines from one seed: array path, separate certified products, fused.

    The separate composition launches the certified ``step_D_pml_real``, the certified
    ``zero_metal_D`` where a wall is live, and the certified
    ``update_E_pml_real_dispersive``. All three engines must agree word for word;
    what differs is the LAUNCH COUNT, which is the only place the fusion is visible.
    """
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (dispersive_kernels, in_seam_coverage,  # noqa: PLC0415
                                       in_seam_passes, step_curl_kernels)
    from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|separate"
    arity = tuple(int(v) for v in spec["arity"])
    reference, ref_grid, ref_pml, host = build(spec, "uniform", case_rng(label))
    separate, sep_grid, sep_pml, _ = build(spec, "uniform", case_rng(label))
    fused, grid, pml, _ = build(spec, "uniform", case_rng(label))
    dtdx = float(grid.dt / grid.dx)
    curl_tables = step_curl_kernels.real_pml_curl_tables(sep_pml, False)
    codes = step_curl_kernels.real_curl_boundary_codes(sep_grid)
    walls = in_seam_coverage.zero_metal_axes(sep_grid)
    fused_tables = real.fused_electric_pair_tables(pml)

    separate_counter = MemoLaunchCounter()
    fused_counter = MemoLaunchCounter()
    rows: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, ref_pml)
        with separate_counter:
            for name in DRIVER_ORDER:
                if name == "step_D":
                    step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
                        separate, curl_tables, codes, dtdx)
                elif name == "zero_metal_D":
                    if any(walls):
                        in_seam_passes.run_pass("zero_metal", separate, "D",
                                                grid=sep_grid)
                elif name == "fill_symmetry_bc_D":
                    if sep_grid.has_symmetry():
                        in_seam_passes.run_pass("fill_symmetry", separate, "D",
                                                grid=sep_grid)
                elif name == "fill_folded_far_ghosts_D":
                    if sep_grid.has_symmetry() and any(
                            row is not None
                            for row in in_seam_coverage.folded_far_rows(sep_grid)):
                        in_seam_passes.run_pass("fill_folded_far", separate, "D",
                                                grid=sep_grid)
                elif name == "update_E":
                    # THE CERTIFIED DISPERSIVE ARM, launched stand-alone: this is the
                    # one place the gate runs the half the fusion absorbs, and the
                    # only route by which "the fused kernel computes what the two
                    # certified products compute" is a MEASUREMENT rather than a
                    # restatement of the array-path comparison.
                    dispersive_kernels.update_E_fused_pml_real_dispersive(
                        separate, sep_pml)
                else:
                    run_pass(name, separate, sep_pml)
        with fused_counter:
            for name in DRIVER_ORDER:
                if name == "step_D":
                    poles, counts = family.pole_bindings(fused)
                    family.launch_dispersive_fused_electric_pair(
                        fused, fused_tables,
                        step_curl_kernels.real_curl_boundary_codes(grid),
                        in_seam_coverage.zero_metal_axes(grid),
                        real.fused_electric_pair_fills(grid), dtdx, poles, counts)
                elif name not in FUSED_PASSES:
                    run_pass(name, fused, pml)
        cp.cuda.runtime.deviceSynchronize()
        against_array = compare(reference, fused, arity)
        against_separate = compare(separate, fused, arity)
        rows.append({"step": step,
                     "fused_vs_array": sum(against_array.values()),
                     "fused_vs_separate": sum(against_separate.values())})
        if against_array or against_separate:
            break

    agreed = all(row["fused_vs_array"] == 0 and row["fused_vs_separate"] == 0
                 for row in rows) and len(rows) == steps
    steps_run = max(len(rows), 1)
    return {
        "passed": bool(agreed and fused_counter.total < separate_counter.total),
        "label": spec["label"],
        "steps": len(rows),
        "per_step": rows,
        "launches": {
            "separate_total": separate_counter.total,
            "separate_named": separate_counter.named(),
            "fused_total": fused_counter.total,
            "fused_named": fused_counter.named(),
            "removed_per_step": (separate_counter.total - fused_counter.total)
            / steps_run,
        },
        "reading": ("all three engines agree word for word; the fusion is visible "
                    "ONLY in the launch count" if agreed else
                    "the three engines do not agree"),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def leg_refusal() -> Dict[str, Any]:
    """The configurations this module refuses BY NAME, with the reason recorded.

    THE PARTITION IS HALF OF THIS LEG. A configuration is only correctly refused here
    if some other product can take it, and the one that matters is the real twin: the
    two must never both admit and never both refuse a run that either could serve.
    """
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import dispersive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []

    def ask(label: str, spec: Dict[str, Any], sources: Any, expect: str,
            mutate=None) -> None:
        fields, grid, pml, _ = build(spec, "uniform", case_rng(f"refusal|{label}"))
        if mutate is not None:
            fields = mutate(fields)
        covered, why = family.covers_dispersive_fused_electric_pair(
            fields, pml, grid, sources)
        twin, twin_why = real.covers_fused_electric_pair(fields, pml, grid, sources)
        rows.append({"label": label, "covered": bool(covered), "reason": str(why),
                     "expect": expect, "real_twin_covered": bool(twin),
                     "real_twin_reason": str(twin_why),
                     "ok": (bool(covered) is (expect == "admit")
                            and not (covered and twin))})

    # NAMED, NEVER INDEXED. This leg once read ``SPECS[2]`` for its arity-zero row and
    # kept reading it after that spec left the sweep, which made a REFUSAL row ask
    # about an admitted configuration and reported it as a predicate defect.
    specs_by_label = {spec["label"]: spec for spec in SPECS}
    base_spec = dict(specs_by_label["corpus_all_periodic"])
    # THE ARITY-ZERO ROW IS BUILT HERE rather than swept: this product refuses it, so
    # it cannot be a spec, and it is the single boolean that keeps the two electric
    # products disjoint -- which makes it the one row this leg must not lose.
    zero_spec = dict(base_spec, label="degenerate_arity_zero", arity=(0, 0, 0))
    source = None

    def electric(grid):
        return VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                            amplitude=1.0)

    # 1. THE PARTITION, from both sides.
    ask("dispersive_run_declared_empty_sources", base_spec, (), "admit")
    ask("degenerate_arity_belongs_to_the_real_twin", zero_spec, (), "refuse")
    # 2. The source seam.
    fields, grid, _pml, _ = build(base_spec, "uniform", case_rng("refusal|src"))
    source = electric(grid)
    ask("undeclared_source_set", base_spec, None, "refuse")
    ask("electric_deposit_carried_by_the_repair", base_spec, (source,), "admit")
    # 3. A conductivity routes the deposit through the OTHER injection path.
    ask("conductive_engine", base_spec, (source,), "refuse",
        mutate=lambda f: _WithFlag(f, "has_conductivity", True))
    # 4. A pole count past the swept cap.
    deep = dict(base_spec)
    deep["arity"] = (dispersive_kernels.POLE_COUNT_CAP + 1,) * 3
    ask("pole_count_past_the_swept_cap", deep, (), "refuse")
    # 5. Complex storage.
    ask("complex_storage", base_spec, (), "refuse",
        mutate=lambda f: _WithFlag(f, "force_complex_fields", True))
    # 6. An off-diagonal row belongs to a different product.
    ask("off_diagonal_chi1inv", base_spec, (), "refuse",
        mutate=lambda f: _WithFlag(f, "has_offdiagonal_epsilon", True))

    return {"passed": all(row["ok"] for row in rows), "rows": rows,
            "checked": len(rows)}


class _WithFlag:
    """One engine attribute overridden, everything else the live object's."""

    __slots__ = ("_fields", "_name", "_value")

    def __init__(self, fields, name, value):
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_value", value)

    def __getattr__(self, name):
        if name == object.__getattribute__(self, "_name"):
            return object.__getattribute__(self, "_value")
        return getattr(object.__getattribute__(self, "_fields"), name)


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _arity(spec) -> Tuple[int, int, int]:
    return tuple(int(v) for v in spec["arity"])


def _driven(component_index: int, minimum: int = 1):
    return lambda spec: _arity(spec)[component_index] >= minimum


def _any_ghost(spec) -> bool:
    return bool(spec.get("symmetry"))


def _folded_axis(letter: str):
    return lambda spec: letter in {axis for axis, _phase in spec.get("symmetry", ())}


def _near_y_ghost_of_Ex_and_driven(spec) -> bool:
    """Is the ``gx_ny_*`` block -- Ex's NEAR ghost on y -- both EMITTED and REACHED?

    THE NEEDLE IS PRESENT IN THE TEXT ON EVERY ROW and its guard is a RUNTIME flag
    (``near_y && j == 0``), so a fold on some OTHER axis compiles the mutation and
    never executes it. Scoring it there would record a defect as uncaught when the
    defect was not present, which is exactly the vacuity this gate is shaped against.
    Measured on the first smoke: ``fold_x_deep_pml`` armed all three ghost needles and
    diverged on none of them, because ``near_y`` is 0 on an x-only fold.
    """
    return _folded_axis("Y")(spec) and _arity(spec)[0] >= 1


def _wall(axis: str):
    index = "xyz".index(axis)
    return lambda spec: (spec["boundaries"][index] == "metallic"
                         and axis.upper() not in
                         {a for a, _p in spec.get("symmetry", ())})


def _anisotropic(spec) -> bool:
    return spec.get("epsilon") == "anisotropic"


def _always(spec) -> bool:
    return True


#: DEVICE-TEXT needles. Each is (label, (old, new, count), must_be_caught, applies).
SOURCE_MUTATIONS: Tuple[Tuple[str, Tuple[str, str, int], Optional[bool],
                              Callable[[Dict[str, Any]], bool]], ...] = (
    # --- THE GHOST CHAIN, and this is the leg this gate exists for ------------
    # The imaged ghost must subtract the DESTINATION's poles. Re-using the source
    # thread's registers is the obvious port and is a different field on every imaged
    # plane. Neither sibling backend can arm this: both refuse a folded grid.
    ("ghost_chain_reads_the_source_poles",
     ("gx_ny_p = gx_ny_p - P_Ex_0[gx_ny_i];",
      "gx_ny_p = gx_ny_p - P_Ex_0[idx];", 1), True,
     _near_y_ghost_of_Ex_and_driven),
    # The ghost's inverse epsilon is the destination's too.
    ("ghost_inverse_epsilon_read_at_the_source",
     ("float gx_ny_s = gx_ny_p * inv_eps_Ex[gx_ny_i];",
      "float gx_ny_s = gx_ny_p * inv_eps_Ex[idx];", 1), True,
     _near_y_ghost_of_Ex_and_driven),
    # The chain must be formed at ALL at a ghost: dropping it leaves the ghost's E
    # computed from a displacement no polarization was removed from.
    ("ghost_chain_dropped",
     ("float gx_ny_p = gx_ny_v;\n            gx_ny_p = gx_ny_p - P_Ex_0[gx_ny_i];",
      "float gx_ny_p = gx_ny_v;", 1), True, _near_y_ghost_of_Ex_and_driven),
    # --- THE CHAIN'S ORDER AND SHAPE -----------------------------------------
    # SEQUENTIAL, LEFT TO RIGHT: ((D - P0) - P1). float32 addition is not associative,
    # so summing first is a different number -- and only at two poles or more, which
    # is why the applicability predicate asks for two.
    ("chain_summed_then_subtracted",
     ("    s_x = s_x - P_Ex_0[idx];\n    s_x = s_x - P_Ex_1[idx];",
      "    s_x = s_x - (P_Ex_0[idx] + P_Ex_1[idx]);", 1), True, _driven(0, 2)),
    ("chain_order_reversed",
     ("    s_x = s_x - P_Ex_0[idx];\n    s_x = s_x - P_Ex_1[idx];",
      "    s_x = s_x - P_Ex_1[idx];\n    s_x = s_x - P_Ex_0[idx];", 1),
     True, _driven(0, 2)),
    # A pole dropped from the chain entirely.
    ("chain_drops_its_last_pole",
     ("    s_y = s_y - P_Ey_1[idx];", "", 1), True, _driven(1, 2)),
    # THE CHAIN'S COMPONENT. Ex's chain is Ex's poles, not Ez's.
    ("chain_takes_another_components_pole",
     ("    s_x = s_x - P_Ex_0[idx];", "    s_x = s_x - P_Ez_0[idx];", 1),
     True, lambda spec: _arity(spec)[0] >= 1 and _arity(spec)[2] >= 1),
    # --- THE SEAM ------------------------------------------------------------
    # The chain opens on the REGISTER the curl left, not on a reload.
    ("chain_opens_on_the_pre_clear_register",
     ("    float s_x = d_x;", "    float s_x = pre_x;", 1), True,
     lambda spec: _arity(spec)[0] >= 1 and (_wall("y")(spec) or _wall("z")(spec))),
    ("seam_takes_the_wrong_component",
     ("    float s_x = d_x;", "    float s_x = d_y;", 1), True, _driven(0)),
    # D on the LEFT of the inverse-epsilon multiply is transcription discipline
    # (IEEE multiply commutes), so this is a NULL that must come back UNCAUGHT --
    # paired with the component swap below, which must be caught. Without the pair,
    # "the operand order is inert" would be a claim about a leg nobody showed
    # could fail.
    ("inverse_epsilon_operand_order_swapped",
     ("    float src_x = s_x * inv_eps_Ex[idx];",
      "    float src_x = inv_eps_Ex[idx] * s_x;", 1), False, _driven(0)),
    ("inverse_epsilon_component_swapped",
     ("    float src_x = s_x * inv_eps_Ex[idx];",
      "    float src_x = s_x * inv_eps_Ez[idx];", 1), True,
     lambda spec: _anisotropic(spec) and _arity(spec)[0] >= 1),
    # --- THE INHERITED CARRY, re-armed on this family -------------------------
    ("wall_clear_uses_the_b_diagonal",
     ("if (wall_x && i == 0) { clr_y = 1; clr_z = 1; }",
      "if (wall_x && i == 0) { clr_x = 1; }", 1), True, _wall("x")),
    ("wall_clear_misses_the_register",
     ("if (own_y && clr_y) { d_y = 0.0f; Dy[idx] = d_y; }",
      "if (own_y && clr_y) { Dy[idx] = 0.0f; }", 1), True, _wall("x")),
    # THE SUB-LATTICE SHADOW, and this defect exists only because the fusion put both
    # halves in one scope: the curl's kms is INTEGER and the constitutive's
    # HALF-INTEGER, and letting the certified name shadow is a half-cell error in the
    # absorber profile -- converged, smooth and wrong. ANCHORED ON THE WHOLE OWN-CELL
    # STATEMENT, not on the coefficient pair: that pair also appears on the three near
    # ghost destinations that keep this thread's index, and a needle matching four
    # lines would arm whichever the replace reached first -- a leg reporting a verdict
    # about a line nobody chose.
    ("constitutive_sublattice_shadow",
     ("constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_half_x[i]);",
      "constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);", 1),
     True, _always),
    # THE GHOST'S COEFFICIENT ROW is the DESTINATION's: a far ghost sits at stored
    # n - 1 on the axis update_E indexes the component on, so reusing the source's
    # pair applies the reflect row's absorber profile to the top plane.
    ("far_ghost_takes_the_source_coefficient_row",
     ("constitutive_apply(Ex, f_w_Ex, gx_x_i, gx_x_s, "
      "kps_x[nx - 1], kms_half_x[nx - 1]);",
      "constitutive_apply(Ex, f_w_Ex, gx_x_i, gx_x_s, kps_x[i], kms_half_x[i]);", 1),
     True, _folded_axis("X")),
    ("curl_shift_direction_flipped",
     ("float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);",
      "float sf = shift_up(Hz, idx, j, ny, sy, bc_y);", 1), True, _always),
)


def _reversed_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {component: list(reversed(arrays)) for component, arrays in plan.items()}


def _rotated_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {"Ex": list(plan["Ez"]), "Ey": list(plan["Ex"]), "Ez": list(plan["Ey"])}


def _own_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {component: list(arrays) for component, arrays in plan.items()}


HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], Optional[bool],
                            Callable[[Dict[str, Any]], bool]], ...] = (
    # THE POLE PLAN IS A LAUNCH ARGUMENT, and this is the door the shipped launcher
    # opens for exactly this leg. Scored at arity >= 2 on some component, where a
    # reordering can be a different float32 number at all.
    ("pole_plan_reversed", {"plan_patch": _reversed_plan}, True,
     lambda spec: max(_arity(spec)) >= 2),
    # The chain belongs to its OWN component.
    ("pole_plan_rotated_across_components", {"plan_patch": _rotated_plan}, True,
     lambda spec: len(set(_arity(spec))) == 1 and min(_arity(spec)) >= 1),
    # A NULL: the plan the launcher would have resolved, handed in explicitly, must
    # change nothing. Without it "caught" above could mean the harness diverges on any
    # plan object at all.
    ("pole_plan_resolved_explicitly", {"plan_patch": _own_plan}, False, _always),
    ("curl_takes_the_half_integer_lattice",
     {"tables_patch": base._half_integer_curl_tables}, True, _always),
    ("constitutive_takes_the_integer_lattice",
     {"tables_patch": base._integer_constitutive_tables}, True, _always),
    ("curl_takes_its_own_integer_lattice",
     {"tables_patch": base._integer_curl_tables}, False, _always),
    ("walls_dropped", {"walls_patch": base._walls_dropped}, True,
     lambda spec: any(_wall(axis)(spec) for axis in "xyz")),
    ("fills_dropped", {"fills_patch": base._fills_dropped}, True, _any_ghost),
    ("fill_phases_flipped", {"fills_patch": base._fill_phases_flipped}, True,
     _any_ghost),
)


def compile_source(source: str, options: Sequence[str]):
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415
    return cp.RawKernel(source, family.KERNEL_NAME, options=tuple(options))


_verdict = base._verdict


def leg_guard_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The SHIPPED source with the contraction guard removed must DIVERGE.

    A guard control that came back IDENTICAL would mean this comparison cannot see a
    rounding change at all, and every "identical" above would be worth much less.
    """
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415

    source = family.dispersive_fused_electric_pair_source(_arity(spec))
    kernel = compile_source(source, ())
    result = run_case(spec, "uniform", steps, kernel=kernel,
                      label_suffix="|unguarded")
    diverged = not result.get("bit_identical", True)
    return {
        "passed": bool(diverged and result.get("launch_counts", {}).get("agree")),
        "guard_removed": list(GUARD_OPTIONS),
        "diverged": diverged,
        "first_divergence": result.get("first_divergence"),
        "differing_words": result.get("differing_words"),
        "reading": ("the comparison IS sensitive to float32 contraction"
                    if diverged else
                    "the unguarded source did NOT diverge; this comparison may not "
                    "see a rounding change at all"),
        "case": result,
    }


# ---------------------------------------------------------------------------
# The deposit
# ---------------------------------------------------------------------------

def leg_deposit(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """Complete driver steps with a real ELECTRIC source inside the seam.

    WHAT THE DECLARED REPAIR IS WORTH, on a DISPERSIVE constitutive half -- which is
    the configuration ``deposit_repair`` has never been gated against on this track.
    ``deposit_repair.apply`` recomputes ``fw_fresh`` through
    ``Fields.displacement_minus_polarization``, which IS this half's source, and it is
    entitled to the pole arrays it reads because ``update_P`` runs AFTER ``update_E``
    (driver.py:3315).

    THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    ``run_dispersive_fused_electric_pair`` directly would test the kernel and skip the
    wiring, and the wiring is what this leg exists for: ``arms.plan_step(..., fuse=
    True)`` is the only route by which ``fused_pairs._install_fused_pair`` brackets the
    launch, and a product whose ``FUSED_PRODUCTS`` row were missing would fail here
    rather than pass on an unbracketed launch.

    THE NULL CONTROL is the same composition asked with an EMPTY source list, injected
    anyway: it MUST diverge, or the bracket is not what made the repaired row pass.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    started = time.time()
    arity = _arity(spec)
    rows: List[Dict[str, Any]] = []
    for repair in (True, False):
        label = f"{spec['label']}|deposit|{'repair' if repair else 'null'}"
        row: Dict[str, Any] = {"label": spec["label"], "repair_wired": repair}
        reference, ref_grid, ref_pml, _ = build(spec, "uniform", case_rng(label))
        actual, grid, pml, _ = build(spec, "uniform", case_rng(label))
        drift = compare(reference, actual, arity)
        if drift:
            row.update(passed=False, why=f"the two builds differ: {drift}")
            rows.append(row)
            continue
        reference_sources = (base._electric_source(reference),)
        actual_sources = (base._electric_source(actual),)
        index = deposit_repair._deposit_index(actual_sources[0])  # noqa: SLF001
        if index is None or len(
                deposit_repair.in_seam_sources(actual_sources, "D")) != 1:
            row.update(passed=False,
                       why="the source built here is not an in-seam D deposit")
            rows.append(row)
            continue
        row["deposit_points"] = int(cp.asarray(index[0]).size)
        row["deposit_magnitude"] = base.deposit_magnitude(
            actual, base._electric_source(actual), steps, float(grid.dt))
        if row["deposit_magnitude"] <= 0.0:
            # A SOURCE THAT DEPOSITS NOTHING makes the repaired and unrepaired walks
            # agree with each other and with a no-op, so the null control would be
            # vacuous for a reason about the fixture.
            row.update(passed=False, why="this source deposits exactly zero")
            rows.append(row)
            continue

        composed = arms.plan_step(actual, pml, grid,
                                  sources=(actual_sources if repair else ()),
                                  fuse=True)
        leading = composed.plans.get("step_D")
        trailing = composed.plans.get("update_E")
        installed = (type(leading).__name__, type(trailing).__name__)
        row["installed_plans"] = list(installed)
        row["selected"] = {slot: composed.selected.get(slot)
                           for slot in ("step_D", "update_E")}
        expected = (("LeadingRepairPlan", "TrailingRepairPlan") if repair
                    else ("CudaFusedPairPlan", "NoopPlan"))
        row["installed_as_expected"] = installed == expected
        if leading is None or trailing is None:
            row.update(passed=False, why="the composer did not fuse the seam",
                       refusals=[reason for key, value in composed.reasons.items()
                                 if key.startswith("fused_pair")
                                 for reason in value])
            rows.append(row)
            continue

        dt = float(grid.dt)
        per_step: List[Dict[str, Any]] = []
        for step in range(1, steps + 1):
            when = (step - 1) * dt
            base._withdraw(reference_sources, reference)
            for name in DRIVER_ORDER:
                run_pass(name, reference, ref_pml)
                if name == "step_D":
                    for source in reference_sources:
                        source.inject(reference, when + 0.5 * dt)
            base._withdraw(actual_sources, actual)
            for name in DRIVER_ORDER:
                if name == "step_D":
                    leading.run()
                    for source in actual_sources:
                        source.inject(actual, when + 0.5 * dt)
                elif name == "update_E":
                    trailing.run()
                elif name not in FUSED_PASSES:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual, arity)
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items()))})
            if difference:
                break
        identical = (len(per_step) == steps
                     and all(r["differing_words"] == 0 for r in per_step))
        repairs = int(getattr(leading, "repairs", 0))
        row.update({
            "passed": bool(row["installed_as_expected"]
                           and (identical and repairs > 0 if repair
                                else not identical and repairs == 0)),
            "bit_identical": identical,
            "deposit_points_repaired": repairs,
            "steps_run": len(per_step),
            "first_divergence": next((r["step"] for r in per_step
                                      if r["differing_words"]), None),
            "per_step": per_step,
        })
        rows.append(row)
    return {
        "passed": all(row.get("passed") for row in rows) and len(rows) == 2,
        "label": spec["label"],
        "rows": rows,
        "reading": ("the bracket is what makes the deposit case pass: unbracketed, "
                    "the same composition diverges"),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

save = base.save


def warm_memo(specs_by_label: Dict[str, Any]) -> Dict[str, Any]:
    """Compile every arity this run will launch, OUTSIDE any counted region.

    ``MemoLaunchCounter`` proxies the shipped compile memo, so a kernel first compiled
    inside a counted region is launched and NOT counted, and the case's launch total
    comes back one short of its step budget -- a failure that looks like a fallback
    and is not one. THE MEMO KEY IS THE SOURCE, so one compile per ARITY covers every
    spec that carries it.
    """
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415

    started = time.time()
    arities = sorted({_arity(spec) for spec in specs_by_label.values()})
    for arity in arities:
        family._get_kernel(arity)
    return {"arities_compiled": [list(a) for a in arities],
            "seconds": round(time.time() - started, 2)}


def run_product(results, out_path, specs, classes, steps):
    cases: List[Dict[str, Any]] = []
    total = len(specs) * len(classes)
    for number, spec in enumerate(specs):
        for index, value_class in enumerate(classes):
            case = run_case(spec, value_class, steps)
            cases.append(case)
            results["product"] = cases
            save(results, out_path)
            log(f"  case {number * len(classes) + index + 1}/{total} "
                f"{spec['label']}|{value_class} arity={case.get('arity')}: "
                f"{'OK' if case.get('passed') else 'FAIL'} "
                f"words={case.get('differing_words')} "
                f"poles={case.get('live_pole_words')} "
                f"({case.get('seconds', 0):.1f} s)")
    return cases


def run_mutations(results, out_path, steps, specs_by_label):
    from meep_gpu.cuda_kernels import dispersive_fused_electric_pair as family  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for label, (old, new, count), must, applies in SOURCE_MUTATIONS:
        scored, caught, legs, unmeasured = 0, 0, [], []
        for spec_label in MUTATION_SPEC_LABELS:
            spec = specs_by_label[spec_label]
            if not applies(spec):
                legs.append({"spec": spec_label, "applicable": False})
                continue
            source = family.dispersive_fused_electric_pair_source(_arity(spec))
            mutated, hits = needle(source, old, new, count)
            if hits != count:
                legs.append({"spec": spec_label, "applicable": True,
                             "needle_hits": hits, "armed": False})
                continue
            kernel = compile_source(mutated, GUARD_OPTIONS)
            case = run_case(spec, "uniform", steps, kernel=kernel,
                            label_suffix=f"|{label}")
            # A CASE THAT NEVER REACHED A COMPARISON IS NOT A VERDICT. A predicate
            # refusal or a build drift returns before the stepping loop and carries no
            # ``bit_identical`` at all; defaulting that to "identical" would record a
            # defect as UNCAUGHT when nothing ran. It is recorded as unmeasured and it
            # FAILS the leg, because a mutation table with an unmeasurable row is a
            # table whose denominator is wrong.
            if "bit_identical" not in case:
                unmeasured.append(spec_label)
                legs.append({"spec": spec_label, "applicable": True, "armed": True,
                             "measured": False, "why": case.get("why")})
                continue
            diverged = not case["bit_identical"]
            scored += 1
            caught += int(diverged)
            legs.append({"spec": spec_label, "applicable": True, "armed": True,
                         "measured": True, "diverged": diverged,
                         "first_divergence": case.get("first_divergence"),
                         "differing_words": case.get("differing_words")})
        verdict = ("UNMEASURABLE LEG" if unmeasured
                   else _verdict(caught, scored, must))
        rows.append({"mutation": label, "must_be_caught": must, "scored": scored,
                     "caught": caught, "verdict": verdict,
                     "unmeasured_specs": unmeasured, "legs": legs})
        results["source_mutations"] = rows
        save(results, out_path)
        log(f"  mutation {label}: {_verdict(caught, scored, must)} "
            f"({caught}/{scored})")
    return rows


def run_host_mutations(results, out_path, steps, specs_by_label):
    rows: List[Dict[str, Any]] = []
    for label, hooks, must, applies in HOST_MUTATIONS:
        scored, caught, legs, unmeasured = 0, 0, [], []
        for spec_label in MUTATION_SPEC_LABELS:
            spec = specs_by_label[spec_label]
            if not applies(spec):
                legs.append({"spec": spec_label, "applicable": False})
                continue
            case = run_case(spec, "uniform", steps, label_suffix=f"|{label}", **hooks)
            # Same rule as the device-text runner: a case that never reached a
            # comparison is recorded as unmeasured and fails the leg, never defaulted
            # to "identical".
            if "bit_identical" not in case:
                unmeasured.append(spec_label)
                legs.append({"spec": spec_label, "applicable": True,
                             "measured": False, "why": case.get("why")})
                continue
            diverged = not case["bit_identical"]
            scored += 1
            caught += int(diverged)
            legs.append({"spec": spec_label, "applicable": True, "measured": True,
                         "diverged": diverged,
                         "first_divergence": case.get("first_divergence")})
        verdict = ("UNMEASURABLE LEG" if unmeasured
                   else _verdict(caught, scored, must))
        rows.append({"mutation": label, "must_be_caught": must, "scored": scored,
                     "caught": caught, "verdict": verdict,
                     "unmeasured_specs": unmeasured, "legs": legs})
        results["host_mutations"] = rows
        save(results, out_path)
        log(f"  host mutation {label}: {_verdict(caught, scored, must)} "
            f"({caught}/{scored})")
    return rows


def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    product = results.get("product", [])
    source_rows = results.get("source_mutations", [])
    host_rows = results.get("host_mutations", [])
    unarmed = [row["mutation"] for row in source_rows + host_rows
               if row["scored"] == 0]
    failed_mutations = [row["mutation"] for row in source_rows + host_rows
                        if row["verdict"] in ("UNCAUGHT", "PARTIAL", "NULL VIOLATED",
                                              "NO LEGS", "UNMEASURABLE LEG")]
    legs = {
        "driver_order": results.get("driver_order", {}).get("passed"),
        "product": all(case.get("passed") for case in product) and bool(product),
        "separate_control": all(row.get("passed")
                                for row in results.get("separate_control", [])),
        "refusal": results.get("refusal", {}).get("passed"),
        "deposit": all(row.get("passed") for row in results.get("deposit", [])),
        "guard_control": results.get("guard_control", {}).get("passed"),
        "source_mutations": not failed_mutations and not unarmed,
    }
    arities = sorted({tuple(case["arity"]) for case in product if "arity" in case})
    return {
        "legs": legs,
        "released": all(bool(value) for value in legs.values()),
        "cases": len(product),
        "cases_passed": sum(1 for case in product if case.get("passed")),
        "steps_per_case": steps,
        "arities_swept": [list(a) for a in arities],
        "mutations_scored": sum(row["scored"] for row in source_rows + host_rows),
        "mutations_caught": sum(row["caught"] for row in source_rows + host_rows),
        "mutations_never_armed": unarmed,
        "mutations_that_failed": failed_mutations,
        "denominators": {
            "board_cell": "D_to_E (cuda_curl/PML, cuda_dispersive/dispersive)",
            "corpus_rows_on_the_cell": 7,
            "corpus_rows_with_an_electric_deposit": 7,
            "corpus_rows_folded": 4,
        },
        "what_this_does_not_claim": [
            "no throughput or dispatch claim of any kind",
            "no verdict about a complex, cylindrical, conductive, BFAST, special-kz, "
            "off-diagonal or nonlinear run -- each is refused BY NAME and none swept",
            "no verdict about the non-dispersive twin, which has its own gate",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="FILE to write gate.json to")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated spec labels, for a smoke run")
    arguments = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)) or ".", exist_ok=True)
    specs = SPECS
    if arguments.only:
        wanted = {name.strip() for name in arguments.only.split(",")}
        specs = tuple(spec for spec in SPECS if spec["label"] in wanted)
    specs_by_label = {spec["label"]: spec for spec in SPECS}

    results: Dict[str, Any] = {
        "gate": "cuda_dispersive_fused_electric_pair",
        "subject": "meep_gpu/cuda_kernels/dispersive_fused_electric_pair.py",
        "kernel": "dispersive_fused_electric_pair_pml_real",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of dispersive_fused_electric_pair_pml_real "
                     "leave every stored volume -- the pole buffers included -- "
                     "byte-identical to stepping's step_D -> the two mirror fills -> "
                     "zero_metal_D -> pole-aware update_E inside a complete driver "
                     "step, including with an electric source deposited in the seam "
                     "and with the chain re-formed at every imaged ghost?"),
        "subnormal_policy": arguments.subnormal_policy,
        "steps": arguments.steps,
        "specs": [spec["label"] for spec in specs],
        "movement_floor_note": MOVEMENT_FLOOR_NOTE,
    }
    if cp is None:
        log("CuPy is not importable on this host; this gate needs a device")
        results["status"] = "refused: no CuPy"
        save(results, arguments.out)
        return 2

    if arguments.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip wraps
    # it, so it records the option tuple NVRTC was really given.
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
        arguments.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, arguments.out)

    # WARM THE UNION, not just the product sweep. Every leg drives kernels through the
    # same shared memo, and a kernel first compiled inside a counted region is launched
    # and NOT counted -- so a spec only a mutation or deposit leg visits has to be
    # warmed here or that leg's launch total comes back short. THE MEMO KEY IS THE
    # SOURCE, so warming is per ARITY rather than per spec.
    results["memo_warm"] = warm_memo(specs_by_label)
    save(results, arguments.out)

    log("driver order")
    results["driver_order"] = base.leg_driver_order()
    save(results, arguments.out)

    log(f"product: {len(specs)} specs x {len(VALUE_CLASSES)} value classes")
    run_product(results, arguments.out, specs, VALUE_CLASSES, arguments.steps)

    log("separate control")
    results["separate_control"] = [
        leg_separate_control(specs_by_label[label], min(arguments.steps, 20))
        for label in SEPARATE_CONTROL_LABELS if label in specs_by_label]
    save(results, arguments.out)

    log("refusals")
    results["refusal"] = leg_refusal()
    save(results, arguments.out)

    log("deposit")
    results["deposit"] = [
        leg_deposit(specs_by_label[label], min(arguments.steps, 24))
        for label in DEPOSIT_SPEC_LABELS if label in specs_by_label]
    save(results, arguments.out)

    log("guard control")
    results["guard_control"] = leg_guard_control(
        specs_by_label["corpus_all_periodic"], min(arguments.steps, 20))
    save(results, arguments.out)

    log(f"source mutations: {len(SOURCE_MUTATIONS)}")
    run_mutations(results, arguments.out, min(arguments.steps, 20), specs_by_label)

    log(f"host mutations: {len(HOST_MUTATIONS)}")
    run_host_mutations(results, arguments.out, min(arguments.steps, 20),
                       specs_by_label)

    results["summary"] = summarize(results, arguments.steps)
    # THE LEDGER'S OWN KEY, and it is spelled here rather than left to a reader of
    # ``summary``: ``rebind_cuda_welds`` binds a weld only where every canonical policy
    # leg carries ``canonical_verdict.released`` true, and it treats a MISSING key as
    # "no verdict" rather than as a failure. A gate that reported its release under
    # some other name would be silently unbindable -- the artifact would look released
    # and nothing in ``fingerprints.json`` would ever point at it.
    results["canonical_verdict"] = {
        "read_from": "summary.released",
        "released": bool(results["summary"]["released"]),
        "reasons": [f"leg {name} did not pass"
                    for name, value in results["summary"]["legs"].items()
                    if not value],
    }
    save(results, arguments.out)
    log(f"released={results['summary']['released']} "
        f"cases={results['summary']['cases_passed']}/{results['summary']['cases']} "
        f"mutations={results['summary']['mutations_caught']}/"
        f"{results['summary']['mutations_scored']}")
    log(f"wrote {arguments.out}")
    return 0 if results["summary"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
