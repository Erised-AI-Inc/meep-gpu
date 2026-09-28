"""Byte gate for the Metal Dcyl |m| >= 1 fused ELECTRIC pair: ``step_D`` into ``update_E``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.cylindrical_fused_electric_pair.cylindrical_fused_electric_pair_coverage`
admits, the host prefix pre-pass followed by ONE dispatch of
``cylindrical_fused_electric_pair_step`` leaves the engine in a state that is
BIT-IDENTICAL, PER COMPLETE STEP, to

  * the CuPy-free NumPy array path (``stepping.step_D`` / ``zero_metal_D`` /
    ``stepping.update_E``, plus the rest of the composer's live pass set), and
  * the two SEPARATELY CERTIFIED Metal products it replaces —
    ``cylindrical_complex.plan_cylindrical_complex_pml_curl`` on ``step_D`` and
    ``...plan_cylindrical_complex_constitutive`` on ``E``, with ``zero_metal_D`` left
    on the HOST between them,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere.

WHY THIS PRODUCT EXISTS AT ALL, and it is the reason this gate carries two legs the
magnetic twin does not. This is the LARGEST cell on the Metal board — 16 reachable
D->E seam-instances — and until this round it carried the verdict ``UNFUSABLE ON
METAL``: 33 pointers against a 30-pointer ceiling, over by THREE. Its magnetic twin
sits at EXACTLY 31 bindings with zero headroom, and the E side's three inverse-epsilon
volumes are what push it over. The signature moved, not the ceiling: the curl half's
six read-only PML coefficient vectors ride in ONE buffer with six element offsets in
the ``Params`` struct the eight non-pointer arguments already ride in.

  * ``binding_ceiling`` compiles FOUR signatures and locates the ceiling by bisection,
    so 31 is a number this host returned rather than one this file cites;
  * ``pack_identity`` measures the packed buffer against the six separate vectors,
    bytes and reads, and the eleven OFFSET mutations are what make a wrong offset a
    caught defect rather than a plausible absorber.

LEGS
  0  expansion            the arm is bound from THIS family's own probe artifact
  1  binding_ceiling      the 41-binding separate-scalar, the 34-binding UNPACKED
                          and a synthetic 32 must all FAIL; the shipped 29 must
                          COMPILE, LAUNCH and read every Params field back — the six
                          pack offsets included — and the ceiling is BISECTED here
  2  pack_identity        one buffer reads as six vectors, on the host and on the
                          device
  3  transcription        both halves are the certified emitters' own bytes,
                          differing only in the three seam lines, and the pack has
                          not reached the arithmetic
  4  product              complete steps, per-step byte compare, launch + prefix-sync
                          counters, movement, subnormal census
  5  value_class          the same on the complex +-0 lattice, plus the band ladder
  6  separate_control     the two ALREADY CERTIFIED products as separate dispatches
                          with the wall clear on the host between them
  7  deposit              a REAL electric ``VolumeSource`` inside the seam, carried
                          through the shipped repair bracket
  8  deposit_null_control the same launch UNBRACKETED, which MUST diverge
  9  byte_neutral_control the register-vs-reload edit must NOT diverge
 10  mutation             armed defects — shader edits, host lattice swaps and PACK
                          OFFSET corruptions — each of which must diverge
 11  refusal              the source seam, the STORAGE split against the real m = 0
                          product (a complex64 m = 0 run is ADMITTED and bracketed
                          since 2026-09-04), the geometry split, by name
 12  disarm               the same harness, shipped bytes, must not diverge

The m = 0 arm (2026-09-04) is driven on three of the product rows, its own armed
defects (the post-add dropped / moved / over-carried / FOLDED into the curl, the Dy
clear dropped, the seam reading the pre-post-add word) are scored on the m = 0 case,
the two arm swaps are scored on both, the in-kernel ``4.0f * dtdx`` spelling is a
CONFIRMED null, and the deposit leg runs on the m = 0 row as well as the |m| = 1 one.

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
    coefficient_pack, cylindrical_complex,
    cylindrical_fused_electric_pair as family,
    cylindrical_fused_magnetic_pair as twin,
    launch as metal_launch, shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

#: The budget every case runs. Twelve, matching every other fused pair's.
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

#: Every stored volume a complete step can touch.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: name -> the grid this case builds. THE ROWS ARE THE MAGNETIC TWIN'S, deliberately:
#: the two products are the same two halves on the two seams, so a divergence between
#: the gates' verdicts is a fact about the seam and not about the fixtures.
#:
#:   THE |m| ARM      ``M_ONE`` emits the on-axis increment; ``M_MANY`` emits the
#:                    hold over ``zrows`` rows and NO increment. DIFFERENT BODIES.
#:   z TERMINATION    a metallic z serves an exact 0.0 past the wall, emits the
#:                    ownership mask's z clauses AND is the only axis that can emit a
#:                    wall clear; a periodic z wraps and emits none of the three.
#:   zrows            ``accurate_fields_near_cylorigin`` selects 1 instead of |m| at
#:                    the same m — a RUNTIME uniform, so it varies the Params record
#:                    rather than the source, which puts that field of the struct
#:                    under test alongside the six new offset fields.
#:   m = 0            the THIRD body (2026-09-04): no i*m/r block, no increment, and
#:                    the m = 0 axis rules on the D side — ``Dz[r=0] += axis_coef *
#:                    Hp[r=0]`` as a post-add, then ``Dy[r=0] = 0``. Its rows are the
#:                    corpus row's z termination, the other one, and a non-power-of-
#:                    two Courant so the new scalar is exercised where scaling is
#:                    inexact.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m1_z_metallic", dict(m=1, z_kind="metallic")),
    ("m_minus1_z_metallic", dict(m=-1, z_kind="metallic")),
    ("m1_z_periodic", dict(m=1, z_kind="periodic")),
    ("m3_z_metallic", dict(m=3, z_kind="metallic")),
    ("m2_z_periodic", dict(m=2, z_kind="periodic")),
    ("m3_accurate_z_metallic",
     dict(m=3, z_kind="metallic", accurate=True, courant=0.25)),
    ("m0_z_metallic", dict(m=0, z_kind="metallic")),
    ("m0_z_periodic", dict(m=0, z_kind="periodic")),
    ("m0_odd_courant_z_metallic",
     dict(m=0, z_kind="metallic", courant=0.2718281828)),
)

#: The case the |m| = 1 mutations are armed on. z METALLIC, so the ownership mask's
#: z clauses and the wall clear are BOTH emitted alongside the axis increment.
M_ONE_CASE = "m1_z_metallic"

#: The case the |m| >= 2 mutations are armed on — the near-axis hold, which the
#: ``M_ONE`` body does not emit at all.
M_MANY_CASE = "m3_z_metallic"

#: The case the m = 0 mutations are armed on — the ``M_ZERO`` body, which emits the
#: D-side post-add and the ``Dy`` clear and none of the |m| >= 1 machinery.
M_ZERO_CASE = "m0_z_metallic"

#: The rows legs ``refusal`` and ``deposit`` build their questions on. The deposit
#: leg runs on BOTH arms that reach a corpus row carrying an in-seam electric source:
#: the |m| = 1 rows and, since 2026-09-04, the m = 0 row
#: (``dipole_in_vacuum_cyl_off_axis.py``, two electric sources).
REFUSAL_CASE: Dict[str, Any] = dict(m=1, z_kind="metallic")
DEPOSIT_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m1_z_metallic", dict(m=1, z_kind="metallic")),
    ("m0_z_metallic", dict(m=0, z_kind="metallic")),
)

#: Bound by ``leg_expansion`` before any other leg runs.
EXPANSION: Optional[str] = None
PROBE_ARTIFACT: Optional[str] = None


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
          value_class: str = values.UNIFORM, scale: float = 1.0,
          ) -> Tuple[Any, Any]:
    """One seeded engine. Called twice (or three times) per case, identically.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the two halves' own gates use. Only the field state is reseeded, and
    identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.
    """
    fields, pml = matrix.cylindrical(**dict(keywords))
    if value_class == values.PM_ZERO_LATTICE:
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
        if array is not None:
            array[...] = ((rng.standard_normal(fields.grid.shape)
                           + 1j * rng.standard_normal(fields.grid.shape))
                          * 0.37 * scale).astype(np.complex64)
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
    silently leave ``zero_metal_D`` running on the host on top of the clear the kernel
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
    """THE EMPTY CENSUS, TURNED INTO A MEASUREMENT."""
    rows: List[Dict[str, Any]] = []
    scale = 1.0
    for rung in range(budget):
        scale *= 1e-9
        fields, pml = build(keywords, seed + rung, scale=scale)
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


def run_case(keywords: Mapping[str, Any], seed: int, steps: int,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             params: Optional[Callable[[Any], Any]] = None,
             value_class: str = values.UNIFORM,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(keywords, seed, value_class)
    actual, actual_pml = build(keywords, seed, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    record = None
    if params is not None:
        # The HOST mutation seam: the defect is in the OFFSETS, so it cannot be
        # reached from the kernel source at all. Building the shipped plan first is
        # what lets a leg read the layout the packer produced.
        probe = family.plan_cylindrical_fused_electric_pair(
            actual, actual_pml, sources=(), residency=Residency())
        assert probe is not None, "the params mutation has no shipped plan to read"
        record = params(probe)
    plan = family.plan_cylindrical_fused_electric_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions,
        params=record)
    if plan is None:
        reasons = family.cylindrical_fused_electric_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False,
                "reason": "the cylindrical complex fused electric pair was refused",
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
    # THE PREFIX PRE-PASS IS PART OF THE CONTRACT, not diagnostics: a plan that
    # skipped it would step every launch after the first from a stale prefix, which is
    # smooth and wrong. One refresh per run, asserted rather than trusted.
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.prefix_syncs == len(per_step))
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
        "prefix_syncs": plan.prefix_syncs,
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "bcz": plan.bcz, "m_arm": plan.m_arm, "zero_rows": plan.zero_rows,
        "zero_metal": list(plan.zero_metal),
        "pack_layout": plan.pack_layout.describe(),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         ) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, same state.

    THIS IS ALSO WHERE THE PACK IS MEASURED AGAINST THE SHIPPED KERNELS: the separate
    side binds the six PML coefficient vectors as six pointers through the CERTIFIED
    curl kernel, the fused side reads the same six through one packed buffer, and the
    two must agree word for word at every complete step.

    NOTE WHAT THE COUNTS SHOW AND WHAT THEY DO NOT. Both sides pay ONE prefix pre-pass
    per step; the fusion does not remove it, and this leg records it on both sides so
    no reader can mistake the dispatch saving for a scan saving.
    """
    reference, reference_pml = build(keywords, seed)
    separate, separate_pml = build(keywords, seed)
    fused, fused_pml = build(keywords, seed)

    separate_residency = Residency()
    curl = cylindrical_complex.plan_cylindrical_complex_pml_curl(
        separate, separate_pml, "step_D", separate_residency)
    electric = cylindrical_complex.plan_cylindrical_complex_constitutive(
        separate, separate_pml, "E", separate_residency)
    fused_residency = Residency()
    plan = family.plan_cylindrical_fused_electric_pair(
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

    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and electric.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches": curl.launches + electric.launches,
        "fused_dispatches": plan.launches,
        "separate_prefix_syncs": curl.prefix_syncs,
        "fused_prefix_syncs": plan.prefix_syncs,
        "dispatches_removed_per_step": (
            (curl.launches + electric.launches - plan.launches)
            // max(len(per_step), 1)),
        "host_passes_in_seam_on_the_separate_side": (
            ["zero_metal_D"] if "zero_metal_D" in live else []),
        "host_passes_in_seam_on_the_fused_side": [],
        "separate_binds_six_coefficient_pointers": True,
        "fused_binds_one_packed_coefficient_buffer": True,
        "note": "the prefix pre-pass is paid on BOTH sides once per step; the fusion "
                "removes one DISPATCH and, on a walled run, one in-seam host pass. "
                "No throughput claim is made from these counts.",
    }


# ---------------------------------------------------------------------------
# The deposit
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str = "Ez") -> Any:
    """A REAL engine source, so ``field_type`` is the engine's own answer.

    ``amplitude=1j`` IS A MEASUREMENT rather than a decoration, and it is inherited
    from the m = 0 sibling's gate where it was established: ``ContinuousEnvelope``'s
    current carries the CW source's ``1/(-i*omega)`` factor, so with a REAL amplitude
    the current is purely imaginary at every t. On COMPLEX storage both planes are
    written, so the deposit would not vanish here as it does on a real grid — but the
    same amplitude is used so the two Dcyl gates deposit the same physical current,
    and THE FLOOR BELOW measures it either way.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    grid = fields.grid
    center = (0.5 * float(grid.shape[0]) / float(grid.resolution), 0.0, 0.0)
    return VolumeSource(grid=grid, component=component,
                        center=center, size=(0.0, 0.0, 0.0),
                        amplitude=1j,
                        envelope=ContinuousEnvelope(frequency=1.0))


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    """``driver.py``'s withdraw pass, on both sides identically."""
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def run_deposit_case(keywords: Mapping[str, Any], seed: int, steps: int,
                     repair: bool = True) -> Dict[str, Any]:
    """Complete driver steps with an ELECTRIC source in the seam, byte compared.

    THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    ``plan_cylindrical_fused_electric_pair`` directly would test the kernel and skip
    the wiring, and the wiring — ``launch._install_fused_pair`` reaching this family
    through its ``FUSED_PAIR_ARMS`` row — is what
    :data:`family.CARRIES_DEPOSIT_REPAIR` claims.
    """
    reference, reference_pml = build(keywords, seed)
    actual, actual_pml = build(keywords, seed)
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
                # THE FLOOR: the injection must MOVE WORDS, or this leg is a no-op
                # agreeing with a no-op.
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
        "prefix_syncs": inner.prefix_syncs,
        "live_passes": list(live),
        "selected": {slot: composed.selected.get(slot)
                     for slot in ("step_D", "update_E")},
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[int, int, Tuple[bool, ...]]:
    """The (bcz, |m| arm, walls) triple the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    kinds = _boundary_kinds(fields.grid, pml)
    bcz = templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC
    return (bcz, cylindrical_complex.m_class(int(fields.grid.m)),
            zero_metal_axes(fields.grid))


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit."""
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


def _edits(base: str) -> Dict[str, str]:
    """Armed source defects, each a plausible transcription slip.

    THE FIRST THREE ARE THE PACK'S OWN, and they exist because this product's prologue
    is new code rather than lifted text: the six aliases are the one place a wrong edit
    reads a valid float from the wrong vector — a plausible absorber, not a crash.
    """
    edits: Dict[str, str] = {
        # --- THE PACK PROLOGUE ------------------------------------------------
        "pack_alias_kmx_reads_sinvx":
            needle(base, "device const float* kmx = cpml + prm.off_kmx;",
                   "device const float* kmx = cpml + prm.off_sinvx;"),
        "pack_alias_x_and_z_swapped":
            needle(base, "device const float* kmx = cpml + prm.off_kmx;\n"
                         "    device const float* sinvx = cpml + prm.off_sinvx;",
                   "device const float* kmx = cpml + prm.off_kmz;\n"
                   "    device const float* sinvx = cpml + prm.off_sinvz;"),
        "pack_alias_sinvz_loses_its_offset":
            needle(base, "device const float* sinvz = cpml + prm.off_sinvz;",
                   "device const float* sinvz = cpml;"),
        # --- THE SEAM ---------------------------------------------------------
        "seam_takes_pre_recurrence_curl":
            needle(base, "float2 src0 = c_mul_field_left(v0, e0[ii]);",
                   "float2 src0 = c_mul_field_left(curl0, e0[ii]);"),
        "seam_takes_the_wrong_component":
            needle(base, "float2 src1 = c_mul_field_left(v1, e1[ii]);",
                   "float2 src1 = c_mul_field_left(v0, e1[ii]);"),
        "seam_drops_the_inverse_permittivity":
            needle(base, "float2 src2 = c_mul_field_left(v2, e2[ii]);",
                   "float2 src2 = v2;"),
        "seam_swaps_the_inverse_permittivity_row":
            needle(base, "float2 src2 = c_mul_field_left(v2, e2[ii]);",
                   "float2 src2 = c_mul_field_left(v2, e0[ii]);"),
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
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = km0[i];",
                   "float kp_0 = kp0[k], km_0 = km0[k];"),
    }
    return edits


def _m_zero_edits(base: str) -> Dict[str, str]:
    """Armed defects in the ``M_ZERO`` D body (2026-09-04), each a transcription slip.

    The m = 0 axis rules are POST-RECURRENCE edits of the stored registers
    (stepping :586-587): ``Dz[r=0] += axis_coef * Hp[r=0]`` and ``Dy[r=0] = 0``, the
    FIELDS only. The fold into the curl is the defect stepping.py:575-580 records as
    measured wrong under PML, and it is armed as a fold rather than described.
    """
    post_add = "        v2 = at_x ? (v2 + inc0) : v2;"
    return {
        "m_zero_post_add_dropped":
            needle(base, post_add, "        // MUTANT: no on-axis Dz increment"),
        "m_zero_Dy_axis_clear_dropped":
            needle(base, "        v1 = at_x ? float2(0.0f, 0.0f) : v1;",
                   "        // MUTANT: Dy[r=0] kept"),
        "m_zero_post_add_lands_on_the_wrong_component":
            needle(base, post_add, "        v0 = at_x ? (v0 + inc0) : v0;"),
        "m_zero_post_add_also_writes_the_auxiliary":
            needle(base, post_add,
                   post_add + "\n        n2 = at_x ? (n2 + inc0) : n2;"),
        "m_zero_post_add_folded_into_the_curl":
            needle(needle(base, post_add, "        // MUTANT: folded instead"),
                   "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n",
                   "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n"
                   "    curl2 = at_x ? -(c_mul_coefficient_left(axis_coef, b)) : curl2;\n"),
        # THE NULL CONTROL: scaling by four is exact, so the in-kernel spelling is
        # the host-rounded word. Declared null and CONFIRMED rather than assumed.
        "m_zero_axis_coef_formed_in_kernel":
            needle(base, "float2 inc0 = c_mul_coefficient_left(axis_coef, b);",
                   "float2 inc0 = c_mul_coefficient_left(4.0f * dtdx, b);"),
        # The seam on the m = 0 body: the E half must read the POST-ADDED axis word.
        "m_zero_seam_reads_the_axis_row_before_the_post_add":
            needle(base, "float2 src2 = c_mul_field_left(v2, e2[ii]);",
                   "float2 src2 = c_mul_field_left(\n"
                   "        at_x ? (v2 - c_mul_coefficient_left(axis_coef, b)) : v2, "
                   "e2[ii]);"),
    }


#: THE ARM SWAPS: the WHOLE body of another m class compiled for a case — the m-class
#: split armed rather than argued. ``(case, arm compiled instead)``.
ARM_SWAPS: Dict[str, Tuple[str, int]] = {
    "m_zero_case_compiled_with_the_m_one_body": (M_ZERO_CASE, cylindrical_complex.M_ONE),
    "m_one_case_compiled_with_the_m_zero_body": (M_ONE_CASE, cylindrical_complex.M_ZERO),
}


def shader_mutations(base: str, base_m_zero: str,
                     specialisations: Mapping[str, Tuple[int, int, Sequence[bool]]],
                     ) -> Dict[str, Dict[str, Any]]:
    """Compile every armed defect against the source of the case it is scored on."""
    mutants: Dict[str, Dict[str, Any]] = {}
    for name, source in _edits(base).items():
        mutants[name] = {
            "case": M_ONE_CASE,
            "functions": {shaders.CONTRACT_OFF:
                          compile_source(source).cylindrical_fused_electric_pair_step}}
    for name, source in _m_zero_edits(base_m_zero).items():
        mutants[name] = {
            "case": M_ZERO_CASE,
            "functions": {shaders.CONTRACT_OFF:
                          compile_source(source).cylindrical_fused_electric_pair_step}}
    for name, (case, other_arm) in ARM_SWAPS.items():
        bcz, arm, walls = specialisations[case]
        assert other_arm != arm, (name, arm)
        swapped = family.cylindrical_fused_electric_pair_source(
            bcz, other_arm, walls, EXPANSION)
        mutants[name] = {
            "case": case,
            "functions": {shaders.CONTRACT_OFF:
                          compile_source(swapped).cylindrical_fused_electric_pair_step}}
    return mutants


#: Mutation id -> what it is DECLARED to do. Anything not named here must be CAUGHT.
#: A DECLARED NULL MUST BE CONFIRMED, not merely permitted.
#:
#: THE TWO PHI OFFSET SWAPS, and they are MEASURED rather than conceded: ``kms_y`` and
#: ``sinv_y`` are each ONE element on a Dcyl grid and both hold exactly ``1.0f``
#: (0x3F800000), because the phi axis carries no PML at all. So pointing ``kmy`` at
#: ``sinvy[0]`` (and ``sinvy`` back at ``kmy[0]``) reads the same word. Their siblings
#: — the SAME two offsets moved the other way, into a NEIGHBOURING vector — are armed
#: as must-catch, which is what stops these reading as "the offsets do not matter".
#:
#: THE m = 0 AXIS COEFFICIENT SPELLED IN-KERNEL is the third: ``4.0f * dtdx`` is the
#: host-rounded ``4.0 * (dt/dx)`` bit for bit because scaling by four is exact, so
#: the row must NOT diverge — and a divergence there would mean the binding and the
#: array path disagree about a word this family ships.
MUTATION_EXPECTATION: Dict[str, str] = {
    "offset_kmy_plus_one_reads_sinvy": "null",
    "offset_sinvy_minus_one_reads_kmy": "null",
    "m_zero_axis_coef_formed_in_kernel": "null",
}


def mutation_must_be_caught(name: str) -> bool:
    return MUTATION_EXPECTATION.get(name, "caught") == "caught"


def byte_neutral_source(base: str) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and ``v0``
    hold the same float2, and a float2 stored to a ``device float2*`` and reloaded is
    bit-identical to the register. This edit is the fusion's central claim written as
    a program, and leg ``byte_neutral_control`` requires it NOT to diverge.
    """
    source = base
    for target in range(3):
        source = needle(
            source,
            f"float2 src{target} = c_mul_field_left(v{target}, e{target}[ii]);",
            f"float2 src{target} = c_mul_field_left(f{target}[ii], e{target}[ii]);")
    return source


#: Where each binding group starts in the plan's argument tuple. Spelled once, so the
#: host mutations and the kernel signature cannot drift apart.
PACK_SLOT = 12
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(22, 28))
IMR_SLOTS = (10, 11)


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: pack the HALF-INTEGER split-field coefficients for the D curl.

    ``step_D`` reads INTEGER positions and ``step_B`` half-integer ones. The kernel
    cannot tell: it is a half-cell error in the absorber profile, not a crash. On this
    family the six pointers became one, so the six-way swap became a one-way one — and
    that is exactly the kind of coupling a pack introduces, armed rather than argued.
    """
    tensor, _layout = coefficient_pack.packed_mirror(
        residency, "mutation:pack_half_integer", np, family.PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kms", "sinv")])
    _rebind(plan, (PACK_SLOT,), (tensor,))


def swap_e_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the INTEGER constitutive coefficients to the E half."""
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_imr_rows(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: exchange the two i*m/r coefficient rows.

    They are per-target rows with different Yee shifts and opposite signs; the kernel
    takes two pointers and never asks which target each belongs to.
    """
    args = list(plan._args)
    args[IMR_SLOTS[0]], args[IMR_SLOTS[1]] = args[IMR_SLOTS[1]], args[IMR_SLOTS[0]]
    plan._args = tuple(args)


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_pack_takes_the_half_integer_lattice": swap_curl_lattice,
    "e_half_takes_the_integer_lattice": swap_e_lattice,
    "imr_rows_exchanged": swap_imr_rows,
}


def _offset_mutation(index: int, delta: int) -> Callable[[Any], Any]:
    """A ``params`` record whose ``index``-th pack offset is off by ``delta``.

    THE HAZARD THE PACK INTRODUCES, ARMED. Six vectors in one allocation are told
    apart by six uints and nothing else, so an offset wrong by one element reads a
    valid float from a real vector — the exact shape of a defect that produces a
    plausible absorber rather than a fault. Built from the SHIPPED plan's own layout.
    """
    def make(plan: Any) -> Any:
        offsets = list(plan.pack_layout.offsets)
        moved = offsets[index] + delta
        if moved < 0:
            raise AssertionError(
                f"offset {index} is already 0; a negative offset is a fault rather "
                f"than a wrong value and this leg would measure the platform")
        offsets[index] = moved
        minus_dtdx, inc_b = cylindrical_complex.axis_increment_scalars(
            _M_FOR_MUTATIONS, plan.dtdx)
        return family._params_tensor(plan.shape, plan.dtdx, plan.zero_rows,
                                     minus_dtdx, inc_b, offsets,
                                     plan.residency.device,
                                     cylindrical_complex.axis_coefficient(plan.dtdx))
    return make


#: The azimuthal order the offset mutations' Params record is rebuilt at — the
#: mutation case's own, so the only field that differs from the shipped record is the
#: one offset the leg moved.
_M_FOR_MUTATIONS = 1

#: id -> (pack index, delta). ELEVEN: every one-element move of every offset except
#: the one that would take ``off_kmx`` negative, which reads before the allocation and
#: is a fault rather than a wrong value.
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

def leg_expansion() -> Dict[str, Any]:
    """The complex arm is bound from THIS family's probe artifact, never a default.

    Four questions: the configured probe must be present and classify an arm; a
    MISSING probe must refuse; an AMBIGUOUS probe must refuse; and a probe carrying
    only the base patterns must refuse.
    """
    global EXPANSION, PROBE_ARTIFACT

    configured = os.environ.get("MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE")
    record = cylindrical_complex.load_expansion_probe()
    arm = cylindrical_complex.expansion_from_probe(record)
    EXPANSION, PROBE_ARTIFACT = arm, configured
    digest = ""
    if configured and Path(configured).is_file():
        digest = hashlib.sha256(Path(configured).read_bytes()).hexdigest()
    missing_refused = cylindrical_complex.expansion_from_probe(None) is None
    ambiguous = dict(record or {})
    ambiguous["patterns"] = {name: "AMBIGUOUS_BOTH"
                             for name in (record or {}).get("patterns", {})}
    ambiguous_refused = cylindrical_complex.expansion_from_probe(ambiguous) is None
    return {
        "passed": bool(arm is not None and configured and digest
                       and missing_refused and ambiguous_refused),
        "probe_artifact": configured,
        "probe_sha256": digest,
        "probe_backend": (record or {}).get("backend"),
        "expansion_arm": arm,
        "missing_probe_refused": missing_refused,
        "ambiguous_probe_refused": ambiguous_refused,
    }


def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, (message.splitlines()[0] if message else "")


def _sweep_source(pointers: int) -> str:
    lines = [f"    device float2* p{n} [[buffer({n})]]," for n in range(pointers)]
    touch = " + ".join(f"p{n}[0]" for n in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n_elem; float dtdx; };", "",
        "kernel void ceiling_sweep(", *lines,
        f"    constant Params& prm [[buffer({pointers})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float2 touch = {touch};",
        *(f"    p{n}[idx] = p{n}[idx] + touch * prm.dtdx;" for n in range(pointers)),
        "}", ""))


def leg_binding_ceiling(base: str) -> Dict[str, Any]:
    """Four signatures compiled, and the ceiling BISECTED on this host.

    1. the SEPARATE-SCALAR shape (41 bindings) FAILS;
    2. the UNPACKED-POINTER shape (33 pointers + one ``Params&`` = 34 bindings)
       FAILS. This is the fused pair built the way every other D/E product on this
       backend is built, over the ceiling by THREE, and it is the measurement behind
       the board's ``UNFUSABLE ON METAL`` verdict for this cell;
    3. a synthetic 32-binding kernel FAILS, which locates the ceiling rather than
       reading it off a constant;
    4. the SHIPPED packed shape (29 bindings) COMPILES, LAUNCHES and reads every
       struct field back — the six pack offsets included, from the REAL packer.
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
    for label, source, key in (
            ("separate_scalar", family.refuted_separate_scalar_source(),
             "separate_scalar"),
            ("unpacked_pointer", family.refuted_unpacked_pointer_source(),
             "unpacked_pointer"),
            ("thirty_second_binding", family.refuted_thirty_second_binding(),
             "thirty_second")):
        ok, error = _compiles(source)
        row[f"{key}_compiled"] = ok
        row[f"{key}_error"] = error
        row[f"{key}_refused_for_the_right_reason"] = (
            not ok and "out of bounds" in error and "buffer" in error)

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
    ceiling_ok = (largest + 1 == MAX_BUFFER_BINDINGS and smallest == largest + 1)
    row["ceiling_measured_equals_declared"] = ceiling_ok
    row["over_the_ceiling_by"] = family.UNPACKED_POINTERS - largest
    row["pack_saves_pointers"] = family.UNPACKED_POINTERS - family.PACKED_POINTERS

    shipped_ok, shipped_error = _compiles(base)
    row["shipped_compiled"] = shipped_ok
    row["shipped_error"] = shipped_error

    names = list(family.PACKED_VECTORS)
    probe_fields = ["nx", "ny", "nz", "n_elem", "zrows", "dtdx", "minus_dtdx",
                    "axis_coef"] + [f"off_{name}" for name in names]
    pointers = "\n".join(f"    device float2*   b{n:<6}[[buffer({n})]],"
                         for n in range(len(probe_fields) + 1))
    struct = ("struct Params {\n"
              "    float2 inc_b;\n"
              "    uint nx; uint ny; uint nz; uint n_elem; uint zrows;\n"
              "    float dtdx; float minus_dtdx; float axis_coef;\n"
              + coefficient_pack.params_fields(names) + "\n};")
    reads = "\n".join(f"    b{n}[idx] = float2(float(prm.{name}), 0.0f);"
                      for n, name in enumerate(probe_fields))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "", struct, "",
        "kernel void packed_probe(", pointers,
        f"    constant Params& prm [[buffer({len(probe_fields) + 1})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }", reads,
        f"    b{len(probe_fields)}[idx] = prm.inc_b;", "}", ""))
    probe_ok, probe_error = _compiles(source)
    if not probe_ok:
        row.update(packed_compiled=False, packed_error=probe_error, passed=False)
        return row
    function = compile_source(source).packed_probe
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, 2, dtype=torch.float32, device="mps")
               for _ in range(len(probe_fields) + 1)]
    shape, dtdx = (3, 4, 5), 0.125
    zero_row_count, minus_dtdx, inc_b = 2, -0.0625, (0.25, 0.5)
    axis_coef = 0.5
    offsets = tuple(range(11, 11 + len(names)))
    params = family._params_tensor(shape, dtdx, zero_row_count, minus_dtdx, inc_b,
                                   offsets, "mps", axis_coef)
    function(*buffers, params)
    torch.mps.synchronize()
    read = [float(buffer.cpu().numpy()[0][0]) for buffer in buffers[:-1]]
    expected = [float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(zero_row_count),
                float(np.float32(dtdx)), float(np.float32(minus_dtdx)),
                float(np.float32(axis_coef))] + [
        float(offset) for offset in offsets]
    incb = [float(value) for value in buffers[-1].cpu().numpy()[0]]
    fields_ok = read == expected and incb == [float(inc_b[0]), float(inc_b[1])]
    row.update(packed_launched=True,
               packed_fields=probe_fields + ["inc_b"],
               packed_fields_read_back=read + incb,
               packed_fields_expected=expected + [float(v) for v in inc_b],
               packed_fields_correct=fields_ok,
               params_itemsize=int(family.params_record_dtype().itemsize),
               passed=bool(row["separate_scalar_refused_for_the_right_reason"]
                           and row["unpacked_pointer_refused_for_the_right_reason"]
                           and row["thirty_second_refused_for_the_right_reason"]
                           and shipped_ok and ceiling_ok and fields_ok
                           and row["over_the_ceiling_by"] == 3))
    return row


def leg_pack_identity(keywords: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """The packed buffer IS the six separate vectors, in bytes and in reads."""
    import torch  # noqa: PLC0415

    fields, pml = build(keywords, seed)
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


def leg_transcription(bcz: int, arm: int, walls: Sequence[bool]) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes, and the pack must not
    have reached the arithmetic."""
    source = family.cylindrical_fused_electric_pair_source(
        bcz, arm, walls, EXPANSION)
    curl = family.certified_cylindrical_curl_body(bcz, arm, EXPANSION)
    head, tail = curl.split(family._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (family._CURL_STORE + tail) in source

    from meep_gpu.metal_kernels import complex_fused_electric_pair as sibling

    certified = sibling.certified_constitutive_body(EXPANSION).splitlines()
    marker = "    // --- update_E (stepping.update_E"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    while spliced and (spliced[0].endswith("--") or spliced[0].strip().startswith("//")):
        spliced.pop(0)
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float2 src{t} = c_mul_field_left(g{t}[ii], e{t}[ii]);",
                 f"    float2 src{t} = c_mul_field_left(v{t}, e{t}[ii]);")
                for t in range(3)]
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


def leg_refusal() -> Dict[str, Any]:
    """The source seam, the m split and the geometry split, each measured by name."""
    fields, pml = build(REFUSAL_CASE, 91000)
    residency = Residency()

    undeclared = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    electric = _volume_source(fields, "Ez")
    electric_coverage = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (electric,), residency)
    electric_plan = family.plan_cylindrical_fused_electric_pair(
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
    indexless_coverage = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (indexless,), residency)
    indexless_named = [r for r in indexless_coverage.reasons
                       if "does not publish the index it writes" in r]
    indexless_plan = family.plan_cylindrical_fused_electric_pair(
        fields, pml, sources=(indexless,), residency=residency)

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (magnetic,), residency)
    # THE MIRROR IMAGE, asked of the magnetic twin rather than asserted about it.
    twin_magnetic = twin.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), Residency())
    twin_electric = twin.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (electric,), Residency())

    m0_fields, m0_pml = matrix.cylindrical(m=0, complex_storage=False)
    m0_coverage = family.cylindrical_fused_electric_pair_coverage(
        m0_fields, m0_pml, (), Residency())
    m0_named = [r for r in m0_coverage.reasons
                if "force_complex_fields is not set" in r]

    # THE m = 0 COMPLEX RUN IS ADMITTED (the M_ZERO arm, 2026-09-04), its plan
    # builds, and the composer installs the deposit bracket around it on a row that
    # carries an electric source — which is the corpus row's shape. And the REAL
    # family refuses the same run on the same clause the other way.
    from meep_gpu.metal_kernels import cylindrical_real  # noqa: PLC0415

    m0c_fields, m0c_pml = matrix.cylindrical(m=0, complex_storage=True)
    m0c_residency = Residency()
    m0c_coverage = family.cylindrical_fused_electric_pair_coverage(
        m0c_fields, m0c_pml, (), m0c_residency)
    m0c_plan = family.plan_cylindrical_fused_electric_pair(
        m0c_fields, m0c_pml, sources=(), residency=m0c_residency)
    m0c_real = cylindrical_real.cylindrical_real_curl_coverage(
        m0c_fields, m0c_pml, "step_D", Residency())
    m0c_real_named = [r for r in m0c_real.reasons if "force_complex_fields=True" in r]
    m0w_fields, m0w_pml = matrix.cylindrical(m=0, complex_storage=True)
    m0w_source = _volume_source(m0w_fields, "Ez")
    m0w_composed = metal_launch.plan_step(m0w_fields, m0w_pml, residency=Residency(),
                                          sources=(m0w_source,), fuse=True)
    m0w_installed = [type(m0w_composed.plans.get(slot)).__name__
                     for slot in ("step_D", "update_E")]
    m0w_wiring_ok = m0w_installed == ["LeadingRepairPlan", "TrailingRepairPlan"]

    cart_fields, cart_pml = matrix.cart()
    cart_coverage = family.cylindrical_fused_electric_pair_coverage(
        cart_fields, cart_pml, (), Residency())
    cart_named = [r for r in cart_coverage.reasons if "cylindrical" in r.lower()]

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
                       and not twin_magnetic.covered and twin_electric.covered
                       and not m0_coverage.covered and m0_named
                       and m0c_coverage.covered and m0c_plan is not None
                       and m0c_plan.m_arm == cylindrical_complex.M_ZERO
                       and not m0c_real.covered and m0c_real_named
                       and m0w_wiring_ok
                       and not cart_coverage.covered and cart_named
                       and absorb_row == ("cylindrical complex",
                                          "cylindrical complex")
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
        "magnetic_source_admitted_here": magnetic_coverage.covered,
        "magnetic_source_refused_by_the_twin": not twin_magnetic.covered,
        "electric_source_admitted_by_the_twin": twin_electric.covered,
        "m_zero_real_storage_refused": not m0_coverage.covered,
        "m_zero_real_storage_named": m0_named[:2],
        "m_zero_complex_storage_admitted": m0c_coverage.covered,
        "m_zero_complex_storage_reasons": list(m0c_coverage.reasons),
        "m_zero_complex_plan_built": m0c_plan is not None,
        "m_zero_complex_plan_arm": None if m0c_plan is None else m0c_plan.m_arm,
        "m_zero_complex_refused_by_the_real_family": not m0c_real.covered,
        "m_zero_complex_refused_by_the_real_family_named": m0c_real_named[:2],
        "m_zero_complex_installed_plans": m0w_installed,
        "m_zero_complex_wiring_installs_the_bracket": m0w_wiring_ok,
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    expansion_row = leg_expansion()
    if not expansion_row["passed"]:
        # THE REFUSED BRANCH IS STAMPED TOO, with the same two lines the release
        # write below carries. This gate has TWO payload write sites and stamped
        # only one, which is the exact shape `meep_gpu/test_gate_provenance.py`
        # exists to catch (its docstring records the same defect found in
        # gate_triton_complex_no_pml_curl on 2026-08-19, with the branches the
        # other way round). A run that dies here is the run whose "which bytes
        # produced this?" is hardest to reconstruct afterwards, so it is the last
        # artifact that should be written blind.
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        refused = {"verdict": "FAIL",
                   "rows": [dict(index=1, total=1, leg="expansion",
                                 label="probe", **expansion_row)]}
        _stamp_provenance(refused)
        args.out.write_text(json.dumps(refused, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
        log("VERDICT FAIL: the expansion probe did not bind an arm")
        return 1

    probe_fields, probe_pml = build(dict(CASES)[M_ONE_CASE], 1)
    bcz, arm, walls = _specialisation(probe_fields, probe_pml)
    assert walls[2], f"{M_ONE_CASE} does not wall z"
    assert arm == cylindrical_complex.M_ONE, (M_ONE_CASE, arm)
    zero_fields, zero_pml = build(dict(CASES)[M_ZERO_CASE], 1)
    bcz0, arm0, walls0 = _specialisation(zero_fields, zero_pml)
    assert arm0 == cylindrical_complex.M_ZERO, (
        f"{M_ZERO_CASE} does not compile the M_ZERO arm; the m = 0 mutations would "
        f"be armed on lines the shipped kernel does not emit")
    specialisations = {M_ONE_CASE: (bcz, arm, walls), M_ZERO_CASE: (bcz0, arm0, walls0)}
    base = family.cylindrical_fused_electric_pair_source(bcz, arm, walls, EXPANSION)
    base_m_zero = family.cylindrical_fused_electric_pair_source(
        bcz0, arm0, walls0, EXPANSION)
    mutants = shader_mutations(base, base_m_zero, specialisations)
    neutral = {shaders.CONTRACT_OFF:
               compile_source(byte_neutral_source(base)
                              ).cylindrical_fused_electric_pair_step}

    controls = ("m1_z_metallic", "m2_z_periodic", "m0_z_metallic")
    total = (1 + 1 + 1 + 2 + 2 * len(CASES) + 1 + len(controls)
             + 2 * len(DEPOSIT_CASES) + 1 + 1
             + len(mutants) + len(HOST_MUTATIONS) + len(OFFSET_MUTATIONS) + 1)
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "expansion",
                     "label": "the_arm_is_bound_from_the_probe", **expansion_row})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "twenty_eight_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling(base)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "pack_identity",
                     "label": "one_buffer_reads_as_six_vectors",
                     **leg_pack_identity(dict(CASES)[M_ONE_CASE], 51000)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_bytes",
                     **leg_transcription(bcz, arm, walls)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "both_halves_are_the_certified_bytes_on_the_m_zero_arm",
                     **leg_transcription(bcz0, arm0, walls0)})
        emit(handle, rows[-1])

        for offset, (name, keywords) in enumerate(CASES):
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, 61000 + offset, args.steps)}
            rows.append(row)
            emit(handle, row)

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
                     **leg_subnormal_ladder(dict(CASES)[M_ONE_CASE], 65000)})
        emit(handle, rows[-1])

        for offset, name in enumerate(controls):
            index += 1
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(dict(CASES)[name], 67000 + offset,
                                          args.steps)}
            rows.append(row)
            emit(handle, row)

        for offset, (name, keywords) in enumerate(DEPOSIT_CASES):
            index += 1
            result = run_deposit_case(keywords, 69000 + 2 * offset, args.steps)
            rows.append({"index": index, "total": total, "leg": "deposit",
                         "label": f"a_real_electric_source_inside_the_seam:{name}",
                         "case": name, "m_arm": cylindrical_complex.m_class(
                             int(keywords["m"])),
                         "steps": args.steps, **result})
            emit(handle, rows[-1])

            index += 1
            result = run_deposit_case(keywords, 69001 + 2 * offset, args.steps,
                                      repair=False)
            diverged = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "deposit_null_control",
                   "label": f"the_unbracketed_launch_must_diverge:{name}",
                   "case": name, "diverged": diverged,
                   "words_the_injection_moved": result.get("words_the_injection_moved"),
                   "passed": bool(diverged and result.get("launches")
                                  and result.get("words_the_injection_moved")),
                   "installed_plans": result.get("installed_plans"),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "the_source_seam_the_m_split_and_the_geometry",
                     **leg_refusal()})
        emit(handle, rows[-1])

        index += 1
        result = run_case(dict(CASES)[M_ONE_CASE], 71000, args.steps,
                          functions=neutral)
        row = {"index": index, "total": total, "leg": "byte_neutral_control",
               "label": "register_replaced_by_a_reload_of_the_same_word",
               "diverged": not result.get("bit_identical", False),
               "passed": bool(result.get("passed")),
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches")}
        rows.append(row)
        emit(handle, row)

        for name, entry in mutants.items():
            index += 1
            result = run_case(dict(CASES)[entry["case"]], 72000 + index, args.steps,
                              functions=entry["functions"])
            caught = not result.get("bit_identical", False)
            wanted = mutation_must_be_caught(name)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "case": entry["case"], "caught": caught,
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
            result = run_case(dict(CASES)[M_ONE_CASE], 72000 + index, args.steps,
                              patch=patch)
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
            result = run_case(dict(CASES)[M_ONE_CASE], 72000 + index, args.steps,
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

        index += 1
        result = run_case(dict(CASES)[M_ONE_CASE], 72000 + index, args.steps)
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
                   "byte_neutral_controls": 1,
                   "deposit_cases": len(DEPOSIT_CASES),
                   "deposit_null_controls": len(DEPOSIT_CASES)},
        "mutation_case": M_ONE_CASE,
        "mutation_cases": {"m_one": M_ONE_CASE, "m_zero": M_ZERO_CASE},
        "arm_swaps": {name: {"case": case, "compiled_arm": other}
                      for name, (case, other) in ARM_SWAPS.items()},
        "mutation_specialisation": {"bcz": bcz, "m_arm": arm,
                                    "zero_metal": list(walls)},
        "mutation_specialisations": {
            name: {"bcz": spec[0], "m_arm": spec[1], "zero_metal": list(spec[2])}
            for name, spec in specialisations.items()},
        "expansion": EXPANSION,
        "expansion_probe": PROBE_ARTIFACT,
        "packed_vectors": list(family.PACKED_VECTORS),
        "binding_counts": {
            "separate_scalars": family.SEPARATE_SCALAR_BINDINGS,
            "unpacked_pointers_plus_struct": family.UNPACKED_POINTER_BINDINGS,
            "packed": family.PACKED_BINDINGS,
            "ceiling": MAX_BUFFER_BINDINGS,
        },
        "corpus_rows_admitted": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "admit_both_halves": 16,
            "and_carrying_an_electric_deposit": 16,
            "note": "the LARGEST cell on the Metal board. All sixteen rows declare "
                    "an ELECTRIC source, so with CARRIES_DEPOSIT_REPAIR at False "
                    "this product would admit ZERO. 16 is an upper bound because "
                    "the census records source FIELD TYPES but not whether a row's "
                    "grid would also satisfy the residency and wall-readability "
                    "clauses this predicate adds. The 194-row census of 2026-09-03 "
                    "(metal_coverage_2026-09-03_complete) carries one more Dcyl "
                    "row, examples:dipole_in_vacuum_cyl_off_axis.py (m = 0, "
                    "complex64, PML, two electric sources), which the M_ZERO arm "
                    "of 2026-09-04 reaches; that row's census verdict is re-cut "
                    "after this gate, not asserted here.",
        },
        "environment": ENVIRONMENT,
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

    _stamp_provenance(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
