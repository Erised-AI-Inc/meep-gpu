"""Byte-identity gate for the off-diagonal E->P weld the rotation hazard refused.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/dispersive_offdiag_fused_polarization_
pair.py``: ONE launch performing the WHOLE off-diagonal dispersive ``update_E``
(driver.py:3313) and the ``update_P`` recurrence (:3315) of one component across every
state that drives it, followed by the remaining components as certified ADE launches
with the host rotation between.

THE CELL IT TAKES, AND THE REFUSAL IT ANSWERS. ``E_to_P
(cuda_dispersive_offdiag/dispersive off-diagonal, cuda_ade/ADE)`` is one seam-instance
-- ``examples/absorbed_power_density.py`` -- the cell this weld was built for. (Since
1.0 the composer refuses that row's update_E arm, and the weld with it, by the chunk
clause below: its x axis wraps and carries a layer on both faces.) The shipped
sibling refuses the cell by name, and refuses the PER-COMPONENT SPLIT specifically:

    "Split per component, launch y would read ``P[x]`` AFTER launch x's rotation
     moved the advanced buffer into that slot -- the driver order reads the OLD one,
     so the weld would diverge, and diverge in a way that still computes."

This product does not split. ``update_E`` stays monolithic, so no rotation happens
inside the launch and the off-diagonal row product has no rotated slot to read. THE
MUTATION ``rotate_before_launch`` REBUILDS THE HAZARD -- it moves the rotation to
before the launch, which is exactly the state the split would have produced -- and if
it is not caught, this gate cannot tell the two shapes apart and its S1 means nothing.

=============================================================================
THE BAR
=============================================================================

The same one the stencil welds are held to, and this file imports their harness
rather than restating it: per COMPLETE DRIVER STEP over all stored volumes as raw
uint32 words, 60 steps, both float32 subnormal policies, both value classes, every
block size, launch counts from two independent counters, read-only volumes bit
unchanged, and a two-sided mutation battery -- with a CONFIRMED NULL that is the
seam's own claim rather than a formality: reading the drive back out of
``f_w_<c>`` instead of taking the ``src`` register must be the IDENTITY, because a
float32 word stored to global and reloaded is the identity on the bits. That leg is
what makes the register hand-off a measurement instead of an argument.

A must-catch mutation that CANNOT differ on one fixture's grid -- the stored-E drive on
a grid where no chunk carries the fused component's own layer -- is declared a
structural null on that arm by the array path's region plan (``STRUCTURAL_NULLS``),
run, and required identical; every other arm must still catch it, and a fixture that
carries the layer is in the sweep for each fused component.

THE DOMAIN: EVERY FIXTURE IS A DIVISION THE COMPOSER CAN HAND THIS WELD. The weld's
update_E half indexes MEEP's chunks on every axis through the chunk record, so its
predicate ends on the composer's own clause
(``arms.StepContext.chunk_division_refusal(CHUNK_AXES_OFFDIAG)``, the clause the
composer asks of the ``dispersive off-diagonal`` arm and of this product), and the
composer installs the weld only over the two arms it selected. On an axis that wraps
AND carries a layer the record's run labels do not name the chunk intervals (on x the
wrapped row 0 joins the low run while its chunk is the high one, as on
absorbed_power_density's x), so dispatch never hands the weld such a division and a
fixture there certifies nothing dispatch can reach. Every axis that carries a layer is
therefore walled; a wrapping axis is in the sweep only without a layer (``bare``),
which is how the fold's periodic termination stays swept
(``periodic_terminated_fold_three_d``); and ``leg_refusal`` measures the weld refusing
the corpus row's wrapping division with the composer's own reason.

WHAT IT DOES NOT CLAIM. No throughput claim (the box is shared and nothing here is
timed) and no dispatch claim (``fastpath.plan_fast_path`` still returns ``None`` on
every branch). The launch SAVING is counted rather than timed: on this cell's one
corpus row the array path runs ``1 + 3`` launches on this seam and the weld runs
``1 + 2``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402
# THE SHARED HARNESS, imported rather than restated: the word comparison, the
# launch counter, the state list and the driver walk are one piece of machinery and
# two copies of them are equal only until someone edits one. Both gates run in the
# same round, so both bind the same bytes.
import gate_cuda_offdiag_stencil_welds as stencil  # noqa: E402

from meep_gpu import stepping  # noqa: E402

log = probe.log
words = stencil.words
differing = stencil.differing
STATE_NAMES = stencil.STATE_NAMES
IMMUTABLE_PML = stencil.IMMUTABLE_PML
MemoLaunchCounter = stencil.MemoLaunchCounter
STEP_PASSES = stencil.STEP_PASSES

SEED = 20260902
STEPS = 60
SCHEDULE_REPEATS = 4
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: The polarization buffers a complete step also touches. Compared beside the
#: stored volumes: a weld that got E right and the recurrence wrong would otherwise
#: pass for a step, and the error would only surface through the next update_E.
POLARIZATION_SLOTS: Tuple[str, ...] = ("P", "P_prev")

#: The corpus row's own boundaries: every axis wraps, and every axis the gate's layer
#: rule resolves carries a layer. The composer refuses this division (the module
#: docstring's DOMAIN); ``leg_refusal`` measures the weld refusing it by the same text.
WRAPPING: Optional[Dict[str, str]] = None

#: Every layered axis walled. On a 2-D grid z is one cell and carries no layer, so it
#: keeps the wrap it had; the record names its one chunk.
WALLED_2D: Dict[str, str] = {"x": "metallic", "y": "metallic"}
WALLED_3D: Dict[str, str] = {"x": "metallic", "y": "metallic", "z": "metallic"}

#: The fixture sweep. Every one folds (the constitutive arm requires it), and the
#: axes swept are the ones that change what the WELD does: the row mask, the number
#: of driving states, the sigma kind, which component the fused launch carries, and
#: the fold's parity and termination. Every one is a division the composer can hand
#: the weld (the module docstring's DOMAIN): a layered axis is walled, and a wrapping
#: axis carries no layer (``bare``, the axes the layer rule in :func:`frame` skips).
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "one_state_all_components", "cell": (1.6, 2.0, 0.0), "dimensions": 2,
     "symmetry": (("Y", 1),), "boundaries": WALLED_2D,
     "rows": {"Ex": ("Ey",), "Ey": ("Ex",)}, "states": 1,
     "driven": ("Ex", "Ey", "Ez"), "sigma_volume": False},
    {"label": "one_state_all_rows_odd_fold", "cell": (1.6, 2.0, 0.0),
     "dimensions": 2, "symmetry": (("Y", -1),), "boundaries": WALLED_2D, "rows": "all",
     "states": 1, "driven": ("Ex", "Ey", "Ez"), "sigma_volume": False},
    {"label": "one_state_volume_sigma", "cell": (1.6, 2.0, 0.0), "dimensions": 2,
     "symmetry": (("Y", 1),), "boundaries": WALLED_2D, "rows": "all", "states": 1,
     "driven": ("Ex", "Ey", "Ez"), "sigma_volume": True},
    {"label": "two_states_mixed_sigma", "cell": (1.6, 2.0, 0.0), "dimensions": 2,
     "symmetry": (("Y", 1),), "boundaries": WALLED_2D, "rows": "all", "states": 2,
     "driven": ("Ex", "Ey", "Ez"), "sigma_volume": "mixed"},
    {"label": "one_state_one_component", "cell": (1.6, 2.0, 0.0), "dimensions": 2,
     "symmetry": (("Y", 1),), "boundaries": WALLED_2D, "rows": "all", "states": 1,
     "driven": ("Ez",), "sigma_volume": False},
    # The fold terminated by a metallic wall at its far face, with the layer there; x is
    # walled too now, so this fixture's boundaries are the walled default.
    {"label": "metallic_terminated_fold", "cell": (1.6, 2.0, 0.0), "dimensions": 2,
     "symmetry": (("Y", 1),), "boundaries": ("metallic", "metallic", "periodic"),
     "rows": "all", "states": 1, "driven": ("Ex", "Ey", "Ez"),
     "sigma_volume": False},
    {"label": "three_d_fold", "cell": (1.2, 2.0, 1.2), "dimensions": 3,
     "symmetry": (("Y", 1),), "boundaries": WALLED_3D, "rows": "all", "states": 1,
     "driven": ("Ex", "Ey", "Ez"), "sigma_volume": False},
    # The single-component weld ON A GRID THAT CARRIES ITS LAYER: a 2-D grid has no z
    # layer, so ``one_state_one_component``'s Ez launch stores the drive everywhere and
    # the stored-E mutation is a structural null there; here it must be caught.
    {"label": "one_state_one_component_three_d", "cell": (1.2, 2.0, 1.2),
     "dimensions": 3, "symmetry": (("Y", 1),), "boundaries": WALLED_3D, "rows": "all",
     "states": 1, "driven": ("Ez",), "sigma_volume": False},
    # THE FOLD'S PERIODIC TERMINATION, inside the domain: the folded y and the
    # non-folded z wrap and carry no layer, so the record names their chunks and the
    # kernel's periodic boundary code runs on both; x keeps the layer the fused Ex
    # needs for the stored-E mutation's arm.
    {"label": "periodic_terminated_fold_three_d", "cell": (1.2, 2.0, 1.2),
     "dimensions": 3, "symmetry": (("Y", 1),), "boundaries": {"x": "metallic"},
     "bare": ("y", "z"), "rows": "all", "states": 1, "driven": ("Ex", "Ey", "Ez"),
     "sigma_volume": False},
)

REDUCED_LABELS: Tuple[str, ...] = (
    "one_state_all_components", "one_state_all_rows_odd_fold",
    "two_states_mixed_sigma")

ALL_ROWS = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")}


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def frame(spec: Dict[str, Any], xp) -> Tuple[Any, Any, Any]:
    """The fixture's ``(fields, grid, pml)`` before any value is written, on ``xp``.

    One geometry for the device build and for the host-side reading of which chunks
    carry which layer (``own_layer_absent``). Every axis of six cells or more carries a
    layer (on the far face only where it folds) unless the spec names it ``bare``.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=spec["cell"],
                dimensions=spec["dimensions"], courant=0.35, xp=xp,
                boundaries=spec["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]))
    bare = tuple("xyz".index(name) for name in spec.get("bare", ()))
    thickness = tuple((0, 0) if grid.shape[axis] < 6 or axis in bare
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, grid, pml


def own_layer_absent(fields, pml, component: str) -> Optional[str]:
    """Why the stored ``component`` IS the drive on every cell of this grid, or None.

    ``update_E`` stores ``(D - P) * chi1inv`` -- the drive, MEEP's plain constitutive
    (update_eh.cpp:162-170) -- on every cell except those whose chunk carries the
    component's OWN-axis layer, where it stores the accumulation and keeps the drive in
    ``f_w_<c>``. When the array path's region plan marks no cell for that form, reading
    the stored E back is the identity on the bits, so a mutation that drives the
    recurrence from it cannot differ on this grid: a STRUCTURAL null (KERNELS.md
    section 7.1), declared by the plan, never by the fixture's label.
    """
    plan = stepping._region_plan(fields, pml)  # noqa: SLF001 - the array path's own plan
    if bool(np.asarray(plan.mask("constitutive", component, np)).any()):
        return None
    return (f"no chunk of this grid carries {component}'s own layer (the array path's "
            f"region plan marks no constitutive cell for it), so the stored "
            f"{component} is the drive on every cell")


def build(spec: Dict[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml)`` on the DEVICE, seeded for this class."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    fields, grid, pml = frame(spec, cp)
    shape = tuple(int(n) for n in grid.shape)
    names = ("Ex", "Ey", "Ez")
    values = (2.0, 2.5, 3.0)
    rows = ALL_ROWS if spec["rows"] == "all" else spec["rows"]
    built = {row: {partner: cp.asarray(
        rng.uniform(-0.25, 0.25, shape).astype(np.float32))
        for partner in partners} for row, partners in rows.items()} or None
    fields.set_epsilon_volumes(
        {n: cp.full(shape, v, cp.float32) for n, v in zip(names, values)},
        {n: cp.full(shape, np.float32(1.0 / v), cp.float32)
         for n, v in zip(names, values)},
        chi1inv_offdiagonal=built)

    kinds = spec["sigma_volume"]
    states = []
    for index in range(spec["states"]):
        volume = kinds if isinstance(kinds, bool) else bool(index % 2 == 0)
        sigma: Dict[str, Any] = {}
        for name in names:
            if name not in spec["driven"]:
                sigma[name] = 0.0
                continue
            sigma[name] = (cp.asarray(
                rng.uniform(0.1, 0.4, shape).astype(np.float32)) if volume
                else 0.25 + 0.03 * index)
        states.append(PolarizationState(
            Susceptibility(1.0 + 0.1 * index, 0.1, "lorentzian"), sigma, grid,
            np.float32))
    fields.polarizations = states

    hosts = (probe.pml_field_hosts(shape, rng, "uniform") if value_class == "uniform"
             else probe.subnormal_band_hosts(STATE_NAMES, shape, rng))
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        source = hosts.get(name)
        if source is None:
            source = rng.normal(0.0, 0.37, size=shape).astype(np.float32)
        array[...] = cp.asarray(np.ascontiguousarray(
            np.asarray(source, dtype=np.float32)))
    # THE RECURRENCE STATE IS SEEDED TOO. A zero P and P_prev make the recurrence's
    # first two terms exactly zero whatever the coefficients are, so a mis-paired
    # c_now/c_prev would only show from step three -- and a leg that seeded them
    # zero would be reporting a pass for the two steps it could not distinguish.
    for state in states:
        for slot in POLARIZATION_SLOTS:
            # A state ALLOCATES only the components it drives, so the slot dict is
            # the enumeration rather than the component list -- an undriven fixture
            # has no P['Ex'] to seed and asking for one is a KeyError, not a zero.
            for array in getattr(state, slot).values():
                array[...] = cp.asarray(
                    rng.normal(0.0, 0.21, size=shape).astype(np.float32))
        # A state that drives nothing allocates no scratch either -- the undriven
        # refusal fixture is the one that has none, and seeding it would be the
        # test inventing a buffer the engine does not make.
        if state._scratch is not None:
            state._scratch[...] = cp.asarray(
                rng.normal(0.0, 0.21, size=shape).astype(np.float32))
    return fields, grid, pml


def state_of(fields) -> Dict[str, Any]:
    """Every stored volume AND every polarization buffer a step can touch.

    The recurrence's own buffers are here because this weld's second half IS the
    recurrence: a comparison over the field volumes alone would miss a wrong P for a
    whole step, and would then report the divergence one step late and in the wrong
    place.
    """
    out = {name: getattr(fields, name) for name in STATE_NAMES
           if getattr(fields, name, None) is not None}
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for slot in POLARIZATION_SLOTS:
            for component, array in sorted(getattr(state, slot).items()):
                out[f"state{index}.{slot}[{component}]"] = array
    return out


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: stencil.to_host(value).copy()
            for name, value in state_of(fields).items()}


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: n for name in sorted(left) if (n := differing(left[name], right[name]))}


def array_step(fields, pml) -> None:
    for name in STEP_PASSES:
        stencil.run_pass(name, fields, pml)
    stepping.update_P(fields, pml)


# ---------------------------------------------------------------------------
# One product case
# ---------------------------------------------------------------------------

def run_case(spec: Dict[str, Any], value_class: str, steps: int, *,
             kernel: Optional[Any] = None, rotate: bool = True,
             rotate_before: bool = False, threads: int = 256, repeat: int = 0,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        dispersive_offdiag_fused_polarization_pair as weld,
    )

    label = f"{spec['label']}/{value_class}{label_suffix}"
    reference, grid_r, pml_r = build(spec, value_class, case_rng(label))
    subject, grid_s, pml_s = build(spec, value_class, case_rng(label))
    seeded = frozen(reference)
    assert not compare(seeded, frozen(subject)), (
        f"{label}: the two engines were not seeded identically")

    read_only = stencil.read_only_snapshot(subject, pml_s)
    census = probe.operand_census({name: stencil.to_host(value)
                                   for name, value in state_of(subject).items()})
    saved = weld._FUSED_THREADS  # noqa: SLF001 - the launcher's own constant
    weld._FUSED_THREADS = int(threads)  # noqa: SLF001
    first_divergence: Optional[Dict[str, Any]] = None
    reports: List[Dict[str, Any]] = []
    try:
        with MemoLaunchCounter() as counter:
            for step in range(steps):
                array_step(reference, pml_r)
                for name in STEP_PASSES:
                    if name == "update_E":
                        if rotate_before:
                            # THE HAZARD, REBUILT. Rotating BEFORE the launch is the
                            # state a per-component split would have produced by the
                            # time the second component's launch read P[x].
                            target = weld.fused_component(subject)
                            for state in weld._polar.component_specs(  # noqa: SLF001
                                    subject)[target]["states"]:
                                weld._rotate(state, target)  # noqa: SLF001
                        report = weld.run_dispersive_offdiag_fused_polarization_pair(
                            subject, grid_s, pml_s, kernel=kernel, rotate=rotate)
                        if not report.get("launched"):
                            raise SystemExit(
                                f"{label}: the weld refused a fixture the sweep "
                                f"admits: {report.get('reason')}")
                        reports.append(report)
                        continue
                    stencil.run_pass(name, subject, pml_s)
                cp.cuda.runtime.deviceSynchronize()
                moved = compare(frozen(reference), frozen(subject))
                if moved and first_divergence is None:
                    first_divergence = {"step": step + 1, "volumes": moved}
                    break
            named, total = counter.named(), counter.total
    finally:
        weld._FUSED_THREADS = saved  # noqa: SLF001

    final = frozen(subject)
    must_move = (list(seeded) if value_class == "uniform"
                 else list(stencil.BAND_MUST_MOVE))
    # AN AUXILIARY WHOSE FORM NO CHUNK OF THIS GRID CARRIES DOES NOT MOVE, BY CONTRACT:
    # MEEP writes f_w only on the cells whose chunk carries the component's own layer
    # and fu only on those that carry its dsigu layer, and on these 2-D fixtures no
    # chunk carries a z layer. Read off the array path's own region plan, and
    # recorded rather than dropped silently.
    plan = getattr(reference, "region_plan", None)

    def form_absent(name: str) -> bool:
        if plan is None or not name.startswith(("f_w_", "fu_")):
            return False
        sub_step, component = (("constitutive", name[4:]) if name.startswith("f_w_")
                               else ("curl", name[3:]))
        try:
            return not bool(np.asarray(plan.mask(sub_step, component, np)).any())
        except Exception:  # noqa: BLE001 - an unreadable plan exempts nothing
            return False

    exempt = sorted(name for name in must_move if form_absent(name))
    must_move = [name for name in must_move if name not in exempt]
    unmoved = sorted(name for name in must_move
                     if name in seeded and not differing(seeded[name], final[name]))
    expected_fused = len(reports)
    expected_tail = sum(len(r.get("tail_launches") or []) for r in reports)
    record = {
        "label": label, "spec": spec["label"], "value_class": value_class,
        "steps": steps, "threads": int(threads), "repeat": repeat,
        "shape": [int(n) for n in grid_s.shape],
        "bit_identical": first_divergence is None,
        "first_divergence": first_divergence,
        "launch_counts": {"memo_named": named, "memo_total": total,
                          "fused_reports": expected_fused,
                          "tail_reports": expected_tail},
        # THE COUNT IS TWO-SIDED: the memo proxy must see exactly the fused launches
        # plus the tail launches, and the fused kernel must appear exactly once per
        # step. An engine that never launched the weld would be byte-identical to
        # the oracle BY CONSTRUCTION.
        "launch_counts_agree": (
            total == expected_fused + expected_tail
            and named.get(weld.KERNEL_NAME) == expected_fused == steps),
        "launch_ledger": ({"weld": reports[0]["launches"],
                           "array_path": reports[0]["array_path_launches"]}
                          if reports else None),
        "read_only_drift": stencil.read_only_drift(read_only, subject, pml_s),
        "operand_census": census,
        "unmoved_volumes": unmoved,
        "exempt_from_the_movement_floor": exempt,
        "non_vacuous": not unmoved,
        "rotated": bool(rotate),
    }
    return record


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

SOURCE_MUTATIONS: Tuple[Tuple[str, Callable[[str], str], str, str], ...] = (
    ("drive_from_the_stored_E",
     lambda s, target: stencil.needle(
         s, f"    float w = src_{target};", f"    float w = {target}[idx];"),
     "CAUGHT",
     "the recurrence driven by the stored E on every cell. On the chunks that carry "
     "the component's layer fields.drive_field returns f_w_<c> (MEEP's w per chunk) "
     "and the stored E there is the accumulated field -- a value that exists, is the "
     "right shape, and is not what update_P consumes. Where no chunk carries that "
     "layer the stored E IS the drive, so the arm is a declared structural null "
     "(STRUCTURAL_NULLS), run and required identical"),
    ("drive_read_back_from_the_volume",
     lambda s, target: stencil.needle(
         s, f"    float w = src_{target};",
         f"    float w = in_h_{target[1].lower()} ? f_w_{target}[idx] : {target}[idx];"),
     "NULL CONFIRMED",
     "THE SEAM'S OWN CLAIM, ARMED AS A NULL. constitutive_apply stored src one line "
     "earlier -- into fw[idx] on the cells whose chunk carries the layer, into f[idx] "
     "elsewhere -- and reading MEEP's per-chunk drive back from those two arrays must "
     "give exactly the register, because a float32 word stored to global and "
     "reloaded is the identity on the bits. If this ever came back CAUGHT the "
     "register hand-off would be an approximation and the whole weld would be "
     "unsound"),
    ("drive_read_back_from_f_w_on_every_cell",
     lambda s, target: stencil.needle(
         s, f"    float w = src_{target};", f"    float w = f_w_{target}[idx];"),
     "CAUGHT",
     "the 0.9.2 drive: f_w_<c> read on every cell. Off the layer's chunks the plain "
     "form leaves f_w untouched, so it is not this step's word -- the partner that "
     "makes the read-back null above a measurement"),
    ("recurrence_coefficients_swapped",
     lambda s, target: stencil.needle(
         s, "((p_0 * two_minus_od_0) - (gamma1_0 * q_0))",
         "((p_0 * gamma1_0) - (two_minus_od_0 * q_0))"),
     "CAUGHT",
     "P^n and P^(n-1) weighted by each other's constant -- a second-order "
     "recurrence that still converges, to the wrong thing"),
    ("recurrence_history_dropped",
     lambda s, target: stencil.needle(
         s, "((p_0 * two_minus_od_0) - (gamma1_0 * q_0))", "(p_0 * two_minus_od_0)"),
     "CAUGHT",
     "the P^(n-1) term removed: a second-order recurrence turned first-order, which "
     "is the defect the certified ADE prologue names in its own comment"),
    ("pole_chain_dropped_from_the_source",
     lambda s, target: stencil.needle(
         s, "    value = value - P_Ex_0[index];\n", "\n"),
     "CAUGHT",
     "dmp_Ex stops subtracting its pole: update_E's source becomes D rather than "
     "D - sum P, which is the coupling between the two halves this weld exists to "
     "carry in one launch"),
)

#: Per-fixture STRUCTURAL NULLS of a source mutation (KERNELS.md section 7.1): name -> a
#: predicate ``(fields, pml, target)`` giving the reason an arm cannot differ on that
#: fixture's grid, or None. Declared by structure, never by label. A declared arm still
#: RUNS and must come back identical; one that diverges fails the scoreboard, and the
#: remaining arms must score the mutation's expected outcome on their own.
STRUCTURAL_NULLS: Dict[str, Callable[[Any, Any, str], Optional[str]]] = {
    "drive_from_the_stored_E": own_layer_absent,
}

HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], str, str], ...] = (
    ("rotation_skipped", {"rotate": False}, "CAUGHT",
     "the launch computes the step and throws it away: without the rotation the "
     "scratch never becomes the live P, so every subsequent step reads a stale "
     "polarization"),
    ("rotate_before_launch", {"rotate_before": True}, "CAUGHT",
     "THE HAZARD THE SIBLING'S REFUSAL NAMES, REBUILT. Rotating BEFORE the launch "
     "puts the state in exactly the shape a per-component split would have produced "
     "by the time the second component's launch read P[x]. If this is not caught, "
     "this gate cannot tell the monolithic weld from the split one and its S1 means "
     "nothing"),
)


def score_source_arms(arms: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One source mutation's outcome over its arms.

    An arm whose anchor is absent (``predicted_null``) did not run; a DECLARED structural
    null ran and must be identical, and is scored apart from the armed arms; every other
    arm is armed. A declared null that diverged decides the outcome by itself, so a
    wrong declaration cannot hide behind the arms that caught.
    """
    armed = [arm for arm in arms if "caught" in arm and "declared_null" not in arm]
    nulls = [arm for arm in arms if "declared_null" in arm]
    diverged = [arm["spec"] for arm in nulls if arm["caught"]]
    outcome = ("DECLARED NULL DIVERGED" if diverged
               else "NOT ARMED" if not armed
               else "NULL CONFIRMED" if not any(arm["caught"] for arm in armed)
               else "CAUGHT" if all(arm["caught"] for arm in armed) else "PARTIAL")
    return {"outcome": outcome, "armed": len(armed),
            "declared_nulls": [arm["spec"] for arm in nulls],
            "declared_nulls_diverged": diverged}


def leg_mutations(specs: Sequence[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        dispersive_offdiag_fused_polarization_pair as weld,
    )
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        ade_sigma_is_volume, offdiag_row_mask,
    )

    scoreboard: Dict[str, Any] = {}
    for name, rewrite, expected, why in SOURCE_MUTATIONS:
        arms: List[Dict[str, Any]] = []
        for spec in specs:
            fields, _grid, pml = build(spec, "uniform", case_rng(spec["label"]))
            target = weld.fused_component(fields)
            states = list(weld._polar.component_specs(  # noqa: SLF001
                fields)[target]["states"])
            index = {t: i for i, (t, _s, _a) in enumerate(weld.ELECTRIC_TERMS)}[target]
            kinds = tuple(bool(ade_sigma_is_volume(state, target))
                          for state in states)
            mask = offdiag_row_mask(fields)
            counts = weld.pole_counts(fields)
            source = weld.kernel_source(mask, counts, index, kinds)
            structural = STRUCTURAL_NULLS.get(name)
            null_reason = structural(fields, pml, target) if structural else None
            try:
                mutated = rewrite(source, target)
            except SystemExit as exc:
                arms.append({"spec": spec["label"], "predicted_null": True,
                             "reason": f"the anchor is not present on this fixture: "
                                       f"{exc}"})
                continue
            kernel = cp.RawKernel(mutated, weld.KERNEL_NAME, options=GUARD_OPTIONS)
            record = run_case(spec, "uniform", steps=min(steps, 8), kernel=kernel,
                              label_suffix=f"/M:{name}")
            arm = {"spec": spec["label"], "target": target,
                   "caught": not record["bit_identical"],
                   "first_divergence": record["first_divergence"]}
            if null_reason is not None:
                arm["declared_null"] = null_reason
            arms.append(arm)
        score = score_source_arms(arms)
        scoreboard[name] = {"why": why, "expected": expected, **score, "arms": arms,
                            "as_required": score["outcome"] == expected}
        log(f"[mutation] {name:34s} {score['outcome']:14s} ({score['armed']} armed, "
            f"{len(score['declared_nulls'])} declared null, "
            f"{len(score['declared_nulls_diverged'])} diverged)")

    for name, kwargs, expected, why in HOST_MUTATIONS:
        arms = []
        for spec in specs:
            record = run_case(spec, "uniform", steps=min(steps, 8),
                              label_suffix=f"/M:{name}", **kwargs)
            arms.append({"spec": spec["label"],
                         "caught": not record["bit_identical"],
                         "first_divergence": record["first_divergence"]})
        outcome = "CAUGHT" if all(a["caught"] for a in arms) else "UNCAUGHT"
        scoreboard[name] = {"why": why, "expected": expected, "outcome": outcome,
                            "arms": arms, "as_required": outcome == expected}
        log(f"[mutation] {name:34s} {outcome:14s} ({len(arms)} armed)")
    return {"scoreboard": scoreboard,
            "passed": all(entry["as_required"] for entry in scoreboard.values())}


# ---------------------------------------------------------------------------
# The structural legs
# ---------------------------------------------------------------------------

def leg_lift() -> Dict[str, Any]:
    """The E half whole, the P half certified, and the pole bound once."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        dispersive_offdiag_fused_polarization_pair as weld,
        dispersive_offdiag_update_e as offdiag_e,
    )

    mask, counts, kinds = (1, 0, 0, 1, 0, 0), (1, 1, 1), (False,)
    source = weld.kernel_source(mask, counts, 0, kinds)
    certified = offdiag_e.dispersive_offdiag_source(mask, counts)
    missing = [line for line in certified.splitlines()
               if line.strip()
               and not line.startswith('extern "C" __global__ void')
               and line != "    float gw_x, float gw_y, float gw_z"
               and line not in source]
    entry = source.index(f'extern "C" __global__ void {weld.KERNEL_NAME}(')
    signature = source[entry:].split("\n) {\n", 1)[0]
    code = re.sub(r"//[^\n]*", "", source)
    record = {
        "kernel": weld.KERNEL_NAME,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "corpus_digest": weld.corpus_digest(),
        "lift_edits": len(weld.LIFT_EDITS),
        "certified_update_E_lines_missing": missing,
        "all_three_components_in_one_launch": all(
            f"constitutive_apply({n}, f_w_{n}, idx, src_{n}," in source
            for n in ("Ex", "Ey", "Ez")),
        "pole_bound_exactly_once": signature.count("P_Ex_0") == 1,
        "p_now_is_not_a_parameter": "p_now" not in code,
        "the_recurrence_is_the_certified_line": (
            "p_out_0[idx] = gamma1inv_0 * (((p_0 * two_minus_od_0) - "
            "(gamma1_0 * q_0)) + (omega0dtsqr_0 * (s_0 * w)));") in source,
        "the_drive_is_the_register": "    float w = src_Ex;" in source
        and "drive_w[idx]" not in source,
        "the_entry_point_is_its_own": weld.KERNEL_NAME != offdiag_e.KERNEL_NAME,
    }
    record["passed"] = (not missing and all(
        value for key, value in record.items()
        if isinstance(value, bool)))
    return record


def leg_refusal() -> Dict[str, Any]:
    """The configurations this weld refuses BY NAME, and the partition it rests on."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        dispersive_offdiag_fused_polarization_pair as weld,
        fused_polarization_pair as diagonal,
    )
    from meep_gpu.cuda_kernels.arms import StepContext  # noqa: PLC0415
    from meep_gpu.cuda_kernels.chunk_coverage import (  # noqa: PLC0415
        CHUNK_AXES_OFFDIAG,
    )

    cases: List[Dict[str, Any]] = []
    base = SPECS[0]
    fields, grid, pml = build(base, "uniform", case_rng("refusal"))
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, pml, grid, ())
    cases.append({"case": "admits its own cell", "covered": covered, "want": True,
                  "reason": why})
    # THE DIVISION DISPATCH NEVER HANDS IT, measured from both sides: the corpus row's
    # wrapping axes with their layers, everything else the admitted fixture's. The
    # composer refuses the update_E arm and this product by the clause it asks of both
    # (``StepContext.chunk_division_refusal`` with the off-diagonal axes), so the weld
    # must refuse by exactly that text -- a predicate that admitted here, or refused
    # for another reason, would certify a domain the composer does not share.
    wrapping = dict(base, label="refusal_wrapping", boundaries=WRAPPING)
    w_fields, w_grid, w_pml = build(wrapping, "uniform", case_rng("refusal_wrapping"))
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        w_fields, w_pml, w_grid, ())
    composer = StepContext(w_fields, w_pml, w_grid).chunk_division_refusal(
        CHUNK_AXES_OFFDIAG)
    cases.append({"case": "refuses the wrapping division with the composer's reason",
                  "covered": covered, "want": False, "reason": why,
                  "composer_reason": composer,
                  "reason_is_the_composer_s": composer is not None and why == composer})
    cases.append({"case": "the DIAGONAL E->P pair refuses the same row",
                  "covered": diagonal.covers_fused_polarization_pair(
                      fields, pml, grid, ())[0], "want": False})
    # A run with no off-diagonal row belongs to the diagonal pair and to this weld
    # not at all -- the partition, measured from both sides.
    plain = dict(base, label="refusal_diagonal", rows={})
    fields, grid, pml = build(plain, "uniform", case_rng("refusal_diagonal"))
    cases.append({"case": "refuses a diagonal run",
                  "covered": weld.covers_dispersive_offdiag_fused_polarization_pair(
                      fields, pml, grid, ())[0], "want": False})
    cases.append({"case": "the DIAGONAL pair serves what this refuses",
                  "covered": diagonal.covers_fused_polarization_pair(
                      fields, pml, grid, ())[0], "want": True})
    undriven = dict(base, label="refusal_undriven", driven=())
    fields, grid, pml = build(undriven, "uniform", case_rng("refusal_undriven"))
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, pml, grid, ())
    cases.append({"case": "refuses a run with nothing driven", "covered": covered,
                  "want": False, "reason": why})
    return {"cases": cases,
            "passed": all(c["covered"] == c["want"]
                          and c.get("reason_is_the_composer_s", True) for c in cases)}


def save(results: Dict[str, Any], path: str) -> None:
    gate_provenance.stamp(results)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _warm_memo(specs: Sequence[Dict[str, Any]]) -> None:
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        dispersive_offdiag_fused_polarization_pair as weld,
    )

    for spec in specs:
        fields, grid, pml = build(spec, "uniform", np.random.default_rng(1))
        report = weld.run_dispersive_offdiag_fused_polarization_pair(
            fields, grid, pml)
        if not report.get("launched"):
            raise SystemExit(f"the memo warm-up could not launch on "
                             f"{spec['label']}: {report.get('reason')}")
    cp.cuda.runtime.deviceSynchronize()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--repeats", type=int, default=SCHEDULE_REPEATS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    started = time.perf_counter()
    results: Dict[str, Any] = {
        "gate": "cuda_dispersive_offdiag_polarization_pair",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of "
                     "dispersive_offdiag_fused_polarization_pair_pml_real, plus the "
                     "certified ADE launches for the components it did not carry, "
                     "leave every stored volume AND every polarization buffer "
                     "byte-identical to stepping's update_E -> update_P inside a "
                     "complete driver step?"),
        "what_it_does_not_claim": [
            "throughput: the box is shared and nothing here is timed",
            "dispatch: fastpath.plan_fast_path still returns None on every branch",
        ],
        "budget_steps": args.steps, "schedule_repeats": args.repeats,
        "block_sizes": list(BLOCK_SIZES),
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        return 0

    results["device_mode"] = True
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    results["driver_order"] = stencil.leg_driver_order()
    results["lift"] = leg_lift()
    log(f"[lift] the E half is whole and the P half certified: "
        f"{results['lift']['passed']}")
    results["refusal"] = leg_refusal()
    log(f"[refusal] the seam clauses hold: {results['refusal']['passed']}")
    save(results, args.out)

    _warm_memo(specs)

    s1: List[Dict[str, Any]] = []
    for spec in specs:
        for value_class in VALUE_CLASSES:
            for repeat in range(args.repeats):
                record = run_case(spec, value_class, args.steps, repeat=repeat,
                                  label_suffix=f"/r{repeat}")
                s1.append(record)
                log(f"[S1] {record['label']:50s} "
                    f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'} "
                    f"launches={record['launch_counts']['memo_total']} "
                    f"ledger={record['launch_ledger']} "
                    f"subnormals={record['operand_census']['subnormals']}")
                save({**results, "S1_subject": s1}, args.out)
    results["S1_subject"] = s1

    s2: List[Dict[str, Any]] = []
    for spec in specs:
        for threads in BLOCK_SIZES:
            record = run_case(spec, "uniform", steps=min(args.steps, 12),
                              threads=threads, label_suffix=f"/b{threads}")
            s2.append(record)
            log(f"[S2] {record['label']:50s} "
                f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'}")
        save({**results, "S2_block_sizes": s2}, args.out)
    results["S2_block_sizes"] = s2

    if not args.skip_mutations:
        results["mutations"] = leg_mutations(specs, args.steps)
        save(results, args.out)

    band = [r for r in s1 if r["value_class"] == "subnormal_band"]
    uniform = [r for r in s1 if r["value_class"] == "uniform"]
    clauses = {
        "S1 every case is bit-identical for the whole budget":
            bool(s1) and all(r["bit_identical"] for r in s1),
        "S1 the launch counts agree between two independent counters":
            bool(s1) and all(r["launch_counts_agree"] for r in s1),
        "S1 nothing read-only drifted": all(not r["read_only_drift"] for r in s1),
        "S1 every case moved the state it must move":
            all(r["non_vacuous"] for r in s1),
        "S2 identical at every block size":
            bool(s2) and all(r["bit_identical"] for r in s2),
        "the subnormal band really contains subnormals":
            bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
        "the uniform class contains none":
            bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                  for r in uniform),
        "the E half is lifted whole and the P half is the certified recurrence":
            results["lift"]["passed"],
        "the seam clauses refuse what the launch cannot serve":
            results["refusal"]["passed"],
        "every mutation scored as required":
            results.get("mutations", {}).get("passed", True),
    }
    results["verdict"] = {
        "clauses": clauses, "passed": all(clauses.values()),
        "denominators": {"S1_cases": len(s1), "S2_cases": len(s2),
                         "fixtures": len(specs), "repeats": args.repeats,
                         "steps_per_case": args.steps},
    }
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"]),
        "reasons": [c for c, v in clauses.items() if not v],
        "what_it_licenses": (
            "one CUDA kernel measured byte-identical to the array path's update_E -> "
            "update_P, over complete driver steps, at every block size, under both "
            "float32 subnormal policies, with the rotation hazard the shipped "
            "sibling's refusal names REBUILT as a mutation and CAUGHT. It licenses "
            "NO throughput claim and NO dispatch claim"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"VERDICT: {'PASS' if results['verdict']['passed'] else 'FAIL'}  "
        f"({results['elapsed_s']} s)")
    return 0 if results["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
