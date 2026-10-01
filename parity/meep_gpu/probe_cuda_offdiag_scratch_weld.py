"""The hand-CUDA stencil weld's arithmetic, measured on the host before a line of PTX.

WHAT THIS PROBE IS FOR. The CUDA board records two off-diagonal cells as
STENCIL-BLOCKED -- ``D_to_E (cuda_curl/PML, cuda_offdiag/off-diagonal)`` (8 reachable
seam-instances) and ``D_to_E (cuda_curl/PML, cuda_folded_offdiag/folded off-diagonal)``
(9) -- with the verdict "the constitutive half reads the curl half's in-place output at
a cell this thread does not own". That verdict is TRUE OF THE IN-PLACE WELD and says
nothing about the SCRATCH-OUTPUT weld the fusion-residue audit identified (§1.1), in
which

* the curl half writes ``D_new``/``fu_new`` to a LAUNCH-LOCAL SCRATCH allocation, so
  nothing in the launch mutates ``D_old``, ``fu_old`` or ``H``;
* the constitutive half takes its OWN cell from a register and RECOMPUTES each foreign
  cell it reads from PRE-LAUNCH state through the same ``__device__`` function;
* the launcher rotates the ``D``/``fu`` bindings after the launch, which is the
  choreography the certified E->P path already ships (``ade_kernels.py:414-424``).

Under that shape the constitutive's foreign reads are a PURE FUNCTION of state no
thread writes, so there is no ordering hazard to have -- but only if the per-cell
closed form the kernel bakes really equals the driver's pass order
``step_D -> fill_symmetry_bc_D -> zero_metal_D -> fill_folded_far_ghosts_D`` byte for
byte, and only if the recompute is genuinely LOAD-BEARING (a weld whose foreign taps
could have read pre-launch D would not need any of this). This probe measures both,
on the arms the two CUDA cells actually name, on a host with no device.

WHAT IT DOES NOT DO. It is not a gate and certifies nothing: NumPy is not NVRTC, and
the substitution, the launch counts and the mutation battery belong to
``gate_cuda_offdiag_stencil_welds.py`` on the device. What it removes is the
possibility of discovering the ARITHMETIC is wrong from inside a device slot.

THE SIBLING MEASUREMENT, AND WHAT IS NEW HERE.
``results/metal_scratch_weld_closed_form_2026-09-01/probe.json`` already measured the
same closed form for the Metal lane over ten fixtures, and its fixture taxonomy and
null battery are lifted here rather than reinvented -- ``closed_form_d_final``'s five
null arms are that probe's, argument for argument, because the resolution is the ARRAY
PATH's arithmetic and is backend-independent. Three things are this probe's own:

1. **the resolution is applied PER CELL through a scalar function**, not as an array
   transform, and every output cell's TAP LEDGER is recorded -- the kernel's foreign
   read is one redirected raw cell and the ledger measures that it is exactly one.
   An array-level identity is compatible with a resolution that needs a whole plane;
   a per-cell one is not, and per-cell purity is the property that licenses the
   recompute.
2. **the race the in-place weld would have is COUNTED** rather than argued: for every
   fixture, how many of the constitutive's D taps land on a cell whose stepped value
   differs from its pre-launch value. That number is the size of the hazard the
   2026-08-20 Triton artifact measured qualitatively, and a zero would mean the whole
   scratch design bought nothing here.
3. **the arms are the CUDA ones**: every fixture is admitted by
   ``covers_real_pml_curl(..., "step_D")`` and by this cell's own constitutive
   predicate (``covers_real_pml_offdiag_constitutive`` /
   ``covers_folded_offdiag_composition``), asked through the census's own CuPy-backend
   proxy, so no case measures a configuration the shipped product would refuse.

Host-only: NumPy, no CuPy. Progress reporting: one flushed line per case, one fsynced JSONL row.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "cuda_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402
from cuda_predicate_battery import _GridWithCupyBackend  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels.coverage import (  # noqa: E402
    OFFDIAG_TRANSVERSE_PARTNERS, covers_real_pml_curl,
    covers_real_pml_offdiag_constitutive,
)
from meep_gpu.cuda_kernels.folded_offdiag_kernels import (  # noqa: E402
    covers_folded_offdiag_composition,
)
from meep_gpu.fields import IYEE_SHIFTS, mirror_parity  # noqa: E402

D_COMPONENTS: Tuple[str, str, str] = ("Dx", "Dy", "Dz")
E_COMPONENTS: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

#: Every stored volume either walk can touch. The comparison reads the whole set,
#: so a weld that got E right and fu wrong is a failure rather than a pass.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The driver's pass order for one complete step with no sources and no poles
#: (driver.py:3283-3316). Inert passes no-op identically on both walks.
FULL_STEP: Tuple[Tuple[str, Callable[[Any, Any], None]], ...] = (
    ("step_B", lambda f, p: stepping.step_B(f, p)),
    ("fill_B", lambda f, p: stepping.fill_symmetry_bc_B(f)),
    ("zero_metal_B", lambda f, p: stepping.zero_metal_B(f)),
    ("far_B", lambda f, p: stepping.fill_folded_far_ghosts_B(f)),
    ("update_H", lambda f, p: stepping.update_H(f, p)),
    ("step_D", lambda f, p: stepping.step_D(f, p)),
    ("fill_D", lambda f, p: stepping.fill_symmetry_bc_D(f)),
    ("zero_metal_D", lambda f, p: stepping.zero_metal_D(f)),
    ("far_D", lambda f, p: stepping.fill_folded_far_ghosts_D(f)),
    ("update_E", lambda f, p: stepping.update_E(f, p)),
)

#: The three D-seam passes the weld absorbs, by the name :data:`FULL_STEP` gives them.
CARRIED_PASSES = ("fill_D", "zero_metal_D", "far_D")


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def restore(fields: Any, snapshot: Dict[str, np.ndarray]) -> None:
    for name, value in snapshot.items():
        getattr(fields, name)[...] = value


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: n for name in sorted(left)
            if (n := differing(left[name], right[name]))}


def seed_state(fields: Any, seed: int) -> None:
    """Physical-band values in every stored volume.

    A zero-filled volume agrees with a zero-filled volume, so the identity below
    would be vacuous without this.
    """
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.normal(0.0, 0.37, size=array.shape).astype(array.dtype)


# ---------------------------------------------------------------------------
# The grid facts the resolution reads
# ---------------------------------------------------------------------------

def grid_facts(fields: Any) -> Dict[str, Any]:
    """Exactly the runtime plan the kernel takes as ``int``/``float`` arguments.

    Read off the same three engine functions the shipped launcher reads
    (``stepping._stored_past_owned``, ``._far_reflect_rows``, ``grid.mirror_phase``),
    so a divergence between this probe and the product cannot come from a second
    reading of the grid.
    """
    grid = fields.grid
    folded = tuple(axis for axis in range(3) if grid.is_mirrored(axis))
    walled = tuple(axis for axis in range(3)
                   if grid.is_metallic(axis) and not grid.is_mirrored(axis))
    far_axes = tuple(axis for axis in folded
                     if stepping._stored_past_owned(grid, axis))
    return {"folded": folded, "walled": walled, "far_axes": far_axes,
            "reflect": stepping._far_reflect_rows(grid),
            "phases": {axis: int(grid.mirror_phase(axis)) for axis in folded},
            "shape": tuple(int(n) for n in grid.shape)}


# ---------------------------------------------------------------------------
# The closed form, PER CELL -- the kernel's ``resolve_D``, in Python
# ---------------------------------------------------------------------------
#
# ONE SCALAR FUNCTION, ONE RAW TAP. This is the whole design claim: the post-pass
# value of component c at ANY cell is the raw stepped value at ONE redirected cell,
# times a product of per-axis parities applied ONE MULTIPLY AT A TIME, or an exact
# zero where the wall clear owns the plane. The kernel calls it for its own cell (to
# fill the scratch) and for each foreign tap the off-diagonal stencil reads.
#
# THE PARITY SPELLING IS THE ARRAY PATH'S -- ``weight * value`` with ``weight`` the
# Python int ``mirror_parity`` returns, exactly as ``stepping._write_mirror_ghost``
# (:1450-1451) and ``._fill_folded_far_ghosts`` (:1524-1534) spell it, and applied
# ONCE PER AXIS in ascending order because those two passes write plane after plane.
#
# THE ORDER OF THE THREE PASSES IS THE DRIVER'S (driver.py:3309, :3310, :3311): the
# near fill, then the wall clear, then the far fill. So a near ghost that lands in a
# cleared plane is CLEARED (the clear runs after), and the far image reads the
# POST-clear, POST-near value at its source row.


def resolve_cell(component: str, cell: Tuple[int, int, int], raw: np.ndarray,
                 facts: Dict[str, Any], *,
                 drop_parity: bool = False,
                 near_source_row: int = 2,
                 drop_far: bool = False,
                 flip_clear_order: bool = False,
                 halo_source_unstepped: bool = False,
                 pre_launch: Optional[np.ndarray] = None,
                 ledger: Optional[List[Tuple[int, int, int]]] = None,
                 ) -> Any:
    """The post-pass value of ``component`` at ``cell``, from the RAW stepped array.

    The keyword arms are the NULL CONTROLS; each must move bytes on a fixture where
    the machinery it disables is live, or the identity is vacuous for that arm.
    ``ledger``, when supplied, records every raw cell this call consulted -- the
    kernel reads exactly one, and the caller asserts it.
    """
    iyee = IYEE_SHIFTS[component]
    own_axis = "xyz".index(component[1])
    shape = facts["shape"]
    scalar = raw.dtype.type
    near_axes = tuple(axis for axis in facts["folded"] if iyee[axis] == 0)
    clear_axes = tuple(axis for axis in facts["walled"] if iyee[axis] == 0)
    is_far_axis = (own_axis in facts["far_axes"]) and not drop_far
    reflect_row = facts["reflect"][own_axis] if is_far_axis else None

    def parity(axis: int) -> int:
        if drop_parity:
            return 1
        return int(mirror_parity(component, axis, facts["phases"][axis]))

    def tap(z: Tuple[int, int, int]) -> Any:
        if ledger is not None:
            ledger.append(z)
        source = raw if not halo_source_unstepped or pre_launch is None else pre_launch
        return source[z]

    def near_value(z: Tuple[int, int, int]) -> Any:
        source = list(z)
        weights = []
        for axis in near_axes:                    # X, Y, Z order: the fill's own
            if z[axis] == 0:
                source[axis] = near_source_row
                weights.append(parity(axis))
        value = tap(tuple(source))
        for weight in weights:                    # nested, one multiply per plane
            value = scalar(weight) * value
        return value

    def cleared(z: Tuple[int, int, int]) -> bool:
        return any(z[axis] == 0 for axis in clear_axes)

    def after_clear(z: Tuple[int, int, int]) -> Any:
        if flip_clear_order:
            # NULL: the parity applied AFTER the clear at a near ghost -- the wrong
            # order, which differs in the sign bit of a zero on odd-parity walls.
            source = list(z)
            weights = []
            for axis in near_axes:
                if z[axis] == 0:
                    source[axis] = near_source_row
                    weights.append(parity(axis))
            base = scalar(0) if cleared(z) else tap(tuple(source))
            for weight in weights:
                base = scalar(weight) * base
            return base
        if cleared(z):
            if ledger is not None:
                ledger.append(z)      # the clear still OWNS this cell: one tap
            return scalar(0)
        return near_value(z)

    if reflect_row is not None and cell[own_axis] == shape[own_axis] - 1:
        redirected = list(cell)
        redirected[own_axis] = reflect_row
        value = after_clear(tuple(redirected))
        # mirror_parity(c, own_axis, phase) IS the far fill's -phase for a shift-1
        # component (fields.py:180-182); nothing else multiplies it.
        return scalar(parity(own_axis)) * value
    return after_clear(cell)


def resolved_volume(component: str, raw: np.ndarray, facts: Dict[str, Any],
                    *, in_place: bool = False, **arms: Any) -> np.ndarray:
    """:func:`resolve_cell` over every cell -- the scratch the kernel writes.

    ``in_place`` is the SCRATCH-ALIASED-TO-STORAGE null: the resolution reads the
    array it is writing, in C-raster order, which is what a weld that skipped the
    scratch allocation would compute.
    """
    out = raw if in_place else np.empty_like(raw)
    source = out if in_place else raw
    nx, ny, nz = facts["shape"]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = resolve_cell(component, (i, j, k), source, facts,
                                            **arms)
    return out


# ---------------------------------------------------------------------------
# The tap ledger and the race census
# ---------------------------------------------------------------------------

def tap_ledger(component: str, facts: Dict[str, Any], raw: np.ndarray
               ) -> Dict[str, Any]:
    """How many RAW cells each resolved cell consults. The kernel reads one."""
    nx, ny, nz = facts["shape"]
    counts: Dict[int, int] = {}
    redirected = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                seen: List[Tuple[int, int, int]] = []
                resolve_cell(component, (i, j, k), raw, facts, ledger=seen)
                counts[len(set(seen))] = counts.get(len(set(seen)), 0) + 1
                if set(seen) != {(i, j, k)}:
                    redirected += 1
    return {"distinct_raw_cells_per_output_cell": {str(k): v
                                                   for k, v in sorted(counts.items())},
            "cells_whose_source_is_not_themselves": redirected}


def in_place_raster_differing(facts: Dict[str, Any], raw: Dict[str, np.ndarray],
                              resolved: Dict[str, np.ndarray]) -> Dict[str, int]:
    """The deterministic host analogue of ``scratch_aliased_to_storage``, measured.

    Resolving in place in C-raster order is what a weld that skipped the scratch
    allocation would compute if the blocks ran in index order. It comes out the
    IDENTITY here -- the redirect sources are stored row 2 on a near axis and the
    reflect row on the own axis, and neither is itself moved by a redirect raster
    order has already applied -- so it is REPORTED rather than asserted. A non-zero
    on some future fixture would be a finding, not a failure of the design.
    """
    out: Dict[str, int] = {}
    for component in D_COMPONENTS:
        scratch = np.array(raw[component], copy=True)
        resolved_volume(component, scratch, facts, in_place=True)
        moved = differing(scratch, resolved[component])
        if moved:
            out[component] = moved
    return out


def constitutive_taps(component_index: int, rows: Dict[str, Sequence[str]],
                      facts: Dict[str, Any]) -> List[Tuple[int, Tuple[int, int, int],
                                                           Tuple[int, int, int]]]:
    """Every (partner component, own cell, tapped cell) the off-diagonal stencil reads.

    Transcribed from ``stepping._offdiagonal_terms`` (stepping.py:1235-1251) the way
    ``offdiag_emitter._term_lines`` assembles it: the pair half a cell DOWN the
    PARTNER's axis and the same pair half a cell UP the component's OWN axis, so the
    four samples are home, down, up and the corner. Only the three that are not
    ``home`` can be foreign.
    """
    nx, ny, nz = facts["shape"]
    name = E_COMPONENTS[component_index]
    own_axis = "xyz".index(name[1])
    partners = rows.get(name, ())
    taps: List[Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]] = []
    for offset in range(2):
        partner_name = E_COMPONENTS[OFFDIAG_TRANSVERSE_PARTNERS[component_index][offset]]
        if partner_name not in partners:
            continue
        partner_axis = "xyz".index(partner_name[1])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    home = (i, j, k)
                    down = list(home)
                    down[partner_axis] = home[partner_axis] - 1
                    up = list(home)
                    up[own_axis] = home[own_axis] + 1
                    corner = list(up)
                    corner[partner_axis] = home[partner_axis] - 1
                    for sample in (down, up, corner):
                        if all(0 <= sample[a] < facts["shape"][a] for a in range(3)):
                            taps.append((partner_axis, home, tuple(sample)))
    return taps


def race_census(rows: Dict[str, Sequence[str]], facts: Dict[str, Any],
                pre_launch: Dict[str, np.ndarray],
                resolved: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """How many foreign taps the IN-PLACE weld would read at an undefined time.

    A tap is HAZARDOUS when the value the constitutive must see (the resolved one)
    differs from the value that cell held before the launch: in an in-place weld the
    load returns one or the other depending on which block ran first, and the answer
    is a schedule. Zero hazardous taps would mean the scratch design bought nothing
    on this fixture, so the number is reported per fixture rather than asserted once.
    """
    total = 0
    hazardous = 0
    for component_index in range(3):
        for partner_axis, _home, sample in constitutive_taps(component_index, rows,
                                                             facts):
            name = D_COMPONENTS[partner_axis]
            total += 1
            before = words(np.asarray(pre_launch[name][sample], dtype=np.float32))
            after = words(np.asarray(resolved[name][sample], dtype=np.float32))
            if before[0] != after[0]:
                hazardous += 1
    return {"foreign_taps": total, "hazardous_taps": hazardous,
            "hazard_fraction": round(hazardous / total, 6) if total else None}


# ---------------------------------------------------------------------------
# The two walks
# ---------------------------------------------------------------------------

def reference_step(fields: Any, pml: Any) -> None:
    for _name, action in FULL_STEP:
        action(fields, pml)


def scheme_step(fields: Any, pml: Any, facts: Dict[str, Any],
                capture: Optional[Dict[str, Any]] = None, **arms: Any) -> None:
    """One complete step with the D seam replaced by the scratch-output weld.

    ``step_D`` supplies the RAW array (in the kernel it is recomputed per tap from
    pre-launch D/fu/H; here it is the same pure function evaluated once, which is the
    identity the device gate re-measures with two independent launch counters). The
    per-cell resolution then maps raw -> final into a SCRATCH array, the write-back IS
    the rotation, and the array path's own ``update_E`` plays the certified
    constitutive half -- consuming exactly the taps the fused kernel re-derives.
    """
    skip_rotation = bool(arms.pop("skip_rotation", False))
    for name, action in FULL_STEP:
        if name in CARRIED_PASSES:
            continue                       # carried by the closed form
        if name == "step_D":
            before = {c: np.array(getattr(fields, c), copy=True) for c in D_COMPONENTS}
            action(fields, pml)
            raw = {c: np.array(getattr(fields, c), copy=True) for c in D_COMPONENTS}
            final = {c: resolved_volume(c, raw[c], facts, pre_launch=before[c], **arms)
                     for c in D_COMPONENTS}
            if capture is not None:
                capture["pre_launch"] = before
                capture["raw"] = raw
                capture["resolved"] = final
            for c in D_COMPONENTS:
                getattr(fields, c)[...] = before[c] if skip_rotation else final[c]
            continue
        action(fields, pml)


# ---------------------------------------------------------------------------
# The arms: no fixture may measure a configuration the product refuses
# ---------------------------------------------------------------------------

def arm_verdicts(fields: Any, pml: Any, family: str) -> Dict[str, Any]:
    """The two CUDA predicates this cell conjoins, asked modulo the CuPy backend."""
    proxy = _GridWithCupyBackend(fields.grid)
    curl_ok, curl_why = covers_real_pml_curl(fields, pml, proxy, "step_D")
    if family == "A":
        const_ok, const_why = covers_real_pml_offdiag_constitutive(fields, pml, proxy)
    else:
        const_ok, const_why = covers_folded_offdiag_composition(fields, pml, proxy)
    return {"curl": {"covered": bool(curl_ok), "reason": str(curl_why)},
            "constitutive": {"covered": bool(const_ok), "reason": str(const_why)}}


def run_case(name: str, family: str, build: Callable[[], Tuple[Any, Any]],
             rows: Dict[str, Sequence[str]], seed: int,
             expect: Dict[str, bool]) -> Dict[str, Any]:
    started = time.time()
    fields, pml = build()
    facts = grid_facts(fields)
    arms = arm_verdicts(fields, pml, family)
    assert arms["curl"]["covered"], (name, "curl arm refuses", arms["curl"]["reason"])
    assert arms["constitutive"]["covered"], (
        name, "constitutive arm refuses", arms["constitutive"]["reason"])

    seed_state(fields, seed)
    pre = frozen(fields)

    reference_step(fields, pml)
    ref1 = frozen(fields)
    reference_step(fields, pml)
    ref2 = frozen(fields)

    restore(fields, pre)
    capture: Dict[str, Any] = {}
    scheme_step(fields, pml, facts, capture=capture)
    got1 = frozen(fields)
    scheme_step(fields, pml, facts)
    got2 = frozen(fields)

    step1 = compare(ref1, got1)
    step2 = compare(ref2, got2)

    # Vacuity floors: the machinery each case exists for must be LIVE on it.
    live = {
        "near": bool(facts["folded"]),
        "far": bool(facts["far_axes"]),
        "odd": any(phase == -1 for phase in facts["phases"].values()),
        "wall": bool(facts["walled"]),
        "corner": len(facts["folded"]) > 1,
    }
    for key, wanted in expect.items():
        assert live.get(key, False) == wanted, (
            f"{name}: expected {key}={wanted} but the fixture answers "
            f"{live.get(key)} -- the case does not exercise what it claims")

    ledger = {c: tap_ledger(c, facts, capture["raw"][c]) for c in D_COMPONENTS}
    for component, entry in ledger.items():
        assert set(entry["distinct_raw_cells_per_output_cell"]) == {"1"}, (
            f"{name}/{component}: a resolved cell consulted "
            f"{sorted(entry['distinct_raw_cells_per_output_cell'])} raw cells; the "
            f"kernel's resolve_D reads exactly one and this shape would not fit it")
    race = race_census(rows, facts, capture["pre_launch"], capture["resolved"])
    assert race["foreign_taps"] > 0, (
        f"{name}: the off-diagonal stencil reads no foreign cell on this fixture, so "
        f"the case measures nothing about a stencil weld")

    nulls: Dict[str, Any] = {}

    def null(tag: str, reachable: bool, reason: str = "", **arm: Any) -> None:
        if not reachable:
            nulls[tag] = {"predicted_null": True, "reason": reason}
            return
        restore(fields, pre)
        scheme_step(fields, pml, facts, **arm)
        moved = compare(ref1, frozen(fields))
        nulls[tag] = {"differing": sum(moved.values()), "volumes": len(moved)}
        assert moved, (f"{name}/{tag}: the null control did not diverge; the "
                       f"identity above is vacuous for this arm")

    null("rotation_skipped", True, skip_rotation=True)
    null("parity_dropped", live["odd"],
         "every live parity is +1 on this fixture; the drop is the identity",
         drop_parity=True)
    null("near_redirect_row_wrong", live["near"],
         "no folded axis: no near redirect exists", near_source_row=1)
    null("far_redirect_dropped", live["far"],
         "no stored-past-owned axis: no far ghost exists", drop_far=True)
    null("clear_order_flipped", live["odd"] and live["wall"],
         "needs an odd parity meeting a cleared plane; not reachable here",
         flip_clear_order=True)
    # THIS PROBE'S OWN ARM. The redirect's SOURCE read from pre-launch D instead of
    # from the recomputed stepped value -- the defect a weld makes when it treats the
    # image plane as carrying state rather than as a view of the step's own output.
    null("halo_source_unstepped", live["near"] or live["far"],
         "no fold: the resolution never redirects, so there is no halo source to "
         "take from the wrong array",
         halo_source_unstepped=True)
    # THE SCRATCH ITSELF IS NOT ARMABLE ON THIS WALK, and saying so is the point.
    # ``scratch_aliased_to_storage`` is a defect about WHEN a load lands relative to
    # another block's store; this walk resolves the whole D volume before the array
    # path's update_E reads any of it, so every tap sees a fully resolved array
    # whichever buffer it names. The deterministic host analogue -- resolving in
    # place in C-raster order (:func:`resolved_volume`'s ``in_place`` arm) -- is the
    # IDENTITY on every fixture here and is recorded as such rather than shipped as a
    # null that always predicts: the redirect sources are stored row 2 on a near axis
    # and the reflect row on the own axis, and neither is itself moved by a redirect
    # that raster order has already applied. The real arm is the device gate's, where
    # binding the scratch to the storage makes the two loads distinguishable.
    nulls["scratch_aliased_to_storage"] = {
        "predicted_null": True,
        "reason": "not expressible on a walk that resolves the whole volume before "
                  "the constitutive reads it; the in-place raster arm is measured "
                  "below and is the identity. Armed on the device gate instead",
        "in_place_raster_differing": in_place_raster_differing(
            facts, capture["raw"], capture["resolved"]),
    }

    record = {
        "case": name, "family": family,
        "grid": facts["shape"], "rows": {k: list(v) for k, v in rows.items()},
        "folded": facts["folded"], "walled": facts["walled"],
        "far_axes": facts["far_axes"], "phases": facts["phases"],
        "reflect_rows": facts["reflect"],
        "arms": arms,
        "step1_differing": step1, "step2_differing": step2,
        "tap_ledger": ledger, "race_census": race,
        "nulls": nulls,
        "elapsed_s": round(time.time() - started, 2),
    }
    ok = not step1 and not step2
    note = ", ".join("%s:%s" % (key, value.get("differing", "predicted"))
                     for key, value in nulls.items())
    log(f"  {name:<34} {'IDENTICAL' if ok else 'DIVERGED ' + str(step1)}"
        f"  hazardous={race['hazardous_taps']}/{race['foreign_taps']}"
        f"  nulls={{{note}}} ({record['elapsed_s']} s)")
    assert ok, (name, step1, step2)
    return record


# ---------------------------------------------------------------------------
# The fixtures -- the 2026-09-01 Metal taxonomy, on the CUDA arms
# ---------------------------------------------------------------------------

_ONE_ROW: Dict[str, Sequence[str]] = {"Ex": ("Ey",)}
_ALL_ROWS: Dict[str, Sequence[str]] = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"),
                                       "Ez": ("Ex", "Ey")}

CASES: Tuple[Tuple[str, str, Callable[[], Tuple[Any, Any]],
                   Dict[str, Sequence[str]], Dict[str, bool]], ...] = (
    # ----- Family A: PML curl x off-diagonal update_E, UNFOLDED (8 instances) -----
    ("A_pml_offdiag_one_row", "A",
     lambda: matrix.cart(rows=_ONE_ROW), _ONE_ROW, dict(near=False)),
    ("A_pml_offdiag_all_rows", "A",
     lambda: matrix.cart(rows=_ALL_ROWS), _ALL_ROWS, dict(near=False)),
    ("A_pml_offdiag_walls_all_rows", "A",
     lambda: matrix.cart(rows=_ALL_ROWS, boundaries="metallic"), _ALL_ROWS,
     dict(near=False, wall=True)),
    # ----- Family B: PML curl x FOLDED off-diagonal update_E (9 instances) --------
    ("B_fold_even", "B",
     lambda: matrix.folded(rows=_ONE_ROW), _ONE_ROW, dict(near=True, far=True)),
    ("B_fold_odd", "B",
     lambda: matrix.folded(phase=-1, rows=_ONE_ROW), _ONE_ROW,
     dict(near=True, far=True, odd=True)),
    ("B_fold_odd_all_rows", "B",
     lambda: matrix.folded(phase=-1, rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, far=True, odd=True)),
    ("B_fold_odd_count", "B",
     lambda: matrix.folded(rows=_ALL_ROWS, extent=2.1), _ALL_ROWS,
     dict(near=True, far=True)),
    # THE OTHER TERMINATION. A folded METALLIC axis has no far ghost at all
    # (``_stored_past_owned`` is False), so the far machinery is dead and the near
    # fill is the whole of the fold -- a mutation caught on one termination is not
    # caught on the other.
    ("B_fold_metallic_termination", "B",
     lambda: matrix.folded(boundaries={"y": "metallic"}, rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, far=False)),
    # THE ONE SHAPE THAT ARMS ``clear_order_flipped``: an ODD parity meeting a
    # CLEARED plane, which needs a wall on an axis the fold does not own.
    ("B_fold_odd_walled", "B",
     lambda: matrix.folded(phase=-1, boundaries={"x": "metallic"}, rows=_ALL_ROWS),
     _ALL_ROWS, dict(near=True, odd=True, wall=True)),
    ("B_fold_two_axes_mixed_phase", "B",
     lambda: matrix.folded(axis="XY", phase=(1, -1), rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, corner=True, odd=True)),
    ("B_fold_3d", "B",
     lambda: matrix.folded(rows=_ALL_ROWS, depth=1.2), _ALL_ROWS,
     dict(near=True, far=True)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write probe.json and progress.jsonl into")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trail = (args.out / "progress.jsonl").open("w")

    log("=" * 78)
    log("THE HAND-CUDA STENCIL WELD -- host arithmetic, no device")
    log("=" * 78)
    log(f"policy      : {os.environ['MEEP_GPU_SUBNORMAL_POLICY']}")
    log(f"cases       : {len(CASES)}")
    started = time.time()
    records: List[Dict[str, Any]] = []
    for index, (name, family, build, rows, expect) in enumerate(CASES, start=1):
        record = run_case(name, family, build, rows, args.seed + index, expect)
        record["case_index"] = f"{index}/{len(CASES)}"
        records.append(record)
        trail.write(json.dumps(record) + "\n")
        trail.flush()
        os.fsync(trail.fileno())
    trail.close()

    payload = {
        "probe": "cuda offdiag scratch-output weld -- host arithmetic",
        "claim": "the per-cell closed-form resolution of the raw stepped D equals the "
                 "driver's fill_symmetry_bc_D / zero_metal_D / fill_folded_far_ghosts_D "
                 "pass order byte for byte over two complete steps, on every fixture "
                 "the two CUDA off-diagonal cells' arms admit; each resolved cell "
                 "consults exactly ONE raw cell; and the foreign taps the in-place weld "
                 "would race on are counted rather than argued",
        "certifies": "NOTHING. NumPy is not NVRTC. The substitution, the launch counts "
                     "and the mutation battery belong to the device gate",
        "policy": os.environ["MEEP_GPU_SUBNORMAL_POLICY"],
        "cases": records,
        "totals": {
            "cases": len(records),
            "identical_step1_and_step2": sum(
                1 for r in records if not r["step1_differing"] and not r["step2_differing"]),
            "foreign_taps": sum(r["race_census"]["foreign_taps"] for r in records),
            "hazardous_taps": sum(r["race_census"]["hazardous_taps"] for r in records),
            "nulls_armed": sum(1 for r in records for v in r["nulls"].values()
                               if "differing" in v),
            "nulls_predicted_unreachable": sum(
                1 for r in records for v in r["nulls"].values()
                if v.get("predicted_null")),
        },
        "elapsed_s": round(time.time() - started, 2),
    }
    (args.out / "probe.json").write_text(json.dumps(payload, indent=1) + "\n")
    log("-" * 78)
    log(f"identical   : {payload['totals']['identical_step1_and_step2']}"
        f"/{payload['totals']['cases']}")
    log(f"foreign taps: {payload['totals']['hazardous_taps']} hazardous of "
        f"{payload['totals']['foreign_taps']}")
    log(f"nulls       : {payload['totals']['nulls_armed']} armed, "
        f"{payload['totals']['nulls_predicted_unreachable']} predicted unreachable")
    log(f"wrote       : {args.out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
