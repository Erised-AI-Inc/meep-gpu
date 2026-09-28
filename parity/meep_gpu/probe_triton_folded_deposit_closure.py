"""Does the FOLDED Triton fused bracket need ``deposit_repair``'s IMAGE CLOSURE?

DEVICE STATUS: **RELEASED 2026-08-30** on the GPU host, RTX A6000 GPU 5, Triton 3.1.0 /
    CuPy 13.5.1, under BOTH subnormal policies —
    ``parity/meep_gpu/results/triton_folded_deposit_closure_2026-08-30/``.
    11 armed cases counted and 1 case measured VACUOUS BY DESIGN, identically under
    ``keep`` (``ieee_keep_ftz_stripped``) and ``flush`` (``meep_x86_flush``): leg A
    bit-identical to the CuPy array path over every allocated volume after every
    complete ``driver.step()``, leg B (point-only repair) diverging on all 11 from
    step 1, leg C (no repair) diverging on all 12. One fused launch per step on
    every leg, counted through the plan's own kernel object.

    THREE RUNS WERE NEEDED AND THE FIRST TWO FAILED HERE, not on arithmetic.
    ``backends.select_backend`` does not exist (the entry point is
    ``guard_kernel_compilation``); the product modules are not registered in
    ``sys.modules`` under their dotted names by the composer's own import path, so
    they must be imported by name; and ``flush`` is unattainable unless MEEP is
    already imported (it owns the only host FTZ knob this package may use) AND
    ``CUPY_ACCELERATORS`` is empty before CuPy is imported (CuPy 13.5.1 fixes its
    reduction dispatch at import). All three are now carried by the probe and its
    runner rather than by whoever runs it next.

WHY THIS GATE EXISTS. On 2026-08-30 three Triton products flipped
``CARRIES_DEPOSIT_REPAIR`` False -> True — :mod:`~meep_gpu.triton_kernels.folded_fused_pair`,
:mod:`~meep_gpu.triton_kernels.folded_fused_magnetic_pair` and
:mod:`~meep_gpu.triton_kernels.complex_fused_magnetic_pair` — discharging 78 clause-refusals.
The device evidence offered for that flip was ``probe_triton_symmetry_composition``, and
it is VACUOUS for the closure on three counts, each checked in this file's
``--no-device`` leg rather than asserted:

  1. its ``run_case`` plans with ``fuse=False``, so the flag was INERT in what it measured;
  2. four of its six cases declare ``"sources": []``, so there is no deposit to repair;
  3. both source cases put the source at center ``(0, 0, 0)``, which lands on stored row 1
     of the folded axis — a row NO fill reads from — so ``repair_cells`` returns the
     deposit and nothing else and a point-only repair is exact THERE BY CONSTRUCTION.

The mechanism the flip actually rests on is the third one. ``deposit_repair.fill_image_rules``
/ ``repair_cells`` (deposit_repair.py:217-258) extend the saved and restored cell set to the
cells the driver's post-injection ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*``
image a deposit into. A point-only repair leaves those images holding the constitutive
result the fused launch computed from the PRE-INJECTION field. That is what this gate
executes.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT, per case:

    With a folded grid, a deposit ON a row a post-injection fill READS FROM, and the
    SHIPPED bracket (``launch._install_fused_pair`` -> ``LeadingRepairPlan`` /
    ``TrailingRepairPlan``) installed at the driver's OWN ``fast.dispatch`` consults,
    one launch of the folded fused kernel per complete ``driver.step()`` leaves every
    allocated volume BIT-IDENTICAL, as uint32, to the CuPy array path — and WITHOUT the
    image closure it does not.

THE THREE LEGS, and rule 3 of the operating rules makes the second and third mandatory:

    A   fused + the shipped ``repair_cells``            -> expect 0 differing words
    B   fused + a POINT-ONLY ``repair_cells``           -> MUST DIVERGE
    C   fused + NO repair (trailing slot made inert)    -> MUST DIVERGE

Leg B is the armed control for the CLOSURE; leg C is the armed control for the repair as
a whole. A case whose leg B does not diverge cannot tell the closure from a point repair:
it is reported ``vacuous_for_closure`` and is NOT counted toward the release. One case is
deliberately built to be exactly that — :data:`ON_PLANE_CONTROL_CASE`, the placement the
2026-08-30 evidence used — so the vacuity is a MEASURED row in this artifact rather than a
claim about another probe.

WHAT DISTINGUISHES A CASE THAT CAN DISCRIMINATE, decided by construction and re-checked
empirically. ``repair_cells`` on the INJECTED component's own constitutive source must
return at least one cell that is not already a deposit point. Both numbers are recorded
per case (``closure_extra_cells``), and a case claiming to be armed whose leg B then does
not diverge is a finding, not a pass — the verdict reports it.

WHAT THIS GATE DOES NOT LICENSE. Nothing about throughput. Nothing about
``fastpath.plan_fast_path``, which still returns ``None`` — the bracket is installed at
the driver's consult points by this harness, exactly as every other Triton composition
gate installs one. Nothing about the OFF-DIAGONAL, NONLINEAR or CYLINDRICAL r = 0 seams,
which ``deposit_repair.repairable`` still refuses by name. And nothing about
:mod:`~meep_gpu.triton_kernels.complex_fused_magnetic_pair`: its own predicate refuses
``grid.has_symmetry()`` outright, so no case it admits can reach a fill at all — see
:func:`unfolded_product_leg`, which measures that refusal rather than assuming it.

Usage::

    # laptop, no CUDA and no Triton — arming, predicate reasons, host-emulated legs
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_deposit_closure.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_deposit_closure.py \\
        --subnormal-policy keep \\
        --out parity/meep_gpu/results/<fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SEED = 20260830

#: One dx at resolution 12 on the cells below, spelled once. Stored row ``r`` of a folded
#: axis sits at ``(r - 1) * DX`` — MEASURED by scanning the source's own ``_point_iy``
#: (see :func:`deposit_rows`), never assumed: every case re-checks the row it landed on
#: and :func:`arming_census` refuses a case whose deposit missed the imaged row.
DX = 1.0 / 12.0

#: The near fill's source row, and the far ghost's, are read from the engine at case
#: build time. These are the CENTERS that put a deposit on them at resolution 12 —
#: ``stepping.MIRROR_SOURCE_INDEX`` is 2, so ``1 * DX``; the folded-periodic reflect row
#: on a 3.0 um cell is 18, so ``17 * DX``.
NEAR_ROW_CENTER = 1 * DX
FAR_ROW_CENTER = 17 * DX

CELL_2D = (3.0, 3.0, 0.0)
CELL_2D_ODD = (3.0, 3.0833333333333335, 0.0)
CELL_3D = (1.5, 1.5, 1.5)

METALLIC_2D = ("metallic", "metallic", "periodic")
PERIODIC_2D = ("periodic", "periodic", "periodic")
METALLIC_3D = ("metallic", "metallic", "metallic")

#: name, cell, boundaries, mirrors, source declaration, seam, steps, why.
#:
#: EVERY REQUIRED PROPERTY IS CARRIED BY A NAMED CASE, not by a sweep that might happen
#: to cover it: a fold on each of two axes and a two-fold pair; a deposit off the mirror
#: plane AND on a row a fill reads from; both seams; a far ghost (folded PERIODIC) as well
#: as near fills; an odd mirror phase; a 3-D fold, because a 2-D case cannot reach the z
#: half of either kernel's carry; and the corpus-realistic SHEET spanning the folded axis.
#:
#: THE COMPONENT IS NOT FREE. Which fill images a deposit is decided by the DEPOSITED
#: component's Yee shifts, so on a Y fold ``Ez``/``Ex`` (shift 0 on y) reach the NEAR fill
#: and ``Ey`` (shift 1) reaches the FAR one; ``Ey`` on a METALLIC fold reaches neither and
#: would be a case that looks armed and is not. :func:`arming_census` is what enforces it.
CASES: Tuple[Tuple[str, Any, Any, Any, Dict[str, Any], str, int, str], ...] = (
    ("near_metallic_Ez_row2", CELL_2D, METALLIC_2D, (("Y", 1),),
     {"component": "Ez", "frequency": 0.7, "center": (0.1, NEAR_ROW_CENTER, 0.0),
      "width": 0.4}, "D", 12,
     "electric seam, near mirror fill: Dz shift 0 on y, deposit on stored row 2"),
    ("near_metallic_odd_Ez_row2", CELL_2D, METALLIC_2D, (("Y", -1),),
     {"component": "Ez", "frequency": 0.7, "center": (0.1, NEAR_ROW_CENTER, 0.0),
      "width": 0.4}, "D", 12,
     "the same near fill under an ODD mirror phase: the image carries -1, so a "
     "closure that dropped the parity would show here and not on the even rows"),
    ("far_periodic_Ey_row18", CELL_2D, PERIODIC_2D, (("Y", 1),),
     {"component": "Ey", "frequency": 0.7, "center": (0.1, FAR_ROW_CENTER, 0.0),
      "width": 0.4}, "D", 12,
     "electric seam, FAR ghost: Dy shift 1 on y, deposit on the reflect row"),
    ("far_periodic_odd_count_Ey", CELL_2D_ODD, PERIODIC_2D, (("Y", 1),),
     {"component": "Ey", "frequency": 0.7, "center": (0.1, FAR_ROW_CENTER, 0.0),
      "width": 0.4}, "D", 12,
     "the far ghost at an ODD full count, where _far_reflect_rows is n-3 and not n-2"),
    ("near_metallic_Hy_row2", CELL_2D, METALLIC_2D, (("Y", 1),),
     {"component": "Hy", "frequency": 0.7, "center": (0.1, NEAR_ROW_CENTER, 0.0),
      "width": 0.4}, "B", 12,
     "MAGNETIC seam, near mirror fill: By shift 0 on y"),
    ("far_periodic_Hx_row18", CELL_2D, PERIODIC_2D, (("Y", 1),),
     {"component": "Hx", "frequency": 0.7, "center": (0.1, FAR_ROW_CENTER, 0.0),
      "width": 0.4}, "B", 12,
     "MAGNETIC seam, FAR ghost: Bx shift 1 on y"),
    ("twofold_metallic_Ez_corner", CELL_2D, METALLIC_2D, (("X", 1), ("Y", 1)),
     {"component": "Ez", "frequency": 0.7,
      "center": (NEAR_ROW_CENTER, NEAR_ROW_CENTER, 0.0), "width": 0.4}, "D", 12,
     "TWO FOLDS: Dz carries a near rule on both axes, so one deposit has three "
     "images and the corner one is reached only by the pair of rules together"),
    ("twofold_periodic_Ez_corner", CELL_2D, PERIODIC_2D, (("X", 1), ("Y", 1)),
     {"component": "Ez", "frequency": 0.7,
      "center": (NEAR_ROW_CENTER, NEAR_ROW_CENTER, 0.0), "width": 0.4}, "D", 12,
     "two folds over PERIODIC outer declarations: near and far rules coexist on "
     "the same component set"),
    ("threeD_zfold_metallic_Ex", CELL_3D, METALLIC_3D, (("Z", 1),),
     {"component": "Ex", "frequency": 0.7, "center": (0.1, 0.1, NEAR_ROW_CENTER),
      "width": 0.4}, "D", 6,
     "a 3-D fold on z: in a 2-D run the whole z half of the carry is compile-time "
     "absent, so no 2-D case reaches it"),
    ("sheet_periodic_Ez_spanning", CELL_2D, PERIODIC_2D, (("Y", 1),),
     {"component": "Ez", "frequency": 0.7, "center": (0.1, 0.0, 0.0),
      "size": (0.0, 3.0, 0.0), "width": 0.4}, "D", 12,
     "THE CORPUS-REALISTIC SHAPE: a planewave/eigenmode-style SHEET spanning the "
     "folded axis, which contains the near fill's source row by construction"),
    ("sheet_periodic_Hy_spanning", CELL_2D, PERIODIC_2D, (("Y", 1),),
     {"component": "Hy", "frequency": 0.7, "center": (0.1, 0.0, 0.0),
      "size": (0.0, 3.0, 0.0), "width": 0.4}, "B", 12,
     "the same sheet on the MAGNETIC seam. Hy is the component the even Y plane "
     "leaves even; Hx/Hz on a sheet centred on the plane are refused by name "
     "(sources._validate_symmetry_parity), which is a refusal and not a gap"),
    ("on_plane_Ez_row1_CONTROL", CELL_2D, METALLIC_2D, (("Y", 1),),
     {"component": "Ez", "frequency": 0.7, "center": (0.1, 0.0, 0.0), "width": 0.4},
     "D", 12,
     "THE VACUOUS PLACEMENT, executed on purpose: center (0,0,0) is the placement "
     "the 2026-08-30 evidence used. It lands on stored row 1, which no fill reads, "
     "so the closure adds no cell and leg B CANNOT diverge. Expected vacuous."),
)

#: The one case above that is expected to be vacuous for the closure. Named, so a run in
#: which it accidentally armed — or in which an armed case went quiet — is a reportable
#: surprise rather than an unremarked change in the pass count.
ON_PLANE_CONTROL_CASE = "on_plane_Ez_row1_CONTROL"

#: Per seam: which slots the pair owns, which component a source of that family deposits
#: into, and which product module implements the folded pair. Mirrors
#: ``launch.FOLDED_FUSED_PAIRS`` rather than restating a second table of slot names.
SEAM_SPEC: Dict[str, Dict[str, Any]] = {
    "D": {"curl": "step_D", "update": "update_E",
          "module": "meep_gpu.triton_kernels.folded_fused_pair",
          "plan_class": "FoldedFusedPairPlan",
          "kernel_factory": "folded_fused_curl_constitutive_D_kernel",
          "coverage": "folded_fused_pair_coverage",
          "target_of": {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"},
          "emulates": ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                       "fill_folded_far_ghosts_D", "update_E")},
    "B": {"curl": "step_B", "update": "update_H",
          "module": "meep_gpu.triton_kernels.folded_fused_magnetic_pair",
          "plan_class": "FoldedFusedMagneticPairPlan",
          "kernel_factory": "folded_fused_curl_constitutive_B_kernel",
          "coverage": "folded_fused_magnetic_pair_coverage",
          "target_of": {"Hx": "Bx", "Hy": "By", "Hz": "Bz"},
          "emulates": ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                       "fill_folded_far_ghosts_B", "update_H")},
}


def log(message: str) -> None:
    print(f"[closure] {message}", flush=True)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    names = (
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/sources.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/folded_fused_pair.py",
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


def save(payload: Dict[str, Any], path: str) -> None:
    """Write the artifact, STAMPED with the shared provenance keys.

    ``gate_provenance.stamp`` adds two things this probe's own bookkeeping does
    not, and both are what makes the result WELDABLE rather than merely readable:

    * ``imported_source_sha256`` — every repo module this process actually
      imported, which is a DIFFERENT FACT from the curated ``source_sha256``
      above (that one is the list this gate MEANS to bind). The distinct key is
      deliberate; see that function's docstring for the two gates that died on
      the collision when one name carried both.
    * ``canonical_verdict`` — this payload's ``verdict.released`` normalised into
      the one shape every reader in the fleet understands.

    WHY IT IS HERE AND NOT ONLY AT THE END. ``rebind_triton_welds.py`` reads
    ``canonical_verdict`` (then ``release``) and takes each welded digest out of
    ``imported_source_sha256``; without both keys this gate's artifact cannot be
    bound into ``triton_kernels/fingerprints.json`` at all, so the deposit-closure
    evidence would stay a file on disk that the record does not cite. Stamping
    inside ``save`` rather than once at the end keeps the partial artifacts (rule
    7) honest too: a run killed mid-case leaves a payload whose verdict and
    imported set describe the cases that actually ran.

    ``stamp`` tolerates the import set GROWING across the progressive saves and
    raises only if a digest for an already-recorded path CHANGES — a source
    rewritten mid-run, where the measurements on either side describe different
    programs. That is the check we want, so it is not caught here.
    """
    import gate_provenance  # noqa: PLC0415

    gate_provenance.stamp(payload)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# The engine objects
# ---------------------------------------------------------------------------

def build_driver(cell, boundaries, mirrors, declaration, xp, seed: int = SEED):
    """One folded PML driver with ONE in-seam source, seeded identically per leg."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    dimensions = 3 if float(cell[2]) > 0.0 else 2
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=xp is not None, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon: the constitutive half multiplies by inverse epsilon at the
    # IMAGED cell's own index, and a uniform material would make a repair that read it
    # at the deposit's index indistinguishable from one that read it at the image's.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(xp.asarray(epsilon) if xp is not None else epsilon)
    folded = {axis.lower() for axis, _phase in mirrors}
    layers: Dict[str, Any] = {
        "x": {"high": 5} if "x" in folded else 5,
        "y": {"high": 5} if "y" in folded else 5,
    }
    if dimensions == 3:
        layers["z"] = {"high": 5} if "z" in folded else 5
    driver.setup_pml(layers)
    driver.add_source(dict(declaration))
    generator = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        values = np.ascontiguousarray(
            generator.uniform(-0.25, 0.25, size=shape).astype(np.float32))
        driver.set_field(name, xp.asarray(values) if xp is not None else values)
    # THE AUXILIARIES TOO. Zero-init is a FIXED POINT of the constitutive recurrence
    # `field + kps*fresh - kms*fw_prev`, so a run that seeded only the primaries could
    # pass without exercising the `f_w` state the repair saves and restores.
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        values = np.ascontiguousarray(
            generator.uniform(-0.05, 0.05, size=shape).astype(np.float32))
        array[...] = xp.asarray(values) if xp is not None else values
    return driver


def inventory(driver) -> Dict[str, Any]:
    """Every allocated volume the step can touch, found by SCANNING rather than listing.

    A hand-written list is a list that stops covering a volume the engine adds; the scan
    plus the ``REQUIRED`` floor below is what keeps the comparison complete.
    """
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found = {name: value for name, value in vars(fields).items()
             if not name.startswith("__")
             and getattr(value, "shape", None) == shape
             and getattr(value, "dtype", None) is not None}
    required = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    missing = [name for name in required if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not complete")
    return found


def words(array) -> np.ndarray:
    """Raw bits, COPIED.

    The copy is load-bearing, not hygiene. ``np.ascontiguousarray`` on an already
    contiguous host array returns that same array and ``.view()`` returns a view of it,
    so a snapshot taken without the copy ALIASES the live volume: every per-step snapshot
    would hold the final state and the step-by-step comparison would compare the last step
    against itself N times. Caught on this harness's own first run — the moved-state
    census reported 0 arrays ever moved on a run whose fields plainly moved.
    """
    host = array.get() if hasattr(array, "get") else array
    return np.ascontiguousarray(host).view(np.uint32).ravel().copy()


def snapshot(driver) -> Dict[str, np.ndarray]:
    return {name: words(array) for name, array in inventory(driver).items()}


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    only = sorted(set(left) ^ set(right))
    if only:
        return {"array": "<inventory>", "reason": "asymmetric", "names": only}
    for name in sorted(left):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape",
                    "left": list(a.shape), "right": list(b.shape)}
        if not np.array_equal(a, b):
            where = int(np.flatnonzero(a != b)[0])
            return {"array": name, "flat_index": where,
                    "left_word": int(a[where]), "right_word": int(b[where]),
                    "differing_words": int(np.count_nonzero(a != b))}
    return None


def differing_words(left: Dict[str, np.ndarray],
                    right: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {name: int(np.count_nonzero(left[name] != right[name]))
            for name in sorted(set(left) & set(right))
            if not np.array_equal(left[name], right[name])}


def moved_arrays(before: Dict[str, np.ndarray],
                 after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


# ---------------------------------------------------------------------------
# Arming: does this case's deposit actually land on a row a fill reads from?
# ---------------------------------------------------------------------------

def deposit_rows(source) -> Dict[str, List[int]]:
    from meep_gpu import deposit_repair  # noqa: PLC0415

    index = deposit_repair._deposit_index(source)
    if index is None:
        return {}
    out = {}
    for axis, column in zip("xyz", index):
        host = column.get() if hasattr(column, "get") else column
        out[axis] = sorted({int(value) for value in np.asarray(host).ravel()})
    return out


def arming_census(driver, seam: str, component: str) -> Dict[str, Any]:
    """How many cells the CLOSURE adds beyond the deposit, on the INJECTED component.

    THE COMPONENT MATTERS AND THE OTHER TWO DO NOT. ``deposit_repair.save`` repairs all
    three constitutive targets at the deposit index, and two of them are cells the
    injection never changed — recomputing those is bit-identical either way (the module's
    own "over-covering is safe" note). So a census over all three would report a case as
    armed whenever ANY component had an image rule, including cases where the injected
    component has none. This counts only the injected component's own target, which is
    the one whose image the injection actually moves.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    spec = SEAM_SPEC[seam]
    target = spec["target_of"][component]
    seam_sources = deposit_repair.in_seam_sources(tuple(driver._sources), seam)
    record: Dict[str, Any] = {
        "injected_component": component,
        "constitutive_target": target,
        "in_seam_sources": len(seam_sources),
        "fill_image_rules": [list(rule) for rule in
                             deposit_repair.fill_image_rules(driver.fields, target)],
        "deposit_rows": {},
        "deposit_cells": 0,
        "closure_cells": 0,
        "closure_extra_cells": 0,
    }
    if not seam_sources:
        record["why_not_armed"] = (
            f"no source is injected in the {seam} seam, so there is no deposit and the "
            f"closure has nothing to close over")
        return record
    source = seam_sources[0]
    index = deposit_repair._deposit_index(source)
    if index is None:
        record["why_not_armed"] = "the source publishes no deposit index"
        return record
    record["deposit_rows"] = deposit_rows(source)
    cells = deposit_repair.repair_cells(driver.fields, target, index)

    def as_set(columns):
        host = [np.asarray(c.get() if hasattr(c, "get") else c).ravel().tolist()
                for c in columns]
        return set(zip(*host))

    points, closure = as_set(index), as_set(cells)
    record["deposit_cells"] = len(points)
    record["closure_cells"] = len(closure)
    record["closure_extra_cells"] = len(closure - points)
    if not record["closure_extra_cells"]:
        record["why_not_armed"] = (
            f"repair_cells on {target} returns no cell outside the deposit itself, so a "
            f"POINT-ONLY repair is exact here by construction and leg B cannot diverge")
    return record


# ---------------------------------------------------------------------------
# The bracket, installed where the DRIVER asks for it
# ---------------------------------------------------------------------------

class CountingKernel:
    """Owns the JIT kernel and counts every launch made through the plan's ``run``.

    BYTES ALONE CANNOT PROVE THE FUSED PATH RAN: a silent fall-back to the array path is
    byte-identical to the array path by construction. The count is what separates "the
    kernel reproduced the array path" from "the kernel never ran".
    """

    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit, self.calls, self.grids = jit, 0, []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*args, **kwargs):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*args, **kwargs)

        return run


class InertPlan:
    """What leg C puts in the trailing slot: the repair, WITHHELD.

    Not a ``NoopPlan``: that sentinel means "the pair absorbed this sub-step and there was
    nothing to repair", which is the truth on a sourceless seam. Here there IS a deposit
    and the repair is deliberately not run, so the class is named for what it is.
    """

    __slots__ = ("slot", "absorbed_by")

    def __init__(self, slot: str, absorbed_by: Any) -> None:
        self.slot, self.absorbed_by = slot, absorbed_by

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class DispatchAdapter:
    """The object the driver consults, holding one leg's plans.

    THE DRIVER'S OWN CONSULT POINTS, not a monkeypatch of ``stepping``. ``FdtdDriver.step``
    asks ``fast.dispatch(name, fields)`` at five sites and calls the array function only
    when the answer is False (driver.py:3292-3315), so installing here exercises the
    engine's real substitution seam — including its contract that True means the WHOLE
    sub-step ran, which is what makes a fused pair's two consults the save and the repair.
    """

    __slots__ = ("plans", "owner", "consults", "dispatched")

    def __init__(self, plans: Dict[str, Any], owner: Any) -> None:
        self.plans, self.owner = plans, owner
        self.consults: Dict[str, int] = {}
        self.dispatched: Dict[str, int] = {}

    def dispatch(self, name: str, fields: Any) -> bool:
        self.consults[name] = self.consults.get(name, 0) + 1
        if fields is not self.owner:
            return False
        entry = self.plans.get(name)
        if entry is None:
            return False
        entry.run()
        self.dispatched[name] = self.dispatched.get(name, 0) + 1
        return True


def install_adapter(driver, plans) -> None:
    """Freeze this driver on ``plans`` before its first step.

    ``FdtdDriver.step`` re-plans only while ``_fast_path_stale``; clearing the flag with
    the adapter in place is what stops ``plan_fast_path`` (which returns ``None`` on every
    branch today) from replacing it at step 1.
    """
    driver._fast_path = DispatchAdapter(plans, driver.fields)
    driver._fast_path_stale = False


class HostFoldedPair:
    """The folded fused product's five passes, for a laptop leg with no Triton.

    STANDS IN FOR THE KERNEL AT THE FIRST CONSULT ONLY, and only under ``--no-device``.
    It runs exactly the product's declared ``REPLACES`` list against the same
    pre-injection field one launch would, so what it changes is whether the arithmetic the
    repair brackets came from one launch or five array calls — which is precisely the
    difference the device leg exists to close. Nothing it measures is device evidence.
    """

    __slots__ = ("fields", "pml", "seam", "runs")

    def __init__(self, fields: Any, pml: Any, seam: str) -> None:
        self.fields, self.pml, self.seam, self.runs = fields, pml, seam, 0

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        from meep_gpu import stepping  # noqa: PLC0415

        self.runs += 1
        for name in SEAM_SPEC[self.seam]["emulates"]:
            function = getattr(stepping, name)
            try:
                function(self.fields, self.pml)
            except TypeError:
                function(self.fields)


def point_only_repair_cells(fields: Any, target: str, index: Tuple[Any, Any, Any]):
    """LEG B's ``repair_cells``: the deposit points, and NOTHING ELSE.

    This is the pre-2026-08-30 behaviour restored for one leg — the repair the flip's
    predicate change would have licensed if the closure did not exist. It is the armed
    control for the closure and must make every armed case diverge.
    """
    xp = fields.grid.xp
    return tuple(xp.asarray(part).reshape(-1) for part in index)


def build_leg_plans(driver, seam: str, leg: str, device: bool,
                    counters: Dict[str, CountingKernel]) -> Dict[str, Any]:
    """The SHIPPED bracket for one leg, or the host emulation of it.

    THE BRACKET IS NOT REBUILT HERE. On the device the plans come from
    ``launch.plan_step(..., fuse=True)``, which routes through
    ``launch._install_folded_fused_pairs`` -> ``_install_fused_pair`` and therefore
    installs the shipped ``LeadingRepairPlan`` / ``TrailingRepairPlan``. A harness that
    assembled its own bracket could not license the one that ships.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    spec = SEAM_SPEC[seam]
    curl_name, update_name = spec["curl"], spec["update"]
    if device:
        from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

        plan = plan_step(driver.fields, driver.pml, fuse=True,
                         sources=tuple(driver._sources), num_warps=1)
        leading = plan.plans.get(curl_name)
        trailing = plan.plans.get(update_name)
        if type(leading).__name__ != "LeadingRepairPlan":
            raise AssertionError(
                f"{seam} seam: the composer put {type(leading).__name__} in {curl_name}, "
                f"not the LeadingRepairPlan the deposit bracket is made of; refusals="
                f"{ {k: list(v) for k, v in plan.reasons.items()} }")
        if type(trailing).__name__ != "TrailingRepairPlan":
            raise AssertionError(
                f"{seam} seam: the composer put {type(trailing).__name__} in "
                f"{update_name}, not the TrailingRepairPlan that runs the repair")
        inner = leading.inner
        if type(inner).__name__ != spec["plan_class"]:
            raise AssertionError(
                f"{seam} seam: the bracket wraps {type(inner).__name__}, not the "
                f"{spec['plan_class']} this gate is about")
        # IMPORTED BY NAME, not read out of ``sys.modules``: the composer reaches these
        # modules through ``launch._folded_fused_pair_entries``, and whether that leaves
        # them registered under their dotted name is an implementation detail this gate
        # must not depend on. It did not, and the first device run died on the KeyError.
        module = importlib.import_module(spec["module"])
        counting = CountingKernel(getattr(module, spec["kernel_factory"])())
        inner._kernel = counting
        counters[leg] = counting
        plans = {curl_name: leading, update_name: trailing}
        if leg == "C":
            plans[update_name] = InertPlan(update_name, inner)
        return plans

    inner = HostFoldedPair(driver.fields, driver.pml, seam)
    seam_sources = deposit_repair.in_seam_sources(tuple(driver._sources), seam)
    if not seam_sources:
        raise AssertionError(
            f"{seam} seam: no in-seam source, so this case cannot exercise a deposit")
    leading = deposit_repair.LeadingRepairPlan(inner, driver.fields, driver.pml,
                                              seam_sources, seam)
    trailing = deposit_repair.TrailingRepairPlan(update_name, leading, driver.fields,
                                                 driver.pml)
    return {curl_name: leading,
            update_name: InertPlan(update_name, inner) if leg == "C" else trailing}


def run_one_leg(case, leg: str, device: bool, xp) -> Dict[str, Any]:
    """One leg of one case: build, install, step, and snapshot after EVERY step."""
    from meep_gpu import deposit_repair  # noqa: PLC0415

    name, cell, boundaries, mirrors, declaration, seam, steps, _why = case
    driver = build_driver(cell, boundaries, mirrors, declaration, xp)
    counters: Dict[str, CountingKernel] = {}
    plans = build_leg_plans(driver, seam, leg, device, counters)
    install_adapter(driver, plans)
    shipped = deposit_repair.repair_cells
    states: List[Dict[str, np.ndarray]] = []
    try:
        if leg == "B":
            deposit_repair.repair_cells = point_only_repair_cells
        for _ in range(steps):
            driver.step()
            if xp is not None:
                xp.cuda.runtime.deviceSynchronize()
            states.append(snapshot(driver))
    finally:
        deposit_repair.repair_cells = shipped
    leading = plans[SEAM_SPEC[seam]["curl"]]
    return {
        "states": states,
        "launches": counters[leg].calls if leg in counters else None,
        "host_runs": getattr(getattr(leading, "inner", None), "runs", None),
        "repairs": int(getattr(leading, "repairs", 0)),
        "adapter": driver._fast_path.dispatched,
    }


def reference_states(case, xp) -> List[Dict[str, np.ndarray]]:
    """The array path, with NOTHING installed: the driver's own order, verbatim."""
    name, cell, boundaries, mirrors, declaration, seam, steps, _why = case
    driver = build_driver(cell, boundaries, mirrors, declaration, xp)
    driver._fast_path = None
    driver._fast_path_stale = False
    out = []
    for _ in range(steps):
        driver.step()
        if xp is not None:
            xp.cuda.runtime.deviceSynchronize()
        out.append(snapshot(driver))
    return out


def run_case(case, device: bool, xp) -> Dict[str, Any]:
    name, cell, boundaries, mirrors, declaration, seam, steps, why = case
    started = time.time()
    row: Dict[str, Any] = {
        "case": name, "why": why, "seam": seam, "steps": steps,
        "cell": list(cell), "boundaries": list(boundaries),
        "mirrors": [list(m) for m in mirrors], "source": dict(declaration),
        "device": device,
    }
    try:
        probe = build_driver(cell, boundaries, mirrors, declaration, xp)
        row["shape"] = [int(n) for n in probe.shape]
        row["arming"] = arming_census(probe, seam, declaration["component"])
        row["armed_by_construction"] = bool(row["arming"]["closure_extra_cells"])
        if device:
            module = importlib.import_module(SEAM_SPEC[seam]["module"])
            verdict = getattr(module, SEAM_SPEC[seam]["coverage"])(
                probe.fields, probe.pml, tuple(probe._sources))
            row["shipped_predicate_admits"] = bool(verdict.covered)
            row["shipped_predicate_reasons"] = list(verdict.reasons)
            if not verdict.covered:
                raise AssertionError(
                    f"the SHIPPED predicate refuses this case ({list(verdict.reasons)}); "
                    f"a leg run around a live refusal would measure a configuration "
                    f"nothing claims")
        del probe

        reference = reference_states(case, xp)
        legs = {leg: run_one_leg(case, leg, device, xp) for leg in ("A", "B", "C")}

        for leg, result in legs.items():
            got = result["states"]
            if len(got) != len(reference):
                raise AssertionError(f"leg {leg}: {len(got)} steps against "
                                     f"{len(reference)} reference steps")
            divergence, at_step = None, None
            for step, (ref, cand) in enumerate(zip(reference, got), start=1):
                divergence = first_divergence(ref, cand)
                if divergence is not None:
                    at_step = step
                    break
            final = differing_words(reference[-1], got[-1])
            row[f"leg_{leg}"] = {
                "identical": divergence is None,
                "first_divergence": divergence,
                "first_divergence_at_step": at_step,
                "final_differing_words": sum(final.values()),
                "final_differing_arrays": final,
                "launches": result["launches"],
                "host_pair_runs": result["host_runs"],
                "repaired_cells_last_step": result["repairs"],
                "dispatched": result["dispatched"] if "dispatched" in result
                else result["adapter"],
            }

        # ARRAYS MUST MOVE. A no-op agreeing with a no-op is trivially identical, so a
        # leg whose compared state never changed is not evidence of anything.
        row["arrays_ever_moved"] = len(moved_arrays(reference[0], reference[-1]))
        row["arrays_total"] = len(reference[0])

        row["leg_A_identical"] = row["leg_A"]["identical"]
        row["leg_B_diverged"] = not row["leg_B"]["identical"]
        row["leg_C_diverged"] = not row["leg_C"]["identical"]
        # THE VACUITY VERDICT, and it is the whole point of the protocol. Withholding the
        # closure must change the answer; where it does not, the case cannot tell a
        # working closure from a missing one and its leg-A pass is worth nothing.
        row["vacuous_for_closure"] = not row["leg_B_diverged"]
        row["vacuous_for_repair"] = not row["leg_C_diverged"]
        if device:
            expected = steps
            row["launch_count_expected"] = expected
            row["launch_counts_correct"] = all(
                row[f"leg_{leg}"]["launches"] == expected for leg in "ABC")
        else:
            row["launch_counts_correct"] = all(
                row[f"leg_{leg}"]["host_pair_runs"] == steps for leg in "ABC")
        row["counted"] = bool(
            row["leg_A_identical"] and row["leg_B_diverged"] and row["leg_C_diverged"]
            and row["launch_counts_correct"] and row["arrays_ever_moved"] > 0)
        row["arming_agrees_with_measurement"] = (
            row["armed_by_construction"] == row["leg_B_diverged"])
    except BaseException as exc:  # noqa: BLE001 - every case reports, none aborts the run
        import traceback
        row["error"] = f"{type(exc).__name__}: {exc}"[:600]
        row["traceback"] = traceback.format_exc()[-1500:]
        row["counted"] = False
    finally:
        row["seconds"] = round(time.time() - started, 3)
    log(f"{row['case']:32s} seam={row['seam']} "
        f"armed={row.get('armed_by_construction')} "
        f"A_identical={row.get('leg_A_identical')} "
        f"B_diverged={row.get('leg_B_diverged')} "
        f"C_diverged={row.get('leg_C_diverged')} "
        f"launches={row.get('leg_A', {}).get('launches')} "
        f"extra_cells={row.get('arming', {}).get('closure_extra_cells')} "
        f"{row.get('error', '')}")
    return row


# ---------------------------------------------------------------------------
# The two structural legs
# ---------------------------------------------------------------------------

def unfolded_product_leg() -> Dict[str, Any]:
    """Can ``complex_fused_magnetic_pair`` reach the image closure AT ALL?

    THIS IS A MEASUREMENT ABOUT THE FLIP, not a gap in the gate. That module flipped
    ``CARRIES_DEPOSIT_REPAIR`` in the same round, and its own predicate refuses
    ``grid.has_symmetry()`` and every ``is_mirrored`` axis by name — so every configuration
    it admits is unfolded, ``fill_image_rules`` there is empty, and ``repair_cells``
    returns the deposit and nothing more. Its flip rests on the POINT repair, which is
    exact on an unfolded grid because there are no images. The leg proves the refusal by
    asking the shipped predicate rather than quoting its comment.
    """
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415
    from meep_gpu import deposit_repair  # noqa: PLC0415

    record: Dict[str, Any] = {"leg": "unfolded_product", "product":
                              "complex_fused_magnetic_pair"}
    try:
        driver = build_driver(CELL_2D, METALLIC_2D, (("Y", 1),),
                              {"component": "Hy", "frequency": 0.7,
                               "center": (0.1, NEAR_ROW_CENTER, 0.0), "width": 0.4},
                              None)
        verdict = product.complex_fused_magnetic_pair_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        symmetry_refusals = [reason for reason in verdict.reasons
                             if "mirror plane" in reason or "folded" in reason]
        record["folded_case_covered"] = bool(verdict.covered)
        record["symmetry_refusals"] = symmetry_refusals
        record["carries_deposit_repair"] = bool(product.CARRIES_DEPOSIT_REPAIR)
        # And on a grid it CAN admit, the closure is empty — measured, not asserted.
        unfolded = build_driver(CELL_2D, METALLIC_2D, (),
                                {"component": "Hy", "frequency": 0.7,
                                 "center": (0.1, 0.2, 0.0), "width": 0.4}, None)
        rules = {target: list(deposit_repair.fill_image_rules(unfolded.fields, target))
                 for target in ("Bx", "By", "Bz")}
        record["unfolded_fill_image_rules"] = rules
        record["closure_is_empty_where_admitted"] = all(not r for r in rules.values())
        record["passed"] = bool(symmetry_refusals
                                and not verdict.covered
                                and record["closure_is_empty_where_admitted"])
        record["finding"] = (
            "complex_fused_magnetic_pair refuses every folded grid by name, so no "
            "configuration it admits reaches a post-injection fill: on the grids it does "
            "admit fill_image_rules is empty for all three magnetic targets and the "
            "closure and a point-only repair are the SAME SET. Its CARRIES_DEPOSIT_REPAIR "
            "flip is licensed by the point repair alone and CANNOT be shown to need the "
            "closure — not because the gate is weak, but because the product cannot get "
            "there.")
    except BaseException as exc:  # noqa: BLE001
        import traceback
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        record["traceback"] = traceback.format_exc()[-800:]
        record["passed"] = False
    log(f"unfolded_product leg: passed={record.get('passed')} "
        f"covered={record.get('folded_case_covered')}")
    return record


def prior_evidence_leg() -> Dict[str, Any]:
    """Re-measure what ``probe_triton_symmetry_composition`` actually exercised.

    THE THREE CLAIMS ARE READ OFF THAT FILE AND ITS OWN ENGINE OBJECTS, never quoted from
    a review: how many of its cases declare a source, which routing flag its ``run_case``
    passes, and — for each case that does declare one — how many cells ``repair_cells``
    adds beyond the deposit. A gate that asserted the prior evidence was vacuous without
    executing it would be doing what it accuses that probe of.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    record: Dict[str, Any] = {"leg": "prior_evidence"}
    try:
        import probe_triton_symmetry_composition as prior  # noqa: PLC0415

        path = os.path.abspath(prior.__file__)
        record["probe"] = os.path.relpath(path, API_ROOT)
        text = open(path, "r", encoding="utf-8").read()
        body = text[text.index("def run_case("):]
        body = body[:body.index("\ndef ")] if "\ndef " in body else body
        record["run_case_fuse_flag"] = ("fuse=False" if "fuse=False" in body
                                        else "fuse=True" if "fuse=True" in body
                                        else "unstated")
        rows = []
        for name, cell, boundaries, mirror_specs, source_specs, _steps in prior.CASES:
            entry: Dict[str, Any] = {"case": name, "declared_sources": len(source_specs)}
            if source_specs:
                declaration = dict(source_specs[0])
                component = declaration["component"]
                seam = "B" if component[0] == "H" else "D"
                declaration.setdefault("width", 0.4)
                declaration.pop("amplitude", None)
                driver = build_driver(cell, boundaries, mirror_specs, declaration, None)
                entry["seam"] = seam
                entry["arming"] = arming_census(driver, seam, component)
                entry["closure_extra_cells"] = entry["arming"]["closure_extra_cells"]
                entry["armed_for_closure"] = bool(entry["closure_extra_cells"])
            else:
                entry["armed_for_closure"] = False
                entry["why"] = "declares no source, so there is no deposit to repair"
            rows.append(entry)
        record["cases"] = rows
        record["cases_total"] = len(rows)
        record["cases_with_a_source"] = sum(1 for r in rows if r["declared_sources"])
        record["cases_armed_for_closure"] = sum(1 for r in rows
                                                if r.get("armed_for_closure"))
        record["passed"] = (record["run_case_fuse_flag"] == "fuse=False"
                            and record["cases_armed_for_closure"] == 0)
        record["finding"] = (
            f"{record['cases_with_a_source']} of {record['cases_total']} cases declare a "
            f"source, {record['cases_armed_for_closure']} of them place a deposit on a "
            f"row a fill reads from, and run_case plans with "
            f"{record['run_case_fuse_flag']} — so the flag flip was inert in that "
            f"measurement and the closure was never exercised.")
    except BaseException as exc:  # noqa: BLE001
        import traceback
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        record["traceback"] = traceback.format_exc()[-800:]
        record["passed"] = False
    log(f"prior_evidence leg: passed={record.get('passed')} "
        f"armed={record.get('cases_armed_for_closure')}/{record.get('cases_total')} "
        f"fuse={record.get('run_case_fuse_flag')}")
    return record


def environment(xp: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    try:
        import triton  # noqa: PLC0415

        payload["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        payload["triton"] = f"absent: {exc!r}"
    if xp is not None:
        payload["cupy"] = xp.__version__
        properties = xp.cuda.runtime.getDeviceProperties(0)
        payload["device"] = properties["name"].decode()
        # THE COMPUTE CAPABILITY, because a Triton weld is a claim about generated
        # PTX and Triton generates PTX for an ARCHITECTURE. fingerprints.json
        # declares validated_compute_capabilities for exactly that reason and
        # test_triton_weld_contract.py binds every weld's host string to it, so a
        # record that cannot say which cc it ran on cannot be welded at all.
        payload["compute_capability"] = f"{properties['major']}.{properties['minor']}"
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-device", action="store_true",
                        help="run the legs that need neither CUDA nor Triton. The "
                             "fused pair is HOST-EMULATED there and nothing the run "
                             "reports is device evidence.")
    parser.add_argument("--only", default=None, help="comma-separated case names")
    parser.add_argument("--subnormal-policy", default="keep",
                        help="the float32 subnormal policy every executor is driven to "
                             "before the first device compile")
    args = parser.parse_args(argv)

    device = not args.no_device
    xp = None
    payload: Dict[str, Any] = {
        "probe": "triton_folded_deposit_closure",
        "device_status": "device" if device else "no-device (host emulation)",
        "source_sha256": source_hashes(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if device:
        import cupy  # noqa: PLC0415
        from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

        xp = cupy
        # MEEP GOES IN FIRST, AND UNCONDITIONALLY — not only for the policy that needs it.
        #
        # ``subnormal_policy._meep_module`` reads ``sys.modules`` and never imports MEEP
        # itself: the package's own boundary forbids it, and ``import meep`` initializes
        # MPI as a side effect. MEEP's ``set_zero_subnormals`` is therefore the ONLY
        # exposure of this process's host FTZ/DAZ bits the package may use, so a 'flush'
        # request in a process without MEEP is refused as UNATTAINABLE — measured, on the
        # first run of this gate, which released under 'keep' and died here under 'flush'.
        #
        # UNCONDITIONALLY, because the two policy artifacts are meant to differ in the
        # POLICY and in nothing else. Importing MEEP only on the flush leg would make the
        # comparison between the two records a comparison between two process setups.
        try:
            import meep as _meep  # noqa: PLC0415,F401

            payload["meep_imported"] = getattr(_meep, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            payload["meep_imported"] = f"absent: {exc!r}"
        # BEFORE THE FIRST DEVICE COMPILE, and strict: a process that asked to keep and
        # quietly did not is a process whose bytes mean nothing, so this refuses at
        # startup rather than writing an artifact whose policy field is a wish.
        subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cupy,
                                                  strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        backends.guard_kernel_compilation(cupy)
        log(f"subnormal policy installed: "
            f"{payload['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}, "
            f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")
    payload["environment"] = environment(xp)

    wanted = {name.strip() for name in args.only.split(",")} if args.only else None
    rows: List[Dict[str, Any]] = []
    for case in CASES:
        if wanted and case[0] not in wanted:
            continue
        rows.append(run_case(case, device, xp))
        payload["cases"] = rows
        save(payload, args.out)          # partial results as they land (rule 7)

    structural = [prior_evidence_leg(), unfolded_product_leg()]
    payload["structural_legs"] = structural

    scored = [row for row in rows if "error" not in row]
    vacuous = [row["case"] for row in scored if row.get("vacuous_for_closure")]
    counted = [row["case"] for row in scored if row.get("counted")]
    expected_vacuous = [name for name in vacuous if name == ON_PLANE_CONTROL_CASE]
    surprise_vacuous = [name for name in vacuous if name != ON_PLANE_CONTROL_CASE]
    control_ran = any(row["case"] == ON_PLANE_CONTROL_CASE for row in scored)
    control_vacuous = ON_PLANE_CONTROL_CASE in vacuous

    verdict: Dict[str, Any] = {
        "cases": len(rows),
        "scored": len(scored),
        "errors": [row["case"] for row in rows if "error" in row],
        "counted": counted,
        "counted_total": len(counted),
        "vacuous_for_closure": vacuous,
        "vacuous_for_closure_expected": expected_vacuous,
        "vacuous_for_closure_UNEXPECTED": surprise_vacuous,
        "vacuous_for_repair": [row["case"] for row in scored
                               if row.get("vacuous_for_repair")],
        "arming_disagreements": [row["case"] for row in scored
                                 if not row.get("arming_agrees_with_measurement")],
        "structural_legs_passed": all(leg.get("passed") for leg in structural),
    }
    reasons: List[str] = []
    if verdict["errors"]:
        reasons.append(f"cases errored: {verdict['errors']}")
    if surprise_vacuous:
        reasons.append(
            f"cases built to be armed did not diverge without the closure and are "
            f"therefore VACUOUS: {surprise_vacuous}")
    if not control_ran or not control_vacuous:
        reasons.append(
            f"the on-plane control did not behave as the vacuous case it is: ran="
            f"{control_ran} vacuous={control_vacuous}. Without it this artifact does "
            f"not demonstrate that the protocol can SEE a vacuous case.")
    for row in scored:
        if row["case"] == ON_PLANE_CONTROL_CASE:
            # The control's job is to be vacuous for the CLOSURE while still needing the
            # repair; its leg A must still match and its leg C must still diverge.
            if not row.get("leg_A_identical") or not row.get("leg_C_diverged"):
                reasons.append(f"{row['case']}: leg A identical="
                               f"{row.get('leg_A_identical')} leg C diverged="
                               f"{row.get('leg_C_diverged')}")
            continue
        if not row.get("counted"):
            reasons.append(
                f"{row['case']}: A_identical={row.get('leg_A_identical')} "
                f"B_diverged={row.get('leg_B_diverged')} "
                f"C_diverged={row.get('leg_C_diverged')} "
                f"launch_counts_correct={row.get('launch_counts_correct')} "
                f"arrays_moved={row.get('arrays_ever_moved')}")
    if not verdict["structural_legs_passed"]:
        reasons.append("a structural leg failed")
    if device and not any(row.get("leg_A", {}).get("launches") for row in scored):
        reasons.append("asked for the device and counted no fused launch; a pass here "
                       "would be the array path wearing the kernel's name")
    verdict["reasons"] = reasons
    verdict["released"] = not reasons and bool(counted)
    payload["verdict"] = verdict
    payload["what_it_does_not_license"] = (
        "nothing about throughput; nothing about fastpath.plan_fast_path, which still "
        "returns None; nothing about the off-diagonal, nonlinear or cylindrical r = 0 "
        "seams, which deposit_repair.repairable refuses by name; and nothing about "
        "complex_fused_magnetic_pair, whose predicate refuses every folded grid so its "
        "flip cannot be shown to need the closure at all")
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save(payload, args.out)
    log(f"verdict released={verdict['released']} counted={verdict['counted_total']} "
        f"vacuous={verdict['vacuous_for_closure']} -> {args.out}")
    for reason in reasons:
        log(f"  WHY NOT: {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
