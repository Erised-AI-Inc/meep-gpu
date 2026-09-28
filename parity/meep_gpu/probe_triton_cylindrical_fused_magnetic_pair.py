#!/usr/bin/env python3
"""Byte gate for the CYLINDRICAL COMPLEX fused magnetic pair: Dcyl ``step_B``
welded into ``update_H``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.cylindrical_fused_magnetic_pair.cylindrical_fused_magnetic_pair_coverage`
admits, ONE launch of ``cyl_complex_fused_curl_constitutive_B`` leaves the engine
in a state that is BIT-IDENTICAL — over every compared volume, on the uint32 word
view — to BOTH

  * the CuPy ARRAY PATH (``stepping.step_B`` -> ``stepping.zero_metal_B`` ->
    ``stepping.update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the cylindrical complex
    curl (``cylindrical_complex.plan_cylindrical_complex_curl`` on ``step_B``) and
    the certified complex constitutive
    (``cylindrical_complex.plan_cylindrical_complex_constitutive`` on side H), with
    ``zero_metal_B`` on the array path between them because no Triton product owns
    it.

``allclose`` appears nowhere.

WHAT THIS GATE REFUSES TO INFER
===============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback is
  byte-identical to the array path BY CONSTRUCTION. Every launch goes through a
  :class:`~gate_triton_complex.CountingKernel` and every row asserts the exact
  launch count.
* **A no-op agreeing with a no-op is trivially identical.** EVERY ROW CARRIES A
  NON-VACUITY FLOOR: the compared state must have MOVED, and on the zero-init rows
  the signed-zero census must be non-zero as well. A zero-initialised constitutive
  leaves every word ``+0.0`` forever, so a deliberately wrong reference still
  reports IDENTICAL there — half of one earlier gate in this project could not
  fail, and this floor is what that cost bought.
* **A mutation that rewrites an unreached line measures nothing.** ``M_ONE`` and
  ``M_MANY`` are ``tl.constexpr`` arms that compile DIFFERENT BODIES, and ``ZM_Z``
  and ``BCZ`` gate whole blocks. So every needle declares the case it is scored on
  and leg ``needle_reachability`` asserts the enclosing guard of that case admits
  it — the dead-branch class, checked rather than hoped for.
* **A name that matches nothing is not a weaker check, it is an absent one.** The
  compared-volume list is asserted non-empty and asserted to contain the six
  magnetic volumes by name.
* **A record cut under one subnormal policy licenses nothing under the other.**
  BOTH policies are run. ``keep`` is
  :data:`~meep_gpu.triton_kernels.complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`
  and is the one the expansion licence is cut under; ``flush`` is run as a
  SEPARATE, SEPARATELY REPORTED cut, and the two are never averaged.

WHAT IT DOES NOT CLAIM. Nothing about throughput. The fusion removes ONE LAUNCH
and, on a walled run, one host wall pass; it does NOT remove the radial prefix
scan, which stays on the array path because its float32 summation order defines
the answer. Every row records the prefix cost on both compositions.

THE FAMILY IS NOT WIRED. ``launch.plan_step`` assigns at most one plan per slot
and this product spans five driver call sites. Nothing in ``launch.py`` names it,
``fastpath.plan_fast_path`` is unchanged, and no default run can reach it.

Rule 7: one flushed line per case, every row appended as it lands, partial results
written to the artifact directory as they are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import gate_triton_complex as shared  # noqa: E402

#: The keep-cut expansion probe artifacts in this tree, newest first. The device
#: legs consume whichever the environment names; the row records which, so the
#: artifact says what licensed its own bytes.
PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/expansion_probe_2026-08-17/expansion_probe_keep.json",
    "parity/meep_gpu/results/complex_expansion_convention_2026-08-16/results/"
    "probe_keep/probe.json",
)

#: The product's own module path, hashed into the artifact.
FAMILY_MODULE = "meep_gpu/triton_kernels/cylindrical_fused_magnetic_pair.py"

#: Every volume the B->H seam can touch. Scanned against ``Fields`` at run time and
#: asserted complete: a renamed volume that silently dropped out of the comparison
#: would make every row pass for a reason that has nothing to do with the kernel.
COMPARED: Tuple[str, ...] = (
    "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
    "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The curl half's SOURCE volumes. Read, never written, by this seam — so they are
#: seeded, compared for CHANGE (a kernel writing its own source is a defect) and
#: excluded from the "must have moved" floor.
SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The names that MUST be present in the scan. Asserted against what ``Fields``
#: really exposes, so a rename fails here rather than disabling the floor it feeds.
REQUIRED: Tuple[str, ...] = ("Bx", "By", "Bz", "Hx", "Hy", "Hz")

#: ``0x80000000`` — the negative-zero word the zero-init census counts.
NEG_ZERO = 0x80000000

#: The two float32 subnormal policies. ``keep`` is the arms' certification policy
#: and the one the expansion licence is cut under; ``flush`` is run as a SEPARATE
#: cut and reported separately. They are never averaged: a keep-cut licence read by
#: a flush process is a broken comparison, not a platform verdict.
POLICIES: Tuple[str, ...] = ("keep", "flush")

#: ``(name, shape, m, courant, z_metallic, accurate, pml_cells, value_class)``.
#:
#: EVERY AXIS IS A COMPILE-TIME ARM OR A VALUE CLASS, NOT DECORATION:
#:   m class      ``M_ONE`` emits an axis-row increment and no near-axis hold;
#:                ``M_MANY`` emits the hold and NO increment. Different bodies.
#:   m sign       flips the i*m/r coefficient AND the sign of the ``M_ONE``
#:                scalar's real word (+0.0 at m > 0, -0.0 at m < 0).
#:   z            METALLIC emits the ownership mask's z clauses AND the only wall
#:                clear this family can emit; PERIODIC emits neither.
#:   accurate     selects ``zero_rows = 1`` instead of |m| at the same m.
#:   value class  ``uniform`` is the ordinary seeded run; ``subnormal_band`` seeds
#:                in the float32 subnormal range, which is the class the two
#:                policies disagree about; ``signed_zero`` is the +-0 lattice, run
#:                against a THIN absorber because a negative ``kms`` is the only
#:                producer of a stored ``-0.0`` from a quiet state.
CASES: Tuple[Tuple[str, Tuple[int, int, int], int, float, bool, bool, int, str], ...] = (
    ("m1_z_metallic",        (20, 1, 24),  1, 0.37, True,  False, 5, "uniform"),
    ("m_minus1_z_metallic",  (20, 1, 24), -1, 0.37, True,  False, 5, "uniform"),
    ("m1_z_periodic",        (20, 1, 24),  1, 0.5,  False, False, 5, "uniform"),
    ("m3_z_metallic",        (13, 1, 11),  3, 0.37, True,  False, 5, "uniform"),
    ("m_minus2_z_periodic",  (12, 1, 64), -2, 0.4472135954999579, False, False, 5,
     "uniform"),
    ("m3_accurate_z_metallic", (20, 1, 24), 3, 0.2777777777777778, True, True, 5,
     "uniform"),
    ("m5_z_metallic",        (64, 1, 12),  5, 0.3141592653589793, True, False, 5,
     "uniform"),
    ("m1_subnormal_band",    (20, 1, 24),  1, 0.37, True,  False, 5, "subnormal_band"),
    ("m3_subnormal_band",    (13, 1, 11),  3, 0.37, True,  False, 5, "subnormal_band"),
    ("m1_signed_zero_thin",  (20, 1, 24),  1, 0.5,  True,  False, 2, "signed_zero"),
    ("m3_signed_zero_thin",  (20, 1, 24),  3, 0.5,  True,  False, 2, "signed_zero"),
    # THE m = 0 ARM UNDER COMPLEX STORAGE (2026-09-04): no i*m/r block, no
    # increment, Bx[0] = 0 on this seam. Courant 0.5 is the corpus row's own
    # (examples:dipole_in_vacuum_cyl_off_axis.py) and the third case is that row's
    # SHAPE (150, 1, 300) with a 50-cell absorber on r-high and z (the lift record
    # carries no absorber depth; 50 cells is 1.0 length unit at its resolution 50).
    ("m0_z_metallic",        (20, 1, 24),  0, 0.5,  True,  False, 5, "uniform"),
    ("m0_z_periodic",        (20, 1, 24),  0, 0.5,  False, False, 5, "uniform"),
    ("m0_corpus_shape",      (150, 1, 300), 0, 0.5, True,  False, 50, "uniform"),
)

#: The value classes whose rows must census a non-zero number of ``-0.0`` words.
#: A ``signed_zero`` row that censuses zero is VACUOUS and FAILS — it is not a pass.
SIGNED_ZERO_CLASSES: Tuple[str, ...] = ("signed_zero",)

#: Launch budgets. ONE launch isolates a first-launch defect; sixty is where a
#: state defect in ``fu_B`` or ``f_w_H`` has had time to reach the low bits of the
#: interior. Both are run on every case.
LAUNCH_BUDGETS: Tuple[int, ...] = (1, 60)

#: Launches per armed mutation. A defect needing more than this to become
#: byte-visible is reported as a NULL with its launch evidence, never as a pass.
MUTATION_LAUNCHES = 8

SEED = 20260820


def log(message: str) -> None:
    print(message, flush=True)


def case_rng(label: str, *parts: Any) -> np.random.Generator:
    """A per-case seed from a DIGEST, never from ``hash()``.

    ``hash()`` of a tuple of strings is salted with ``PYTHONHASHSEED``, so a
    hash-seeded gate draws a different fixture every process and a failing case
    cannot be replayed. This is reproducible across processes and machines.
    """
    key = "|".join([label] + [str(part) for part in parts]).encode("utf-8")
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(key).digest()[:4], "big"))


# ===========================================================================
# Fixture
# ===========================================================================

def build_engine(xp, shape, m, courant, z_metallic, accurate, pml_cells,
                 complex_storage: bool = True):
    """A real complex cylindrical Grid/Fields/PML at a chosen shape.

    ``resolution = 1`` with a cell size equal to the cell count lands on the shape
    exactly. z is declared explicitly because ``Grid``'s own default is periodic
    and both terminations are in the corpus.

    NOTE THE INVARIANT-AXIS PEC TRAP IS NOT RE-BOUGHT HERE: phi is the invariant
    axis and is left PERIODIC. Declaring it metallic is refused by ``Grid`` by name
    (grid.py:842-877), and a case that tripped that refusal would report a family
    failure that was really a fixture failure.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(m),
                boundaries=({"z": "metallic"} if z_metallic else None),
                accurate_fields_near_cylorigin=bool(accurate),
                courant=float(courant), xp=xp)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    fields = Fields(grid=grid, force_complex_fields=bool(complex_storage))
    fields.enable_pml_storage()
    thickness = {"x": (0, pml_cells), "z": (pml_cells if z_metallic else 0)}
    return grid, fields, PML(grid=grid, thickness=thickness)


def seed_state(xp, fields, shape, value_class: str, label: str) -> Dict[str, int]:
    """Fill every compared volume in the named VALUE CLASS.

    NONZERO EVERYWHERE on the seeded rows, auxiliaries included: a zero ``fu``
    makes ``fu*kms`` exactly zero on the first launch whatever ``kms`` is, so a
    mis-indexed coefficient would only show from the second.

    ``signed_zero`` is the +-0 LATTICE — every word an exact zero, half of them
    negative — which is the only class in which the recurrence's ``+0.0 - (+-0.0)``
    spellings can differ at all, and it is run only against a THIN absorber because
    a NEGATIVE ``kms`` is the only producer of a stored ``-0.0`` from a quiet state.
    """
    rng = case_rng(label, value_class, shape)
    written = 0
    for name in COMPARED + SOURCES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if value_class == "uniform":
            host = (rng.standard_normal(shape).astype(np.float32)
                    + 1j * rng.standard_normal(shape).astype(np.float32))
        elif value_class == "subnormal_band":
            # The float32 subnormal range: [1.4e-45, 1.18e-38). This is the class
            # the two policies disagree about, and the reason both are run.
            scale = np.float32(1.0e-40)
            host = ((rng.standard_normal(shape).astype(np.float32) * scale)
                    + 1j * (rng.standard_normal(shape).astype(np.float32) * scale))
        elif value_class == "signed_zero":
            signs = rng.integers(0, 2, size=shape).astype(np.float32)
            host = ((np.float32(0.0) * (1 - 2 * signs))
                    + 1j * (np.float32(0.0) * (1 - 2 * rng.integers(
                        0, 2, size=shape).astype(np.float32))))
        else:
            raise ValueError(f"unknown value class {value_class!r}")
        array[...] = xp.asarray(np.ascontiguousarray(host.astype(np.complex64)))
        written += int(np.count_nonzero(host))
    return {"nonzero_words_written": written}


def to_host(array) -> np.ndarray:
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


def words(array) -> np.ndarray:
    return np.ascontiguousarray(to_host(array)).view(np.uint32).ravel()


def inventory(fields) -> Dict[str, Any]:
    """Every compared volume, found by name and ASSERTED complete."""
    found = {name: getattr(fields, name) for name in COMPARED
             if getattr(fields, name, None) is not None}
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the compared-volume scan lost {missing}; the comparison inventory is "
            f"not complete and every row below would pass for that reason")
    if not found:
        raise AssertionError("the compared-volume scan is EMPTY")
    return found


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: words(array) for name, array in inventory(fields).items()}


def source_snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: words(getattr(fields, name)) for name in SOURCES
            if getattr(fields, name, None) is not None}


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape"}
        if not np.array_equal(a, b):
            bad = np.flatnonzero(a != b)
            index = int(bad[0])
            return {"array": name, "differing_words": int(bad.size),
                    "first_word": index,
                    "left": hex(int(a[index])), "right": hex(int(b[index]))}
    return None


def total_differing(left: Dict[str, np.ndarray],
                    right: Dict[str, np.ndarray]) -> int:
    return int(sum(np.count_nonzero(left[name] != right[name])
                   for name in sorted(set(left) & set(right))))


def moved_arrays(before: Dict[str, np.ndarray],
                 after: Dict[str, np.ndarray]) -> List[str]:
    return sorted(name for name in before
                  if not np.array_equal(before[name], after[name]))


def negative_zero_census(state: Dict[str, np.ndarray]) -> int:
    return int(sum(np.count_nonzero(value == NEG_ZERO) for value in state.values()))


def nan_census(fields) -> Dict[str, int]:
    """Non-finite stored words. A NaN's payload is IEEE-unspecified, so a uint32
    compare over one is not a measurement; a clean row with a nonzero count FAILS
    rather than being compared through."""
    nans = infinities = 0
    for array in inventory(fields).values():
        host = to_host(array)
        nans += int(np.count_nonzero(np.isnan(host)))
        infinities += int(np.count_nonzero(np.isinf(host)))
    return {"nan_words": nans, "inf_words": infinities}


def minimum_bound_kms(pml, half_integer: bool) -> float:
    """The smallest ``kms`` word over the sub-lattice the CURL binds.

    THE PRODUCER OF A STORED ``-0.0`` IN A SIGNED-ZERO ROW, and the only one: the
    recurrence is ``fu *= kms; fu -= curl; ...``, so from an all-``+0.0`` state a
    negative-zero word can only appear where some ``kms`` is NEGATIVE. Whether one
    is negative is a property of the ABSORBER THICKNESS, not of the harness, and it
    is recorded per row so a vacuous row can be told from a passing one.
    """
    suffix = "_h" if half_integer else ""
    return min(float(np.min(to_host(getattr(pml, f"kms_{axis}{suffix}"))))
               for axis in "xyz")


# ===========================================================================
# The three routes
# ===========================================================================

def coefficient_dict(pml, half_integer: bool) -> Dict[str, Any]:
    suffix = "_h" if half_integer else ""
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def constitutive_dict(pml) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def array_seam(fields, pml) -> None:
    """The array path's own B->H seam, in driver order (driver.py:3281-3289)."""
    from meep_gpu import stepping  # noqa: PLC0415

    stepping.step_B(fields, pml)
    stepping.zero_metal_B(fields)
    stepping.update_H(fields, pml)


def separate_plans(fields, pml, probe):
    """The two ALREADY CERTIFIED Triton products this fusion replaces."""
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    curl = cyl.plan_cylindrical_complex_curl(fields, pml, "step_B", probe=probe)
    constitutive = cyl.plan_cylindrical_complex_constitutive(
        fields, pml, "H", probe=probe)
    return curl, constitutive


def separate_seam(fields, pml, curl, constitutive) -> None:
    """The separate composition: certified curl, HOST wall clear, certified H.

    ``zero_metal_B`` stays on the array path because no Triton product owns it —
    which is exactly the in-seam host pass the fusion removes on a walled run.
    """
    from meep_gpu import stepping  # noqa: PLC0415

    curl.run()
    stepping.zero_metal_B(fields)
    constitutive.run()


# ===========================================================================
# One row
# ===========================================================================

def run_case(xp, case, launches: int, policy: str, probe,
             mutant: Any = None, expansion_override: Optional[int] = None,
             ) -> Dict[str, Any]:
    """Three routes in lockstep from ONE seed; stop at the FIRST byte divergence."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_fused_magnetic_pair as product)

    (label, shape, m, courant, z_metallic, accurate, pml_cells, value_class) = case
    built = [build_engine(xp, shape, m, courant, z_metallic, accurate, pml_cells)
             for _ in range(3)]
    for _grid, fields, _pml in built:
        seed_state(xp, fields, shape, value_class, label)
    (_g0, reference, reference_pml) = built[0]
    (_g1, separate, separate_pml) = built[1]
    (_g2, fused, fused_pml) = built[2]

    row: Dict[str, Any] = {
        "case": label, "shape": list(shape), "m": m, "courant": courant,
        "z": "metallic" if z_metallic else "periodic", "accurate": accurate,
        "pml_cells": pml_cells, "value_class": value_class,
        "launches_budget": launches, "policy": policy,
        "mutated": mutant is not None,
        "expansion_override": expansion_override,
        "min_bound_kms_curl_lattice": minimum_bound_kms(fused_pml, True),
    }

    counter = shared.CountingKernel(
        mutant if mutant is not None
        else product.cyl_complex_fused_curl_constitutive_B_kernel())
    plan = product.plan_cylindrical_fused_magnetic_pair(
        fused, fused_pml, sources=(), num_warps=1, kernel=counter, probe=probe)
    if plan is None:
        verdict = product.cylindrical_fused_magnetic_pair_coverage(
            fused, fused_pml, (), probe=probe)
        row.update({"passed": False, "refused": True,
                    "reasons": list(verdict.reasons)})
        return row
    if expansion_override is not None:
        # THE ARM, OVERRIDDEN. Not a source rewrite: the constexpr IS the licence,
        # so flipping it here measures whether the licence is load-bearing.
        plan.expansion = int(expansion_override)
    row["plan"] = repr(plan)
    row["plan_replaces"] = list(plan.replaces)
    row["licensed_expansion"] = plan.expansion

    curl, constitutive = separate_plans(separate, separate_pml, probe)
    row["separate_products"] = {"curl": type(curl).__name__ if curl else None,
                                "constitutive": (type(constitutive).__name__
                                                 if constitutive else None)}
    if curl is None or constitutive is None:
        row.update({"passed": False,
                    "reason": "a separately certified half was refused"})
        return row

    opening = snapshot(fused)
    opening_sources = source_snapshot(fused)
    row["negative_zero_census_at_seed"] = negative_zero_census(opening)
    # EVER-MOVED ACROSS THE SEQUENCE, not final-versus-seed.
    #
    # MEASURED ON THE GPU HOST 2026-08-20, and it is a harness defect of the class this
    # project has already paid for: on the +-0 lattice `fu_Bz` has a PERIOD-2 orbit
    # (its recurrence is `n2 = (fu*km_x - curl) * si_x` with every operand a signed
    # zero, so the sign map is affine and flips), which returns it to exactly its
    # seeded pattern at every EVEN launch budget. A final-versus-seed floor then
    # reported the 60-launch row VACUOUS over a volume that had moved at all sixty
    # launches, while the byte comparison was identical throughout. A floor that
    # fires for a reason other than the one it exists for is not a stricter check.
    ever_moved: set = set()
    previous = opening
    per_launch: List[Dict[str, Any]] = []
    started = time.time()
    divergence: Optional[Dict[str, Any]] = None
    control: Optional[Dict[str, Any]] = None
    for launch in range(1, launches + 1):
        array_seam(reference, reference_pml)
        separate_seam(separate, separate_pml, curl, constitutive)
        plan.run()
        xp.cuda.runtime.deviceSynchronize()
        after_reference = snapshot(reference)
        after_separate = snapshot(separate)
        after_fused = snapshot(fused)
        divergence = (first_divergence(after_fused, after_reference)
                      or first_divergence(after_fused, after_separate))
        control = first_divergence(after_separate, after_reference)
        per_launch.append({
            "launch": launch,
            "fused_vs_array": total_differing(after_fused, after_reference),
            "fused_vs_separate_certified": total_differing(after_fused,
                                                           after_separate),
            "oracle_control_vs_array": total_differing(after_separate,
                                                        after_reference),
            "negative_zero_census": negative_zero_census(after_reference),
            "kernel_launches": counter.launches,
        })
        ever_moved.update(moved_arrays(previous, after_fused))
        previous = after_fused
        if divergence is not None or control is not None:
            row["diverged_at_launch"] = launch
            break
    row["elapsed_seconds"] = time.time() - started

    final = snapshot(fused)
    ever_moved = sorted(ever_moved)
    sources_changed = moved_arrays(opening_sources, source_snapshot(fused))
    peak_census = max(entry["negative_zero_census"] for entry in per_launch)
    row.update({
        "per_launch": per_launch,
        "launches_observed": counter.launches,
        "first_divergence": divergence,
        "control_divergence": control,
        "arrays_compared": sorted(final),
        "arrays_moved": ever_moved,
        "arrays_never_moved": sorted(set(final) - set(ever_moved)),
        "source_volumes_changed": sources_changed,
        "negative_zero_census_peak": peak_census,
        "nan_census": nan_census(fused),
        "prefix_scans_fused": counter.launches,
        "prefix_scans_separate": counter.launches,
    })

    failures: List[str] = []
    if divergence is not None:
        failures.append(f"byte divergence: {divergence}")
    if control is not None:
        failures.append(
            f"an ORACLE control diverged from the array path, so this row could "
            f"not have measured the fused launch: {control}")
    if counter.launches != launches:
        failures.append(f"the fused kernel launched {counter.launches} times, "
                        f"not {launches}: this row may be a silent fallback")
    # THE NON-VACUITY FLOOR. A row whose state never moved compares a no-op with a
    # no-op and passes for a reason that has nothing to do with the kernel.
    if row["arrays_never_moved"]:
        failures.append(f"these volumes NEVER MOVED, so this row is vacuous over "
                        f"them: {row['arrays_never_moved']}")
    if sources_changed:
        failures.append(f"the kernel WROTE its own source volumes: {sources_changed}")
    if row["nan_census"]["nan_words"] or row["nan_census"]["inf_words"]:
        failures.append(f"non-finite stored words: {row['nan_census']}")
    # THE SIGNED-ZERO FLOOR. A +-0 row whose census is zero cannot discriminate the
    # class it exists for, whatever its differing-word count says.
    if value_class in SIGNED_ZERO_CLASSES and peak_census == 0:
        failures.append(
            f"the signed-zero census peaked at 0 with min bound kms "
            f"{row['min_bound_kms_curl_lattice']:+.4f}: this row is VACUOUS for "
            f"the class it exists to measure and must not be read as a pass")
    row["failures"] = failures
    row["passed"] = not failures
    return row


# ===========================================================================
# Mutations
# ===========================================================================

_TEMPORARY: List[str] = []


def shipped_block() -> str:
    """The shipped kernel definition, read out of the file rather than re-typed.

    RETURNED AT ITS OWN INDENTATION — the kernel is defined inside
    ``if triton is not None:``, so every body line carries eight spaces. Every
    needle in :data:`MUTATIONS` is written against THIS text, so a needle and the
    source it edits cannot disagree about indentation; :func:`dedented` is what
    turns it into something importable.

    Parsing the source is what keeps this from being a MIRRORED EVALUATOR: a gate
    that re-implemented the kernel's arithmetic would mirror a planted defect
    instead of executing it. Measured on this project 2026-08-20: three planted
    assembly defects each left 86 of 87 mirrored tests passing.
    """
    text = (API_ROOT / FAMILY_MODULE).read_text(encoding="utf-8")
    start = text.index("    def cyl_complex_fused_curl_constitutive_B(")
    end = text.index("else:  # pragma: no cover - laptop path")
    return text[start:end].rstrip() + "\n"


def dedented(block: str) -> str:
    """One indentation level off, so the block is importable at module scope."""
    return "\n".join(line[4:] if line.startswith("    ") else line
                     for line in block.splitlines()) + "\n"


def compile_mutated(source: str, entry: str):
    """Compile one mutated kernel from a REAL FILE.

    Triton reads source through ``inspect``, so an ``exec``'d body raises at first
    launch. The constexpr codes and the three certified multiply helpers are
    re-declared in the header because a ``@triton.jit`` body may not read a plain
    module global.
    """
    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import (\n"
        "    _mul_coefficient_left, _mul_field_left)\n"
        "from meep_gpu.triton_kernels.cylindrical_complex import (\n"
        "    _mul_general_coefficient_left)\n"
        "PERIODIC = tl.constexpr(0)\n"
        "METALLIC = tl.constexpr(1)\n"
        "M_ZERO = tl.constexpr(0)\n"
        "M_ONE = tl.constexpr(1)\n"
        "M_MANY = tl.constexpr(2)\n\n"
        # shipped_block() starts at `def`, so the DECORATOR is supplied here. Without
        # it the mutated module defines a plain Python function, `CountingKernel`
        # raises "'function' object is not subscriptable", and the whole mutation
        # battery is unarmed. Measured on the GPU host 2026-08-20.
        "@triton.jit\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_cylfused.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cylfused_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, entry)


def _renamed(source: str, suffix: str) -> Tuple[str, str]:
    name = f"cyl_complex_fused_curl_constitutive_B__{suffix}"
    return (source.replace("def cyl_complex_fused_curl_constitutive_B(",
                           f"def {name}(", 1), name)


#: Every armed source defect, as DATA: name -> (case, old, new, expectation).
#:
#: ``expectation`` is ``"catch"`` or ``"null"``. A NULL is never merely permitted —
#: it carries a measured reason in :data:`NULL_REASONS` and the leg reports it as
#: NULL CONFIRMED rather than as a pass.
#:
#: THE CASE EACH IS SCORED ON IS DECLARED, because ``M_CLASS`` and ``BCZ`` are
#: ``tl.constexpr`` guards: a needle armed on a case whose arm the shipped kernel
#: does not compile would rewrite an unreached line and report UNCAUGHT while
#: measuring nothing.
MUTATIONS: Dict[str, Tuple[str, str, str, str]] = {
    # ---- THE SEAM ITSELF ---------------------------------------------------
    "seam_takes_pre_recurrence_curl":
        ("m1_z_metallic", "        src_re = v0_re\n        src_im = v0_im\n",
         "        src_re = curl0_re\n        src_im = curl0_im\n", "catch"),
    "seam_takes_the_wrong_component":
        ("m1_z_metallic", "        src_re = v1_re\n        src_im = v1_im\n",
         "        src_re = v0_re\n        src_im = v0_im\n", "catch"),
    # ---- THE WALL CLEAR (ZM_Z; ZM_X/ZM_Y are dead on every Dcyl grid) -------
    "zero_metal_dropped":
        ("m1_z_metallic",
         "        if ZM_Z:\n            v2_re = tl.where(at_z, 0.0, v2_re)\n"
         "            v2_im = tl.where(at_z, 0.0, v2_im)\n",
         "        if ZM_Z:\n            pass\n", "catch"),
    "zero_metal_clears_only_the_real_plane":
        ("m1_z_metallic", "            v2_im = tl.where(at_z, 0.0, v2_im)\n", "",
         "catch"),
    "zero_metal_uses_the_r_row":
        ("m1_z_metallic", "            v2_re = tl.where(at_z, 0.0, v2_re)",
         "            v2_re = tl.where(at_r, 0.0, v2_re)", "catch"),
    # ---- THE CYLINDRICAL CURL ----------------------------------------------
    "imr_partners_swapped":
        ("m1_z_metallic",
         "            m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, c_re, c_im,",
         "            m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, a_re, a_im,",
         "catch"),
    "imr_row_indexed_by_z":
        ("m1_z_metallic", "            q0_re = tl.load(c0 + 2 * i, mask=live, other=0.0)",
         "            q0_re = tl.load(c0 + 2 * k, mask=live, other=0.0)", "catch"),
    "imr_sign_flipped":
        ("m1_z_metallic", "            curl0_re = curl0_re - m0_re",
         "            curl0_re = curl0_re + m0_re", "catch"),
    "prefix_row_stride_dropped":
        ("m1_z_metallic",
         "            pu_re = tl.load(pfx + 2 * (idx + nyz), mask=live, other=0.0)",
         "            pu_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)", "catch"),
    "prefix_replaced_by_the_four_operand_grouping":
        ("m1_z_metallic",
         "            curl2_re, curl2_im = _mul_coefficient_left(dtdx, pu_re - pd_re,\n"
         "                                                       pu_im - pd_im, EXPANSION)",
         "            curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re,\n"
         "                                                       t2_im, EXPANSION)",
         "catch"),
    # ---- THE |m| = 1 AXIS INCREMENT (M_ONE body only) ----------------------
    "axis_increment_not_negated":
        ("m1_z_metallic",
         "                curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)",
         "                curl0_re = tl.where(at_r, inc_re, curl0_re)", "catch"),
    "axis_increment_reads_the_axis_row":
        ("m1_z_metallic", "                off = nyz + j * nz + k",
         "                off = j * nz + k", "catch"),
    # ---- THE |m| >= 2 NEAR-AXIS HOLD (M_MANY body only) --------------------
    "axis_hold_skips_the_auxiliaries":
        ("m3_z_metallic", "            n0_re = tl.where(near, 0.0, n0_re)\n", "",
         "catch"),
    "axis_hold_covers_only_row_zero":
        ("m3_z_metallic", "            near = i < ZERO_ROWS", "            near = i == 0",
         "catch"),
    # ---- THE CERTIFIED COMPLEX BASE ----------------------------------------
    "curl_parens_flattened":
        ("m1_z_metallic",
         "        t1_re = ((a_z_re - a_re) + (c_re - c_r_re))",
         "        t1_re = (a_z_re - a_re + c_re - c_r_re)", "catch"),
    "recurrence_axis_pair_swapped":
        ("m1_z_metallic",
         "        x_re, x_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)",
         "        x_re, x_im = _mul_field_left(p0_re, p0_im, km_z, EXPANSION)",
         "catch"),
    "fu_store_dropped":
        ("m1_z_metallic", "        tl.store(u0 + 2 * idx, n0_re, mask=live)\n", "",
         "catch"),
    "flux_store_dropped":
        ("m1_z_metallic", "        tl.store(f0 + 2 * idx, v0_re, mask=live)\n", "",
         "catch"),
    "fw_store_dropped":
        ("m1_z_metallic", "        tl.store(w0 + 2 * idx, src_re, mask=live)\n", "",
         "catch"),
    "constitutive_coefficient_index_moved":
        ("m1_z_metallic", "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
         "        kp_0 = tl.load(kp0 + j, mask=live, other=0.0)", "catch"),
    "constitutive_prev_read_after_write":
        ("m1_z_metallic",
         "        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.\n"
         "        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)\n"
         "        src_re = v0_re\n        src_im = v0_im\n"
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n",
         "        src_re = v0_re\n        src_im = v0_im\n"
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n"
         "        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)\n"
         "        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)\n",
         "catch"),
    # ---- MEASURED PREDICTED NULLS -----------------------------------------
    # Both were MEASURED on the Metal twin on 2026-08-20 and are re-measured here
    # rather than inherited. They are REAL choices that a byte gate on a B-only
    # seam cannot hold, and they are recorded as such.
    "curl_parens_flattened_on_the_invariant_term":
        ("m1_z_metallic",
         "        t0_re = ((c_p_re - c_re) + (b_re - b_z_re))",
         "        t0_re = (c_p_re - c_re + b_re - b_z_re)", "null"),
    # ARMED AS A CATCH ON THE +-0 LATTICE, and that placement is the whole finding.
    # The increment REPLACES row 0, and after the ownership mask that row is exactly
    # +0.0 — so `+0.0 + x` and `x` differ ONLY at x = -0.0. A RANDOM seed reaches
    # that class with probability zero and a ZERO-INIT state cannot manufacture it
    # on this seam either (Bx's split-field dsig axis is phi, whose kms is 1.0, so
    # the `fu * kms` minuend from an all-+0.0 state is +0.0). MEASURED BOTH WAYS ON
    # 2026-08-20: 0 differing words under random seeding and under zero-init with a
    # thin absorber, and CAUGHT on the +-0 LATTICE, where `fu` itself carries -0.0
    # words and `-0.0 * 1.0` keeps the sign. So the choice is byte-visible and the
    # row is a CATCH — on the one value class that can see it.
    "axis_increment_accumulates":
        ("m1_signed_zero_thin",
         "                curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)\n"
         "                curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)",
         "                curl0_re = tl.where(at_r, curl0_re - inc_re, curl0_re)\n"
         "                curl0_im = tl.where(at_r, curl0_im - inc_im, curl0_im)",
         "catch"),
    # ---- THE m = 0 AXIS RULE ON THIS SEAM (M_ZERO body; 2026-09-04) -----------
    "m0_bx_axis_zero_dropped":
        ("m0_z_metallic",
         "                v0_re = tl.where(at_r, 0.0, v0_re)\n"
         "                v0_im = tl.where(at_r, 0.0, v0_im)\n",
         "                v0_re = v0_re\n                v0_im = v0_im\n", "catch"),

}

NULL_REASONS: Dict[str, str] = {
    "curl_parens_flattened_on_the_invariant_term":
        "t0 leads with the PHI SELF-DIFFERENCE (c_p - c). phi has n = 1, so the "
        "wrap returns the SAME element and the difference is an exact +0.0; both "
        "associations then reduce to `b - b_z` and no word can differ. The "
        "identical edit on t1, which carries no invariant-axis operand, IS armed "
        "as a catch in the row above. Measured on the Metal twin 2026-08-20: "
        "0 words against 528 for t1.",
}


#: How each ``tl.constexpr`` guard this kernel carries is evaluated for one case.
#:
#: FAIL-CLOSED BY CONSTRUCTION: a condition that is not in this table makes
#: :func:`guard_chain_holds` return ``None`` and the leg FAIL, rather than letting
#: an unmodelled guard pass silently. That is the whole point — the dead-branch
#: class is a mutation reported UNCAUGHT because the case never entered the block,
#: and a guard evaluator that shrugged at an unknown condition would re-buy it.
GUARD_RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    # BACKWARD is bound to 0 by every builder (the product is the B seam), so the
    # `if BACKWARD:` arms of the verbatim-copied curl are DEAD on every case. A
    # needle inside one measures nothing and this table is what says so.
    "if BACKWARD:": lambda case: False,
    "if BCZ == METALLIC:": lambda case: case["z_metallic"],
    "if M_CLASS == M_ONE:": lambda case: abs(case["m"]) == 1,
    # The m = 0 arm (2026-09-04): the i*m/r block is compiled OUT there, the
    # m = 0 axis pair compiled IN, and the near-axis hold is its own arm now.
    "if M_CLASS != M_ZERO:": lambda case: case["m"] != 0,
    "if M_CLASS == M_ZERO:": lambda case: case["m"] == 0,
    "if M_CLASS == M_MANY:": lambda case: abs(case["m"]) >= 2,
    # On a Dcyl grid `Grid` cannot report r or phi walled, so ZM_X and ZM_Y are
    # dead on EVERY admitted configuration — which is why the shipped product
    # refuses a grid whose wall table says otherwise, and why no needle here is
    # armed inside either.
    "if ZM_X:": lambda case: False,
    "if ZM_Y:": lambda case: False,
    "if ZM_Z:": lambda case: case["z_metallic"],
}


def _first_line(needle: str) -> str:
    for line in needle.splitlines():
        if line.strip():
            return line
    raise AssertionError(f"empty needle: {needle!r}")


def guard_chain(block: str, needle: str) -> List[str]:
    """Every ``if`` / ``else`` header ENCLOSING the needle, outermost first.

    Parsed out of the shipped text by indentation rather than declared in a table
    beside it: a hand-kept table of "which guard is this under" is exactly the
    thing that drifts, and when it drifts a needle silently stops being scored on
    the case that can reach it.

    An ``else:`` is resolved to the ``if`` it belongs to and returned as
    ``"else of <condition>"``, so the evaluator never has to guess a polarity.
    """
    head = _first_line(needle)
    position = block.index(needle)
    prefix = block[:position].splitlines()
    want = len(head) - len(head.lstrip())
    chain: List[str] = []
    index = len(prefix) - 1
    while index >= 0:
        line = prefix[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index -= 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent < want:
            if stripped.startswith("else:"):
                # Walk back to the `if` at the SAME indent this `else` belongs to.
                partner = index - 1
                while partner >= 0:
                    candidate = prefix[partner]
                    text = candidate.strip()
                    if text and not text.startswith("#"):
                        depth = len(candidate) - len(candidate.lstrip())
                        if depth == indent and text.startswith(("if ", "elif ")):
                            chain.append(f"else of {text}")
                            break
                        if depth < indent:
                            chain.append("else of <unresolved>")
                            break
                    partner -= 1
                else:
                    chain.append("else of <unresolved>")
                want = indent
            elif stripped.startswith(("if ", "elif ")):
                chain.append(stripped)
                want = indent
            else:
                want = indent
        index -= 1
    chain.reverse()
    return [entry for entry in chain if not entry.startswith("def ")]


def guard_chain_holds(chain: Sequence[str],
                      case: Dict[str, Any]) -> Optional[bool]:
    """Does this case COMPILE every guard the needle sits under? ``None`` = unknown.

    ``None`` is a refusal, never an assumed True: an unmodelled guard is exactly
    the situation in which a mutation can report UNCAUGHT while measuring nothing.
    """
    for entry in chain:
        negate = entry.startswith("else of ")
        condition = entry[len("else of "):] if negate else entry
        rule = GUARD_RULES.get(condition)
        if rule is None:
            return None
        value = bool(rule(case))
        if negate:
            value = not value
        if not value:
            return False
    return True


def leg_needle_reachability() -> Dict[str, Any]:
    """Every needle must be PRESENT, and its ENCLOSING GUARDS must hold on its case.

    THE DEAD-BRANCH CHECK, and the reason this leg exists. A mutation can rewrite
    REAL lines the scored case never reaches, because they sit under a
    ``tl.constexpr`` guard that case does not enter; it then reports UNCAUGHT while
    measuring nothing. This kernel carries FOUR such guards — ``BACKWARD`` (bound
    to 0, so its arms are dead on every case), ``BCZ``, ``M_CLASS`` and the three
    ``ZM_*`` flags — so the guard chain of every needle is PARSED OUT OF THE
    SHIPPED TEXT by indentation and evaluated against the case the needle is scored
    on. An unmodelled condition FAILS the leg rather than passing.

    A rewrite that DELETES an ``if`` counts as its own guard, so the deletion
    needles carry the ``if`` line in the text they match and are attributed to the
    chain ABOVE it.
    """
    block = shipped_block()
    cases = {case[0]: case for case in CASES}
    rows: List[Dict[str, Any]] = []
    for name, (case_name, old, _new, expectation) in MUTATIONS.items():
        case = cases[case_name]
        context = {"m": case[2], "z_metallic": case[4], "accurate": case[5]}
        present = old in block
        chain = guard_chain(block, old) if present else []
        holds = guard_chain_holds(chain, context) if present else None
        rows.append({
            "mutation": name, "case": case_name, "expectation": expectation,
            "needle_present": present,
            "enclosing_guards": chain,
            "case_compiles_every_guard": holds,
            "case_m": context["m"],
            "case_z": "metallic" if context["z_metallic"] else "periodic"})
    # THE MARKER-NAME FLOOR. A set of names that matches nothing disables the
    # check it feeds SILENTLY, so every condition this evaluator models is
    # asserted to occur in the shipped source. A rule for a guard the kernel no
    # longer carries is a rule that will never fire.
    markers = {condition: (condition in block) for condition in GUARD_RULES}
    ok = all(entry["needle_present"] and entry["case_compiles_every_guard"] is True
             for entry in rows)
    return {"passed": bool(ok and all(markers.values())),
            "rows": rows,
            "guard_conditions_present_in_the_shipped_source": markers,
            "guard_rules_modelled": sorted(GUARD_RULES),
            "note": "BACKWARD is bound to 0, so the `if BACKWARD:` arms of the "
                    "verbatim-copied curl are DEAD on every case and no needle is "
                    "armed inside one; ZM_X and ZM_Y are dead on every Dcyl grid "
                    "for the same kind of reason. Both are modelled as False here "
                    "so a needle placed inside either FAILS this leg rather than "
                    "reporting UNCAUGHT downstream."}


def leg_transcription() -> Dict[str, Any]:
    """The two halves must be the certified kernels' own lines, VERBATIM.

    PARSED OUT OF THE SHIPPED TEXT, never re-implemented — a test that
    re-implemented the kernel's assembly would MIRROR a planted defect instead of
    executing it. Measured on this project 2026-08-20: three planted assembly
    defects each left 86 of 87 mirrored tests passing.

    The certified sources are read from THEIR files and every arithmetic statement
    of the curl half and of the constitutive half is required to appear in the
    fused body character for character.
    """
    fused = dedented(shipped_block())
    curl = (API_ROOT / "meep_gpu/triton_kernels/cylindrical_complex.py").read_text(
        encoding="utf-8")
    constitutive = (
        API_ROOT / "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py"
    ).read_text(encoding="utf-8")

    def statements(text: str, start: str, end: str) -> List[str]:
        """LOGICAL statements, not physical lines.

        Continuation lines are joined by bracket depth and runs of whitespace are
        collapsed, so a REFLOW — which changes no token and no float32 bit — does
        not fire this check, while any changed token, operand order or paren does.
        Comparing physical lines instead would make the leg fire on line wrapping
        and would have to be relaxed by hand, which is how a transcription check
        stops being one.
        """
        block = text[text.index(start):text.index(end)]
        out: List[str] = []
        pending = ""
        depth = 0
        for line in block.splitlines():
            stripped = line.strip()
            if not depth and (not stripped or stripped.startswith("#")
                              or stripped.startswith('"')):
                continue
            pending = (pending + " " + stripped).strip() if pending else stripped
            depth += stripped.count("(") - stripped.count(")")
            if depth <= 0:
                out.append(" ".join(pending.split()))
                pending, depth = "", 0
        if pending:
            out.append(" ".join(pending.split()))
        return out

    curl_lines = statements(
        curl, "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    # --- stores: u then f (kernels.py:193-198 order), both planes ---")
    constitutive_lines = statements(
        constitutive, "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
        "else:  # pragma: no cover - laptop path")
    # The END anchor is EXCLUSIVE, so the fused body is read to a sentinel PAST
    # its last statement and that statement is added back by name. An end anchor
    # that swallowed the last line would make one certified statement look missing
    # on every run, which is a check that fails for a reason of its own.
    _LAST = "    tl.store(h2 + 2 * idx + 1, a_im, mask=live)"
    assert _LAST in fused, "the fused body no longer ends at the H2 store"
    fused_statements = set(statements(
        fused, "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)", _LAST))
    fused_statements.add(_LAST.strip())

    # The curl half is carried verbatim EXCEPT for the register renames the
    # cylindrical curl and this weld share (none) — so every line must be present.
    curl_missing = [line for line in curl_lines if line not in fused_statements]
    constitutive_missing = [line for line in constitutive_lines
                            if line not in fused_statements]
    # The three seam lines the weld ADDS, and the only ones.
    seam_lines = [f"src_re = v{n}_re" for n in range(3)]
    seam_present = [line for line in seam_lines if line in fused_statements]
    # And the three the weld REMOVES: the constitutive reload of B.
    reload_absent = not any(f"src_re = tl.load(g{n}" in fused for n in range(3))
    return {
        "passed": bool(not curl_missing and not constitutive_missing
                       and len(seam_present) == 3 and reload_absent),
        "certified_curl_statements": len(curl_lines),
        "curl_statements_missing_from_the_fused_body": curl_missing,
        "certified_constitutive_statements": len(constitutive_lines),
        "constitutive_statements_missing_from_the_fused_body": constitutive_missing,
        "seam_lines_present": seam_present,
        "constitutive_reload_of_B_absent": reload_absent,
        "note": "parsed out of the shipped files; nothing here re-implements the "
                "kernel's arithmetic, which is what stops a planted defect from "
                "being mirrored instead of executed.",
    }


def leg_refusal(xp, probe) -> Dict[str, Any]:
    """The source seam, the m split and the geometry split, each measured by name.

    THE SOURCE CLAUSE COSTS THIS FAMILY ZERO CORPUS ROWS — every one of the sixteen
    cylindrical rows declares electric sources only — which is exactly why it has to
    be tested here rather than left to the corpus.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_fused_magnetic_pair as product)
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    _grid, fields, pml = build_engine(xp, (20, 1, 24), 1, 0.37, True, False, 5)
    seed_state(xp, fields, (20, 1, 24), "uniform", "refusal")

    undeclared = product.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, None, probe=probe)
    magnetic = VolumeSource(grid=fields.grid, component="Hy",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    magnetic_cov = product.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,), probe=probe)
    electric = VolumeSource(grid=fields.grid, component="Ez",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    electric_cov = product.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (electric,), probe=probe)
    magnetic_plan = product.plan_cylindrical_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), probe=probe)

    # m = 0 UNDER COMPLEX STORAGE IS ADMITTED (2026-09-04, the M_ZERO arm); the
    # REAL-storage m = 0 run is still cylindrical_triton's and must be refused BY
    # NAME on the storage clause — the split moved from m to force_complex_fields
    # and both directions are measured here.
    _g0, m0_fields, m0_pml = build_engine(xp, (20, 1, 24), 0, 0.37, True, False, 5)
    m0_cov = product.cylindrical_fused_magnetic_pair_coverage(
        m0_fields, m0_pml, (), probe=probe)
    _g1, real_fields, real_pml = build_engine(xp, (20, 1, 24), 0, 0.37, True, False,
                                              5, complex_storage=False)
    m0_real = product.cylindrical_fused_magnetic_pair_coverage(
        real_fields, real_pml, (), probe=probe)

    return {
        "passed": bool(not undeclared.covered
                       and any("was not declared" in r for r in undeclared.reasons)
                       and not magnetic_cov.covered
                       and any("is magnetic" in r and "driver.py:3283" in r
                               for r in magnetic_cov.reasons)
                       and magnetic_plan is None
                       and electric_cov.covered
                       and m0_cov.covered
                       and not m0_real.covered
                       and any("force_complex_fields" in r
                               for r in m0_real.reasons)),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": [r for r in undeclared.reasons
                             if "was not declared" in r],
        "magnetic_source_refused": not magnetic_cov.covered,
        "magnetic_plan_is_none": magnetic_plan is None,
        "magnetic_named": [r for r in magnetic_cov.reasons if "is magnetic" in r],
        "electric_source_admitted": electric_cov.covered,
        "electric_reasons": list(electric_cov.reasons),
        "m_zero_complex_admitted": m0_cov.covered,
        "m_zero_complex_reasons": list(m0_cov.reasons)[:3],
        "m_zero_real_storage_refused": not m0_real.covered,
        "m_zero_real_storage_named": [r for r in m0_real.reasons
                                      if "force_complex_fields" in r][:2],
        "note": "the source clause costs ZERO corpus rows on this family, which is "
                "why it is measured here rather than left to the corpus.",
    }


# ===========================================================================
# Main
# ===========================================================================

def probe_record(explicit: Optional[str]) -> Tuple[Optional[Dict[str, Any]],
                                                   List[str], Optional[str]]:
    """The keep-cut expansion probe artifact, or a NAMED refusal.

    A PLACEHOLDER IS REFUSED BY NAME: `expansion_license` takes the verdict dict,
    not a path, and handing it a string produces "the expansion licence is str, not
    the verdict dict" — a refusal about the harness reported as one about the
    corpus. So the artifact is loaded here and the loaded record is what travels.
    """
    candidates = ([explicit] if explicit else
                  [str(API_ROOT / relative) for relative in PROBE_CANDIDATES])
    reasons: List[str] = []
    for path in candidates:
        if not path or not Path(path).is_file():
            reasons.append(f"probe artifact absent: {path}")
            continue
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - unreadable == missing
            reasons.append(f"probe artifact unreadable ({path}): {exc}")
            continue
        if not isinstance(record, dict):
            reasons.append(f"probe artifact is not a verdict dict: {path}")
            continue
        return record, reasons, path
    return None, reasons, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True,
                        help="results DIRECTORY; partial rows land here as they do")
    parser.add_argument("--probe-artifact", default=None)
    parser.add_argument("--launches", type=int, default=None,
                        help="override the largest launch budget")
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    results: Dict[str, Any] = {
        "gate": "triton_cylindrical_fused_magnetic_pair",
        "family_module": FAMILY_MODULE,
        "cases": [list(case) for case in CASES],
        "policies": list(POLICIES),
        "launch_budgets": list(LAUNCH_BUDGETS),
        "rows": [], "skipped": {}, "failures": [],
    }

    def flush() -> None:
        from gate_provenance import stamp as _stamp  # noqa: PLC0415
        _stamp(results)
        (out / "gate.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

    # --- laptop legs, which run everywhere ---------------------------------
    results["rows"].append({"leg": "needle_reachability", "device": False,
                            **leg_needle_reachability()})
    log(f"[needle_reachability] passed={results['rows'][-1]['passed']}")
    flush()
    results["rows"].append({"leg": "transcription", "device": False,
                            **leg_transcription()})
    log(f"[transcription] passed={results['rows'][-1]['passed']}")
    flush()

    try:
        import cupy as cp  # noqa: PLC0415
        import triton  # noqa: F401, PLC0415
        device_available, why = True, None
    except Exception as exc:  # noqa: BLE001
        cp, device_available, why = None, False, f"{type(exc).__name__}: {exc}"

    if not device_available:
        results["skipped"]["device"] = (
            f"{why} — the device legs run on the measurement machine; NOTHING was "
            f"measured here")
        log(f"[device] SKIPPED cleanly: {results['skipped']['device']}")
        results["verdict"] = "SKIPPED_DEVICE"
        results["elapsed_seconds"] = time.time() - started
        flush()
        return 0

    budgets = (LAUNCH_BUDGETS if args.launches is None
               else (1, int(args.launches)))
    record, reasons, probe_path = probe_record(args.probe_artifact)
    if record is None:
        results["failures"].extend(reasons)
        results["verdict"] = "REFUSED"
        results["elapsed_seconds"] = time.time() - started
        flush()
        log("REFUSED: " + "; ".join(reasons))
        return 1
    results["probe_artifact"] = probe_path
    results["probe_sha256"] = hashlib.sha256(
        Path(probe_path).read_bytes()).hexdigest()

    for policy in POLICIES:
        # THE POLICY IS INSTALLED BEFORE ANY DEVICE COMPILE. One authority per
        # process phase; the two cuts are reported separately and never averaged.
        stamp = shared.install_ftz_strip(policy)
        policy_reasons = shared.probe_record_policy_reasons(record)
        results.setdefault("policy_stamps", {})[policy] = {
            "install": stamp, "probe_policy_reasons": list(policy_reasons)}
        if policy != "keep" and not policy_reasons:
            # A keep-cut record read by a flush process is a BROKEN COMPARISON, not
            # a platform verdict. It is recorded and the flush cut is reported as
            # UNLICENSED rather than silently averaged with the keep cut.
            results["policy_stamps"][policy]["licence"] = (
                "the expansion record is keep-cut; the flush cut below is reported "
                "as a SEPARATE measurement and licenses nothing")

        from meep_gpu.triton_kernels import complex_fields as cx  # noqa: PLC0415
        expansion = cx._resolve_expansion(record)
        results["policy_stamps"][policy]["expansion"] = expansion
        if expansion is None:
            results["failures"].append(
                f"[{policy}] the expansion licence refused this record")
            flush()
            continue

        for case in CASES:
            for launches in budgets:
                row = run_case(cp, case, launches, policy, record)
                row["leg"] = "product"
                results["rows"].append(row)
                log(f"[product/{policy}] {case[0]} x{launches}: "
                    f"passed={row.get('passed')} "
                    f"launches={row.get('launches_observed')} "
                    f"moved={len(row.get('arrays_moved', []))} "
                    f"census={row.get('negative_zero_census_peak')} "
                    f"({row.get('elapsed_seconds', 0):.1f} s)")
                flush()

        # THE ARM OVERRIDE. The constexpr IS the licence; flipping it measures
        # whether the licence is load-bearing on this family. On the cylindrical
        # arm it is EXPECTED TO BE INERT (every general complex product here takes
        # a coefficient whose real word is a zero, and there is no Bloch rotation),
        # so the row is scored as a NULL with that reason and reported, never
        # silently dropped.
        other = 0 if int(expansion) == 1 else 1
        row = run_case(cp, CASES[0], MUTATION_LAUNCHES, policy, record,
                       expansion_override=other)
        row["leg"] = "expansion_override"
        row["expectation"] = "null"
        row["reason"] = (
            "the two arms differ only where a complex product's coefficient has a "
            "NONZERO real word; every general complex product here takes one whose "
            "real word is a zero (the i*m/r rows are exactly +0.0, the |m| = 1 "
            "scalar is +-0.0) and this family compiles no Bloch rotation. The "
            "binding is INERT here and is still made from a measured artifact, "
            "because a widening would make it live.")
        row["null_confirmed"] = row.get("first_divergence") is None
        row["passed"] = bool(row["null_confirmed"]
                             and row.get("launches_observed") == MUTATION_LAUNCHES)
        results["rows"].append(row)
        log(f"[expansion_override/{policy}] arm {expansion}->{other}: "
            f"null_confirmed={row['null_confirmed']}")
        flush()

    # --- refusal + mutations, under the certification policy only ----------
    shared.install_ftz_strip("keep")
    results["rows"].append({"leg": "refusal", "device": True, "policy": "keep",
                            **leg_refusal(cp, record)})
    log(f"[refusal] passed={results['rows'][-1]['passed']}")
    flush()

    cases = {case[0]: case for case in CASES}
    block = shipped_block()
    for name, (case_name, old, new, expectation) in MUTATIONS.items():
        if old not in block:
            results["rows"].append({
                "leg": "mutation", "label": name, "passed": False,
                "reason": "the needle is ABSENT from the shipped source; this "
                          "mutation armed NOTHING"})
            flush()
            continue
        mutated, entry = _renamed(dedented(block.replace(old, new, 1)), name)
        kernel = compile_mutated(mutated, entry)
        row = run_case(cp, cases[case_name], MUTATION_LAUNCHES, "keep", record,
                       mutant=kernel)
        diverged = row.get("first_divergence") is not None
        launched = row.get("launches_observed") == MUTATION_LAUNCHES
        result_row = {
            "leg": "mutation", "label": name, "case": case_name,
            "expectation": expectation,
            "caught": diverged, "null_confirmed": not diverged,
            "launches_observed": row.get("launches_observed"),
            "first_divergence": row.get("first_divergence"),
            "arrays_moved": len(row.get("arrays_moved", [])),
            "negative_zero_census_peak": row.get("negative_zero_census_peak"),
            # A CATCH breaks the launch loop at its first divergence, so it owes
            # AT LEAST ONE launch, not the whole budget; a NULL runs the budget out
            # and owes all of it. Demanding the full budget of a catch scored every
            # caught mutation as a failure — measured on the GPU host 2026-08-20.
            "passed": bool(diverged and row.get("launches_observed", 0) >= 1
                           if expectation == "catch"
                           else (not diverged and launched)),
        }
        if expectation == "null":
            result_row["reason"] = NULL_REASONS[name]
        results["rows"].append(result_row)
        log(f"[mutation] {name} on {case_name}: expectation={expectation} "
            f"caught={diverged} launches={row.get('launches_observed')} "
            f"passed={result_row['passed']}")
        flush()

    # THE PREDICTED NULLS, RE-RUN ON THE SIGNED-ZERO CASE. A null measured only on
    # a random seed is a statement about the seed; these are re-measured where the
    # +-0 class is live and the census proves the row is not vacuous.
    for name in ("axis_increment_accumulates",):
        case_name, old, new, _expectation = MUTATIONS[name]
        mutated, entry = _renamed(dedented(block.replace(old, new, 1)),
                                  name + "_uniform")
        kernel = compile_mutated(mutated, entry)
        # THE SAME NEEDLE ON THE UNIFORM CLASS, where it is expected to be NULL. The
        # pair of rows is the measurement: the choice is real and is visible ONLY on
        # the +-0 lattice, so a gate that swept random seeds alone would have
        # reported it uncaught and called the kernel certified.
        row = run_case(cp, cases["m1_z_metallic"], MUTATION_LAUNCHES, "keep",
                       record, mutant=kernel)
        diverged = row.get("first_divergence") is not None
        census = row.get("negative_zero_census_peak", 0)
        results["rows"].append({
            "leg": "value_class_sensitivity", "label": name,
            "case": "m1_z_metallic", "value_class": "uniform",
            "caught": diverged, "null_confirmed": not diverged,
            "negative_zero_census_peak": census,
            "min_bound_kms_curl_lattice": row.get("min_bound_kms_curl_lattice"),
            "launches_observed": row.get("launches_observed"),
            "expectation": "null",
            "passed": bool(not diverged
                           and row.get("launches_observed") == MUTATION_LAUNCHES),
            "reason":
                "the same needle that is a CATCH on the +-0 lattice is NULL here, "
                "and the pair is the measurement: the increment's REPLACE-versus-"
                "ACCUMULATE choice differs only at a -0.0 operand, which a random "
                "seed reaches with probability zero. A gate that swept uniform "
                "seeds alone would have reported this defect uncaught and called "
                "the kernel certified."})
        log(f"[value_class_sensitivity] {name} on uniform: caught={diverged} "
            f"census={census} passed={results['rows'][-1]['passed']}")
        flush()

    # --- HOST mutations: what no source rewrite can reach -------------------
    host_rows = run_host_mutations(cp, cases["m1_z_metallic"], record)
    results["rows"].extend(host_rows)
    for entry in host_rows:
        log(f"[host_mutation] {entry['label']}: caught={entry['caught']} "
            f"passed={entry['passed']}")
    flush()

    rows = results["rows"]
    results["verdict"] = ("PASS" if all(row.get("passed") for row in rows)
                          else "FAIL")
    results["failed_rows"] = [row.get("label", row.get("case", row.get("leg")))
                              for row in rows if not row.get("passed")]
    results["elapsed_seconds"] = time.time() - started
    # THE POLICY THE WELD IS CUT UNDER, in the key the ledger tools read. This probe
    # cuts BOTH policies in one run and reports them side by side under
    # ``policy_stamps``; ``seed_triton_welds.policy_line`` reads a top-level
    # ``subnormal_policy`` block (``resolved`` / ``policy`` / ``cache_dir``) and
    # refused this artifact by name on 2026-09-12 for want of it. The keep cut is the
    # release (the flush cut is reported as a separate measurement and licenses
    # nothing), so the keep install stamp is the block.
    keep_stamp = (results.get("policy_stamps") or {}).get("keep") or {}
    if isinstance(keep_stamp.get("install"), dict):
        results["subnormal_policy"] = dict(keep_stamp["install"])
    import triton_device_identity  # noqa: PLC0415

    results["environment"] = triton_device_identity.record({
        "numpy": np.__version__,
        "cupy": getattr(cp, "__version__", None),
        "subnormal_policy_stamps": results.get("policy_stamps"),
    })
    results["source_sha256"] = {
        FAMILY_MODULE: hashlib.sha256(
            (API_ROOT / FAMILY_MODULE).read_bytes()).hexdigest(),
        "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    flush()
    log(f"VERDICT {results['verdict']} in {results['elapsed_seconds']:.1f}s; "
        f"artifact {out / 'gate.json'}")
    for path in _TEMPORARY:
        try:
            os.unlink(path)
        except OSError:
            pass
    return 0 if results["verdict"] == "PASS" else 1


def run_host_mutations(xp, case, probe) -> List[Dict[str, Any]]:
    """Corrupt the TABLES the host binds — defects no source rewrite can reach.

    The kernel takes twelve coefficient pointers and two i*m/r rows and never asks
    which lattice or which target they came from, so each of these is a silent
    half-cell or wrong-sign error rather than a crash.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_fused_magnetic_pair as product)

    (label, shape, m, courant, z_metallic, accurate, pml_cells, value_class) = case
    rows: List[Dict[str, Any]] = []

    def one(name: str, corrupt: Callable[[Any, Any], None]) -> Dict[str, Any]:
        built = [build_engine(xp, shape, m, courant, z_metallic, accurate,
                              pml_cells) for _ in range(2)]
        for _g, fields, _p in built:
            seed_state(xp, fields, shape, value_class, label)
        _g0, reference, reference_pml = built[0]
        _g1, actual, actual_pml = built[1]
        counter = shared.CountingKernel(
            product.cyl_complex_fused_curl_constitutive_B_kernel())
        plan = product.plan_cylindrical_fused_magnetic_pair(
            actual, actual_pml, sources=(), num_warps=1, kernel=counter,
            probe=probe)
        assert plan is not None
        corrupt(plan, actual_pml)
        for _ in range(MUTATION_LAUNCHES):
            array_seam(reference, reference_pml)
            plan.run()
        xp.cuda.runtime.deviceSynchronize()
        divergence = first_divergence(snapshot(actual), snapshot(reference))
        caught = divergence is not None
        return {"leg": "host_mutation", "label": name, "case": label,
                "caught": caught, "null_confirmed": not caught,
                "first_divergence": divergence,
                "launches_observed": counter.launches,
                "passed": bool(caught and counter.launches == MUTATION_LAUNCHES)}

    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    def integer_lattice(plan, pml) -> None:
        """The B curl reads HALF-INTEGER positions; hand it the integer ones."""
        plan._curl_coefficients = tuple(
            CupyPointer(_flat(getattr(pml, f"{stem}_{axis}")))
            for axis in "xyz" for stem in ("kms", "sinv"))

    def half_integer_h(plan, pml) -> None:
        """``update_H`` takes kps_a/kms_a; hand it the half-integer lattice."""
        plan._h_coefficients = tuple(
            CupyPointer(_flat(getattr(pml, f"{stem}_{axis}_h")))
            for axis in "xyz" for stem in ("kps", "kms"))

    def swapped_imr(plan, pml) -> None:
        """Bind target 2's i*m/r row to target 0 and vice versa. The two differ by
        the target's OWN radial Yee shift AND by the sign of the term."""
        plan._imr_rows = tuple(reversed(plan._imr_rows))

    for name, corrupt in (("curl_takes_the_integer_lattice", integer_lattice),
                          ("h_half_takes_the_half_integer_lattice", half_integer_h),
                          ("imr_rows_swapped_between_targets", swapped_imr)):
        rows.append(one(name, corrupt))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
