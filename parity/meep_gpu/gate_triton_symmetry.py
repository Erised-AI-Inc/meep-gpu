"""Bit-identity gate for the folded-grid (mirror symmetry) Triton kernels.

WHY THIS IS A SEPARATE FILE AND NOT A BRANCH OF THE SHARED PROBE. The shared
probe (``probe_fused_kernel_bit_identity.py``) and the shared track adapter
(``track_triton_pml.py``) are being edited RIGHT NOW by the concurrent fused-pair
workflow, so this round does not touch either. Everything reusable is IMPORTED
from the shared probe and nothing is re-implemented: the byte comparator, the ULP
key, the curl grouping, the split-field recurrence, the synthetic coefficient
tables, the field seeding, the artifact writer. What is added here is only what
does not exist there — the MIRROR branches of the ghost rule and the fold's
top-plane ownership mask — plus the two legs the fold needs (the ghost-fill
kernel, and a real-engine substitution). Merging these branches into
``reference_shift``/``reference_mask`` and ``PML_BOUNDARY_SETS`` is a one-hour
edit once those files are free; it is listed with the rest of the deferred
integration in ``meep_gpu/triton_kernels/symmetry.py``'s module docstring.

THE GATE MUST STAY AT SUB-STEP GRANULARITY, and this is not a preference. At
whole-step granularity NEITHER fold mask is observable: the driver's fill passes
overwrite exactly the planes the masks protect, so a kernel carrying neither is
bytewise-identical after a complete step (measured, both terminations, 24 steps,
with a positive control that differs). A whole-step check would certify a
mask-less symmetry kernel as correct. Every leg below is per-sub-step.

Legs, in order:

* ``synthetic``   — the sweep. Shapes x boundary sets x dtdx x plane phase x
  reflect-row parity x sub-step, guarded and unguarded, against a reference
  transcribed from ``stepping.py`` here rather than imported from it.
* ``mutations``   — six defects injected into the shipped kernel's SOURCE, each
  of which must be caught (two are nulls that must NOT be).
* ``fill``        — the ghost-fill kernel against the array path's own
  ``fill_symmetry_bc_*`` / ``fill_folded_far_ghosts_*``, on real folded grids.
* ``engine``      — ``stepping.step_B``/``step_D`` on a real folded CuPy grid with
  a real PML layer, substituted sub-step by sub-step, over several steps.
* ``bench``       — the fill kernel against the array path's fills, timed.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=5 python -u gate_triton_symmetry.py \\
        --out results/triton_symmetry_<date>/gate.json

Every case prints one flushed line as it lands and the JSON artifact is rewritten
incrementally (the progress-reporting rule).

THE ARTIFACT STATES ITS OWN OUTCOME. The last thing written is ``verdict``, a
dict carrying ``pass`` plus the clause behind every leg above — the shape
``gate_provenance.read_verdict`` already reads, so "did this gate release?" is
answerable mechanically rather than by knowing this file. ``main`` exits 1 on a
false verdict, and a leg that raises is RECORDED and the run continues: a gate
that dies writes no verdict, and a verdict-less artifact reads as unreadable,
which is not the same claim as refused. See "The recorded verdict" below.
"""

from __future__ import annotations

import argparse
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cupy as cp  # noqa: E402

# Everything reusable comes from the shared probe. Nothing below re-implements
# any of it — a second byte comparator or a second recurrence would make this
# gate's verdict about this file rather than about the contract.
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# The subnormal-policy machinery — the ftz strip, its resolution, its counters
# and its stamp — is the shared gate's, imported for the same reason the probe
# is. A second policy authority in this process is exactly how a record gets
# cut under one policy and labelled with another.
import gate_triton_complex as gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, mirror_parity  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import symmetry  # noqa: E402

SEED = probe.SEED
bit_compare = probe.bit_compare
combine = probe.combine
_face = probe._face

install_ftz_strip = gate.install_ftz_strip
ftz_strip_license_reasons = gate.ftz_strip_license_reasons
policy_stamp = gate.policy_stamp

#: The boundary strings this file adds to the shared probe's two.
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
MIRROR_METALLIC = "mirror_metallic"
MIRROR_PERIODIC = "mirror_periodic"

#: String -> the kernel constexpr the plan binds.
CODE_OF = {
    PERIODIC: symmetry.CODE_PERIODIC,
    METALLIC: symmetry.CODE_METALLIC,
    MIRROR_METALLIC: symmetry.CODE_MIRROR_METALLIC,
    MIRROR_PERIODIC: symmetry.CODE_MIRROR_PERIODIC,
}
IS_FOLD = (MIRROR_METALLIC, MIRROR_PERIODIC)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The reference — the shared probe's, with the two MIRROR branches added
# ---------------------------------------------------------------------------

def reference_shift(xp: Any, field: Any, axis: int, boundary: str, backward: bool,
                    component: str, phase: int,
                    reflect_row: Optional[int]) -> Any:
    """``field[i-1]`` / ``field[i+1]`` under one axis's ghost rule, folds included.

    ``probe.reference_shift`` with the two branches ``stepping._shift_down``
    (:1787) and ``_shift_up`` (:1723) take on a folded axis:

    * near face (``_shift_down``, the D sub-step's direction)::

          shifted[0] = mirror_parity(component, axis, phase) * field[2]

      on EITHER termination — the mirror images stored cell
      ``MIRROR_SOURCE_INDEX = 2``, MEEP's ``io = -2`` halved origin.

    * far face (``_shift_up``, the B sub-step's direction)::

          shifted[-1] = mirror_parity(...) * field[reflect_row]   # folded PERIODIC
          shifted[-1] = 0                                          # folded METALLIC

    The unfolded branches delegate to the shared probe, so a regression in the
    periodic/metallic ghost rule is caught by the same code that catches it for
    the shipped kernel.
    """
    if boundary in (PERIODIC, METALLIC):
        return probe.reference_shift(xp, field, axis, boundary, backward)
    shifted = xp.roll(field, 1 if backward else -1, axis=axis)
    parity = mirror_parity(component, axis, phase)
    if backward:
        shifted[_face(axis, 0)] = parity * field[_face(axis,
                                                       symmetry.MIRROR_SOURCE_INDEX)]
        return shifted
    if boundary == MIRROR_PERIODIC:
        if reflect_row is None:
            raise ValueError("a folded PERIODIC axis needs a reflect row")
        shifted[_face(axis, -1)] = parity * field[_face(axis, reflect_row)]
        return shifted
    shifted[_face(axis, -1)] = 0
    return shifted


def reference_curl(xp: Any, sources: Dict[str, Any], term, dtdx: Any,
                   backward: bool, boundaries: Sequence[str], phases: Sequence[int],
                   reflect_rows: Sequence[Optional[int]], grouping: str) -> Any:
    """dtdx * the discrete curl, with the shared probe's two groupings.

    ``array_order`` is ``stepping._curl_from_operands`` (:1601), the contract::

        dtdx * ((shifted_first - first) + (second - shifted_second))
    """
    _target, g1, a1, g2, a2, _dsig, _dsigu, _iyee = term
    first, second = sources[g1], sources[g2]
    shifted_first = reference_shift(xp, first, a1, boundaries[a1], backward,
                                    g1, phases[a1], reflect_rows[a1])
    shifted_second = reference_shift(xp, second, a2, boundaries[a2], backward,
                                     g2, phases[a2], reflect_rows[a2])
    if grouping == "array_order":
        return dtdx * ((shifted_first - first) + (second - shifted_second))
    if grouping == "kernel_order":
        return dtdx * (((shifted_first - first) + second) - shifted_second)
    raise ValueError(f"unknown grouping {grouping!r}")


def reference_mask(curl: Any, iyee: Sequence[int],
                   boundaries: Sequence[str]) -> None:
    """``stepping._mask_non_owned_cells`` (:1865) in full, in place.

    TWO clauses, not one. The shared probe carries the first only, because the
    shipped kernel only ever sees metallic axes:

    1. shift 0 on a non-periodic axis -> cell 0 (``is_mirrored or is_metallic or
       is_axis``); the fold and the wall land on the same plane for different
       reasons;
    2. shift 1 on a FOLDED PERIODIC axis -> the LAST cell. That slot is past
       MEEP's owned window (``owns``, vec.cpp:445-462) and the fill pass, not the
       curl, is what writes it.
    """
    for axis in range(3):
        if iyee[axis] == 0 and boundaries[axis] != PERIODIC:
            curl[_face(axis, 0)] = 0
        if iyee[axis] == 1 and boundaries[axis] == MIRROR_PERIODIC:
            curl[_face(axis, -1)] = 0


def reference_pml_step(xp: Any, arrays: Dict[str, Any], coefficients: Dict[str, Any],
                       dtdx: Any, sub_step: str, boundaries: Sequence[str],
                       phases: Sequence[int],
                       reflect_rows: Sequence[Optional[int]],
                       grouping: str = "array_order") -> None:
    """One real-field PML curl sub-step, in place, on a possibly folded grid.

    The recurrence itself is ``probe.reference_recurrence`` — the shared
    transcription of ``stepping._apply_pml_update`` — untouched: a fold changes
    the ghost rule and the ownership mask and nothing else.
    """
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    for term in terms:
        target, _g1, _a1, _g2, _a2, dsig, dsigu, iyee = term
        curl = reference_curl(xp, arrays, term, dtdx, backward, boundaries,
                              phases, reflect_rows, grouping)
        reference_mask(curl, iyee, boundaries)
        probe.reference_recurrence(
            arrays[target], arrays["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


# ---------------------------------------------------------------------------
# The sweep product
# ---------------------------------------------------------------------------
#
# Why each axis is in the product rather than trimmed:
#
# * dtdx — 0.35 is MANDATORY. At 0.5 the scaling is exact in binary and the
#   FMA/associativity discrepancies vanish, so a gate testing only 0.5 certifies
#   broken kernels.
# * shapes — an ODD extent on the folded axis (the reflect_row = n-3 case, the
#   only place the full count's parity still shows), a non-cube (so an i<->k index
#   swap cannot pass), and a 2-D sheet with an invariant nz = 1.
# * boundaries — both folds alone, both folds ASYMMETRIC against a different rule
#   on the other axes (a per-axis bug cannot hide behind a uniform declaration),
#   and a TWO-PLANE set. The unfolded sets are carried too: with no fold this
#   kernel must reduce to the shipped one exactly.
# * phase — F2 predicts the CURL is phase-independent (the parity only ever
#   multiplies a ghost the mask discards). That is a prediction the gate checks
#   rather than a fact it assumes, so -1 is in the product.
# * reflect row — n-2 (even full count) and n-3 (odd). The kernel never reads it;
#   the REFERENCE does, and that is the measurement behind "the ghost is dead".
# * sub_step — step_B and step_D shift in opposite directions, read different
#   coefficient sub-lattices, and mask DIFFERENT planes (cell 0 on the shift-0
#   axes, the last cell on the shift-1 ones). Neither implies the other.

SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (13, 11, 9),     # odd non-cube
    (12, 14, 10),    # even non-cube
    (9, 7, 1),       # 2-D sheet, invariant z
    (17, 5, 3),      # thin and asymmetric
)
BOUNDARY_SETS: Tuple[Tuple[str, str, str], ...] = (
    (PERIODIC, MIRROR_METALLIC, PERIODIC),
    (PERIODIC, MIRROR_PERIODIC, PERIODIC),
    (MIRROR_PERIODIC, PERIODIC, METALLIC),     # asymmetric, folded X
    (METALLIC, MIRROR_METALLIC, PERIODIC),     # asymmetric, folded Y
    (PERIODIC, PERIODIC, MIRROR_PERIODIC),     # folded Z
    (MIRROR_PERIODIC, MIRROR_METALLIC, PERIODIC),   # two planes, mixed
    (MIRROR_PERIODIC, MIRROR_PERIODIC, MIRROR_METALLIC),  # three planes
    (PERIODIC, PERIODIC, PERIODIC),            # the unfolded control...
    (METALLIC, PERIODIC, METALLIC),            # ...and its walled sibling
)
DTDX: Tuple[float, ...] = probe.PML_DTDX          # (0.5, 0.35) — 0.35 is mandatory
PHASES: Tuple[int, ...] = (1, -1)
ROW_PARITIES: Tuple[str, ...] = ("even_count", "odd_count")
MULTI_STEP_COUNT = 6


def reflect_rows_for(shape: Sequence[int], boundaries: Sequence[str],
                     row_parity: str) -> Tuple[Optional[int], ...]:
    """The reference's far-face image row per axis.

    ``stepping._far_reflect_rows`` is ``n_full - stored + 2``, which is
    ``stored - 2`` at an even full count and ``stored - 3`` at an odd one. The
    synthetic leg has no ``Grid`` to ask, so both are swept — and the engine leg
    below asks a real one, which is what pins the formula itself.
    """
    offset = 2 if row_parity == "even_count" else 3
    return tuple(int(shape[axis]) - offset if boundaries[axis] == MIRROR_PERIODIC
                 else None for axis in range(3))


def viable(shape: Sequence[int], boundaries: Sequence[str]) -> Optional[str]:
    """Why this (shape, boundaries) pair cannot be run, or None."""
    for axis in range(3):
        if boundaries[axis] not in IS_FOLD:
            continue
        if int(shape[axis]) <= symmetry.MIRROR_SOURCE_INDEX + 1:
            return (f"axis {axis} has {shape[axis]} cells; a folded axis needs "
                    f"more than {symmetry.MIRROR_SOURCE_INDEX} to reflect from "
                    f"and one more to hold a distinct far ghost")
    return None


# ---------------------------------------------------------------------------
# The synthetic leg
# ---------------------------------------------------------------------------

def one_case(shape, boundaries, dtdx, phase, row_parity, sub_step, guard,
             kernel=None, host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One launch of one sub-step against both reference groupings, bytewise."""
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "dtdx": repr(dtdx), "phase": phase, "row_parity": row_parity,
        "sub_step": sub_step, "guard": guard,
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "host_mutation": host_mutation,
    }
    skip = viable(shape, boundaries)
    if skip:
        case["skipped"] = skip
        return case

    rng = np.random.default_rng(SEED)
    arrays = probe.make_pml_fields(tuple(shape), rng)
    half_integer = sub_step == "step_B"
    coefficients = probe.synthetic_coefficients(cp, tuple(shape), half_integer)
    flat = probe.flatten_coefficients(coefficients)

    reference = {name: array.copy() for name, array in arrays.items()}
    phases = (phase, phase, phase)
    rows = reflect_rows_for(shape, boundaries, row_parity)
    ref_array = {name: array.copy() for name, array in reference.items()}
    reference_pml_step(cp, ref_array, coefficients, dtdx, sub_step, boundaries,
                       phases, rows, "array_order")
    ref_kernel = {name: array.copy() for name, array in reference.items()}
    reference_pml_step(cp, ref_kernel, coefficients, dtdx, sub_step, boundaries,
                       phases, rows, "kernel_order")

    codes = [CODE_OF[b] for b in boundaries]
    if host_mutation == "fold_as_periodic":
        codes = [symmetry.CODE_PERIODIC if b in IS_FOLD else CODE_OF[b]
                 for b in boundaries]
    elif host_mutation == "swap_fold_termination":
        codes = [symmetry.CODE_MIRROR_PERIODIC if b == MIRROR_METALLIC
                 else symmetry.CODE_MIRROR_METALLIC if b == MIRROR_PERIODIC
                 else CODE_OF[b] for b in boundaries]
    plan = symmetry.plan_folded_from_arrays(sub_step, arrays, flat, codes,
                                            float(dtdx), kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets, aux = _names(sub_step)
    case["vs_array_order"] = combine(
        {name: bit_compare(arrays[name], ref_array[name])
         for name in targets + aux})
    case["vs_kernel_order"] = combine(
        {name: bit_compare(arrays[name], ref_kernel[name])
         for name in targets + aux})
    return case


def _names(sub_step: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if sub_step == "step_B":
        return ("Bx", "By", "Bz"), ("fu_Bx", "fu_By", "fu_Bz")
    return ("Dx", "Dy", "Dz"), ("fu_Dx", "fu_Dy", "fu_Dz")


def multi_step_case(shape, boundaries, dtdx, phase, row_parity,
                    kernel=None) -> Dict[str, Any]:
    """B and D alternating for several sub-steps from one state.

    The auxiliary is state: a kernel that gets ``field`` right and ``fu`` wrong is
    correct for exactly one launch and wrong forever after. Consecutive sub-steps
    are what turn that into a visible divergence.
    """
    case: Dict[str, Any] = {"shape": list(shape), "boundaries": list(boundaries),
                            "dtdx": repr(dtdx), "phase": phase,
                            "row_parity": row_parity, "steps": MULTI_STEP_COUNT}
    skip = viable(shape, boundaries)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 7)
    arrays = probe.make_pml_fields(tuple(shape), rng)
    reference = {name: array.copy() for name, array in arrays.items()}
    tables = {sub: probe.synthetic_coefficients(cp, tuple(shape), sub == "step_B")
              for sub in ("step_B", "step_D")}
    flat = {sub: probe.flatten_coefficients(tables[sub]) for sub in tables}
    codes = [CODE_OF[b] for b in boundaries]
    rows = reflect_rows_for(shape, boundaries, row_parity)
    phases = (phase, phase, phase)

    for _ in range(MULTI_STEP_COUNT):
        for sub_step in ("step_B", "step_D"):
            symmetry.plan_folded_from_arrays(
                sub_step, arrays, flat[sub_step], codes, float(dtdx),
                kernel=kernel).run()
            reference_pml_step(cp, reference, tables[sub_step], dtdx, sub_step,
                               boundaries, phases, rows, "array_order")
    cp.cuda.runtime.deviceSynchronize()
    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
             "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
    case["vs_array_order"] = combine(
        {name: bit_compare(arrays[name], reference[name]) for name in names})
    return case


#: The multi-step sub-sweep, as data rather than as a comprehension inside
#: ``main``. TWO CONSUMERS NEED THE SAME LIST: the leg that runs it, and the
#: verdict clause that has to know how many cases a COMPLETE leg produces. With
#: the product written out at the call site the clause could only compare a
#: count against itself, and a leg that died at case 30 would record 30/30 and
#: read as a pass.
MULTI_STEP_COMBOS: Tuple[Tuple[Any, ...], ...] = tuple(
    (shape, boundaries, dtdx, phase, row_parity)
    for shape in SHAPES[:2]
    for boundaries in BOUNDARY_SETS[:6]
    for dtdx in DTDX
    for phase in (1,)
    for row_parity in ROW_PARITIES)


def run_multi_step(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """The multi-step sub-sweep: several sub-steps from one state, per combination."""
    cases: List[Dict[str, Any]] = []
    for index, (shape, boundaries, dtdx, phase, row_parity) in enumerate(
            MULTI_STEP_COMBOS, start=1):
        started = time.time()
        case = multi_step_case(shape, boundaries, dtdx, phase, row_parity)
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        if case.get("skipped"):
            log(f"[multi] {index}/{len(MULTI_STEP_COMBOS)} SKIPPED "
                f"{case['skipped'][:60]}")
        else:
            log(f"[multi] {index}/{len(MULTI_STEP_COMBOS)} "
                f"{'x'.join(str(n) for n in shape)} "
                f"{'/'.join(b[:4] for b in boundaries)} dtdx={dtdx!r} "
                f"phase={phase:+d} {row_parity} x{MULTI_STEP_COUNT}: "
                f"identical={case['vs_array_order']['bit_identical']} "
                f"({case['seconds']} s)")
        results["multi_step"] = {
            "ran": sum(1 for c in cases if not c.get("skipped")),
            "identical": sum(int(c.get("vs_array_order", {}).get("bit_identical",
                                                                 False))
                             for c in cases),
            "expected": expected_multi_step_cases(),
            "cases": cases}
        save(results, out_path)
    return results["multi_step"]


def run_synthetic(results: Dict[str, Any], out_path: str,
                  kernel=None, host_mutation: Optional[str] = None,
                  label: str = "gate", guards=(False, True),
                  shapes: Sequence[Tuple[int, int, int]] = SHAPES) -> Dict[str, Any]:
    """The full product, both arrays per component, guarded and unguarded.

    ``shapes`` narrows only the MUTATION legs, and only on the axis a mutation
    cannot hide behind: every mutation here is a per-axis or per-plane defect, so
    an odd non-cube and an even non-cube exercise all of them. The gate leg itself
    always runs the full product.
    """
    cases: List[Dict[str, Any]] = []
    combos = [(shape, boundaries, dtdx, phase, row_parity, sub_step)
              for shape in shapes
              for boundaries in BOUNDARY_SETS
              for dtdx in DTDX
              for phase in PHASES
              for row_parity in ROW_PARITIES
              for sub_step in ("step_B", "step_D")]
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for shape, boundaries, dtdx, phase, row_parity, sub_step in combos:
            index += 1
            started = time.time()
            case = one_case(shape, boundaries, dtdx, phase, row_parity, sub_step,
                            guard, kernel=kernel, host_mutation=host_mutation)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            if case.get("skipped"):
                log(f"[{label}] {index}/{total} SKIPPED {case['skipped'][:60]}")
            else:
                verdict = case["vs_array_order"]
                log(f"[{label}] {index}/{total} guard={guard} {sub_step} "
                    f"{'x'.join(str(n) for n in shape)} "
                    f"{'/'.join(b[:4] for b in boundaries)} dtdx={dtdx!r} "
                    f"phase={phase:+d} {row_parity}: "
                    f"identical={verdict['bit_identical']} "
                    f"(differing={verdict['differing_floats']}/"
                    f"{verdict['total_floats']}, "
                    f"maxulp={verdict.get('max_ulp', 0)}) "
                    f"({case['seconds']} s)")
            results.setdefault("synthetic", {})[label] = summarize(cases)
            # Every 25 cases rather than every case: the artifact carries every
            # case, so rewriting it per case is quadratic in the sweep length and
            # the sweep is thousands of cases long. 25 keeps the progress file
            # useful (the progress-reporting rule) without making the write the measurement.
            if index % 25 == 0:
                save(results, out_path)
    summary = summarize(cases)
    summary["cases"] = cases
    results.setdefault("synthetic", {})[label] = summary
    save(results, out_path)
    return summary


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass counts as N/N per guard, never as the word 'matches'."""
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in cases:
        bucket = per_guard.setdefault(str(case.get("guard")),
                                      {"identical": 0, "ran": 0, "skipped": 0})
        if case.get("skipped"):
            bucket["skipped"] += 1
            continue
        bucket["ran"] += 1
        bucket["identical"] += int(case["vs_array_order"]["bit_identical"])
    guarded = [c for c in cases if c.get("guard") is False and not c.get("skipped")]
    return {
        "per_guard": per_guard,
        "guarded_ran": len(guarded),
        "guarded_identical": sum(int(c["vs_array_order"]["bit_identical"])
                                 for c in guarded),
        "guarded_pass": bool(guarded) and all(
            c["vs_array_order"]["bit_identical"] for c in guarded),
    }


# ---------------------------------------------------------------------------
# Source mutations
# ---------------------------------------------------------------------------
# Each transform returns (mutated source, sites hit); a mutation that matched
# NOTHING is refused, because a needle that has drifted away from the kernel is
# not exercising anything.

_TOP_MASK = re.compile(r"tl\.where\(last_[xyz], 0\.0, (curl\d)\)")
_CELL0_IF = re.compile(r"if BC([XYZ]) != PERIODIC:")
_TOP_IF = re.compile(r"if BC([XYZ]) == MIRROR_PERIODIC:")
_GHOST_IF = re.compile(r"if BC([XYZ]) == PERIODIC:")
_STENCIL = re.compile(r"dtdx \* \(\((\w+) - (\w+)\) \+ \((\w+) - (\w+)\)\)")
_FU_STORE = re.compile(r"\n *tl\.store\(u\d \+ idx, n\d, mask=live\)")
_RECURRENCE = re.compile(
    r"n(?P<t>\d) = \(\(p(?P=t) \* km_(?P<s>[xyz])\) - curl(?P=t)\) \* si_(?P=s)\n"
    r"(?P<pad>\s+)v(?P=t) = \(\(\(tl\.load\(f(?P=t) \+ idx, mask=live, other=0\.0\)"
    r" \* km_(?P<u>[xyz])\) \+ n(?P=t)\) - p(?P=t)\) \* si_(?P=u)")


def drop_top_plane_mask(source: str) -> Tuple[str, int]:
    """(a) The fold's own mask is gone. CAUGHT ONLY ON A ``MIRROR_PERIODIC`` CASE.

    Rewritten to ``curl0 = curl0`` rather than deleted: the statement is the whole
    body of a constexpr ``if`` and removing it is a syntax error, and a mutation
    that fails to compile measures nothing.
    """
    return _TOP_MASK.sub(r"\1", source), len(_TOP_MASK.findall(source))


def drop_cell_zero_mask_on_a_fold(source: str) -> Tuple[str, int]:
    """(b) The cell-0 mask narrows back to walls only, so a folded axis loses it.

    ``!= PERIODIC`` -> ``== METALLIC`` in the mask block. The ghost-rule block
    spells its test ``== PERIODIC``, so this needle cannot reach it.
    """
    return _CELL0_IF.sub(r"if BC\1 == METALLIC:", source), \
        len(_CELL0_IF.findall(source))


def top_plane_mask_on_a_metallic_fold_too(source: str) -> Tuple[str, int]:
    """(c) The top mask fires on BOTH folds. Deletes a plane the metallic fold steps.

    The classification getting it backwards in this direction is the silent half
    of risk 5: a spurious top mask on a folded METALLIC axis wipes a stepped
    plane. ``== MIRROR_PERIODIC`` -> ``>= MIRROR_METALLIC``, i.e. any mirror.
    """
    return _TOP_IF.sub(r"if BC\1 >= MIRROR_METALLIC:", source), \
        len(_TOP_IF.findall(source))


def fold_ghost_wraps(source: str) -> Tuple[str, int]:
    """(d) A folded axis WRAPS instead of terminating. Caught on the metallic fold.

    ``== PERIODIC`` -> ``!= METALLIC`` in the ghost block. On a folded PERIODIC
    axis both ghosts are masked away, so this is invisible there — which is the
    point, and the leg records it: the ONLY place the folded ghost value is live
    is the top plane of a folded METALLIC axis, where it must be exactly 0.0.
    """
    return _GHOST_IF.sub(r"if BC\1 != METALLIC:", source), \
        len(_GHOST_IF.findall(source))


def regroup_stencil(source: str) -> Tuple[str, int]:
    """(e) ``dtdx*((a-b)+(c-d))`` -> ``((a-b)+c)-d``. The associativity family."""
    return _STENCIL.sub(r"dtdx * (\1 - \2 + \3 - \4)", source), \
        len(_STENCIL.findall(source))


def drop_fu_store(source: str) -> Tuple[str, int]:
    """(f) ``f`` right on every launch, ``fu`` never written — the auxiliary defect."""
    return _FU_STORE.sub("", source), len(_FU_STORE.findall(source))


def swap_dsig_dsigu(source: str) -> Tuple[str, int]:
    """(g) The recurrence reads dsigu's coefficients where dsig's belong."""
    def swap(match: "re.Match[str]") -> str:
        t, s, u, pad = match["t"], match["s"], match["u"], match["pad"]
        return (f"n{t} = ((p{t} * km_{u}) - curl{t}) * si_{u}\n"
                f"{pad}v{t} = (((tl.load(f{t} + idx, mask=live, other=0.0) "
                f"* km_{s}) + n{t}) - p{t}) * si_{s}")
    return _RECURRENCE.sub(swap, source), len(_RECURRENCE.findall(source))


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "drop_top_plane_mask": drop_top_plane_mask,
    "drop_cell_zero_mask_on_a_fold": drop_cell_zero_mask_on_a_fold,
    "top_plane_mask_on_a_metallic_fold_too": top_plane_mask_on_a_metallic_fold_too,
    "fold_ghost_wraps": fold_ghost_wraps,
    "regroup_stencil": regroup_stencil,
    "drop_fu_store": drop_fu_store,
    "swap_dsig_dsigu": swap_dsig_dsigu,
}

#: Host-side mutations: the kernel is the shipped one, the PLAN lies to it.
HOST_MUTATIONS = ("fold_as_periodic", "swap_fold_termination")

_TEMPORARY: List[str] = []


def compile_mutated(source: str):
    """Compile a mutated copy of the shipped kernel from a real file on disk.

    Triton reads a kernel's text with ``inspect.getsource``, so the mutated
    function has to live in a file; an ``exec``-ed one raises ``OSError`` at first
    launch. The four boundary constexprs are re-declared in the header because a
    ``@triton.jit`` body may not read a plain module global.
    """
    header = (
        "import triton\nimport triton.language as tl\n"
        f"PERIODIC = tl.constexpr({symmetry.CODE_PERIODIC})\n"
        f"METALLIC = tl.constexpr({symmetry.CODE_METALLIC})\n"
        f"MIRROR_METALLIC = tl.constexpr({symmetry.CODE_MIRROR_METALLIC})\n"
        f"MIRROR_PERIODIC = tl.constexpr({symmetry.CODE_MIRROR_PERIODIC})\n\n")
    # encoding= is not optional: the kernel's prose carries non-ASCII and the
    # measurement host's default locale is ASCII, so the default open() dies
    # inside the mutation leg — which is a leg that reports nothing rather than
    # a leg that reports a pass, but it is still a disarmed gate.
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_fold.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_fold_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
    sys.modules[spec.name] = module                      # type: ignore[union-attr]
    spec.loader.exec_module(module)                      # type: ignore[union-attr]
    return module


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Each defect must change the answer. A leg that reports a pass is a failure."""
    shipped = textwrap.dedent(inspect.getsource(symmetry.pml_curl_step_folded.fn))
    out: Dict[str, Any] = {}
    for name, transform in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "the mutation matched nothing; the needle has "
                                  "drifted away from the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            save(results, out_path)
            continue
        module = compile_mutated(mutated)
        summary = run_synthetic(results, out_path,
                                kernel=module.pml_curl_step_folded,
                                label="mutation:" + name, guards=(False,),
                                shapes=SHAPES[:2])
        caught = summary["guarded_identical"] < summary["guarded_ran"]
        out[name] = {"sites": hits, "ran": summary["guarded_ran"],
                     "identical": summary["guarded_identical"],
                     "caught": caught,
                     "caught_on": _caught_on(results["synthetic"]
                                             ["mutation:" + name]["cases"])}
        log(f"[mut] {name}: sites={hits} identical="
            f"{summary['guarded_identical']}/{summary['guarded_ran']} "
            f"CAUGHT={caught} on {out[name]['caught_on']}")
        results["mutations"] = out
        save(results, out_path)
    for name in HOST_MUTATIONS:
        summary = run_synthetic(results, out_path, host_mutation=name,
                                label="mutation:" + name, guards=(False,),
                                shapes=SHAPES[:2])
        caught = summary["guarded_identical"] < summary["guarded_ran"]
        out[name] = {"host": True, "ran": summary["guarded_ran"],
                     "identical": summary["guarded_identical"], "caught": caught,
                     "caught_on": _caught_on(results["synthetic"]
                                             ["mutation:" + name]["cases"])}
        log(f"[mut] {name}: identical={summary['guarded_identical']}/"
            f"{summary['guarded_ran']} CAUGHT={caught} on {out[name]['caught_on']}")
        results["mutations"] = out
        save(results, out_path)
    results["mutations"] = out
    save(results, out_path)
    return out


def _caught_on(cases: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Which boundary families a mutation is visible in — the informative half.

    ``drop_top_plane_mask`` is caught ONLY on a ``MIRROR_PERIODIC`` case;
    ``fold_ghost_wraps`` only on a ``MIRROR_METALLIC`` one. A gate that reported
    a bare "caught" would hide the fact that a sweep without both folds proves
    nothing about one of the two masks.
    """
    counts: Dict[str, int] = {}
    for case in cases:
        if case.get("skipped") or case["vs_array_order"]["bit_identical"]:
            continue
        key = "/".join(sorted({b for b in case["boundaries"] if b in IS_FOLD})
                       or {"none"})
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# The ghost-fill leg — against the array path's own fill passes
# ---------------------------------------------------------------------------
#
# The reference here IS the engine, deliberately, and the reason is that this
# sub-step has no arithmetic to transcribe independently: its only operation is a
# multiply by exactly +/-1, which is exact in float32 for every input including
# subnormals and signed zeros. What can go wrong is INDEXING — which plane, which
# source row, which component, in which order — and the engine's own pass is the
# only statement of that worth comparing against.

FILL_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"axis": "Y", "boundaries": "periodic", "extent": 2.0, "phase": 1,
     "cell": (0.8, 2.0, 0.8)},
    {"axis": "Y", "boundaries": "periodic", "extent": 2.1, "phase": 1,
     "cell": (0.8, 2.1, 0.8)},          # ODD full count -> reflect row n-3
    {"axis": "Y", "boundaries": "periodic", "extent": 2.0, "phase": -1,
     "cell": (0.8, 2.0, 0.8)},          # odd plane
    {"axis": "Y", "boundaries": "metallic", "extent": 2.0, "phase": 1,
     "cell": (0.8, 2.0, 0.8)},          # no far fill at all
    {"axis": "X", "boundaries": "periodic", "extent": 2.1, "phase": 1,
     "cell": (2.1, 0.8, 0.8)},          # folded X, odd
    {"axis": "Z", "boundaries": "periodic", "extent": 2.0, "phase": -1,
     "cell": (0.8, 0.8, 2.0)},          # folded Z, odd plane
    {"axis": "XY", "boundaries": "periodic", "extent": 2.0, "phase": 1,
     "cell": (2.0, 2.0, 0.8)},          # TWO planes: the doubly-unowned corner
)


def build_folded_fields(spec: Dict[str, Any], xp=cp):
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axis"])
    grid = Grid(resolution=10.0, cell_size=spec["cell"], boundaries=spec["boundaries"],
                symmetry=planes, xp=xp)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return grid, fields


def seed_fields(fields, names: Sequence[str], offset: int = 0) -> None:
    rng = np.random.default_rng(SEED + 31 + offset)
    for index, name in enumerate(names):
        host = rng.uniform(-1.0, 1.0,
                           size=fields.grid.shape).astype(np.float32)
        getattr(fields, name)[...] = cp.asarray(np.ascontiguousarray(host))


def run_fill(results: Dict[str, Any], out_path: str,
             kernel=None, entry_mutation: Optional[str] = None,
             label: str = "gate") -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in FILL_GRIDS:
        for family in ("B", "D"):
            started = time.time()
            case: Dict[str, Any] = {"spec": dict(spec), "family": family,
                                    "entry_mutation": entry_mutation}
            grid, fields = build_folded_fields(spec)
            names = symmetry.GHOST_FILL_FAMILIES[family]["targets"]
            seed_fields(fields, names)
            reference = Fields(grid=grid, force_complex_fields=False)
            reference.enable_pml_storage()
            for name in names:
                getattr(reference, name)[...] = getattr(fields, name)

            if family == "B":
                stepping.fill_symmetry_bc_B(reference)
                stepping.fill_folded_far_ghosts_B(reference)
            else:
                stepping.fill_symmetry_bc_D(reference)
                stepping.fill_folded_far_ghosts_D(reference)

            entries = list(symmetry.ghost_fill_axis_entries(grid, family))
            if entry_mutation == "reflect_row_n_minus_two":
                for entry in entries:
                    if entry["far"]:
                        entry["reflect_row"] = int(grid.stored_cells(
                            entry["axis"])) - 2
            elif entry_mutation == "reverse_axis_order":
                entries = list(reversed(entries))
            elif entry_mutation == "flip_far_parity":
                for entry in entries:
                    entry["phase"] = -int(entry["phase"])
            elif entry_mutation == "drop_far_fill":
                for entry in entries:
                    entry["far"] = False
            elif entry_mutation == "swap_near_and_far_shifts":
                for entry in entries:
                    entry["shifts"] = tuple(1 - int(s) for s in entry["shifts"])
            plan = symmetry.plan_mirror_ghost_fill_from_arrays(
                family, {name: getattr(fields, name) for name in names},
                entries, kernel=kernel)
            plan.run()
            cp.cuda.runtime.deviceSynchronize()

            case["entries"] = [{k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in entry.items()} for entry in entries]
            case["verdict"] = combine(
                {name: bit_compare(getattr(fields, name), getattr(reference, name))
                 for name in names})
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            log(f"[fill:{label}] {spec['axis']}/{spec['boundaries']}/"
                f"phase{spec['phase']:+d}/n={grid.stored_cells(0)}x"
                f"{grid.stored_cells(1)}x{grid.stored_cells(2)} {family}: "
                f"identical={case['verdict']['bit_identical']} "
                f"(differing={case['verdict']['differing_floats']}/"
                f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
            results.setdefault("fill", {})[label] = {
                "ran": len(cases),
                "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
                "cases": cases}
            save(results, out_path)
    return results["fill"][label]


#: Fill-kernel source mutations, aimed at ``mirror_ghost_fill`` and nothing else.
_FILL_NEAR_SOURCE = re.compile(r"base \+ 2 \* stride")


def near_source_row_three(source: str) -> Tuple[str, int]:
    """The near ghost images stored cell 3 instead of ``MIRROR_SOURCE_INDEX = 2``.

    A whole cell wrong on every folded run, on one plane, smooth and plausible.
    """
    return _FILL_NEAR_SOURCE.sub("base + 3 * stride", source), \
        len(_FILL_NEAR_SOURCE.findall(source))


FILL_SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "near_source_row_three": near_source_row_three,
}

#: Entry (host) mutations, and whether each MUST be caught. ``reverse_axis_order``
#: is a NULL and it is here to be measured, not asserted: every fill is a multiply
#: by exactly +/-1 from a plane no other axis's fill writes, so two axes' fills
#: COMMUTE bitwise and the doubly-unowned corner carries ph_x*ph_y either way.
#: The plan still launches in X, Y, Z order — matching the array path is what makes
#: that a fact about this configuration rather than a property to rely on — and the
#: leg records the null so a future reader does not read "uncaught" as "disarmed".
FILL_ENTRY_MUTATIONS: Dict[str, bool] = {
    "reflect_row_n_minus_two": True,
    "flip_far_parity": True,
    "drop_far_fill": True,
    "swap_near_and_far_shifts": True,
    "reverse_axis_order": False,
}


def run_fill_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, must_catch in FILL_ENTRY_MUTATIONS.items():
        summary = run_fill(results, out_path, entry_mutation=name,
                           label="mutation:" + name)
        caught = summary["identical"] < summary["ran"]
        out[name] = {"ran": summary["ran"], "identical": summary["identical"],
                     "caught": caught, "must_catch": must_catch,
                     "as_expected": caught == must_catch,
                     "caught_on": [c["spec"] for c in summary["cases"]
                                   if not c["verdict"]["bit_identical"]]}
        log(f"[fill-mut] {name}: identical={summary['identical']}/"
            f"{summary['ran']} CAUGHT={caught} (must_catch={must_catch})")
        results["fill_mutations"] = out
        save(results, out_path)
    shipped = textwrap.dedent(inspect.getsource(symmetry.mirror_ghost_fill.fn))
    for name, transform in FILL_SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "the mutation matched nothing"}
            log(f"[fill-mut] {name}: NEEDLE MISSED")
            continue
        module = compile_mutated(mutated)
        summary = run_fill(results, out_path, kernel=module.mirror_ghost_fill,
                           label="mutation:" + name)
        caught = summary["identical"] < summary["ran"]
        out[name] = {"sites": hits, "ran": summary["ran"],
                     "identical": summary["identical"], "caught": caught,
                     "must_catch": True, "as_expected": caught}
        log(f"[fill-mut] {name}: sites={hits} identical={summary['identical']}/"
            f"{summary['ran']} CAUGHT={caught}")
        results["fill_mutations"] = out
        save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The engine leg — the real array path, on a real folded grid, sub-step by sub-step
# ---------------------------------------------------------------------------

ENGINE_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"axis": "Y", "boundaries": "periodic", "cell": (1.6, 2.0, 1.2), "phase": 1},
    {"axis": "Y", "boundaries": "periodic", "cell": (1.6, 2.1, 1.2), "phase": 1},
    {"axis": "Y", "boundaries": "metallic", "cell": (1.6, 2.0, 1.2), "phase": 1},
    {"axis": "Y", "boundaries": "periodic", "cell": (1.6, 2.0, 1.2), "phase": -1},
    {"axis": "X", "boundaries": "periodic", "cell": (2.1, 1.2, 1.2), "phase": 1},
    {"axis": "Z", "boundaries": "metallic", "cell": (1.2, 1.2, 2.0), "phase": 1},
    {"axis": "XY", "boundaries": "periodic", "cell": (2.0, 2.0, 1.2), "phase": 1},
)
ENGINE_STEPS = 4


def run_engine(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """``stepping.step_B``/``step_D`` against the plan, from identical state.

    THE STRONGEST LEG, and the one the synthetic sweep cannot be: it uses the REAL
    graded PML tables at the STORED extent of a folded axis, the real
    ``_boundary_kinds`` resolution, the real ``_far_reflect_rows``, and the
    predicate + plan builder the engine would use. What it is NOT is an
    independent transcription, which is why the synthetic leg exists as well.
    """
    cases: List[Dict[str, Any]] = []
    for spec in ENGINE_GRIDS:
        started = time.time()
        planes = tuple(Mirror(name, spec["phase"]) for name in spec["axis"])
        grid = Grid(resolution=10.0, cell_size=spec["cell"],
                    boundaries=spec["boundaries"], symmetry=planes, xp=cp)
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness=2)
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
        seed_fields(fields, names)

        verdict = symmetry.folded_pml_curl_coverage(fields, pml)
        case: Dict[str, Any] = {
            "spec": dict(spec), "shape": list(grid.shape),
            "covered": verdict.covered, "reasons": list(verdict.reasons),
            "codes": list(symmetry.folded_axis_kinds(grid, pml)[0] or ()),
            "kinds": list(stepping._boundary_kinds(grid, pml)),
            "reflect_rows": list(stepping._far_reflect_rows(grid)),
            "stored": [grid.stored_cells(a) for a in range(3)],
            "owned": [grid.owned_cells(a) for a in range(3)],
        }
        if not verdict.covered:
            case["error"] = "the predicate refused a configuration the gate built"
            cases.append(case)
            log(f"[engine] {spec} REFUSED: {verdict.reasons}")
            continue

        reference = Fields(grid=grid, force_complex_fields=False)
        reference.enable_pml_storage()
        for name in names:
            getattr(reference, name)[...] = getattr(fields, name)

        parts: Dict[str, Any] = {}
        for _ in range(ENGINE_STEPS):
            for sub_step, targets in (("step_B", ("Bx", "By", "Bz")),
                                      ("step_D", ("Dx", "Dy", "Dz"))):
                plan = symmetry.plan_folded_pml_curl(fields, pml, sub_step)
                plan.run()
                getattr(stepping, sub_step)(reference, pml)
                cp.cuda.runtime.deviceSynchronize()
                for name in targets + tuple("fu_" + t for t in targets):
                    parts[name] = bit_compare(getattr(fields, name),
                                              getattr(reference, name))
        case["verdict"] = combine(parts)
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[engine] {spec['axis']}/{spec['boundaries']}/phase{spec['phase']:+d} "
            f"shape={tuple(grid.shape)} codes={case['codes']} "
            f"rows={case['reflect_rows']}: "
            f"identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results["engine"] = {
            "ran": len(cases),
            "identical": sum(int(c.get("verdict", {}).get("bit_identical", False))
                             for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["engine"]


# ---------------------------------------------------------------------------
# The fill benchmark
# ---------------------------------------------------------------------------

BENCH_CELLS: Tuple[float, ...] = (3.2, 6.4, 12.8, 25.6)
BENCH_REPEATS = 40


def run_bench(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Array-path fills vs the fill kernel, per step, at four sizes.

    The array path's cost is six strided whole-plane assignments on a folded
    periodic axis (two on a metallic one) and it is LAUNCH-bound, not
    bandwidth-bound — which is exactly the shape of cost one kernel removes.
    """
    rows: List[Dict[str, Any]] = []
    for boundaries in ("periodic", "metallic"):
        for cell in BENCH_CELLS:
            # Per-axis, because z is INVARIANT in a 2-D run and Grid refuses a
            # wall there: an invariant direction has no outer face at all.
            grid = Grid(resolution=20.0, cell_size=(cell, cell, 0.0),
                        dimensions=2, boundaries=(boundaries, boundaries,
                                                  "periodic"),
                        symmetry=(Mirror("Y", 1),), xp=cp)
            fields = Fields(grid=grid, force_complex_fields=False)
            fields.enable_pml_storage()
            names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
            seed_fields(fields, names)
            plans = {family: symmetry.plan_mirror_ghost_fill(fields, family)
                     for family in ("B", "D")}
            if any(plan is None for plan in plans.values()):
                continue

            def array_path() -> None:
                stepping.fill_symmetry_bc_B(fields)
                stepping.fill_folded_far_ghosts_B(fields)
                stepping.fill_symmetry_bc_D(fields)
                stepping.fill_folded_far_ghosts_D(fields)

            def kernel_path() -> None:
                plans["B"].run()
                plans["D"].run()

            row = {"boundaries": boundaries, "shape": list(grid.shape),
                   "array_us": _time(array_path), "kernel_us": _time(kernel_path)}
            row["speedup"] = row["array_us"] / row["kernel_us"]
            rows.append(row)
            log(f"[bench] {boundaries} {tuple(grid.shape)}: array "
                f"{row['array_us']:.1f} us  kernel {row['kernel_us']:.1f} us  "
                f"{row['speedup']:.1f}x")
            results["bench"] = rows
            save(results, out_path)
    return {"rows": rows}


def _time(callable_: Callable[[], None]) -> float:
    for _ in range(5):
        callable_()
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    for _ in range(BENCH_REPEATS):
        callable_()
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - started) / BENCH_REPEATS * 1e6


# ---------------------------------------------------------------------------
# The recorded verdict
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Every leg above decides something — 512/512 guarded, 0/512
# unguarded, 14 mutations caught, 7/7 on real folded grids — and until now the
# gate wrote all of it down and never wrote down what it CONCLUDED. Measured
# 2026-08-19: ``gate_provenance.read_verdict`` returns ``released=None`` on the
# 2026-08-10 artifact, not because the gate failed but because there is no shape
# in it to read, and a weld cannot cite a record that does not state its own
# outcome. This section states it, in the ``verdict``-dict shape that
# ``read_verdict`` already understands (``verdict.pass``, the shape
# ``gate_triton_cylindrical`` writes) — not an eleventh spelling of one fact.
#
# TWO RULES SHAPE EVERY CLAUSE BELOW.
#
# 1. A COUNT IS COMPARED AGAINST WHAT A COMPLETE LEG PRODUCES, never against
#    itself. The artifact is rewritten every 25 cases (progress reporting), and every leg's
#    running summary is a ratio of what has landed SO FAR — so a sweep killed at
#    case 300 leaves ``300/300 identical`` behind it. Against itself that reads
#    as a pass; against the 512 the constants say a complete sweep runs, it does
#    not. The expected counts are therefore derived from the sweep constants,
#    which is also what makes them follow an edit to those constants.
#
# 2. A LEG THAT FAILS MUST MAKE THE VERDICT FALSE, NOT MAKE THE PROCESS RAISE.
#    A gate that dies writes no verdict at all, and ``read_verdict`` reports
#    that as ``released=None`` — UNREADABLE, which is deliberately not the same
#    claim as refused. A refusal that arrives dressed as a silence is the hole
#    this whole exercise is closing, so ``main`` runs each leg inside
#    :func:`_run_leg`, records what it raised, and lets the clause read the
#    absence.
#
# ``bench`` IS NOT A CLAUSE. It is a timing measurement with no expected value —
# there is no speedup this gate refuses to release below — and a gate must not
# be able to release on a leg it asserts nothing about. It is reported, and the
# release rests on the five legs that compare bytes.

#: The legs a release rests on. Order is the order they run in.
CERTIFYING_LEGS: Tuple[str, ...] = ("synthetic", "mutations", "fill",
                                    "fill_mutations", "engine")

#: Everything ``main`` runs by default: the five above, plus the timing leg.
DEFAULT_LEGS: Tuple[str, ...] = CERTIFYING_LEGS + ("bench",)


def expected_synthetic_cases(shapes: Sequence[Tuple[int, int, int]] = SHAPES,
                             boundary_sets: Sequence[Tuple[str, str, str]]
                             = BOUNDARY_SETS) -> int:
    """How many cases ONE guard value of the synthetic sweep runs to completion.

    Derived from the same constants ``run_synthetic`` builds its product from,
    with the same skip rule (:func:`viable`) applied — so a shape added to
    ``SHAPES`` moves both the sweep and the number the verdict holds it to.
    """
    per_pair = len(DTDX) * len(PHASES) * len(ROW_PARITIES) * 2  # both sub-steps
    return sum(per_pair for shape in shapes for boundaries in boundary_sets
               if viable(shape, boundaries) is None)


def expected_multi_step_cases() -> int:
    """How many of :data:`MULTI_STEP_COMBOS` a complete multi-step leg runs."""
    return sum(1 for shape, boundaries, _, _, _ in MULTI_STEP_COMBOS
               if viable(shape, boundaries) is None)


def expected_fill_cases() -> int:
    """Grids x families — what a complete ghost-fill leg compares."""
    return 2 * len(FILL_GRIDS)


def _synthetic_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Guarded identical on every case, and the unguarded control identical on none.

    BOTH HALVES ARE REQUIRED. The guarded half alone cannot distinguish a kernel
    that matches from a comparator that would call anything identical; the
    unguarded half (``enable_fp_fusion=True``) is the positive control that says
    the comparison can see a difference, and it is only evidence while it fails
    on every case it runs. Measured 2026-08-10: 512/512 and 0/512.
    """
    record = (results.get("synthetic") or {}).get("gate") or {}
    expected = expected_synthetic_cases()
    unguarded = (record.get("per_guard") or {}).get("True") or {}
    detail = {
        "guarded": f"{record.get('guarded_identical')}/{record.get('guarded_ran')}",
        "unguarded_identical": f"{unguarded.get('identical')}/{unguarded.get('ran')}",
        "expected_per_guard": expected,
    }
    ok = (int(record.get("guarded_ran", 0)) == expected
          and int(record.get("guarded_identical", -1)) == expected
          and int(unguarded.get("ran", 0)) == expected
          and int(unguarded.get("identical", -1)) == 0)
    return ok, detail


def _multi_step_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Several sub-steps from one state, identical on every combination.

    A separate clause from the single-launch sweep because it is a separate
    claim: the ``fu_*`` auxiliary is STATE, and a kernel that writes the target
    correctly and the auxiliary wrongly is right for exactly one launch.
    """
    record = results.get("multi_step") or {}
    expected = expected_multi_step_cases()
    ok = (int(record.get("ran", 0)) == expected
          and int(record.get("identical", -1)) == expected)
    return ok, {"identical": f"{record.get('identical')}/{record.get('ran')}",
                "expected": expected, "steps_each": MULTI_STEP_COUNT}


def _mutations_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Every curl defect changes the answer, and every one of them was tried.

    THE NAME LIST IS CHECKED, not just the entries present. A mutation whose
    needle drifted off the kernel records an ``error`` and no ``caught`` key, and
    a leg that died halfway records nothing at all for the rest — both of which
    would otherwise leave a shorter dict in which every entry says ``caught``.
    """
    record = results.get("mutations") or {}
    names = tuple(SOURCE_MUTATIONS) + tuple(HOST_MUTATIONS)
    expected_cases = expected_synthetic_cases(SHAPES[:2])
    missing = [name for name in names if name not in record]
    errored = [name for name in names if (record.get(name) or {}).get("error")]
    not_caught = [name for name in names
                  if name in record and not (record[name] or {}).get("caught")]
    short = [name for name in names
             if name in record and "ran" in (record[name] or {})
             and int(record[name]["ran"]) != expected_cases]
    ok = not (missing or errored or not_caught or short)
    return ok, {"caught": f"{len(names) - len(missing) - len(not_caught)}/{len(names)}",
                "missing": missing, "errored": errored,
                "not_caught": not_caught, "short_sweeps": short,
                "cases_each_expected": expected_cases}


def _fill_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """The ghost-fill kernel identical to the array path's own fills, every grid."""
    record = (results.get("fill") or {}).get("gate") or {}
    expected = expected_fill_cases()
    ok = (int(record.get("ran", 0)) == expected
          and int(record.get("identical", -1)) == expected)
    return ok, {"identical": f"{record.get('identical')}/{record.get('ran')}",
                "expected": expected}


def _fill_mutations_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Every fill defect lands as EXPECTED — including the one that must not bite.

    ``reverse_axis_order`` is a measured null (two axes' fills commute bitwise),
    and ``as_expected`` is what this clause reads, not ``caught``: a null that
    started being caught is as much a change in the contract as a real defect
    that stopped being.
    """
    record = results.get("fill_mutations") or {}
    names = tuple(FILL_ENTRY_MUTATIONS) + tuple(FILL_SOURCE_MUTATIONS)
    expected_cases = expected_fill_cases()
    missing = [name for name in names if name not in record]
    errored = [name for name in names if (record.get(name) or {}).get("error")]
    unexpected = [name for name in names
                  if name in record and not (record[name] or {}).get("as_expected")]
    short = [name for name in names
             if name in record and "ran" in (record[name] or {})
             and int(record[name]["ran"]) != expected_cases]
    ok = not (missing or errored or unexpected or short)
    nulls = {name: (record.get(name) or {}).get("caught")
             for name, must_catch in FILL_ENTRY_MUTATIONS.items() if not must_catch}
    return ok, {
        "as_expected": f"{len(names) - len(missing) - len(unexpected)}/{len(names)}",
        "missing": missing, "errored": errored, "not_as_expected": unexpected,
        "short_sweeps": short, "measured_nulls_caught": nulls,
        "cases_each_expected": expected_cases}


def _engine_clause(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """``stepping.step_B``/``step_D`` substituted on real folded grids, identical.

    A REFUSED CONFIGURATION FAILS THIS CLAUSE. The predicate declining a grid the
    gate built is not a neutral outcome here: this leg exists to exercise the
    real coverage predicate and the real plan builder, and a refusal means the
    kernel was never launched on that grid — an untested configuration, recorded
    as though the leg had run.
    """
    record = results.get("engine") or {}
    expected = len(ENGINE_GRIDS)
    refused = [case.get("spec") for case in (record.get("cases") or [])
               if not case.get("covered") or case.get("error")]
    ok = (int(record.get("ran", 0)) == expected
          and int(record.get("identical", -1)) == expected
          and not refused)
    return ok, {"identical": f"{record.get('identical')}/{record.get('ran')}",
                "expected": expected, "refused": refused,
                "steps_each": ENGINE_STEPS}


#: clause name -> reader. Every certifying leg is here; the synthetic leg carries
#: two clauses because it makes two independent claims.
_CLAUSES: Tuple[Tuple[str, str, Callable[[Dict[str, Any]],
                                         Tuple[bool, Dict[str, Any]]]], ...] = (
    ("synthetic", "synthetic_guarded_identical_and_control_bites", _synthetic_clause),
    ("synthetic", "multi_step_identical", _multi_step_clause),
    ("mutations", "every_curl_mutation_caught", _mutations_clause),
    ("fill", "ghost_fills_identical", _fill_clause),
    ("fill_mutations", "every_fill_mutation_as_expected", _fill_mutations_clause),
    ("engine", "engine_substitution_identical", _engine_clause),
)


def verdict(results: Dict[str, Any], legs: Sequence[str]) -> Dict[str, Any]:
    """The gate's own reading of its own rows. Every clause stated, none inferred.

    Returns the ``verdict`` dict the artifact carries. ``pass`` is true only when
    the run covered every certifying leg AND every clause reads true; a clause
    whose reader raises on a half-written record is FALSE with the exception
    recorded, because a verdict function that dies takes the verdict with it.
    """
    out: Dict[str, Any] = {
        "legs_requested": list(legs),
        "certifying_legs": list(CERTIFYING_LEGS),
        "clauses": {},
        "detail": {},
        "reasons": [],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    requested = set(legs)

    # COMPLETENESS FIRST. ``--legs bench`` must not be able to produce a released
    # artifact: with no certifying leg in the run there is nothing to be wrong,
    # and "no clause failed" is not the same statement as "the gate certified".
    missing_legs = [name for name in CERTIFYING_LEGS if name not in requested]
    out["clauses"]["covers_every_certifying_leg"] = not missing_legs
    if missing_legs:
        out["reasons"].append(
            "this run did not include the certifying leg(s) "
            + ", ".join(missing_legs))

    for leg_name, clause_name, reader in _CLAUSES:
        if leg_name not in requested:
            continue
        try:
            ok, detail = reader(results)
        except Exception as exc:  # noqa: BLE001 - a verdict must not die reading
            ok, detail = False, {"error": f"{type(exc).__name__}: {exc}"}
        out["clauses"][clause_name] = bool(ok)
        out["detail"][clause_name] = detail
        if not ok:
            out["reasons"].append(f"{clause_name}: {json.dumps(detail, default=str)}")

    # A LEG THAT RAISED, STATED SEPARATELY. Its clause is already false (a leg
    # that died wrote a short record, or none), but a leg can also raise AFTER
    # its record is complete — in the summary log, or in its own final save — and
    # that failure would otherwise leave no mark on the verdict at all.
    raised = sorted(name for name in (results.get("leg_errors") or {})
                    if name in CERTIFYING_LEGS)
    out["clauses"]["no_certifying_leg_raised"] = not raised
    if raised:
        out["reasons"].append("certifying leg(s) raised: " + ", ".join(raised))

    # THE POLICY IS A RELEASE CLAUSE, NOT A DECORATION. Every clause above is a
    # byte comparison, and a byte comparison is a statement about one float32
    # subnormal policy; a run whose strip was never installed or never exercised
    # compared bytes this engine does not ship, and the weld quoting it would
    # name a policy the run did not hold.
    stamp = results.get("subnormal_policy") or {}
    policy_reasons = list(results.get("subnormal_policy_license_reasons") or ())
    named = str(stamp.get("resolved") or "") in ("keep", "flush")
    out["clauses"]["cut_under_a_named_subnormal_policy"] = bool(
        named and not policy_reasons)
    out["detail"]["cut_under_a_named_subnormal_policy"] = {
        "policy": stamp.get("policy"), "resolved": stamp.get("resolved"),
        "requested": stamp.get("requested"),
        "resolved_from": stamp.get("resolved_from"),
        "nvrtc_calls": stamp.get("nvrtc_calls"),
        "ftz_removed": stamp.get("ftz_removed"),
        "cache_preexisting": stamp.get("cache_preexisting"),
        "measured": stamp.get("measured"),
        "license_reasons": policy_reasons}
    if not out["clauses"]["cut_under_a_named_subnormal_policy"]:
        out["reasons"].append(
            "subnormal policy: " + "; ".join(
                policy_reasons or ["the artifact names no policy"]))

    bench = results.get("bench")
    out["bench_rows"] = len(bench) if isinstance(bench, list) else None
    out["bench_note"] = ("a timing measurement with no expected value; reported, "
                         "never a release clause")

    out["pass"] = bool(out["clauses"]) and all(out["clauses"].values())
    return out


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
        json.dump(results, handle, indent=1, sort_keys=True, default=str)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=",".join(DEFAULT_LEGS))
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    # THE SHIP POLICY, INSTALLED BEFORE THE FIRST DEVICE COMPILE. This gate's
    # ORACLE IS CuPy — reference_pml_step is called with ``cp``, never ``np`` —
    # and CuPy appends '-ftz=true' to every NVRTC compile unconditionally
    # (cupy/cuda/compiler.py:552) while Triton keeps subnormals natively. Without
    # this call the comparison is a FLUSHING oracle against a KEEPING kernel and
    # the identity it records names no policy at all, which is the _mixed_policy
    # debt three welds already carry. Raises at startup on a cache directory that
    # could serve one policy's binaries under the other's name.
    install_ftz_strip()

    import triton  # noqa: PLC0415

    results: Dict[str, Any] = {
        "device": probe.device_info(),
        "subnormal_policy": policy_stamp("cupy"),  # refreshed after the legs
        "triton_version": triton.__version__,
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "sweep": {"shapes": [list(s) for s in SHAPES],
                  "boundary_sets": [list(b) for b in BOUNDARY_SETS],
                  "dtdx": [repr(v) for v in DTDX],
                  "phases": list(PHASES),
                  "row_parities": list(ROW_PARITIES)},
        "kernel_sha256": _digest(symmetry.__file__),
        "granularity": "sub-step; a whole-step comparison cannot see either fold mask",
        "legs_requested": list(legs),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save(results, args.out)

    def run_leg(name: str, body: Callable[[], None]) -> None:
        """One leg, or the record of why it did not finish.

        A LEG IS NOT ALLOWED TO KILL THE RUN. An uncaught exception here writes
        no verdict at all, and ``read_verdict`` reports a verdict-less artifact
        as ``released=None`` — unreadable, which is deliberately NOT the claim
        "refused". Recording the exception and carrying on turns a leg that blew
        up into a verdict that says false, and leaves the legs after it measured
        rather than lost.
        """
        if name not in legs:
            return
        log(f"[leg] {name}")
        try:
            body()
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            import traceback  # noqa: PLC0415
            results.setdefault("leg_errors", {})[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}
            log(f"[leg] {name} RAISED {type(exc).__name__}: {exc}")
            save(results, args.out)

    def synthetic_leg() -> None:
        summary = run_synthetic(results, args.out)
        log(f"[SUMMARY] synthetic guarded {summary['guarded_identical']}/"
            f"{summary['guarded_ran']} identical; unguarded "
            f"{summary['per_guard'].get('True', {})}")
        multi = run_multi_step(results, args.out)
        log(f"[SUMMARY] multi-step {multi['identical']}/{multi['ran']} identical")

    def fill_leg() -> None:
        summary = run_fill(results, args.out)
        log(f"[SUMMARY] fill {summary['identical']}/{summary['ran']} identical")

    def engine_leg() -> None:
        summary = run_engine(results, args.out)
        log(f"[SUMMARY] engine {summary['identical']}/{summary['ran']} identical")

    run_leg("synthetic", synthetic_leg)
    run_leg("mutations", lambda: run_mutations(results, args.out))
    run_leg("fill", fill_leg)
    run_leg("fill_mutations", lambda: run_fill_mutations(results, args.out))
    run_leg("engine", engine_leg)
    run_leg("bench", lambda: run_bench(results, args.out))

    # REFRESHED AFTER THE LEGS, not only at startup: the strip's counters mean
    # nothing until this run's compiles have gone through them.
    results["subnormal_policy"] = policy_stamp("cupy")
    results["subnormal_policy_license_reasons"] = ftz_strip_license_reasons()

    results["verdict"] = verdict(results, legs)
    save(results, args.out)
    log(f"VERDICT: pass={results['verdict']['pass']} "
        f"{json.dumps(results['verdict']['clauses'], sort_keys=True)}")
    for reason in results["verdict"]["reasons"]:
        log(f"  REASON {reason}")
    log(f"[done] {args.out}")
    return 0 if results["verdict"]["pass"] else 1


def _digest(path: str) -> str:
    import hashlib  # noqa: PLC0415

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
