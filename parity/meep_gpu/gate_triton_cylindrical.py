"""Bit-identity gate for the cylindrical (Dcyl, m = 0) Triton curl kernel.

WHY THIS IS A SEPARATE FILE AND NOT A BRANCH OF THE SHARED PROBE. The shared probe
(``probe_fused_kernel_bit_identity.py``) and the shared track adapter
(``track_triton_pml.py``) are being edited RIGHT NOW by two concurrent workflows, so
this round does not touch either — the same call ``gate_triton_symmetry.py`` made,
and for the same reason. Everything reusable is IMPORTED from the shared probe and
nothing is re-implemented: the byte comparator, the ULP-ordered key, the combiner,
the synthetic coefficient tables, the seeded field builder. What is added here is
only what does not exist there — the cylindrical prefix, the axis ghost rule, the
m = 0 axis rules — plus the two legs a Dcyl kernel needs (a real-engine substitution,
and the not-yet-measured constitutive claim).

THE REFERENCE IS NOT WRITTEN HERE. :func:`reference_cyl_step` and its four helpers
are the transcription validated on the laptop BEFORE any of this ran: NumPy,
``lift_simulation`` on a real ``mp.Simulation``, ``stepping.step_B``/``step_D``
driven alternately on the same seeded field set, uint32-view comparison of all
eighteen volumes after every sub-step — **400/400 steps bytewise identical at four
Courant numbers** (0.5, 0.37, pi/10, 1/sqrt(5); three non-power-of-two), with 8 of 9
mutations caught at step 0 and a clean control at 0/9 false positives. It is carried
over verbatim, with its stepping.py line references, rather than rewritten.

THE STEP BUDGET, AND WHY THE LONG-RUN LEGS ARE NOT A PASS/FAIL CLAUSE. 400 steps is
a NUMPY budget: it certifies the transcription, not the kernel. §16 of the plan
measured that a 1-ULP flip amplifies at **0.220 decades/step** and that ``2d_pml`` —
certified 60/60 identical — FIRST DIFFERS AT STEP 67, so a budget can certify
something that diverges just past it. It also measured the other half: the array path
flipped against ITSELF by one ULP, with Triton nowhere, reaches the SAME plateau. So
a long-run divergence is evidence about the kernel only when the ``controls`` leg
says the array path is otherwise deterministic and that this configuration amplifies.
This gate therefore runs a long budget, REPORTS the step at which each row first
differs as a MEASURED budget, and puts the pass/fail weight on the single-launch
sweep, the mutations and the controls.

Legs, in order:

* ``synthetic``    — the sweep. shapes x Courant x sub-step, guarded AND unguarded,
  the shipped plan against :func:`reference_cyl_step` on the same seeded arrays.
* ``multi``        — B and D alternating for the step budget from one state. The
  auxiliary is state: a kernel right in ``field`` and wrong in ``fu`` is correct for
  exactly one launch.
* ``controls``     — array path against array path (``null``, must be clean) and
  against itself with one mantissa bit flipped (``one_ulp``, must diverge). Without
  both, a long-run divergence number means nothing.
* ``engine``       — a REAL cylindrical ``Grid``/``Fields``/``PML``, ``stepping.step_B``/
  ``step_D`` on one field set against ``plan_cylindrical_curl`` (the engine route,
  through the shipped predicate, with the engine's own ``StepScratch``) on a clone,
  compared bytewise after every sub-step. This is the leg that certifies the
  PREDICATE, the PREFIX and the scratch path, not only the kernel.
* ``constitutive`` — the claim finding E made and did not measure: that the EXISTING
  ``kernels.constitutive_step`` is already right on a Dcyl grid. Bytes against
  ``stepping.update_H``/``update_E`` on a real cylindrical grid.
* ``mutations``    — the nine defects the recon established, carried over including
  ``axis_ghost_sign``, whose expected verdict is NOT-CAUGHT-BECAUSE-DEAD.
* ``bench``        — the kernel against the array path, timed, on the benchmark
  case's own shape.

Usage (the GPU host, one CLEAR device)::

    CUDA_VISIBLE_DEVICES=7 python -u gate_triton_cylindrical.py \\
        --out results/triton_cylindrical_<date>/gate.json

Every case prints one flushed line as it lands and the JSON artifact is rewritten
incrementally (the progress-reporting rule).

THE CUPY VERSION IS PART OF THE VERDICT and is recorded in the artifact. Everything
downstream of the radial prefix is bit-identical only for a given ``cupy.cumsum``
summation order; a CuPy bump is a correctness event for this kernel exactly as a
Triton bump is for ``kernels.py``, and a silent one.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cupy as cp  # noqa: E402

import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: E402
from meep_gpu.triton_kernels import kernels as _kernels  # noqa: E402
from meep_gpu.triton_kernels.launch import (  # noqa: E402
    plan_constitutive_from_arrays,
)

SEED = probe.SEED
bit_compare = probe.bit_compare
combine = probe.combine

#: The Dcyl ghost triple, as strings, exactly as ``stepping._boundary_kinds``
#: resolves it. The reference reads these; the kernel compiles them in.
BOUNDARIES: Tuple[str, str, str] = cyl.CYLINDRICAL_BOUNDARY_KINDS

#: ``_mirror_phases``'s r slot at m = 0: ``(-1)^m`` = +1.
MIRROR_PHASE: float = 1.0

FIELD_NAMES: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")


def log(message: str) -> None:
    print(message, flush=True)


# ===========================================================================
# The reference — VERBATIM from the transcription validated on NumPy at 400/400
# ===========================================================================
#
# Transcribed from (meep_gpu/stepping.py, this tree):
#   step_B                          :262-377      (cylindrical branch :323-369, :403-404)
#   step_D                          :379-457      (cylindrical branch :410-423, :456-457)
#   cylindrical_rderiv_prefix       :1257-1305
#   _shift_up / _shift_down         :1723 / :1787 (CYL_AXIS branches :1782, :1836-1846)
#   _curl_from_operands             :1601
#   _mask_non_owned_cells           :1865        (the is_axis clause :1899-1904)
#   _apply_pml_update               :1905
#   _cylindrical_axis_zero_B / _D   :648 / :560

# stepping.B_CURL_TERMS / D_CURL_TERMS (:213-223) + fields.IYEE_SHIFTS (:214).
B_TERMS = (("Bx", "Ez", 1, "Ey", 2, "y", "z", (0, 1, 1)),
           ("By", "Ex", 2, "Ez", 0, "z", "x", (1, 0, 1)),
           ("Bz", "Ey", 0, "Ex", 1, "x", "y", (1, 1, 0)))
D_TERMS = (("Dx", "Hz", 1, "Hy", 2, "y", "z", (1, 0, 0)),
           ("Dy", "Hx", 2, "Hz", 0, "z", "x", (0, 1, 0)),
           ("Dz", "Hy", 0, "Hx", 1, "x", "y", (0, 0, 1)))


def _face(axis, index):
    return (slice(None),) * axis + (index,)


def rderiv_prefix(xp, f_p, ir0):
    """stepping.cylindrical_rderiv_prefix (:1257), the unpooled branch.

    THE SCAN. On CuPy this is ``cupy.cumsum``, which is deterministic but is NOT a
    sequential float32 accumulation — so it, and not exact arithmetic, is the oracle.
    """
    real_dtype = f_p.real.dtype
    counts = xp.arange(f_p.shape[0], dtype=xp.float64) + ir0
    weights = counts.reshape(-1, 1, 1).astype(real_dtype)
    divisor = (counts[1:] - 0.5).reshape(-1, 1, 1).astype(real_dtype)
    weighted = f_p * weights
    increment = xp.zeros_like(f_p)
    increment[1:] = (weighted[1:] - weighted[:-1]) / divisor
    return xp.cumsum(increment, axis=0)


def shift_up(xp, field, axis, boundary):
    """stepping._shift_up (:1723): PERIODIC wraps, METALLIC/CYL_AXIS serve 0 at the far face."""
    shifted = xp.roll(field, -1, axis=axis)
    if boundary == "periodic":
        return shifted
    shifted[_face(axis, -1)] = 0
    return shifted


def shift_down(xp, field, axis, boundary, component, mirror_phase):
    """stepping._shift_down (:1787). The CYL_AXIS near ghost is r_to_minus_r (:1836)."""
    shifted = xp.roll(field, 1, axis=axis)
    if boundary == "periodic":
        return shifted
    if boundary == "metallic":
        shifted[_face(axis, 0)] = 0
        return shifted
    if boundary == "axis":
        sign = -1.0 if component[1] in ("x", "y") else 1.0
        sign *= mirror_phase
        shifted[_face(axis, 0)] = sign * field[_face(axis, 0)]
        return shifted
    raise ValueError(boundary)


def curl_from(first, shifted_first, second, shifted_second, dtdx):
    """stepping._curl_from_operands (:1601). DO NOT flatten the parentheses."""
    return dtdx * ((shifted_first - first) + (second - shifted_second))


def mask(curl, iyee, boundaries):
    """stepping._mask_non_owned_cells (:1865), the metallic + cylindrical-axis clauses."""
    for axis in range(3):
        if iyee[axis] != 0:
            continue
        if boundaries[axis] in ("metallic", "axis"):
            curl[_face(axis, 0)] = 0


def recurrence(field, fu, curl, kms, sinv, kms_u, sinv_u):
    """stepping._apply_pml_update (:1905), in the array path's operand order."""
    fu_previous = fu.copy()
    fu *= kms
    fu -= curl
    fu *= sinv
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u


def reference_cyl_step(xp, sub_step, fields, coefficients, dtdx, boundaries, m,
                       mirror_phase):
    """One cylindrical curl sub-step, in place. m = 0 only.

    Kernel-shaped: explicit per-axis ghosts, explicit masks, the split-field
    recurrence in the array path's operand order — i.e. what a Triton kernel
    computes, written so the comparison is a statement about the kernel rather than
    about NumPy broadcasting.
    """
    if m != 0:
        raise NotImplementedError("m != 0 needs complex storage and the i*m/r coupling")
    backward = sub_step == "step_D"
    terms = D_TERMS if backward else B_TERMS
    sources = ("Hx", "Hy", "Hz") if backward else ("Ex", "Ey", "Ez")
    snap = {name: fields[name] for name in sources}

    if backward:
        # step_D :416-419 — Hp prefixed at ir0 = 0.5, consumed by the Dz term only.
        prefix = rderiv_prefix(xp, snap["Hy"], 0.5)
    else:
        # step_B :322-333 — Ep extended by its ZERO WALL ROW, ir0 = 0.0.
        rows = snap["Ey"].shape[0]
        extended = xp.empty((rows + 1,) + snap["Ey"].shape[1:], dtype=snap["Ey"].dtype)
        extended[:rows] = snap["Ey"]
        extended[rows] = 0
        prefix_ext = rderiv_prefix(xp, extended, 0.0)

    for target, g1, a1, g2, a2, dsig, dsigu, iyee in terms:
        src = dict(snap)
        if backward and target == "Dz":
            src["Hy"] = prefix                      # step_D :425-426
        if backward:
            sf = shift_down(xp, src[g1], a1, boundaries[a1], g1, mirror_phase)
            ss = shift_down(xp, src[g2], a2, boundaries[a2], g2, mirror_phase)
        else:
            sf = shift_up(xp, src[g1], a1, boundaries[a1])
            ss = shift_up(xp, src[g2], a2, boundaries[a2])
        curl = curl_from(src[g1], sf, src[g2], ss, dtdx)
        if (not backward) and target == "Bz":
            # step_B :343-346 — the WHOLE curl is replaced, and the grouping differs:
            # one subtract then one multiply, not the four-operand form above.
            curl = dtdx * (prefix_ext[1:] - prefix_ext[:-1])
        mask(curl, iyee, boundaries)
        recurrence(fields[target], fields["fu_" + target], curl,
                   coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
                   coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])

    if backward:
        # _cylindrical_axis_zero_D (:560), m = 0 branch (:583-585).
        fields["Dz"][_face(0, 0)] += (4.0 * dtdx) * fields["Hy"][_face(0, 0)]
        fields["Dy"][_face(0, 0)] = 0
    else:
        fields["Bx"][_face(0, 0)] = 0               # _cylindrical_axis_zero_B (:648)


# ===========================================================================
# The sweep product
# ===========================================================================
#
# Why each axis is in the product rather than trimmed:
#
# * dtdx — 0.5 ALONE CERTIFIES BROKEN KERNELS. At 0.5 the scaling is exact in binary
#   and the FMA/associativity discrepancies vanish. The three non-power-of-two values
#   are the ones the transcription was validated at, so a disagreement here is
#   between the kernel and a reference already pinned at those numbers.
# * shapes — nphi is ALWAYS 1 (Dcyl is 2.5-D), so the sweep varies nr and nz: a
#   non-square (an i<->k index swap cannot pass), a tall-thin and a short-wide (the
#   prefix runs along r, so the two extremes exercise a long scan and a wide one),
#   an odd pair, and the benchmark case's own 320x320.
# * sub_step — B and D shift in opposite directions, read different coefficient
#   sub-lattices, mask DIFFERENT planes, use the prefix DIFFERENTLY (a replaced curl
#   versus a substituted operand) and apply DIFFERENT axis rules. Neither implies
#   the other; both are always run.
# * guard — enable_fp_fusion False (the shipped value; must be identical) and True
#   (the control; must NOT be identical at a non-power-of-two Courant, which is what
#   proves the gate bites rather than comparing a kernel to itself).

SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (20, 1, 40),     # the laptop-validated lift: res 10, r = z = 4 with PML
    (13, 1, 9),      # odd, non-square
    (64, 1, 8),      # tall and thin: a long radial scan
    (8, 1, 64),      # short and wide: many scan columns, few rows
)
BENCH_SHAPE: Tuple[int, int, int] = (320, 1, 320)

#: 0.5 first so a failure at it is visible as a failure of something other than
#: rounding; the rest are the transcription's own four minus the duplicate.
DTDX: Tuple[float, ...] = (0.5, 0.37, 0.3141592653589793, 0.4472135954999579)

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Default hardware step budget. See the module docstring for why it is not 400.
DEFAULT_STEPS = 1000


def coefficient_tables(shape: Tuple[int, int, int],
                       half_integer: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Seeded kms/sinv in [0.5, 1.0], per axis AND per Yee offset, plus flat views.

    ``probe.synthetic_coefficients`` — a uniform table of ones would hide the
    coefficient indexing completely, and these never touch 1.0. They are used in
    preference to a real Dcyl layer for the synthetic leg BECAUSE a real one carries
    zero sigma on phi and at r = 0, i.e. exactly where a mis-index would hide. The
    ``engine`` leg runs the real layer.
    """
    tables = probe.synthetic_coefficients(cp, shape, half_integer)
    return tables, probe.flatten_coefficients(tables)


def seeded_arrays(shape: Tuple[int, int, int], offset: int = 0) -> Dict[str, Any]:
    """The eighteen volumes, NONZERO everywhere including the auxiliaries.

    A zero ``fu`` makes ``fu*kms`` exactly zero on the first sub-step whatever
    ``kms`` is, so a mis-indexed coefficient would only show from step two.
    """
    return probe.make_pml_fields(shape, np.random.default_rng(SEED + offset))


def first_difference_bits(a_dev: Any, b_dev: Any) -> Optional[Dict[str, Any]]:
    """The BITS at the first differing element, and whether either is subnormal.

    A count alone cannot tell a kernel defect from the amplifying float32
    disagreement plan §16 recorded (``2d_pml``: certified 60/60 identical, first
    differs at step 67, twelve arrays, 112 floats, ``0x004fbb9e`` against
    ``0x00000000``). The bits can: a seed that is one ULP apart, or a subnormal
    against a flushed zero, is that phenomenon; a whole plane apart is not.

    CuPy's flush-to-zero applies to its REDUCTION arithmetic too, so a magnitude
    computed on device under-reports inside the subnormal band (§16.5). Everything
    here is computed on the HOST, in float64, off the raw bits.
    """
    a = probe.to_host(a_dev).ravel().view(np.uint32)
    b = probe.to_host(b_dev).ravel().view(np.uint32)
    where = np.flatnonzero(a != b)
    if where.size == 0:
        return None
    index = int(where[0])
    fa = a[index:index + 1].view(np.float32)[0]
    fb = b[index:index + 1].view(np.float32)[0]

    def classify(bits: int, value: np.float32) -> str:
        exponent = (int(bits) >> 23) & 0xFF
        if exponent == 0:
            return "zero" if (int(bits) & 0x7FFFFF) == 0 else "subnormal"
        if exponent == 0xFF:
            return "inf_or_nan"
        return "normal"

    # ULP distance in the monotone ordered key, the same one bit_compare uses.
    key_a = probe._ordered_key(a[index:index + 1].view(np.float32))
    key_b = probe._ordered_key(b[index:index + 1].view(np.float32))
    return {
        "n_differing": int(where.size),
        "flat_index": index,
        "array_path_bits": f"0x{int(a[index]):08x}",
        "triton_bits": f"0x{int(b[index]):08x}",
        "array_path_value": float(fa),
        "triton_value": float(fb),
        "array_path_class": classify(int(a[index]), fa),
        "triton_class": classify(int(b[index]), fb),
        "ulp_gap": int(abs(int(key_a[0]) - int(key_b[0]))),
    }


def divergence_record(left: Dict[str, Any], right: Dict[str, Any],
                      step_index: int) -> Optional[Dict[str, Any]]:
    """Per-component counts PLUS the bits at the first difference, or None."""
    components: Dict[str, Any] = {}
    for name in FIELD_NAMES:
        detail = first_difference_bits(left[name], right[name])
        if detail is not None:
            components[name] = detail
    if not components:
        return None
    return {"step": step_index,
            "n_components": len(components),
            "total_differing_floats": sum(c["n_differing"]
                                          for c in components.values()),
            "components": components}


def _names(sub_step: str) -> Tuple[str, ...]:
    if sub_step == "step_B":
        return ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    return ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")


# ===========================================================================
# Leg 1 — the synthetic sweep
# ===========================================================================

def mutated_prefix_builder(sub_step: str, mutation: str) -> Callable[..., Any]:
    """``cylindrical_triton.cylindrical_prefix`` built wrong in exactly one way.

    Installed by MONKEYPATCHING the module function the shipped plan calls, rather
    than by editing the plan: the plan is the artifact under test and a wrapper that
    replaced it would be testing the wrapper.
    """
    def build(xp, sub, sources, scratch=None):
        spec = cyl.SUB_STEPS[sub]
        source = sources[spec["prefix_component"]]
        ir0 = spec["prefix_ir0"]
        if mutation == "ir0_swapped":
            ir0 = 0.5 if sub == "step_B" else 0.0
        if mutation == "no_wall_row" and spec["extend_wall_row"]:
            # The recon's defect exactly: prefix the UNEXTENDED Ep, then terminate
            # the forward difference with `_shift_up`'s far-face zero. Rows 0..nr-1
            # are the unextended prefix and row nr is 0, so the kernel's load stays
            # in bounds and the defect is the missing wall row and nothing else.
            base = rderiv_prefix(xp, source, ir0)
            rows = base.shape[0]
            padded = xp.zeros((rows + 1,) + base.shape[1:], dtype=base.dtype)
            padded[:rows] = base
            return padded
        if spec["extend_wall_row"]:
            rows = source.shape[0]
            extended = xp.empty((rows + 1,) + source.shape[1:], dtype=source.dtype)
            extended[:rows] = source
            extended[rows] = 0
            return rderiv_prefix(xp, extended, ir0)
        return rderiv_prefix(xp, source, ir0)

    return build


def one_case(shape, dtdx, sub_step, guard, kernel=None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One launch of one sub-step against the reference, bytewise."""
    case: Dict[str, Any] = {
        "shape": list(shape), "dtdx": repr(dtdx), "sub_step": sub_step,
        "guard": guard, "host_mutation": host_mutation,
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "power_of_two_dtdx": bool(dtdx == 0.5),
    }
    arrays = seeded_arrays(tuple(shape))
    half_integer = sub_step == "step_B"
    if host_mutation == "coeff_lattice_swap":
        half_integer = not half_integer
    _tables, flat = coefficient_tables(tuple(shape), half_integer)
    reference_tables, _ = coefficient_tables(tuple(shape), sub_step == "step_B")

    reference = {name: array.copy() for name, array in arrays.items()}
    reference_cyl_step(cp, sub_step, reference, reference_tables, dtdx,
                       BOUNDARIES, 0, MIRROR_PHASE)

    plan = cyl.plan_cylindrical_curl_from_arrays(
        sub_step, arrays, flat, float(dtdx), cp, kernel=kernel)
    shipped_prefix = cyl.cylindrical_prefix
    if host_mutation in ("ir0_swapped", "no_wall_row"):
        cyl.cylindrical_prefix = mutated_prefix_builder(sub_step, host_mutation)
    try:
        plan.run(guard=guard)
        cp.cuda.runtime.deviceSynchronize()
    finally:
        cyl.cylindrical_prefix = shipped_prefix

    case["comparison"] = combine({name: bit_compare(arrays[name], reference[name])
                                  for name in _names(sub_step)})
    return case


def run_synthetic(results: Optional[Dict[str, Any]], out_path: Optional[str],
                  kernel=None, host_mutation: Optional[str] = None,
                  label: str = "synthetic",
                  guards: Sequence[bool] = (False, True),
                  shapes: Sequence[Tuple[int, int, int]] = SHAPES) -> Dict[str, Any]:
    """The sweep. ``results``/``out_path`` may be None — the mutation legs pass None
    so a sub-run cannot overwrite the artifact with its own partial record."""
    cases: List[Dict[str, Any]] = []
    combos = [(shape, dtdx, sub_step) for shape in shapes for dtdx in DTDX
              for sub_step in SUB_STEPS]
    total = len(guards) * len(combos)
    index = 0
    started = time.time()
    summary: Dict[str, Any] = {}
    for guard in guards:
        for shape, dtdx, sub_step in combos:
            index += 1
            case = one_case(shape, dtdx, sub_step, guard, kernel=kernel,
                            host_mutation=host_mutation)
            cases.append(case)
            log(f"  {label} {index}/{total} shape={tuple(shape)} dtdx={dtdx:.6g} "
                f"{sub_step} guard={guard} "
                f"identical={case['comparison']['bit_identical']} "
                f"n_diff={case['comparison']['differing_floats']} "
                f"({time.time() - started:.1f}s)")
            summary = summarize(cases)
            if results is not None and out_path is not None:
                results[label] = summary
                save(results, out_path)
    return summary


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarded and unguarded counted SEPARATELY, and unguarded split by Courant.

    The unguarded control is only expected to differ where the arithmetic can:
    at dtdx = 0.5 the scaling is exact in binary, so an unguarded row that comes back
    identical there is information about the Courant number, not about the guard.
    """
    guarded = [c for c in cases if c["guard"] is False and "comparison" in c]
    unguarded = [c for c in cases if c["guard"] is True and "comparison" in c]
    nptwo = [c for c in unguarded if not c["power_of_two_dtdx"]]
    ident = lambda rows: sum(bool(c["comparison"]["bit_identical"]) for c in rows)
    return {
        "cases": list(cases),
        "guarded_total": len(guarded),
        "guarded_identical": ident(guarded),
        "unguarded_total": len(unguarded),
        "unguarded_identical": ident(unguarded),
        "unguarded_non_power_of_two_total": len(nptwo),
        "unguarded_non_power_of_two_identical": ident(nptwo),
    }


# ===========================================================================
# Leg 2 — the multi-step budget on synthetic arrays
# ===========================================================================

def run_multi_step(results: Dict[str, Any], out_path: str, steps: int,
                   kernel=None) -> Dict[str, Any]:
    """B and D alternating from one state for ``steps`` steps, compared every step.

    The auxiliary is state. A kernel that gets ``field`` right and ``fu`` wrong is
    correct for exactly one launch and wrong forever after, and only consecutive
    sub-steps turn that into a visible divergence.
    """
    rows: List[Dict[str, Any]] = []
    for shape in SHAPES[:2]:
        for dtdx in (0.3141592653589793, 0.5):
            rows.append(one_multi_step(shape, dtdx, steps, kernel))
            log(f"  multi shape={tuple(shape)} dtdx={dtdx:.6g} steps={steps} "
                f"first_divergence={rows[-1]['first_divergence']}")
            results["multi"] = {"rows": rows, "steps": steps}
            save(results, out_path)
    results["multi"] = {
        "rows": rows, "steps": steps,
        "identical_rows": sum(r["first_divergence"] is None for r in rows),
        "total_rows": len(rows),
    }
    save(results, out_path)
    return results["multi"]


def one_multi_step(shape, dtdx, steps: int, kernel=None) -> Dict[str, Any]:
    arrays = seeded_arrays(tuple(shape), offset=7)
    reference = {name: array.copy() for name, array in arrays.items()}
    tables = {sub: coefficient_tables(tuple(shape), sub == "step_B")
              for sub in SUB_STEPS}
    first: Optional[Dict[str, Any]] = None
    started = time.time()
    for step_index in range(steps):
        for sub_step in SUB_STEPS:
            plan = cyl.plan_cylindrical_curl_from_arrays(
                sub_step, arrays, tables[sub_step][1], float(dtdx), cp, kernel=kernel)
            plan.run()
            reference_cyl_step(cp, sub_step, reference, tables[sub_step][0], dtdx,
                               BOUNDARIES, 0, MIRROR_PHASE)
        if first is None:
            cp.cuda.runtime.deviceSynchronize()
            first = divergence_record(reference, arrays, step_index)
        if steps >= 100 and (step_index + 1) % 100 == 0:
            log(f"    multi {tuple(shape)} dtdx={dtdx:.6g} step {step_index + 1}/{steps} "
                f"first_divergence={first} ({time.time() - started:.1f}s)")
    cp.cuda.runtime.deviceSynchronize()
    return {"shape": list(shape), "dtdx": repr(dtdx), "steps": steps,
            "first_divergence": first,
            "final": combine({name: bit_compare(arrays[name], reference[name])
                              for name in FIELD_NAMES})}


# ===========================================================================
# Leg 3 — the real engine: predicate, prefix, scratch and kernel together
# ===========================================================================

def build_engine(shape: Tuple[int, int, int], courant: float):
    """A real cylindrical ``Grid``/``Fields``/``PML`` on CuPy, at a chosen shape.

    ``resolution = 1`` with a cell size equal to the cell count lands on the shape
    exactly (``meep_cell_count`` is ``int(size*a + 0.5)``). z is METALLIC because a
    lifted ``mp.Simulation`` with no ``k_point`` runs ``use_bloch = false``; ``Grid``'s
    own default is periodic, so the declaration is explicit here.
    """
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=0, boundaries={"z": "metallic"},
                courant=float(courant), xp=cp)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    return grid, fields, PML(grid=grid, thickness=thickness)


def run_engine(results: Dict[str, Any], out_path: str, steps: int,
               kernel=None) -> Dict[str, Any]:
    """``stepping.step_B``/``step_D`` against the shipped plan, on a real Dcyl grid.

    Two independent engines are built and seeded identically; one is stepped by the
    array path and the other by :func:`cylindrical_triton.plan_cylindrical_curl` —
    the ENGINE route, so the shipped coverage predicate has to admit the run, the
    shipped prefix has to be built from the engine's own ``StepScratch``, and the
    shipped sub-lattice binding has to be right. Bytes compared after every sub-step.
    """
    rows: List[Dict[str, Any]] = []
    for shape in ((20, 1, 40), (13, 1, 9)):
        for courant in (0.3141592653589793, 0.5):
            rows.append(one_engine_case(shape, courant, steps, kernel))
            log(f"  engine shape={tuple(shape)} courant={courant:.6g} steps={steps} "
                f"covered={rows[-1]['covered']} "
                f"first_divergence={rows[-1]['first_divergence']}")
            results["engine"] = {"rows": rows, "steps": steps}
            save(results, out_path)
    results["engine"] = {
        "rows": rows, "steps": steps,
        "identical_rows": sum(r["first_divergence"] is None and r["covered"]
                              for r in rows),
        "total_rows": len(rows),
    }
    save(results, out_path)
    return results["engine"]


def one_engine_case(shape, courant, steps: int, kernel=None) -> Dict[str, Any]:
    grid_a, fields_a, pml_a = build_engine(shape, courant)     # array path
    grid_b, fields_b, pml_b = build_engine(shape, courant)     # Triton path

    verdict = cyl.cylindrical_curl_coverage(fields_b, pml_b)
    row: Dict[str, Any] = {
        "shape": list(shape), "courant": repr(courant), "steps": steps,
        "covered": bool(verdict.covered), "reasons": list(verdict.reasons),
        "boundaries": list(stepping._boundary_kinds(grid_b, pml_b)),
        "dtdx": repr(float(grid_b.dt / grid_b.dx)),
        "first_divergence": None,
    }
    if not verdict.covered:
        return row

    rng = np.random.default_rng(SEED + 21)
    for name in FIELD_NAMES:
        values = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=tuple(shape)).astype(np.float32))
        getattr(fields_a, name)[...] = cp.asarray(values)
        getattr(fields_b, name)[...] = cp.asarray(values)

    plans = {sub: cyl.plan_cylindrical_curl(fields_b, pml_b, sub)
             for sub in SUB_STEPS}
    if any(plan is None for plan in plans.values()):
        row["first_divergence"] = {"step": -1, "components": {"plan": "None"}}
        return row
    if kernel is not None:
        for plan in plans.values():
            plan._kernel = kernel

    started = time.time()
    for step_index in range(steps):
        for sub_step in SUB_STEPS:
            getattr(stepping, sub_step)(fields_a, pml_a)
            plans[sub_step].run()
        if row["first_divergence"] is None:
            cp.cuda.runtime.deviceSynchronize()
            row["first_divergence"] = divergence_record(
                {name: getattr(fields_a, name) for name in FIELD_NAMES},
                {name: getattr(fields_b, name) for name in FIELD_NAMES},
                step_index)
        if steps >= 100 and (step_index + 1) % 100 == 0:
            log(f"    engine {tuple(shape)} c={courant:.6g} step {step_index + 1}/{steps} "
                f"first_divergence={row['first_divergence']} "
                f"({time.time() - started:.1f}s)")
    cp.cuda.runtime.deviceSynchronize()
    row["final"] = combine({name: bit_compare(getattr(fields_a, name),
                                              getattr(fields_b, name))
                            for name in FIELD_NAMES})
    return row


# ===========================================================================
# Leg 3b — the NULL CONTROLS, without which a long-run divergence means nothing
# ===========================================================================
#
# A multi-step leg that diverges at step 40-ish has TWO possible causes and they look
# identical in a count: a defect in the kernel, or the amplifying float32
# disagreement plan §16 measured (a 1-ULP or subnormal seed growing at 0.220
# decades/step, which reaches the same plateau when the ARRAY PATH is flipped against
# ITSELF and Triton is nowhere). Neither the step number nor the component name tells
# them apart. These two controls do:
#
# * ``null``   — two array-path engines, identically seeded, stepped by
#   ``stepping.step_B``/``step_D`` and by nothing else. It answers "is the array path
#   even deterministic, and is the harness's own comparison quiet?" and it MUST be
#   identical for the whole budget.
# * ``one_ulp`` — the array path against itself with the low mantissa bit of ONE
#   NORMAL float flipped after the first step. It answers "does THIS configuration
#   amplify a last-bit disagreement at all?" — if it diverges the way the Triton leg
#   does, then no implementation that is not bit-exact could stay close, and the
#   kernel is not the reason.

def run_null_controls(results: Dict[str, Any], out_path: str,
                      steps: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for shape in ((20, 1, 40), (13, 1, 9)):
        for courant in (0.3141592653589793, 0.5):
            for control in ("null", "one_ulp"):
                rows.append(one_null_case(shape, courant, steps, control))
                log(f"  control={control} shape={tuple(shape)} "
                    f"courant={courant:.6g} steps={steps} "
                    f"first_divergence_step="
                    f"{(rows[-1]['first_divergence'] or {}).get('step')}")
                results["controls"] = {"rows": rows, "steps": steps}
                save(results, out_path)
    nulls = [r for r in rows if r["control"] == "null"]
    ulps = [r for r in rows if r["control"] == "one_ulp"]
    results["controls"] = {
        "rows": rows, "steps": steps,
        "null_identical": sum(r["first_divergence"] is None for r in nulls),
        "null_total": len(nulls),
        "one_ulp_diverged": sum(r["first_divergence"] is not None for r in ulps),
        "one_ulp_total": len(ulps),
    }
    save(results, out_path)
    return results["controls"]


def one_null_case(shape, courant, steps: int, control: str) -> Dict[str, Any]:
    """Array path against array path — Triton nowhere in it."""
    _grid_a, fields_a, pml_a = build_engine(shape, courant)
    _grid_b, fields_b, pml_b = build_engine(shape, courant)
    rng = np.random.default_rng(SEED + 21)
    for name in FIELD_NAMES:
        values = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=tuple(shape)).astype(np.float32))
        getattr(fields_a, name)[...] = cp.asarray(values)
        getattr(fields_b, name)[...] = cp.asarray(values)

    row: Dict[str, Any] = {"control": control, "shape": list(shape),
                           "courant": repr(courant), "steps": steps,
                           "first_divergence": None, "seed_flip": None}
    for step_index in range(steps):
        for sub_step in SUB_STEPS:
            getattr(stepping, sub_step)(fields_a, pml_a)
            getattr(stepping, sub_step)(fields_b, pml_b)
        if control == "one_ulp" and step_index == 0 and row["seed_flip"] is None:
            # Flip the low mantissa bit of one NORMAL float, §16.3's middle row.
            flat = fields_b.fu_Bz.reshape(-1)
            host = probe.to_host(flat).view(np.uint32)
            exponents = (host >> 23) & 0xFF
            normal = np.flatnonzero((exponents != 0) & (exponents != 0xFF))
            if normal.size:
                index = int(normal[normal.size // 2])
                flipped = np.uint32(host[index] ^ np.uint32(1))
                flat[index] = cp.asarray(
                    np.array([flipped], dtype=np.uint32).view(np.float32))[0]
                row["seed_flip"] = {"component": "fu_Bz", "flat_index": index,
                                    "before": f"0x{int(host[index]):08x}",
                                    "after": f"0x{int(flipped):08x}"}
            continue
        if row["first_divergence"] is None:
            cp.cuda.runtime.deviceSynchronize()
            row["first_divergence"] = divergence_record(
                {name: getattr(fields_a, name) for name in FIELD_NAMES},
                {name: getattr(fields_b, name) for name in FIELD_NAMES},
                step_index)
    cp.cuda.runtime.deviceSynchronize()
    row["final"] = combine({name: bit_compare(getattr(fields_a, name),
                                              getattr(fields_b, name))
                            for name in FIELD_NAMES})
    return row


# ===========================================================================
# Leg 4 — the constitutive claim, MEASURED
# ===========================================================================

def run_constitutive(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Is ``kernels.constitutive_step`` already right on a Dcyl grid?

    Finding E ARGUED it is — ``stepping.update_H``/``update_E`` carry no cylindrical
    branch, they are element-wise with per-axis coefficient tables indexed on the
    component's own axis, and cylindrical changes nothing there except n_phi = 1. An
    argument from absence is exactly what this project's rules refuse to ship, so it
    is measured here: the shipped constitutive plan against the shipped array-path
    sub-step, on a real cylindrical grid, bytewise.
    """
    rows: List[Dict[str, Any]] = []
    for shape in ((20, 1, 40), (13, 1, 9)):
        for side in ("H", "E"):
            rows.append(one_constitutive_case(shape, side))
            log(f"  constitutive shape={tuple(shape)} side={side} "
                f"covered={rows[-1]['covered']} identical={rows[-1]['identical']}")
            results["constitutive"] = {"rows": rows}
            save(results, out_path)
    results["constitutive"] = {
        "rows": rows,
        "identical": sum(bool(r["identical"]) for r in rows),
        "total": len(rows),
    }
    save(results, out_path)
    return results["constitutive"]


def one_constitutive_case(shape, side: str) -> Dict[str, Any]:
    grid_a, fields_a, pml_a = build_engine(shape, 0.3141592653589793)
    grid_b, fields_b, pml_b = build_engine(shape, 0.3141592653589793)
    verdict = cyl.cylindrical_constitutive_coverage(fields_b, pml_b, side)
    row: Dict[str, Any] = {"shape": list(shape), "side": side,
                           "covered": bool(verdict.covered),
                           "reasons": list(verdict.reasons), "identical": None}
    if not verdict.covered:
        return row

    spec = {"H": (("Hx", "Hy", "Hz"), ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
                  ("Bx", "By", "Bz"), False),
            "E": (("Ex", "Ey", "Ez"), ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
                  ("Dx", "Dy", "Dz"), True)}[side]
    targets, aux, sources, half_integer = spec

    rng = np.random.default_rng(SEED + 33)
    names = tuple(targets) + tuple(aux) + tuple(sources)
    for name in names:
        values = np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=tuple(shape)).astype(np.float32))
        getattr(fields_a, name)[...] = cp.asarray(values)
        getattr(fields_b, name)[...] = cp.asarray(values)

    getattr(stepping, "update_H" if side == "H" else "update_E")(fields_a, pml_a)

    arrays = {name: getattr(fields_b, name) for name in names}
    if side == "E":
        for index, component in enumerate(("Ex", "Ey", "Ez")):
            arrays["inv_eps_" + component] = fields_b.inverse_epsilon_for(component)
    suffix = "_h" if half_integer else ""
    flat = {f"{stem}_{axis}": getattr(pml_b, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}
    plan_constitutive_from_arrays(side, arrays, flat).run()
    cp.cuda.runtime.deviceSynchronize()

    comparison = combine({name: bit_compare(getattr(fields_a, name),
                                            getattr(fields_b, name))
                          for name in tuple(targets) + tuple(aux)})
    row["identical"] = bool(comparison["bit_identical"])
    row["comparison"] = comparison
    return row


# ===========================================================================
# Leg 5 — the mutations
# ===========================================================================
#
# NINE defects, the ones the recon established against the reference, re-aimed at
# the SHIPPED artifact: five at the kernel's own source, three at the host side (the
# prefix's ir0, the wall row, the coefficient sub-lattice), and one — ``axis_ghost_sign``
# — whose expected verdict is NOT CAUGHT.
#
# ``axis_ghost_sign`` IS THE POINT OF INCLUDING IT. The kernel deliberately omits the
# r_to_minus_r near ghost because the ghost is unobservable: the only terms taking a
# shift-down along r are Dp and Dz, both of whose curls the ownership mask zeroes at
# row 0 — the only row the ghost writes. Measured at m = 0, 1 AND 2 over 20 full
# step pairs: inverting that whole plane changed nothing. So the mutation that PUTS
# THE GHOST BACK must come back UNCAUGHT, and a run reporting it caught would mean
# the mask had been broken, not the ghost fixed. Dropping it quietly would leave the
# module's stated design decision looking tested when it is not.

KERNEL_SOURCE = textwrap.dedent(inspect.getsource(cyl.cyl_pml_curl_step.fn))

_B_PREFIX_CURL = re.compile(
    r"curl2 = dtdx \* \(tl\.load\(pfx \+ idx \+ nyz, mask=live, other=0\.0\)\n"
    r"(?P<pad>\s+)- tl\.load\(pfx \+ idx, mask=live, other=0\.0\)\)")
_AXIS_MASK_R = re.compile(r"curl(?P<t>\d) = tl\.where\(at_r, 0\.0, curl(?P=t)\)")
_DZ_AXIS_ADD = re.compile(
    r"v2 = tl\.where\(at_r, v2 \+ four_dtdx \* tl\.load\(hp \+ idx, mask=live, other=0\.0\),\n"
    r"(?P<pad>\s+)v2\)")
_AXIS_ZERO = re.compile(r"v(?P<t>[01]) = tl\.where\(at_r, 0\.0, v(?P=t)\)")
# Anchored with the indentation CAPTURED, not spelled: ``inspect.getsource`` returns
# the kernel at its in-file indent and ``textwrap.dedent`` removes only the common
# prefix, so a hard-coded column is a needle that silently matches nothing the first
# time the enclosing block moves.
_LOAD_BLOCK = re.compile(
    r"(?P<ind>[ \t]*)c_p = tl\.load\(g2 \+ o_p, mask=vp, other=0\.0\)\n")
_D_PREFIX_LOAD = re.compile(
    r"(?P<ind>[ \t]*)p_down = tl\.load\(pfx \+ o_r, mask=vr, other=0\.0\)\n")


def _bz_generic_curl(source: str) -> Tuple[str, int]:
    """Bz keeps the four-operand curl of Ey/Ex instead of the prefix difference.

    The whole cylindrical B-side substitution, deleted. Recon: 1560 Bz floats at
    step 0.
    """
    return _B_PREFIX_CURL.subn("curl2 = curl2", source)


def _bz_flat_grouping(source: str) -> Tuple[str, int]:
    """``dtdx*a - dtdx*b`` in place of ``dtdx*(a - b)``.

    The prefix difference cannot be distributed in float32. Recon: 211 Bz floats.
    """
    return _B_PREFIX_CURL.subn(
        lambda m: ("curl2 = (dtdx * tl.load(pfx + idx + nyz, mask=live, other=0.0)\n"
                   f"{m['pad']}- dtdx * tl.load(pfx + idx, mask=live, other=0.0))"),
        source)


def _no_axis_mask(source: str) -> Tuple[str, int]:
    """The is_axis clause of ``_mask_non_owned_cells`` dropped; the z clause kept.

    Rewritten to an identity rather than deleted: the statements are the whole body
    of a ``constexpr`` ``if`` and removing them is a syntax error — a mutation that
    fails to compile measures nothing. Recon: 40 fu_Bx + 39 fu_Dy.
    """
    return _AXIS_MASK_R.subn(lambda m: f"curl{m['t']} = curl{m['t']}", source)


def _no_dz_axis_add(source: str) -> Tuple[str, int]:
    """The m = 0 on-axis ``Dz += 4*Courant*Hp`` dropped. Recon: 40 Dz floats."""
    return _DZ_AXIS_ADD.subn(lambda m: "v2 = v2", source)


def _no_axis_zero(source: str) -> Tuple[str, int]:
    """``Br[0] = 0`` and ``Dp[0] = 0`` dropped. Recon: 40 Bx + 40 Dy."""
    return _AXIS_ZERO.subn(lambda m: f"v{m['t']} = v{m['t']}", source)


def _axis_ghost_sign(source: str) -> Tuple[str, int]:
    """PUT THE r_to_minus_r NEAR GHOST BACK. Expected verdict: NOT CAUGHT.

    ``_shift_down``'s CYL_AXIS branch (stepping.py:1883-1893) images stored row 0
    with a sign of -1 for r- and phi-direction components and +1 for z. The kernel
    serves an exact 0.0 there instead, because the row that consumes it is masked.
    This mutation implements the branch faithfully; if the gate catches it, the MASK
    is broken, not the ghost.
    """
    def add_ghost(match: "re.Match[str]") -> str:
        pad = match["ind"]
        return (match.group(0)
                + f"{pad}ghost_r = live & (si < 0)\n"
                + f"{pad}if BACKWARD:\n"
                + f"{pad}    b_r = tl.where(ghost_r, -b, b_r)\n"
                + f"{pad}    c_r = tl.where(ghost_r, c, c_r)\n")

    def add_prefix_ghost(match: "re.Match[str]") -> str:
        return (match.group(0)
                + f"{match['ind']}p_down = tl.where(ghost_r, -p_here, p_down)\n")

    out, first = _LOAD_BLOCK.subn(add_ghost, source, count=1)
    out, second = _D_PREFIX_LOAD.subn(add_prefix_ghost, out, count=1)
    return out, first + second


#: name -> (transform | None for a host mutation, expected verdict)
MUTATIONS: Tuple[Tuple[str, Optional[Callable[[str], Tuple[str, int]]], bool], ...] = (
    ("bz_generic_curl", _bz_generic_curl, True),
    ("bz_flat_grouping", _bz_flat_grouping, True),
    ("no_axis_mask", _no_axis_mask, True),
    ("no_dz_axis_add", _no_dz_axis_add, True),
    ("no_axis_zero", _no_axis_zero, True),
    ("axis_ghost_sign", _axis_ghost_sign, False),   # DEAD CODE — must NOT be caught
    ("no_wall_row", None, True),
    ("ir0_swapped", None, True),
    ("coeff_lattice_swap", None, True),
)

_TEMPORARY: List[str] = []


def compile_mutated(source: str) -> Any:
    """Compile a mutated copy of the shipped kernel from a REAL FILE.

    Triton reads a kernel's text with ``inspect.getsource``, so the mutated function
    has to live on disk; an ``exec``-ed one raises ``OSError: could not get source
    code`` at first launch.
    """
    header = "import triton\nimport triton.language as tl\n\n"
    # encoding= is not decoration: the kernel's prose carries em dashes, and the
    # default locale on the measurement host is ASCII, which turned the whole
    # mutation leg into a UnicodeEncodeError at the first needle.
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_cyl.py", delete=False,
                                         encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    name = "triton_mutated_cyl_" + str(len(_TEMPORARY))
    spec = importlib.util.spec_from_file_location(name, handle.name)
    module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)                     # type: ignore[union-attr]
    return module.cyl_pml_curl_step


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Each defect in turn, over a narrowed sweep. Every one must land as predicted."""
    rows: List[Dict[str, Any]] = []
    shapes = SHAPES[:2]
    for name, transform, expect_caught in MUTATIONS:
        row: Dict[str, Any] = {"mutation": name, "expected_caught": expect_caught,
                               "sites": None}
        kernel = None
        host_mutation = None
        if transform is not None:
            mutated, sites = transform(KERNEL_SOURCE)
            row["sites"] = int(sites)
            if sites == 0:
                row["error"] = ("the needle matched nothing — the mutation has "
                                "drifted from the kernel and is exercising nothing")
                rows.append(row)
                log(f"  mutation {name}: NEEDLE MATCHED NOTHING")
                results["mutations"] = {"rows": rows}
                save(results, out_path)
                continue
            if mutated == KERNEL_SOURCE:
                row["error"] = "the transform returned the shipped source unchanged"
                rows.append(row)
                results["mutations"] = {"rows": rows}
                save(results, out_path)
                continue
            kernel = compile_mutated(mutated)
        else:
            host_mutation = name

        summary = run_synthetic(None, None, kernel=kernel,
                                host_mutation=host_mutation,
                                label=f"mutation:{name}", guards=(False,),
                                shapes=shapes)
        caught = summary["guarded_identical"] < summary["guarded_total"]
        row.update({
            "guarded_total": summary["guarded_total"],
            "guarded_identical": summary["guarded_identical"],
            "caught": bool(caught),
            "as_expected": bool(caught == expect_caught),
        })
        rows.append(row)
        log(f"  mutation {name}: caught={caught} expected={expect_caught} "
            f"({summary['guarded_identical']}/{summary['guarded_total']} identical)")
        results["mutations"] = {"rows": rows}
        save(results, out_path)

    results["mutations"] = {
        "rows": rows,
        "as_expected": sum(bool(r.get("as_expected")) for r in rows),
        "total": len(rows),
        "caught": sum(bool(r.get("caught")) for r in rows),
        "dead_by_construction": [r["mutation"] for r in rows
                                 if r["expected_caught"] is False],
    }
    save(results, out_path)
    return results["mutations"]


def run_clean_control(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """The shipped kernel over the same narrowed sweep the mutations use.

    Without it a mutation leg that reports "caught" everywhere is indistinguishable
    from a harness that reports "caught" unconditionally.
    """
    summary = run_synthetic(None, None, label="clean_control",
                            guards=(False,), shapes=SHAPES[:2])
    results["clean_control"] = {
        "guarded_total": summary["guarded_total"],
        "guarded_identical": summary["guarded_identical"],
        "false_positive": summary["guarded_identical"] != summary["guarded_total"],
    }
    save(results, out_path)
    log(f"  clean control: {summary['guarded_identical']}/{summary['guarded_total']} "
        f"identical")
    return results["clean_control"]


# ===========================================================================
# Leg 6 — throughput
# ===========================================================================

def run_bench(results: Dict[str, Any], out_path: str, repeats: int = 50,
              warmup: int = 10) -> Dict[str, Any]:
    """The kernel's curl pair against the array path's, on the benchmark case's shape.

    Timed with a device synchronize around the whole repeat block, not per launch:
    the case is LAUNCH-BOUND, so a per-launch synchronize would measure the
    synchronize. The prefix is INSIDE both legs, because it is inside both paths.
    """
    rows: List[Dict[str, Any]] = []
    for shape in (BENCH_SHAPE, (640, 1, 640)):
        rows.append(one_bench(shape, repeats, warmup))
        log(f"  bench shape={tuple(shape)} array={rows[-1]['array_ms']:.3f} ms "
            f"triton={rows[-1]['triton_ms']:.3f} ms "
            f"speedup={rows[-1]['speedup']:.2f}x")
        results["bench"] = {"rows": rows}
        save(results, out_path)
    return results["bench"]


def one_bench(shape, repeats: int, warmup: int) -> Dict[str, Any]:
    grid, fields, pml = build_engine(tuple(shape), 0.5)
    rng = np.random.default_rng(SEED + 41)
    for name in FIELD_NAMES:
        getattr(fields, name)[...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=tuple(shape)).astype(np.float32)))
    plans = {sub: cyl.plan_cylindrical_curl(fields, pml, sub) for sub in SUB_STEPS}

    def array_leg():
        for sub in SUB_STEPS:
            getattr(stepping, sub)(fields, pml)

    def triton_leg():
        for sub in SUB_STEPS:
            plans[sub].run()

    timings = {}
    for label, leg in (("array", array_leg), ("triton", triton_leg)):
        for _ in range(warmup):
            leg()
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            leg()
        cp.cuda.runtime.deviceSynchronize()
        timings[label] = (time.perf_counter() - started) / repeats * 1e3

    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    return {"shape": list(shape), "cells": cells, "repeats": repeats,
            "array_ms": timings["array"], "triton_ms": timings["triton"],
            "speedup": timings["array"] / timings["triton"],
            "array_mcells_s": cells / timings["array"] / 1e3,
            "triton_mcells_s": cells / timings["triton"] / 1e3}


# ===========================================================================
# Artifact
# ===========================================================================

def save(results: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
        json.dump(results, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())


def environment() -> Dict[str, Any]:
    """Every version that decides this gate's answer — CuPy above all.

    Everything downstream of the radial prefix is bit-identical only for a given
    ``cupy.cumsum`` summation order, so a CuPy bump is a correctness event for this
    kernel and the version has to be IN the artifact rather than in a log.
    """
    import triton  # noqa: PLC0415

    out: Dict[str, Any] = {
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "block": cyl.DEFAULT_BLOCK,
        "enable_fp_fusion_in_source": _kernels.ENABLE_FP_FUSION,
    }
    try:
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        out["device"] = device["name"].decode()
        out["compute_capability"] = f"{device['major']}.{device['minor']}"
    except Exception as exc:  # noqa: BLE001
        out["device"] = f"unavailable: {exc}"
    try:
        out["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        out["nvidia_smi"] = f"unavailable: {exc}"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="the step budget for the multi and engine legs")
    parser.add_argument("--legs", default="synthetic,multi,engine,controls,"
                                          "constitutive,clean,mutations,bench")
    args = parser.parse_args(argv)
    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]

    results: Dict[str, Any] = {
        "what": "bit-identity gate for the cylindrical (Dcyl, m = 0) Triton curl kernel",
        "environment": environment(),
        "step_budget": args.steps,
        "step_budget_note": (
            "400 steps is the NumPy budget the transcription was certified to; this "
            "run's number is what may be claimed for the kernel. A budget can "
            "certify something that diverges just past it (2d_pml: 60/60 identical, "
            "first differs at step 67)."),
        "boundaries": list(BOUNDARIES),
        "m": 0,
        "legs": legs,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(results, args.out)
    log(f"gate_triton_cylindrical -> {args.out}")
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
    log(f"  environment: {json.dumps(results['environment'], default=str)}")

    if "synthetic" in legs:
        log("leg: synthetic")
        run_synthetic(results, args.out)
    if "multi" in legs:
        log("leg: multi")
        run_multi_step(results, args.out, args.steps)
    if "engine" in legs:
        log("leg: engine")
        run_engine(results, args.out, args.steps)
    if "controls" in legs:
        log("leg: null controls (array path against itself, Triton nowhere)")
        run_null_controls(results, args.out, args.steps)
    if "constitutive" in legs:
        log("leg: constitutive")
        run_constitutive(results, args.out)
    if "clean" in legs:
        log("leg: clean control")
        run_clean_control(results, args.out)
    if "mutations" in legs:
        log("leg: mutations")
        run_mutations(results, args.out)
    if "bench" in legs:
        log("leg: bench")
        run_bench(results, args.out)

    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["verdict"] = verdict(results)
    save(results, args.out)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
    log(f"VERDICT: {json.dumps(results['verdict'], default=str)}")
    return 0 if results["verdict"]["pass"] else 1


def verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """The gate's own reading of its rows. Every clause is stated, none inferred."""
    out: Dict[str, Any] = {"clauses": {}}
    synthetic = results.get("synthetic")
    if synthetic:
        out["clauses"]["guarded_all_identical"] = (
            synthetic["guarded_identical"] == synthetic["guarded_total"]
            and synthetic["guarded_total"] > 0)
        out["clauses"]["unguarded_control_bites"] = (
            synthetic["unguarded_non_power_of_two_identical"] == 0
            and synthetic["unguarded_non_power_of_two_total"] > 0)
        out["guarded"] = f"{synthetic['guarded_identical']}/{synthetic['guarded_total']}"
        out["unguarded_non_power_of_two"] = (
            f"{synthetic['unguarded_non_power_of_two_identical']}/"
            f"{synthetic['unguarded_non_power_of_two_total']}")
    # THE LONG-RUN LEGS ARE NOT A PASS/FAIL CLAUSE, and that is deliberate. Plan §16
    # established that every covered configuration stops being byte-equal at some step
    # in the 20-70 neighbourhood because a subnormal or 1-ULP seed AMPLIFIES at 0.220
    # decades/step — and that the array path reaches the same plateau when flipped
    # against ITSELF with Triton nowhere. So a long-run divergence is only evidence
    # about the kernel if the NULL control is clean and the ONE-ULP control is not.
    # What the long-run legs report is therefore a MEASURED BUDGET, not a verdict.
    for leg in ("multi", "engine"):
        record = results.get(leg)
        if not record:
            continue
        steps = [(r["first_divergence"] or {}).get("step") for r in record["rows"]]
        reached = [s for s in steps if s is not None]
        out[f"{leg}_first_divergence_steps"] = steps
        out[f"{leg}_certified_steps"] = (min(reached) if reached
                                         else record.get("steps"))
    controls = results.get("controls")
    if controls:
        out["clauses"]["null_control_is_clean"] = (
            controls["null_identical"] == controls["null_total"]
            and controls["null_total"] > 0)
        out["clauses"]["one_ulp_control_amplifies"] = (
            controls["one_ulp_diverged"] == controls["one_ulp_total"]
            and controls["one_ulp_total"] > 0)
    constitutive = results.get("constitutive")
    if constitutive:
        out["clauses"]["constitutive_identical"] = (
            constitutive["identical"] == constitutive["total"]
            and constitutive["total"] > 0)
    control = results.get("clean_control")
    if control:
        out["clauses"]["clean_control_no_false_positive"] = not control["false_positive"]
    mutations = results.get("mutations")
    if mutations:
        out["clauses"]["mutations_all_as_expected"] = (
            mutations["as_expected"] == mutations["total"]
            and mutations["total"] > 0)
        out["mutations"] = f"{mutations['as_expected']}/{mutations['total']} as expected"
    out["pass"] = bool(out["clauses"]) and all(out["clauses"].values())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
