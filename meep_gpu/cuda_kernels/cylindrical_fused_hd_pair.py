"""The COMPLEX Dcyl H->D product, any m: ``update_H`` + the radial increment in one
launch, ``xp.cumsum`` untouched on the array path, the certified complex cylindrical
``step_D`` curl in the second launch. THREE launches per seam where the D side takes
seven today.

THE CELL: ``H_to_D (cuda_complex/complex, cuda_cyl_complex/cylindrical complex)`` --
18 seam-instances on the standing census (``cuda_predicate_coverage_2026-09-06_hd``):
16 ``buildable_not_built`` (the examples ``dipole_in_vacuum_cyl_{off,on}_axis``,
``disc_extraction_efficiency``, ``disc_radiation_pattern``, ``extraction_eff_ldos``,
``perturbation_theory``, ``planar_cavity_ldos``, ``point_dipole_cyl``, ``ring-cyl``; the
tests ``TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_2_1``, ``TestLDOS.test_ldos_cyl``,
``TestLDOS.test_ldos_ext_eff``, ``TestPMLCylindrical.test_pml_cyl_{1_1_0,2_2_0,3_3_0}``,
``TestRingCyl.test_ring_cyl``) and 2 ``withdraw_seam`` (``cylinder_cross_section.py``,
``zone_plate.py``), which this product REFUSES BY NAME exactly as :mod:`.fused_hd_pair`
refuses its five. ``ny = 1`` on every row, so the scan unit is a radial column; three
rows have exactly one column.

The shape is the real sibling's (:mod:`.cylindrical_real_fused_hd_pair`, whose docstring
carries the seam argument and the launch algebra) over complex64 storage and every
``m_class``: launch 1 = the certified ``update_H_pml_complex_bloch`` body lifted PURE
(scratch output) at the thread's own cell, the same body RECOMPUTED at the one-cell
backward radial neighbour for the increment, no store of the neighbour; ``xp.cumsum``
untouched; launch 2 = the certified ``cyl_step_D_pml_complex`` through its own launcher
with the prefix handed in. The ``i*m/r`` couplings, the |m| = 1 fold, the m = 0 axis
post-add and the |m| >= 2 zeroing all live in launch 2 and read the ROTATED ``H``
pointwise, exactly where the array path reads ``fields.get_H`` after ``update_H``.

=============================================================================
THE TWO COMPLEX SPELLINGS ARE MEASURED, AND THEY INVERT THE NUMPY PRESCRIPTION
=============================================================================

``weights`` and ``divisor`` are float32 row vectors and ``f_p`` is complex64 here. CuPy's
``multiply`` and ``true_divide`` carry ``FF->F`` and no mixed ``Ff`` loop, so the float32
operand is cast to ``complex<float>(d)`` -- imaginary ``T()`` = +0.0f
(``cupy/_core/include/cupy/complex/complex_inl.h:28``) -- and the COMPLEX/COMPLEX
operators run: ``operator*`` at ``arithmetic.h:75-79`` (four products) and ``operator/``
at ``:96-110`` (the SCALED algorithm). Neither is NumPy's ``a * (float32(1)/d)``, which
is what the Metal backend's complex scan must spell because ITS oracle is NumPy.
Measured on device (RTX A6000, CuPy 13.5.1, both float32 subnormal policies, 13 corpus
radial extents x both ir0 x four value classes -- ``lanes/cyl_round/cupy_probe/FINDINGS.md``):

* **divide** = the scaled algorithm with ``rhs = (d, +0.0f)`` and EVERY zero-valued
  term LEFT IN (:func:`div_coefficient_source`): **0 of 14,387,104 complex64 words**
  per policy. NumPy's ``a * (float32(1)/d)`` misses 793,105 / 820,455 of them (keep /
  flush); the real family's componentwise ``/`` misses 3,584,917 / 3,591,312; the
  algorithm with its zero terms "tidied" away misses 15,598 / 60,038 on the edge class.
* **multiply** ``Hy * w`` = :mod:`.complex_emitter`'s **FMA_V1** ``mul_field_left``,
  verbatim, whatever EXPANSION arm the constitutive half runs under. NVRTC contracts
  CuPy's own ufunc body; the uncontracted four-product spelling differs on the SIGN OF A
  FLUSHED ZERO when ``Hy.re * 0.5`` underflows at row 0 (``ir0 = 0.5``) with
  ``Hy.re < 0``: 6 of 4,005,000 words under ``flush``, 0 with the fma spelling. The
  random batteries never exercise it; the gate PLANTS the row-0 case.
* row 0 = ``cf_zero()`` = (+0.0f, +0.0f) (the array path assigns the integer 0);
  the subtract is the certified componentwise ``cf_sub``.

Both spellings are carried as source text here AND as armed mutations in the gate,
because the real family's comment says "spell ``/``", the Metal verdict says "spell
``a*(1/d)``", and both are right on their own backend and wrong on this one.

ON THE DIVISIONS. The certified complex families perform no floating-point division
(``div.rn.f32``'s ptxas expansion carries ``.FTZ`` range checks). This launch performs
the two reciprocals CuPy's own header performs, in the same expressions, compiled by
the same NVRTC -- and that is the measurement above: 0 words against ``cupy.true_divide``
under BOTH policies, including the edge class of subnormals and tiny normals. It is
licensed by that number, not by the rule.

=============================================================================
ARM, LICENCE, INSTALLATION
=============================================================================

Every entry point takes ``expansion``; the arbiter is
``triton_kernels.complex_fields.expansion_license`` and the predicate asks
``coverage.complex_expansion_refusal(license, subnormal_policy)`` first, as every
complex family does. The increment's multiply does NOT vary with the arm (above).

:data:`HOISTS_THE_WITHDRAW` False, :data:`CARRIES_DEPOSIT_REPAIR` False,
:data:`INSTALLABLE` False for the reason the real sibling gives: the cylindrical launch
algebra favours this product (+2 launches per step per row against both neighbours)
and the composer's ``4 - pairs`` rule does not know it; the flip is a measured
composition gate plus a rule change, not a flag.

NOT WIRED, NOT DISPATCHED: not in ``fused_pairs``' tables until the wiring change
lands; ``fastpath.plan_fast_path`` never names this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import complex_emitter
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import covers_real_pml_complex_constitutive
from .cylindrical_coverage import covers_pml_cylindrical_complex_curl
from .cylindrical_prefix import PREFIX_COMPONENT, PREFIX_IR0
from . import cylindrical_complex_kernels as _cyl
from . import fused_hd_pair as _hd
from .. import withdraw_hoist as _withdraw_hoist

# The complex constitutive TABLE builder imports CuPy at scope; taken defensively so
# the predicate and the emitter answer on a host with no device.
try:
    from . import complex_pml_kernels
except Exception:  # noqa: BLE001 - no CuPy on this host
    complex_pml_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_cylindrical_fused_hd_pair"
KERNEL_NAME = "update_H_increment_pml_cyl_complex"
CURL_KERNEL_NAME = "cyl_step_D_pml_complex"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS
PREFIX_SOURCE: str = PREFIX_COMPONENT["step_D"]
PREFIX_IR0_VALUE: float = PREFIX_IR0["step_D"]
LAUNCHES_PER_RUN = 3

#: CERTIFIED BY ``parity/meep_gpu/gate_cuda_cylindrical_fused_hd_pair.py`` on device
#: under both float32 subnormal policies and both EXPANSION arms (campaign
#: ``cuda_cylindrical_fused_hd_pair_2026-09-07``). The declaration is the FINAL bytes
#: the campaign was cut against; the record block that backs it is written by the
#: record tools in the wiring change, and ``test_kernel_partition.py`` is red until then.
CERTIFIED_KERNELS = (
    "update_H_increment_pml_cyl_complex",
)
UNCERTIFIED_KERNELS = {}

CARRIES_DEPOSIT_REPAIR = False
HOISTS_THE_WITHDRAW = False
INSTALLABLE = False
INSTALLABLE_REASON = (
    "slot arbitration, measured on the Cartesian cell and NOT YET measured on this one. "
    "On a cylindrical row the D-side seam is 7 launches today and this product takes it "
    "to 3, saving 4, while each neighbouring cylindrical pair saves 1 -- displacing both "
    "is +2 launches per step per row in this product's favour -- but the composer's "
    "rule (_neighbouring_seam_claimant / _spans_may_absorb) counts launches as 4 - "
    "pairs and would refuse or lose the tie, and no composition gate has yet counted "
    "device launches per step on a lifted cylindrical row with this product installed "
    "against the two neighbours installed. INSTALLABLE stays False -- credited on the "
    "board as served by predicate admission, as fused_hd_pair is -- and the flip is a "
    "MEASURED step: that composition gate, then a per-family launches_saved weight in "
    "the arbitration, a rule change in fused_pairs.py")
WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration once wired, it is not in the composer's tables until the wiring "
    "change lands, and no timing of any kind has been taken of this shape. The launch "
    "figures in this module are COUNTS.")

_FUSED_THREADS = 256
#: ``--fmad=false`` is CORRECTNESS: the constitutive half's accumulations are
#: contraction candidates, and the FMA_V1 arm's own fusions are explicit ``__fmaf_rn``
#: so the flag removes the accidental ones and leaves the transcribed ones. On the
#: increment itself the probe measured the unguarded build at 0 words -- there is no
#: contractible pattern left once the multiply is an explicit fma and the divide's sums
#: were measured neutral both ways -- and the gate records that as a NON-BITING control
#: with this reason rather than dropping it.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "CURL_KERNEL_NAME", "FAMILY",
    "HOISTS_THE_WITHDRAW", "H_TARGETS", "INCREMENT_SPELLING", "INSTALLABLE",
    "INSTALLABLE_REASON", "KERNEL_NAME", "LAUNCHES_PER_RUN", "PREFIX_IR0_VALUE",
    "PREFIX_SOURCE", "REPLACES", "SCRATCH_VOLUMES", "SEAM", "SLOT",
    "UNCERTIFIED_KERNELS", "WHAT_A_RELEASE_DOES_NOT_LICENSE",
    "assert_bindings_are_disjoint", "constitutive_prelude",
    "covers_cylindrical_fused_hd_pair", "device_sources", "div_coefficient_source",
    "increment_source", "kernel_source", "launch_constitutive_and_increment",
    "launch_curl", "mul_field_left_increment_source", "raw_update_H_cell_source",
    "resolve", "run_cylindrical_fused_hd_pair", "scan_and_rotate", "signature",
    "source_digest",
]


# ---------------------------------------------------------------------------
# The declared edits and the measured spelling -- DATA
# ---------------------------------------------------------------------------

INCREMENT_SPELLING: Tuple[Dict[str, str], ...] = (
    {"line": "cf wi = mul_field_left_increment(own_h[1], weights[i]);",
     "transcribes": "xp.multiply(f_p, weights) -- CuPy's complex<float>*complex<float> "
                    "with rhs (w, +0.0f), as NVRTC contracts it: the FMA_V1 mul_field_left",
     "control": "the NAIVE four-product spelling: 6 of 4,005,000 words under flush on "
                "the planted row-0 case (sign of a flushed zero); 0 elsewhere"},
    {"line": "cf wim1 = mul_field_left_increment(halo_h[1], weights[i - 1]);",
     "transcribes": "the same multiply at row i - 1 on the RECOMPUTED Hy_new[i-1]",
     "control": "reading the stored pre-launch Hy[idx - nyz] instead: must be caught"},
    {"line": "cf diff = cf_sub(wi, wim1);",
     "transcribes": "xp.subtract(weighted[1:], weighted[:-1]) -- componentwise",
     "control": "(none: cf_sub is the certified componentwise subtract)"},
    {"line": "cf_store(increment, idx, cf_div_coefficient(diff, divisor[i - 1]));",
     "transcribes": "increment[1:] /= divisor -- CuPy's scaled complex/complex divide "
                    "with rhs (d, +0.0f), zero terms kept",
     "control": "numpy's z * (1.0f/d): 793,105 / 820,455 of 14,387,104; componentwise "
                "re/d, im/d: 3,584,917 / 3,591,312; zero terms tidied: 15,598 / 60,038"},
    {"line": "cf_store(increment, idx, cf_zero());",
     "transcribes": "increment[_face(0, 0)] = 0 -- the word pair (+0.0f, +0.0f)",
     "control": "a sign-carrying zero or a non-zero row 0 moves the whole prefix"},
)

#: Every line of certified CONSTITUTIVE text this product does not lift verbatim, with
#: the reason. No arithmetic is in this table.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void constitutive_apply(",
     "became": "__device__ __forceinline__ cf constitutive_apply_pure(",
     "why": "the launch writes SCRATCH, so the helper stores nothing: it returns the "
            "stepped word pair and hands the split-field value back through an "
            "out-parameter; f and fw become const."},
    {"line": "    float* f, float* fw, int idx, cf src, float kps, float kms\n)",
     "became": "    const float* f, const float* fw, int idx, cf src, float kps, "
               "float kms,\n    cf* fw_out\n)",
     "why": "the parameter list of the pure form; nothing else moves."},
    {"line": "    cf_store(fw, idx, src);",
     "became": "    *fw_out = src;",
     "why": "the store moves to the caller, which addresses the scratch for its own "
            "cell and stores nothing for the foreign recompute."},
    {"line": "    cf_store(f, idx, a);",
     "became": "    return a;",
     "why": "same value, returned instead of stored; the two separate accumulations "
            "above it are untouched."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;",
     "became": "(removed -- idx is a parameter of raw_update_H_cell)",
     "why": "the certified body becomes a __device__ function evaluated at an "
            "ARBITRARY cell; the decode below it STAYS, because it indexes the absorber "
            "profile at the recomputed cell."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);",
     "became": "    h_out[0] = constitutive_apply_pure(f0, w0, idx, s0, kps_x[i], "
               "kms_x[i], &w_out[0]);",
     "why": "capture both values; the argument list and the coefficient pairing are "
            "untouched. Three lines, one per component, derived by prefix and suffix."},
)

#: The one helper this family LIFTS out of the arm block and renames: the FMA_V1
#: ``mul_field_left``. Renamed because the increment must spell it under the NAIVE arm
#: too, and the NAIVE arm's prelude already defines ``mul_field_left`` the other way.
INCREMENT_MULTIPLY_SOURCE_ARM = "FMA_V1"

_BODY_ANCHOR = "\n) {\n"
_THREAD_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    "    if (idx >= nx * ny * nz) return;\n")
_INDEX_DECODE = ("    int k = idx % nz;\n"
                 "    int j = (idx / nz) % ny;\n"
                 "    int i = idx / (ny * nz);\n")
_APPLY_SIGNATURE = "__device__ __forceinline__ void constitutive_apply(\n"
_APPLY_SIGNATURE_PURE = "__device__ __forceinline__ cf constitutive_apply_pure(\n"
_APPLY_PARAMETERS = "    float* f, float* fw, int idx, cf src, float kps, float kms\n)"
_APPLY_PARAMETERS_PURE = ("    const float* f, const float* fw, int idx, cf src, "
                          "float kps, float kms,\n    cf* fw_out\n)")
_APPLY_FW_STORE = "    cf_store(fw, idx, src);\n"
_APPLY_FW_STORE_PURE = (
    "    // THE FIRST STORE, HANDED BACK INSTEAD OF PERFORMED: the caller writes it to\n"
    "    // the SCRATCH volume for its own cell and discards it for a foreign\n"
    "    // recompute, so nothing this launch reads is a word this launch wrote.\n"
    "    *fw_out = src;\n")
_APPLY_F_STORE = "    cf_store(f, idx, a);\n"
_APPLY_F_STORE_PURE = (
    "    // THE SECOND STORE, LIKEWISE: same two separate accumulations, same order,\n"
    "    // the value RETURNED so the caller stores it to scratch and the increment\n"
    "    // consumes it from a register.\n"
    "    return a;\n")
_MUL_FIELD_LEFT_HEAD = "__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"


def _needle(source: str, old: str, new: str, what: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified text carries {count} copies of {what}, not one; this "
            f"product LIFTS that text rather than retyping it and cannot splice around "
            f"its absence")
    return source.replace(old, new, 1)


def _arm(expansion) -> int:
    return complex_emitter.normalized_expansion(expansion)


def _prelude(arm: int) -> str:
    """The certified complex prelude ``complex_source`` builds: HEAD, the arm block, TAIL."""
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[arm]  # noqa: SLF001
            + complex_emitter._TAIL)  # noqa: SLF001


def constitutive_prelude(expansion) -> str:
    """The certified complex prelude with ``constitutive_apply`` made PURE.

    Four anchored edits, all in :data:`CONSTITUTIVE_LIFT_EDITS`. ``cshift_up`` /
    ``cshift_dn`` / ``pml_apply`` ride along unused: they are the certified TAIL's and
    lifting the tail whole is what keeps the shared mutation needles resolving.
    """
    prelude = _prelude(_arm(expansion))
    prelude = _needle(prelude, _APPLY_SIGNATURE, _APPLY_SIGNATURE_PURE,
                      "constitutive_apply's declaration")
    prelude = _needle(prelude, _APPLY_PARAMETERS, _APPLY_PARAMETERS_PURE,
                      "constitutive_apply's parameter list")
    prelude = _needle(prelude, _APPLY_FW_STORE, _APPLY_FW_STORE_PURE,
                      "constitutive_apply's split-field store")
    prelude = _needle(prelude, _APPLY_F_STORE, _APPLY_F_STORE_PURE,
                      "constitutive_apply's field store")
    return prelude


def mul_field_left_increment_source() -> str:
    """The FMA_V1 ``mul_field_left``, lifted from the arm block and RENAMED.

    Read off ``complex_emitter._ARM_SOURCE[FMA_V1]`` rather than retyped, so the
    increment's multiply IS the certified spelling that measured 0 of 14,387,104 words
    (and 0 of 6 on the flushed-zero class the NAIVE spelling misses).
    """
    block = complex_emitter._ARM_SOURCE[  # noqa: SLF001
        complex_emitter.EXPANSIONS[INCREMENT_MULTIPLY_SOURCE_ARM]]
    if block.count(_MUL_FIELD_LEFT_HEAD) != 1:
        raise AssertionError(
            "the FMA_V1 arm block no longer declares mul_field_left once; the "
            "increment's multiply has no certified text to lift")
    start = block.index(_MUL_FIELD_LEFT_HEAD)
    end = block.index("}\n", start) + len("}\n")
    body = block[start:end]
    if "__fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f)" not in body:
        raise AssertionError(
            "the FMA_V1 mul_field_left no longer spells its real part as the explicit "
            "fma of the field's real part; the spelling that measured 0 words is gone")
    return ("\n// CuPy's complex<float> * complex<float>(w, +0.0f) AS NVRTC CONTRACTS IT --\n"
            "// complex_emitter._ARM_SOURCE[FMA_V1].mul_field_left, lifted and renamed so\n"
            "// the increment spells it under EITHER constitutive arm. The uncontracted\n"
            "// four-product form differs on the sign of a FLUSHED zero when Hy.re * 0.5\n"
            "// underflows at row 0 (6 of 4,005,000 words under flush; the gate plants it).\n"
            + body.replace("mul_field_left(", "mul_field_left_increment(", 1))


def div_coefficient_source() -> str:
    """``complex64 / float32`` as CuPy's device path spells it: ``arithmetic.h:96-110``
    with ``rhs = (d, +0.0f)`` and every zero-valued term LEFT IN. Measured 0 of
    14,387,104 words under both policies; the three other spellings bite."""
    return r'''
// cupy/_core/include/cupy/complex/arithmetic.h:96-110 operator/(complex<T>, complex<T>)
// with rhs = complex<float>(d) = (d, +0.0f) (complex_inl.h:28) -- the SCALED algorithm,
// transcribed with every zero-valued term LEFT IN. They decide the sign of a zero and
// the rounding on the edge class: the "tidied" form misses 15,598 / 60,038 words there.
// NOT numpy's z * (1.0f / d) (793,105 / 820,455 of 14,387,104 words) and NOT the real
// family's componentwise '/' (3,584,917 / 3,591,312). Measured, not chosen.
__device__ __forceinline__ cf cf_div_coefficient(cf z, float d) {
    float s = fabsf(d) + fabsf(0.0f);
    float oos = 1.0f / s;
    float ars = z.re * oos;
    float ais = z.im * oos;
    float brs = d * oos;
    float bis = 0.0f * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    cf q;
    q.re = ((ars * brs) + (ais * bis)) * oos;
    q.im = ((ais * brs) - (ars * bis)) * oos;
    return q;
}
'''


def raw_update_H_cell_source(expansion) -> str:
    """The certified ``update_H_pml_complex_bloch`` body as a ``__device__`` function
    of a CELL, storing nothing: both results come back through out-parameters."""
    arm = _arm(expansion)
    source = complex_emitter.complex_source("update_H", arm)
    prelude = _prelude(arm)
    if not source.startswith(prelude):
        raise AssertionError(
            "complex_source('update_H') no longer begins with the prelude this module "
            "lifts separately; the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError("the certified update_H signature terminator is not unique")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified update_H body does not end with a brace")
    body = body[: -len("}\n")]
    body = _needle(
        body, _THREAD_PREAMBLE,
        "    // THE CELL ARRIVES AS A FLAT INDEX; the bounds guard left with the thread\n"
        "    // index. THE DECODE BELOW STAYS: it indexes the absorber profile at the\n"
        "    // recomputed cell rather than at the thread's own.\n",
        "the constitutive body's thread preamble")
    if body.count(_INDEX_DECODE) != 1:
        raise AssertionError(
            "the certified update_H body no longer carries exactly one index decode")
    for component, axis in enumerate("xyz"):
        coordinate = "ijk"[component]
        old = (f"    constitutive_apply(f{component}, w{component}, idx, s{component}, "
               f"kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);\n")
        new = (f"    h_out[{component}] = constitutive_apply_pure(f{component}, "
               f"w{component}, idx, s{component}, kps_{axis}[{coordinate}], "
               f"kms_{axis}[{coordinate}], &w_out[{component}]);\n")
        body = _needle(body, old, new, f"component {component}'s constitutive_apply call")
    if "constitutive_apply(" in body:
        raise AssertionError("a storing constitutive_apply call survived the capture")
    unpack = [
        "    // The certified body's own names, bound to the pack (float32 WORD views of",
        "    // complex64 volumes; cf_load does the doubling).",
        "    const int nx = weld.nx; const int ny = weld.ny;",
        "    const int nz = weld.nz;",
        "    (void) nx;   // the removed bounds guard's operand",
        "    const float* __restrict__ f0 = weld.Hx;",
        "    const float* __restrict__ f1 = weld.Hy;",
        "    const float* __restrict__ f2 = weld.Hz;",
        "    const float* __restrict__ w0 = weld.f_w_Hx;",
        "    const float* __restrict__ w1 = weld.f_w_Hy;",
        "    const float* __restrict__ w2 = weld.f_w_Hz;",
        "    const float* __restrict__ g0 = weld.Bx;",
        "    const float* __restrict__ g1 = weld.By;",
        "    const float* __restrict__ g2 = weld.Bz;",
    ]
    for stem in ("kps", "kms"):
        for axis in "xyz":
            unpack.append(f"    const float* __restrict__ {stem}_{axis} = weld.{stem}_{axis};")
    return ("\n// The certified update_H_pml_complex_bloch, evaluated at ONE ARBITRARY CELL\n"
            "// and storing nothing: a pure function of H, f_w_H and B, none of which this\n"
            "// launch writes.\n"
            "__device__ __forceinline__ void raw_update_H_cell(\n"
            "    int idx, const WeldArgs& weld, cf* h_out, cf* w_out\n"
            ") {\n" + "\n".join(unpack) + "\n" + body + "}\n")


def signature() -> str:
    return '''
extern "C" __global__ void update_H_increment_pml_cyl_complex(
    // THE SCRATCH OUTPUTS (float32 word views of complex64), a DIFFERENT allocation
    // from every pre-launch volume below.
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // THE INCREMENT: the pre-cumsum stage of cylindrical_rderiv_prefix(Hy_new, 0.5).
    float* __restrict__ increment,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // The array path's own cached float32 row vectors: nx rows and nx - 1 rows.
    const float* __restrict__ weights, const float* __restrict__ divisor,
    // Extents in COMPLEX CELLS.
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z,
    const float* __restrict__ kms_x, const float* __restrict__ kms_y,
    const float* __restrict__ kms_z
) {
'''


def increment_source() -> str:
    lines = [
        "",
        "    // --- THE INCREMENT (stepping.py:1293-1302 at THIS cell), complex64. Row 0",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1293-1302->1322-1331
        "    // is (+0.0f, +0.0f); rows i >= 1 difference Hy_new at i (register) and",
        "    // i - 1 (RECOMPUTED from pre-launch state, never a read of another",
        "    // thread's store). The multiply is the FMA_V1 mul_field_left whatever arm",
        "    // the constitutive half runs, and the divide is CuPy's scaled algorithm;",
        "    // both MEASURED (module docstring). NOT numpy's z * (1/d), NOT the real",
        "    // family's componentwise '/'.",
        "    const int nyz = ny * nz;",
        "    const int i = idx / nyz;",
        "    if (i == 0) {",
        "        cf_store(increment, idx, cf_zero());",
        "        return;",
        "    }",
        "    cf halo_h[3];",
        "    cf halo_w[3];",
        "    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);",
        "    cf wi = mul_field_left_increment(own_h[1], weights[i]);",
        "    cf wim1 = mul_field_left_increment(halo_h[1], weights[i - 1]);",
        "    cf diff = cf_sub(wi, wim1);",
        "    cf_store(increment, idx, cf_div_coefficient(diff, divisor[i - 1]));",
        "",
    ]
    return "\n".join(lines)


def kernel_source(expansion) -> str:
    """The whole device source of launch 1 under one EXPANSION arm."""
    weld = "\n".join([
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        _hd.weld_args_construction().rstrip("\n"),
        "",
        "    // --- update_H at the thread's OWN cell into registers, stored to SCRATCH.",
        "    cf own_h[3];",
        "    cf own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    cf_store(Hx_out, idx, own_h[0]); cf_store(Hy_out, idx, own_h[1]);",
        "    cf_store(Hz_out, idx, own_h[2]);",
        "    cf_store(f_w_Hx_out, idx, own_w[0]); cf_store(f_w_Hy_out, idx, own_w[1]);",
        "    cf_store(f_w_Hz_out, idx, own_w[2]);",
    ])
    source = (constitutive_prelude(expansion) + mul_field_left_increment_source()
              + div_coefficient_source() + _hd.weld_args_struct()
              + raw_update_H_cell_source(expansion) + signature() + weld
              + increment_source() + "}\n")
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source launch 1 can emit, keyed by arm name -- for a digest."""
    return {name: kernel_source(name) for name in sorted(complex_emitter.EXPANSIONS)}


def source_digest() -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for name, source in device_sources().items():
        digest.update(name.encode("ascii"))
        digest.update(source.encode("ascii"))
    return digest.hexdigest()


def _get_kernel(expansion, source: Optional[str] = None):
    import cupy as cp  # noqa: PLC0415

    arm = _arm(expansion)
    code = kernel_source(arm) if source is None else source
    key = _kernel_cache_key(f"{KERNEL_NAME}_arm{arm}", True, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_cylindrical_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                     sources: Any = None, license: Any = None,
                                     subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION: the certified complex constitutive predicate on the H side (asked
    FIRST; it asks the licence first of all), the certified complex Dcyl curl predicate
    on ``step_D`` (every integer m), the seam's withdraw clause, the boundary
    resolution launch 2 binds, the increment's radial extent, and the rotation's six
    volumes.
    """
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False, so nothing would perform the withdraw "
            f"before its launches"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]
    try:
        _cyl.cylindrical_complex_boundary_codes(grid)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the cylindrical curl's boundary resolution refuses this grid: "
                       f"{type(exc).__name__}: {exc}")
    source = getattr(fields, PREFIX_SOURCE, None)
    if source is None:
        return False, f"{PREFIX_SOURCE} is not allocated; the increment is built from it"
    if int(source.shape[0]) < 2:
        return False, (f"{PREFIX_SOURCE} has {int(source.shape[0])} radial row(s); the "
                       f"increment differences rows i and i - 1")
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def _prefix_row_vectors(fields: Any) -> Tuple[Any, Any]:
    from ..stepping import _cylindrical_rderiv_weights  # noqa: PLC0415

    source = getattr(fields, PREFIX_SOURCE)
    xp = fields.grid.xp
    rows = int(source.shape[0])
    real_dtype = source.real.dtype
    key = ("cyl_rderiv", rows, float(PREFIX_IR0_VALUE), real_dtype)
    scratch = getattr(fields, "scratch", None)
    if scratch is not None and hasattr(scratch, "constant"):
        weights, divisor = scratch.constant(
            key, lambda: _cylindrical_rderiv_weights(xp, rows, PREFIX_IR0_VALUE,
                                                     real_dtype))
    else:
        weights, divisor = _cylindrical_rderiv_weights(xp, rows, PREFIX_IR0_VALUE,
                                                       real_dtype)
    flat_w, flat_d = weights.reshape(-1), divisor.reshape(-1)
    if int(flat_w.shape[0]) != rows or int(flat_d.shape[0]) != rows - 1:
        raise ValueError("the prefix row vectors do not match the radial extent")
    if flat_w.dtype != real_dtype or flat_d.dtype != real_dtype:
        raise ValueError("the prefix row vectors are not the storage's real dtype")
    return flat_w, flat_d


def resolve(fields: Any, grid: Any, pml: Any, expansion) -> Dict[str, Any]:
    """Everything both launches need, resolved ONCE per frozen configuration -- the
    ``i*m/r`` rows and every scalar included, so launch 2 adds no per-step array work
    (the census plan caches them the same way)."""
    import cupy as cp  # noqa: PLC0415

    if complex_pml_kernels is None:
        raise RuntimeError(
            "complex_pml_kernels is not importable on this host (it imports CuPy at "
            "module scope), so the constitutive tables cannot be built; the predicate "
            "and the emitter need neither and still run")
    arm = _arm(expansion)
    source = getattr(fields, PREFIX_SOURCE)
    weights, divisor = _prefix_row_vectors(fields)
    dtdx = float(grid.dt / grid.dx)
    m = int(grid.m)
    rows = int(source.shape[0])
    return {
        "arm": arm,
        "constitutive": complex_pml_kernels.complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER["H"]),
        "curl": _cyl.cylindrical_complex_curl_tables(pml, _cyl.HALF_INTEGER["step_D"]),
        "codes": _cyl.cylindrical_complex_boundary_codes(grid),
        "m_class": _cyl.m_class(m),
        "zero_rows": _cyl.zero_rows(m, bool(grid.accurate_fields_near_cylorigin)),
        "imr_rows": _cyl.imr_rows_for("step_D", grid.xp, m, dtdx, rows, source.dtype),
        "increment_scalars": _cyl.axis_increment_scalars(m, dtdx),
        "axis_coef": _cyl.axis_coefficient(dtdx),
        "scratch": _hd.fused_hd_pair_scratch(fields),
        "increment": cp.empty_like(source),
        "prefix": cp.empty_like(source),
        "weights": weights, "divisor": divisor,
        "dtdx": dtdx,
        "launches": 0, "cumsum_calls": 0, "curl_launches": 0,
    }


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", state["scratch"][name])
    visit("increment", state["increment"])
    visit("prefix", state["prefix"])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz",
                 "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    visit("weights", state["weights"])
    visit("divisor", state["divisor"])
    for index, row in enumerate(state["imr_rows"]):
        visit(f"imr{index}", row)
    for axis in "xyz":
        visit(f"kps_{axis}", state["constitutive"][f"kps_{axis}"])
        visit(f"kms_{axis}", state["constitutive"][f"kms_{axis}"])
    if collisions:
        raise ValueError(
            "this product binds every field, scratch, buffer and table argument "
            "__restrict__, and these arguments alias: " + "; ".join(collisions))
    return len(bound)


def launch_constitutive_and_increment(fields: Any, state: Dict[str, Any],
                                      kernel: Optional[Any] = None,
                                      threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """LAUNCH 1 under the resolved arm. ``kernel`` and ``threads`` are the gate's doors."""
    word_view = _cyl.word_view
    nx, ny, nz = (int(n) for n in getattr(fields, PREFIX_SOURCE).shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    tables = state["constitutive"]
    arguments: List[Any] = [word_view(state["scratch"][name]) for name in SCRATCH_VOLUMES]
    arguments.append(word_view(state["increment"]))
    arguments += [word_view(getattr(fields, name)) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz")]
    arguments += [state["weights"], state["divisor"]]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz)]
    arguments += [tables[f"kps_{axis}"] for axis in "xyz"]
    arguments += [tables[f"kms_{axis}"] for axis in "xyz"]
    (kernel or _get_kernel(state["arm"]))((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "kernel": KERNEL_NAME,
            "arm": complex_emitter.EXPANSION_NAMES[state["arm"]]}


def scan_and_rotate(fields: Any, state: Dict[str, Any], rotate: bool = True) -> Any:
    """``xp.cumsum(increment, axis=0, out=prefix)`` -- the oracle, untouched -- then the
    rotation. Counted as one launch."""
    xp = fields.grid.xp
    xp.cumsum(state["increment"], axis=0, out=state["prefix"])
    state["cumsum_calls"] += 1
    state["launches"] += 1
    if rotate:
        state["scratch"] = _hd.rotate_into_fields(fields, state["scratch"])
    return state["prefix"]


def launch_curl(fields: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """LAUNCH 2: the certified ``cyl_step_D_pml_complex`` through its own launcher on the
    gridless route, every derived quantity resolved once and the prefix handed in."""
    _cyl.step_cylindrical_complex(
        "step_D", fields, state["arm"],
        dtdx=state["dtdx"], tables=state["curl"], boundary_codes=state["codes"],
        imr_rows=state["imr_rows"], prefix=state["prefix"],
        m_class_code=state["m_class"], zero_rows_count=state["zero_rows"],
        increment_scalars=state["increment_scalars"], axis_coef=state["axis_coef"])
    state["launches"] += 1
    state["curl_launches"] += 1
    return {"launched": True, "kernel": CURL_KERNEL_NAME}


def run_cylindrical_fused_hd_pair(fields: Any, grid: Any, pml: Any, expansion, *,
                                  sources: Any = None, license: Any = None,
                                  subnormal_policy: Any = None,
                                  state: Optional[Dict[str, Any]] = None,
                                  kernel: Optional[Any] = None,
                                  threads: int = _FUSED_THREADS,
                                  rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve once, then launch 1 -> cumsum -> rotate -> launch 2.
    A refusal is returned, not raised; ``state`` carries the scratch between steps."""
    covered, reason = covers_cylindrical_fused_hd_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if state is None:
        state = resolve(fields, grid, pml, expansion)
    assert_bindings_are_disjoint(fields, state)
    before = state["launches"]
    launch_constitutive_and_increment(fields, state, kernel, threads)
    scan_and_rotate(fields, state, rotate)
    launch_curl(fields, state)
    return {"launched": True, "state": state, "rotated": bool(rotate),
            "launches_this_run": state["launches"] - before,
            "launches_per_run": LAUNCHES_PER_RUN,
            "raw_kernel_launches_this_run": 2, "cumsum_calls_this_run": 1,
            "replaces": REPLACES, "arm": complex_emitter.EXPANSION_NAMES[state["arm"]]}
