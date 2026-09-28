#!/usr/bin/env python
"""DEVICE byte gate for the Triton Dcyl COMPLEX H->D product:
``meep_gpu/triton_kernels/cylindrical_fused_hd_pair.py``.

The complex-storage twin of ``gate_triton_cylindrical_real_fused_hd_pair.py``, on the
``(cylindrical complex -> cylindrical complex PML)`` H->D cell: every m the complex
family carries (m = 0 under complex64 storage, |m| = 1, |m| >= 2 with and without the
accurate axis branch) and both z terminations. Same shape -- ``update_H`` plus the
pre-scan increment in launch 1, the array module's ``cumsum`` untouched, the certified
complex cylindrical curl in launch 2 -- same four reference engines, same evidence
standard (uint32 words per complete driver step, both subnormal policies from empty
caches, every zero beside a control that bites). Read the real gate and the kit first;
this file states only what the complex arithmetic adds.

THE TWO SPELLINGS UNDER TEST INVERT AGAINST THE REAL FAMILY, and each has a measured
wrong neighbour that a careful reader would copy:

* the multiply ``Hy * w`` is the FMA_V1 ``mul_field_left`` arrangement with EXPLICIT
  fused multiply-adds -- the spelling of CuPy's contracted ``complex<float> *
  complex<float>(w)``. The four-product form differs on the SIGN of a flushed zero when
  ``Hy.re * 0.5`` underflows at row 0; a random battery does not reach it (0 of
  14,387,104 in the NVRTC record), so ``increment_stage`` PLANTS that class and measures
  the four-product form against ``cupy.multiply`` there, under both policies;
* the divide ``/ d`` is CuPy's SCALED complex division with every zero-valued term kept
  and both reciprocals correctly rounded. numpy's ``a * (float32(1)/d)`` -- correct on
  the NumPy-engined backend -- is wrong here (793,105 / 820,455 of 14,387,104 words,
  NVRTC); componentwise ``re/d, im/d`` -- the real family's spelling -- is wrong on a
  quarter of all words; tidying the zero terms out is wrong on the edge class.

Both are armed here as kernel mutations AND as increment-stage variants compiled by
Triton, so the design record's NVRTC numbers are replaced by this backend's own.

THE FLUSH RECORD IS A DIFFERENT CLAIM. Every complex arm in this package is certified
under ``keep`` (``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY``); under ``flush``
the predicate refuses by that clause, the composer selects nothing on the seam, and the
two composition references and the lifted rows are REFUSED and named. What the flush
record measures is the product's arithmetic from arrays with the ``EXPANSION`` arm
FORCED to the one the keep-cut probe licenses, against the array path and the two
certified singles built the same way. ``composer_reachable_under_this_policy`` in the
record says which reading applies.

    MEEP_GPU_COMPLEX_EXPANSION_PROBE=<keep-cut probe.json> \\
      CUDA_VISIBLE_DEVICES=<n> CUPY_CACHE_DIR=<empty, token ftz_stripped> \\
      TRITON_CACHE_DIR=<empty> PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_cylindrical_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_cylindrical_hd_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_complex as cx  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels.offdiag_scratch_weld import twin_table  # noqa: E402

GATE = "triton_cylindrical_fused_hd_pair"
CELL_ARMS: Tuple[str, str] = family.ARMS
CENSUS = kit.CENSUS
SEAM_RECORD = kit.SEAM_RECORD
MODES = kit.MODES

SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/cylindrical_fused_hd_pair.py",
    "meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py",
    "meep_gpu/triton_kernels/fused_hd_pair.py",
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/cylindrical_complex.py",
    "meep_gpu/triton_kernels/complex_fields.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/stepping.py",
    "meep_gpu/withdraw_hoist.py",
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/fastpath.py",
    "parity/meep_gpu/triton_cylindrical_hd_gate_kit.py",
    "parity/meep_gpu/gate_triton_cylindrical_fused_hd_pair.py",
)

#: The synthetic fixtures: the three m classes the kernel compiles (M_ZERO, M_ONE,
#: M_MANY with both near-axis row counts), both z terminations, one single-column
#: (nz = 1) periodic case on the ring-cyl / perturbation_theory shape class, and
#: Courant numbers below the |m| >= 2 accurate-branch ceiling where that branch is on.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m0_zm", {"shape": (12, 16), "m": 0, "courant": 0.35, "pml": 3,
               "z_boundary": "metallic"}),
    ("m1_zm", {"shape": (12, 16), "m": 1, "courant": 0.37, "pml": 3,
               "z_boundary": "metallic"}),
    ("mneg1_zp", {"shape": (12, 16), "m": -1, "courant": 0.31, "pml": 3,
                  "z_boundary": "periodic"}),
    ("m2_acc_zm", {"shape": (14, 10), "m": 2, "courant": 0.25, "pml": 2,
                   "z_boundary": "metallic", "accurate": True}),
    ("m3_zm", {"shape": (12, 16), "m": 3, "courant": 0.25, "pml": 3,
               "z_boundary": "metallic"}),
    ("m1_nz1_zp", {"shape": (16, 1), "m": 1, "courant": 0.37, "pml": 3,
                   "z_boundary": "periodic"}),
    # THE MUTATION CASE: 96 radial rows so the ladder reaches the values where the two
    # wrong divide spellings part from the scaled algorithm (numpy's reciprocal
    # coincides with it wherever d * (1/d) rounds to exactly 1.0, which holds for
    # every d < 41 -- MEASURED NULL on a 12-row fixture 2026-09-07), and 192 cells so
    # the launch is still one program.
    ("m1_nr96_nz2", {"shape": (96, 2), "m": 1, "courant": 0.37, "pml": 3,
                     "z_boundary": "periodic"}),
    # FIVE PROGRAMS (1280 cells at BLOCK = 256): the halo recompute crosses program
    # boundaries here, as it does on every corpus row.
    ("m1_nr40_nz32", {"shape": (40, 32), "m": 1, "courant": 0.35, "pml": 4,
                      "z_boundary": "metallic"}),
)
MUTATION_CASE = "m1_nr96_nz2"

KERNEL_NAME = "cyl_complex_update_H_increment"

# ---------------------------------------------------------------------------
# The increment-stage variants (compiled by Triton from these statements)
# ---------------------------------------------------------------------------

_FMA_MULTIPLY = (
    "wh_re = tl.math.fma(fr_i, w_i, (fi_i * 0.0) * -1.0)\n"
    "wh_im = tl.math.fma(fr_i, 0.0, fi_i * w_i)\n"
    "wb_re = tl.math.fma(fr_b, w_im1, (fi_b * 0.0) * -1.0)\n"
    "wb_im = tl.math.fma(fr_b, 0.0, fi_b * w_im1)")
_NAIVE_MULTIPLY = (
    "wh_re = (fr_i * w_i) - (fi_i * 0.0)\n"
    "wh_im = (fr_i * 0.0) + (fi_i * w_i)\n"
    "wb_re = (fr_b * w_im1) - (fi_b * 0.0)\n"
    "wb_im = (fr_b * 0.0) + (fi_b * w_im1)")
_SCALED_DIVIDE = (
    "s = tl.abs(d_im1) + 0.0\n"
    "oos = tl.math.div_rn(1.0, s)\n"
    "ars = d_re * oos\n"
    "ais = d_im * oos\n"
    "brs = d_im1 * oos\n"
    "bis = 0.0 * oos\n"
    "s2 = (brs * brs) + (bis * bis)\n"
    "oos2 = tl.math.div_rn(1.0, s2)\n"
    "q_re = ((ars * brs) + (ais * bis)) * oos2\n"
    "q_im = ((ais * brs) - (ars * bis)) * oos2")

INCREMENT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "multiply": _FMA_MULTIPLY, "divide": _SCALED_DIVIDE, "expected": "IDENTICAL",
        # Contraction ON is RECORDED, not asserted: with explicit fmas there is no
        # contractible multiply-add left except the divide's sums, whose fused-in
        # addends are zero-valued (neutral away from the subnormal boundary). The
        # design record measured 0 / 0 for NVRTC; this is Triton's own number.
        # MEASURED under keep (2026-09-07 smoke, 65 cases, 8,991,940 words): 0.
        "also_fusion_on": True, "expected_fusion_on": {"keep": "IDENTICAL", "flush": "RECORDED"},
    },
    "numpy_reciprocal": {
        "multiply": _FMA_MULTIPLY,
        "divide": ("inv = tl.math.div_rn(1.0, d_im1)\n"
                   "q_re = tl.math.fma(d_re, inv, (d_im * 0.0) * -1.0)\n"
                   "q_im = tl.math.fma(d_re, 0.0, d_im * inv)"),
        "expected": "DIFFERS",
        "why": "numpy's complex64 / float32 IS a * (float32(1)/d) and the Metal verdict "
               "prescribes it; CuPy's is the scaled algorithm, and this spelling is wrong "
               "here on ~5.5 % of words (NVRTC record)",
    },
    "componentwise": {
        "multiply": _FMA_MULTIPLY,
        "divide": "q_re = tl.math.div_rn(d_re, d_im1)\nq_im = tl.math.div_rn(d_im, d_im1)",
        "expected": "DIFFERS",
        "why": "the real family's `/` carried over componentwise: wrong on ~25 % of words",
    },
    "zero_terms_tidied": {
        "multiply": _FMA_MULTIPLY,
        "divide": ("s = tl.abs(d_im1)\n"
                   "oos = tl.math.div_rn(1.0, s)\n"
                   "ars = d_re * oos\n"
                   "ais = d_im * oos\n"
                   "brs = d_im1 * oos\n"
                   "s2 = brs * brs\n"
                   "oos2 = tl.math.div_rn(1.0, s2)\n"
                   "q_re = (ars * brs) * oos2\n"
                   "q_im = (ais * brs) * oos2"),
        "expected": {"edge": "DIFFERS", "planted": "RECORDED", "default": "IDENTICAL"},
        # MEASURED under keep (2026-09-07 smoke): 7,662 words on 13 of 13 edge cases, 0 on
        # every uniform / wide_dynamic / cancelling / planted case.
        "why": "the scaled algorithm with its zero-valued terms simplified away: wrong "
               "exactly where a product is a signed zero or underflows (the edge class; "
               "15,598 / 60,038 words in the NVRTC record), invisible elsewhere",
    },
    "first_reciprocal_div_full": {
        "multiply": _FMA_MULTIPLY,
        "divide": _SCALED_DIVIDE.replace("oos = tl.math.div_rn(1.0, s)", "oos = 1.0 / s"),
        "expected": "DIFFERS",
        "why": "the FIRST reciprocal (1 / |d|, the integer ladder 1..nr-1 on this seam) "
               "through Triton's div.full.f32. MEASURED 2026-09-07 (keep smoke, 65 cases): "
               "631,870 of 8,991,940 words, biting on every case -- the approximate "
               "division does NOT land on the correctly rounded word for the ladder",
    },
    "second_reciprocal_div_full": {
        "multiply": _FMA_MULTIPLY,
        "divide": _SCALED_DIVIDE.replace("oos2 = tl.math.div_rn(1.0, s2)",
                                         "oos2 = 1.0 / s2"),
        "expected": {"keep": "IDENTICAL", "flush": "RECORDED"},
        "why": "the SECOND reciprocal (1 / (brs*brs)) through div.full.f32. MEASURED NULL "
               "2026-09-07 (keep smoke, 65 cases): 0 of 8,991,940 words. brs = d * (1/|d|) "
               "is 1.0 to within an ulp for every ladder value, and 1 / (1 +- ulp) is the "
               "one place div.full.f32 lands on the correctly rounded word; recorded as an "
               "equivalence of THIS ladder, not a licence for `/`",
    },
    "four_product_multiply": {
        "multiply": _NAIVE_MULTIPLY, "divide": _SCALED_DIVIDE,
        "expected": "IDENTICAL",
        "why": "the uncontracted four-product complex multiply (the certified "
               "constitutive's NAIVE arm) inside the INCREMENT. MEASURED 0 of 8,991,940 "
               "words under BOTH policies (2026-09-07), planted class included, and the "
               "null is structural: the four-product form differs from the fused one only "
               "on the SIGN of a flushed zero in `Hy*w`, and that sign cannot reach the "
               "increment -- it survives the subtract only when the other weighted term is "
               "itself a zero, and then the scaled divide's `(ars*brs) + (ais*bis)` "
               "launders it unless `d_im < 0`, which forces that other term's real word "
               "to be +0.0 and the difference vanishes. The multiply SUB-LEG is where the "
               "spelling is byte-visible (planted class under flush) and is what pins it",
    },
}

MULTIPLY_VARIANTS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "multiply_here": ("wh_re = tl.math.fma(fr_i, w_i, (fi_i * 0.0) * -1.0)\n"
                          "wh_im = tl.math.fma(fr_i, 0.0, fi_i * w_i)"),
        "expected": "IDENTICAL",
    },
    "four_product": {
        "multiply_here": ("wh_re = (fr_i * w_i) - (fi_i * 0.0)\n"
                          "wh_im = (fr_i * 0.0) + (fi_i * w_i)"),
        # MEASURED 2026-09-07: 0 under keep on every class; under flush the planted
        # class DIFFERS on exactly nz words per case (the row-0 sign of the flushed
        # `Hy.re * 0.5`) and the edge class on 0-1 words.
        "expected": {"keep": "IDENTICAL",
                     "flush": {"planted": "DIFFERS", "edge": "RECORDED",
                               "default": "IDENTICAL"}},
    },
    "componentwise": {
        "multiply_here": "wh_re = fr_i * w_i\nwh_im = fi_i * w_i",
        "expected": {"edge": "DIFFERS", "planted": "DIFFERS", "default": "IDENTICAL"},
    },
    "coefficient_left": {
        "multiply_here": ("wh_re = tl.math.fma(w_i, fr_i, (0.0 * fi_i) * -1.0)\n"
                          "wh_im = tl.math.fma(w_i, fi_i, 0.0 * fr_i)"),
        # The other fma ORIENTATION (complex_fields._mul_coefficient_left's). MEASURED
        # 2026-09-07: 0 under keep; under flush 1-2 words on three edge cases -- the
        # orientation decides which zero-valued addend is fused with a flushed product,
        # which is why complex_fields keeps its call-site orientation table.
        "expected": {"keep": "IDENTICAL", "flush": {"edge": "RECORDED", "default": "IDENTICAL"}},
    },
}

# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

_HALO_CALL = ("        h_re, h_im = _h_tap_complex(1, i - 1, j, k, inner,\n"
              "                                    hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,\n"
              "                                    kp0, km0, kp1, km1, kp2, km2, ny, nz, "
              "EXPANSION)\n")
_DIVIDE_CALL = "        q_re, q_im = _div_coefficient_scaled(d_re, d_im, d_im1)\n"
_SUMS = ("        q_re = ((ars * brs) + (ais * bis)) * oos2\n"
         "        q_im = ((ais * brs) - (ars * bis)) * oos2\n")

MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m_numpy_reciprocal_divide": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _DIVIDE_CALL,
        "new": ("        q_re, q_im = _mul_field_left_fma(d_re, d_im, "
                "tl.math.div_rn(1.0, d_im1))\n"),
        "why": "numpy's a * (float32(1)/d) -- the sibling backend's verdict, correct "
               "there because that engine is NumPy -- in place of CuPy's scaled complex "
               "division (793,105 / 820,455 of 14,387,104 words, NVRTC record)",
    },
    "m_componentwise_divide": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _DIVIDE_CALL,
        "new": ("        q_re = tl.math.div_rn(d_re, d_im1)\n"
                "        q_im = tl.math.div_rn(d_im, d_im1)\n"),
        "why": "the real family's `/` carried over componentwise (3,584,917 / 3,591,312 "
               "of 14,387,104 words, NVRTC record)",
    },
    "m_first_reciprocal_spelled_with_slash": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        oos = tl.math.div_rn(1.0, s)\n",
        "new": "        oos = 1.0 / s\n",
        "why": "the scaled divide's first reciprocal through Triton's div.full.f32 "
               "instead of the correctly rounded intrinsic",
    },
    "m_second_reciprocal_spelled_with_slash": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        # Under flush the policy refuses to compile `/` at all (div.full.f32 has no
        # uniform .ftz lowering through the flush executor), so the null is keep-only.
        "expected_override": {"keep": "NULL", "flush": "REFUSED BY THE POLICY"},
        "old": "        oos2 = tl.math.div_rn(1.0, s2)\n",
        "new": "        oos2 = 1.0 / s2\n",
        "why": "the scaled divide's second reciprocal through div.full.f32. MEASURED NULL on "
               "the increment leg (2026-09-07 keep smoke: 0 of 8,991,940 words over 65 "
               "cases): its divisor brs*brs is 1.0 to within an ulp on this integer ladder, "
               "where the approximate division lands on the correctly rounded word. Armed "
               "beside the FIRST reciprocal, which bites on every case, so the two record "
               "which reciprocal the correctly rounded intrinsic is load-bearing for",
    },
    "m_zero_terms_tidied_out": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "old": _SUMS,
        "new": ("        q_re = (ars * brs) * oos2\n"
                "        q_im = (ais * brs) * oos2\n"),
        "why": "PREDICTED NULL on this smooth fixture: dropping the zero-valued cross "
               "terms changes a word only where a product is a signed zero or underflows, "
               "which a seeded normal field does not reach; increment_stage measures the "
               "same edit on the edge class, where it DIFFERS (15,598 / 60,038 words in "
               "the NVRTC record)",
    },
    "m_four_product_multiply_in_the_increment": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "old": ("        out_re = tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)\n"
                "        out_im = tl.math.fma(z_re, 0.0, z_im * c)\n"),
        "new": ("        out_re = (z_re * c) - (z_im * 0.0)\n"
                "        out_im = (z_re * 0.0) + (z_im * c)\n"),
        "why": "MEASURED NULL, and structurally so: the four-product form differs from "
               "the fused arrangement only on the sign of a FLUSHED zero when Hy.re * 0.5 "
               "underflows at row 0 (6 of 4,005,000 words under flush in the NVRTC "
               "record, and nz words per planted case in this gate's multiply sub-leg), "
               "and that sign cannot reach the increment through the subtract and the "
               "scaled divide (increment_stage's four_product_multiply variant: 0 of "
               "8,991,940 words under both policies, planted class included). The "
               "multiply sub-leg is where the spelling is pinned",
    },
    "m_subtract_reversed": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        d_re = wh_re - wb_re\n",
        "new": "        d_re = wb_re - wh_re\n",
        "why": "the real part's difference taken the wrong way round",
    },
    "m_weight_row_misaligned": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        w_im1 = tl.load(wgt + i - 1, mask=inner, other=0.0)\n",
        "new": "        w_im1 = tl.load(wgt + i, mask=inner, other=0.0)\n",
        "why": "the neighbour's field weighted by the OWN row's r/dr",
    },
    "m_axis_row_not_zeroed": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        q_re = tl.where(inner, q_re, 0.0)\n",
        "new": "        q_re = q_re + 0.0\n",
        "why": "the array path assigns 0 to row 0; left un-zeroed the axis row's real "
               "word carries Hy[0].re * w[0] into every prefix of its column",
    },
    "m_halo_reads_the_stored_Hy": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _HALO_CALL,
        "new": ("        h_re = tl.load(hi1 + 2 * (idx - nyz), mask=inner, other=0.0)\n"
                "        h_im = tl.load(hi1 + 2 * (idx - nyz) + 1, mask=inner, other=0.0)\n"),
        "why": "the backward neighbour read from the STORED pre-launch Hy -- the "
               "previous step's field -- instead of recomputed; the purity ledger is the "
               "floor that makes this a claim about the kernel",
    },
    "m_halo_reads_the_neighbours_scratch": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "old": _HALO_CALL,
        "new": ("        h_re = tl.load(ho1 + 2 * (idx - nyz), mask=inner, other=0.0)\n"
                "        h_im = tl.load(ho1 + 2 * (idx - nyz) + 1, mask=inner, other=0.0)\n"),
        "why": "the backward neighbour read from the scratch this launch is WRITING. "
               "MEASURED NULL 2026-09-07 on a one-program fixture (the own-cell store "
               "precedes the neighbour load in program order, same warp); across "
               "programs it is the block-schedule race, not deterministically "
               "observable, so it is recorded rather than armed and the STORED-Hy halo "
               "is the deterministic control",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo1 + 2 * idx, s1_re, mask=live)\n",
        "new": "        tl.store(wi1 + 2 * idx, s1_re, mask=live)\n",
        "why": "the split-field history's real word written into the PRE-LAUNCH buffer: "
               "a neighbour's recompute then reads B where it needs B_prev, and the "
               "rotation publishes an unwritten scratch",
    },
    "m_rotation_skipped": {
        "target": "host", "host": "rotation_skipped", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the plan launches and does NOT move the engine's references onto the "
               "freshly written scratch",
    },
    "m_fp_fusion_on": {
        "target": "host", "host": "fusion_on", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "why": "PREDICTED NULL under the FMA_V1 arm: every complex product in both "
               "launches is an EXPLICIT fma or a componentwise add of fma results, so no "
               "implicit contraction can form; the scaled divide's two sums fuse a "
               "zero-valued addend, which is neutral away from the subnormal boundary "
               "(the NVRTC record measured 0 / 0 for the complex primary built with "
               "contraction on). Recorded rather than dropped so a compiler that found "
               "a contractible pattern here would be caught as a CAUGHT-where-NULL",
    },
    "m_scan_skipped": {
        "target": "host", "host": "scan_skipped", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the increment handed to launch 2 UNSCANNED",
    },
    "m_ir0_zero_ladder": {
        "target": "host", "host": "ir0_zero_ladder", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the two invariant row vectors rebuilt at ir0 = 0.0 where step_D prefixes "
               "Hp at 0.5",
    },
    "m_curl_reads_the_pre_launch_H": {
        "target": "host", "host": "curl_reads_pre_launch_H", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launch 2's three magnetic pointers redirected to the PRE-LAUNCH volumes: "
               "the curl consumes the state the seam's first half has not yet produced",
    },
    "m_constitutive_takes_the_half_integer_lattice": {
        "target": "host", "host": "half_integer", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the constitutive half bound the `_h` PML sub-lattice: a half-cell-wrong "
               "absorber profile that compiles, launches and converges",
    },
    "m_withdraw_after_step_D": {
        "target": "host", "host": "withdraw_after_step_D", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the seam's electric withdraw performed AFTER the launch; driven in the "
               "withdraw leg on a fixture carrying an integrated electric source",
    },
}

BYTE_NEUTRAL: Dict[str, Any] = {
    "tag": "m_first_product_of_each_sum_as_explicit_fma",
    "old": _SUMS,
    "new": ("        q_re = tl.math.fma(ars, brs, ais * bis) * oos2\n"
            "        q_im = tl.math.fma(ais, brs, (ars * bis) * -1.0) * oos2\n"),
    "why": "the scaled divide's two sums with their first product fused explicitly. "
           "PREDICTED NULL (measured 0 / 0 for NVRTC): the fused-in addend is a "
           "zero-valued product, so either arrangement reproduces CuPy -- and confirming "
           "it here is what shows the divide's identity does not hinge on the one "
           "arrangement the kernel happens to spell",
}


# ---------------------------------------------------------------------------
# The family adapter
# ---------------------------------------------------------------------------

def _probe_record() -> Tuple[Optional[str], Any]:
    path = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
    if not path:
        return None, None
    return path, complex_fields.load_expansion_probe(path)


class Spec(kit.FamilySpec):
    gate = GATE
    family = family
    complex_storage = True
    cell_arms = CELL_ARMS
    sources = SOURCES
    cases = CASES
    mutation_case = MUTATION_CASE
    kernel_name = KERNEL_NAME
    mutations = MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    increment_variants = INCREMENT_VARIANTS
    multiply_variants = MULTIPLY_VARIANTS

    def driver_keywords(self, keywords: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(keywords)

    def pml_spec(self, keywords: Mapping[str, Any]) -> Any:
        thickness = int(keywords.get("pml", 3))
        nr, nz = keywords["shape"]
        spec: Dict[str, Any] = {"x": {"high": thickness}}
        if keywords.get("z_boundary", "metallic") == "metallic" and int(nz) > 2 * thickness + 1:
            spec["z"] = thickness
        return spec

    # -- the forced arm ----------------------------------------------------------
    def forced_expansion(self) -> Optional[Dict[str, Any]]:
        path, record = _probe_record()
        if record is None:
            raise SystemExit(
                f"the complex gate needs {complex_fields.PROBE_PATH_ENVIRONMENT} to name a "
                f"keep-cut expansion probe artifact; none is readable at {path!r}")
        licence = complex_fields.expansion_license(record)
        if licence["expansion"] is None:
            raise SystemExit(f"the probe artifact licenses no arm: {licence['refusals']}")
        return {"probe": path, "expansion": int(licence["expansion"]),
                "arm": licence.get("arm"), "basis": licence.get("basis"),
                "what_forced_means": (
                    "the arm the keep-cut probe licenses, bound from arrays without the "
                    "predicate, because under this policy the predicate refuses by the "
                    "certification clause and the composer selects nothing")}

    def _expansion(self, forced: bool) -> int:
        if forced:
            return int(self.forced_expansion()["expansion"])
        code = complex_fields._resolve_expansion(None)  # noqa: SLF001
        if code is None:
            raise RuntimeError("no EXPANSION arm is licensed under the policy in force: "
                               + "; ".join(complex_fields._expansion_reasons(None)))  # noqa: SLF001
        return int(code)

    def _from_arrays_inputs(self, driver: Any) -> Dict[str, Any]:
        fields, pml, grid = driver.fields, driver.pml, driver.fields.grid
        arrays = {name: getattr(fields, name)
                  for name in family.ROTATED + family.IN_PLACE + family.CONSTITUTIVE_SOURCES}
        for name, twin in twin_table(fields, family.ROTATED).items():
            arrays["scratch_" + name] = twin
        kinds = stepping._boundary_kinds(grid, pml)  # noqa: SLF001
        return {
            "arrays": arrays,
            "curl_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                          for axis in "xyz" for stem in ("kms", "sinv")},
            "constitutive_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                                  for axis in "xyz" for stem in ("kps", "kms")},
            "dtdx": grid.dt / grid.dx, "m": int(grid.m),
            "accurate": bool(grid.accurate_fields_near_cylorigin),
            "bcz": 1 if kinds[2] == "metallic" else 0,
        }

    def build_weld(self, driver: Any, kernel: Any = None, curl_kernel: Any = None,
                   forced: bool = False) -> Any:
        sources = tuple(getattr(driver, "_sources", ()))
        if not forced:
            plan = family.plan_cylindrical_fused_hd_pair(
                driver.fields, driver.pml, sources, kernel=kernel, curl_kernel=curl_kernel)
            if plan is not None:
                return plan
            # The withdraw leg's fixture is refused BY DESIGN (a standing integrated
            # electric source); the gate's from-arrays route builds the same plan there
            # so the hoist and its two null controls can be driven.
            verdict = family.cylindrical_fused_hd_pair_coverage(driver.fields, driver.pml,
                                                                sources)
            if not all("standing integrated" in reason for reason in verdict.reasons):
                raise RuntimeError("the product refused a fixture this gate expects it to "
                                   f"admit: {verdict.reasons}")
        inputs = self._from_arrays_inputs(driver)
        return family.plan_cylindrical_fused_hd_pair_from_arrays(
            inputs["arrays"], inputs["curl_flat"], inputs["constitutive_flat"],
            inputs["dtdx"], inputs["m"], inputs["accurate"], inputs["bcz"],
            self._expansion(forced=forced), driver.fields, driver.xp, driver.fields.scratch,
            kernel=kernel, curl_kernel=curl_kernel)

    def singles(self, driver: Any, forced: bool = False) -> Tuple[Any, Any]:
        fields, pml = driver.fields, driver.pml
        if not forced:
            return (cx.plan_cylindrical_complex_constitutive(fields, pml, "H"),
                    cx.plan_cylindrical_complex_curl(fields, pml, "step_D"))
        inputs = self._from_arrays_inputs(driver)
        expansion = self._expansion(forced=True)
        side = complex_fields.CONSTITUTIVE_SIDES["H"]
        h_arrays = {name: getattr(fields, name)
                    for name in tuple(side["targets"]) + tuple(side["aux"]) + tuple(side["sources"])}
        constitutive = complex_fields.plan_complex_constitutive_from_arrays(
            "H", h_arrays, inputs["constitutive_flat"], expansion)
        spec = triton_launch.SUB_STEPS["step_D"]
        d_arrays = {name: getattr(fields, name) for name in spec["targets"] + spec["sources"]}
        d_arrays.update({"fu_" + name: getattr(fields, "fu_" + name) for name in spec["targets"]})
        curl = cx.plan_cylindrical_complex_curl_from_arrays(
            "step_D", d_arrays, inputs["curl_flat"], inputs["dtdx"], inputs["m"],
            inputs["accurate"], inputs["bcz"], expansion, driver.xp,
            scratch=fields.scratch)
        return constitutive, curl

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.cylindrical_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernels(self, owner: Any) -> Dict[str, Any]:
        from meep_gpu.triton_kernels import kernels as tkernels  # noqa: PLC0415
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as pair_d,
            cylindrical_fused_magnetic_pair as pair_b,
        )

        name = type(owner).__name__
        if name == "CylindricalFusedHdPairPlan":
            return {"_kernel": family.cyl_complex_update_H_increment_kernel(),
                    "_curl_kernel": cx.cyl_complex_pml_curl_step}
        if name == "ComplexConstitutivePlan":
            return {"_kernel": complex_fields.bloch_constitutive_step}
        if name == "CylindricalComplexCurlPlan":
            return {"_kernel": cx.cyl_complex_pml_curl_step}
        if name == "ConstitutivePlan":
            return {"_kernel": tkernels.constitutive_step}
        if name == "PmlCurlPlan":
            return {"_kernel": tkernels.pml_curl_step}
        if name == "CylindricalFusedMagneticPairPlan":
            return {"_kernel": pair_b.cyl_complex_fused_curl_constitutive_B_kernel()}
        if name == "CylindricalFusedElectricPairPlan":
            return {"_kernel": pair_d.cyl_complex_fused_curl_constitutive_D_kernel()}
        return {}

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        fields, pml = driver.fields, driver.pml
        grid = fields.grid

        class _Inactive:
            is_active = False

        return [
            {"name": "undeclared_source_list", "fields": fields, "pml": pml, "sources": None,
             "must_refuse": True, "needle": "was not declared"},
            {"name": "no_sources", "fields": fields, "pml": pml, "sources": (),
             "must_refuse": False},
            {"name": "integrated_electric_withdraw", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(),), "must_refuse": True,
             "needle": "standing integrated"},
            {"name": "non_integrated_electric_source", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(integrated=False),), "must_refuse": False},
            {"name": "integrated_magnetic_source", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(component="Hz", field_type="B"),),
             "must_refuse": False},
            {"name": "not_cylindrical", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, grid=kit.GridView(grid, cylindrical=False)),
             "needle": "not cylindrical"},
            {"name": "real_storage", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, force_complex_fields=False),
             "needle": "force_complex_fields is not set"},
            {"name": "one_radial_row", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, grid=kit.GridView(
                 grid, shape=(1, 1, int(grid.shape[2])))),
             "needle": "radial extent is 1"},
            {"name": "no_step_scratch", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, scratch=None), "needle": "StepScratch"},
            {"name": "inactive_absorber", "fields": fields, "pml": _Inactive(), "sources": (),
             "must_refuse": True, "first_half": "cylindrical complex constitutive half"},
        ]

    def half_integer_rebind(self, plan: Any, pml: Any) -> Any:
        from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

        plan._h_coeff = tuple(  # noqa: SLF001
            CupyPointer(_flat(getattr(pml, f"{stem}_{axis}_h")))
            for axis in "xyz" for stem in ("kps", "kms"))
        return plan

    # -- launch 2's magnetic arguments: cyl_complex_pml_curl_step(f0, f1, f2, u0, u1,
    #    u2, g0, g1, g2, pfx, c0, c2, ...) -- the three sources at 6..8, word views.
    def word_view(self, array: Any) -> Any:
        return complex_fields._word_view(array)  # noqa: SLF001

    def curl_prelaunch_positions(self) -> Dict[int, int]:
        return {6: 0, 7: 1, 8: 2}

    def default_curl_kernel(self) -> Any:
        return cx.cyl_complex_pml_curl_step

    def policy_reaches_composer(self, policy: Optional[str]) -> Tuple[bool, str]:
        reasons: List[str] = list(complex_fields.expansion_certification_reasons(policy))
        path, record = _probe_record()
        if record is None:
            reasons.append(f"no expansion probe artifact is named by "
                           f"{complex_fields.PROBE_PATH_ENVIRONMENT} (got {path!r})")
        elif policy is not None:
            reasons.extend(complex_fields.expansion_policy_reasons(record, policy))
        return (not reasons, "; ".join(reasons))

    def composition_refusal_needle(self) -> str:
        return "float32 subnormal policy"

    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        certified = family.certified_constitutive_tail()
        lifted = family.lifted_constitutive_tail()
        measures["constitutive_chars"] = len(certified)
        if certified != lifted:
            findings.append(
                f"_h_cell_complex is NOT complex_fields.bloch_constitutive_step's own body "
                f"with the declared lift edits: {len(certified)} chars against {len(lifted)}")
        statements = [line for line in certified.splitlines()
                      if line.strip() and not line.strip().startswith("#")]
        measures["constitutive_statements"] = len(statements)
        if len(statements) < 40:
            findings.append(f"the lifted complex constitutive body is only {len(statements)} "
                            f"statements; the anchors have crossed")
        for edit in family.CONSTITUTIVE_LIFT_EDITS:
            if not edit.get("why"):
                findings.append(f"the lift edit {edit.get('line')!r} carries no reason")
        for stem in ("f", "w", "g", "e"):
            for target in range(3):
                if f"{stem}{target} +" in lifted:
                    findings.append(f"the lifted body still indexes {stem}{target}")
        launch1 = family.increment_statements()
        measures["launch1_statements"] = len(launch1)
        for tag, needle in family.INCREMENT_SPELLING:
            hits = sum(1 for line in launch1 if needle in line)
            measures[f"spelling_{tag}"] = hits
            if hits != 1:
                findings.append(f"the increment statement {tag!r} appears {hits} times, not once")
        for needle in family.DIVIDE_SPELLING + family.MULTIPLY_SPELLING:
            if sum(1 for line in launch1 if line == needle) != 1:
                findings.append(f"the spelling {needle!r} is not exactly one statement of "
                                f"the launch-1 kernel's helpers")
        slashes = [line for line in launch1
                   if " = " in line and not line.startswith(('"', "'"))
                   and " / " in line.split("#", 1)[0] and "//" not in line]
        measures["slash_divisions_in_launch1"] = len(slashes)
        if slashes:
            findings.append(f"launch 1 divides with `/` (div.full.f32): {slashes}")
        # The increment's multiply is NOT the constitutive's arm-gated helper.
        if any("_mul_field_left(" in line and "_mul_field_left_fma(" not in line
               for line in launch1):
            findings.append("the increment reaches the arm-gated _mul_field_left; the "
                            "constitutive's EXPANSION must not leak into the increment")
        spec = cx.PREFIX[family.CURL_SUB_STEP]
        if (family.PREFIX_COMPONENT, float(family.PREFIX_IR0)) != (
                spec["component"], float(spec["ir0"])):
            findings.append("the product's prefix declaration is not cylindrical_complex's")
        if spec["extend_wall_row"]:
            findings.append("the D side's prefix now extends a wall row")
        measures["prefix"] = {"component": family.PREFIX_COMPONENT, "ir0": family.PREFIX_IR0}
        tree = ast.parse((API_ROOT / "meep_gpu" / "triton_kernels" / "kernels.py").read_text(
            encoding="utf-8"))
        certified_block = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
        if family.DEFAULT_BLOCK != certified_block:
            findings.append(f"DEFAULT_BLOCK {family.DEFAULT_BLOCK} is not kernels.py's "
                            f"{certified_block}")
        try:
            measures["sub_lattice_suffixes"] = list(family._sub_lattice_suffixes())  # noqa: SLF001
        except AssertionError as error:
            findings.append(str(error))
        measures["launches_per_run"] = family.LAUNCHES_PER_RUN
        measures["kernel_launches_per_run"] = family.KERNEL_LAUNCHES_PER_RUN
        if family.LAUNCHES_PER_RUN != family.KERNEL_LAUNCHES_PER_RUN + 1:
            findings.append("LAUNCHES_PER_RUN does not count exactly two kernels and one scan")
        return findings, measures

    def ghost_observability(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        curl_text = Path(cx.__file__).read_text(encoding="utf-8")
        product_text = Path(family.__file__).read_text(encoding="utf-8")
        for needle in ("curl0_re =", "curl2_re =", "def cyl_complex_pml_curl_step"):
            if needle in product_text:
                findings.append(f"the product module carries curl text ({needle!r})")
        measures["launch2_is_the_module_kernel_object"] = (
            "_cx.cyl_complex_pml_curl_step" in product_text)
        if not measures["launch2_is_the_module_kernel_object"]:
            findings.append("the plan does not launch cylindrical_complex.cyl_complex_pml_curl_step")
        guarded = "p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)" in curl_text
        masked = ("curl2_re = tl.where(at_r, 0.0, curl2_re)" in curl_text
                  and "curl2_im = tl.where(at_r, 0.0, curl2_im)" in curl_text)
        measures["prefix_backward_tap_guarded"] = guarded
        measures["target2_zeroed_at_the_axis_row_under_BACKWARD"] = masked
        if not guarded:
            findings.append("the certified complex curl no longer guards the prefix's "
                            "backward tap")
        if not masked:
            findings.append("the certified complex curl no longer zeroes Dz's curl at the "
                            "axis row")
        measures["what_this_licenses"] = (
            "the r near ghost the prefix's backward tap would read at row 0 is masked and "
            "its target's curl is zeroed there on every m class, so no edit to what that "
            "ghost serves can be byte-visible on the D side; this product passes the "
            "scan's output to the unchanged kernel")
        return findings, measures

    def arbitration_incumbents(self) -> Tuple[str, str]:
        return ("fused pair B (cylindrical complex)", "fused pair D (cylindrical complex)")

    def product_row(self) -> Dict[str, str]:
        return {"curl_slot": "update_H", "module": "cylindrical_fused_hd_pair",
                "coverage": "cylindrical_fused_hd_pair_coverage",
                "builder": "plan_cylindrical_fused_hd_pair",
                "label": "fused pair H->D (cylindrical complex)"}


SPEC = Spec()

leg_driver_order = lambda: kit.leg_driver_order(SPEC)  # noqa: E731
leg_transcription = lambda: kit.leg_transcription(SPEC)  # noqa: E731
leg_purity_ledger = lambda: kit.leg_purity_ledger(SPEC)  # noqa: E731
leg_ghost_observability = lambda: kit.leg_ghost_observability(SPEC)  # noqa: E731
lift_basis = lambda results: kit.lift_basis(SPEC, results)  # noqa: E731
run_product = kit.run_product
drive = kit.drive
evaluate_row = kit.evaluate_row
leg_lift = kit.leg_lift
_is_private = kit._is_private  # noqa: SLF001


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the gate; see the module docstring and the kit for what each leg measures."""
    return kit.main(SPEC, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    sys.exit(main())
