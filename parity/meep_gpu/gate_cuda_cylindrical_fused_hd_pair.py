#!/usr/bin/env python3
"""Device byte gate for the CUDA COMPLEX Dcyl H->D two-launch product, any m.

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/cylindrical_fused_hd_pair.py``: launch
1 (slot ``update_H``) computes the certified ``update_H_pml_complex_bloch`` body, lifted
PURE per EXPANSION arm, at the thread's own cell into launch-local scratch and RECOMPUTES
it at the one-cell backward radial neighbour to form the pre-``cumsum`` increment of
``stepping.cylindrical_rderiv_prefix(Hy_new, 0.5)`` over complex64 storage;
``xp.cumsum`` runs untouched on the array path; the bindings rotate; launch 2 is the
certified ``cyl_step_D_pml_complex`` with the prefix handed in -- the ``i*m/r``
couplings, the |m| = 1 fold, the m = 0 axis post-add and the |m| >= 2 zeroing all read
the ROTATED H pointwise there.

THE TWO COMPLEX SPELLINGS INVERT THE NUMPY PRESCRIPTION, AND THE INVERSION IS ARMED. On
this backend ``complex64 / float32`` is CuPy's SCALED complex division with every
zero-valued term kept, and ``complex64 * float32`` is the FMA_V1 ``mul_field_left`` as
NVRTC contracts CuPy's own ufunc body. NumPy's ``a * (float32(1)/d)`` -- right on Metal,
prescribed by the adjudication's section 9 -- is WRONG here on ~5.5 % of words, and the
real family's componentwise ``/`` on ~25 %; both are armed mutations that must be CAUGHT
in the fused kernel, and the ``spelling`` leg measures every candidate against CuPy's
own ufuncs on standalone kernels. The NAIVE four-product multiply's 6-word class (the
sign of a flushed zero when ``Hy.re * 0.5`` underflows at row 0) is PLANTED on a
fixture and armed inside the fused kernel, with the prediction -- written before the
run, with the algebra -- that the divide's ``((ars*brs) + (ais*bis))`` launders the sign
so the fused increment cannot move; the class is measured to bite on the bare multiply
under ``flush``. A prediction refuted is a finding, recorded either way.

Everything else is the real sibling's gate over complex64 storage, every integer m, both
z terminations, the ``accurate_fields_near_cylorigin`` branch, one-column rows and a
corpus-sized extent: four references per complete driver step, three launch counters
per weld step, the floors, the halo ledger, the scan-order restatement per case, the
driver-level sync and withdraw legs, the arbitration as shipped and wired in-process,
and the corpus lift over the cell's 18 rows (16 driven, 2 refused BY NAME on the
standing integrated withdraw).

STANDALONE: not in ``fused_pairs``' tables; the wiring is a patch in the lane directory.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cylindrical_hd_gate_common as common  # noqa: E402
from cylindrical_hd_gate_common import (Adapter, case_rng, cp, differing,  # noqa: E402
                                        encodes_ascii, log, mutation_needles_resolve)
import gate_cuda_cylindrical_real_fused_hd_pair as real_gate  # noqa: E402

SUBJECT_PACKAGE = "cuda_kernels"
SEED = 20260907
FAMILY_NAME = "cuda_cylindrical_fused_hd_pair"
BLOCK_KEY = "cuda_cylindrical_fused_hd_pair_gate"
CELL_ARMS: Tuple[str, str] = ("cuda_complex/complex", "cuda_cyl_complex/cylindrical complex")

#: The fixtures: every m class, both z terminations, odd extents, the accurate-origin
#: branch, one-column rows, a thin absorber, a corpus-sized extent. ``plant`` marks the
#: fixtures that ALSO run the planted row-0 class.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "m0_r16_z20_metallic", "shape": (16, 1, 20), "m": 0, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "planted")},
    {"label": "m0_r16_z20_periodic", "shape": (16, 1, 20), "m": 0, "z_kind": "periodic"},
    {"label": "m+1_r16_z20_metallic", "shape": (16, 1, 20), "m": 1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "planted", "signed_zero")},
    # THE DIVISOR-LADDER FIXTURE. On the D side the divisor is the INTEGER ladder
    # d = 1, 2, 3, ... (ir0 = 0.5), and CuPy's scaled divide reduces EXACTLY to
    # numpy's re * fl(1/d) wherever d * fl(1/d) rounds to 1 in float32 -- which is every
    # d < 41 (first exceptions 41, 47, 55, 61, 82, 83, 94, 97, ...; measured on the
    # host before this fixture was added). A 16-row fixture therefore cannot tell the
    # two spellings apart on non-zero data; this one reaches row 64.
    {"label": "m+1_r64_z12_metallic", "shape": (64, 1, 12), "m": 1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "signed_zero")},
    {"label": "m-1_r16_z20_periodic", "shape": (16, 1, 20), "m": -1, "z_kind": "periodic"},
    {"label": "m+1_r9_z17_metallic", "shape": (9, 1, 17), "m": 1, "z_kind": "metallic"},
    {"label": "m+2_r9_z17_metallic", "shape": (9, 1, 17), "m": 2, "z_kind": "metallic"},
    {"label": "m+3_r9_z17_periodic", "shape": (9, 1, 17), "m": 3, "z_kind": "periodic"},
    {"label": "m+5_r12_z48_metallic", "shape": (12, 1, 48), "m": 5, "z_kind": "metallic"},
    {"label": "m-1_r40_z16_metallic", "shape": (40, 1, 16), "m": -1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "signed_zero")},
    # The accurate-origin branch drops the default treatment's zero rows near the axis,
    # and the array path refuses it BY NAME above courant 1/(|m| + 0.5) (0.2857 at |m| = 3):
    # the near-axis update is a smooth growing mode there, not a crash. The fixture-wide
    # default (0.35) tripped that refusal on the first campaign cut (attempt 1, all five
    # per-fixture legs errored on this row only), so the row carries its own courant.
    {"label": "m+3_accurate_r16_z20", "shape": (16, 1, 20), "m": 3, "z_kind": "metallic",
     "accurate": True, "courant": 0.25},
    # ONE-COLUMN ROWS: a one-cell z axis must be PERIODIC (Grid refuses a metallic unit
    # axis by name), which is what the three one-column corpus rows carry.
    {"label": "m+1_r80_z1_periodic", "shape": (80, 1, 1), "m": 1, "z_kind": "periodic"},
    {"label": "m0_thin_r16_z20_metallic", "shape": (16, 1, 20), "m": 0, "z_kind": "metallic",
     "thin_absorber": True, "classes": ("uniform", "subnormal_band", "planted")},
    {"label": "m+1_r120_z120_metallic", "shape": (120, 1, 120), "m": 1, "z_kind": "metallic"},
)
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("m+1_r16_z20_metallic", "m+1_r64_z12_metallic",
                                         "m0_thin_r16_z20_metallic", "m+2_r9_z17_metallic")
REDUCED_LABELS: Tuple[str, ...] = ("m0_r16_z20_metallic", "m+1_r16_z20_metallic",
                                   "m+2_r9_z17_metallic")

#: The FMA_V1 ``mul_field_left`` body as this family lifts it, anchored on the RENAMED
#: head so the needle cannot land on the arm block's own copy.
_MUL_HEAD = ("cf mul_field_left_increment(cf z, float c) {\n    cf o;\n"
             "    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);\n"
             "    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);")

DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "numpy_reciprocal_divide": {
        "old": "    cf_store(increment, idx, cf_div_coefficient(diff, divisor[i - 1]));",
        "new": ("    { float r = 1.0f / divisor[i - 1]; cf q; q.re = diff.re * r; "
                "q.im = diff.im * r; cf_store(increment, idx, q); }"),
        "why": ("numpy's a * (float32(1)/d) -- the adjudication's section 9 prescription, "
                "right on Metal whose oracle is NumPy, WRONG here: 793,105 / 820,455 of "
                "14,387,104 words in the probe"),
        "live_on": ("rows whose integer divisor d has d * fl(1/d) != 1 in float32 (41, 47, "
                    "55, 61, 82, ...) -- the r64 fixture -- and the signed-zero class "
                    "everywhere (the sum's zero sign)"),
        "classes": ("uniform", "signed_zero"),
    },
    "componentwise_divide": {
        "old": "    cf_store(increment, idx, cf_div_coefficient(diff, divisor[i - 1]));",
        "new": ("    { cf q; q.re = diff.re / divisor[i - 1]; q.im = diff.im / divisor[i - 1]; "
                "cf_store(increment, idx, q); }"),
        "why": ("the real family's IEEE '/' carried onto complex storage: 3,584,917 / "
                "3,591,312 of 14,387,104 words in the probe"),
        "live_on": "every fixture",
    },
    "simplified_scaled_divide": {
        "score_increment": True,
        "old": "    float s = fabsf(d) + fabsf(0.0f);",
        "new": "    float s = fabsf(d);",
        "also": [("    float bis = 0.0f * oos;\n    s = (brs * brs) + (bis * bis);",
                  "    float bis = 0.0f;\n    s = (brs * brs);"),
                 ("    q.re = ((ars * brs) + (ais * bis)) * oos;\n"
                  "    q.im = ((ais * brs) - (ars * bis)) * oos;",
                  "    q.re = (ars * brs) * oos;\n    q.im = (ais * brs) * oos;")],
        "why": ("the scaled algorithm with its zero-valued terms 'tidied' away: 15,598 / "
                "60,038 words on the edge class in the probe (zero signs and the rounding "
                "of the sum)"),
        "live_on": "the signed-zero class (a -0.0 difference against a +0.0 zero term)",
        "classes": ("uniform", "subnormal_band", "signed_zero"),
    },
    "naive_four_product_multiply": {
        "score_increment": True,
        "old": _MUL_HEAD,
        "new": ("cf mul_field_left_increment(cf z, float c) {\n    cf o;\n"
                "    o.re = (z.re * c) - (z.im * 0.0f);\n"
                "    o.im = (z.re * 0.0f) + (z.im * c);"),
        "why": ("the uncontracted four-product multiply (CuPy's ufunc body as written, "
                "not as NVRTC contracts it): the sign of a FLUSHED zero when Hy.re * 0.5 "
                "underflows at row 0 -- 6 of 4,005,000 words on the bare multiply under flush"),
        "live_on": "the planted row-0 class, under flush",
        "classes": ("planted",),
        "predicted_null": (
            "ARMED WITH THE PLANT AND PREDICTED NULL IN THE FUSED KERNEL, BY ALGEBRA: the "
            "class puts +0.0 (naive) against -0.0 (fma) in wim1.re at row 1. The subtract "
            "wi.re - wim1.re preserves that sign only when wi.re is -0.0, which needs "
            "Hy_new[1].im >= 0; the divide's q.re = ((ars*brs) + (ais*bis)) * oos then "
            "preserves it only when ais*bis is -0.0, which needs diff.im < 0, i.e. "
            "Hy_new[1].im < Hy_new[0].im / 3 < 0. The two conditions contradict, so the "
            "sum launders the sign and no increment word can move. The class IS measured to "
            "bite on the bare multiply in the spelling leg (under flush); here the "
            "prediction is recorded held or refuted"),
    },
    "componentwise_multiply": {
        "score_increment": True,
        "old": _MUL_HEAD,
        "new": ("cf mul_field_left_increment(cf z, float c) {\n    cf o;\n"
                "    o.re = z.re * c;\n"
                "    o.im = z.im * c;"),
        "why": ("the zero cross terms dropped from the increment's multiply: same values, "
                "different zero signs -- 9,583 / 40,322 words on the edge class in the probe"),
        "live_on": "the signed-zero class (Hy.re = -0.0 with Hy.im < 0)",
        "classes": ("uniform", "subnormal_band", "signed_zero"),
    },
    "fmad_default_build": {
        "old": None, "new": None, "options": (),
        "why": ("the same source built WITHOUT --fmad=false. The FMA_V1 arm's fusions are "
                "explicit __fmaf_rn and the increment's multiply is one; the divide's sums "
                "are the only contractible pattern and were measured neutral both ways"),
        "live_on": "recorded as a non-biting control with the reason",
        "classes": ("uniform", "subnormal_band"),
        "predicted_null": (
            "the probe measured the unguarded build at 0 / 14,387,104 on the primary "
            "complex spelling under both policies: with explicit fmas there is no "
            "contractible multiply-add left except the divide's sums, where contraction "
            "is neutral away from the subnormal boundary. Recorded, not dropped; the "
            "constitutive half's fmas are the arm's own"),
    },
    "forward_difference": {
        "old": ("    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);\n"
                "    cf wi = mul_field_left_increment(own_h[1], weights[i]);\n"
                "    cf wim1 = mul_field_left_increment(halo_h[1], weights[i - 1]);"),
        "new": ("    raw_update_H_cell((i + 1 < nx) ? idx + nyz : idx, weld, halo_h, halo_w);\n"
                "    cf wi = mul_field_left_increment(own_h[1], weights[i]);\n"
                "    cf wim1 = mul_field_left_increment(halo_h[1], weights[(i + 1 < nx) ? i + 1 : i]);"),
        "why": "the halo taken FORWARD along r",
        "live_on": "every fixture",
    },
    "neighbour_reads_stored_H": {
        "old": "    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);",
        "new": ("    halo_h[0] = cf_load(weld.Hx, idx - nyz); halo_h[1] = cf_load(weld.Hy, idx - nyz); "
                "halo_h[2] = cf_load(weld.Hz, idx - nyz);\n"
                "    halo_w[0] = cf_zero(); halo_w[1] = cf_zero(); halo_w[2] = cf_zero();"),
        "why": "the neighbour's Hy read from the PRE-launch volume instead of recomputed",
        "live_on": "every fixture",
    },
    "row0_not_zero": {
        "old": "        cf_store(increment, idx, cf_zero());",
        "new": "        cf_store(increment, idx, mul_field_left_increment(own_h[1], weights[i]));",
        "why": "the sum's start is the weighted row instead of (+0.0, +0.0)",
        "live_on": "every fixture",
    },
    "accumulations_regrouped": {
        "old": ("    a = cf_add(a, mul_coefficient_left(kps, src));\n"
                "    a = cf_sub(a, mul_coefficient_left(kms, prev));"),
        "new": ("    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src), "
                "mul_coefficient_left(kms, prev)));"),
        "why": "the two SEPARATE accumulations flattened to f + (kps*src - kms*prev)",
        "live_on": "every fixture with an active absorber",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n\n    // NO ghost rule"),
        "new": "    int k = 0;\n    int j = 0;\n    int i = 0;\n\n    // NO ghost rule",
        "why": "the recomputed cell's absorber coefficients taken from the origin",
        "live_on": "every fixture with an active absorber",
    },
    "f_w_H_written_in_place": {
        "old": "    *fw_out = src;",
        "new": "    cf_store(const_cast<float*>(fw), idx, src);",
        "why": "the split-field history stored IN PLACE: racy by construction",
        "live_on": "every fixture (schedule-dependent)",
        "predicted_null": ("schedule-dependent by construction; the deterministic twin "
                           "foreign_history_reads_the_post_store_value must be CAUGHT"),
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    cf prev = cf_load(fw, idx);",
        "new": "    cf prev = src;",
        "why": "the recurrence reads the value the in-place arrangement would have handed it",
        "live_on": "every fixture with an active absorber",
    },
    "halo_weight_is_own_row": {
        "old": "    cf wim1 = mul_field_left_increment(halo_h[1], weights[i - 1]);",
        "new": "    cf wim1 = mul_field_left_increment(halo_h[1], weights[i]);",
        "why": "the neighbour weighted by the thread's own radius",
        "live_on": "every fixture",
    },
    "divisor_index_shifted": {
        "old": "    cf_store(increment, idx, cf_div_coefficient(diff, divisor[i - 1]));",
        "new": "    cf_store(increment, idx, cf_div_coefficient(diff, divisor[(i < nx - 1) ? i : i - 1]));",
        "why": "the divisor ladder read one row late",
        "live_on": "every fixture",
    },
}

HOST_MUTATIONS: Tuple[str, ...] = (
    "rotation_skipped", "curl_launched_first", "stale_prefix", "ir0_zero_ladder",
    "swap_constitutive_sublattice", "scratch_aliased_to_storage", "wrong_m_class",
    "unlicensed_arm")

BYTE_NEUTRAL: Dict[str, str] = {
    "old": "    cf wi = mul_field_left_increment(own_h[1], weights[i]);",
    "new": "    cf wi = mul_field_left_increment(cf_load(Hy_out, idx), weights[i]);",
}

#: The eleven statements of CuPy's scaled complex division, as measured (the probe's
#: ``cupy_scaled``), pinned so the transcription leg asserts the text rather than trusting
#: the module docstring.
MEASURED_DIVIDE_LINES: Tuple[str, ...] = (
    "    float s = fabsf(d) + fabsf(0.0f);",
    "    float oos = 1.0f / s;",
    "    float ars = z.re * oos;",
    "    float ais = z.im * oos;",
    "    float brs = d * oos;",
    "    float bis = 0.0f * oos;",
    "    s = (brs * brs) + (bis * bis);",
    "    oos = 1.0f / s;",
    "    q.re = ((ars * brs) + (ais * bis)) * oos;",
    "    q.im = ((ais * brs) - (ars * bis)) * oos;",
)


# ---------------------------------------------------------------------------
# The spelling leg: every candidate against CuPy's own ufuncs, standalone kernels
# ---------------------------------------------------------------------------

_CF_PRELUDE = r'''
typedef struct { float re; float im; } cf;
__device__ __forceinline__ cf cf_load(const float* g, int idx) {
    cf z; z.re = g[2 * idx]; z.im = g[2 * idx + 1]; return z;
}
__device__ __forceinline__ void cf_store(float* g, int idx, cf z) {
    g[2 * idx] = z.re; g[2 * idx + 1] = z.im;
}
'''

_SPELL_KERNEL = r'''
extern "C" __global__ void spell(
    float* __restrict__ out, const float* __restrict__ a,
    const float* __restrict__ d, int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    int i = idx / (ny * nz);
    cf z = cf_load(a, idx);
    cf_store(out, idx, @OP@(z, d[i]));
}
'''

_MUL_CANDIDATES: Dict[str, str] = {
    "naive_four_product": r'''
__device__ __forceinline__ cf cmul(cf z, float w) {
    cf o; o.re = (z.re * w) - (z.im * 0.0f); o.im = (z.re * 0.0f) + (z.im * w); return o;
}
''',
    "componentwise": r'''
__device__ __forceinline__ cf cmul(cf z, float w) {
    cf o; o.re = z.re * w; o.im = z.im * w; return o;
}
''',
}

_DIV_CANDIDATES: Dict[str, str] = {
    "numpy_reciprocal": r'''
__device__ __forceinline__ cf cdiv(cf z, float d) {
    float r = 1.0f / d; cf q; q.re = z.re * r; q.im = z.im * r; return q;
}
''',
    "componentwise": r'''
__device__ __forceinline__ cf cdiv(cf z, float d) {
    cf q; q.re = z.re / d; q.im = z.im / d; return q;
}
''',
    "simplified_scaled": r'''
__device__ __forceinline__ cf cdiv(cf z, float d) {
    float s = fabsf(d); float oos = 1.0f / s;
    float ars = z.re * oos; float ais = z.im * oos; float brs = d * oos;
    s = brs * brs; oos = 1.0f / s;
    cf q; q.re = (ars * brs) * oos; q.im = (ais * brs) * oos; return q;
}
''',
}


def complex_field(rng, shape: Tuple[int, int, int], cls: str) -> np.ndarray:
    out = np.empty(shape, dtype=np.complex64)
    view = out.view(np.float32).reshape(shape[0], shape[1], shape[2], 2)
    if cls == "planted_row0":
        # THE 6-WORD CLASS, DENSE: row 0 holds z.re in -[2^-126, 2^-125) and z.im < 0 on
        # every column, so w[0] = 0.5 underflows the product on every word of the row.
        view[..., 0] = rng.standard_normal(shape).astype(np.float32)
        view[..., 1] = rng.standard_normal(shape).astype(np.float32)
        mant = rng.integers(0, 1 << 23, (shape[1], shape[2])).astype(np.uint32)
        tiny = ((np.uint32(1) << np.uint32(31)) | (np.uint32(1) << np.uint32(23)) | mant)
        view[0, :, :, 0] = tiny.astype(np.uint32).view(np.float32)
        view[0, :, :, 1] = -np.abs(rng.standard_normal((shape[1], shape[2]))).astype(np.float32) - 0.1
        return out
    view[..., 0] = real_gate.spelling_field(rng, shape, cls)
    view[..., 1] = real_gate.spelling_field(rng, shape, cls)
    return out


def leg_spelling_complex(adapter: "ComplexAdapter") -> Dict[str, Any]:
    """Every divide and multiply candidate against CuPy's own ``true_divide`` and
    ``multiply`` (the array path's statements), on standalone kernels. The family's two
    spellings must be 0 under this policy; the controls must bite where the probe said
    they do; the naive multiply's 6-word class is POLICY-CONDITIONAL (bites under flush
    on the planted row-0 class; 0 under keep) and is recorded as such."""
    family = adapter.family
    shapes = ((163, 1, 175), (1335, 1, 163), (80, 1, 1))
    classes = ("uniform", "wide_dynamic", "edge", "planted_row0")
    family_mul = family.mul_field_left_increment_source().replace(
        "mul_field_left_increment(", "cmul(", 1)
    family_div = family.div_coefficient_source().replace("cf_div_coefficient(", "cdiv(", 1)
    kernels: Dict[Tuple[str, str], Any] = {}

    def kernel(body: str, op: str) -> Any:
        key = (body, op)
        if key not in kernels:
            kernels[key] = cp.RawKernel(_CF_PRELUDE + body + _SPELL_KERNEL.replace("@OP@", op),
                                        "spell", options=("--fmad=false",))
        return kernels[key]

    cases: List[Dict[str, Any]] = []
    for shape in shapes:
        for cls in classes:
            rng = case_rng(SEED, f"cspelling/{shape}/{cls}")
            a = cp.asarray(complex_field(rng, shape, cls))
            counts = np.arange(shape[0], dtype=np.float64) + 0.5
            w = cp.asarray(counts.reshape(-1, 1, 1).astype(np.float32))
            d = cp.asarray((counts + 0.5).reshape(-1, 1, 1).astype(np.float32))
            mul_ap = cp.multiply(a, w)
            div_ap = cp.true_divide(a, d)
            cells = int(a.size)
            entry: Dict[str, Any] = {"shape": list(shape), "class": cls,
                                     "words": 2 * cells, "multiply": {}, "divide": {}}

            def run(body: str, op: str, vector) -> Any:
                out = cp.empty_like(a)
                kernel(body, op)(((cells + 255) // 256,), (256,),
                                 (out.view(cp.float32), a.view(cp.float32), vector.reshape(-1),
                                  np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])))
                cp.cuda.runtime.deviceSynchronize()
                return out

            entry["multiply"]["family_fma_v1"] = differing(run(family_mul, "cmul", w), mul_ap)
            for name, body in _MUL_CANDIDATES.items():
                entry["multiply"][name] = differing(run(body, "cmul", w), mul_ap)
            entry["divide"]["family_scaled"] = differing(run(family_div, "cdiv", d), div_ap)
            for name, body in _DIV_CANDIDATES.items():
                entry["divide"][name] = differing(run(body, "cdiv", d), div_ap)
            cases.append(entry)
            log(f"[spelling] {shape} {cls:12s} mul={entry['multiply']} div={entry['divide']}")

    def agg(kind: str, name: str) -> int:
        return sum(c[kind][name] for c in cases)

    planted = [c for c in cases if c["class"] == "planted_row0"]
    naive_on_plant = sum(c["multiply"]["naive_four_product"] for c in planted)
    out = {
        "cases": cases, "denominator": len(cases), "words": sum(c["words"] for c in cases),
        "policy": adapter.policy,
        "family_multiply_differing": agg("multiply", "family_fma_v1"),
        "family_divide_differing": agg("divide", "family_scaled"),
        "controls": {
            "divide_numpy_reciprocal": agg("divide", "numpy_reciprocal"),
            "divide_componentwise": agg("divide", "componentwise"),
            "divide_simplified_scaled": agg("divide", "simplified_scaled"),
            "multiply_componentwise": agg("multiply", "componentwise"),
            "multiply_naive_four_product_all_classes": agg("multiply", "naive_four_product"),
            "multiply_naive_four_product_on_the_planted_class": naive_on_plant,
        },
        "naive_multiply_policy_conditional": {
            "expected": ("bites under flush on the planted row-0 class (the sign of a "
                         "flushed zero); 0 under keep, where nothing flushes"),
            "bit_here": naive_on_plant > 0,
            "consistent_with_the_policy": (naive_on_plant > 0) == (adapter.policy == "flush"),
        },
    }
    out["passed"] = bool(
        out["family_multiply_differing"] == 0 and out["family_divide_differing"] == 0
        and out["controls"]["divide_numpy_reciprocal"] > 0
        and out["controls"]["divide_componentwise"] > 0
        and out["controls"]["divide_simplified_scaled"] > 0
        and out["controls"]["multiply_componentwise"] > 0
        and out["naive_multiply_policy_conditional"]["consistent_with_the_policy"])
    return out


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class ComplexAdapter(Adapter):
    complex_storage = True
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band")
    seed = SEED
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    wiring_arms_row = ("complex", "cylindrical complex")
    #: THE UNLICENSED ARM IS DEGENERATE ON THIS FAMILY EXCEPT UNDER FLUSH. Every
    #: multiply launch 1 and 2 perform has a zero-imaginary coefficient (the i*m/r row's
    #: real word is exactly +0.0), so the FMA_V1 and NAIVE spellings agree on every
    #: rounded product AND on every zero sign -- fma(a, b, +-0) rounds a*b once, as the
    #: naive product does -- and differ ONLY where a product UNDERFLOWS and the flush
    #: policy zeroes the naive form's product before its subtract (the 6-word class).
    #: Under keep nothing flushes and the arm is a true null; under flush it must be
    #: caught on a class whose products underflow.
    host_mutation_classes = {"unlicensed_arm": ("uniform", "subnormal_band", "planted")}
    #: (An earlier draft declared the unlicensed arm NULL under keep by that algebra; the
    #: 4-step smoke CAUGHT it on the subnormal-band class of every fixture under BOTH
    #: policies, so it is required caught. The uniform class is where it is degenerate.)

    def __init__(self, policy: Optional[str], licence: Optional[Dict[str, Any]]) -> None:
        from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_fused_hd_pair as family  # noqa: PLC0415

        self.family = family
        self.policy = policy
        self.license = licence
        self.arm = (complex_emitter.normalized_expansion(licence["arm"])
                    if licence and licence.get("arm") else None)
        self.other_arm = None if self.arm is None else 1 - int(self.arm)

    @staticmethod
    def real_plane(name: str, shape: Tuple[int, int, int], value_class: str, rng) -> np.ndarray:
        # THE SIGNED-ZERO CLASS, DENSE ON THE INCREMENT'S OWN VOLUMES: the real planes of
        # Hy, By and f_w_Hy are signed zeros on EVERY cell (so Hy_new.re is a signed
        # zero everywhere and the increment's zero-sign classes are hit on every word),
        # the other real planes on half their cells, the imaginary planes uniform.
        if value_class == "signed_zero" and name in ("Hy", "By", "f_w_Hy"):
            return np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0).astype(np.float32)
        return Adapter.real_plane(name, shape, value_class, rng)

    def plant(self, fields, grid, value_class: str) -> None:
        """THE ROW-0 TINY-NORMAL PLANT on top of a uniform seed. With no absorber on
        phi (kps_y = kms_y = 1 everywhere) ``Hy_new = (Hy + By) - f_w_Hy_prev`` exactly, so
        the pre-launch state is chosen to make ``Hy_new[0] = (-tiny_normal, negative)`` --
        the class where ``Hy.re * 0.5`` underflows -- and ``Hy_new[1].re = -0.0`` with
        ``Hy_new[1].im > 0``, the only arrangement in which the subtract preserves the
        sign the multiply spelling decides (the divide then launders it; see the armed
        mutation's prediction)."""
        if value_class != "planted":
            return
        shape = tuple(int(n) for n in grid.shape)
        columns = (shape[1], shape[2])
        rng = np.random.default_rng(self.seed + 7)
        mant = rng.integers(0, 1 << 23, columns).astype(np.uint32)
        tiny = ((np.uint32(1) << np.uint32(31)) | (np.uint32(1) << np.uint32(23)) | mant)
        row0_by = np.empty(columns, dtype=np.complex64)
        row0_by.real = tiny.astype(np.uint32).view(np.float32)
        row0_by.imag = -(0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        row1_by = np.empty(columns, dtype=np.complex64)
        row1_by.real = np.full(columns, -0.0, dtype=np.float32)
        row1_by.imag = (0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        zeros = np.zeros(columns, dtype=np.complex64)
        row1_hy = np.empty(columns, dtype=np.complex64)
        row1_hy.real = np.full(columns, -0.0, dtype=np.float32)
        row1_hy.imag = (0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        fields.Hy[0] = cp.asarray(zeros)
        fields.f_w_Hy[0] = cp.asarray(zeros)
        fields.By[0] = cp.asarray(row0_by)
        fields.Hy[1] = cp.asarray(row1_hy)
        fields.f_w_Hy[1] = cp.asarray(zeros)
        fields.By[1] = cp.asarray(row1_by)

    # -- the product -------------------------------------------------------------
    def predicate(self, fields, pml, grid, sources):
        return self.family.covers_cylindrical_fused_hd_pair(
            fields, pml, grid, sources, self.license, self.policy)

    def resolve(self, fields, grid, pml, **overrides):
        state = self.family.resolve(fields, grid, pml, overrides.pop("expansion", self.arm))
        state.update(overrides)
        return state

    def launch1(self, fields, state, kernel=None, threads=None):
        return self.family.launch_constitutive_and_increment(
            fields, state, kernel, threads or self.family._FUSED_THREADS)  # noqa: SLF001

    def scan_rotate(self, fields, state, rotate=True):
        return self.family.scan_and_rotate(fields, state, rotate)

    def launch2(self, fields, state):
        return self.family.launch_curl(fields, state)

    def kernel_source(self) -> str:
        return self.family.kernel_source(self.arm)

    def source_digest(self) -> str:
        return self.family.source_digest()

    def weld_state_mutation(self, name, fields, grid, pml, state):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as cyl  # noqa: PLC0415

        if name == "ir0_zero_ladder":
            source = self.prefix_source_volume(fields)
            weights, divisor = common.stepping._cylindrical_rderiv_weights(  # noqa: SLF001
                cp, int(source.shape[0]), 0.0, source.real.dtype)
            state["weights"], state["divisor"] = weights.reshape(-1), divisor.reshape(-1)
            return {"applied": "row vectors rebuilt at ir0 = 0.0"}
        if name == "swap_constitutive_sublattice":
            state["constitutive"] = complex_pml_kernels.complex_constitutive_tables(pml, True)
            return {"applied": "update_H's coefficients from the HALF-INTEGER sub-lattice"}
        if name == "scratch_aliased_to_storage":
            state["scratch"] = {vol: getattr(fields, vol) for vol in self.family.SCRATCH_VOLUMES}
            return {"applied": "the scratch twins bound to the live H/f_w_H"}
        if name == "stale_prefix":
            return {"applied": "the curl handed a prefix of the PRE-launch Hy (in the arrangement)"}
        if name == "wrong_m_class":
            wrong = cyl.M_ONE if state["m_class"] == cyl.M_ZERO else cyl.M_ZERO
            state["m_class"], state["zero_rows"] = wrong, 0
            return {"applied": f"launch 2 told m_class {wrong} instead of {cyl.m_class(int(grid.m))}"}
        if name == "unlicensed_arm":
            fresh = self.family.resolve(fields, grid, pml, self.other_arm)
            state.update(fresh)
            return {"applied": f"both launches compiled under arm {self.other_arm}, the one the "
                               f"platform was NOT licensed for"}
        raise KeyError(name)

    # -- the references ------------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as cyl  # noqa: PLC0415

        tables_h = complex_pml_kernels.complex_constitutive_tables(pml, False)
        tables_e = complex_pml_kernels.complex_constitutive_tables(pml, True)
        scratch = getattr(fields, "scratch", None)
        arm = self.arm
        return {
            "step_B": lambda: cyl.step_cylindrical_complex("step_B", fields, arm, grid=grid,
                                                           pml=pml, scratch=scratch),
            "update_H": lambda: complex_pml_kernels.update_fused_pml_complex(
                "H", fields, arm, tables=tables_h),
            "step_D": lambda: cyl.step_cylindrical_complex("step_D", fields, arm, grid=grid,
                                                           pml=pml, scratch=scratch),
            "update_E": lambda: complex_pml_kernels.update_fused_pml_complex(
                "E", fields, arm, tables=tables_e),
        }

    def released_pairs(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as electric,
            cylindrical_fused_magnetic_pair as magnetic)

        scratch = getattr(fields, "scratch", None)
        return {
            "step_B": lambda: magnetic.run_cylindrical_fused_magnetic_pair(
                fields, grid, pml, dtdx, self.arm, sources=(), license=self.license,
                subnormal_policy=self.policy, scratch=scratch),
            "step_D": lambda: electric.run_cylindrical_fused_electric_pair(
                fields, grid, pml, dtdx, self.arm, sources=(), license=self.license,
                subnormal_policy=self.policy, scratch=scratch),
        }

    def released_pair_replaces(self):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as electric,
            cylindrical_fused_magnetic_pair as magnetic)

        return {"step_B": tuple(magnetic.REPLACES), "step_D": tuple(electric.REPLACES)}

    def driver_kwargs(self):
        return {"force_complex_fields": True, "m": 1}

    def driver_source(self, integrated: bool):
        return {"component": "Ez", "center": (4.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                "frequency": 1.0,
                "source_type": "continuous" if integrated else "gaussian",
                **({"is_integrated": True} if integrated else {"fwidth": 0.2})}

    # -- the family's own host legs -------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as cyl  # noqa: PLC0415

        family = self.family
        per_arm: Dict[str, Any] = {}
        for arm_name, arm in sorted(complex_emitter.EXPANSIONS.items()):
            certified_prelude = (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[arm]  # noqa: SLF001
                                 + complex_emitter._TAIL)  # noqa: SLF001
            emitted_prelude = family.constitutive_prelude(arm)
            # THE INVERSE OF THE DECLARED EDITS reproduces the certified prelude EXACTLY:
            # every edit is anchored, and nothing else moved.
            inverted = emitted_prelude
            inverted = inverted.replace(family._APPLY_F_STORE_PURE, family._APPLY_F_STORE, 1)  # noqa: SLF001
            inverted = inverted.replace(family._APPLY_FW_STORE_PURE, family._APPLY_FW_STORE, 1)  # noqa: SLF001
            inverted = inverted.replace(family._APPLY_PARAMETERS_PURE, family._APPLY_PARAMETERS, 1)  # noqa: SLF001
            inverted = inverted.replace(family._APPLY_SIGNATURE_PURE, family._APPLY_SIGNATURE, 1)  # noqa: SLF001
            certified_body = complex_emitter.complex_source("update_H", arm)[len(certified_prelude):]
            certified_body = certified_body.split("\n) {\n", 1)[1][: -len("}\n")]
            raw_cell = family.raw_update_H_cell_source(arm)
            decode = ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                      "    int i = idx / (ny * nz);\n")
            source = family.kernel_source(arm)
            below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
            curl_source = cyl.cylindrical_complex_source("step_D", arm)
            facts = {
                "prelude_edits_invert_to_the_certified_text": inverted == certified_prelude,
                "declared_edit_anchors_resolve_in_the_certified_text": all(
                    entry["line"] in certified_prelude or entry["line"] in certified_body
                    or entry["became"].startswith("(removed")
                    or "constitutive_apply(f0" in entry["line"]
                    for entry in family.CONSTITUTIVE_LIFT_EDITS),
                "decode_is_carried_verbatim": decode in raw_cell,
                "thread_preamble_is_gone": "blockIdx.x" not in raw_cell,
                "three_calls_captured": raw_cell.count("constitutive_apply_pure(") == 3,
                "no_storing_call_survives": "constitutive_apply(" not in raw_cell.replace(
                    "constitutive_apply_pure(", ""),
                "certified_body_lines_survive": all(
                    line in raw_cell for line in certified_body.splitlines()
                    if line.strip() and "blockIdx" not in line and "return;" not in line
                    and "constitutive_apply(" not in line),
                "declared_increment_lines_present_once": {
                    e["line"]: source.count(e["line"]) for e in family.INCREMENT_SPELLING},
                "no_stale_magnetic_read_below_the_weld": {
                    name: below.count(f"weld.{name}") for name in family.H_TARGETS},
                "halo_is_a_recompute": "raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);" in below,
                "divide_is_the_scaled_algorithm": all(
                    line in family.div_coefficient_source() for line in MEASURED_DIVIDE_LINES),
                "increment_divide_is_not_componentwise_and_not_reciprocal":
                    " / divisor" not in below and "1.0f / divisor" not in source,
                "increment_multiply_is_the_fma_v1_lift":
                    family.mul_field_left_increment_source().count(_MUL_HEAD) == 1,
                "fma_v1_lift_matches_the_arm_block": (
                    complex_emitter._ARM_SOURCE[complex_emitter.EXPANSIONS["FMA_V1"]]  # noqa: SLF001
                    .count(_MUL_HEAD.replace("mul_field_left_increment(", "mul_field_left(")) == 1),
                "row0_is_cf_zero": "cf_store(increment, idx, cf_zero());" in below,
                "kernel_name_declared_once": source.count(f"void {family.KERNEL_NAME}(") == 1,
                "source_is_ascii": source.isascii() and encodes_ascii(source),
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "launch2": {
                    "curl_kernel_name": family.CURL_KERNEL_NAME,
                    "curl_kernel_is_the_certified_one":
                        f"void {family.CURL_KERNEL_NAME}(" in curl_source
                        and family.CURL_KERNEL_NAME in cyl.CERTIFIED_KERNELS,
                    "curl_source_sha256": hashlib.sha256(curl_source.encode("utf-8")).hexdigest(),
                    "launcher_hands_in_the_prefix":
                        "prefix=state[\"prefix\"]" in inspect.getsource(family.launch_curl),
                },
            }
            facts["passed"] = bool(
                facts["prelude_edits_invert_to_the_certified_text"]
                and facts["declared_edit_anchors_resolve_in_the_certified_text"]
                and facts["decode_is_carried_verbatim"] and facts["thread_preamble_is_gone"]
                and facts["three_calls_captured"] and facts["no_storing_call_survives"]
                and facts["certified_body_lines_survive"]
                and all(n == 1 for n in facts["declared_increment_lines_present_once"].values())
                and sum(facts["no_stale_magnetic_read_below_the_weld"].values()) == 0
                and facts["halo_is_a_recompute"] and facts["divide_is_the_scaled_algorithm"]
                and facts["increment_divide_is_not_componentwise_and_not_reciprocal"]
                and facts["increment_multiply_is_the_fma_v1_lift"]
                and facts["fma_v1_lift_matches_the_arm_block"] and facts["row0_is_cf_zero"]
                and facts["kernel_name_declared_once"] and facts["source_is_ascii"]
                and facts["launch2"]["curl_kernel_is_the_certified_one"]
                and facts["launch2"]["launcher_hands_in_the_prefix"])
            per_arm[arm_name] = facts
        licensed = self.kernel_source()
        out = {
            "per_arm": per_arm,
            "the_two_arms_emit_different_sources":
                family.kernel_source("NAIVE") != family.kernel_source("FMA_V1"),
            "licensed_arm": self.arm,
            "compile_options": list(family._COMPILE_OPTIONS),  # noqa: SLF001
            "source_sha256_all_arms": family.source_digest(),
            "mutation_needles_resolve_on_the_licensed_arm": mutation_needles_resolve(self, licensed),
            "byte_neutral_needle_resolves": licensed.count(BYTE_NEUTRAL["old"]) == 1,
        }
        out["passed"] = bool(
            all(f["passed"] for f in per_arm.values())
            and out["the_two_arms_emit_different_sources"]
            and out["compile_options"] == ["--fmad=false"]
            and out["mutation_needles_resolve_on_the_licensed_arm"]["passed"]
            and out["byte_neutral_needle_resolves"])
        return out

    def leg_refusal(self) -> Dict[str, Any]:
        from meep_gpu.fields import Fields  # noqa: PLC0415
        from meep_gpu.grid import Grid  # noqa: PLC0415
        from meep_gpu.pml import PML  # noqa: PLC0415
        from meep_gpu.sources import (ContinuousEnvelope, GaussianEnvelope,  # noqa: PLC0415
                                      VolumeSource)

        family = self.family
        checks: Dict[str, Any] = {}

        def dcyl(m: int = 1, complex_storage: bool = True, shape=(16, 1, 20), **kwargs):
            grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                        cylindrical=True, m=m, boundaries={"z": "metallic"}, courant=0.35,
                        xp=cp, **kwargs)
            fields = Fields(grid=grid, force_complex_fields=complex_storage)
            fields.enable_field_storage()
            fields.enable_pml_storage()
            return fields, PML(grid=grid, thickness={"x": (0, 3), "z": 3}), grid

        def verdict(name: str, expected: bool, thunk, names: Optional[str] = None,
                    licence: Any = "default", policy: Any = "default") -> None:
            try:
                fields, pml, grid, sources = thunk()
                covered, reason = family.covers_cylindrical_fused_hd_pair(
                    fields, pml, grid, sources,
                    self.license if licence == "default" else licence,
                    self.policy if policy == "default" else policy)
            except Exception as error:  # noqa: BLE001
                checks[name] = {"covered": False, "expected": expected,
                                "reason": f"construction refused: {error!r}"[:300],
                                "passed": expected is False,
                                "note": "refused by the engine before the predicate was asked"}
                return
            entry = {"covered": bool(covered), "expected": expected, "reason": reason}
            entry["passed"] = bool(covered) == expected
            if names is not None:
                entry["names_the_reason"] = names in reason
                entry["passed"] = entry["passed"] and entry["names_the_reason"]
            checks[name] = entry

        for m in (0, 1, -3):
            verdict(f"admits_a_plain_complex_dcyl_run_m{m}", True, lambda m=m: (*dcyl(m=m), ()))
        verdict("admits_the_accurate_origin_branch", True,
                lambda: (*dcyl(m=2, accurate_fields_near_cylorigin=True), ()))
        verdict("refuses_an_undeclared_source_set", False, lambda: (*dcyl(), None), "was not declared")

        def integrated():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        verdict("refuses_a_standing_integrated_electric_withdraw", False, integrated,
                "standing integrated")

        def plain_source():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
            return fields, pml, grid, [source]

        verdict("admits_a_non_integrated_electric_source", True, plain_source)

        def magnetic():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Hz", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        verdict("admits_an_integrated_magnetic_source", True, magnetic)
        verdict("refuses_real_storage", False, lambda: (*dcyl(m=0, complex_storage=False), ()),
                "real float32")
        verdict("refuses_without_a_licence", False, lambda: (*dcyl(), ()), "no expansion licence",
                licence=None)
        other = "flush" if self.policy == "keep" else "keep"
        verdict("refuses_a_licence_cut_under_the_other_policy", False, lambda: (*dcyl(), ()),
                "cut under", policy=other)

        def no_pml():
            fields, _pml, grid = dcyl()
            return fields, None, grid, ()

        verdict("refuses_a_run_with_no_pml", False, no_pml, "no active PML")

        def cartesian():
            grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 1.6), courant=0.35, xp=cp,
                        boundaries=("periodic", "periodic", "periodic"))
            fields = Fields(grid=grid, force_complex_fields=True)
            fields.enable_field_storage()
            fields.enable_pml_storage()
            return fields, PML(grid=grid, thickness=2), grid, ()

        verdict("refuses_a_cartesian_grid", False, cartesian, "cylindrical")
        verdict("refuses_a_bloch_phase", False, lambda: (*dcyl(k_point=(0.0, 0.0, 0.1)), ()))
        verdict("refuses_a_single_radial_row", False, lambda: (*dcyl(shape=(1, 1, 20)), ()))

        class _Beta:
            def __init__(self, grid):
                self._grid = grid

            def __getattr__(self, item):
                return getattr(self._grid, item)

            beta = 0.3

        def beta():
            fields, pml, grid = dcyl()
            return fields, pml, _Beta(grid), ()

        verdict("refuses_special_kz", False, beta, "beta")

        def missing_volume():
            fields, pml, grid = dcyl()

            class _NoHz:
                def __getattr__(self, item):
                    return getattr(fields, item)

                Hz = None

            return _NoHz(), pml, grid, ()

        verdict("refuses_an_unallocated_rotated_volume", False, missing_volume, "Hz")
        return {"checks": checks, "denominator": len(checks),
                "passed": all(entry["passed"] for entry in checks.values())}

    def leg_spelling(self) -> Dict[str, Any]:
        return leg_spelling_complex(self)

    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import registry  # noqa: PLC0415
        from meep_gpu.cuda_kernels.fused_pairs import CudaFusedPairPlan  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context):
            covered, reason = family.covers_cylindrical_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources,
                context.license_for(registry.LICENSE_COMPLEX), context.subnormal_policy)
            return Coverage(bool(covered), (str(reason),))

        def plan(context):
            held: Dict[str, Any] = {}

            def resolve(ctx):
                if "state" not in held:
                    licence = ctx.license_for(registry.LICENSE_COMPLEX) or {}
                    held["state"] = family.resolve(ctx.fields, ctx.grid, ctx.pml, licence.get("arm"))
                return {"state": held["state"]}

            def launch(fields, arguments):
                return family.run_cylindrical_fused_hd_pair(
                    fields, fields.grid, context.pml, arguments["state"]["arm"],
                    sources=context.sources,
                    license=context.license_for(registry.LICENSE_COMPLEX),
                    subnormal_policy=context.subnormal_policy, state=arguments["state"])

            return CudaFusedPairPlan(family=family.FAMILY, label="cylindrical complex fused H/D pair",
                                     kernel_label=family.KERNEL_NAME,
                                     replaces=tuple(family.REPLACES),
                                     slots=("update_H", "step_D"), context=context,
                                     resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": "cylindrical_fused_hd_pair"}


def make_adapter(policy: Optional[str], licence: Optional[Dict[str, Any]]) -> ComplexAdapter:
    return ComplexAdapter(policy, licence)


def runtime_reasons() -> List[str]:
    return common.hd_gate.runtime_reasons()


def main(argv=None) -> int:
    return common.run_gate(make_adapter, os.path.abspath(__file__), FAMILY_NAME, BLOCK_KEY,
                           __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
