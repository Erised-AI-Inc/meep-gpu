"""Gate: ``meep_gpu.cuda_kernels.complex_beta_fused_hd_pair`` -- the COMPLEX beta H->D weld.

ONE CUDA kernel welding ``stepping.update_H`` into the special_kz ``stepping.step_D``
under COMPLEX64 storage on a ``grid.beta != 0`` run, measured BIT-IDENTICAL as uint32
words per COMPLETE DRIVER STEP against FOUR independent arrangements of the seam,
under both float32 subnormal policies, on synthetic fixtures and on the four lifted
corpus rows of its cell.

THE CELL, AND WHY IT IS ONE TEXT
=============================================================================
``results/fusion_matrix_cuda_2026-09-07_wired/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(cuda_complex_beta/complex beta -> cuda_complex_beta/complex beta)`` -- **4
  instances**: ``tests:TestSpecialKz.test_special_kz``,
  ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` and the two
  ``tests:TestEigCoeffs.test_binary_grating_special_kz_*`` legs.

Three of the four are folded (Mirror(Y), periodic termination) and one is not, and
ONE TEXT serves all four: ``complex_beta_kernels``' device source is the FOLDED
complex curl with the beta insert, so a fold is a runtime boundary code here. The
released complex H->D product needed two variants; this one needs none, and the
fixtures below still carry both folded and unfolded grids because the RUNTIME CODES
differ even where the text does not.

THE FOUR REFERENCES, and what each one rules out
=============================================================================
``array``               ``stepping``'s own ten passes. THE ORACLE.
``singles``             the certified ``step_D_pml_complex_beta`` and the certified
                        complex constitutive kernel at their own slots, everything
                        else on the array path. Rules out a divergence that is the
                        certified halves' rather than the weld's.
``composition_today``   what the composer BUILDS on these rows today: the released
                        ``cuda_complex_beta_fused_magnetic_pair`` at ``step_B`` and
                        ``cuda_complex_beta_fused_electric_pair`` at ``step_D``.
                        Rules out "identical to a composition nobody runs".
``unfused``             all four slot-path slots as certified singles.
``weld``                the subject.

Plus ``weld_composed`` on the launch-structure leg only.

WHAT THIS SEAM ADDS THAT NO SIBLING H->D GATE CAN MEASURE
=============================================================================
THE BETA PARTNER. ``stepping._special_kz_beta_term`` consumes the SAME-CELL magnetic
snapshot, which the array path takes AFTER ``update_H``; welded, that snapshot must be
this launch's own ``own_h[...]`` register. ``beta_partner_reads_stale_h`` is the armed
defect and it is one sub-step behind on exactly two of the six curl terms.

WHAT THE ARITHMETIC LEGS ADD THAT A WHOLE-STEP IDENTITY CANNOT
=============================================================================
CuPy's ``complex64 * float32`` is the four-product form CONTRACTED by NVRTC: the
uncontracted transcription differs on the SIGN OF A FLUSHED ZERO -- 6 words of
4,005,000 on (flush, edge) in ``lanes/cyl_round/cupy_probe``, all on tiny normals at a
row the multiply underflows. A RANDOM battery does not reach that class. The
``spelling`` leg therefore runs the family's own multiply against ``cupy.multiply`` on
a PLANTED row-0 tiny-normal fixture beside the ordinary classes, and the mutation
battery arms the naive four-product form as a device defect on that same plant.

**THIS FAMILY PERFORMS NO DIVISION** and the ``spelling`` leg records that as a
measured absence rather than omitting it: the reciprocal the recurrence needs is
``sinv``, computed host-side by ``PML``. CuPy's ``complex64 / float32`` is the SCALED
complex/complex algorithm with the zero-valued terms kept -- NOT numpy's reciprocal
multiply, and NOT the real family's true divide -- and it has no site here. The
absence is asserted against the emitted text, with a planted divide as the control.

THE BETA COEFFICIENT'S SIGNED ZERO IS PASSED THROUGH AND IS NOT CLAIMED TO MATTER.
``complex_beta_kernels``' own device leg measured ``synthesise_the_zero_real_word``
UNCAUGHT 0/12 under BOTH policies, because ``mul_imag_coefficient_left`` consumes the
imaginary word alone. This gate re-arms the same needle on the H->D seam and RECORDS
whatever it measures; a null here is a null, not a defence.

NOT WIRED, AND THE ARBITRATION LEG SAYS SO IN BOTH ARRANGEMENTS
=============================================================================
The family is not in ``fused_pairs.FUSED_PRODUCTS`` / ``registry`` / ``arms`` yet.
The ``arbitration`` leg measures the composer AS SHIPPED and again WITH the wiring
rows patched in-process, and it ASSERTS the substantive claims only -- refused BY
NAME, the refusal names ``INSTALLABLE = False``, the selection unchanged, both
released neighbours keeping their slots -- while RECORDING the positional ones, which
a wiring round inverts by design.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import beta_hd_gate_common as kit  # noqa: E402

cp = kit.cp
log = kit.log
probe = kit.probe
differing = kit.differing
to_host = kit.to_host

FAMILY_NAME = "cuda_complex_beta_fused_hd_pair"
BLOCK_KEY = "cuda_complex_beta_fused_hd_pair_gate"
GATE_FILE = "gate_complex_beta_fused_hd_pair.py"

#: The env var the parent hands the child its expansion licence through. A child that
#: had to re-read the probe would be re-deciding the arm the parent already decided.
LICENCE_ENV = "MEEP_GPU_CXBHD_GATE_LICENCE"

#: The board cell, in the census's ``(update_H arm, step_D arm)`` spelling.
CELL_ARMS: Tuple[str, str] = ("cuda_complex_beta/complex beta",
                              "cuda_complex_beta/complex beta")

#: The betas of the four corpus rows this cell holds, from
#: ``complex_beta_kernels``' own admission: ``test_special_kz`` -0.3907,
#: ``test_eigsrc_kz_0_complex`` 0.2, and the two binary-grating legs -0.685 / -0.912.
#: Fixtured across BOTH SIGNS, because the coefficient's sign is what decides whether
#: the real word Python's complex multiply leaves is ``-0.0`` or ``+0.0``.
CORPUS_BETAS: Tuple[float, ...] = (-0.3907, 0.2, -0.685, -0.912)

#: THE PLANT. ``complex64 * float32`` on this device is the CONTRACTED four-product
#: form, and the uncontracted transcription differs only where a cross term
#: UNDERFLOWS: the sign of a flushed zero. The class is ``|re|`` and ``|im|`` tiny
#: NORMALS with both parts negative, so ``z.im * c`` with a small coefficient falls
#: into the subnormal range and ``flush`` zeroes it while ``keep`` does not. Planted on
#: row 0 of EVERY stored volume, because ``mul_field_left`` -- the helper whose
#: spelling the defect changes -- is called by ``pml_apply`` on ``D``, ``fu_D`` and the
#: curl, NOT on ``H`` or ``B``; a plant on the magnetic volumes alone leaves the
#: operands order 1 and nothing underflows.
PLANT_MAGNITUDE = np.float32(1.5e-38)

#: The synthetic fixtures. Both fold terminations, both full-count parities, a fully
#: periodic grid (the ONLY specialisation on which the shifted tap's PERIODIC WRAP is
#: observable), a live Bloch phase on an UNFOLDED axis, and both signs of beta.
#:
#: A PHASE ON A FOLDED AXIS IS NOT FIXTURED. ``complex_folded_kernels._fold_reasons``
#: refuses that configuration by name and ``complex_beta_kernels`` inherits the
#: refusal; the folded fixtures below phase unfolded axes only, which is what the four
#: corpus rows carry. The invariant z axis is forced periodic by ``Grid`` at
#: ``dimensions=2``, so no fixture walls, folds or phases it.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "periodic_k0", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[0],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "periodic_kx", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[1],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.3, 0.0, 0.0)},
    # THE BRILLOUIN EDGE, where ``Grid._axis_bloch_phase`` returns EXACTLY -1+0j: the
    # rotation is then a sign flip with an exact zero imaginary part, which is where a
    # dropped zero cross term hides.
    {"label": "brillouin_edge_kx", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[2],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.5, 0.0, 0.0)},
    {"label": "walls_xy", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[3],
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "wall_x_only", "cell": (1.6, 1.6, 0.0), "beta": 0.3321611318837033,
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    # ---- the folded grids: three of this cell's four rows are Mirror(Y) periodic ----
    {"label": "fold_y_even_k0", "cell": (1.6, 2.0, 0.0), "beta": CORPUS_BETAS[0],
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_odd_kx", "cell": (1.6, 2.0, 0.0), "beta": CORPUS_BETAS[2],
     "boundaries": None, "symmetry": (("Y", -1),), "k_point": (0.3, 0.0, 0.0)},
    {"label": "fold_y_odd_count", "cell": (1.6, 2.1, 0.0), "beta": CORPUS_BETAS[1],
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_metallic_termination", "cell": (1.6, 2.0, 0.0),
     "beta": CORPUS_BETAS[3], "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
)

#: The fixtures the mutations are scored on: a live Bloch phase, a wall on each
#: in-plane axis, both fold terminations, and both signs of beta.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("periodic_kx", "walls_xy", "fold_y_odd_kx",
                                         "fold_y_metallic_termination")

REDUCED_LABELS: Tuple[str, ...] = ("periodic_kx", "walls_xy", "brillouin_edge_kx",
                                   "fold_y_odd_kx", "fold_y_metallic_termination")


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # ---- THE BETA CLAUSES: this seam's own ----
    "beta_partner_reads_stale_h": {
        "old": "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
        "new": ("        curl = cf_sub(curl, mul_imag_coefficient_left("
                "bp, cf_load(g1, idx)));"),
        "why": ("THE DEFECT THIS WELD CREATES AND NO SIBLING H->D SEAM HAS. The beta "
                "partner is the SAME-CELL magnetic snapshot, which the array path "
                "takes AFTER update_H; this reverts it to the PRE-LAUNCH volume, "
                "leaving the finite-difference stencil correct and putting exactly ONE "
                "of the six curl terms one sub-step behind. Every magnitude stays "
                "plausible and the run converges"),
        "live_on": "every fixture (beta is nonzero on all of them)",
    },
    "drop_the_beta_term": {
        "old": "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n",
        "new": "        // beta term removed\n",
        "why": ("the analytic d/dz factor is dropped from target 0 entirely -- a "
                "beta = 0 run on one component, which is a DIFFERENT physical problem "
                "that still converges"),
        "live_on": "every fixture",
    },
    "swap_the_beta_signs": {
        "old": "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
        "new": "        curl = cf_sub(curl, mul_imag_coefficient_left(bm, f_2));",
        "why": ("the +1 call site takes the -1 coefficient (stepping.py:443 against "  # stepping.py live lines for the frozen device-text citation(s) in this string: 443->472
                ":445). Complex negation is exact, so the two words differ only in "
                "sign -- and the cross-polarisation coupling reverses"),
        "live_on": "every fixture",
    },
    "beta_added_not_subtracted": {
        "old": "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
        "new": "        curl = cf_add(curl, mul_imag_coefficient_left(bp, f_2));",
        "why": ("stepping.py:784 returns -(c * partner) and :443 ADDS it; this drops "  # stepping.py live lines for the frozen device-text citation(s) in this string: 784->811, 443->472
                "the negation. IEEE defines subtraction as addition of the negation "
                "and complex add is plane-wise, so the single subtract in the "
                "certified text is the same bits -- and this control is what shows "
                "that reading is load-bearing"),
        "live_on": "every fixture",
    },
    "beta_scaled_by_dtdx": {
        "old": "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
        "new": ("        curl = cf_sub(curl, mul_coefficient_left(dtdx, "
                "mul_imag_coefficient_left(bp, f_2)));"),
        "why": ("the beta term is an ANALYTIC derivative and carries NO dtdx "
                "(stepping.py:758-762, :797). Scaling it by the finite-difference "
                "factor rescales the TE/TM coupling by the Courant number"),
        "live_on": "every fixture",
    },
    "beta_on_the_third_component": {
        # TARGET 2 gets a beta increment it must not have. The anchor is target 2's
        # curl line together with its FIRST mask, which is unique: targets 0 and 1
        # carry their beta line (and its comment block) between those two statements.
        "old": ("        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), "
                "cf_sub(f_2, ss)));\n"
                "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();"),
        "new": ("        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), "
                "cf_sub(f_2, ss)));\n"
                "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n"
                "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();"),
        "why": ("a beta increment on the component MEEP gives none (step_db.cpp:"
                "148-176 runs cc over d_c in {X, Y} only). The component PATTERN is "
                "part of the transcription: targets 0 and 1 must move and target 2 "
                "must not"),
        "live_on": "every fixture",
    },
    "beta_after_the_mask": {
        # THE COMPOSITION NEEDLE the certified single's own gate carries, re-armed on
        # the weld: the insert moved BELOW the ownership mask, which a FOLD makes
        # reach twice as far. Invisible in the interior; live on the mask planes only.
        "old": ("        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n"
                "        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();"),
        "new": "        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();",
        "also": ((("        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) "
                   "curl = cf_zero();\n"
                   "        pml_apply(f0, u0,"),
                  ("        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) "
                   "curl = cf_zero();\n"
                   "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n"
                   "        pml_apply(f0, u0,")),),
        "why": ("the beta increment moves from ABOVE the ownership mask to BELOW it, "
                "so a masked plane keeps a live increment the array path zeroes "
                "(stepping.py:472 sits after :458 and before :479). Invisible in the "
                "interior, which is why it is scored on fixtures carrying a wall or a "
                "fold on a beta target"),
        "live_on": "walls_xy, fold_y_odd_kx, fold_y_metallic_termination",
    },
    # ---- THE WELD CLAUSES ----
    "foreign_tap_reads_stale_H": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return cf_load(comp == 0 ? weld.Hx\n"
                "                        : comp == 1 ? weld.Hy : weld.Hz,\n"
                "                        idx - stride);"),
        "why": ("the foreign tap loads the PRE-LAUNCH magnetic field instead of "
                "recomputing update_H there -- which is what an unfused step_D would "
                "read and is exactly one sub-step behind"),
        "live_on": "every fixture",
    },
    "foreign_tap_reads_B": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return cf_load(comp == 0 ? weld.Bx\n"
                "                        : comp == 1 ? weld.By : weld.Bz,\n"
                "                        idx - stride);"),
        "why": ("the tap reads the constitutive SOURCE rather than its result. Under "
                "mu = 1 outside the absorber H and B are close, so this is the "
                "plausible wrong answer rather than an obvious one"),
        "live_on": "every fixture",
    },
    "own_cell_reads_stale_H": {
        "old": "        cf f_1 = own_h[2];",
        "new": "        cf f_1 = cf_load(g2, idx);",
        "why": ("the thread's OWN cell load reverts to the pre-launch volume, so "
                "target 0's curl differences an H one sub-step behind on one of its "
                "two terms"),
        "live_on": "every fixture",
    },
    "f_w_H_written_in_place": {
        "old": "    *fw_out = src;",
        "new": "    cf_store(const_cast<float*>(fw), idx, src);",
        "why": ("the split-field history is stored IN PLACE, so a foreign recompute "
                "reading fw at a cell another block already wrote gets B where its "
                "recurrence needs B_prev. RACY BY CONSTRUCTION"),
        "live_on": "every fixture (schedule-dependent)",
        "predicted_null": ("schedule-dependent by construction; the deterministic twin "
                           "foreign_history_reads_the_post_store_value carries the "
                           "same arithmetic consequence and MUST be caught"),
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    cf prev = cf_load(fw, idx);",
        "new": "    cf prev = src;",
        "why": ("THE DETERMINISTIC TWIN of the in-place store: the recurrence reads "
                "the value the in-place arrangement would have handed it, with no "
                "race to depend on"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    "the_wrapped_lane_is_not_phased": {
        # ANCHORED ON THE RECOMPUTE'S OWN WRAP, not on the bare phase line: the
        # certified ``cshift_up`` rides along in the lifted prelude and spells the
        # same line, so a one-line needle finds more than one site.
        "old": ("    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);\n"
                "    if (ph) w = rotate_field_left(w, phase);"),
        "new": ("    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);\n"
                "    if (ph) w = w;"),
        "why": ("the Bloch factor is dropped from the wrapped lane, which is the "
                "band-structure error: every magnitude stays plausible and only the "
                "phase moves"),
        "live_on": "periodic_kx, brillouin_edge_kx, fold_y_odd_kx",
    },
    "the_phase_is_applied_to_the_near_neighbour_too": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) {\n"
                "        cf n = resolve_H(comp, idx - stride, weld);\n"
                "        if (ph) n = rotate_field_left(n, phase);\n"
                "        return n;\n    }"),
        "why": ("the phase is applied to EVERY lane rather than to the wrapped one "
                "only -- ``_apply_bloch_phase`` multiplies the single plane "
                "_face(axis, -1) of the rolled buffer and nothing else"),
        "live_on": "periodic_kx, brillouin_edge_kx, fold_y_odd_kx",
    },
    "the_periodic_wrap_reads_the_near_row": {
        "old": "    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);",
        "new": "    cf w = resolve_H(comp, idx + stride, weld);",
        "why": ("the periodic wrap reads the row ABOVE instead of the far row. Unlike "
                "the metallic ghost this IS observable: on a periodic axis the "
                "ownership mask does not zero the curl the wrapped tap feeds"),
        "live_on": "every fixture (their x axis wraps)",
    },
    "the_ghost_serves_the_clamped_cells_constitutive": {
        # ANCHORED ON THE RECOMPUTE'S OWN GHOST, with the wrap line carried along:
        # the certified ``cshift_up`` and ``cshift_dn`` ride along in the lifted
        # prelude and spell the ghost line identically, so a one-line needle finds
        # THREE sites and lands on the wrong one. Measured on this gate's host smoke.
        "old": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) "
                "return cf_zero();\n"
                "    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);"),
        "new": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) "
                "return resolve_H(comp, idx, weld);\n"
                "    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);"),
        "why": ("the metallic and mirror ghosts serve update_H's value at the clamped "
                "cell instead of the exact cf_zero(). A DERIVED NULL for BOTH codes: "
                "``ghost_observability`` reads off the emitted text that every "
                "redirected tap is masked at exactly the plane its constant ghost "
                "fires on, under BC_METALLIC and BC_MIRROR_PERIODIC alike, so nothing "
                "either ghost serves can reach step_D's output. The control that MUST "
                "bite is the PERIODIC sibling above, whose branch is the one the mask "
                "does not cover"),
        "live_on": "no fixture -- derived null; see ghost_observability",
        "predicted_null": ("derived from the emitted text by the ghost_observability "
                           "leg, for BOTH constant-ghost codes: the set of (tap, "
                           "plane) pairs a ghost fires on and the set the ownership "
                           "mask zeroes COINCIDE. Measured uncaught 0/8 on this gate's "
                           "first mutation smoke, which is what sent the derivation "
                           "back to cover the mirror arm as well as the metallic one"),
    },
    "accumulations_regrouped": {
        "old": ("    a = cf_add(a, mul_coefficient_left(kps, src));\n"
                "    a = cf_sub(a, mul_coefficient_left(kms, prev));"),
        "new": ("    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src),\n"
                "                         mul_coefficient_left(kms, prev)));"),
        "why": ("the two SEPARATE accumulations become one expression, which is a "
                "different float32 number because addition is not associative. MEEP's "
                "step_update_EDHB accumulates left to right and the certified body "
                "transcribes that; flattening it reads as a slightly weaker absorber"),
        "live_on": "every fixture with an active absorber",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n\n    // NO ghost rule"),
        "new": ("    int k = 0;\n    int j = 0;\n    int i = 0;\n\n"
                "    // NO ghost rule"),
        "why": ("the recomputed cell's absorber coefficients are taken from the origin "
                "instead of from the recomputed cell -- the half-cell class of error, "
                "converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "naive_four_product_multiply": {
        "old": ("__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"
                "    cf o;\n"
                "    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);\n"
                "    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);\n"),
        "new": ("__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"
                "    cf o;\n"
                "    o.re = (z.re * c) - (z.im * 0.0f);\n"
                "    o.im = (z.re * 0.0f) + (z.im * c);\n"),
        "why": ("CuPy's complex64 * float32 is the four-product form CONTRACTED by "
                "NVRTC; the uncontracted transcription differs on the SIGN OF A "
                "FLUSHED ZERO. Measured 6 words of 4,005,000 on (flush, edge) by "
                "lanes/cyl_round/cupy_probe -- a class a random battery does not "
                "reach, which is why the planted_row0 value class exists"),
        "live_on": "every policy -- see the refutation below",
        "arm_only": "FMA_V1",
        # ARMED UNDER BOTH POLICIES, AND THAT IS A MEASURED REFUTATION OF THE
        # INHERITED PREDICTION. The released complex H->D gate declares this defect
        # NULL under ``keep``, reasoning that "under keep nothing flushes, so the two
        # forms agree bit for bit: fma(a, b, +-0) rounds a*b once, zero signs
        # included". This gate's first mutation smoke (2026-09-07, GPU 3, reduced
        # fixtures) CAUGHT it under ``keep``. The reasoning is incomplete rather than
        # wrong: the addend is an exact +-0 either way, so the two forms differ
        # wherever ``z.re * c`` itself lands on an exact zero and the two spellings
        # disagree about ITS SIGN -- which ``keep`` reaches too, because a product of
        # two band-class operands underflows past the smallest SUBNORMAL and is zero
        # under either policy. So the needle is required to bite under both, which is
        # the stronger claim, and the record carries the refutation rather than the
        # inherited null.
    },
    "fmad_default_build": {
        "old": None,
        "new": None,
        "options": (),
        "why": ("the same source built WITHOUT --fmad=false, so NVRTC may contract "
                "the accidental multiply-adds the transcription does not spell"),
        "live_on": "every fixture",
        "null_under": ("keep", "flush"),
        "null_reason": (
            "PREDICTED NULL ON THE COMPLEX PRIMARY, and the prediction is a prior "
            "MEASUREMENT rather than this run's excuse: lanes/cyl_round/cupy_probe "
            "measured the same source built with and without --fmad=false at 0 of "
            "14,387,504 words under both policies on the complex spelling. THE CAUSE "
            "is that with the arm's multiplies spelled as explicit __fmaf_rn there is "
            "no contractible multiply-add left for the flag to remove -- and that "
            "cause is MEASURED here rather than argued: the fmad_attribution sub-leg "
            "builds the UNCONTRACTED arm (NAIVE) both ways and requires those two to "
            "differ. The control stays armed because it must still COMPILE and RUN"),
    },
}

#: The one armed edit required NOT to diverge: the own-cell register replaced by a
#: RELOAD of the scratch word this thread has just stored -- the same value by a
#: different route.
BYTE_NEUTRAL: Dict[str, str] = {
    "old": "        cf f_1 = own_h[2];",
    "new": "        cf f_1 = cf_load(Hz_out, idx);",
    "why": ("the own-cell register is replaced by a RELOAD of the scratch word pair "
            "this thread has just stored -- the same value by a different route. Two "
            "float32 words stored to global memory and loaded back are the identity "
            "on the bits, so an edit that diverged here would mean the harness rather "
            "than the weld decides the answer"),
}

HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swap_constitutive_sublattice",
                                   "unconjugated_backward_phase",
                                   "swap_beta_coefficient_words",
                                   "synthesise_the_zero_real_word")


# ---------------------------------------------------------------------------
# The multiply spellings, on standalone kernels against CuPy's own ufuncs
# ---------------------------------------------------------------------------

_SPELLING_PRELUDE = r'''
typedef struct { float re; float im; } cf;
__device__ __forceinline__ cf cf_load(const float* g, int idx) {
    cf z; z.re = g[2 * idx]; z.im = g[2 * idx + 1]; return z;
}
__device__ __forceinline__ void cf_store(float* g, int idx, cf z) {
    g[2 * idx] = z.re; g[2 * idx + 1] = z.im;
}
'''

#: The four candidate multiply spellings. ``family_fma_v1`` is the one the emitted
#: kernel calls (through ``complex_emitter._ARM_SOURCE[FMA_V1].mul_field_left``, lifted
#: whole); the rest are the controls the probe measured biting.
_MULTIPLY_ARMS: Dict[str, str] = {
    "family_fma_v1": (
        "    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);\n"
        "    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);\n"),
    "naive_four_product": (
        "    o.re = (z.re * c) - (z.im * 0.0f);\n"
        "    o.im = (z.re * 0.0f) + (z.im * c);\n"),
    "componentwise": (
        "    o.re = z.re * c;\n"
        "    o.im = z.im * c;\n"),
    "coefficient_left": (
        "    o.re = __fmaf_rn(c, z.re, (0.0f * z.im) * -1.0f);\n"
        "    o.im = __fmaf_rn(c, 0.0f, 0.0f * z.re);\n"),
}


def _spelling_kernel(body: str, name: str):
    source = (_SPELLING_PRELUDE
              + "__device__ __forceinline__ cf mul(cf z, float c) {\n    cf o;\n"
              + body + "    return o;\n}\n"
              + f'extern "C" __global__ void {name}(\n'
              "    float* __restrict__ out, const float* __restrict__ z,\n"
              "    const float* __restrict__ c, int n\n) {\n"
              "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
              "    if (idx >= n) return;\n"
              "    cf_store(out, idx, mul(cf_load(z, idx), c[idx]));\n}\n")
    return cp.RawKernel(source, name, options=("--fmad=false",)), source


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class ComplexBetaAdapter(kit.Adapter):
    """The COMPLEX beta H->D weld's answers to the shared legs' questions."""

    complex_storage = True
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band", "planted_row0")
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    #: (update_H's arm, step_D's arm). Both are this family's own ``complex beta``
    #: arm: ``covers_complex_beta_fused_hd_pair`` opens with
    #: ``complex_beta_kernels.covers_complex_beta_constitutive(..., "H")`` and
    #: continues into ``covers_complex_beta_curl(..., "step_D")``, which ARE that
    #: arm's two predicates. The board's cell is the same pair of labels.
    wiring_arms_row = ("complex beta", "complex beta")
    wiring_module = "complex_beta_fused_hd_pair"
    released_neighbours = ("cuda_complex_beta_fused_magnetic_pair",
                           "cuda_complex_beta_fused_electric_pair")
    env_stem = "MEEP_GPU_CXBHD_GATE"

    def __init__(self) -> None:
        from meep_gpu.cuda_kernels import complex_beta_fused_hd_pair as family  # noqa: PLC0415
        self.family = family

    # -- fixtures -------------------------------------------------------------
    def plant(self, fields, grid, value_class: str) -> Dict[str, Any]:
        """Plant the tiny-normal row-0 class on EVERY stored volume.

        WHERE THE PLANT HAS TO GO IS A MEASUREMENT, not a guess, and the released
        complex H->D gate got it wrong on its first cut: planting only the MAGNETIC
        volumes left ``naive_four_product_multiply`` uncaught under ``flush``, because
        ``mul_field_left`` is called by ``pml_apply`` on ``D``, ``fu_D`` and the curl,
        NOT on ``H`` or ``B``. The plant also varies the SIGNS across the plane: the
        defect is the sign of a FLUSHED zero, so both parts must appear with both signs
        for the class to be reached at all.
        """
        if value_class != "planted_row0":
            return {"planted": False}
        shape = tuple(int(n) for n in grid.shape)
        magnitude = float(PLANT_MAGNITUDE)
        j = np.arange(shape[1])[:, None]
        k = np.arange(shape[2])[None, :]
        real_sign = np.where((j + k) % 2 == 0, -1.0, 1.0)
        imag_sign = np.where(j % 2 == 0, -1.0, 1.0) * np.ones_like(k)
        plane = np.empty(shape[1:], dtype=np.complex64)
        plane.real = (real_sign * magnitude).astype(np.float32)
        plane.imag = (imag_sign * magnitude).astype(np.float32)
        planted: List[str] = []
        for name in kit.STATE_NAMES:
            array = getattr(fields, name, None)
            if array is None:
                continue
            host = to_host(array).copy()
            host[0, :, :] = plane
            array[...] = cp.asarray(np.ascontiguousarray(host))
            planted.append(name)
        return {"planted": True, "volumes": planted,
                "cells": len(planted) * int(np.prod(shape[1:])),
                "magnitude": magnitude,
                "why": ("row 0 of EVERY stored volume set to tiny NORMALS of all four "
                        "sign combinations. Multiplied by any coefficient below 1 -- "
                        "every sinv, and kms inside the absorber -- the product "
                        "underflows, and under flush the sign of the flushed zero is "
                        "what separates the naive four-product multiply from the "
                        "contracted one. A random battery reaches this class with "
                        "probability ~0")}

    # -- the product ----------------------------------------------------------
    def predicate(self, fields, pml, grid, sources) -> Tuple[bool, str]:
        return self.family.covers_complex_beta_fused_hd_pair(
            fields, pml, grid, sources, self.license, self.policy)

    def resolve(self, fields, grid, pml, dtdx: float, **overrides) -> Dict[str, Any]:
        return self.family.resolve(fields, grid, pml, self.arm, dtdx=dtdx, **overrides)

    def assert_disjoint(self, fields, state) -> int:
        return self.family.assert_bindings_are_disjoint(fields, state)

    def launch(self, fields, state, kernel=None, threads: Optional[int] = None):
        return self.family.launch_complex_beta_fused_hd_pair(
            fields, state, kernel,
            self.family._FUSED_THREADS if threads is None else int(threads))  # noqa: SLF001

    def rotate(self, fields, state) -> None:
        state["scratch"] = self.family.rotate_into_fields(fields, state["scratch"])

    def boundary_codes(self, grid):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        return complex_folded_kernels.folded_complex_boundary_codes(grid)

    def kernel_source(self, source: Optional[str] = None) -> str:
        return self.family.kernel_source(self.arm) if source is None else source

    def compile(self, source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
        kernel = cp.RawKernel(
            self.kernel_source(source), self.kernel_name(),
            options=self.compile_options() if options is None else tuple(options))
        kernel.compile()
        return kernel

    def attribution_source(self) -> Optional[str]:
        """The UNCONTRACTED arm's text.

        The shipped arm spells every fusion as an explicit ``__fmaf_rn`` and leaves
        ``--fmad=false`` nothing to remove, so an attribution run on the shipped text
        would measure the harness rather than the spelling. NAIVE is the arm that
        carries contractible multiply-adds.
        """
        return self.family.kernel_source("NAIVE")

    def attribution_arm_name(self) -> str:
        return "NAIVE"

    # -- the references -------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx: float) -> Dict[str, Callable[[], None]]:
        """The certified single-slot launchers for a COMPLEX beta run.

        The two CURL slots go to ``complex_beta_kernels``' certified beta pair, and
        the two CONSTITUTIVE slots to the certified complex pair -- which is what
        ``covers_complex_beta_constitutive`` admits and what the census records at
        ``update_H``/``update_E`` on all four corpus rows, because that family ships no
        constitutive kernel of its own.
        """
        from meep_gpu.cuda_kernels import (complex_beta_kernels,  # noqa: PLC0415
                                           complex_emitter, complex_pml_kernels)

        constitutive = {side: complex_pml_kernels.complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER[side]) for side in ("H", "E")}

        def curl(sub_step: str):
            return lambda: complex_beta_kernels.step_complex_beta(
                sub_step, fields, self.arm, grid=grid, pml=pml, dtdx=dtdx)

        return {
            "step_B": curl("step_B"),
            "step_D": curl("step_D"),
            "update_H": lambda: complex_pml_kernels.update_fused_pml_complex(
                "H", fields, self.arm, tables=constitutive["H"]),
            "update_E": lambda: complex_pml_kernels.update_fused_pml_complex(
                "E", fields, self.arm, tables=constitutive["E"]),
        }

    def released_pairs(self, fields, grid, pml, dtdx: float):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            complex_beta_fused_electric_pair as electric,
            complex_beta_fused_magnetic_pair as magnetic)

        def run_magnetic():
            return magnetic.run_complex_beta_fused_magnetic_pair(
                fields, grid, pml, dtdx, self.arm, sources=(),
                license=self.license, subnormal_policy=self.policy)

        def run_electric():
            return electric.run_complex_beta_fused_electric_pair(
                fields, grid, pml, dtdx, self.arm, sources=(),
                license=self.license, subnormal_policy=self.policy)

        return magnetic, electric, run_magnetic, run_electric

    def composer_licenses(self) -> Optional[Dict[str, Any]]:
        return {"complex": self.license} if self.license is not None else None

    def child_env(self) -> Dict[str, str]:
        """The expansion licence, handed down rather than re-derived.

        THE FULL VERDICT, not the arm alone: every complex predicate checks the
        verdict's shape, its refusal list, its basis and the policy it was cut under,
        and a child that re-read the probe would be re-deciding the arm the parent
        already decided -- with nothing comparing the two.
        """
        return {LICENCE_ENV: json.dumps(self.license, default=str)}

    # -- the host legs --------------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        """Is every welded byte the certified family's, and does the lift INVERT?

        TWO TEXTS (one per arm) AND TWO BYTE EQUALITIES. The equalities are the
        strongest thing this leg has and they are NEW measurements rather than
        restatements: this family's curl transform applied to the CERTIFIED PLAIN
        complex ``step_D`` text returns exactly what
        ``complex_fused_hd_pair.curl_pieces("plain", arm)`` returns, and applied to the
        CERTIFIED FOLDED text exactly what its ``"folded"`` variant returns -- byte for
        byte, on BOTH arms. The beta weld is therefore the RELEASED, GATED transform
        plus the beta insert and the beta-partner assertion.
        """
        from meep_gpu.cuda_kernels import (complex_emitter,  # noqa: PLC0415
                                           complex_folded_kernels,
                                           complex_fused_hd_pair)

        family = self.family
        rows: List[Dict[str, Any]] = []
        equalities: List[Dict[str, Any]] = []
        for arm_name in sorted(complex_emitter.EXPANSIONS):
            source = family.kernel_source(arm_name)
            prelude = family.constitutive_prelude(arm_name)
            certified_prelude, _ = family._curl_source(arm_name).split(  # noqa: SLF001
                '\nextern "C" __global__ void ', 1)
            inverted = prelude
            for old, new in complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS:
                inverted = inverted.replace(new, old, 1)
            arm_block = complex_emitter._ARM_SOURCE[  # noqa: SLF001
                complex_emitter.EXPANSIONS[arm_name]]
            rows.append({
                "arm": arm_name, "kernel": family.KERNEL_NAME,
                "lines": len(source.splitlines()),
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "ascii": kit.cyl.encodes_ascii(source),
                "the_lift_inverts_to_the_certified_prelude":
                    inverted == certified_prelude,
                "the_arm_block_is_present_whole": arm_block.strip() in source,
                "the_kernel_name_is_declared":
                    f'extern "C" __global__ void {family.KERNEL_NAME}(' in source,
                "cshift_dn_recompute_keeps_the_near_neighbour_branch":
                    "if (ia > 0) return resolve_H(comp, idx - stride, weld);" in source,
                "cshift_dn_recompute_keeps_the_wrap_arithmetic":
                    "resolve_H(comp, idx + (na - 1) * stride, weld)" in source,
                "cshift_dn_recompute_keeps_the_phase_on_the_wrapped_lane_only":
                    "if (ph) w = rotate_field_left(w, phase);" in source,
                "the_ghost_is_still_an_exact_cf_zero":
                    ("if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) "
                     "return cf_zero();") in source,
                "no_magnetic_pointer_survives_in_the_curl_half":
                    "cshift_dn(g" not in source,
                # MINUS THE DECLARATION: the helper's own signature spells its name
                # too, and a count that included it would report seven taps and pass a
                # weld that had lost one.
                "shifted_taps": source.count("cshift_dn_recompute(") - 1,
                "own_cell_registers": source.count("own_h["),
                # THE FOLDED DELTAS, which this family inherits whole because its text
                # IS the folded one with an insert.
                "the_third_boundary_code_is_defined":
                    "#define BC_MIRROR_PERIODIC 2" in source,
                "the_ghost_is_widened_to_the_metallic_zero":
                    "bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC" in source,
                "the_mask_splits_by_termination":
                    source.count("BC_MIRROR_PERIODIC && ") >= 6,
                # THE BETA DELTAS.
                "the_two_beta_increments_are_present":
                    source.count("mul_imag_coefficient_left(b")
                    == family.BETA_INSERT_LINES,
                "the_beta_helper_is_one_line_over_rotate_field_left":
                    "    return rotate_field_left(c, z);" in source,
                "the_four_beta_words_are_bound":
                    "float bp_re, float bp_im, float bm_re, float bm_im" in source,
                "both_beta_partners_are_weld_registers":
                    "mul_imag_coefficient_left(bp, f_2)" in source
                    and "mul_imag_coefficient_left(bm, f_1)" in source
                    and "cf f_2 = own_h[1];" in source
                    and "cf f_1 = own_h[0];" in source,
            })
            mine_plain = family.curl_pieces(
                arm_name, complex_emitter.complex_source("step_D", arm_name))
            mine_folded = family.curl_pieces(
                arm_name, complex_folded_kernels.folded_source("step_D", arm_name))
            equalities.append({
                "arm": arm_name,
                "the_transform_on_the_certified_plain_text_is_the_released_transform":
                    mine_plain == complex_fused_hd_pair.curl_pieces("plain", arm_name),
                "the_transform_on_the_certified_folded_text_is_the_released_transform":
                    mine_folded == complex_fused_hd_pair.curl_pieces("folded", arm_name),
            })
        needles = kit.mutation_needles_resolve(self)
        keys = [key for key in rows[0] if isinstance(rows[0][key], bool)]
        checks = {key: all(bool(row.get(key, True)) for row in rows) for key in keys}
        checks["every_text_redirects_six_taps_and_six_own_loads"] = all(
            row["shifted_taps"] == family.HALO_TAPS
            and row["own_cell_registers"] >= family.OWN_LOAD_EDITS for row in rows)
        checks["the_two_arms_emit_different_texts"] = len(
            {row["sha256"] for row in rows}) == len(rows)
        checks["the_transform_is_the_released_one_on_both_certified_texts"] = all(
            row["the_transform_on_the_certified_plain_text_is_the_released_transform"]
            and row["the_transform_on_the_certified_folded_text_is_the_released_transform"]
            for row in equalities)
        checks["every_armed_needle_resolves_to_exactly_one_site"] = needles["passed"]
        return {"texts": rows, "denominator": len(rows),
                "released_transform_equalities": equalities,
                "mutation_needles": needles, "checks": checks,
                "device_sources": sorted(family.device_sources()),
                "source_digest": family.source_digest(),
                "what_the_equalities_license": (
                    "that this family's curl transform IS the released complex H->D "
                    "weld's, over BOTH of that product's certified input texts and "
                    "BOTH arms, with the beta text as its only other input -- so every "
                    "clause gate_cuda_complex_fused_hd_pair.py measured about that "
                    "transform carries, and what THIS gate adds is the beta insert and "
                    "the beta partner"),
                "passed": all(checks.values())}

    def leg_refusal(self) -> Dict[str, Any]:
        """Every configuration this predicate must refuse, BY NAME, on a host."""
        from meep_gpu import withdraw_hoist  # noqa: PLC0415

        family = self.family
        named: List[Dict[str, Any]] = []

        class _HostGrid:
            xp = np
            beta = 0.0
            dimensions = 2
            cylindrical = False

            @staticmethod
            def has_symmetry():
                return False

            @staticmethod
            def is_mirrored(axis):  # noqa: ARG004
                return False

        class _BetaGrid(_HostGrid):
            beta = 0.25

        class _Fields:
            force_complex_fields = True

        def check(label: str, thunk, expect_covered: bool, names: str = "") -> None:
            try:
                covered, reason = thunk()
            except Exception as error:  # noqa: BLE001
                named.append({"case": label, "raised": repr(error)[:300],
                              "passed": False})
                return
            ok = bool(covered) == expect_covered and (
                not names or names.lower() in str(reason).lower())
            named.append({"case": label, "covered": bool(covered),
                          "reason": str(reason)[:300],
                          "expected_covered": expect_covered, "names": names,
                          "passed": ok})

        # 1. beta = 0 belongs to the certified complex pair or the folded one.
        check("beta_is_zero",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _HostGrid(), (), self.license, self.policy),
              False, "grid.beta is zero")

        # 2. REAL storage with beta is special_kz_curl's, named.
        class _RealFields:
            force_complex_fields = False

        check("real_storage_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _RealFields(), None, _BetaGrid(), (), self.license, self.policy),
              False, "special_kz_curl")

        # 3. CYLINDRICAL with beta: MEEP aborts.
        class _Cylindrical(_BetaGrid):
            cylindrical = True

        check("cylindrical_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _Cylindrical(), (), self.license, self.policy),
              False, "cylindrical")

        # 4. A 3-D grid with beta.
        class _ThreeD(_BetaGrid):
            dimensions = 3

        check("three_dimensional_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _ThreeD(), (), self.license, self.policy),
              False, "dimensions")

        # 5. NO EXPANSION LICENCE -- refused by the certified constitutive predicate
        #    first of all, because the arm is compiled into the binary at this seam
        #    and there is no later rung to check it at.
        check("no_expansion_licence",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _BetaGrid(), (), None, None),
              False, "constitutive half")

        # 6. An undeclared source set: the seam clause cannot infer an absence.
        check("sources_undeclared",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _BetaGrid(), None, self.license, self.policy),
              False, "")

        # 7. The withdraw refusal's text, on a synthetic standing source.
        class _Withdrawing:
            field_type = "electric"
            is_integrated = True
            _n_source_points = 1

            def withdraw(self, fields):  # noqa: ARG002
                return None

        seam_reasons = withdraw_hoist.seam_withdraw_reasons(
            _Fields(), (_Withdrawing(),), undeclared="undeclared",
            refusal=lambda index, source: (
                f"source {index} ({type(source).__name__}) has a standing integrated "
                f"electric withdraw"),
            hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
        withdraw_case = {
            "reasons": [str(text)[:220] for text in seam_reasons],
            "passed": bool(seam_reasons)
                      and "standing integrated" in str(seam_reasons[0])}

        # 8. The arm refuses rather than defaulting.
        arm_case = {"passed": False}
        try:
            family.kernel_source("SOMETHING")
        except ValueError as error:
            arm_case = {"raised": str(error)[:220], "passed": True}

        # 9. The transform refuses a text whose beta partner is NOT the weld register.
        stale = {"passed": False}
        try:
            corrupted = family._curl_source("FMA_V1").replace(  # noqa: SLF001
                "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
                "        cf f_9 = cf_load(g1, idx);\n"
                "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_9));", 1)
            family.curl_pieces("FMA_V1", corrupted)
            stale = {"passed": False,
                     "note": "the transform accepted a beta partner it did not redirect"}
        except AssertionError as error:
            stale = {"raised": str(error)[:300], "passed": True}
        except Exception as error:  # noqa: BLE001
            stale = {"raised": repr(error)[:300], "passed": False}

        record = {"named": named, "withdraw_text": withdraw_case,
                  "unknown_arm_refused": arm_case,
                  "the_transform_refuses_a_stale_beta_partner": stale,
                  "denominator": len(named) + 3}
        record["passed"] = bool(all(row["passed"] for row in named)
                                and withdraw_case["passed"] and arm_case["passed"]
                                and stale["passed"])
        return record

    def leg_spelling(self) -> Dict[str, Any]:
        """``complex64 * float32``, measured against ``cupy.multiply`` word for word.

        THE CLASS THE PLANT EXISTS FOR. On uniform data the naive four-product form
        and the contracted one AGREE bit for bit (``fma(a, b, +-0)`` rounds ``a*b``
        once, zero signs included). They separate only where a cross term UNDERFLOWS,
        and only under ``flush``, where the flushed zero's sign is lost.

        THE DIVIDE IS RECORDED AS AN ABSENCE WITH A CONTROL THAT BITES. CuPy's
        ``complex64 / float32`` is the SCALED complex/complex algorithm with the
        zero-valued terms kept -- not numpy's reciprocal multiply, and not the REAL
        beta sibling's true divide. This family has no division site; the assertion is
        armed by planting one and requiring the family's own emitter to refuse it.
        """
        if cp is None:
            return {"passed": False, "error": "no CuPy"}
        family = self.family
        rng = np.random.default_rng(self.seed + 991)
        cases: List[Dict[str, Any]] = []
        for shape_label, n in (("small", 4096), ("large", 1 << 20)):
            for value_class in ("uniform", "wide_dynamic", "signed_zero", "edge",
                                "planted_tiny_normal"):
                if value_class == "uniform":
                    re = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                    im = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                    coefficient = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                elif value_class == "wide_dynamic":
                    scale = np.float32(10.0) ** rng.integers(
                        -12, 12, size=n).astype(np.float32)
                    re = rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * scale
                    im = rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * scale
                    coefficient = (rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                                   * np.float32(10.0) ** rng.integers(
                                       -12, 12, size=n).astype(np.float32))
                elif value_class == "signed_zero":
                    # HALF THE WORDS ARE EXACT ZEROS OF RANDOM SIGN, and the
                    # coefficient carries an exact +-0.0 row: the class the zero cross
                    # terms decide, invisible on data that never holds a zero.
                    re = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                    im = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                    zeroed = rng.integers(0, 2, size=n) == 0
                    signs = np.where(rng.integers(0, 2, size=n) == 0,
                                     np.float32(-0.0),
                                     np.float32(0.0)).astype(np.float32)
                    re = np.where(zeroed, signs, re).astype(np.float32)
                    im = np.where(rng.integers(0, 2, size=n) == 0,
                                  -np.abs(im).astype(np.float32), im).astype(np.float32)
                    coefficient = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                    coefficient[::4] = np.float32(0.0)
                    coefficient[1::4] = np.float32(-0.0)
                elif value_class == "edge":
                    # +-0, +-subnormals, +-tiny normals, +-1e30, ASSEMBLED FROM UINT32
                    # BIT PATTERNS so that no host float operation touches a subnormal
                    # and both policy legs see identical inputs.
                    patterns = np.array(
                        [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF,
                         0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7149F2CA],
                        dtype=np.uint32)
                    re = np.frombuffer(
                        patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                        dtype=np.float32).copy()
                    im = np.frombuffer(
                        patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                        dtype=np.float32).copy()
                    coefficient = np.frombuffer(
                        patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                        dtype=np.float32).copy()
                else:
                    # BOTH PARTS NEGATIVE TINY NORMALS against a small coefficient, so
                    # z.im * c underflows: the class where the flushed zero's SIGN is
                    # the whole difference.
                    re = np.full(n, -float(PLANT_MAGNITUDE), dtype=np.float32)
                    im = np.full(n, -float(PLANT_MAGNITUDE), dtype=np.float32)
                    coefficient = np.full(n, np.float32(0.5), dtype=np.float32)
                    coefficient[1::2] = np.float32(-0.5)
                host = np.empty(n, dtype=np.complex64)
                host.real, host.imag = re, im
                z = cp.asarray(host)
                c = cp.asarray(coefficient)
                oracle = cp.multiply(z, c)
                row: Dict[str, Any] = {"shape": shape_label, "n": n,
                                       "value_class": value_class, "words": 2 * n,
                                       "arms": {}}
                for arm_name, body in _MULTIPLY_ARMS.items():
                    kernel, _source = _spelling_kernel(body, f"spell_{arm_name}")
                    out = cp.empty_like(z)
                    threads = 256
                    kernel(((n + threads - 1) // threads,), (threads,),
                           (out.view(cp.float32), z.view(cp.float32), c, np.int32(n)))
                    cp.cuda.runtime.deviceSynchronize()
                    row["arms"][arm_name] = int(differing(to_host(oracle), to_host(out)))
                cases.append(row)
        family_zero = all(row["arms"]["family_fma_v1"] == 0 for row in cases)
        biting = {arm: sum(row["arms"][arm] for row in cases) for arm in _MULTIPLY_ARMS}
        naive_planted = sum(row["arms"]["naive_four_product"] for row in cases
                            if row["value_class"] == "planted_tiny_normal")
        naive_other = sum(row["arms"]["naive_four_product"] for row in cases
                          if row["value_class"] != "planted_tiny_normal")
        armed = {"refused": None, "passed": False}
        try:
            family._assert_the_measured_absence_of_a_divide(  # noqa: SLF001
                family.kernel_source(self.arm).replace(
                    "    cf own_h[3];", "    float bad = 1.0f / dtdx;", 1))
        except AssertionError as error:
            armed = {"refused": str(error)[:300], "passed": True}
        record = {
            "cases": cases, "denominator": len(cases), "policy": self.policy,
            "family_multiply_is_zero_on_every_case": family_zero,
            "control_totals": biting,
            "naive_four_product": {
                "on_the_planted_class": naive_planted,
                "on_every_other_class": naive_other,
                "expected": ("bites under flush on the planted tiny-normal class (the "
                             "sign of a flushed zero); 0 under keep, where nothing "
                             "flushes, and 0 on uniform data under either policy"),
                "bit_here": bool(naive_planted)},
            "every_control_that_drops_a_term_bites": all(
                biting[arm] > 0 for arm in ("componentwise", "coefficient_left")),
            "cases_with_no_biting_control": [
                f"{row['shape']}/{row['value_class']}" for row in cases
                if not any(row["arms"][arm] for arm in _MULTIPLY_ARMS
                           if arm != "family_fma_v1")],
            "the_divide_has_no_site_here": {
                "planted_divide_is_refused": armed,
                "why": ("the reciprocal this recurrence needs is sinv, computed "
                        "host-side by PML, so CuPy's SCALED complex/complex divide "
                        "algorithm -- which the cylindrical H->D products must get "
                        "right, and which is NOT the real beta sibling's float32 true "
                        "divide -- has nothing to be right about here. Asserted "
                        "against the emitted text, with a planted divide as the "
                        "control")},
            "passed": bool(family_zero
                           and all(biting[arm] > 0 for arm in
                                   ("componentwise", "coefficient_left"))
                           and armed["passed"]),
        }
        return record

    # -- host mutations -------------------------------------------------------
    def host_mutation(self, name: str, spec: Mapping[str, Any],
                      steps: int) -> Dict[str, Any]:
        family = self.family
        if name == "rotation_skipped":
            record = kit.drive(self, spec, "uniform", steps,
                               modes=("array", "weld"), rotate=False)
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("the launcher never swaps the H/f_w_H bindings, so the "
                            "driver keeps stepping the PRE-launch magnetic field")}
        if name == "scratch_aliased_to_storage":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "alias"))
            state = self.resolve(fields, grid, pml, dtdx)
            state["scratch"] = {volume: getattr(fields, volume)
                                for volume in family.SCRATCH_VOLUMES}
            try:
                self.assert_disjoint(fields, state)
                return {"caught": False, "refused": None,
                        "why": "the disjointness check accepted the in-place weld"}
            except ValueError as error:
                return {"caught": True, "refused": str(error)[:400],
                        "why": ("the scratch IS the storage, which makes the launch "
                                "the IN-PLACE weld the board refused")}
        if name == "swap_constitutive_sublattice":
            from meep_gpu.cuda_kernels import (complex_emitter,  # noqa: PLC0415
                                               complex_pml_kernels)
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "swap"))
            half = complex_pml_kernels.complex_constitutive_tables(pml, True)
            tables = complex_pml_kernels.complex_curl_tables(
                pml, complex_emitter.HALF_INTEGER["step_D"])
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"constitutive": half,
                                                "tables": tables})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("update_H's kps/kms taken from the HALF-INTEGER "
                            "sub-lattice -- a half-cell error in the absorber "
                            "profile, converged and smooth")}
        if name == "unconjugated_backward_phase":
            from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
            phased = next((case for case in self.specs if any(case["k_point"])), spec)
            fields, grid, pml, dtdx, _plant = self.build(
                phased, "uniform", kit.case_rng(self.seed, "phase"))
            # THE FORWARD factor where the backward one belongs: the classic sign
            # error, in which every magnitude stays plausible and only the phase moves.
            flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
            record = kit.drive(self, phased, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"phase_flags": flags,
                                                "phase_values": values})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "fixture": phased["label"],
                    "why": ("the FORWARD Bloch factor bound where the backward one "
                            "belongs -- the classic sign error")}
        if name == "swap_beta_coefficient_words":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "beta"))
            plus, minus = family.beta_coefficients(grid)
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"beta": (minus, plus)})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "coefficients": [list(plus), list(minus)],
                    "why": ("the two host-rounded call-site coefficients are handed "
                            "over in the wrong order. Only the HOST transcription can "
                            "be wrong here, which is why this is a host mutation")}
        if name == "synthesise_the_zero_real_word":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "planted_row0", kit.case_rng(self.seed, "zero"))
            (plus_re, plus_im), (minus_re, minus_im) = family.beta_coefficients(grid)
            record = kit.drive(self, spec, "planted_row0", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"beta": ((0.0, plus_im),
                                                         (0.0, minus_im))})
            caught = not record["identical"]["weld"]
            return {"caught": caught, "passed": True,
                    "first_divergence": record["first_divergence"]["weld"],
                    "real_words_replaced": [float(plus_re), float(minus_re)],
                    "predicted_null": (
                        "PREDICTED NULL and INHERITED as a measurement rather than an "
                        "argument: complex_beta_kernels' own device leg measured this "
                        "needle UNCAUGHT 0/12 under BOTH policies, because "
                        "mul_imag_coefficient_left consumes the coefficient's "
                        "IMAGINARY word alone. Re-armed here on the H->D seam and on "
                        "the PLANTED class, which is the class the earlier record "
                        "named as the one that would discriminate it. Whichever way it "
                        "lands is a result: a CATCH refutes the inherited finding on "
                        "this seam and a null confirms it"),
                    "prediction_held": not caught,
                    "why": ("the beta coefficient's real word -- a SIGNED zero left "
                            "by Python's own complex multiply, negative exactly when "
                            "the coefficient is negative on the magnetic side -- is "
                            "replaced by a synthesised +0.0")}
        raise ValueError(f"no such host mutation {name!r}")

    # -- wiring ---------------------------------------------------------------
    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415
        from meep_gpu.cuda_kernels.registry import LICENSE_COMPLEX  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context: Any) -> Coverage:
            covered, reason = family.covers_complex_beta_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources,
                context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
            return Coverage(bool(covered), (str(reason),))

        def plan(context: Any):
            held: Dict[str, Any] = {}

            def resolve(ctx: Any) -> Dict[str, Any]:
                if "state" not in held:
                    held["state"] = family.resolve(
                        ctx.fields, ctx.grid, ctx.pml,
                        ctx.license_for(LICENSE_COMPLEX)["arm"])
                return {"state": held["state"]}

            def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
                return family.run_complex_beta_fused_hd_pair(
                    fields, context.grid, context.pml, arguments["state"]["arm"],
                    sources=context.sources, state=arguments["state"], check=False)

            return fused_pairs.CudaFusedPairPlan(
                family=family.FAMILY, label="complex beta fused H/D pair",
                kernel_label=family.KERNEL_NAME, replaces=tuple(family.REPLACES),
                slots=("update_H", "step_D"), context=context,
                resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": self.wiring_module}


LICENCES_WHAT = (
    "ONE CUDA kernel measured byte-identical, per COMPLETE DRIVER STEP and as uint32 "
    "words over every stored volume, to four independent arrangements of the "
    "update_H -> step_D seam on a COMPLEX64 special_kz (grid.beta != 0) run, at every "
    "block size and at three seed scales, on the fixtures and corpus rows this record "
    "names, under the installed float32 subnormal policy and the arm that policy's "
    "expansion probe licenses. It licenses NO throughput claim, NO dispatch claim and "
    "NO composition claim: the family declares INSTALLABLE = False, is not in the "
    "composer's tables on disk, and a credited seam-instance is PREDICATE ADMISSION. "
    "It licenses nothing about the beta coefficient's SIGNED ZERO real word either -- "
    "see this record's synthesise_the_zero_real_word entry for what was measured.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = kit.make_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    adapter = ComplexBetaAdapter()

    if args.lift_child:
        licence = json.loads(os.environ.get(LICENCE_ENV, "null"))
        adapter.license = licence
        adapter.arm = (licence or {}).get("arm")
        return kit.lift_child(
            adapter, args.lift_child, args.lift_child_target,
            json.loads(args.lift_child_cases), args.lift_child_out,
            args.lift_steps, args.lift_child_progress, args.subnormal_policy,
            bool(args.import_meep_for_host_policy), BLOCK_KEY)

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    return kit.run_campaign(adapter, args, GATE_FILE, FAMILY_NAME, BLOCK_KEY,
                            LICENCES_WHAT)


if __name__ == "__main__":
    raise SystemExit(main())
