"""The COMPLEX special_kz electric seam in one launch: complex beta ``step_D``
welded into complex ``update_E``.

The ``complex beta PML`` -> ``complex beta run`` cell's D->E product, and the
twin of :mod:`.complex_beta_fused_magnetic_pair`. Both halves are already
certified and neither is re-derived here:

* the curl half is :func:`.special_kz.beta_bloch_pml_curl_step` (K2) on
  ``step_D`` — the certified complex curl body plus the constexpr-gated ±i
  coefficient insert, with the ``-1j`` coefficient words this electric side
  binds (S:771-772);
* the constitutive half is :func:`.complex_fields.bloch_constitutive_step`'s
  ``SCALE = 1`` arm, admitted on a beta run by the shipped
  :func:`.special_kz.beta_run_complex_constitutive_coverage`;
* and the WELD is the one substitution
  :func:`.complex_fused_electric_pair.complex_fused_curl_constitutive_D` already
  makes between exactly those two bodies on a beta = 0 complex run.

SO THIS KERNEL IS ``complex_fused_curl_constitutive_D``, CHARACTER FOR CHARACTER,
PLUS THE SAME STATEMENTS ``beta_bloch_pml_curl_step`` ADDS TO
``bloch_pml_curl_step`` — four scalar word arguments, ``HAS_BETA``, and the
seven-statement insert between the ``dtdx`` curl and the ownership mask (the
array path's order, S:438-445 after :429, before :450). The gate's transcription
leg asserts exactly that, statement by statement, against both shipped sources.

THE FOURTH OPERAND ORIENTATION AND ITS LICENCE are the magnetic twin's, stated
once there (module docstring §"THE INSERT"): :data:`PRODUCT_PROBE_PATTERNS` is
:data:`.special_kz.BETA_PROBE_PATTERNS`, the fifth pattern is measured
AMBIGUOUS_BOTH with valid evidence in
``results/complex_expansion_beta_extension_2026-09-01/probe_keep/probe.json``
(keep policy, base-four table character-identical to the standing
``complex_expansion_convention_2026-08-16`` artifact), and
``beta_expansion_license`` grants FMA_V1 with basis ``measured``.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

ONE seam-instance, from ``results/fusion_matrix_triton_2026-09-01_foldedbeta``
at the D->E cell (``complex beta PML``, ``complex beta run``):

    tests:TestSpecialKz.test_special_kz   shape (500, 1, 1)   beta = -0.3907...

The board scored the instance "reachable; a slot has NO admitting arm (array
path)" with ``curl_admitters: []`` — the same unmeasured-fifth-pattern refusal
the probe extension discharges. The row declares ONE electric source
(``source_field_types == ['D']``), injected in THIS seam (``in_seam_source:
true``, ``live_in_seam_passes: ['fill_D']``), so :data:`CARRIES_DEPOSIT_REPAIR`
is the whole reason this product serves anything: with the flag False the source
clause refuses the cell's only row.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

Each is answered exactly as the beta-less electric twin answers it — the beta
insert sits entirely INSIDE the curl half and touches no pass:

* **the electric sources — CARRIED, through the shipped deposit repair**
  (:class:`~..deposit_repair.LeadingRepairPlan` /
  :class:`~..deposit_repair.TrailingRepairPlan`); the clause below passes
  ``carries_repair=CARRIES_DEPOSIT_REPAIR`` into
  :func:`~..deposit_repair.seam_source_reasons` so ``repairable`` decides,
  refusing BY NAME every seam it cannot invert. An undeclared ``sources`` is a
  REFUSAL, never an assumed ``()``.
* **``fill_symmetry_bc_D`` / ``fill_folded_far_ghosts_D`` — PROVABLY NO-OPS
  HERE** (stepping.py:1481-1482, :1565-1566 against the family's own fold
  refusal), re-checked by this predicate anyway.
* **``zero_metal_D`` — CARRIED INLINE**, through
  :func:`.coverage.zero_metal_axes`, on BOTH word planes, with the D SIDE'S
  component map: an x wall clears the TANGENTIAL ``Dy``/``Dz`` and leaves
  ``Dx`` (:data:`WALL_CLEARED_COMPONENTS`, re-derived from the shipped
  ``IYEE_SHIFTS`` by the predicate, which refuses on a disagreement — the
  beta-less twin's first device run is the measured reason this clause exists).

``BACKWARD`` IS 1 AND ``SCALE`` IS 1 — ``SUB_STEPS['step_D']['backward']`` and
the E side's ``inv_eps`` multiply (``_mul_field_left``, an orientation the curl
half already launches; no fifth orientation beyond the beta insert's). The
coefficient lattices are the mirror image of the magnetic twin's: INTEGER on the
step_D curl, HALF-INTEGER on E.

NOT WIRED, and not an arm. Nothing in ``launch.py`` names this module;
``fastpath.plan_fast_path`` is unchanged.

DEVICE STATUS: see ``parity/meep_gpu/probe_triton_complex_beta_fused_electric_pair.py``
and its artifact. No byte-identity claim is made anywhere in this file until
that gate has run and released.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _mul_coefficient_left,
    _mul_field_left,
    _phase_arguments,
    _rotate_field_left,
    _word_view,
    bloch_phase_table,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .special_kz import (
    BETA_PROBE_PATTERNS,
    _mul_imag_coefficient_left,
    beta_bloch_pml_curl_coverage,
    beta_curl_coefficients,
    beta_expansion_from_probe,
    beta_run_complex_constitutive_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and
#: the flag IS the product: the cell's one corpus row declares an electric
#: source injected in this seam, so with the flag False the source clause
#: refuses everything this product exists to serve. The flag reaches only
#: ``deposit_repair.repairable``, which goes on refusing BY NAME every seam the
#: repair cannot invert; the gate's carry legs measure the shipped
#: LeadingRepairPlan/TrailingRepairPlan bracket doing work rather than assuming
#: it.
CARRIES_DEPOSIT_REPAIR = True

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The three volumes the curl half writes, in kernel argument order. READ from
#: :data:`.launch.SUB_STEPS` rather than spelled, because the wall clause below
#: indexes ``IYEE_SHIFTS`` by these names.
CURL_TARGETS: Tuple[str, ...] = tuple(SUB_STEPS[CURL_SUB_STEP]["targets"])

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: ``ComplexConstitutivePlan``'s ``scale`` for side E, restated and pinned the
#: same way. The constexpr that switches on the ``inv_eps`` multiply — the ONE
#: arithmetic difference between this body's constitutive half and the magnetic
#: twin's.
SCALE = 1

#: WHICH CURL TARGETS ``zero_metal_D`` CLEARS, PER WALLED AXIS — the D side's
#: map: the TWO TANGENTIAL components (an x wall clears Dy and Dz and leaves
#: Dx), the exact complement of the B side's. A CLAIM ABOUT THE KERNEL, not a
#: derivation: the predicate recomputes the same map from ``fields.IYEE_SHIFTS``
#: and refuses when they differ — the clause the beta-less twin's first device
#: run (wrong in both directions at once on every metallic grid) is the measured
#: reason for.
WALL_CLEARED_COMPONENTS: Tuple[Tuple[int, ...], ...] = ((1, 2), (0, 2), (0, 1))

#: The probe patterns this product's operand orientations require — the EXTENDED
#: set (the beta insert's imaginary-coefficient-left product; module docstring).
PRODUCT_PROBE_PATTERNS: Tuple[str, ...] = tuple(BETA_PROBE_PATTERNS)

#: The complex-multiply device functions this kernel is allowed to call, and the
#: only ones — FOUR names, the twin's three plus the beta insert's.
LICENSED_MULTIPLY_HELPERS: Tuple[str, ...] = (
    "_rotate_field_left", "_mul_field_left", "_mul_coefficient_left",
    "_mul_imag_coefficient_left")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "CURL_TARGETS", "LICENSED_MULTIPLY_HELPERS", "PRODUCT_PROBE_PATTERNS",
    "REPLACES", "SCALE", "WALL_CLEARED_COMPONENTS",
    "ComplexBetaFusedElectricPairPlan",
    "complex_beta_fused_curl_constitutive_D",
    "complex_beta_fused_curl_constitutive_D_kernel",
    "complex_beta_fused_electric_pair_coverage",
    "plan_complex_beta_fused_electric_pair",
    "plan_complex_beta_fused_electric_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``complex_fields.METALLIC``, restated for the
    # reason that module restates ``kernels``'. ONLY METALLIC IS NAMED: the
    # certified complex curl body branches on ``== METALLIC`` and takes periodic
    # as the else.
    METALLIC = tl.constexpr(1)

    @triton.jit
    def complex_beta_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz (complex64 as words)
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        e0, e1, e2,                       # inverse epsilon, float32 VOLUMES
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        pxr, pxi, pyr, pyi, pzr, pzi,     # per-axis complex64-rounded phase (conj: BACKWARD)
        bp_re, bp_im, bm_re, bm_im,       # complex64-rounded ±sign beta coefficients (S:770-784)
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see the twin
        SCALE: tl.constexpr,              # bound to 1 by every builder; see the twin
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Complex beta ``step_D`` + ``zero_metal_D`` + complex ``update_E``, one launch.

        ``complex_fused_electric_pair.complex_fused_curl_constitutive_D``'s body
        with EXACTLY the additions ``special_kz.beta_bloch_pml_curl_step`` makes
        to ``complex_fields.bloch_pml_curl_step``: the four beta words,
        ``HAS_BETA``, and the insert between the curl and the ownership mask.
        ``HAS_BETA = 0`` must build byte-identical to the plain complex fused
        electric pair — the gate's identity leg pins that.

        The beta partners are the CENTER word pairs, loaded before the phase
        section and never touched by it; the ``-1j`` of this electric side lives
        in the host-bound coefficient words (S:771-772), rounded once through
        numpy.complex64 with the signed-zero real word passed through.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from special_kz.beta_bloch_pml_curl_step (itself
        # complex_fields.bloch_pml_curl_step plus the beta insert); the only edit
        # below its stores is the wall clear applied to the v registers.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == METALLIC:
            vx = live & (si >= 0) & (si < nx)
        else:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        if BCY == METALLIC:
            vy = live & (sj >= 0) & (sj < ny)
        else:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        if BCZ == METALLIC:
            vz = live & (sk >= 0) & (sk < nz)
        else:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

        ox = si * nyz + j * nz + k
        oy = i * nyz + sj * nz + k
        oz = i * nyz + j * nz + sk

        # --- wrapped-lane predicates, one plane per axis ------------------------
        if BACKWARD:
            wx, wy, wz = i == 0, j == 0, k == 0
        else:
            wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

        # --- loads: two words per operand --------------------------------------
        a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
        b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
        b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
        c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
        c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
        a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
        a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
        a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
        a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
        b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
        b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
        b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
        b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
        c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
        c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
        c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
        c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

        # --- Bloch phase on the wrapped lane, BEFORE the difference -------------
        # SHIFTED operands only; the beta partners (a, b centers) are never
        # rotated. The host passed the CONJUGATE for this BACKWARD sub-step
        # (S:1818-1822).
        if PHX:
            rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
            b_x_re = tl.where(wx, rot_re, b_x_re)
            b_x_im = tl.where(wx, rot_im, b_x_im)
            rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
            c_x_re = tl.where(wx, rot_re, c_x_re)
            c_x_im = tl.where(wx, rot_im, c_x_im)
        if PHY:
            rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
            a_y_re = tl.where(wy, rot_re, a_y_re)
            a_y_im = tl.where(wy, rot_im, a_y_im)
            rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
            c_y_re = tl.where(wy, rot_re, c_y_re)
            c_y_im = tl.where(wy, rot_im, c_y_im)
        if PHZ:
            rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
            a_z_re = tl.where(wz, rot_re, a_z_re)
            a_z_im = tl.where(wz, rot_im, a_z_im)
            rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
            b_z_re = tl.where(wz, rot_re, b_z_re)
            b_z_im = tl.where(wz, rot_im, b_z_im)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
        t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
        t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
        t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
        t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
        t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
        curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
        curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

        # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order
        # (S:438-445 against S:450). Coefficient LEFT (S:784).
        if HAS_BETA:
            t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)
            curl0_re = curl0_re - t_re
            curl0_im = curl0_im - t_im
            t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)
            curl1_re = curl1_re - t_re
            curl1_im = curl1_im - t_im

        # --- ownership mask (stepping._mask_non_owned_cells) -------------------
        # ON THIS SUB-STEP THE BACKWARD ARM IS THE LIVE ONE, masking TWO
        # components per metallic axis.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY == METALLIC:
                curl0_re = tl.where(at_y, 0.0, curl0_re)
                curl0_im = tl.where(at_y, 0.0, curl0_im)
            if BCZ == METALLIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
            if BCX == METALLIC:
                curl1_re = tl.where(at_x, 0.0, curl1_re)
                curl1_im = tl.where(at_x, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            if BCX == METALLIC:
                curl2_re = tl.where(at_x, 0.0, curl2_re)
                curl2_im = tl.where(at_x, 0.0, curl2_im)
            if BCY == METALLIC:
                curl2_re = tl.where(at_y, 0.0, curl2_re)
                curl2_im = tl.where(at_y, 0.0, curl2_im)
        else:
            if BCX == METALLIC:
                curl0_re = tl.where(at_x, 0.0, curl0_re)
                curl0_im = tl.where(at_x, 0.0, curl0_im)
            if BCY == METALLIC:
                curl1_re = tl.where(at_y, 0.0, curl1_re)
                curl1_im = tl.where(at_y, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
        p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
        q_re = q_re - curl0_re
        q_im = q_im - curl0_im
        n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)
        e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
        r_re = (r_re + n0_re) - p0_re
        r_im = (r_im + n0_im) - p0_im
        v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

        p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
        p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
        q_re = q_re - curl1_re
        q_im = q_im - curl1_im
        n1_re, n1_im = _mul_field_left(q_re, q_im, si_z, EXPANSION)
        e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
        r_re = (r_re + n1_re) - p1_re
        r_im = (r_im + n1_im) - p1_im
        v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

        p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
        p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
        q_re = q_re - curl2_re
        q_im = q_im - curl2_im
        n2_re, n2_im = _mul_field_left(q_re, q_im, si_x, EXPANSION)
        e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
        r_re = (r_re + n2_re) - p2_re
        r_im = (r_im + n2_im) - p2_im
        v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Applied to the REGISTERS, before both the store and the constitutive
        # read. BOTH planes; THE COMPONENT MAP IS THE D SIDE'S — the two
        # TANGENTIAL components per walled axis (WALL_CLEARED_COMPONENTS), the
        # exact complement of the magnetic twin's, measured wrong-in-both-
        # directions when the twin's map was carried here by the beta-less
        # product's first device run.
        if ZM_X:
            v1_re = tl.where(at_x, 0.0, v1_re)
            v1_im = tl.where(at_x, 0.0, v1_im)
            v2_re = tl.where(at_x, 0.0, v2_re)
            v2_im = tl.where(at_x, 0.0, v2_im)
        if ZM_Y:
            v0_re = tl.where(at_y, 0.0, v0_re)
            v0_im = tl.where(at_y, 0.0, v0_im)
            v2_re = tl.where(at_y, 0.0, v2_re)
            v2_im = tl.where(at_y, 0.0, v2_im)
        if ZM_Z:
            v0_re = tl.where(at_z, 0.0, v0_re)
            v0_im = tl.where(at_z, 0.0, v0_im)
            v1_re = tl.where(at_z, 0.0, v1_re)
            v1_im = tl.where(at_z, 0.0, v1_im)

        # --- stores: u then f (kernels.py:193-198 order), both planes ----------
        tl.store(u0 + 2 * idx, n0_re, mask=live)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
        tl.store(u1 + 2 * idx, n1_re, mask=live)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
        tl.store(u2 + 2 * idx, n2_re, mask=live)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)

        # ==================== the constitutive half ===========================
        # Verbatim from complex_fields.bloch_constitutive_step's SCALE=1 arm, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half
        # just produced. `prev` is read BEFORE the `w` store (S:2083-2085); the
        # inv_eps multiply happens BETWEEN the source read and that store
        # (complex_fields.py:783-786), so `f_w_E*` holds `D * inv_eps` and not
        # `D`; the two accumulations stay separate and left-to-right, each a
        # coefficient-LEFT zero-imaginary complex product (S:2086-2087,
        # S:2093-2095).
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v0_re
        src_im = v0_im
        if SCALE:
            # inv_eps is a float32 volume at the COMPLEX cell index — `+ idx`, not
            # `+ 2 * idx`. D LEFT (S:982-984).
            ie = tl.load(e0 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
        tl.store(w0 + 2 * idx, src_re, mask=live)
        tl.store(w0 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h0 + 2 * idx, a_re, mask=live)
        tl.store(h0 + 2 * idx + 1, a_im, mask=live)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(w1 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v1_re
        src_im = v1_im
        if SCALE:
            ie = tl.load(e1 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
        tl.store(w1 + 2 * idx, src_re, mask=live)
        tl.store(w1 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h1 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h1 + 2 * idx, a_re, mask=live)
        tl.store(h1 + 2 * idx + 1, a_im, mask=live)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(w2 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v2_re
        src_im = v2_im
        if SCALE:
            ie = tl.load(e2 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
        tl.store(w2 + 2 * idx, src_re, mask=live)
        tl.store(w2 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h2 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h2 + 2 * idx, a_re, mask=live)
        tl.store(h2 + 2 * idx + 1, a_im, mask=live)

else:  # pragma: no cover - laptop path
    complex_beta_fused_curl_constitutive_D = None  # type: ignore[assignment]


def complex_beta_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if complex_beta_fused_curl_constitutive_D is None:
        raise ImportError(
            "the complex beta fused electric D/E kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return complex_beta_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_beta_fused_electric_pair_coverage(fields: Any, pml: Any,
                                              sources: Any = None,
                                              probe: Any = None) -> Coverage:
    """May ONE launch span complex beta ``step_D`` -> wall -> complex ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam
    clauses — the beta-less electric twin's construction with the halves swapped
    to :func:`.special_kz.beta_bloch_pml_curl_coverage` (EXTENDED probe set) and
    :func:`.special_kz.beta_run_complex_constitutive_coverage` (BASE set, its
    certified kernel's own contract). Nothing is weakened; ``probe`` is
    forwarded to both halves unchanged.
    """
    reasons: List[str] = []

    curl = beta_bloch_pml_curl_coverage(fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"complex beta curl half: {reason}"
                       for reason in curl.reasons)
    constitutive = beta_run_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, probe=probe)
    if not constitutive.covered:
        reasons.extend(f"complex beta constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3296-3299). THIS IS THE CLAUSE THE WHOLE PRODUCT TURNS ON: the
    # cell's one corpus row declares an electric source, so with
    # `carries_repair=False` this arm admits nothing. The flag is passed, not
    # assumed: `repairable` still refuses by name every seam it cannot invert.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3296), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # THE TWO SYMMETRY PASSES — no-ops on every admitted configuration
    # (stepping.py:1481-1482, :1565-1566 against the family's own fold refusal),
    # RE-CHECKED HERE ANYWAY rather than read off another module's guard.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_D (driver.py:3300) "
            "and fill_folded_far_ghosts_D (:3302) run inside this seam and this "
            "weld carries neither")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two symmetry passes "
                f"in this seam are no longer no-ops and this weld carries neither")

    # THE WALL CLEAR is carried inline, so the grid must be able to answer which
    # axes are walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # ...AND THE COMPONENT MAP IT BAKES MUST BE THE ENGINE'S OWN. Recomputed from
    # the shipped ``IYEE_SHIFTS`` by exactly ``_zero_metal``'s rule — shift 0 on
    # the walled axis — rather than compared against a second transcription.
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, name in enumerate(CURL_TARGETS)
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    if derived != WALL_CLEARED_COMPONENTS:
        reasons.append(
            f"stepping._zero_metal clears {derived} per walled axis on "
            f"{list(CURL_TARGETS)}, but this kernel bakes "
            f"{WALL_CLEARED_COMPONENTS}; the inline wall clear would zero one plane "
            f"and leave another live")

    # A susceptibility makes the constitutive source (D - sum P) rather than D
    # (fields.py:1096-1105). The constitutive half already refuses it
    # (`_beta_complex_grid_reasons` clause 7); restated because THIS kernel bakes
    # the plain product and a reader should not have to chase the other predicate.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: this kernel bakes the plain "
            "constitutive product, whose source is D and not (D - sum P)")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL on every launch
    # (SCALE=1), so `Fields` must be able to hand one over per target component.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the E-side constitutive "
            "half cannot bind the inv_eps volumes its SCALE=1 arm loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ComplexBetaFusedElectricPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    :class:`.complex_fused_electric_pair.ComplexFusedElectricPairPlan` plus the
    four host-rounded beta coefficient words and ``HAS_BETA``. The complex
    volumes are bound as float32 WORD VIEWS at plan time; the ``inv_eps``
    volumes are bound RAW (one real coefficient per complex cell, indexed at
    ``+ idx``).
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "scale", "bc", "zero_metal",
        "phased", "phase_values", "beta_words", "has_beta", "expansion",
        "block", "num_warps",
        "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inv_eps", "_e_coefficients", "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phased, phase_values,
                 beta_words, expansion: int, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_beta: int = 1) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64
        # precision, and Triton types a Python float argument as fp32.
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.scale = SCALE
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        # ((bp_re, bp_im), (bm_re, bm_im)) — already complex64-rounded words
        # with the signed-zero real parts passed through (float() keeps them).
        self.beta_words = tuple(tuple(float(word) for word in pair)
                                for pair in beta_words)
        self.has_beta = int(has_beta)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        self._e_aux = tuple(CupyPointer(_word_view(a)) for a in e_aux)
        # NOT word-viewed, and not optional: SCALE is 1 on every launch, so a
        # placeholder binding here would be read as an epsilon.
        if inverse_epsilon is None:
            raise ValueError(
                "the E-side fused pair loads inv_eps on every launch (SCALE=1); "
                "a placeholder binding would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(a)) for a in e_coefficients)
        # THE INT32 WORD BOUND, refused here as well because the from-arrays
        # route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else complex_beta_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        (bp_re, bp_im), (bm_re, bm_im) = self.beta_words
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inv_eps, *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            bp_re, bp_im, bm_re, bm_im,
            BACKWARD=self.backward,
            SCALE=self.scale,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            HAS_BETA=self.has_beta,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexBetaFusedElectricPairPlan(shape={self.shape}, "
                f"bc={self.bc}, zero_metal={self.zero_metal}, "
                f"phased={self.phased}, beta_words={self.beta_words!r}, "
                f"has_beta={self.has_beta}, expansion={self.expansion}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_complex_beta_fused_electric_pair(fields: Any, pml: Any,
                                          sources: Any = None,
                                          block: Optional[int] = None,
                                          num_warps: Optional[int] = 1,
                                          kernel: Any = None,
                                          probe: Any = None,
                                          ) -> Optional[ComplexBetaFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal. The EXPANSION constexpr binds through
    :func:`.special_kz.beta_expansion_from_probe` — the EXTENDED pattern set —
    never through the base-four resolver.
    """
    if not complex_beta_fused_electric_pair_coverage(fields, pml, sources,
                                                     probe=probe).covered:
        return None
    from .complex_fields import load_expansion_probe  # noqa: PLC0415

    record = probe if probe is not None else load_expansion_probe()
    expansion = beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phased, values = _phase_arguments(bloch_phase_table(grid, kinds),
                                      backward=bool(curl_spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=False, complex_storage=True)
    return ComplexBetaFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), phased, values, beta_words, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # THE MIRROR IMAGE OF THE MAGNETIC TWIN'S CHOICE: INTEGER lattice on the
        # step_D curl, HALF-INTEGER on E; the gate carries a mutation for a swap.
        [getattr(pml, f"{stem}_{axis}_h") for axis in "xyz"
         for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


def plan_complex_beta_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], dtdx: float, expansion: int,
        beta_words,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        has_beta: int = 1) -> ComplexBetaFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs. ``arrays`` is keyed by component name plus
    ``fu_D*``, ``f_w_E*`` and ``inv_eps_Ex``... (float32); ``phases`` is the
    per-axis ``Optional[complex]`` table and the conjugation for this backward
    sub-step is applied HERE; ``beta_words`` is normally
    :func:`.special_kz.beta_curl_coefficients`'s ``magnetic=False`` output, or a
    deliberately wrong pair on a mutation leg; ``has_beta=0`` is the identity
    leg's certified-kernel arm.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    phased, values = _phase_arguments(tuple(phases),
                                      backward=bool(curl_spec["backward"]))
    return ComplexBetaFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, phased, values, beta_words,
        int(expansion), DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )
