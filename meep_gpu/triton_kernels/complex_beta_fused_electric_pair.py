"""The COMPLEX special_kz electric seam in one launch: complex beta ``step_D``
welded into complex ``update_E``.

The ``complex beta PML`` -> ``complex beta run`` cell's D->E product, and the
twin of :mod:`.complex_beta_fused_magnetic_pair`. Both halves are already
certified and neither is re-derived here:

* the curl half is :func:`.special_kz.beta_bloch_pml_curl_step` (K2) on
  ``step_D`` — the complex curl body plus the constexpr-gated beta pass after the
  forms, the quarter turn by ``-i`` this electric side binds (the D side's word is
  the negated one, :func:`.special_kz.beta_curl_coefficients`);
* the constitutive half is :func:`.complex_fields.bloch_constitutive_step`'s
  ``SCALE = 1`` arm, admitted on a beta run by the shipped
  :func:`.special_kz.beta_run_complex_constitutive_coverage`;
* and the WELD is the one substitution
  :func:`.complex_fused_electric_pair.complex_fused_curl_constitutive_D` already
  makes between exactly those two bodies on a beta = 0 complex run.

SO THIS KERNEL IS ``complex_fused_curl_constitutive_D``, CHARACTER FOR CHARACTER,
PLUS THE SAME STATEMENTS ``beta_bloch_pml_curl_step`` ADDS TO
``bloch_pml_curl_step`` — two scalar word arguments, ``HAS_BETA``, and the beta
pass after the curl forms and before the stores (MEEP's order: the pass runs after
every component's curl, step_db.cpp:161-175). The gate's transcription
leg asserts exactly that, statement by statement, against both shipped sources.

THE PASS is the magnetic twin's, stated once there (module docstring §"THE PASS").

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

ONE seam-instance, from ``results/fusion_matrix_triton_2026-09-01_foldedbeta``
at the D->E cell (``complex beta PML``, ``complex beta run``):

    tests:TestSpecialKz.test_special_kz   shape (500, 1, 1)   beta = -0.3907...

The board scored the instance "reachable; a slot has NO admitting arm (array
path)" with ``curl_admitters: []``. The row declares ONE electric source
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
    _word_view,
    bloch_phase_table,
    connection_arguments,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .special_kz import (
    beta_bloch_pml_curl_coverage,
    beta_curl_coefficients,
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

#: The complex-multiply device functions this kernel is allowed to call, and the
#: only one: the partner connection. The beta pass's quarter turn has a real
#: operand and is formed part by part, inline; the test scans the shipped body for
#: any other helper.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect",)

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "CURL_TARGETS", "MULTIPLY_HELPERS",
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

    # The forms, the fold ladder, the record layout and the connection, read by
    # the kernel below under the names the complex pair binds them.
    from .complex_fields import _connect  # noqa: E402
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

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
        cb,                               # chunk record: the forms' membership and presence
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
        beta_plus, beta_minus,            # f32 quarter-turn words per sign (B: +c, D: -c)
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        SCALE: tl.constexpr,              # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """Complex beta ``step_D`` + ``zero_metal_D`` + complex ``update_E``, one launch.

        ``complex_fused_electric_pair.complex_fused_curl_constitutive_D``'s body, character for character, plus
        ``special_kz.beta_bloch_pml_curl_step``'s beta pass after the six part forms and
        before the wall clear -- the array path's order (the curl, its beta pass, then the
        seam, then update_E). The pass is a quarter turn of the CENTER partner's parts by
        the side's real word, negated, masked to (+0, +0) on the cells the curl's mask
        drops, through the beta ladder per part; no complex product is formed. The
        registers it leaves are the words both the store and the constitutive half read.
        ``HAS_BETA = 0`` builds ``complex_fused_curl_constitutive_D`` exactly.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # complex_fields.bloch_pml_curl_step's body, statement for statement; the
        # only edit below its last form is the wall clear on the v registers.
        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
        # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

        # --- loads: two words per operand ------------------------------------------
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

        # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
        # Target 0 reads c along y and b along z, target 1 a along z and c along x,
        # target 2 b along x and a along y (stepping's term table). `tl.where` is a
        # bitwise select, so every cell off plane 0 keeps its loaded words.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
        b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
        a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
        c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
        b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
        a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
        if CNX:
            r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
            c1_re = tl.where(at_x, r_re, c_re)
            c1_im = tl.where(at_x, r_im, c_im)
            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
            b2_re = tl.where(at_x, r_re, b_re)
            b2_im = tl.where(at_x, r_im, b_im)
        if CNY:
            r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
            c0_re = tl.where(at_y, r_re, c_re)
            c0_im = tl.where(at_y, r_im, c_im)
            r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
            a2_re = tl.where(at_y, r_re, a_re)
            a2_im = tl.where(at_y, r_im, a_im)
        if CNZ:
            r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
            b0_re = tl.where(at_z, r_re, b_re)
            b0_im = tl.where(at_z, r_im, b_im)
            r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
            a1_re = tl.where(at_z, r_re, a_re)
            a1_im = tl.where(at_z, r_im, a_im)

        # --- the curl (stepping._curl_from_operands), per part ---------------------
        # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
        # association with both partners, the one-partner form where the record says the
        # other partner's derivative axis is invariant, an exact +0 with neither. The
        # dtdx scale is the real scalar times each part (part_products.scale).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
        curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
        curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
        curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
        curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
        curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
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

        # --- the PML forms (stepping._apply_curl_regions), per part ----------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
        # select per target serves both parts; fu is read and written on the
        # recurrence's cells only (the auxiliary contract), both words.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
        p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl2_im, km_x, si_x, km_y, si_y, du2)

        # --- the special-kz beta pass (stepping._special_kz_beta_term, then
        # stepping._apply_folded_increment), per part ------------------------------
        # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
        # applied to the registers the forms returned. The partners are the CENTER
        # words as loaded, never connected: target 0 takes the second source at sign
        # +1, target 1 the first at sign -1; target 2 gets nothing. The increment is
        # the negated quarter turn (part_products.quarter_turn): i*c*(x + iy) is
        # (-(c*y), c*x), so its negation is (c*y, -(c*x)), the two negations of the
        # real part cancelling exactly. The words beta_plus/beta_minus already carry
        # the side's i or -i (special_kz.beta_curl_coefficients). Masked to (+0, +0)
        # on the cells the curl's own mask drops; each part takes the beta ladder.
        if HAS_BETA:
            q0_re = beta_plus * b_im
            q0_im = (beta_plus * b_re) * -1.0
            q1_re = beta_minus * a_im
            q1_im = (beta_minus * a_re) * -1.0
            if BACKWARD:
                if BCY == METALLIC:
                    q0_re = tl.where(at_y, 0.0, q0_re)
                    q0_im = tl.where(at_y, 0.0, q0_im)
                if BCZ == METALLIC:
                    q0_re = tl.where(at_z, 0.0, q0_re)
                    q0_im = tl.where(at_z, 0.0, q0_im)
                if BCX == METALLIC:
                    q1_re = tl.where(at_x, 0.0, q1_re)
                    q1_im = tl.where(at_x, 0.0, q1_im)
                if BCZ == METALLIC:
                    q1_re = tl.where(at_z, 0.0, q1_re)
                    q1_im = tl.where(at_z, 0.0, q1_im)
            else:
                if BCX == METALLIC:
                    q0_re = tl.where(at_x, 0.0, q0_re)
                    q0_im = tl.where(at_x, 0.0, q0_im)
                if BCY == METALLIC:
                    q1_re = tl.where(at_y, 0.0, q1_re)
                    q1_im = tl.where(at_y, 0.0, q1_im)
            v0_re, n0_re = fold_ladder(v0_re, n0_re, q0_re, si_y, si_z, du0)
            v0_im, n0_im = fold_ladder(v0_im, n0_im, q0_im, si_y, si_z, du0)
            v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)
            v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)

        # --- the seam: stepping.zero_metal_D, between the two sub-steps ---------
        # Applied to the REGISTERS, before both the store and the constitutive read,
        # so the two consumers see the one value the array path leaves in D. BOTH
        # parts. THE COMPONENT MAP IS THE COMPLEMENT OF THE MAGNETIC TWIN'S:
        # `_zero_metal` clears every component whose Yee shift on the WALLED axis is
        # ZERO, which on the D side is the two TANGENTIAL components
        # (:data:`WALL_CLEARED_COMPONENTS`, re-derived from IYEE_SHIFTS by the
        # predicate, which refuses on a disagreement).
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

        # --- stores: fu (recurrence cells only), then the fields, both parts ----
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)

        # ==================== the constitutive half ===========================
        # complex_fields.bloch_constitutive_step's SCALE=1 arm, per part, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half just
        # produced -- the substitution kernels.fused_curl_constitutive_D
        # makes on the real path. The record's dsigw line on the half-integer lattice
        # picks the form; f_w is read and written on the accumulation's cells only,
        # `prev` before the store.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        src_re = v0_re
        src_im = v0_im
        if SCALE:
            # inv_eps is a float32 volume at the COMPLEX cell index -- `+ idx`.
            ie = tl.load(e0 + idx, mask=live, other=0.0)
            src_re = src_re * ie
            src_im = src_im * ie
        tl.store(w0 + 2 * idx, src_re, mask=live & dw0)
        tl.store(w0 + 2 * idx + 1, src_im, mask=live & dw0)
        a_re = tl.load(h0 + 2 * idx, mask=live & dw0, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        tl.store(h0 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0),
                 mask=live)
        tl.store(h0 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0), mask=live)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=live & dw1, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        src_re = v1_re
        src_im = v1_im
        if SCALE:
            # inv_eps is a float32 volume at the COMPLEX cell index -- `+ idx`.
            ie = tl.load(e1 + idx, mask=live, other=0.0)
            src_re = src_re * ie
            src_im = src_im * ie
        tl.store(w1 + 2 * idx, src_re, mask=live & dw1)
        tl.store(w1 + 2 * idx + 1, src_im, mask=live & dw1)
        a_re = tl.load(h1 + 2 * idx, mask=live & dw1, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        tl.store(h1 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1),
                 mask=live)
        tl.store(h1 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1), mask=live)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=live & dw2, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        src_re = v2_re
        src_im = v2_im
        if SCALE:
            # inv_eps is a float32 volume at the COMPLEX cell index -- `+ idx`.
            ie = tl.load(e2 + idx, mask=live, other=0.0)
            src_re = src_re * ie
            src_im = src_im * ie
        tl.store(w2 + 2 * idx, src_re, mask=live & dw2)
        tl.store(w2 + 2 * idx + 1, src_im, mask=live & dw2)
        a_re = tl.load(h2 + 2 * idx, mask=live & dw2, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        tl.store(h2 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2),
                 mask=live)
        tl.store(h2 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2), mask=live)

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
                                              sources: Any = None) -> Coverage:
    """May ONE launch span complex beta ``step_D`` -> wall -> complex ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam
    clauses — the beta-less electric twin's construction with the halves swapped
    to :func:`.special_kz.beta_bloch_pml_curl_coverage` and
    :func:`.special_kz.beta_run_complex_constitutive_coverage` (its certified
    kernel's own contract). Nothing is weakened.
    """
    reasons: List[str] = []

    curl = beta_bloch_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"complex beta curl half: {reason}"
                       for reason in curl.reasons)
    constitutive = beta_run_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
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

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason. The
    ``inv_eps`` volumes are the exception and are bound RAW: they are float32 with
    one real coefficient per complex cell, and the kernel indexes them at ``+ idx``
    (``fields.py:1203-1204``; the same binding
    :class:`.complex_fields.ComplexConstitutivePlan` makes).
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "scale", "bc", "zero_metal",
        "connections", "connection_values", "beta_words", "has_beta", "block", "num_warps",
        "chunk_bounds",
        "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inv_eps", "_e_coefficients", "_cb", "_grid",
        "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, connections,
                 connection_values, beta_words, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_beta: int = 1, *,
                 chunk_bounds: Any) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64
        # precision, and Triton types a Python float argument as fp32, so the two
        # are the same bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.scale = SCALE
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        # (plus, minus): one float32-rounded real word per sign, the side's i or -i
        # already applied (special_kz.beta_curl_coefficients); float() keeps the bits.
        words = tuple(beta_words)
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real quarter-turn words of "
                             f"special_kz.beta_curl_coefficients(..., "
                             f"complex_storage=True), got {beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        self._e_aux = tuple(CupyPointer(_word_view(a)) for a in e_aux)
        # NOT word-viewed, and not optional. Unlike ComplexConstitutivePlan, which
        # binds the source views as placeholders on the H side because its SCALE
        # constexpr compiles the loads away, THIS product is E-only: SCALE is 1 on
        # every launch, so a placeholder here would be read as an epsilon.
        if inverse_epsilon is None:
            raise ValueError(
                "the E-side fused pair loads inv_eps on every launch (SCALE=1); "
                "a placeholder binding would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(a)) for a in e_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in
        # int32, so the complex cell count must leave room for the doubling —
        # the same halving the complex predicates apply (module docstring of
        # complex_fields, "Word addressing"). Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexBetaFusedElectricPairPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else complex_beta_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inv_eps, *self._e_coefficients,
            self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            *self.beta_words,
            BACKWARD=self.backward,
            SCALE=self.scale,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexBetaFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, connections={self.connections}, "
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_beta_fused_electric_pair(fields: Any, pml: Any, sources: Any = None,
                                     block: Optional[int] = None,
                                     num_warps: Optional[int] = 1,
                                     kernel: Any = None,
                                     chunk_layout: Any = None,
                                     ) -> Optional[ComplexBetaFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly. ``chunk_layout`` is the run's chunk division, ``None`` for
    the one-rank default (``launch.chunk_bounds_for``).
    """
    if not complex_beta_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    connections, values = connection_arguments(bloch_phase_table(grid, kinds),
                                               backward=bool(curl_spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=False, complex_storage=True)
    return ComplexBetaFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # THE MIRROR IMAGE OF THE MAGNETIC TWIN'S CHOICE. The curl takes the
        # INTEGER lattice on step_D (SUB_STEPS['step_D']['suffix'] == "") and the
        # constitutive the HALF-INTEGER one on E (stepping.py:1015 against the D
        # curl's half_integer=False). The kernel takes both and never asks which is
        # which, so a swap here is a silent half-cell error in the absorber
        # profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}_h") for axis in "xyz"
         for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_complex_beta_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], dtdx: float,
        beta_words,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1, *,
        chunk_bounds: Any) -> ComplexBetaFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    ``beta_words`` is ``(plus, minus)`` — normally
    :func:`.special_kz.beta_curl_coefficients`'s output, or a deliberately wrong
    pair on a mutation leg; ``has_beta=0`` is the identity leg's root-pair arm.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    is keyed by component name plus ``fu_D*``, ``f_w_E*`` and ``inv_eps_Ex``...
    (float32); ``phases`` is the per-axis ``Optional[complex]`` table (None =
    unphased) and the conjugation for a backward sub-step is applied HERE, exactly
    as the engine route applies it — this product is backward-only, so it always
    is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(curl_spec["backward"]))
    return ComplexBetaFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block,
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
        chunk_bounds=chunk_bounds,
    )
