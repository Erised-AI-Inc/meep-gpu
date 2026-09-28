"""The FOLDED special_kz MAGNETIC seam in one launch: folded beta ``step_B``
welded into ``update_H``, mirror fills carried inline.

The ``folded real beta PML`` -> ``folded beta run`` cell's B->H product, and the
MAGNETIC TWIN of :mod:`.folded_beta_fused_electric_pair`. Both halves are
already certified and neither is re-derived here:

* the curl half is :func:`.folded_complex.folded_beta_pml_curl_step` (K3a) taken
  through its ``BACKWARD == 0`` arm — ``symmetry.pml_curl_step_folded``'s
  certified body plus THE SAME three-line beta insert
  ``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``;
* the fills, the wall clear and the constitutive half are
  :func:`.folded_fused_magnetic_pair.folded_fused_curl_constitutive_B`'s,
  byte-copied — the B-family ownership restructure (near on the component's OWN
  axis with the MOVED coefficient index, far on the two others, parity products
  measured against the array path's own three passes), the ``_carry_ghost``
  device function (IMPORTED, not copied, so the two modules cannot drift) and
  the B-family three-row ``zero_metal_B`` block included.

SO THIS KERNEL IS ``folded_fused_curl_constitutive_B``, CHARACTER FOR CHARACTER,
PLUS THE SAME THREE LINES ``folded_beta_pml_curl_step`` ADDS TO
``pml_curl_step_folded``::

    if HAS_BETA:
        curl0 = curl0 - (beta_plus * b)
        curl1 = curl1 - (beta_minus * a)

placed AFTER the ``dtdx`` curl and BEFORE BOTH ownership masks — the array
path's order (S:356-363 after :342, before :369 on the B side), and K3a's own
placement. Real storage is MEEP's implicit-i trick: the SAME sign convention as
the electric twin (the ±i lives only in complex storage; MEASURED on
``special_kz.beta_curl_coefficients``, whose ``magnetic=`` flag changes no byte
on the real path). A reader must be able to diff this body against the two
shipped sources and see nothing else moved; the gate's transcription leg asserts
exactly that.

===========================================================================
WHY THIS CELL EXISTS, AND WHAT IT IS WORTH
===========================================================================

ONE corpus row, TWO seam-instances — the electric twin serves the other.
``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``: shape (420, 212, 1),
beta = 0.2, Mirror(Y) over a PERIODIC declaration (so the FAR fill is live), an
active PML, real storage, and FOUR sources — two electric, two magnetic
(``source_field_types ['D', 'D', 'B', 'B']``). The 2026-08-31 board scores::

    ceiling 1 (rows 1)  B->H  folded real beta PML -> folded beta run  NO PRODUCT

with ``in_seam_source: true`` (the TWO MAGNETIC sources land in THIS seam,
driver.py:3283-3284) and ``in_seam_source_blocks: false``. So
:data:`CARRIES_DEPOSIT_REPAIR` is the product here exactly as it is on the
electric side: at False the source clause refuses the only row the cell has.
The plain folded magnetic pair set the B-seam repair precedent on 2026-08-30;
this module rides the same shipped bracket.

BETA IS 2-D ONLY (grid.py:690), so no admitted configuration can fold z:
``NEAR_Z``/``FAR_Z`` and the z carry blocks are compile-time absent on every
configuration this predicate can admit. The transcription still carries them
verbatim — the body is ``folded_fused_curl_constitutive_B`` plus the insert,
nothing removed.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3281-3289)::

    step_B -> MAGNETIC SOURCES -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — CARRIED, through the SHIPPED deposit repair** with
  the fold image closure (``deposit_repair.repair_cells``), exactly as the
  electric twin carries its electric ones;
* **the two fills — CARRIED INLINE**, byte-for-byte the plain folded magnetic
  product's carry, with its measured ownership rule and moved near-side
  coefficient index;
* **``zero_metal_B`` — CARRIED INLINE**, three rows, byte-copied.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; wired, it
would contend for ``step_B`` with the ``folded real beta PML`` arm, which IS
wired — and ``_select_slot`` (launch.py:1983-1996) leaves a slot with two
admitters UNSELECTED. Nothing in ``launch.py`` names this module;
``fastpath.plan_fast_path`` is unchanged.

Import contract: importable WITHOUT Triton — the predicate and the plan builder
(to ``None``) must answer on the laptop that is the merge bar.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a
dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .folded_complex import (
    folded_beta_pml_curl_coverage,
    folded_beta_run_constitutive_coverage,
)
from .special_kz import beta_curl_coefficients
from .symmetry import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_SOURCE_INDEX,
    _far_reflect_rows,
    _stored_past_owned_reader,
    folded_axis_kinds,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between one seam-instance and zero: the single
#: corpus row declares TWO MAGNETIC sources, which the driver injects between the
#: two halves (driver.py:3283-3284). The flag is a claim about the PLAN this module
#: builds and is only ever changed in the same edit as that wiring. The B-seam
#: repair itself is the one the plain folded magnetic pair already rides
#: (CARRIES_DEPOSIT_REPAIR flipped there 2026-08-30 under a released gate); what
#: is new here is only the beta arm on the curl half, which the repair never reads.
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
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289). Declared, never inferred from the slot name. THE
#: INJECTION IS NOT ONE OF THE FIVE: it is carried by the deposit-repair bracket
#: around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: The two codes a folded axis can take. The NEAR fill runs on either; the FAR
#: fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``).
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without importing :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "RELEASED 2026-09-01 on the GPU host GPU 3 (verified physically empty before the "
    "run), NVIDIA RTX A6000, cc 8.6, Triton 3.1.0 / CuPy 13.5.1, under the 'keep' "
    "float32 subnormal policy (ieee_keep_ftz_stripped) through an ftz-stripped "
    "cache, RE-RUN in place after this paragraph was written -- the gate pins "
    "this module in source_sha256, so the run that certifies it has to be the "
    "one that saw the paragraph. Artifact: "
    "parity/meep_gpu/results/triton_folded_beta_2026-08-31/"
    "folded_beta_fused_magnetic_pair/gate.json, release.released = true. "
    "31 device legs over the electric twin's six folded 2-D grids, every one "
    "of 26 allocated volumes BIT-IDENTICAL on the uint32 view per COMPLETE "
    "driver.step() against BOTH oracles -- the CuPy array path and the three "
    "separately certified Triton products this launch replaces (the 'folded "
    "real beta PML' curl on step_B, the mirror ghost fill on fill_B, and the "
    "'folded beta run' constitutive on update_H; zero_metal_B and the far fill "
    "counted on the array path in the separate oracle). 6 CARRY legs repairing "
    "in-seam MAGNETIC deposits through the SHIPPED deposit_repair bracket with "
    "its fold-image closure, and 2 NULL CONTROLS with the bracket removed that "
    "BOTH diverge as they must. 3 REDUCTION legs, one per value class: "
    "HAS_BETA = 0 reproduces "
    "folded_fused_magnetic_pair.folded_fused_curl_constitutive_B bit for bit, "
    "and the leg records per class whether it also separates the pair from the "
    "array path. 15 kernel mutations: 14 CAUGHT and 1 declared NULL with its "
    "premise measured per run (the near image rebilled at the source row is "
    "byte-invisible because the integer-lattice H coefficients are equal at "
    "rows 0 and 2 on every admissible folded configuration — a folded axis "
    "absorbs on its HIGH face only; the FIRST device run is what found it, and "
    "the catchable far-side rebilling was added in its place). 3/3 host "
    "mutations caught, 5/5 refusals, 4/4 no-device legs. "
    "|| WHAT THE RUN DID NOT ESTABLISH: nothing is TIMED here and no throughput "
    "claim is admissible; and the product is NOT WIRED, so nothing in a default "
    "run reaches this plan. This module has no fingerprints.json entry -- the "
    "gate artifact's own source_sha256 map is the binding, and the fusion matrix "
    "reads it.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "DEVICE_STATUS", "MIRROR_CODES", "REPLACES",
    "FoldedBetaFusedMagneticPairPlan",
    "folded_beta_fused_curl_constitutive_B",
    "folded_beta_fused_curl_constitutive_B_kernel",
    "folded_beta_fused_magnetic_pair_coverage",
    "mirror_phases",
    "plan_folded_beta_fused_magnetic_pair",
    "plan_folded_beta_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    from .folded_fused_magnetic_pair import _carry_ghost  # noqa: E402 - certified

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def folded_beta_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx, By, Bz
        u0, u1, u2,                       # curl auxiliaries: fu_Bx, fu_By, fu_Bz
        g0, g1, g2,                       # curl sources: Ex, Ey, Ez
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx, Hy, Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx, f_w_Hy, f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        BACKWARD: tl.constexpr,           # bound to 0 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # compiled in only for beta != 0 runs
        BLOCK: tl.constexpr,
    ):
        """beta ``step_B`` + ``fill_B`` + ``zero_metal_B`` + far fill + ``update_H``.

        :func:`.folded_fused_magnetic_pair.folded_fused_curl_constitutive_B`'s body
        with :func:`.folded_complex.folded_beta_pml_curl_step`'s constexpr-gated
        insert placed where the array path places it — after the ``dtdx`` curl and
        before BOTH ownership masks (S:356-363 after :342, before :369). Nothing
        else moves.

        ``BACKWARD`` IS CARRIED AND MUST BE 0: both carries below are transcribed
        for the B family's Yee shifts. ``HAS_BETA`` MUST BE 1 on any admitted
        configuration; it is carried rather than hard-coded so the gate has a
        certified-kernel arm — at ``HAS_BETA = 0`` this body must reproduce
        :func:`.folded_fused_magnetic_pair.folded_fused_curl_constitutive_B` bit
        for bit, which is a leg rather than an argument.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny
        # Index 0 on any axis, as a per-lane tensor: the near fill's destination
        # coefficient index.
        origin = idx * 0

        # ======================= the curl half ================================
        # Verbatim from symmetry.pml_curl_step_folded via
        # folded_fused_magnetic_pair.folded_fused_curl_constitutive_B; the beta
        # insert is the ONLY added statement group.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == PERIODIC:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        else:
            vx = live & (si >= 0) & (si < nx)
        if BCY == PERIODIC:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        else:
            vy = live & (sj >= 0) & (sj < ny)
        if BCZ == PERIODIC:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        else:
            vz = live & (sk >= 0) & (sk < nz)

        ox = si * nyz + j * nz + k
        oy = i * nyz + sj * nz + k
        oz = i * nyz + j * nz + sk

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_y = tl.load(g0 + oy, mask=vy, other=0.0)
        a_z = tl.load(g0 + oz, mask=vz, other=0.0)
        b_x = tl.load(g1 + ox, mask=vx, other=0.0)
        b_z = tl.load(g1 + oz, mask=vz, other=0.0)
        c_x = tl.load(g2 + ox, mask=vx, other=0.0)
        c_y = tl.load(g2 + oy, mask=vy, other=0.0)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- the beta term (stepping._special_kz_beta_term), CENTER partners only
        # AFTER the dtdx curl, BEFORE BOTH ownership masks — the array path's
        # order, and folded_beta_pml_curl_step's own placement. No dtdx on the
        # term (analytic derivative, S:733-735); real storage takes the SAME sign
        # convention as the electric twin (the implicit-i trick).
        if HAS_BETA:
            curl0 = curl0 - (beta_plus * b)
            curl1 = curl1 - (beta_minus * a)

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells :1865) -----
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY != PERIODIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ != PERIODIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX != PERIODIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ != PERIODIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX != PERIODIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY != PERIODIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX != PERIODIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY != PERIODIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ != PERIODIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        # --- ownership mask, the TOP plane of a folded PERIODIC axis ------------
        # The BACKWARD == 0 arm: Bx (0,1,1) By (1,0,1) Bz (1,1,0) — shift 1 on the
        # two OTHER axes, six lines against the D arm's three.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BCY == MIRROR_PERIODIC:
            curl0 = tl.where(last_y, 0.0, curl0)
        if BCZ == MIRROR_PERIODIC:
            curl0 = tl.where(last_z, 0.0, curl0)
        if BCX == MIRROR_PERIODIC:
            curl1 = tl.where(last_x, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl1 = tl.where(last_z, 0.0, curl1)
        if BCX == MIRROR_PERIODIC:
            curl2 = tl.where(last_x, 0.0, curl2)
        if BCY == MIRROR_PERIODIC:
            curl2 = tl.where(last_y, 0.0, curl2)

        # --- ownership of the fill destinations --------------------------------
        own0, own1, own2 = live, live, live
        if NEAR_X:
            own0 = own0 & (i != 0)
        if FAR_Y:
            own0 = own0 & (j != ny - 1)
        if FAR_Z:
            own0 = own0 & (k != nz - 1)
        if NEAR_Y:
            own1 = own1 & (j != 0)
        if FAR_X:
            own1 = own1 & (i != nx - 1)
        if FAR_Z:
            own1 = own1 & (k != nz - 1)
        if NEAR_Z:
            own2 = own2 & (k != 0)
        if FAR_X:
            own2 = own2 & (i != nx - 1)
        if FAR_Y:
            own2 = own2 & (j != ny - 1)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=own0, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=own1, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=own2, other=0.0) * km_y) + n2) - p2) * si_y

        # --- the seam: stepping.zero_metal_B (driver.py:3286) ------------------
        if ZM_X:
            v0 = tl.where(at_x, 0.0, v0)
        if ZM_Y:
            v1 = tl.where(at_y, 0.0, v1)
        if ZM_Z:
            v2 = tl.where(at_z, 0.0, v2)

        # fu is written at EVERY cell, destinations included: the array path's
        # step_B writes it everywhere and fill_symmetry_bc_B does not touch it.
        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=own0)
        tl.store(f1 + idx, v1, mask=own1)
        tl.store(f2 + idx, v2, mask=own2)

        # ==================== the constitutive half ===========================
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=own0, other=0.0)   # BEFORE the store.
        src0 = v0
        tl.store(w0 + idx, src0, mask=own0)
        a0v = tl.load(h0 + idx, mask=own0, other=0.0)
        a0v = a0v + kp_0 * src0
        a0v = a0v - km_0 * prev0
        tl.store(h0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1, other=0.0)
        src1 = v1
        tl.store(w1 + idx, src1, mask=own1)
        a1v = tl.load(h1 + idx, mask=own1, other=0.0)
        a1v = a1v + kp_1 * src1
        a1v = a1v - km_1 * prev1
        tl.store(h1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2, other=0.0)
        src2 = v2
        tl.store(w2 + idx, src2, mask=own2)
        a2v = tl.load(h2 + idx, mask=own2, other=0.0)
        a2v = a2v + kp_2 * src2
        a2v = a2v - km_2 * prev2
        tl.store(h2 + idx, a2v, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # Byte-copied from folded_fused_curl_constitutive_B — the measured
        # ownership rule, the moved coefficient index on the NEAR half, and the
        # ownership AND on every carry mask. The values imaged already carry the
        # beta increment: the fills run AFTER step_B on the array path and image
        # the array the beta term was added to.
        kp_d0 = tl.load(kp0 + origin, mask=live, other=0.0)
        km_d0 = tl.load(km0 + origin, mask=live, other=0.0)
        kp_d1 = tl.load(kp1 + origin, mask=live, other=0.0)
        km_d1 = tl.load(km1 + origin, mask=live, other=0.0)
        kp_d2 = tl.load(kp2 + origin, mask=live, other=0.0)
        km_d2 = tl.load(km2 + origin, mask=live, other=0.0)

        near_i = live & (i == 2)
        near_j = live & (j == 2)
        near_k = live & (k == 2)
        far_i = live & (i == rx)
        far_j = live & (j == ry)
        far_k = live & (k == rz)
        dn_x = -2 * nyz
        dn_y = -2 * nz
        dn_z = -2
        df_x = (nx - 1 - rx) * nyz
        df_y = (ny - 1 - ry) * nz
        df_z = (nz - 1 - rz)

        # --- component 0 (Bx): near on x, far on y and z -----------------------
        if FAR_Y:
            _carry_ghost(f0, w0, h0, idx + df_y, -PHY * v0, kp_0, km_0, own0 & far_j)
        if FAR_Z:
            _carry_ghost(f0, w0, h0, idx + df_z, -PHZ * v0, kp_0, km_0, own0 & far_k)
        if FAR_Y and FAR_Z:
            _carry_ghost(f0, w0, h0, idx + df_y + df_z, PHY * PHZ * v0,
                         kp_0, km_0, own0 & far_j & far_k)
        if NEAR_X:
            _carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_d0, km_d0, own0 & near_i)
        if NEAR_X and FAR_Y:
            _carry_ghost(f0, w0, h0, idx + dn_x + df_y, -PHX * PHY * v0,
                         kp_d0, km_d0, own0 & near_i & far_j)
        if NEAR_X and FAR_Z:
            _carry_ghost(f0, w0, h0, idx + dn_x + df_z, -PHX * PHZ * v0,
                         kp_d0, km_d0, own0 & near_i & far_k)
        if NEAR_X and (FAR_Y and FAR_Z):
            _carry_ghost(f0, w0, h0, idx + dn_x + df_y + df_z,
                         PHX * PHY * PHZ * v0, kp_d0, km_d0,
                         own0 & near_i & far_j & far_k)

        # --- component 1 (By): near on y, far on x and z -----------------------
        if FAR_X:
            _carry_ghost(f1, w1, h1, idx + df_x, -PHX * v1, kp_1, km_1, own1 & far_i)
        if FAR_Z:
            _carry_ghost(f1, w1, h1, idx + df_z, -PHZ * v1, kp_1, km_1, own1 & far_k)
        if FAR_X and FAR_Z:
            _carry_ghost(f1, w1, h1, idx + df_x + df_z, PHX * PHZ * v1,
                         kp_1, km_1, own1 & far_i & far_k)
        if NEAR_Y:
            _carry_ghost(f1, w1, h1, idx + dn_y, PHY * v1, kp_d1, km_d1, own1 & near_j)
        if NEAR_Y and FAR_X:
            _carry_ghost(f1, w1, h1, idx + dn_y + df_x, -PHY * PHX * v1,
                         kp_d1, km_d1, own1 & near_j & far_i)
        if NEAR_Y and FAR_Z:
            _carry_ghost(f1, w1, h1, idx + dn_y + df_z, -PHY * PHZ * v1,
                         kp_d1, km_d1, own1 & near_j & far_k)
        if NEAR_Y and (FAR_X and FAR_Z):
            _carry_ghost(f1, w1, h1, idx + dn_y + df_x + df_z,
                         PHY * PHX * PHZ * v1, kp_d1, km_d1,
                         own1 & near_j & far_i & far_k)

        # --- component 2 (Bz): near on z, far on x and y -----------------------
        if FAR_X:
            _carry_ghost(f2, w2, h2, idx + df_x, -PHX * v2, kp_2, km_2, own2 & far_i)
        if FAR_Y:
            _carry_ghost(f2, w2, h2, idx + df_y, -PHY * v2, kp_2, km_2, own2 & far_j)
        if FAR_X and FAR_Y:
            _carry_ghost(f2, w2, h2, idx + df_x + df_y, PHX * PHY * v2,
                         kp_2, km_2, own2 & far_i & far_j)
        if NEAR_Z:
            _carry_ghost(f2, w2, h2, idx + dn_z, PHZ * v2, kp_d2, km_d2, own2 & near_k)
        if NEAR_Z and FAR_X:
            _carry_ghost(f2, w2, h2, idx + dn_z + df_x, -PHZ * PHX * v2,
                         kp_d2, km_d2, own2 & near_k & far_i)
        if NEAR_Z and FAR_Y:
            _carry_ghost(f2, w2, h2, idx + dn_z + df_y, -PHZ * PHY * v2,
                         kp_d2, km_d2, own2 & near_k & far_j)
        if NEAR_Z and (FAR_X and FAR_Y):
            _carry_ghost(f2, w2, h2, idx + dn_z + df_x + df_y,
                         PHZ * PHX * PHY * v2, kp_d2, km_d2,
                         own2 & near_k & far_i & far_j)

else:  # pragma: no cover - laptop path
    folded_beta_fused_curl_constitutive_B = None  # type: ignore[assignment]


def folded_beta_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_beta_fused_curl_constitutive_B is None:
        raise ImportError(
            "the folded beta fused magnetic B/H kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return folded_beta_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def mirror_phases(grid: Any) -> Tuple[int, int, int]:
    """``PHX``/``PHY``/``PHZ``: the declared plane phase per axis, 0 where unfolded.

    The same derivation as :func:`.folded_fused_magnetic_pair.mirror_phases`,
    restated because this module's builder bakes the result into its own
    constexprs and may not read its inputs off another module's guard.
    """
    phases: List[int] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            phases.append(0)
            continue
        value = _call(grid, "mirror_phase", axis, default=None)
        phases.append(int(value) if value in (1, -1) else 0)
    return (phases[0], phases[1], phases[2])


def folded_beta_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None) -> Coverage:
    """May ONE launch span beta ``step_B`` -> fills -> wall -> ``update_H``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. The two
    halves are the arms ``launch.plan_step`` really selects on a folded beta row
    today (``folded real beta PML`` on ``step_B``, ``folded beta run`` on
    ``update_H``). The seam clauses are
    :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`'s,
    restated for the same kernel geometry.
    """
    reasons: List[str] = []

    curl = folded_beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded beta curl half: {reason}"
                       for reason in curl.reasons)
    magnetic = folded_beta_run_constitutive_coverage(fields, pml,
                                                     CONSTITUTIVE_SIDE)
    if not magnetic.covered:
        reasons.extend(f"folded beta constitutive half: {reason}"
                       for reason in magnetic.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything.
    # The one corpus row declares TWO magnetic sources, injected BETWEEN the two
    # halves (driver.py:3283-3284); CARRIES_DEPOSIT_REPAIR is what lets the repair
    # carry them, with `repairable`'s fold clauses live. An ELECTRIC source is
    # injected in the D/E half and does not disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR CARRY'S OWN CONDITIONS, transcribed from
        # folded_fused_magnetic_pair_coverage — the same kernel geometry, the same
        # ownership move, the same row checks.
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
            reader = _stored_past_owned_reader()
            if reader is None:  # pragma: no cover - folded_axis_kinds refused first
                reasons.append(
                    "stepping._stored_past_owned is not importable; the far carry "
                    "cannot be checked against the array path's own predicate")
                break
            past_owned = bool(reader(grid, axis))
            if periodic != past_owned:
                reasons.append(
                    f"axis {axis} code {int(code)} and "
                    f"stepping._stored_past_owned {past_owned} disagree; the kernel "
                    f"would mask the top plane and image the far ghost on different "
                    f"axes than the array path")
            if not periodic or len(shape) != 3:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(
                    f"axis {axis} is a folded PERIODIC axis with no reflect row "
                    f"from stepping._far_reflect_rows; the far carry has nothing "
                    f"to image")
                continue
            row, extent = int(row), int(shape[axis])
            if not (0 <= row < extent - 1):
                reasons.append(
                    f"axis {axis} reflect row {row} is outside [0, {extent - 1}); "
                    f"the far carry would image the plane it writes, or read "
                    f"outside the allocation")
            if row == 0:
                reasons.append(
                    f"axis {axis} reflect row is 0, the plane the NEAR fill writes: "
                    f"the far carry would image a ghost rather than an owned cell")
            if extent - 1 == MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} stores {extent} cells, so the far carry's write "
                    f"plane IS the near fill's read plane "
                    f"(cell {MIRROR_SOURCE_INDEX})")

        # THE NEAR FILL IMAGES STORED CELL 2 FROM THAT CELL'S OWN LANE.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                        f"the near fill images stored cell {MIRROR_SOURCE_INDEX} and "
                        f"this kernel images it from that cell's own lane")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall there; "
                    f"stepping._zero_metal skips a folded axis, so the two passes "
                    f"have drifted and the carry's disjointness no longer holds")

        # The phases the kernel bakes must be exactly the ones the array path uses.
        phases = mirror_phases(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
                reasons.append(
                    f"axis {axis} is folded but its mirror phase resolves to "
                    f"{phases[axis]!r}, not +1 or -1")

    # The wall clear is carried inline, so the grid must be able to answer which
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips
    # a plane the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedBetaFusedMagneticPairPlan:
    """One allocation-free launch for FIVE of the driver's magnetic passes.

    :class:`.folded_fused_magnetic_pair.FoldedFusedMagneticPairPlan`'s bindings
    plus the two host-rounded beta scalars and the ``HAS_BETA`` constexpr.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "beta_plus", "beta_minus", "has_beta",
        "backward", "bc", "zero_metal", "phases", "reflect", "near", "far",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_h_targets", "_h_aux", "_h_coefficients", "_grid", "_kernel",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_beta: int = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (special_kz.beta_curl_coefficients);
        # float() keeps the bits, and Triton types a Python float argument as fp32.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = int(has_beta)
        self.backward = BACKWARD
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phases = tuple(int(value) for value in phases)
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT — the same rule as the
        # plain folded plan, for the same reason.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        if not any(self.near):
            raise ValueError(
                "no axis is folded: this product exists to carry the mirror fills "
                "inside the B seam of a beta run, and an unfolded beta grid belongs "
                "to beta_fused_magnetic_pair")
        for axis, code in enumerate(self.bc):
            folded = code in MIRROR_CODES
            if folded and self.phases[axis] not in (1, -1):
                raise ValueError(
                    f"axis {axis} is folded but its baked phase is "
                    f"{self.phases[axis]!r}, not +1 or -1")
            if folded and self.zero_metal[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a wall clear; "
                    f"stepping._zero_metal skips a folded axis")
            if code == CODE_MIRROR_PERIODIC and not (
                    0 <= self.reflect[axis] < int(shape[axis]) - 1):
                raise ValueError(
                    f"axis {axis} is folded PERIODIC with reflect row "
                    f"{self.reflect[axis]}, which is outside "
                    f"[0, {int(shape[axis]) - 1}); the far carry would image the "
                    f"plane it writes, or read outside the allocation")
            if code != CODE_MIRROR_PERIODIC and self.reflect[axis] != -1:
                raise ValueError(
                    f"axis {axis} is not folded PERIODIC but carries reflect row "
                    f"{self.reflect[axis]}; stepping._far_reflect_rows answers None "
                    f"there and the kernel would read a row nothing wrote")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._h_targets = tuple(CupyPointer(array) for array in h_targets)
        self._h_aux = tuple(CupyPointer(array) for array in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(array)) for array in h_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs. Dropping
        # it is not a slowdown, it is a DISARMING.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_beta_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            self.reflect[0], self.reflect[1], self.reflect[2],
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phases[0], PHY=self.phases[1], PHZ=self.phases[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedBetaFusedMagneticPairPlan(shape={self.shape}, "
                f"bc={self.bc}, phases={self.phases}, "
                f"zero_metal={self.zero_metal}, near={self.near}, far={self.far}, "
                f"reflect={self.reflect}, beta_plus={self.beta_plus!r}, "
                f"has_beta={self.has_beta}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_folded_beta_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None) -> Optional[FoldedBetaFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not folded_beta_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    # THE SAME FUNCTION THE CURL PLAN CALLS, with `magnetic` from the sub-step.
    # On REAL storage the two flag values are MEASURED IDENTICAL.
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(CURL_SUB_STEP == "step_B"),
                                         complex_storage=False)
    return FoldedBetaFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        mirror_phases(grid), plus, minus,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H. The kernel takes both and never asks which is
        # which; the gate carries a mutation for the swap.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_beta_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, phases,
        dtdx: float, beta_plus: float, beta_minus: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1,
        reflect: Any = (None, None, None)) -> FoldedBetaFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.
    ``kernel=`` carries the mutation override and ``has_beta=0`` the reduction
    leg's certified-kernel arm, which must reproduce
    ``folded_fused_magnetic_pair.folded_fused_curl_constitutive_B`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedBetaFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, phases, beta_plus, beta_minus,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        reflect=reflect,
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )
