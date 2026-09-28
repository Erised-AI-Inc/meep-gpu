"""The FOLDED special_kz ELECTRIC seam in one launch: folded beta ``step_D``
welded into ``update_E``, mirror fills carried inline.

The ``folded real beta PML`` -> ``folded beta run`` cell's D->E product. Both
halves are already certified and neither is re-derived here:

* the curl half is :func:`.folded_complex.folded_beta_pml_curl_step` (K3a) —
  ``symmetry.pml_curl_step_folded``'s certified body plus THE SAME three-line
  beta insert ``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``,
  taken through its ``BACKWARD == 1`` arm;
* the fills, the wall clear and the constitutive half are
  :func:`.folded_fused_pair.folded_fused_curl_constitutive_D`'s, byte-copied —
  the ownership restructure, the chained near/far composition, the
  ``_carry_ghost_E`` device function (IMPORTED, not copied, so the two modules
  cannot drift) and the D-family six-row ``zero_metal_D`` block included;
* and the WELD is the one substitution ``folded_fused_curl_constitutive_D``
  already makes between the folded curl and ``update_E`` on a beta = 0 run.

SO THIS KERNEL IS ``folded_fused_curl_constitutive_D``, CHARACTER FOR CHARACTER,
PLUS THE SAME THREE LINES ``folded_beta_pml_curl_step`` ADDS TO
``pml_curl_step_folded``::

    if HAS_BETA:
        curl0 = curl0 - (beta_plus * b)
        curl1 = curl1 - (beta_minus * a)

placed AFTER the ``dtdx`` curl and BEFORE BOTH ownership masks — the array
path's order (S:438-445 after :429, before :450), and K3a's own placement. A
reader must be able to diff this body against those two and see nothing else
moved; the gate's transcription leg asserts exactly that, statement by
statement, against both shipped sources.

WHY THE BETA SLOT IS LOAD-BEARING UNDER A FOLD — ``folded_complex.py``'s module
docstring carries the measurement: the fold WIDENS the ownership mask from
METALLIC-only at cell 0 to non-PERIODIC at cell 0 PLUS ``MIRROR_PERIODIC`` at
the top plane, and beta touches targets 0 and 1 on both sub-steps, so one folded
axis makes BOTH mask arms fire on BOTH beta targets. A kernel inserting beta
below the masks leaves a live increment on the mirror plane and on the far ghost
plane, invisible in the interior. THE FILLS THEN IMAGE THE STEPPED VALUE, beta
included: the carries below read ``v0``/``v1``/``v2`` after the recurrence, so a
ghost cell's beta content is its source lane's, exactly as
``stepping._write_mirror_ghost`` images the array the beta term was added to.

===========================================================================
WHY THIS CELL EXISTS, AND WHAT IT IS WORTH
===========================================================================

ONE corpus row, TWO seam-instances — this module's B->H twin serves the other.
``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``: shape (420, 212, 1),
beta = 0.2, Mirror(Y) over a PERIODIC declaration (so the FAR fill is live),
an active PML, real storage, and FOUR sources — two electric, two magnetic
(``source_field_types ['D', 'D', 'B', 'B']``). The 2026-08-31 board
(``results/fusion_matrix_triton_2026-08-31_plainrepair7``) scores the cell::

    ceiling 1 (rows 1)  D->E  folded real beta PML -> folded beta run  NO PRODUCT

with ``in_seam_source: true`` and ``in_seam_source_blocks: false`` — the
injection lands between the two halves and ``deposit_repair`` can carry it. So
:data:`CARRIES_DEPOSIT_REPAIR` is not one clause of this product, it IS the
product: at False the source clause refuses the only row the cell has.

BETA IS 2-D ONLY, and that bounds the kernel's reachable branches structurally:
``Grid`` refuses a nonzero beta off a zero-z-extent cell BY NAME (grid.py:690;
MEEP fields.cpp:546-547 aborts on the same combination), so no admitted
configuration can fold z. ``NEAR_Z``/``FAR_Z`` and every ``gv_*z`` register are
compile-time absent on EVERY configuration this predicate can admit — unlike the
plain folded family, where 2-D was a case-table gap the 3-D cases had to close.
The transcription still carries those blocks verbatim: the body is
``folded_fused_curl_constitutive_D`` plus the insert, nothing removed.

THE SIBLINGS ON THE OTHER BACKEND are ``metal_kernels/folded_beta_real_fused_pair.py``
and its magnetic twin, which serve the same cells there. This module is the
Triton twin: the same certified halves, each transcribed from ITS OWN backend's
body.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.**
  :class:`.deposit_repair.LeadingRepairPlan` saves the state at the deposit
  points AND at every cell the two post-injection fills image them into
  (``deposit_repair.repair_cells``), and :class:`.deposit_repair.TrailingRepairPlan`
  recomputes them after, in the driver's own order. IGNORANCE IS STILL NEVER AN
  EMPTY SET: an undeclared ``sources`` is a REFUSAL, and a source that cannot
  publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — CARRIED INLINE**,
  byte-for-byte the plain folded product's carry: the near fill's chained
  parity-then-clear, the far fill's clear-then-parity, and every composition,
  measured 156/156 against the array path's own three passes
  (``results/d_side_composition_law_2026-08-21/``);
* **``zero_metal_D`` — CARRIED INLINE** through :func:`.coverage.zero_metal_axes`,
  six rows on this side, byte-copied from ``folded_fused_curl_constitutive_D``.

The corpus row is all-periodic with ``has_metallic == false``, so the inline
``zero_metal_D`` carry compiles to three ``False`` flags ON THAT ROW — the gate
scores every wall-clear mutation on a WALLED case instead, where the flags are
real.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; wired, it
would also contend for ``step_D`` with the ``folded real beta PML`` arm, which
IS wired and admits the same configurations — and ``_select_slot``
(launch.py:1983-1996) leaves a slot with two admitters UNSELECTED, taking the
certified curl off the device with it. Nothing in ``launch.py`` names this
module; ``fastpath.plan_fast_path`` is unchanged.

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
#: corpus row declares TWO ELECTRIC sources, which the driver injects between the
#: two halves (driver.py:3294-3299). The flag is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is
#: only ever changed in the same edit as that wiring.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a cylindrical r = 0 axis, a fold
#: whose fill map it cannot read, a fold too short to hold the near fill's source
#: row, and an absorber whose split-field recurrence never ran. On THIS family the
#: fold clauses are live: ``_folded_seam_reasons`` reads the same fill map the
#: kernel's forward carry implements, so the repair's inverse and the kernel's
#: forward image are the same closed form or the predicate refuses.
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

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. THE
#: INJECTION IS NOT ONE OF THE FIVE: it is carried by the deposit-repair bracket
#: around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The two codes a folded axis can take. The NEAR fill runs on either; the FAR
#: fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``).
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without importing :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "RELEASED 2026-09-01 on the GPU host GPU 3 (verified physically empty before the "
    "run), NVIDIA RTX A6000, cc 8.6, Triton 3.1.0 / CuPy 13.5.1, under the 'keep' "
    "float32 subnormal policy (ieee_keep_ftz_stripped) through an ftz-stripped "
    "cache, RE-RUN in place after this paragraph was written -- the gate pins "
    "this module in source_sha256, so the run that certifies it has to be the "
    "one that saw the paragraph. Artifact: "
    "parity/meep_gpu/results/triton_folded_beta_2026-08-31/"
    "folded_beta_fused_electric_pair/gate.json, release.released = true. "
    "31 device legs over six folded 2-D grids (both terminations, both "
    "parities, odd counts, two-fold composition depth 3), every one of 26 "
    "allocated volumes BIT-IDENTICAL on the uint32 view per COMPLETE "
    "driver.step() against BOTH oracles -- the CuPy array path and the three "
    "separately certified Triton products this launch replaces (the 'folded "
    "real beta PML' curl on step_D, the mirror ghost fill on fill_D, and the "
    "'folded beta run' constitutive on update_E; zero_metal_D and the far fill "
    "counted on the array path in the separate oracle). 6 CARRY legs repairing "
    "in-seam electric deposits through the SHIPPED deposit_repair bracket with "
    "its fold-image closure, and 2 NULL CONTROLS with the bracket removed that "
    "BOTH diverge as they must. 3 REDUCTION legs, one per value class: "
    "HAS_BETA = 0 reproduces folded_fused_pair.folded_fused_curl_constitutive_D "
    "bit for bit, and the leg records per class whether it also separates the "
    "pair from the array path. 14/14 kernel mutations caught, 3/3 host "
    "mutations caught, 5/5 refusals, 4/4 no-device legs. "
    "|| WHAT THE RUN DID NOT ESTABLISH: nothing is TIMED here and no throughput "
    "claim is admissible; and the product is NOT WIRED, so nothing in a default "
    "run reaches this plan. This module has no fingerprints.json entry -- the "
    "gate artifact's own source_sha256 map is the binding, and the fusion matrix "
    "reads it.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "DEVICE_STATUS", "MIRROR_CODES", "REPLACES",
    "FoldedBetaFusedElectricPairPlan",
    "folded_beta_fused_curl_constitutive_D",
    "folded_beta_fused_curl_constitutive_D_kernel",
    "folded_beta_fused_electric_pair_coverage",
    "mirror_phases",
    "plan_folded_beta_fused_electric_pair",
    "plan_folded_beta_fused_electric_pair_from_arrays",
]


if triton is not None:

    from .folded_fused_pair import _carry_ghost_E  # noqa: E402 - certified, shared

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def folded_beta_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx, Dy, Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz
        g0, g1, g2,                       # curl sources: Hx, Hy, Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex, Ey, Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex, f_w_Ey, f_w_Ez
        ie0, ie1, ie2,                    # inverse epsilon, per target
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # compiled in only for beta != 0 runs
        BLOCK: tl.constexpr,
    ):
        """beta ``step_D`` + ``fill_D`` + ``zero_metal_D`` + far fill + ``update_E``.

        :func:`.folded_fused_pair.folded_fused_curl_constitutive_D`'s body with
        :func:`.folded_complex.folded_beta_pml_curl_step`'s constexpr-gated insert
        placed where the array path places it — after the ``dtdx`` curl and before
        BOTH ownership masks (S:438-445 after :429, before :450). Nothing else
        moves.

        ``BACKWARD`` IS CARRIED AND MUST BE 1: both carries below are transcribed
        for the D family's Yee shifts. ``HAS_BETA`` MUST BE 1 on any admitted
        configuration — the beta family's clause is inverted and requires a nonzero
        ``grid.beta``. It is carried rather than hard-coded so the gate has a
        certified-kernel arm: at ``HAS_BETA = 0`` this body must reproduce
        :func:`.folded_fused_pair.folded_fused_curl_constitutive_D` bit for bit,
        which is a leg rather than an argument.

        The remaining constexprs and runtime rows are exactly the plain folded
        product's; see that kernel's docstring for the ownership restructure and
        the chained near/far composition.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from symmetry.pml_curl_step_folded via
        # folded_fused_pair.folded_fused_curl_constitutive_D; the beta insert is
        # the ONLY added statement group.

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
        # term (analytic derivative, S:733-735); the subtraction IS the array
        # path's `curl + (-(c*g))` by IEEE-754's definition of subtraction.
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

        # --- ownership mask, the TOP plane of a folded PERIODIC axis -----------
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BCX == MIRROR_PERIODIC:
            curl0 = tl.where(last_x, 0.0, curl0)
        if BCY == MIRROR_PERIODIC:
            curl1 = tl.where(last_y, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl2 = tl.where(last_z, 0.0, curl2)

        # --- ownership of the fill destinations --------------------------------
        own0, own1, own2 = live, live, live
        if NEAR_Y:
            own0 = own0 & (j != 0)
            own2 = own2 & (j != 0)
        if NEAR_Z:
            own0 = own0 & (k != 0)
            own1 = own1 & (k != 0)
        if NEAR_X:
            own1 = own1 & (i != 0)
            own2 = own2 & (i != 0)
        if FAR_X:
            own0 = own0 & (i != nx - 1)
        if FAR_Y:
            own1 = own1 & (j != ny - 1)
        if FAR_Z:
            own2 = own2 & (k != nz - 1)

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

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        # fu is written at EVERY cell, destinations included: the array path's
        # step_D writes it everywhere and fill_symmetry_bc_D does not touch it.
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
        src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)
        tl.store(w0 + idx, src0, mask=own0)
        a0v = tl.load(e0 + idx, mask=own0, other=0.0)
        a0v = a0v + kp_0 * src0
        a0v = a0v - km_0 * prev0
        tl.store(e0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=own1, other=0.0)
        tl.store(w1 + idx, src1, mask=own1)
        a1v = tl.load(e1 + idx, mask=own1, other=0.0)
        a1v = a1v + kp_1 * src1
        a1v = a1v - km_1 * prev1
        tl.store(e1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=own2, other=0.0)
        tl.store(w2 + idx, src2, mask=own2)
        a2v = tl.load(e2 + idx, mask=own2, other=0.0)
        a2v = a2v + kp_2 * src2
        a2v = a2v - km_2 * prev2
        tl.store(e2 + idx, a2v, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # Byte-copied from folded_fused_curl_constitutive_D — the chained
        # composition, the moved coefficient index on the FAR half, and the
        # ownership AND on every carry mask. The values imaged already carry the
        # beta increment: the fills run AFTER step_D on the array path and image
        # the array the beta term was added to.
        top_x = idx * 0 + (nx - 1)
        top_y = idx * 0 + (ny - 1)
        top_z = idx * 0 + (nz - 1)
        kp_f0 = tl.load(kp0 + top_x, mask=live, other=0.0)
        km_f0 = tl.load(km0 + top_x, mask=live, other=0.0)
        kp_f1 = tl.load(kp1 + top_y, mask=live, other=0.0)
        km_f1 = tl.load(km1 + top_y, mask=live, other=0.0)
        kp_f2 = tl.load(kp2 + top_z, mask=live, other=0.0)
        km_f2 = tl.load(km2 + top_z, mask=live, other=0.0)

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

        # --- component 0 (Dx): near on y and z, far on x -----------------------
        if NEAR_Y:
            gv_0y = PHY * v0
            if ZM_Y:
                gv_0y = tl.where(at_y, 0.0, gv_0y)
            if ZM_Z:
                gv_0y = tl.where(at_z, 0.0, gv_0y)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,
                           kp_0, km_0, own0 & near_j)
        if NEAR_Z:
            gv_0z = PHZ * v0
            if ZM_Y:
                gv_0z = tl.where(at_y, 0.0, gv_0z)
            if ZM_Z:
                gv_0z = tl.where(at_z, 0.0, gv_0z)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_z, gv_0z,
                           kp_0, km_0, own0 & near_k)
        if NEAR_Y and NEAR_Z:
            gv_0yz = PHZ * (PHY * v0)
            if ZM_Y:
                gv_0yz = tl.where(at_y, 0.0, gv_0yz)
            if ZM_Z:
                gv_0yz = tl.where(at_z, 0.0, gv_0yz)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y + dn_z, gv_0yz,
                           kp_0, km_0, own0 & near_j & near_k)
        if FAR_X:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x, -PHX * v0,
                           kp_f0, km_f0, own0 & far_i)
        if FAR_X and NEAR_Y:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y, -PHX * gv_0y,
                           kp_f0, km_f0, own0 & far_i & near_j)
        if FAR_X and NEAR_Z:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_z, -PHX * gv_0z,
                           kp_f0, km_f0, own0 & far_i & near_k)
        if FAR_X and (NEAR_Y and NEAR_Z):
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y + dn_z, -PHX * gv_0yz,
                           kp_f0, km_f0, own0 & far_i & near_j & near_k)

        # --- component 1 (Dy): near on x and z, far on y -----------------------
        if NEAR_X:
            gv_1x = PHX * v1
            if ZM_X:
                gv_1x = tl.where(at_x, 0.0, gv_1x)
            if ZM_Z:
                gv_1x = tl.where(at_z, 0.0, gv_1x)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x, gv_1x,
                           kp_1, km_1, own1 & near_i)
        if NEAR_Z:
            gv_1z = PHZ * v1
            if ZM_X:
                gv_1z = tl.where(at_x, 0.0, gv_1z)
            if ZM_Z:
                gv_1z = tl.where(at_z, 0.0, gv_1z)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_z, gv_1z,
                           kp_1, km_1, own1 & near_k)
        if NEAR_X and NEAR_Z:
            gv_1xz = PHZ * (PHX * v1)
            if ZM_X:
                gv_1xz = tl.where(at_x, 0.0, gv_1xz)
            if ZM_Z:
                gv_1xz = tl.where(at_z, 0.0, gv_1xz)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x + dn_z, gv_1xz,
                           kp_1, km_1, own1 & near_i & near_k)
        if FAR_Y:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y, -PHY * v1,
                           kp_f1, km_f1, own1 & far_j)
        if FAR_Y and NEAR_X:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x, -PHY * gv_1x,
                           kp_f1, km_f1, own1 & far_j & near_i)
        if FAR_Y and NEAR_Z:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_z, -PHY * gv_1z,
                           kp_f1, km_f1, own1 & far_j & near_k)
        if FAR_Y and (NEAR_X and NEAR_Z):
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x + dn_z, -PHY * gv_1xz,
                           kp_f1, km_f1, own1 & far_j & near_i & near_k)

        # --- component 2 (Dz): near on x and y, far on z -----------------------
        if NEAR_X:
            gv_2x = PHX * v2
            if ZM_X:
                gv_2x = tl.where(at_x, 0.0, gv_2x)
            if ZM_Y:
                gv_2x = tl.where(at_y, 0.0, gv_2x)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x, gv_2x,
                           kp_2, km_2, own2 & near_i)
        if NEAR_Y:
            gv_2y = PHY * v2
            if ZM_X:
                gv_2y = tl.where(at_x, 0.0, gv_2y)
            if ZM_Y:
                gv_2y = tl.where(at_y, 0.0, gv_2y)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_y, gv_2y,
                           kp_2, km_2, own2 & near_j)
        if NEAR_X and NEAR_Y:
            gv_2xy = PHY * (PHX * v2)
            if ZM_X:
                gv_2xy = tl.where(at_x, 0.0, gv_2xy)
            if ZM_Y:
                gv_2xy = tl.where(at_y, 0.0, gv_2xy)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x + dn_y, gv_2xy,
                           kp_2, km_2, own2 & near_i & near_j)
        if FAR_Z:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z, -PHZ * v2,
                           kp_f2, km_f2, own2 & far_k)
        if FAR_Z and NEAR_X:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x, -PHZ * gv_2x,
                           kp_f2, km_f2, own2 & far_k & near_i)
        if FAR_Z and NEAR_Y:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_y, -PHZ * gv_2y,
                           kp_f2, km_f2, own2 & far_k & near_j)
        if FAR_Z and (NEAR_X and NEAR_Y):
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x + dn_y, -PHZ * gv_2xy,
                           kp_f2, km_f2, own2 & far_k & near_i & near_j)

else:  # pragma: no cover - laptop path
    folded_beta_fused_curl_constitutive_D = None  # type: ignore[assignment]


def folded_beta_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_beta_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded beta fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return folded_beta_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def mirror_phases(grid: Any) -> Tuple[int, int, int]:
    """``PHX``/``PHY``/``PHZ``: the declared plane phase per axis, 0 where unfolded.

    The same derivation as :func:`.folded_fused_pair.mirror_phases`, restated here
    because this module's builder bakes the result into its own constexprs and may
    not read its inputs off another module's guard.
    """
    phases: List[int] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            phases.append(0)
            continue
        value = _call(grid, "mirror_phase", axis, default=None)
        phases.append(int(value) if value in (1, -1) else 0)
    return (phases[0], phases[1], phases[2])


def folded_beta_fused_electric_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None) -> Coverage:
    """May ONE launch span beta ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. The two
    halves are the arms ``launch.plan_step`` really selects on a folded beta row
    today (``folded real beta PML`` on ``step_D``, ``folded beta run`` on
    ``update_E``), so this predicate is narrower than the pair the planner already
    puts on the device, never wider. The seam clauses are
    :func:`.folded_fused_pair.folded_fused_pair_coverage`'s, restated for the same
    kernel geometry.
    """
    reasons: List[str] = []

    curl = folded_beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded beta curl half: {reason}"
                       for reason in curl.reasons)
    electric = folded_beta_run_constitutive_coverage(fields, pml,
                                                     CONSTITUTIVE_SIDE)
    if not electric.covered:
        reasons.extend(f"folded beta constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything.
    # The one corpus row declares TWO electric sources, injected BETWEEN the two
    # halves (driver.py:3294-3299); CARRIES_DEPOSIT_REPAIR is what lets the repair
    # carry them, and `repairable`'s fold clauses are live here — the repair must
    # be able to read the same fill map this kernel's forward carry implements.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR CARRY'S OWN CONDITIONS, transcribed from
        # folded_fused_pair_coverage — the same kernel geometry, the same ownership
        # move, the same row checks. See that predicate for why each is a refusal
        # rather than a soft error.
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        reader = _stored_past_owned_reader()
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
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
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A susceptibility makes the constitutive source (D - sum P) rather than D.
    # `folded_beta_run_constitutive_coverage` already refuses it on the E side;
    # restated because this kernel bakes the plain product and a reader should not
    # have to chase the other predicate to learn that.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: this kernel bakes the plain "
            "constitutive product, whose source is D and not (D - sum P)")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, at the owned cell and at
    # every imaged ghost, so `Fields` must be able to hand one over per component.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedBetaFusedElectricPairPlan:
    """One allocation-free launch for FIVE of the driver's electric passes.

    :class:`.folded_fused_pair.FoldedFusedPairPlan`'s bindings plus the two
    host-rounded beta scalars and the ``HAS_BETA`` constexpr, which is exactly the
    delta :class:`.special_kz.BetaPmlCurlPlan` carries over
    :class:`.launch.PmlCurlPlan`.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "beta_plus", "beta_minus", "has_beta",
        "backward", "bc", "zero_metal", "phases", "reflect", "near", "far",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_grid",
        "_kernel",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
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
                "inside the D seam of a beta run, and an unfolded beta grid belongs "
                "to beta_fused_electric_pair")
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
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self._inverse_epsilon = tuple(CupyPointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_beta_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
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
        return (f"FoldedBetaFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"phases={self.phases}, zero_metal={self.zero_metal}, "
                f"near={self.near}, far={self.far}, reflect={self.reflect}, "
                f"beta_plus={self.beta_plus!r}, has_beta={self.has_beta}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_folded_beta_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None) -> Optional[FoldedBetaFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not folded_beta_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    # THE SAME FUNCTION THE CURL PLAN CALLS, with `magnetic` from the sub-step —
    # identical arithmetic to the array path's per-call-site computation. On REAL
    # storage the two flag values are MEASURED IDENTICAL (the ±i lives only on the
    # complex path); it is written this way because that is what the array path
    # computes per call site.
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(CURL_SUB_STEP == "step_B"),
                                         complex_storage=False)
    return FoldedBetaFusedElectricPairPlan(
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
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both
        # and never asks which is which; the gate carries a mutation for the swap.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_beta_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, phases,
        dtdx: float, beta_plus: float, beta_minus: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1,
        reflect: Any = (None, None, None)) -> FoldedBetaFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    additionally carries ``inv_eps_Ex``...; ``kernel=`` carries the mutation
    override and ``has_beta=0`` the reduction leg's certified-kernel arm, which
    must reproduce ``folded_fused_pair.folded_fused_curl_constitutive_D`` bit for
    bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedBetaFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, phases, beta_plus, beta_minus,
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
        reflect=reflect,
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )
