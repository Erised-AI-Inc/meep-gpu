"""The FOLDED special_kz ELECTRIC seam in one launch: folded beta ``step_D``
welded into ``update_E``, mirror fills carried inline.

The ``folded real beta PML`` -> ``folded beta run`` cell's D->E product. Both
halves are already certified and neither is re-derived here:

* the curl half is :func:`.folded_complex.folded_beta_pml_curl_step` (K3a) —
  ``symmetry.pml_curl_step_folded``'s body plus THE SAME beta pass
  ``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``,
  taken through its ``BACKWARD == 1`` arm;
* the fills, the wall clear and the constitutive half are
  :func:`.folded_fused_pair.folded_fused_curl_constitutive_D`'s, byte-copied —
  the ownership restructure, the chained near/far composition, the
  ``_carry_ghost_E`` device function (IMPORTED, not copied, so the two modules
  cannot drift) and the D-family six-row ``zero_metal_D`` block included;
* and the WELD is the one substitution ``folded_fused_curl_constitutive_D``
  already makes between the folded curl and ``update_E`` on a beta = 0 run.

SO THIS KERNEL IS ``folded_fused_curl_constitutive_D``, CHARACTER FOR CHARACTER,
PLUS THE PASS ``folded_beta_pml_curl_step`` ADDS TO ``pml_curl_step_folded``,
after the three curl forms and before the stores — MEEP's order (the pass runs after
every component's curl, step_db.cpp:161-175), the array path's
(``stepping._apply_folded_increment``) and K3a's own placement::

    if HAS_BETA:
        q0 = (beta_plus * b) * -1.0
        q1 = (beta_minus * a) * -1.0
        ...  (q0, q1 masked by the curl's own ownership clauses, both mask arms)
        v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
        v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)

A reader must be able to diff this body against those two and see nothing else
moved; the gate's transcription leg asserts exactly that, statement by
statement, against both shipped sources.

WHY THE BETA SLOT IS LOAD-BEARING UNDER A FOLD — ``folded_complex.py``'s module
docstring carries the measurement: the fold WIDENS the ownership mask from
METALLIC-only at cell 0 to non-PERIODIC at cell 0 PLUS ``MIRROR_PERIODIC`` at
the top plane, and beta touches targets 0 and 1 on both sub-steps, so one folded
axis makes BOTH mask arms fire on BOTH beta targets. An increment the pass left
unmasked would be live on the mirror plane and on the far ghost plane, invisible in
the interior. THE FILLS THEN IMAGE THE STEPPED VALUE, beta
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

    # The forms, the fold ladder and the chunk record's word offsets, under the names
    # the folded pair binds them; its ghost carries are its own device functions.
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402
    from .folded_fused_pair import _carry_ghost_E  # noqa: E402 - shared


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
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """Folded beta ``step_D`` + the fills + ``zero_metal_D`` + ``update_E``, one launch.

        ``folded_fused_pair.folded_fused_curl_constitutive_D``'s body, character for character, plus the
        beta pass of ``special_kz.beta_pml_curl_step``
        after the curl forms and before the wall clear and the ghost carries -- the array
        path's order (the curl, its beta pass, then the seam passes, then update_E). The
        pass's increment carries the folded curl's two ownership blocks, so a cell the
        curl leaves to a fill keeps its word through the pass, and every carried ghost
        images the registers the pass leaves. The carries are the folded pair's own
        device functions. ``HAS_BETA = 0`` builds ``folded_fused_curl_constitutive_D`` exactly.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from symmetry.pml_curl_step_folded; the only edits below its
        # stores are the ownership masks the near-fill carry needs.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # PERIODIC wraps; every other rule serves an exact 0.0 past the face. On a
        # folded axis that zero stands in for a value nothing reads: its only
        # consumer is the plane the cell-0 mask below zeroes.
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
        # kernels.pml_curl_step's three calls: MEEP's left-to-right association with
        # both partners, the one-partner form where the record says the other
        # partner's derivative axis is invariant, an exact +0 with neither.
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
        curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
        curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells :1865) -----
        # Every target whose Yee shift is 0 on a non-periodic axis. `!= PERIODIC`
        # rather than `== METALLIC`: _mask_non_owned_cells asks `is_mirrored or
        # is_metallic or is_axis`, and _boundary_kinds resolves exactly those to a
        # non-periodic kind.
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
        # symmetry.pml_curl_step_folded's second added block, BYTE-COPIED from its
        # BACKWARD == 1 arm (symmetry.py:294-301). It was absent while this family
        # refused MIRROR_PERIODIC; carrying fill_folded_far_ghosts_D brings the axis
        # in, so the block comes with it. Dx:(1,0,0) Dy:(0,1,0) Dz:(0,0,1) — shift 1
        # on its OWN axis only, which is exactly the set the far carry writes, and
        # three lines against the B arm's six.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BCX == MIRROR_PERIODIC:
            curl0 = tl.where(last_x, 0.0, curl0)
        if BCY == MIRROR_PERIODIC:
            curl1 = tl.where(last_y, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl2 = tl.where(last_z, 0.0, curl2)

        # --- ownership of the fill destinations --------------------------------
        # Component m is a NEAR destination at stored cell 0 of every folded axis
        # that is NOT its own and a FAR destination at the top plane of its own axis
        # when that axis is folded PERIODIC (IYEE_SHIFTS: Dx (1,0,0), Dy (0,1,0),
        # Dz (0,0,1) — shift 0 on the two others, 1 on its own). Those lanes do NOT
        # load or store D, E or f_w_E: the source lane writes all three for them.
        # The masks are what make the carry race-free — a lane that merely discarded
        # the value would still have READ a word another lane writes.
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

        # --- the PML forms (stepping._apply_curl_regions) ---------------------
        # The PML coefficient vectors are built at the STORED extent on a folded
        # axis, so the n_a indexing needs no fold-aware change.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # Which cells take the recurrence: the component's dsigu layer on its own
        # sub-lattice, verbatim from the folded curl (integer here, BACKWARD = 1).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=own0, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)

        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=own1, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)

        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=own2, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the special-kz beta pass (stepping._special_kz_beta_term, then
        # stepping._apply_folded_increment) ----------------------------------------
        # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
        # applied to the registers the forms returned. The partners are the CENTER
        # loads: target 0 takes the second source at sign +1 (Bx <- Ey, Dx <- Hy),
        # target 1 the first at sign -1 (By <- Ex, Dy <- Hx); target 2 gets nothing.
        # The increment is the NEGATED product (no dtdx: an analytic derivative),
        # masked to +0 on the cells the curl's own mask drops, and the beta ladder
        # subtracts it (forms.fold_ladder).
        if HAS_BETA:
            q0 = (beta_plus * b) * -1.0
            q1 = (beta_minus * a) * -1.0
            if BACKWARD:
                if BCY != PERIODIC:
                    q0 = tl.where(at_y, 0.0, q0)
                if BCZ != PERIODIC:
                    q0 = tl.where(at_z, 0.0, q0)
                if BCX != PERIODIC:
                    q1 = tl.where(at_x, 0.0, q1)
                if BCZ != PERIODIC:
                    q1 = tl.where(at_z, 0.0, q1)
                if BCX == MIRROR_PERIODIC:
                    q0 = tl.where(last_x, 0.0, q0)
                if BCY == MIRROR_PERIODIC:
                    q1 = tl.where(last_y, 0.0, q1)
            else:
                if BCX != PERIODIC:
                    q0 = tl.where(at_x, 0.0, q0)
                if BCY != PERIODIC:
                    q1 = tl.where(at_y, 0.0, q1)
                if BCY == MIRROR_PERIODIC:
                    q0 = tl.where(last_y, 0.0, q0)
                if BCZ == MIRROR_PERIODIC:
                    q0 = tl.where(last_z, 0.0, q0)
                if BCX == MIRROR_PERIODIC:
                    q1 = tl.where(last_x, 0.0, q1)
                if BCZ == MIRROR_PERIODIC:
                    q1 = tl.where(last_z, 0.0, q1)
            v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
            v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Byte-copied from dispersive_fused_pair's ZM block — the same six rows
        # _zero_metal writes for the D family (Dy/Dz on an x wall, Dx/Dz on y,
        # Dx/Dy on z). Applied to the REGISTER, before both the store and the
        # constitutive read, so the two consumers see the one value the array path
        # leaves in D. A walled axis is never a folded axis (_zero_metal
        # :2237-2239), so a wall's cleared plane (cell 0 of a walled axis) is never
        # a fold's destination plane; the predicate CHECKS that rather than leaning
        # on it. It does NOT follow that the clear and the near fill are unrelated
        # on this family — a component folded on y is still cleared on a z wall, and
        # that is why each near-ghost block below re-applies these lines AFTER its
        # parity, in the driver's order.
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        # fu is written on the recurrence's cells ONLY (the auxiliary contract),
        # destinations included: the array path's step_D writes it on every such
        # cell, a ghost row reading its image row's membership, and neither fill
        # touches it (the fills write `field`, never `fu_field`).
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=own0)
        tl.store(f1 + idx, v1, mask=own1)
        tl.store(f2 + idx, v2, mask=own2)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.constitutive_step's SCALE=1 arm, with
        # `tl.load(g + idx)` replaced by the register the curl half just produced
        # and `live` by the component's ownership mask. The form is keyed on the
        # dsigw line of the component's own axis on the half-integer lattice.
        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z
        # (stepping.E_CONSTITUTIVE_TERMS :227) — the component's OWN axis, which
        # is exactly the axis the NEAR fill never images along and exactly the one
        # the FAR fill does. That is what lets a near-only destination reuse these
        # six registers unchanged and what makes the far half load its own pair.
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
        prev0 = tl.load(w0 + idx, mask=own0 & dw0, other=0.0)   # BEFORE the store.
        src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)
        tl.store(w0 + idx, src0, mask=own0 & dw0)
        a0v = tl.load(e0 + idx, mask=own0 & dw0, other=0.0)
        a0v = constitutive_form(a0v, prev0, src0, kp_0, km_0, dw0)
        tl.store(e0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1 & dw1, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=own1, other=0.0)
        tl.store(w1 + idx, src1, mask=own1 & dw1)
        a1v = tl.load(e1 + idx, mask=own1 & dw1, other=0.0)
        a1v = constitutive_form(a1v, prev1, src1, kp_1, km_1, dw1)
        tl.store(e1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2 & dw2, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=own2, other=0.0)
        tl.store(w2 + idx, src2, mask=own2 & dw2)
        a2v = tl.load(e2 + idx, mask=own2 & dw2, other=0.0)
        a2v = constitutive_form(a2v, prev2, src2, kp_2, km_2, dw2)
        tl.store(e2 + idx, a2v, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # stepping._write_mirror_ghost (:1451):  D[..0..]  = +phase * D[..2..]
        # stepping._fill_folded_far_ghosts (:1529-1532): D[..-1..] = -phase * D[..r..]
        # and every COMPOSITION of the two, which is what a cell that is a near
        # destination on one axis and a far one on another carries.
        #
        # THE COMPOSITION IS CHAINED, NOT A PRODUCT OF PARITIES, and on this family
        # that is a byte-level distinction rather than a stylistic one. The driver
        # runs fill_symmetry_bc_D (:3300), then zero_metal_D (:3301), then
        # fill_folded_far_ghosts_D (:3302), so
        #
        #     a NEAR ghost is  clear(parity * v)  — the clear applied AFTER the
        #                                           parity, which is why each block
        #                                           below re-applies the ZM lines;
        #     a FAR  ghost is  parity * (the value already standing at its source),
        #                                           with NO clear after it, because
        #                                           :3302 is the last pass before
        #                                           update_E;
        #     a FAR-over-NEAR cell is the far parity times THE NEAR GHOST THIS LANE
        #                                           JUST COMPOSED — not the raw
        #                                           product -PH_m * PH_a * v.
        #
        # MEASURED, NOT ARGUED. parity/meep_gpu/measure_d_side_composition_law.py
        # scores this rule and four transcription errors against the array path's
        # own three passes over 156 grid declarations: this rule 156/156, and the
        # four alternatives caught on 84, 36, 36 and 70 cases respectively, with no
        # vacuous case. Reversing either order, or flattening the composite, is one
        # of those four.
        #
        # THE PARITY IS +PHASE ON A SHIFT-0 COMPONENT AND -PHASE ON A SHIFT-1 ONE
        # (fields.mirror_parity :117 is `phase * (1 - 2 * iyee)`), so the near carry
        # takes `PH?` and the far carry `-PH?`, and +-1.0 is exact in float32.
        #
        # THE COEFFICIENT INDEX MOVES ONLY FOR THE FAR HALF. update_E indexes
        # component m on axis m (stepping.E_CONSTITUTIVE_TERMS :227); the near fill
        # images along an axis that is NOT m, so its destination sits at the SAME
        # coefficient index as the lane that owns it and reuses that lane's pair —
        # a reload there would reload the identical word. The far fill images along
        # exactly axis m, so its destination reads kps/kms at the TOP row rather
        # than reusing the source lane's at the reflect row.
        #
        # THE NEAR SOURCE INDEX IS THE LITERAL 2, as it is in
        # symmetry.mirror_ghost_fill (`base + 2 * stride`), rather than the named
        # MIRROR_SOURCE_INDEX: a jit body that closes over a module-level Python int
        # is a Triton-version question this file has no way to measure without a
        # device. The literal is pinned to the name by test, so the two cannot drift.
        #
        # The destination coefficient pair for the FAR half, per component, at the
        # top row of that component's own axis. A scalar `nx - 1` would not
        # broadcast against a masked vector load, so it is built as a per-lane
        # tensor the same way the magnetic twin builds its `origin`.
        top_x = idx * 0 + (nx - 1)
        top_y = idx * 0 + (ny - 1)
        top_z = idx * 0 + (nz - 1)
        kp_f0 = tl.load(kp0 + top_x, mask=live, other=0.0)
        km_f0 = tl.load(km0 + top_x, mask=live, other=0.0)
        kp_f1 = tl.load(kp1 + top_y, mask=live, other=0.0)
        km_f1 = tl.load(km1 + top_y, mask=live, other=0.0)
        kp_f2 = tl.load(kp2 + top_z, mask=live, other=0.0)
        km_f2 = tl.load(km2 + top_z, mask=live, other=0.0)
        # ... and its membership there, for the same reason: the FAR destination's
        # own-axis row is the top row, so its select is keyed on the top row's word.
        # A NEAR-only destination keeps the source lane's coordinate on that axis,
        # and so its flag (dw0, dw1, dw2 above).
        dw_f0 = record_in(cb, RECORD_HALF_INTEGER + 0, top_x)
        dw_f1 = record_in(cb, RECORD_HALF_INTEGER + 4, top_y)
        dw_f2 = record_in(cb, RECORD_HALF_INTEGER + 8, top_z)

        # The per-axis source lane and destination offset, once. Each is read only
        # under a mask its own axis's constexpr gates, so an axis that carries no
        # fill contributes no lane and no load.
        #
        # EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWN OWNERSHIP MASK, and
        # that is a DEFECT THE MAGNETIC TWIN'S DEVICE GATE FOUND rather than a
        # precaution (its results/run_farcarry2/, 2026-08-20). With two fills a lane
        # can be the SOURCE of one and the DESTINATION of the other: on a grid
        # folding x PERIODIC and y, the lane at (i = rx, j = 0) is the far source
        # for Dx on x and is itself the near fill's destination on y. It would then
        # write the composite cell with a `v` built from an `f0` load its own
        # ownership mask zeroed, racing the lane that legitimately owns it.
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

        # THE THREE-TERM GUARDS ARE PARENTHESISED, AND THAT IS A MEASURED PLATFORM
        # FACT rather than a style choice: Triton 3.1.0's frontend refuses a chained
        # boolean outright — "chained boolean operators (A or B or C) are not
        # supported; use parentheses to split the chain" — and it refuses at COMPILE
        # time on the device, so `if NEAR_Y and NEAR_Z and FAR_X:` is a kernel that
        # does not build rather than one that builds wrong.

        # --- component 0 (Dx): near on y and z, far on x -----------------------
        # The near ghosts are composed FIRST and kept in registers, because the
        # far-over-near cells below are the far parity applied to exactly these
        # values. The ZM lines re-applied to each are zero_metal_D's six rows for
        # this component (Dx clears on a y wall and on a z wall), in the driver's
        # parity-then-clear order.
        if NEAR_Y:
            gv_0y = PHY * v0
            if ZM_Y:
                gv_0y = tl.where(at_y, 0.0, gv_0y)
            if ZM_Z:
                gv_0y = tl.where(at_z, 0.0, gv_0y)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,
                           kp_0, km_0, dw0, own0 & near_j)
        if NEAR_Z:
            gv_0z = PHZ * v0
            if ZM_Y:
                gv_0z = tl.where(at_y, 0.0, gv_0z)
            if ZM_Z:
                gv_0z = tl.where(at_z, 0.0, gv_0z)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_z, gv_0z,
                           kp_0, km_0, dw0, own0 & near_k)
        if NEAR_Y and NEAR_Z:
            # The doubly-unowned corner. Y is applied before Z
            # (_fill_symmetry_ghost_cells :1441-1447), so the value is
            # phase_z * (phase_y * v) — two exact +-1.0 multiplies, in the array
            # path's own order, then the clear.
            gv_0yz = PHZ * (PHY * v0)
            if ZM_Y:
                gv_0yz = tl.where(at_y, 0.0, gv_0yz)
            if ZM_Z:
                gv_0yz = tl.where(at_z, 0.0, gv_0yz)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y + dn_z, gv_0yz,
                           kp_0, km_0, dw0, own0 & near_j & near_k)
        if FAR_X:
            # NO ZM AFTER THE PARITY: :3302 is the last pass before update_E, so
            # the far ghost is -phase times the value standing at its source, which
            # is `v0` read after the clear lines above.
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x, -PHX * v0,
                           kp_f0, km_f0, dw_f0, own0 & far_i)
        if FAR_X and NEAR_Y:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y, -PHX * gv_0y,
                           kp_f0, km_f0, dw_f0, own0 & far_i & near_j)
        if FAR_X and NEAR_Z:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_z, -PHX * gv_0z,
                           kp_f0, km_f0, dw_f0, own0 & far_i & near_k)
        if FAR_X and (NEAR_Y and NEAR_Z):
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y + dn_z, -PHX * gv_0yz,
                           kp_f0, km_f0, dw_f0, own0 & far_i & near_j & near_k)

        # --- component 1 (Dy): near on x and z, far on y -----------------------
        if NEAR_X:
            gv_1x = PHX * v1
            if ZM_X:
                gv_1x = tl.where(at_x, 0.0, gv_1x)
            if ZM_Z:
                gv_1x = tl.where(at_z, 0.0, gv_1x)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x, gv_1x,
                           kp_1, km_1, dw1, own1 & near_i)
        if NEAR_Z:
            gv_1z = PHZ * v1
            if ZM_X:
                gv_1z = tl.where(at_x, 0.0, gv_1z)
            if ZM_Z:
                gv_1z = tl.where(at_z, 0.0, gv_1z)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_z, gv_1z,
                           kp_1, km_1, dw1, own1 & near_k)
        if NEAR_X and NEAR_Z:
            # X is applied before Z: phase_z * (phase_x * v).
            gv_1xz = PHZ * (PHX * v1)
            if ZM_X:
                gv_1xz = tl.where(at_x, 0.0, gv_1xz)
            if ZM_Z:
                gv_1xz = tl.where(at_z, 0.0, gv_1xz)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x + dn_z, gv_1xz,
                           kp_1, km_1, dw1, own1 & near_i & near_k)
        if FAR_Y:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y, -PHY * v1,
                           kp_f1, km_f1, dw_f1, own1 & far_j)
        if FAR_Y and NEAR_X:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x, -PHY * gv_1x,
                           kp_f1, km_f1, dw_f1, own1 & far_j & near_i)
        if FAR_Y and NEAR_Z:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_z, -PHY * gv_1z,
                           kp_f1, km_f1, dw_f1, own1 & far_j & near_k)
        if FAR_Y and (NEAR_X and NEAR_Z):
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x + dn_z, -PHY * gv_1xz,
                           kp_f1, km_f1, dw_f1, own1 & far_j & near_i & near_k)

        # --- component 2 (Dz): near on x and y, far on z -----------------------
        if NEAR_X:
            gv_2x = PHX * v2
            if ZM_X:
                gv_2x = tl.where(at_x, 0.0, gv_2x)
            if ZM_Y:
                gv_2x = tl.where(at_y, 0.0, gv_2x)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x, gv_2x,
                           kp_2, km_2, dw2, own2 & near_i)
        if NEAR_Y:
            gv_2y = PHY * v2
            if ZM_X:
                gv_2y = tl.where(at_x, 0.0, gv_2y)
            if ZM_Y:
                gv_2y = tl.where(at_y, 0.0, gv_2y)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_y, gv_2y,
                           kp_2, km_2, dw2, own2 & near_j)
        if NEAR_X and NEAR_Y:
            # X is applied before Y: phase_y * (phase_x * v).
            gv_2xy = PHY * (PHX * v2)
            if ZM_X:
                gv_2xy = tl.where(at_x, 0.0, gv_2xy)
            if ZM_Y:
                gv_2xy = tl.where(at_y, 0.0, gv_2xy)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x + dn_y, gv_2xy,
                           kp_2, km_2, dw2, own2 & near_i & near_j)
        if FAR_Z:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z, -PHZ * v2,
                           kp_f2, km_f2, dw_f2, own2 & far_k)
        if FAR_Z and NEAR_X:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x, -PHZ * gv_2x,
                           kp_f2, km_f2, dw_f2, own2 & far_k & near_i)
        if FAR_Z and NEAR_Y:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_y, -PHZ * gv_2y,
                           kp_f2, km_f2, dw_f2, own2 & far_k & near_j)
        if FAR_Z and (NEAR_X and NEAR_Y):
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x + dn_y, -PHZ * gv_2xy,
                           kp_f2, km_f2, dw_f2, own2 & far_k & near_i & near_j)

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
    """One allocation-free launch for FIVE of the driver's electric passes."""

    __slots__ = (
        "beta_words", "has_beta",
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "phases",
        "reflect", "near", "far", "chunk_bounds",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_cb", "_grid",
        "_kernel",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 reflect: Any = (None, None, None),
                 beta_words: Any = None, has_beta: int = 1,
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phases = tuple(int(value) for value in phases)
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT. The kernel branches its
        # curl half on the four codes and its two carries on these booleans; if
        # they could be passed independently they could disagree, and a kernel that
        # masked the top plane on one axis set and imaged the far ghost on another
        # is a plane of wrong values rather than a crash.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        if not any(self.near):
            raise ValueError(
                "no axis is folded: this product exists to carry the mirror fills "
                "inside the D seam, and an unfolded grid belongs to the certified "
                "curl and constitutive pair")
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
        # (plus, minus): the two host-rounded beta words (special_kz.beta_curl_coefficients);
        # float() keeps the bits.
        words = tuple(beta_words) if beta_words is not None else ()
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real beta words of "
                             f"special_kz.beta_curl_coefficients, got {beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        # The chunk record, required and checked against this extent; ONE record
        # serves both halves, as in launch.FusedPairPlan.
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedBetaFusedElectricPairPlan")
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        self._inverse_epsilon = tuple(CupyPointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of this kernel. Dropping it is
        # not a slowdown, it is a DISARMING — every mutation leg would then launch
        # the shipped kernel and report the defect as uncaught.
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
            *self._e_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.reflect[0], self.reflect[1], self.reflect[2],
            *self.beta_words,
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
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_beta_fused_electric_pair(fields: Any, pml: Any, sources: Any = None,
                           block: Optional[int] = None,
                           num_warps: Optional[int] = 1,
                           kernel: Any = None,
                           chunk_layout: Any = None) -> Optional[FoldedBetaFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``); its record is what both halves' forms key on.
    """
    if not folded_beta_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    words = beta_curl_coefficients(fields.grid.beta, fields.grid.dt,
                                   magnetic=False, complex_storage=False)
    return FoldedBetaFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        mirror_phases(grid),
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
        # and never asks which is which, so a swap here is a silent half-cell
        # error in the absorber profile; the gate carries a mutation for exactly
        # it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count
        # and `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        beta_words=words,
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_beta_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, phases,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None), beta_words: Any = None, has_beta: int = 1, *,
        chunk_bounds: Any) -> FoldedBetaFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    ``beta_words`` is ``(plus, minus)``, normally
    :func:`.special_kz.beta_curl_coefficients`'s output or a deliberately wrong pair
    on a mutation leg; ``has_beta=0`` is the identity leg's folded-pair arm.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

    ``chunk_bounds`` is REQUIRED and has no default, as in
    ``launch.plan_from_arrays``: a plan without the record of the division the
    caller built is refused by name.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedBetaFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, phases,
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
        beta_words=beta_words, has_beta=has_beta,
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
