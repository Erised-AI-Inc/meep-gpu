"""The Dcyl m = 0 ELECTRIC seam in one launch: ``step_D`` welded into ``update_E``.

THE ELECTRIC TWIN of :mod:`.cylindrical_real_fused_magnetic_pair`, over the SAME two
certified halves on the other seam. It claims the cell the Triton fusion matrix ranks
second among the unbuilt ones::

    ceiling  3  (rows 3)  D->E  cylindrical PML -> cylindrical   NO FUSED PRODUCT

— ``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_{0_0,1_0}`` and
``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``, the same three rows its magnetic
twin serves, measured in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-31_cells``.

**ALL THREE ROWS DECLARE AN ELECTRIC SOURCE** (``source_field_types == ['D']``), so
under the source clause as the MAGNETIC twin states it — with
:data:`CARRIES_DEPOSIT_REPAIR` at False — this arm would admit NOTHING AT ALL. The
ceiling of 3 is the count with the deposit repair carried, and this module declares
the flag in the same change as the wiring that brackets its launch. That is the
whole difference between the two twins' source clauses, and it is the reverse of
their situation on the magnetic seam, where the same three rows are electric-only and
the clause costs the twin nothing.

EVERYTHING HERE IS TRANSCRIBED, and every arithmetic line carries the function it
came from:

* the curl half is :func:`.cylindrical_triton.cyl_pml_curl_step`'s body, character
  for character, taken through its ``BACKWARD == 1`` arms — the prefix substitution
  on target 2, the D-family ownership mask, and the m = 0 axis rules
  (``Dz[0] += 4*Courant*Hp[0]`` then ``Dp[0] = 0``). The magnetic twin carries the
  SAME lines and never compiles them, because its ``BACKWARD`` is bound to 0;
* the wall clear is :func:`.kernels.fused_curl_constitutive_D`'s ZM block — each
  axis clears the two TANGENTIAL D components, which is a different six rows from
  the magnetic twin's three;
* the constitutive half is :func:`.kernels.fused_curl_constitutive_D`'s, which is
  :func:`.kernels.constitutive_step`'s ``SCALE = 1`` arm with ``tl.load(g + idx)``
  replaced by the register the curl half computed.

A reader can diff this body against those two and see that nothing moved. If you find
yourself deriving here, you have taken a wrong turn.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between ``step_D`` and ``update_E``
(driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.** The
  injection lands between the two halves, so the fused launch consumes a
  pre-injection ``D``; :class:`.deposit_repair.LeadingRepairPlan` saves the two
  stateful arrays at the deposit points before the launch and
  :class:`.deposit_repair.TrailingRepairPlan` recomputes them after, in the driver's
  own order. IGNORANCE IS STILL NEVER AN EMPTY SET: ``Fields`` does not hold the
  source list, so an undeclared ``sources`` is a REFUSAL, and a source that cannot
  publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` — PROVABLY INERT, not carried.**
  ``stepping._fill_symmetry_ghost_cells`` (:1437-1439) returns before touching a
  cell unless ``grid.has_symmetry()``, and
  :func:`.cylindrical_triton.cylindrical_curl_coverage` refuses a mirror plane on
  every axis, twice — once through ``has_symmetry`` and once per axis through
  ``is_mirrored``. A pass that cannot execute is not a pass this kernel has to
  absorb, and the difference from "refused" is worth the two words;
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301), through
  :func:`.coverage.zero_metal_axes`, which is IMPORTED rather than re-spelled. It is
  NOT vacuous: all three corpus rows declare a metallic z wall
  (``metallic == [False, False, True]``), so ``ZM_Z`` fires on ``Dr`` and ``Dp`` at
  stored cell 0 of z on every one of them;
* **``fill_folded_far_ghosts_D`` — PROVABLY INERT**, for the same reason and at the
  same line (:1516-1517), plus it needs an axis whose stored extent exceeds its
  owned one, which only a folded periodic axis has.

So :data:`REPLACES` names THREE passes and not five. The two it omits are omitted
because they cannot run, and the predicate is what guarantees that.

===========================================================================
WHAT IS AND IS NOT CYLINDRICAL ABOUT THIS PRODUCT
===========================================================================

**The curl half is deeply cylindrical, and MORE so than the magnetic twin's.** Three
things that are compile-time absent at ``BACKWARD = 0`` are live here:

1. ``Dz``'s curl takes the BACKWARD difference of the radial prefix
   (``stepping.step_D`` :425-426) — ``dtdx * ((p_down - p_here) + (a - a_p))``, the
   four-operand grouping, rather than the two-operand forward difference ``Bz``
   takes;
2. the ownership mask is the D family's: ``Dy`` at r = 0 and at z = 0, ``Dz`` at
   r = 0 (``IYEE_SHIFTS``: Dx(1,0,0), Dy(0,1,0), Dz(0,0,1)), against the B family's
   two rows;
3. the m = 0 axis rules are ``_cylindrical_axis_zero_D`` (stepping.py:589):
   ``Dz[0] += (4*Courant) * Hp[0]`` — a POST-add applied to the UPDATED field and
   deliberately NOT folded into the curl (measured: the fold breaks m = 0 under PML)
   — and then ``Dp[0] = 0``. The magnetic side has only ``Br[0] = 0``.

**The constitutive half is not cylindrical at all**, and that is a FINDING this
product inherits rather than an assumption it makes:
:func:`.cylindrical_triton.cylindrical_constitutive_coverage` records that
``stepping.update_E`` carries NO cylindrical branch — it is element-wise with
per-axis coefficient tables indexed on the component's own axis — and its gate
MEASURED the shipped ``ConstitutivePlan`` bytewise identical to
``update_H``/``update_E`` on a real Dcyl grid, 4/4
(``parity/meep_gpu/results/triton_cylindrical_2026-08-10/gate.json``).

**THE PREFIX STAYS ON THE ARRAY PATH, and it is UPSTREAM of the seam, not inside
it.** ``cyl_pml_curl_step`` consumes a radial prefix scan the host computes
(:func:`.cylindrical_triton.cylindrical_prefix`); this plan computes the prefix and
then makes ONE launch, exactly as :class:`.cylindrical_triton.CylindricalCurlPlan`
does, and that launch is what replaces the separate constitutive launch. The pair
goes from TWO device launches to ONE, with the prefix's own launches unchanged on
both sides of the comparison.

``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy of
``cyl_pml_curl_step`` rather than a hand-specialised one. It cannot be 0: the wall
clear below is the D family's six rows, the constitutive half is the E side with its
inverse-permittivity multiply, and the axis rules are ``_cylindrical_axis_zero_D``'s.

A CUPY VERSION IS PART OF THIS PRODUCT'S CORRECTNESS CONTRACT, inherited whole from
:mod:`.cylindrical_triton`: everything downstream of the radial scan is bit-identical
only for a given ``cupy.cumsum`` summation order. A CuPy bump is a correctness event
for this plan exactly as a Triton bump is, and a SILENT one. The gate records the
CuPy version; re-run it on any bump.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans three driver call sites; there is no slot
it can claim without a composition rule nothing has measured. Nothing in
``launch.py`` names this module, ``fastpath.plan_fast_path`` is unchanged, and
dispatch stays disabled — the same deferral its magnetic twin ships under.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    zero_metal_axes,
)
from .cylindrical_triton import (
    DEFAULT_BLOCK,
    SUB_STEPS,
    cylindrical_constitutive_coverage,
    cylindrical_curl_coverage,
    cylindrical_prefix,
    _float32,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between three seam-instances and zero: every one of
#: the three corpus rows declares an ELECTRIC source, which the driver injects between
#: the two halves (driver.py:3294-3299). The flag is a claim about the PLAN this
#: module builds -- that the leading slot saves and the trailing slot restores -- and
#: is only ever changed in the same edit as that wiring.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a FOLD whose fill map it cannot read
#: (including the cylindrical r = 0 axis under a mirror plane), and an absorber whose
#: split-field recurrence never ran. The rows this admits carry no fold at all, so
#: that clause is quiet here rather than absent.
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

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding is
#: visible without importing the curl module's table; the host suite pins them equal.
BACKWARD = 1

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). THREE, not five: ``fill_symmetry_bc_D`` and
#: ``fill_folded_far_ghosts_D`` both return before touching a cell unless
#: ``grid.has_symmetry()`` (stepping.py:1481-1483, :1565-1566), and the predicate
#: refuses a mirror plane on every axis. THE INJECTION IS NOT ONE OF THE THREE: it is
#: carried by the deposit repair bracket around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The two driver passes this product does NOT carry, with the reason each cannot
#: execute on a configuration the predicate admits. Named as data so the gate can
#: assert them inert rather than assume it.
INERT_PASSES: Dict[str, str] = {
    "fill_symmetry_bc_D": ("stepping._fill_symmetry_ghost_cells:1437-1439 returns "
                           "unless grid.has_symmetry(); this predicate refuses a "
                           "mirror plane on every axis"),
    "fill_folded_far_ghosts_D": ("stepping._fill_folded_far_ghosts:1516-1517 returns "
                                 "unless grid.has_symmetry(), and additionally needs "
                                 "an axis whose stored extent exceeds its owned one"),
}

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "INERT_PASSES", "REPLACES",
    "CylindricalRealFusedElectricPairPlan", "cyl_real_fused_curl_constitutive_D",
    "cyl_real_fused_curl_constitutive_D_kernel",
    "cylindrical_real_fused_electric_pair_coverage",
    "plan_cylindrical_real_fused_electric_pair",
    "plan_cylindrical_real_fused_electric_pair_from_arrays",
]


if triton is not None:

    @triton.jit
    def cyl_real_fused_curl_constitutive_D(
        f0, f1, f2,                   # curl targets:    Dx, Dy, Dz            (in/out)
        u0, u1, u2,                   # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz  (in/out)
        g0, g1, g2,                   # curl sources:    Hx, Hy, Hz            (in)
        pfx,                          # the ARRAY-PATH radial prefix
        hp,                           # stored Hp, for the m = 0 on-axis Dz add
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Ex, Ey, Ez
        w0, w1, w2,                   # constitutive aux: f_w_Ex, f_w_Ey, f_w_Ez
        ie0, ie1, ie2,                # inverse epsilon, per target
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx, four_dtdx,
        BACKWARD: tl.constexpr,       # bound to 1 by every builder; see below
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``zero_metal_D`` + ``update_E`` on a Dcyl grid at m = 0.

        NO BOUNDARY CONSTEXPRS, for the reason
        :func:`.cylindrical_triton.cyl_pml_curl_step` gives: this body serves exactly
        one declaration — the Dcyl triple in ``CYLINDRICAL_BOUNDARY_KINDS`` (``axis``,
        ``periodic``, ``metallic``) — so the ghost rules are compiled in directly and
        the predicate refuses anything else rather than a constexpr selecting it.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy
        rather than a hand-specialised one. It cannot be 0: the wall-clear block below
        is the D family's six rows, the constitutive half is the E side, and the axis
        rules are ``_cylindrical_axis_zero_D``'s.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` carry ``stepping.zero_metal_D`` inline. They are
        NOT the boundary kinds: ``_zero_metal`` asks ``grid.is_metallic(axis) and not
        grid.is_mirrored(axis)`` and returns early unless ``grid.has_metallic``, which
        is what :func:`.coverage.zero_metal_axes` transcribes. On a Dcyl grid ``ZM_Y``
        is structurally false — phi is the one-cell invariant axis and ``Grid``
        refuses a PEC there by name (grid.py:842-877) — and the row is carried anyway,
        because a block that is absent cannot be measured and a block whose guard is
        false costs nothing.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from cylindrical_triton.cyl_pml_curl_step; the only edit below
        # its stores is the wall clear the seam needs, inserted between the axis
        # rules and the stores exactly where the driver runs it.

        # --- the ghost rule, per axis -----------------------------------------
        # r (axis 0) is CYL_AXIS. Forward: ``_shift_up``'s CYL_AXIS branch shares
        # the METALLIC one — a hard zero at the FAR r face, the PEC wall
        # (stepping.py:1829). Backward: the near ghost is r_to_minus_r and is
        # DELIBERATELY NOT IMPLEMENTED — it is unobservable, see
        # cylindrical_triton's module docstring. An exact 0.0 is served instead,
        # which is what the masked load delivers without dereferencing anything.
        # phi (axis 1) is PERIODIC on a length-1 axis: the wrap returns the SAME
        # element, which is what makes the difference an exact +0.0. Computed, not
        # elided — see the signed-zero trap in that docstring.
        # z (axis 2) is METALLIC: zero past the wall on both faces.
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vr = live & (si >= 0) & (si < nx)
        vz = live & (sk >= 0) & (sk < nz)
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        vp = live

        o_r = si * nyz + j * nz + k
        o_p = i * nyz + sj * nz + k
        o_z = i * nyz + j * nz + sk

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_p = tl.load(g0 + o_p, mask=vp, other=0.0)
        a_z = tl.load(g0 + o_z, mask=vz, other=0.0)
        b_r = tl.load(g1 + o_r, mask=vr, other=0.0)
        b_z = tl.load(g1 + o_z, mask=vz, other=0.0)
        c_r = tl.load(g2 + o_r, mask=vr, other=0.0)
        c_p = tl.load(g2 + o_p, mask=vp, other=0.0)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens -
        curl0 = dtdx * ((c_p - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_r))
        curl2 = dtdx * ((b_r - b) + (a - a_p))

        # --- the cylindrical substitution on target 2 --------------------------
        if BACKWARD:
            # step_D :425-426 — the Dz term's `first` source is the prefix, and the
            # backward machinery is otherwise unchanged.
            p_here = tl.load(pfx + idx, mask=live, other=0.0)
            p_down = tl.load(pfx + o_r, mask=vr, other=0.0)
            curl2 = dtdx * ((p_down - p_here) + (a - a_p))
        else:
            # step_B :343-346 — Bz's WHOLE curl is the forward difference of the
            # EXTENDED prefix: one subtract, one multiply. The four-operand grouping
            # above is a different float32 number.
            curl2 = dtdx * (tl.load(pfx + idx + nyz, mask=live, other=0.0)
                            - tl.load(pfx + idx, mask=live, other=0.0))

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) --
        # Per axis, for every target whose Yee shift there is 0 (fields.IYEE_SHIFTS):
        #   B: Bx(0,1,1) -> r ;  By(1,0,1) -> phi only, which is periodic: nothing ;
        #      Bz(1,1,0) -> z
        #   D: Dx(1,0,0) -> phi (nothing) + z ;  Dy(0,1,0) -> r + z ;  Dz(0,0,1) -> r
        at_r, at_z = i == 0, k == 0
        if BACKWARD:
            curl0 = tl.where(at_z, 0.0, curl0)
            curl1 = tl.where(at_r, 0.0, curl1)
            curl1 = tl.where(at_z, 0.0, curl1)
            curl2 = tl.where(at_r, 0.0, curl2)
        else:
            curl0 = tl.where(at_r, 0.0, curl0)
            curl2 = tl.where(at_z, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        # dsig/dsigu follow vec.hpp's cycle_direction: target 0 takes (y, z),
        # target 1 (z, x), target 2 (x, y) — the same triple on both sides.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y

        # --- the m = 0 axis rules, folded into the STORED value ----------------
        # B side (_cylindrical_axis_zero_B, stepping.py:677): Br[0] = 0.
        # D side (_cylindrical_axis_zero_D, :589): Dz[0] += (4*Courant)*Hp[0] — a
        # POST-add, applied to the updated field and NOT folded into the curl
        # (measured: the fold breaks m = 0 under PML) — then Dp[0] = 0.
        if BACKWARD:
            v1 = tl.where(at_r, 0.0, v1)
            v2 = tl.where(at_r, v2 + four_dtdx * tl.load(hp + idx, mask=live, other=0.0),
                          v2)
        else:
            v0 = tl.where(at_r, 0.0, v0)

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Byte-copied from kernels.fused_curl_constitutive_D's ZM block: each axis
        # clears the two TANGENTIAL D components, which is SIX rows against the
        # magnetic twin's three. Applied to the REGISTER, AFTER the m = 0 axis rules
        # and before both the store and the constitutive read — the driver's own
        # order (step_D, which ends with the axis rules, then zero_metal_D, then
        # update_E) — so the two consumers see the one value the array path leaves
        # in D.
        #
        # THE CARTESIAN BLOCK'S OWN NAMES, so it is a byte-copy: on a Dcyl grid axis
        # 0 IS r and axis 2 IS z (CYLINDRICAL_BOUNDARY_KINDS), so `at_x` is the curl
        # half's `at_r` and `at_z` is its own. `at_y` is spelled out because the curl
        # half never needs it: phi masks nothing.
        at_x, at_y = at_r, j == 0
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.fused_curl_constitutive_D, which is itself
        # constitutive_step's SCALE=1 arm with `src0 = tl.load(g0 + idx)` replaced by
        # the register the curl half just computed. Component 0 takes its coefficient
        # from axis x, 1 from y, 2 from z (stepping.E_CONSTITUTIVE_TERMS :227) — the
        # component's OWN axis. Nothing here is cylindrical: stepping.update_E
        # carries no Dcyl branch, which is the finding
        # cylindrical_constitutive_coverage records and its gate measured 4/4.
        #
        # D IS ON THE LEFT of the inverse-epsilon multiply because the array path
        # writes `source * fields.inverse_epsilon_for(component)`
        # (stepping.py:1011-1013); float multiplication is bitwise commutative but a
        # transcription is not a place to rely on that.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(h0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(h0 + idx, a0, mask=live)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(h1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(h1 + idx, a1, mask=live)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(h2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(h2 + idx, a2, mask=live)

else:  # pragma: no cover - laptop path
    cyl_real_fused_curl_constitutive_D = None  # type: ignore[assignment]


def cyl_real_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if cyl_real_fused_curl_constitutive_D is None:
        raise ImportError(
            "the Dcyl m = 0 fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return cyl_real_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_real_fused_electric_pair_coverage(fields: Any, pml: Any,
                                                  sources: Any = None) -> Coverage:
    """May ONE launch span ``step_D`` -> wall -> ``update_E`` on a Dcyl m = 0 grid?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the same
    construction
    :func:`.cylindrical_real_fused_magnetic_pair.cylindrical_real_fused_magnetic_pair_coverage`
    uses on the other seam.
    """
    reasons: List[str] = []

    curl = cylindrical_curl_coverage(fields, pml)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    electric = cylindrical_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not electric.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # ELECTRIC source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. All three corpus rows of this cell
    # declare exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
    # NOTHING; it is True, and the plan below brackets its launch with the repair.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so an
    # undeclared `sources` is itself a REFUSAL. A MAGNETIC source is injected in the
    # B/H half and does not disqualify this pair.
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

    # THE TWO FILL PASSES MUST BE INERT, not merely absent from this kernel.
    # `_fill_symmetry_ghost_cells` (:1437-1439) and `_fill_folded_far_ghosts`
    # (:1516-1517) both return before touching a cell unless `grid.has_symmetry()`.
    # The cylindrical curl predicate already refuses a mirror plane; RESTATED here
    # because this product's REPLACES tuple names three passes and not five, and a
    # reader must not have to chase another predicate to learn why the other two are
    # missing. "Another module already guards it" is exactly the reasoning this
    # package refuses.
    if bool(getattr(grid, "has_symmetry", lambda: False)()
            if callable(getattr(grid, "has_symmetry", None))
            else getattr(grid, "has_symmetry", False)):
        reasons.append(
            "a mirror plane is active: fill_symmetry_bc_D and "
            "fill_folded_far_ghosts_D then do real work inside this seam "
            "(driver.py:3300, :3302) and this kernel carries neither")
    for axis in range(3):
        reader = getattr(grid, "is_mirrored", None)
        if callable(reader) and bool(reader(axis)):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two fill passes in "
                f"this seam stop being inert")

    # The wall clear is carried inline, so the grid must be able to answer which axes
    # are walled; an unanswerable one compiles to ZM=False and silently skips a plane
    # the array path clears. NOT VACUOUS on this cell: every corpus row that reaches
    # it declares a metallic z wall.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able to
    # hand one over per E component. An absent accessor would be a TypeError inside
    # the builder rather than a refusal, which is the wrong way for an uncovered
    # configuration to fail. The magnetic twin needs no such clause: its constitutive
    # half is the SCALE = 0 arm and multiplies by nothing.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalRealFusedElectricPairPlan:
    """The array-path radial prefix, then ONE launch for three of the driver's passes.

    IT IS NOT ALLOCATION-FREE, and cannot be, for the reason
    :class:`.cylindrical_triton.CylindricalCurlPlan` gives: the prefix is recomputed
    every launch from the current sources. With the engine's ``StepScratch`` threaded
    through it allocates nothing beyond what the array path already pools. That is
    inherited unchanged — the fusion this product performs is between the CURL and
    the CONSTITUTIVE, and the prefix sits upstream of both.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "four_dtdx", "backward", "zero_metal", "block",
        "num_warps", "xp", "scratch", "_targets", "_aux", "_sources", "_source_map",
        "_curl_coefficients", "_e_targets", "_e_aux", "_inverse_epsilon",
        "_e_coefficients", "_grid", "_kernel", "_pointer",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, xp: Any,
                 scratch: Any = None, kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not "
                f"(the exp(i*m*phi) dependence is analytic)")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply, and Triton types a
        # Python float argument as fp32 — so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        # The 4*Courant of the m = 0 on-axis Dz add, ROUNDED TO FLOAT32 HERE because
        # the array path forms it as a float64 product and NEP-50 casts it weakly
        # (stepping.py:619). UNLIKE the magnetic twin this argument is READ on every
        # launch: BACKWARD is 1 here.
        self.four_dtdx = float(_float32(4.0 * float(dtdx)))
        self.backward = BACKWARD
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.xp = xp
        self.scratch = scratch
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        # The raw arrays, by component name: the prefix is computed from them per
        # launch, so the plan holds the arrays and not only their addresses.
        self._source_map = {name: array for name, array
                            in zip(SUB_STEPS[CURL_SUB_STEP]["sources"], sources)}
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self._inverse_epsilon = tuple(CupyPointer(array)
                                      for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown,
        # it is a DISARMING — every mutation leg would then launch the shipped kernel
        # and report the defect as uncaught.
        self._kernel = kernel

    def prefix(self) -> Any:
        """This launch's radial prefix, from the shipped array-path scan."""
        return cylindrical_prefix(self.xp, CURL_SUB_STEP, self._source_map,
                                  scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Prefix, then ONE launch. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else cyl_real_fused_curl_constitutive_D_kernel())
        # Hp for the m = 0 on-axis Dz add. READ on every launch here — the argument
        # is the same one the magnetic twin binds and never dereferences.
        hp = self._pointer(
            self._source_map[SUB_STEPS[CURL_SUB_STEP]["prefix_component"]])
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.prefix()), hp,
            *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx, self.four_dtdx,
            BACKWARD=self.backward,
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"CylindricalRealFusedElectricPairPlan(shape={self.shape}, m=0, "
                f"zero_metal={self.zero_metal}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_cylindrical_real_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        ) -> Optional[CylindricalRealFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not cylindrical_real_fused_electric_pair_coverage(
            fields, pml, sources).covered:
        return None
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return CylindricalRealFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx, zero_metal_axes(grid),
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
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        grid.xp, scratch=getattr(fields, "scratch", None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_cylindrical_real_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], zero_metal, dtdx: float, xp: Any,
        scratch: Any = None, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalRealFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat``
    and ``constitutive_flat`` are keyed on the sub-lattice the caller ALREADY
    selected, so the gate can hand over a swapped pair and measure that the swap is
    caught; ``arrays`` additionally carries ``inv_eps_Ex``... .
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalRealFusedElectricPairPlan(
        shape, dtdx, zero_metal, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, scratch=scratch, kernel=kernel, num_warps=num_warps,
    )
