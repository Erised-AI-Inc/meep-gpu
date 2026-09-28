"""The Dcyl m = 0 magnetic seam in one launch: ``step_B`` welded into ``update_H``.

The REAL cylindrical arm's B->H product, and the smallest cell on the fusion matrix
with **no attrition at all**: every row that drives it is reachable, and the shipped
predicate admits every one. Its sibling :mod:`.cylindrical_fused_magnetic_pair`
takes the |m| >= 1 complex arm; this file takes the m = 0 real one, and the two row
sets are disjoint (measured: 3 and 16, intersection empty).

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY, with a comment naming the source it
came from. The curl half is :func:`.cylindrical_triton.cyl_pml_curl_step`'s body
character for character, ``BACKWARD`` arms and all; the wall clear and the
constitutive half are :func:`.kernels.fused_curl_constitutive_B`'s, which is itself
:func:`.kernels.constitutive_step`'s ``SCALE = 0`` arm. A reader can diff this body
against those two and see that nothing moved, and the gate's transcription leg is
what turns that from a claim into a measurement. If you find yourself deriving here,
you have taken a wrong turn.

===========================================================================
WHAT IT IS WORTH — measured before it was built, on BOTH backends' censuses
===========================================================================

``parity/meep_gpu/results/fusion_matrix_{triton,metal}_2026-08-20`` rank the cells
the 186-row corpus drives that no fused product covers, by the CEILING at that cell
— instances with no in-seam source — not by rows reaching it::

    reach 3  (of 3 rows)  B_to_H  (cylindrical m=0, cylindrical m=0)   FITS; NOT BUILT

Re-measured here against the census this file prices against
(``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired_convention``, read on
``covered_modulo_backend`` because that lift ran on NumPy)::

    admit the cylindrical m = 0 curl at step_B                 3 rows
    admit the cylindrical constitutive at update_H             3 rows
    admit BOTH halves                                          3 rows
    ... and declare NO MAGNETIC source                         3 rows

3 -> 3 -> 3 -> 3. NO CLAUSE IN THE LADDER CUTS ANYTHING, which is rare enough on
this board to be worth stating: the three rows
(``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0`` and ``_1_0``,
``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``) all declare an ELECTRIC source
only, so the magnetic seam is empty on every one. The Metal census
(``results/metal_coverage_tranche6_2026-08-19``) funnels to the SAME three rows
through its own independently written predicates.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between ``step_B`` and ``update_H``
(driver.py:3281-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME.** ``driver.step`` injects them between
  the halves (driver.py:3283-3284), so a fused pair would consume a pre-injection
  ``B``. Ignorance is never an empty set: ``Fields`` does not hold the source list,
  so an undeclared ``sources`` is a REFUSAL and not an assumed ``()``. On this cell
  the clause costs nothing — all three corpus rows are electric-only — and it is
  still checked, because "the corpus happens not to" is not a kernel property.
* **``fill_symmetry_bc_B`` — PROVABLY INERT, not carried.**
  ``stepping._fill_symmetry_ghost_cells`` (:1437-1439) returns before touching a
  single cell unless ``grid.has_symmetry()``, and
  :func:`.cylindrical_triton.cylindrical_curl_coverage` refuses a mirror plane on
  every axis, twice — once through ``has_symmetry`` and once per axis through
  ``is_mirrored``. A pass that cannot execute is not a pass this kernel has to
  absorb, and the difference from "refused" is worth the two words: the folded
  pairs CARRY this fill because there it does real work.
* **``zero_metal_B`` — CARRIED INLINE** (driver.py:3286;
  ``stepping._zero_metal`` :2208-2247), through :func:`.coverage.zero_metal_axes`,
  which is IMPORTED rather than re-spelled. It is NOT vacuous here: all three
  corpus rows declare a metallic z wall (``metallic = [False, False, True]``), so
  ``ZM_Z`` fires on ``Bz`` at stored cell 0 of z on every one of them. A kernel
  that skipped it would leave a plane of un-cleared ``Bz`` for ``update_H`` to
  consume.
* **``fill_folded_far_ghosts_B`` — PROVABLY INERT, not carried**, for the same
  reason and at the same line: ``_fill_folded_far_ghosts`` (:1516-1517) returns
  unless ``grid.has_symmetry()``, and it additionally needs an axis whose stored
  extent exceeds its owned one, which only a folded periodic axis has.

So :data:`REPLACES` names THREE passes and not five. The two it omits are omitted
because they cannot run, and the predicate is what guarantees that.

===========================================================================
WHAT IS AND IS NOT CYLINDRICAL ABOUT THIS PRODUCT
===========================================================================

**The curl half is deeply cylindrical.** ``Bz``'s whole curl is replaced by the
forward difference of the EXTENDED radial prefix (``stepping.step_B`` :343-346), the
r = 0 ownership mask drops ``Bx``'s curl row 0, and the m = 0 axis rule writes
``Br[0] = 0`` (``_cylindrical_axis_zero_B`` :648). All three are
``cyl_pml_curl_step``'s, transcribed here unchanged.

**The constitutive half is not cylindrical at all**, and that is a FINDING this
product inherits rather than an assumption it makes:
:func:`.cylindrical_triton.cylindrical_constitutive_coverage` records that
``stepping.update_H`` carries NO cylindrical branch — it is element-wise with
per-axis coefficient tables indexed on the component's own axis, and Dcyl changes
nothing there except that axis 1 has n = 1 — and its gate MEASURED the shipped
``ConstitutivePlan`` bytewise identical to ``update_H``/``update_E`` on a real Dcyl
grid, 4/4 (``parity/meep_gpu/results/triton_cylindrical_2026-08-10/gate.json``). So
the constitutive half welded in below is the ORDINARY one, byte for byte.

**THE PREFIX STAYS ON THE ARRAY PATH, and it is UPSTREAM of the seam, not inside
it.** ``cyl_pml_curl_step`` consumes a radial prefix scan the host computes
(:func:`.cylindrical_triton.cylindrical_prefix`, which calls the shipped
``stepping.cylindrical_rderiv_prefix``); that decision, and the measurements behind
it, belong to :mod:`.cylindrical_triton` and are unchanged here. What matters for
FUSION is only that the prefix runs BEFORE the curl, so no launch of this kernel
straddles it: this plan computes the prefix and then makes ONE launch, exactly as
:class:`.cylindrical_triton.CylindricalCurlPlan` does, and that launch is what
replaces the separate constitutive launch. The pair therefore goes from TWO device
launches to ONE, with the prefix's own launches unchanged on both sides of the
comparison.

``BACKWARD`` IS CARRIED AND MUST BE 0. It keeps the curl half a verbatim copy of
``cyl_pml_curl_step`` rather than a hand-specialised one. It cannot be 1: the wall
clear below is the B family's rows (component ``m`` on axis ``m``), the constitutive
half is the H side, and — decisively — an electric source deposits into ``D``
between ``step_D`` and ``update_E`` on 157 of the corpus's 186 rows, which no launch
can straddle.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans three driver passes; there is no slot it
can claim without a composition rule nothing has measured. ``launch.py`` already
says so for this cell in its own words — "``fuse=True`` leaves all four plans
separate with named cylindrical refusals because the CuPy prefix and the axis rules
must be carried explicitly by a new pair kernel" (cylindrical_triton.py, the
CENTRAL-COMPOSER INTEGRATION note). This is that kernel; the composition rule is
still not written. Nothing in ``launch.py`` names this module,
``fastpath.plan_fast_path`` is unchanged, and dispatch stays disabled.

A CUPY VERSION IS PART OF THIS PRODUCT'S CORRECTNESS CONTRACT, inherited whole from
:mod:`.cylindrical_triton`: everything downstream of the radial scan is bit-identical
only for a given ``cupy.cumsum`` summation order. A CuPy bump is a correctness event
for this plan exactly as a Triton bump is, and a SILENT one. The gate records the
CuPy version; re-run it on any bump.

DEVICE STATUS: **RELEASED 2026-08-20**, the GPU host RTX A6000 GPU 6 (verified empty by
    UUID before and after), Triton 3.1.0 / CuPy 13.5.1, under the ``keep`` float32
    subnormal policy (``ieee_keep_ftz_stripped``).
    ``parity/meep_gpu/results/triton_cylindrical_real_fused_magnetic_pair_2026-08-20/``:

    * **4/4 device cases, 10 complete driver steps each, bit-identical** on the
      uint32 view of every allocated volume, to BOTH the CuPy array path and the two
      separately certified Triton products this launch replaces — one fused launch
      per step, 24 of 26 volumes moved (the two that did not are the read-only
      material inputs);
    * both ARMED HARNESS mutations refused as they must: the un-substituted route
      agrees bytewise and is caught by the launch counter alone, and the frozen seam
      agrees trivially and is caught by the moved-state census alone;
    * **9/10 kernel mutations caught**, including the wall clear, the seam's ORDER,
      the m = 0 axis rule, Bz's prefix substitution and the constitutive coefficient
      axis; the tenth (a commuted multiply) is a declared null, confirmed as one;
    * 3/3 refusals on real CuPy drivers.

    A FIRST RUN FOUND A DEFECT, and it was in the HARNESS. ``m5_ownership_mask_dropped``
    came back UNCAUGHT: with the ELECTRIC constitutive history left at zero in the
    fixture, ``Ep`` is exactly ``+0.0`` on both planes the curl's ownership mask
    touches — the m = 0 axis rule sets ``Dp[0] = 0`` and ``zero_metal_D`` clears
    ``Dy`` at z = 0 — so the mask was writing a value already present and the
    mutation could not fail. Seeding ``f_w_E*`` makes it observable, and the gate's
    ``mask_observability`` leg now measures BOTH arms so a future harness cannot
    quietly un-seed it.

    STILL NOT WIRED. A weld licenses the claim, not a dispatch: nothing in a default
    run reaches this plan.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    zero_metal_axes,
)
from .cylindrical_triton import (
    COVERED_M,
    CYLINDRICAL_AXIS_FLAGS,
    CYLINDRICAL_BOUNDARY_KINDS,
    DEFAULT_BLOCK,
    SUB_STEPS,
    cylindrical_constitutive_coverage,
    cylindrical_curl_coverage,
    cylindrical_prefix,
    _float32,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

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

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding is
#: visible without importing the curl module's table; the test file pins them equal.
BACKWARD = 0

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289). THREE, not five: ``fill_symmetry_bc_B`` and
#: ``fill_folded_far_ghosts_B`` both return before touching a cell unless
#: ``grid.has_symmetry()`` (stepping.py:1481-1483, :1565-1566), and the predicate
#: refuses a mirror plane on every axis. Declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The two driver passes this product does NOT carry, with the reason each cannot
#: execute on a configuration the predicate admits. Named as data so the gate can
#: assert them inert rather than assume it.
INERT_PASSES: Dict[str, str] = {
    "fill_symmetry_bc_B": ("stepping._fill_symmetry_ghost_cells:1437-1439 returns "
                           "unless grid.has_symmetry(); this predicate refuses a "
                           "mirror plane on every axis"),
    "fill_folded_far_ghosts_B": ("stepping._fill_folded_far_ghosts:1516-1517 returns "
                                 "unless grid.has_symmetry(), and additionally needs "
                                 "an axis whose stored extent exceeds its owned one"),
}

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "INERT_PASSES", "REPLACES",
    "CylindricalRealFusedMagneticPairPlan", "cyl_real_fused_curl_constitutive_B",
    "cyl_real_fused_curl_constitutive_B_kernel",
    "cylindrical_real_fused_magnetic_pair_coverage",
    "plan_cylindrical_real_fused_magnetic_pair",
    "plan_cylindrical_real_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    @triton.jit
    def cyl_real_fused_curl_constitutive_B(
        f0, f1, f2,                   # curl targets:    Bx, By, Bz            (in/out)
        u0, u1, u2,                   # curl auxiliaries: fu_Bx, fu_By, fu_Bz  (in/out)
        g0, g1, g2,                   # curl sources:    Ex, Ey, Ez            (in)
        pfx,                          # the ARRAY-PATH radial prefix
        hp,                           # stored Hp for the m=0 D-side axis add; unread here
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Hx, Hy, Hz
        w0, w1, w2,                   # constitutive aux: f_w_Hx, f_w_Hy, f_w_Hz
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, INTEGER lattice
        nx, ny, nz, n_elem, dtdx, four_dtdx,
        BACKWARD: tl.constexpr,       # bound to 0 by every builder; see below
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_B`` + ``zero_metal_B`` + ``update_H`` on a Dcyl grid at m = 0.

        NO BOUNDARY CONSTEXPRS, for the reason :func:`.cylindrical_triton.cyl_pml_curl_step`
        gives: this body serves exactly one declaration — the Dcyl triple in
        ``CYLINDRICAL_BOUNDARY_KINDS`` (``axis``, ``periodic``, ``metallic``) — so the
        ghost rules are compiled in directly and the predicate refuses anything else
        rather than a constexpr selecting it.

        ``BACKWARD`` IS CARRIED AND MUST BE 0. It keeps the curl half a verbatim copy
        rather than a hand-specialised one. It cannot be 1: the wall-clear block below
        is the B family's (component ``m`` on axis ``m``, ``IYEE_SHIFTS``) and the
        constitutive half is the H side.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` carry ``stepping.zero_metal_B`` inline. They are
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

        # --- the seam: stepping.zero_metal_B (driver.py:3286) ------------------
        # Byte-copied from kernels.fused_curl_constitutive_B's ZM block. Applied to
        # the REGISTER, AFTER the m = 0 axis rules and before both the store and the
        # constitutive read — the driver's own order (step_B, which ends with the
        # axis rules, then zero_metal_B, then update_H) — so the two consumers see
        # the one value the array path leaves in B.
        #
        # THE CARTESIAN BLOCK'S OWN NAMES, so it is a byte-copy: on a Dcyl grid
        # axis 0 IS r and axis 2 IS z (CYLINDRICAL_BOUNDARY_KINDS), so `at_x` is the
        # curl half's `at_r` and `at_z` is its own. `at_y` is spelled out because
        # the curl half never needs it: phi masks nothing.
        at_x, at_y = at_r, j == 0
        if ZM_X:
            v0 = tl.where(at_x, 0.0, v0)
        if ZM_Y:
            v1 = tl.where(at_y, 0.0, v1)
        if ZM_Z:
            v2 = tl.where(at_z, 0.0, v2)

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.fused_curl_constitutive_B, which is itself
        # constitutive_step's SCALE=0 arm with `src0 = tl.load(g0 + idx)` replaced by
        # the register the curl half just computed. Component 0 takes its
        # coefficient from axis x, 1 from y, 2 from z (stepping.H_CONSTITUTIVE_TERMS
        # :226) — the component's OWN axis. Nothing here is cylindrical:
        # stepping.update_H carries no Dcyl branch, which is the finding
        # cylindrical_constitutive_coverage records and its gate measured 4/4.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        src0 = v0
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(h0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(h0 + idx, a0, mask=live)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        src1 = v1
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(h1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(h1 + idx, a1, mask=live)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        src2 = v2
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(h2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(h2 + idx, a2, mask=live)

else:  # pragma: no cover - laptop path
    cyl_real_fused_curl_constitutive_B = None  # type: ignore[assignment]


def cyl_real_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if cyl_real_fused_curl_constitutive_B is None:
        raise ImportError(
            "the cylindrical m=0 fused magnetic B/H kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return cyl_real_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_real_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                                  sources: Any = None) -> Coverage:
    """May ONE launch span ``step_B`` -> wall -> ``update_H`` on a Dcyl m = 0 grid?

    A conjunction of the two halves' own predicates plus the seam clauses. Nothing
    is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. Same
    construction as :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage`.
    """
    reasons: List[str] = []

    curl = cylindrical_curl_coverage(fields, pml)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    magnetic = cylindrical_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not magnetic.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in magnetic.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E half and does not disqualify this pair.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no magnetic source" from not being told would be the
    # over-covering refusal this clause exists to prevent.
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
            "a mirror plane is active: fill_symmetry_bc_B and "
            "fill_folded_far_ghosts_B then do real work inside this seam "
            "(driver.py:3285, :3287) and this kernel carries neither")
    for axis in range(3):
        reader = getattr(grid, "is_mirrored", None)
        if callable(reader) and bool(reader(axis)):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two fill passes in "
                f"this seam stop being inert")

    # The wall clear is carried inline, so the grid must be able to answer which
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips a
    # plane the array path clears. NOT VACUOUS on this cell: every corpus row that
    # reaches it declares a metallic z wall.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalRealFusedMagneticPairPlan:
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
        "_curl_coefficients", "_h_targets", "_h_aux", "_h_coefficients", "_grid",
        "_kernel", "_pointer",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients, xp: Any, scratch: Any = None,
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
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
        # (stepping.py:619). It is unread at BACKWARD = 0 and still bound, because a
        # null makes the launcher's failure a TypeError far from its cause.
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
        self._h_targets = tuple(CupyPointer(array) for array in h_targets)
        self._h_aux = tuple(CupyPointer(array) for array in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(array)) for array in h_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg, which
        # compiles a deliberately broken copy of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def prefix(self) -> Any:
        """This launch's radial prefix, from the shipped array-path scan."""
        return cylindrical_prefix(self.xp, CURL_SUB_STEP, self._source_map,
                                  scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Prefix, then ONE launch. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else cyl_real_fused_curl_constitutive_B_kernel())
        # Hp for the m = 0 D-side axis add. At BACKWARD = 0 the kernel never reads
        # it, but the argument still has to type, so it is bound to a real float32
        # volume of the right shape rather than to a null.
        hp = self._pointer(self._source_map[SUB_STEPS[CURL_SUB_STEP]["prefix_component"]])
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.prefix()), hp,
            *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx, self.four_dtdx,
            BACKWARD=self.backward,
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"CylindricalRealFusedMagneticPairPlan(shape={self.shape}, m=0, "
                f"zero_metal={self.zero_metal}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_cylindrical_real_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        ) -> Optional[CylindricalRealFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not cylindrical_real_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return CylindricalRealFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive the
        # INTEGER one on H (stepping.py:948 vs the B curl's half_integer=True). The
        # kernel takes both and never asks which is which, so a swap here is a silent
        # half-cell error in the absorber profile; the gate carries a mutation for it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        grid.xp, scratch=getattr(fields, "scratch", None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_cylindrical_real_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], zero_metal, dtdx: float, xp: Any,
        scratch: Any = None, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalRealFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat``
    and ``constitutive_flat`` are keyed on the sub-lattice the caller ALREADY
    selected, so the gate can hand over a swapped pair and measure that the swap is
    caught.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalRealFusedMagneticPairPlan(
        shape, dtdx, zero_metal, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, scratch=scratch, kernel=kernel, num_warps=num_warps,
    )
