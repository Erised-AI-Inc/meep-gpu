"""The Dcyl m = 0 magnetic seam in one launch: ``step_B`` welded into ``update_H``.

The REAL cylindrical arm's B->H product. Its sibling :mod:`.cylindrical_fused_magnetic_pair`
takes the complex-storage arm; the two are split by storage, as their single sub-step
products are.

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY of a shipped body. The curl half is
:func:`.cylindrical_triton.cyl_pml_curl_step`'s, ``BACKWARD`` arms and all, up to its
stores: the presence-selected curls, Bz's ``dtdx`` times the radial difference, the
ownership mask, the chunk-keyed PML forms with the auxiliary contract on ``fu``, and the
m = 0 axis rules. The wall clear and the constitutive half are
:func:`.kernels.fused_curl_constitutive_B`'s (the chunk-keyed ``dsigw`` forms of
:func:`.kernels.constitutive_step`'s ``SCALE = 0`` arm with MEEP's start of H), with the
source load replaced by the register the curl half computed. The gate's transcription
leg (``parity/meep_gpu/probe_triton_cylindrical_real_fused_magnetic_pair.py``) is what
turns that from a claim into a measurement.

THE RADIAL SCAN IS UPSTREAM OF THE SEAM. The plan launches
:func:`.cylindrical_triton.cyl_radial_difference` (one
:class:`.cylindrical_triton.RadialScan`, bound once), then this kernel; no launch
straddles the scan, and the fusion removes the separate constitutive launch.

THE FIVE PASSES IN THE SEAM. The driver runs ``step_B -> magnetic sources ->
fill_symmetry_bc_B -> zero_metal_B -> fill_folded_far_ghosts_B -> update_H``:

* **the magnetic sources -- REFUSED BY NAME** (an undeclared source list is a refusal,
  never an assumed empty set): ``driver.step`` injects them between the halves, so a
  fused launch would consume a pre-injection ``B``;
* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` -- PROVABLY INERT, not
  carried**: both return before touching a cell unless ``grid.has_symmetry()``, and the
  predicate refuses a mirror plane on every axis;
* **``zero_metal_B`` -- CARRIED INLINE** through :func:`.coverage.zero_metal_axes`,
  applied to the registers after the m = 0 axis rule and before both the store and the
  constitutive read, the driver's own order.

So :data:`REPLACES` names THREE passes and :data:`INERT_PASSES` the other two.

``BACKWARD`` IS CARRIED AND BOUND TO 0, so the curl half stays a verbatim copy; the
wall clear is the B family's rows and the constitutive half is the H side.

DEVICE STATUS. The weld released on 2026-08-20 measured the previous forms and does not
license this text; certification is owed at the combined device round.
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
    RadialScan,
    courant_constants,
    cylindrical_constitutive_coverage,
    cylindrical_curl_coverage,
    radial_scan_starts,
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

    # The forms and the record layout, read by the kernel. Imported only when Triton
    # is: both modules import it at module scope.
    from .forms import (axis_present, constitutive_form, curl_form, curl_stencil,
                        record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE
    from .cylindrical_triton import COURANT_FOUR

    @triton.jit(do_not_specialize=["start_h"])
    def cyl_real_fused_curl_constitutive_B(
        f0, f1, f2,                   # curl targets:    Bx, By, Bz            (in/out)
        u0, u1, u2,                   # curl auxiliaries: fu_Bx, fu_By, fu_Bz  (in/out)
        g0, g1, g2,                   # curl sources:    Ex, Ey, Ez            (in)
        rad,                          # the radial difference (cyl_radial_difference)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Hx, Hy, Hz
        w0, w1, w2,                   # constitutive aux: f_w_Hx, f_w_Hy, f_w_Hz
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, INTEGER lattice
        cb,                           # chunk record: both halves' membership and presence
        courant,                      # float64 [4*Courant, Courant, -Courant]
        nx, ny, nz, n_elem, dtdx, start_h,  # start_h: constitutive_step's start of H
        BACKWARD: tl.constexpr,       # bound to 0 by every builder; see below
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_B`` + ``zero_metal_B`` + ``update_H`` on a Dcyl grid at m = 0.

        The curl half is :func:`.cylindrical_triton.cyl_pml_curl_step`'s body up to its
        stores; the wall clear and the constitutive half are
        :func:`.kernels.fused_curl_constitutive_B`'s. ``ZM_X``/``ZM_Y``/``ZM_Z`` carry
        ``stepping.zero_metal_B`` inline (:func:`.coverage.zero_metal_axes`); ``ZM_Y`` is
        structurally false on a Dcyl grid and carried anyway. ``start_h`` is
        :func:`.kernels.constitutive_step`'s: the registers the curl computed ARE the B
        words MEEP's first ``update_eh`` copies from.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis ---------------------------------------------
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
        a_z = tl.load(g0 + o_z, mask=vz, other=0.0)
        b_z = tl.load(g1 + o_z, mask=vz, other=0.0)
        c_r = tl.load(g2 + o_r, mask=vr, other=0.0)
        c_p = tl.load(g2 + o_p, mask=vp, other=0.0)

        # --- the curls (stepping._curl_from_operands, by partner presence) ----------
        # Target 0 differences g2 = Ez/Hz along phi and g1 = Ey/Hy along z; target 1
        # g0 = Ex/Hx along z and g2 along r (stepping's term table). phi carries no
        # partner on a Dcyl grid, so the presence word selects the one-partner form for
        # target 0 and the full left-to-right bracket for target 1. Target 2's WHOLE
        # curl is dtdx times the radial difference (step_db.cpp:86-119).
        presence = tl.load(cb + RECORD_PRESENCE)
        pr = axis_present(presence, 0)
        pp = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_p, c, b, b_z, dtdx, pp, pz)
        curl1 = curl_stencil(a_z, a, c, c_r, dtdx, pz, pr)
        curl2 = dtdx * tl.load(rad + idx, mask=live, other=0.0)

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) ------
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

        # --- the PML forms (stepping._apply_curl_regions) ---------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction: target 0 takes (phi, z),
        # target 1 (z, r), target 2 (r, phi). The recurrence runs where the target's
        # chunk carries its dsigu layer -- the record line of that axis on the
        # target's own sub-lattice (half-integer for B, integer for D) -- and fu is
        # read and written there only.
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

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)
        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)
        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the m = 0 axis rules, on the UPDATED words ------------------------------
        # B (stepping._cylindrical_axis_zero_B): Br[0] = 0.
        # D (_cylindrical_axis_zero_D): Dz[0] += realnum(double(4*Courant) *
        # double(Hp[0])) -- a double product rounded once, then a float32 add into the
        # updated field; Hp is the g1 centre `b` (the stored Hy under the layer) --
        # then Dp[0] = 0. The two touch different components.
        if BACKWARD:
            v1 = tl.where(at_r, 0.0, v1)
            dfcnd = tl.cast(tl.cast(b, tl.float64) * tl.load(courant + COURANT_FOUR),
                            tl.float32)
            v2 = tl.where(at_r, v2 + dfcnd, v2)
        else:
            v0 = tl.where(at_r, 0.0, v0)

        # --- the seam: zero_metal_B (the root pair's ZM block) ---------------
        # The root pair's names: on a Dcyl grid axis 0 IS r and axis 2 IS z, so
        # `at_x` is the curl half's `at_r`; `at_y` is spelled out because phi masks
        # nothing in the curl half.
        at_x, at_y = at_r, j == 0
        # --- the seam: stepping.zero_metal_B, between the two sub-steps ------------
        # Applied to the REGISTER, before both the store and the constitutive read, so
        # the two consumers see the one value the array path leaves in B.
        if ZM_X:
            v0 = tl.where(at_x, 0.0, v0)
        if ZM_Y:
            v1 = tl.where(at_y, 0.0, v1)
        if ZM_Z:
            v2 = tl.where(at_z, 0.0, v2)

        # fu on the recurrence's cells ONLY (the auxiliary contract); the field everywhere.
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ================================
        # Verbatim from constitutive_step's SCALE=0 arm, with `src0 = tl.load(g0+idx)`
        # replaced by the register v0 (and v1/v2). The three source pointers and the
        # three inv_eps placeholders are gone with it.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        # Which cells take the accumulation: the component's own (dsigw) layer on the
        # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
        # (the constitutive term tables), so each target's record line is again a
        # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        dw0 = record_in(cb, 0, i)         # H: the integer sub-lattice
        dw1 = record_in(cb, 4, j)
        dw2 = record_in(cb, 8, k)
        # MEEP's start of H (see the docstring).
        start = start_h != 0
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        src0 = v0
        tl.store(w0 + idx, src0, mask=live & dw0)
        a0 = tl.load(h0 + idx, mask=live & dw0, other=0.0)
        prev0 = tl.where(start, src0, prev0)    # MEEP's start of H
        a0 = tl.where(start, src0, a0)
        a0 = constitutive_form(a0, prev0, src0, kp_0, km_0, dw0)
        tl.store(h0 + idx, a0, mask=live)

        # --- component 1 -----------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        src1 = v1
        tl.store(w1 + idx, src1, mask=live & dw1)
        a1 = tl.load(h1 + idx, mask=live & dw1, other=0.0)
        prev1 = tl.where(start, src1, prev1)    # MEEP's start of H
        a1 = tl.where(start, src1, a1)
        a1 = constitutive_form(a1, prev1, src1, kp_1, km_1, dw1)
        tl.store(h1 + idx, a1, mask=live)

        # --- component 2 -----------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        src2 = v2
        tl.store(w2 + idx, src2, mask=live & dw2)
        a2 = tl.load(h2 + idx, mask=live & dw2, other=0.0)
        prev2 = tl.where(start, src2, prev2)    # MEEP's start of H
        a2 = tl.where(start, src2, a2)
        a2 = constitutive_form(a2, prev2, src2, kp_2, km_2, dw2)
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
    """The radial scan, then ONE launch for three of the driver's passes.

    Allocation-free per step: the scan's output and bindings, the chunk record and the
    Courant words are bound at plan time. ``chunk_bounds`` is REQUIRED and ONE record
    serves both halves: the curl half reads its entries on the curl's sub-lattice and the
    constitutive half on its own, the rule that picks each half's coefficient table.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "courant", "backward", "zero_metal", "block",
        "num_warps", "chunk_bounds", "scan", "_targets", "_aux", "_sources",
        "_curl_coefficients", "_h_targets", "_h_aux", "_h_coefficients", "_cb", "_courant",
        "_h_state", "_grid", "_kernel", "_pointer",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, courant: float, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients, xp: Any, *, chunk_bounds: Any,
                 scan_starts: Any, kernel: Any = None, scan_kernel: Any = None,
                 num_warps: Optional[int] = 1, h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not "
                f"(the exp(i*m*phi) dependence is analytic)")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "CylindricalRealFusedMagneticPairPlan")
        # float(): the array path multiplies a float32 volume by a Python float, which
        # NumPy/CuPy cast to float32 before the multiply, and Triton types a Python
        # float argument as fp32 -- so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        # The double Courant of the m = 0 axis add travels as float64 words.
        self.courant = float(courant)
        self._courant = CupyPointer(xp.asarray(courant_constants(self.courant)))
        self.backward = BACKWARD
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        spec = SUB_STEPS[CURL_SUB_STEP]
        source_map = dict(zip(spec["sources"], sources))
        self.scan = RadialScan(CURL_SUB_STEP, source_map[spec["prefix_component"]],
                               scan_starts, xp, block=self.block, kernel=scan_kernel)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._h_targets = tuple(CupyPointer(array) for array in h_targets)
        self._h_aux = tuple(CupyPointer(array) for array in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(array)) for array in h_coefficients)
        self._h_state = h_state
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg, which
        # compiles a deliberately broken copy of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING -- every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """The scan, then ONE launch. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        self.scan.run(guard)
        kernel = (self._kernel if self._kernel is not None
                  else cyl_real_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        state = self._h_state
        start = 0 if state is None or state.h_separated else 1
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.scan.out),
            *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients,
            self._cb, self._courant,
            nx, ny, nz, self.n_elem, self.dtdx, start,
            BACKWARD=self.backward,
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def __repr__(self) -> str:
        return (f"CylindricalRealFusedMagneticPairPlan(shape={self.shape}, m=0, "
                f"zero_metal={self.zero_metal}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_cylindrical_real_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, chunk_layout: Any = None,
        ) -> Optional[CylindricalRealFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``chunk_layout`` is the run's division (``None`` for the one-rank default); the chunk
    record and the scan's starts line both follow it.
    """
    if not cylindrical_real_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from .launch import chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return CylindricalRealFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, float(grid.courant), zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive the
        # INTEGER one on H. The kernel takes both and never asks which is which, so a
        # swap here is a silent half-cell error in the absorber profile; the gate
        # carries a mutation for it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        grid.xp, kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        scan_starts=radial_scan_starts(fields, pml, CURL_SUB_STEP, chunk_layout),
        h_state=fields,
    )


def plan_cylindrical_real_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], zero_metal, dtdx: float, courant: float,
        xp: Any, *, chunk_bounds: Any, scan_starts: Any, block: Optional[int] = None,
        kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalRealFusedMagneticPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat``
    and ``constitutive_flat`` are keyed on the sub-lattice the caller ALREADY selected.
    ``chunk_bounds`` and ``scan_starts`` are REQUIRED.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalRealFusedMagneticPairPlan(
        shape, dtdx, courant, zero_metal, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        scan_kernel=scan_kernel, num_warps=num_warps,
    )
