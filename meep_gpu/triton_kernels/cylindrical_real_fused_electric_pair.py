"""The Dcyl m = 0 ELECTRIC seam in one launch: ``step_D`` welded into ``update_E``.

THE ELECTRIC TWIN of :mod:`.cylindrical_real_fused_magnetic_pair`, over the same two
shipped bodies on the other seam.

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY of a shipped body:

* the curl half is :func:`.cylindrical_triton.cyl_pml_curl_step`'s body taken through
  its ``BACKWARD == 1`` arms up to its stores -- Dz's ``dtdx`` times the backward radial
  difference, the D-family ownership mask, the chunk-keyed PML forms and the m = 0 axis
  rules (``Dz[0] += realnum(double(4*Courant) * double(Hp[0]))`` then ``Dp[0] = 0``);
* the wall clear is :func:`.kernels.fused_curl_constitutive_D`'s ZM block -- each axis
  clears the two TANGENTIAL D components;
* the constitutive half is :func:`.kernels.fused_curl_constitutive_D`'s, the chunk-keyed
  ``dsigw`` forms of :func:`.kernels.constitutive_step`'s ``SCALE = 1`` arm with the
  source load replaced by the register the curl half computed, D on the left of the
  inverse-epsilon multiply.

THE RADIAL SCAN IS UPSTREAM OF THE SEAM: the plan launches
:func:`.cylindrical_triton.cyl_radial_difference`, then this kernel.

THE FIVE PASSES IN THE SEAM. ``step_D -> electric sources -> fill_symmetry_bc_D ->
zero_metal_D -> fill_folded_far_ghosts_D -> update_E``:

* **the electric sources -- CARRIED through the shipped deposit repair.** The injection
  lands between the halves, so the fused launch consumes a pre-injection ``D``;
  :class:`.deposit_repair.LeadingRepairPlan` saves the stateful words at the deposit
  points before the launch and :class:`.deposit_repair.TrailingRepairPlan` recomputes
  them after, in the driver's own order. An undeclared source list is a refusal, and a
  source that cannot publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` -- PROVABLY INERT**, for
  the magnetic twin's reason;
* **``zero_metal_D`` -- CARRIED INLINE** through :func:`.coverage.zero_metal_axes`,
  after the m = 0 axis rules and before both the store and the constitutive read.

``BACKWARD`` IS CARRIED AND BOUND TO 1.

DEVICE STATUS. The weld released on 2026-08-31 measured the previous forms and does not
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
    zero_metal_axes,
)
from .cylindrical_triton import (
    DEFAULT_BLOCK,
    SUB_STEPS,
    RadialScan,
    courant_constants,
    cylindrical_constitutive_coverage,
    cylindrical_curl_coverage,
    radial_scan_starts,
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

    # The forms and the record layout, read by the kernel. Imported only when Triton
    # is: both modules import it at module scope.
    from .forms import (axis_present, constitutive_form, curl_form, curl_stencil,
                        record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE
    from .cylindrical_triton import COURANT_FOUR

    @triton.jit
    def cyl_real_fused_curl_constitutive_D(
        f0, f1, f2,                   # curl targets:    Dx, Dy, Dz            (in/out)
        u0, u1, u2,                   # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz  (in/out)
        g0, g1, g2,                   # curl sources:    Hx, Hy, Hz            (in)
        rad,                          # the radial difference (cyl_radial_difference)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                   # constitutive targets: Ex, Ey, Ez
        w0, w1, w2,                   # constitutive aux: f_w_Ex, f_w_Ey, f_w_Ez
        ie0, ie1, ie2,                # diagonal inverse epsilon volumes
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, HALF-INTEGER lattice
        cb,                           # chunk record: both halves' membership and presence
        courant,                      # float64 [4*Courant, Courant, -Courant]
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,       # bound to 1 by every builder; see below
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``zero_metal_D`` + ``update_E`` on a Dcyl grid at m = 0.

        The curl half is :func:`.cylindrical_triton.cyl_pml_curl_step`'s body up to its
        stores, through its ``BACKWARD`` arms; the wall clear and the constitutive half
        are :func:`.kernels.fused_curl_constitutive_D`'s. ``ZM_*`` carry
        ``stepping.zero_metal_D`` inline (:func:`.coverage.zero_metal_axes`).
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

        # --- the seam: zero_metal_D (the root pair's ZM block) ---------------
        # The root pair's names: on a Dcyl grid axis 0 IS r and axis 2 IS z, so
        # `at_x` is the curl half's `at_r`; `at_y` is spelled out because phi masks
        # nothing in the curl half.
        at_x, at_y = at_r, j == 0
        # zero_metal_D: each axis clears the two tangential D components.
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        # fu on the recurrence's cells ONLY (the auxiliary contract); the field everywhere.
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ================================
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # Which cells take the accumulation: the component's own (dsigw) layer on the
        # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
        # (the constitutive term tables), so each target's record line is again a
        # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)   # E: the half-integer sub-lattice
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)
        src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live & dw0)
        a0 = constitutive_form(tl.load(e0 + idx, mask=live & dw0, other=0.0),
                                 prev0, src0, kp_0, km_0, dw0)
        tl.store(e0 + idx, a0, mask=live)

        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live & dw1)
        a1 = constitutive_form(tl.load(e1 + idx, mask=live & dw1, other=0.0),
                                 prev1, src1, kp_1, km_1, dw1)
        tl.store(e1 + idx, a1, mask=live)

        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live & dw2)
        a2 = constitutive_form(tl.load(e2 + idx, mask=live & dw2, other=0.0),
                                 prev2, src2, kp_2, km_2, dw2)
        tl.store(e2 + idx, a2, mask=live)

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
    """The radial scan, then ONE launch for three of the driver's passes.

    Allocation-free per step: the scan's output and bindings, the chunk record and the
    Courant words are bound at plan time. ``chunk_bounds`` is REQUIRED and ONE record
    serves both halves: the curl half reads its entries on the curl's sub-lattice and the
    constitutive half on its own, the rule that picks each half's coefficient table.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "courant", "backward", "zero_metal", "block",
        "num_warps", "chunk_bounds", "scan", "_targets", "_aux", "_sources",
        "_curl_coefficients", "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_cb", "_courant",
        "_grid", "_kernel", "_pointer",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, courant: float, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, xp: Any, *, chunk_bounds: Any,
                 scan_starts: Any, kernel: Any = None, scan_kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not "
                f"(the exp(i*m*phi) dependence is analytic)")
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "CylindricalRealFusedElectricPairPlan")
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
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        self._inverse_epsilon = tuple(CupyPointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
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
                  else cyl_real_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.scan.out),
            *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            self._cb, self._courant,
            nx, ny, nz, self.n_elem, self.dtdx,
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
        num_warps: Optional[int] = 1, kernel: Any = None, chunk_layout: Any = None,
        ) -> Optional[CylindricalRealFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``chunk_layout`` is the run's division (``None`` for the one-rank default); the chunk
    record and the scan's starts line both follow it.
    """
    if not cylindrical_real_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from .launch import chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return CylindricalRealFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx, float(grid.courant), zero_metal_axes(grid),
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
        # HALF-INTEGER one on E. The kernel takes both and never asks which is which,
        # so a swap here is a silent half-cell error in the absorber profile; the gate
        # carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        grid.xp, kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        scan_starts=radial_scan_starts(fields, pml, CURL_SUB_STEP, chunk_layout),
    )


def plan_cylindrical_real_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], zero_metal, dtdx: float, courant: float,
        xp: Any, *, chunk_bounds: Any, scan_starts: Any, block: Optional[int] = None,
        kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalRealFusedElectricPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat``
    and ``constitutive_flat`` are keyed on the sub-lattice the caller ALREADY selected.
    ``chunk_bounds`` and ``scan_starts`` are REQUIRED.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalRealFusedElectricPairPlan(
        shape, dtdx, courant, zero_metal, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        scan_kernel=scan_kernel, num_warps=num_warps,
    )
