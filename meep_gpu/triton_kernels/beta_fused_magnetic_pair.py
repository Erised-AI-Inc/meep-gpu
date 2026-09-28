"""The special_kz REAL-beta magnetic seam in one launch: beta ``step_B`` welded
into ``update_H``.

The ``real beta PML`` -> ``real beta run`` cell's B->H product. Both halves are
already certified and neither is re-derived here:

* the curl half is :func:`.special_kz.beta_pml_curl_step` — itself
  ``kernels.pml_curl_step``'s certified body plus ONE constexpr-gated insert;
* the constitutive half is ``kernels.constitutive_step``'s ``SCALE = 0`` arm,
  admitted on a beta run by the shipped
  :func:`.special_kz.beta_run_constitutive_coverage` ("the constitutive sub-steps
  read nothing beta-dependent");
* and the WELD is the one substitution :func:`.kernels.fused_curl_constitutive_B`
  already makes between exactly those two bodies on a beta = 0 run.

SO THIS KERNEL IS ``fused_curl_constitutive_B``, CHARACTER FOR CHARACTER, PLUS THE
SAME THREE LINES ``beta_pml_curl_step`` ADDS TO ``pml_curl_step``. A reader must be
able to diff this body against those two and see nothing else moved; the gate's
transcription leg asserts exactly that, statement by statement, against both
shipped sources.

===========================================================================
THE THREE LINES, AND WHERE THEY GO
===========================================================================

``stepping._special_kz_beta_term`` (S:733-735, :770, :784) adds an analytic
``d/dz -> i*2*pi*beta`` term to the curl. The array path inserts it AFTER the
``dtdx`` curl and BEFORE the ownership mask (S:356-363 after :342, before :369),
and that is where it sits here. Transcribed from
:func:`.special_kz.beta_pml_curl_step`, not re-derived:

* the beta partners are the CENTER loads the curl already made — target 0 takes
  the second source's center ``b`` at sign +1 (Bx <- Ey) and target 1 the first
  source's center ``a`` at sign -1 (By <- Ex); target 2 gets nothing, because
  ``step_db.cpp:148-176`` runs ``cc`` over ``d_c`` in {X, Y} only;
* NO ``dtdx`` multiplies the term (it is an analytic derivative, S:733-735);
* ``curl - (c * g)`` is the array path's ``curl + (-(c * g))`` by IEEE-754's
  definition of subtraction, which is how ``beta_pml_curl_step`` spells it and
  therefore how it is spelled here;
* real storage is MEEP's implicit-i trick, so BOTH sub-steps take the same sign
  (the ±i lives only in complex storage). This product is the B side only.

``beta_plus``/``beta_minus`` are host-rounded ONCE by
:func:`.special_kz.beta_curl_coefficients` and passed as float32 words, exactly as
:class:`.special_kz.BetaPmlCurlPlan` passes them.

``HAS_BETA`` IS CARRIED AND MUST BE 1 on any configuration this product's predicate
admits — clause 12 of the beta family is INVERTED (``grid.beta`` must be nonzero),
so a beta = 0 run is the ordinary :func:`.coverage.fused_pair_coverage`'s and is
refused here by name. It is carried rather than hard-coded for
``beta_pml_curl_step``'s own reason: it keeps the curl half a verbatim copy rather
than a hand-specialised one, and it gives the gate a certified-kernel arm
(``HAS_BETA = 0`` must reproduce ``fused_curl_constitutive_B`` bit for bit).

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3282-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME.** The driver injects them BETWEEN the
  halves (driver.py:3283-3284), so a fused pair would consume a pre-injection
  ``B``. Ignorance is never an empty set: ``Fields`` does not hold the source list,
  so an undeclared ``sources`` is a REFUSAL and not an assumed ``()``.
* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` — PROVABLY NO-OPS
  HERE**, not carried and not refused. Both return at their first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482, :1565-1566), and clause 5 of
  ``_beta_real_grid_reasons`` refuses every fold outright ("the folded real beta
  curl is Phase B"). THE PREDICATE RE-CHECKS IT ANYWAY rather than reading its own
  coverage off another module's guard.
* **``zero_metal_B`` — CARRIED INLINE** (driver.py:3286; ``stepping._zero_metal``
  :2206-2247), through :func:`.coverage.zero_metal_axes`, which is IMPORTED rather
  than re-spelled so the predicate and the compile-time choice cannot disagree.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

ONE seam-instance, reachable, from ``results/fusion_matrix_triton_2026-08-20_closed/``
at the B->H cell (``real beta PML``, ``real beta run``):

    examples:refl-angular-kz2d.py   shape (1200, 1, 1)   beta = 0.3321611

It declares an ELECTRIC source only (``source_field_types == ['D']``), so the
magnetic seam is clear; it is all-periodic with ``has_metallic == false``, so the
inline ``zero_metal_B`` carry compiles to three ``False`` flags ON THAT ROW — which
is why the gate scores every wall-clear mutation on a WALLED case instead, where
the flags are real. The D/E partner cell is worth ZERO on this corpus (that row
injects electrically between ``step_D`` and ``update_E``) and is not built.

THE OTHER BACKEND BUILT THIS FIRST: ``metal_kernels/beta_fused_magnetic_pair.py``,
certified by ``parity/meep_gpu/gate_metal_below_the_cut_fused_pairs.py``
(``results/below_the_cut_fused_pairs_2026-08-20/``), and worth the same one row
there.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot
it can claim without a composition rule nothing has measured. Wired, it would also
contend for ``step_B`` with the ``real beta PML`` arm, which IS wired and admits
the same configurations — and ``_select_slot`` (launch.py:1983-1996) leaves a slot
with two admitters UNSELECTED, taking the certified curl off the device with it.
Nothing in ``launch.py`` names this module; ``fastpath.plan_fast_path`` is
unchanged.

DEVICE STATUS: see ``parity/meep_gpu/probe_triton_beta_fused_magnetic_pair.py``
and its artifact. A weld licenses a claim, not a dispatch.
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
    _call,
    zero_metal_axes,
)
from .special_kz import (
    beta_curl_coefficients,
    beta_pml_curl_coverage,
    beta_run_constitutive_coverage,
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

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3282-3289). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills return at
#: their first line without a mirror plane — and the inert two are listed anyway:
#: what a launch REPLACES is what the driver would otherwise have called.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "REPLACES",
    "BetaFusedMagneticPairPlan",
    "beta_fused_curl_constitutive_B",
    "beta_fused_curl_constitutive_B_kernel",
    "beta_fused_magnetic_pair_coverage",
    "plan_beta_fused_magnetic_pair",
    "plan_beta_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``kernels.METALLIC``, restated for the reason
    # every sibling restates it — importing a ``tl.constexpr`` wrapper and
    # re-wrapping it is not the same object, and the comparison in the kernel body
    # is against a literal code. ONLY METALLIC IS NAMED: both certified bodies
    # branch on ``== METALLIC`` and take periodic as the else.
    METALLIC = tl.constexpr(1)

    @triton.jit
    def beta_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx,By,Bz
        u0, u1, u2,                       # curl auxiliaries: fu_Bx,fu_By,fu_Bz
        g0, g1, g2,                       # curl sources: Ex,Ey,Ez
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
        BACKWARD: tl.constexpr,           # bound to 0 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        HAS_BETA: tl.constexpr,           # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """Beta ``step_B`` + ``zero_metal_B`` + ``update_H``, one launch.

        ``kernels.fused_curl_constitutive_B``'s body with
        ``special_kz.beta_pml_curl_step``'s ``HAS_BETA`` insert in the one place
        the array path puts it. Nothing else moves.

        ``BACKWARD`` IS CARRIED AND MUST BE 0. It keeps the curl half a verbatim
        copy rather than a hand-specialised one. It cannot be 1: the D half's seam
        carries the ELECTRIC injection and an ``inv_eps`` scaling this body does
        not have.

        ``SCALE`` IS NOT CARRIED. This is the H side, where
        ``kernels.constitutive_step``'s ``SCALE = 0`` arm reads ``B`` directly; the
        ``SCALE = 1`` arm's ``inv_eps`` multiply belongs to the E side.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes`` — the question
        ``stepping._zero_metal`` asks (the grid's own declaration), NOT the
        resolved ghost rule, which an invariant axis softens to periodic while the
        wall clear still fires there.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from special_kz.beta_pml_curl_step, which is itself
        # kernels.pml_curl_step plus the HAS_BETA insert.

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
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
        # No dtdx on the term (analytic derivative, S:733-735); the subtraction IS
        # the array path's `curl + (-(c*g))`. Verbatim from beta_pml_curl_step.
        if HAS_BETA:
            curl0 = curl0 - (beta_plus * b)
            curl1 = curl1 - (beta_minus * a)

        # --- ownership mask (stepping._mask_non_owned_cells) -------------------
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY == METALLIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ == METALLIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX == METALLIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ == METALLIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX == METALLIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY == METALLIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX == METALLIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY == METALLIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ == METALLIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
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

        # --- the seam: stepping.zero_metal_B (driver.py:3286) ------------------
        # Applied to the REGISTER, before both the store and the constitutive
        # read, so the two consumers see the one value the array path leaves in B.
        # Verbatim from kernels.fused_curl_constitutive_B.
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
        # constitutive_step's SCALE=0 arm with `src0 = tl.load(g0+idx)` replaced
        # by the register v0. `prev` is read BEFORE the `w` store.
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
    beta_fused_curl_constitutive_B = None  # type: ignore[assignment]


def beta_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if beta_fused_curl_constitutive_B is None:
        raise ImportError(
            "the beta fused magnetic B/H kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return beta_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def beta_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                      sources: Any = None) -> Coverage:
    """May ONE launch span beta ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    construction :func:`.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage` both use.

    The two halves are the arms ``launch.plan_step`` really selects on a beta row
    today (``real beta PML`` on ``step_B``, ``real beta run`` on ``update_H``), so
    this predicate is narrower than the pair the planner already puts on the
    device, never wider.
    """
    reasons: List[str] = []

    curl = beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"beta curl half: {reason}" for reason in curl.reasons)
    constitutive = beta_run_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"beta constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E seam and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no magnetic source" from not being told would
    # be the over-covering refusal this clause exists to prevent.
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

    # THE TWO SYMMETRY PASSES. Both return at their first line without
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566), and clause 5 of
    # `_beta_real_grid_reasons` already refuses every fold. RE-CHECKED HERE
    # ANYWAY, not inferred: this module's coverage may not be read off another
    # module's guard, and if the beta tranche ever admits a fold this weld would
    # silently swallow two passes that had started doing work.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_B (driver.py:3285) "
            "and fill_folded_far_ghosts_B (:3287) run inside this seam and this "
            "weld carries neither")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two symmetry passes "
                f"in this seam are no longer no-ops and this weld carries neither")

    # THE WALL CLEAR is carried inline, so the grid must be able to answer which
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

class BetaFusedMagneticPairPlan:
    """One allocation-free launch for five of the driver's magnetic call sites.

    :class:`.launch.FusedPairPlan`'s ``pair='B'`` bindings plus the two
    host-rounded beta scalars and the ``HAS_BETA`` constexpr, which is exactly the
    delta :class:`.special_kz.BetaPmlCurlPlan` carries over
    :class:`.launch.PmlCurlPlan`. No inverse epsilon: that is the E side's.
    """

    __slots__ = ("shape", "n_elem", "dtdx", "beta_plus", "beta_minus", "has_beta",
                 "backward", "bc", "zero_metal", "block", "num_warps",
                 "_targets", "_aux", "_sources", "_curl_coefficients",
                 "_h_targets", "_h_aux", "_h_coefficients", "_grid", "_kernel")

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
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
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.block = int(block)
        # ONE WARP, the measured default for a fused pair
        # (launch.FUSED_DEFAULT_NUM_WARPS: fusing doubles the dependency chain a
        # program has to hide, and at Triton's default of 4 the fusion is a 16 %
        # LOSS on the pair). Explicit None remains "take Triton's default".
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._h_targets = tuple(CupyPointer(a) for a in h_targets)
        self._h_aux = tuple(CupyPointer(a) for a in h_aux)
        self._h_coefficients = tuple(CupyPointer(_flat(a)) for a in h_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else beta_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BetaFusedMagneticPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, beta_plus={self.beta_plus!r}, "
                f"has_beta={self.has_beta}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_beta_fused_magnetic_pair(fields: Any, pml: Any, sources: Any = None,
                                  block: Optional[int] = None,
                                  num_warps: Optional[int] = 1,
                                  kernel: Any = None,
                                  ) -> Optional[BetaFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not beta_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE SAME FUNCTION THE CURL PLAN CALLS, with `magnetic` from the sub-step —
    # identical arithmetic to the array path's per-call-site computation, and not
    # a second spelling of it.
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(CURL_SUB_STEP == "step_B"),
                                         complex_storage=False)
    return BetaFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), plus, minus,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H (stepping.py:948 against the B curl's
        # half_integer=True). The kernel takes both and never asks which is which,
        # so a swap here is a silent half-cell error in the absorber profile; the
        # gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


def plan_beta_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        dtdx: float, beta_plus: float, beta_minus: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1,
        ) -> BetaFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``kernel=``
    carries the mutation override and ``has_beta=0`` the identity leg's
    certified-kernel arm, which must reproduce
    ``kernels.fused_curl_constitutive_B`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return BetaFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, beta_plus, beta_minus,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )
