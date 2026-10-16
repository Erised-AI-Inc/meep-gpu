"""COMPLEX cylindrical (Dcyl, complex64 storage at every m) PML curl -- ONE curl kernel,
the shared radial scan, and no new constitutive kernel.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid under
complex64 storage (``force_complex_fields=True``) with an active split-field PML, at
every integer m: the curls, the radial sum, the i*m/r pass, the |m| = 1 axis-row
increments and the per-|m| axis rules. The real-storage m = 0 run is
:mod:`.cylindrical_triton`'s; the two products are split by STORAGE, and each predicate
refuses the other's by name. The constitutive sub-steps need no cylindrical kernel:
``stepping.update_H`` / ``update_E`` carry no cylindrical branch, so
:func:`plan_cylindrical_complex_constitutive` builds
:class:`.complex_fields.ComplexConstitutivePlan` behind its own predicate.

EVERY STORED PART IS STEPPED WITH REAL OPERANDS ALONE, as MEEP's ``DOCMP`` loop steps
``f[c][0]`` and ``f[c][1]`` (update_eh.cpp:149, step_db.cpp:47): a complex64 volume is a
float32 word pair at ``2*idx`` / ``2*idx + 1``, every product has a real operand, and no
complex multiply and no multiply-add appears anywhere in this file. The forms are the
array path's (:mod:`meep_gpu.stepping` of this tree), transcribed with their grouping:

1. THE RADIAL SUM. Bz's and Dz's WHOLE curl is ``dtdx`` times MEEP's per-chunk radial
   difference of ``f_rderiv_int`` (``stepping.cylindrical_rderiv_difference``,
   step_db.cpp:86-119), computed by :func:`.cylindrical_triton.cyl_radial_difference`
   -- the column-serial scan has one home, and on complex storage each stored part is
   one scan column. The plan launches the scan, then the curl.

2. THE OTHER CURLS (``stepping._curl_from_operands``) through
   :func:`.forms.curl_stencil` with the chunk record's partner-presence word, per part:
   phi is invariant, so Bx/Dx take ``(-dtdx) * (g2s - g2)`` and By/Dy MEEP's
   left-to-right ``dtdx * (((g1s - g1) + g2) - g2s)``. The ownership mask
   (``stepping._mask_non_owned_cells``) zeroes row 0 of every target whose r Yee shift
   is 0 and, on a metallic z wall, cell 0 of every target whose z shift is 0.

3. THE |m| = 1 AXIS-ROW INCREMENTS (``stepping._cylindrical_axis_increment_B`` / ``_D``,
   step_db.cpp:347-397), which REPLACE curl row 0 after the mask, before the PML forms,
   so the recurrence applies MEEP's axis ladder:

   * B, Bx: ``bracket = (Ep[0] - Ep[0, z+1]) - partner`` with ``partner =
     (m*Ez[1].im, -(m*Ez[1].re))`` -- MEEP's ``((1-2*cmp)*m) * f[Ez][1-cmp]`` at the
     FIRST OFF-AXIS row;
   * D, Dy: ``bracket = (Hr[0] - Hr[0, z-1]) - 2*Hz[0]``;

   each part ``realnum(double(+-Courant) * double(bracket))`` -- a DOUBLE product rounded
   once (step_db.cpp:377; ``stepping._realnum_double_product``), ``-Courant`` on B and
   ``+Courant`` on D -- and the curl row takes its negation. The double Courant words are
   bound as a ``float64`` vector (:func:`.cylindrical_triton.courant_constants`): a scalar
   kernel argument rides the fp32 ABI and would round them first.

4. THE PML FORMS (``stepping._apply_curl_regions``) through :func:`.forms.curl_form`
   keyed on the chunk record ``cb``, per part: the recurrence on the target's dsigu
   cells, the dsig-only form elsewhere, ``fu`` read and written on the recurrence's
   cells only.

5. THE i*m/r PASS (``stepping._cylindrical_imr_increment``, step_db.cpp:177-294), a pass
   of its own AFTER every component's curl, compiled out at m = 0. Per (target,
   partner, sign) -- Bx <- Ez (+1), Bz <- Ex (-1); Dx <- Hz (-1), Dz <- Hx (+1) -- the
   increment is the quarter turn of the partner by the REAL row ``rinv = the_m /
   here_r``, a float32 quotient (``the_m = realnum(sign*2*m*Courant)``, ``here_r =
   realnum(max(2*ir + iyee_r, 1))``), built on the host the array path's way and bound
   as a device row (:func:`imr_rinv_row`), so the kernel divides nothing:
   ``inc = (-(rinv*g.im), rinv*g.re)``, the negation spelled ``* -1.0``. It is masked to
   ``+0`` on the target's unowned cells and applied through MEEP's additive ladder
   (``stepping._apply_folded_increment``): ``df = sinv*inc``; ``f - df`` outside the
   target's dsigu cells, ``fu - df`` and ``f - sinv_u*df`` inside them. Applied in
   register after the curl's form: a float32 stored and reloaded is the identity, so the
   in-register order (curl form, then the ladder) is the array path's two passes. The
   partners are the centre loads the curl already makes (``c`` = g2 = Ez/Hz for target
   0, ``a`` = g0 = Ex/Hx for target 2).

6. THE PER-|m| AXIS RULES, on the updated words (``stepping._cylindrical_axis_zero_B`` /
   ``_D``): m = 0 -- ``Br[0] = 0``; ``Dz[0] += realnum(double(4*Courant) *
   double(Hp[0]))`` per part, then ``Dp[0] = 0``; |m| = 1 -- ``Dz[0] = 0`` (the field
   only), nothing on B; |m| >= 2 -- all three components AND their ``fu`` held at ``+0``
   on rows ``[0, ZERO_ROWS)`` (``|m|``, or 1 on the accurate branch;
   ``stepping._cylindrical_axis_rows``), ``fu`` on every cell of those rows, as the array
   path writes it.

THE r AXIS NEEDS NO GHOST CODE. ``stepping._shift_up``'s CYL_AXIS branch is the metallic
far-face zero, which a masked load delivers. The near ghost (``r_to_minus_r``) reaches
only row 0 of a shift-down along r, which only Dy takes (partner Hz), and Dy's r Yee
shift is 0, so the mask drops that row and the |m| = 1 increment then replaces it. The
kernel serves an exact ``+0`` there; a mutation that puts a faithful ghost back must
come out NOT caught.

``launch.plan_step`` selects these plans for the strict predicates below; the fast path
(``fastpath``) reaches them only through ``plan_step``, never by importing this module.

Import contract: importable WITHOUT Triton -- the predicates and the plan builders (to
``None``) answer on a host without it; the kernel degrades to
``complex_fields._UnavailableKernel``, and ``ENABLE_FP_FUSION`` is imported from
:mod:`.kernels` only inside ``run()``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .complex_fields import _UnavailableKernel

try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .complex_fields import (  # READ-ONLY: restate, never edit, the complex base
    ComplexConstitutivePlan,
    _complex_layout_reasons,
    _word_view,
)
from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
)
from .cylindrical_triton import (  # the radial scan's one home, and its bindings
    COURANT_WORDS,
    RadialScan,
    courant_constants,
    radial_scan_starts,
)
from .launch import SUB_STEPS, CupyPointer, _flat, chunk_bounds_for

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Restated rather
# than imported: importing kernels.py pulls Triton in unconditionally and defeats the
# host-only route.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: ``M_CLASS`` arms. ``M_ZERO`` is m = 0 UNDER COMPLEX64 STORAGE: the i*m/r pass is
#: compiled out (the array path forms it only for ``grid.m != 0``), no axis-row increment
#: exists, and the axis rules are the m = 0 pair. ``M_ONE`` is |m| = 1 (the axis-row
#: increments on Bx and Dy, Dz zeroed on the axis), ``M_MANY`` |m| >= 2 (the near-axis
#: zeroing of all six volumes).
M_ZERO = tl.constexpr(0)
M_ONE = tl.constexpr(1)
M_MANY = tl.constexpr(2)

#: Word indices of :func:`.cylindrical_triton.courant_constants`, as constexprs a kernel
#: body may read.
COURANT_FOUR = tl.constexpr(COURANT_WORDS["four"])
COURANT_PLUS = tl.constexpr(COURANT_WORDS["plus"])
COURANT_MINUS = tl.constexpr(COURANT_WORDS["minus"])

#: The boundary triple this family requires on r and phi, and the two it admits on z.
REQUIRED_R_KIND = "axis"
REQUIRED_PHI_KIND = "periodic"
ADMITTED_Z_KINDS: Tuple[str, ...] = ("metallic", "periodic")

#: ``grid.is_axis`` on the three axes. Refused unless it is exactly this.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The smallest radial extent this product may step. The |m| = 1 axis increment reads
#: the FIRST OFF-AXIS row (``stepping._cylindrical_axis_increment_B``'s
#: ``xp.take(Ez, 1, axis=0)``) and the array path itself raises ``IndexError`` below
#: this -- see the clause in :func:`_cylindrical_geometry_reasons`.
MINIMUM_RADIAL_ROWS = 2

#: The i*m/r pass's terms -- step_db.cpp:177-294 via ``stepping._CYLINDRICAL_IMR_B`` /
#: ``_D``. ``(target index, partner register, sign)``; the partner register names which
#: CENTRE load of the curl the term reuses ("c" = g2 = Ez/Hz, "a" = g0 = Ex/Hx). Target
#: 1 gets no term on either side.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "c", +1.0), (2, "a", -1.0)),   # Bx <- Ez (+1), Bz <- Ex (-1)
    "step_D": ((0, "c", -1.0), (2, "a", +1.0)),   # Dx <- Hz (-1), Dz <- Hx (+1)
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0.
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}   # Bx / Dy

#: Per-sub-step radial-sum source, as ``stepping.step_B`` / ``step_D`` pass it: Ep = Ey
#: with its stored wall row at ``ir0 = 0.0`` (B), Hp = Hy at ``ir0 = 0.5`` (D).
PREFIX: Dict[str, Dict[str, Any]] = {
    "step_B": {"component": "Ey", "ir0": 0.0, "extend_wall_row": True},
    "step_D": {"component": "Hy", "ir0": 0.5, "extend_wall_row": False},
}

CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: Elements per program -- complex CELLS, not words. Restated from
#: ``kernels.DEFAULT_BLOCK`` for the same import reason; no autotune, because the kernel
#: writes its own inputs in place.
DEFAULT_BLOCK = 256

__all__ = [
    "ADMITTED_Z_KINDS",
    "AXIS_INCREMENT_TARGET",
    "CylindricalComplexCurlPlan",
    "IMR_TERMS",
    "MINIMUM_RADIAL_ROWS",
    "PREFIX",
    "cylindrical_complex_constitutive_coverage",
    "cylindrical_complex_curl_coverage",
    "cyl_complex_pml_curl_step",
    "imr_rinv_row",
    "m_class",
    "plan_cylindrical_complex_constitutive",
    "plan_cylindrical_complex_curl",
    "plan_cylindrical_complex_curl_from_arrays",
    "zero_rows",
]


# ---------------------------------------------------------------------------
# Host-side constants that must be computed the array path's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``M_CLASS`` for one azimuthal order: 0 at m = 0, 1 at |m| = 1, 2 above."""
    if int(m) == 0:
        return 0
    return 1 if abs(int(m)) == 1 else 2


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows``: ``slice(0, 1)`` on the ACCURATE branch and
    ``slice(0, |m|)`` on MEEP's default near-axis rule. Zero below |m| = 2, where
    neither branch applies.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_rinv_row(target: str, sign: float, m: int, courant: float, rows: int) -> Any:
    """The i*m/r pass's REAL per-row coefficient, as a host float32 vector.

    ``stepping._cylindrical_imr_increment``'s ``_build``, transcribed: ``the_m`` is the
    double product ``sign*2*m*Courant`` (left to right) rounded once to ``realnum``;
    ``here_r`` is the target's doubled r coordinate ``2*ir + iyee_r`` (its OWN radial
    Yee shift, ``fields.IYEE_SHIFTS``), clamped to at least 1 and converted to
    ``realnum``; ``rinv = the_m / here_r`` is a float32 QUOTIENT -- MEEP's ``realnum``
    division, not a double one rounded once. The clamp only reaches row 0 of a shift-0
    target, which the ownership mask zeroes.
    """
    import numpy  # noqa: PLC0415

    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    iyee_r = IYEE_SHIFTS[target][0]
    the_m = numpy.float32(sign * 2.0 * int(m) * float(courant))
    r_doubled = 2 * numpy.arange(int(rows), dtype=numpy.int64) + iyee_r
    here_r = numpy.maximum(r_doubled, 1).astype(numpy.float32)
    return numpy.ascontiguousarray(the_m / here_r)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

# The forms and the record layout, read by the kernel below. Imported only when Triton
# is: both modules import it at module scope.
if _TRITON_IMPORT_ERROR is None:
    from .forms import axis_present, curl_form, curl_stencil, record_in
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE


@triton.jit
def cyl_complex_pml_curl_step(
    f0, f1, f2,                   # targets:     Bx,By,Bz or Dx,Dy,Dz (c8 as words)
    u0, u1, u2,                   # auxiliaries: fu_B*    or fu_D*    (c8 as words)
    g0, g1, g2,                   # sources:     Ex,Ey,Ez or Hx,Hy,Hz (c8 as words)
    rad,                          # the radial difference (c8 as words)
    q0, q2,                       # i*m/r rinv rows for targets 0 and 2 (f32, nr long)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, f32, ONE sub-lattice
    cb,                           # chunk record: the forms' membership and presence
    courant,                      # float64 [4*Courant, Courant, -Courant]
    nx, ny, nz, n_elem, dtdx,     # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    mf,                           # float32(m): the |m| = 1 B-side partner's factor
    BACKWARD: tl.constexpr,       # 0 = B (forward differences), 1 = D
    BCZ: tl.constexpr,            # PERIODIC or METALLIC; r is METALLIC, phi PERIODIC
    M_CLASS: tl.constexpr,        # M_ZERO, M_ONE or M_MANY
    ZERO_ROWS: tl.constexpr,      # |m| or 1 under M_MANY; unused otherwise
    BLOCK: tl.constexpr,
):
    """One COMPLEX cylindrical curl sub-step: the module docstring's six steps, per part."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis -------------------------------------------------
    # r is CYL_AXIS: the far face is the metallic zero and the near ghost is not
    # implemented (unobservable; module docstring). phi is PERIODIC on one cell.
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vr = live & (si >= 0) & (si < nx)
    sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    vp = live
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        vz = live

    o_r = si * nyz + j * nz + k
    o_p = i * nyz + sj * nz + k
    o_z = i * nyz + j * nz + sk

    # --- loads: two words per operand -----------------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_z_re = tl.load(g0 + 2 * o_z, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * o_z + 1, mask=vz, other=0.0)
    b_z_re = tl.load(g1 + 2 * o_z, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * o_z + 1, mask=vz, other=0.0)
    c_r_re = tl.load(g2 + 2 * o_r, mask=vr, other=0.0)
    c_r_im = tl.load(g2 + 2 * o_r + 1, mask=vr, other=0.0)
    c_p_re = tl.load(g2 + 2 * o_p, mask=vp, other=0.0)
    c_p_im = tl.load(g2 + 2 * o_p + 1, mask=vp, other=0.0)

    # --- 1-2. the curls, per part ---------------------------------------------------
    presence = tl.load(cb + RECORD_PRESENCE)
    pr = axis_present(presence, 0)
    pp = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0_re = curl_stencil(c_p_re, c_re, b_re, b_z_re, dtdx, pp, pz)
    curl0_im = curl_stencil(c_p_im, c_im, b_im, b_z_im, dtdx, pp, pz)
    curl1_re = curl_stencil(a_z_re, a_re, c_re, c_r_re, dtdx, pz, pr)
    curl1_im = curl_stencil(a_z_im, a_im, c_im, c_r_im, dtdx, pz, pr)
    curl2_re = dtdx * tl.load(rad + 2 * idx, mask=live, other=0.0)
    curl2_im = dtdx * tl.load(rad + 2 * idx + 1, mask=live, other=0.0)

    # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) ----------
    #   B: Bx(0,1,1) -> r ; By(1,0,1) -> phi only, periodic: nothing ; Bz(1,1,0) -> z
    #   D: Dx(1,0,0) -> phi (nothing) + z ; Dy(0,1,0) -> r + z ; Dz(0,0,1) -> r
    # The z clauses fire only when z is METALLIC. Both parts take +0.
    at_r, at_z = i == 0, k == 0
    if BACKWARD:
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        curl1_re = tl.where(at_r, 0.0, curl1_re)
        curl1_im = tl.where(at_r, 0.0, curl1_im)
        curl2_re = tl.where(at_r, 0.0, curl2_re)
        curl2_im = tl.where(at_r, 0.0, curl2_im)
    else:
        curl0_re = tl.where(at_r, 0.0, curl0_re)
        curl0_im = tl.where(at_r, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- 3. the |m| = 1 axis-row increment, AFTER the mask, REPLACING row 0 -----------
    if M_CLASS == M_ONE:
        if BACKWARD:
            # Dy: (Hr[0] - Hr[0, z-1]) - 2*Hz[0]; Hr = g0 centre `a` and its z-down
            # neighbour, Hz = g2 centre `c`. Times +Courant in double, rounded once.
            s_re = (a_re - a_z_re) - (2.0 * c_re)
            s_im = (a_im - a_z_im) - (2.0 * c_im)
            factor = tl.load(courant + COURANT_PLUS)
            inc_re = tl.cast(tl.cast(s_re, tl.float64) * factor, tl.float32)
            inc_im = tl.cast(tl.cast(s_im, tl.float64) * factor, tl.float32)
            curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)
            curl1_im = tl.where(at_r, inc_im * -1.0, curl1_im)
        else:
            # Bx: (Ep[0] - Ep[0, z+1]) - partner, with partner = (m*Ez[1].im,
            # -(m*Ez[1].re)) read at the FIRST OFF-AXIS row of this lane's column.
            # Times -Courant in double, rounded once.
            off = nyz + j * nz + k
            e1_re = tl.load(g2 + 2 * off, mask=live, other=0.0)
            e1_im = tl.load(g2 + 2 * off + 1, mask=live, other=0.0)
            partner_re = mf * e1_im
            partner_im = (mf * e1_re) * -1.0
            s_re = (b_re - b_z_re) - partner_re
            s_im = (b_im - b_z_im) - partner_im
            factor = tl.load(courant + COURANT_MINUS)
            inc_re = tl.cast(tl.cast(s_re, tl.float64) * factor, tl.float32)
            inc_im = tl.cast(tl.cast(s_im, tl.float64) * factor, tl.float32)
            curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)
            curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)

    # --- 4. the PML forms (stepping._apply_curl_regions), per part --------------------
    # dsig/dsigu cycle: target 0 -> (phi, z), 1 -> (z, r), 2 -> (r, phi). One record
    # select per target serves both parts; fu read and written on its cells only.
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

    # --- 5. the i*m/r pass, in register after the curl's form -------------------------
    if M_CLASS != M_ZERO:
        r0 = tl.load(q0 + i, mask=live, other=0.0)
        r2 = tl.load(q2 + i, mask=live, other=0.0)
        # quarter turn by the real row: (-(rinv*g.im), rinv*g.re)
        i0_re = (r0 * c_im) * -1.0
        i0_im = r0 * c_re
        i2_re = (r2 * a_im) * -1.0
        i2_im = r2 * a_re
        # the targets' unowned cells keep +0 (the same mask as their curls)
        if BACKWARD:
            if BCZ == METALLIC:
                i0_re = tl.where(at_z, 0.0, i0_re)
                i0_im = tl.where(at_z, 0.0, i0_im)
            i2_re = tl.where(at_r, 0.0, i2_re)
            i2_im = tl.where(at_r, 0.0, i2_im)
        else:
            i0_re = tl.where(at_r, 0.0, i0_re)
            i0_im = tl.where(at_r, 0.0, i0_im)
            if BCZ == METALLIC:
                i2_re = tl.where(at_z, 0.0, i2_re)
                i2_im = tl.where(at_z, 0.0, i2_im)
        # MEEP's additive ladder (stepping._apply_folded_increment): target 0's dsig
        # is phi and dsigu z, target 2's dsig r and dsigu phi.
        d_re = si_y * i0_re
        d_im = si_y * i0_im
        v0_re = tl.where(du0, v0_re - (si_z * d_re), v0_re - d_re)
        v0_im = tl.where(du0, v0_im - (si_z * d_im), v0_im - d_im)
        n0_re = n0_re - d_re
        n0_im = n0_im - d_im
        d_re = si_x * i2_re
        d_im = si_x * i2_im
        v2_re = tl.where(du2, v2_re - (si_y * d_re), v2_re - d_re)
        v2_im = tl.where(du2, v2_im - (si_y * d_im), v2_im - d_im)
        n2_re = n2_re - d_re
        n2_im = n2_im - d_im

    # --- 6. the per-|m| axis rules, on the updated words ------------------------------
    um0 = live & du0
    um1 = live & du1
    um2 = live & du2
    if M_CLASS == M_ZERO:
        if BACKWARD:

            v1_re = tl.where(at_r, 0.0, v1_re)
            v1_im = tl.where(at_r, 0.0, v1_im)
            four = tl.load(courant + COURANT_FOUR)
            h_re = tl.cast(tl.cast(b_re, tl.float64) * four, tl.float32)
            h_im = tl.cast(tl.cast(b_im, tl.float64) * four, tl.float32)
            v2_re = tl.where(at_r, v2_re + h_re, v2_re)
            v2_im = tl.where(at_r, v2_im + h_im, v2_im)
        else:
            v0_re = tl.where(at_r, 0.0, v0_re)
            v0_im = tl.where(at_r, 0.0, v0_im)
    if M_CLASS == M_ONE:
        if BACKWARD:
            v2_re = tl.where(at_r, 0.0, v2_re)
            v2_im = tl.where(at_r, 0.0, v2_im)
    if M_CLASS == M_MANY:
        near = i < ZERO_ROWS
        v0_re = tl.where(near, 0.0, v0_re)
        v0_im = tl.where(near, 0.0, v0_im)
        v1_re = tl.where(near, 0.0, v1_re)
        v1_im = tl.where(near, 0.0, v1_im)
        v2_re = tl.where(near, 0.0, v2_re)
        v2_im = tl.where(near, 0.0, v2_im)
        n0_re = tl.where(near, 0.0, n0_re)
        n0_im = tl.where(near, 0.0, n0_im)
        n1_re = tl.where(near, 0.0, n1_re)
        n1_im = tl.where(near, 0.0, n1_im)
        n2_re = tl.where(near, 0.0, n2_re)
        n2_im = tl.where(near, 0.0, n2_im)
        # the array path zeroes every fu word of the near rows, inside a layer or not
        um0 = live & (du0 | near)
        um1 = live & (du1 | near)
        um2 = live & (du2 | near)

    # --- stores: fu on its cells, then the field everywhere, both words --------------
    tl.store(u0 + 2 * idx, n0_re, mask=um0)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=um0)
    tl.store(u1 + 2 * idx, n1_re, mask=um1)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=um1)
    tl.store(u2 + 2 * idx, n2_re, mask=um2)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=um2)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


# ---------------------------------------------------------------------------
# Coverage — the POSITIVE refusal enumeration
# ---------------------------------------------------------------------------
#
# WRITTEN POSITIVELY AND ENUMERATING. Every requirement is named and checked and
# coverage is never inferred from the absence of a known blocker — the two worst
# defects in this project were silent wrong answers, not crashes.
#
# These predicates do NOT call ``coverage._grid_reasons`` (its clauses 2, 6 and 7
# refuse this whole domain) and do NOT call
# ``complex_fields._complex_grid_reasons`` (its clause 4 refuses cylindrical
# outright). Both are RIGHT for their own products and must stay; their
# non-cylindrical clauses are restated below, individually, with the cylindrical
# ones replaced by positive Dcyl requirements. That duplication is deliberate and
# it is the reason neither shared clause may be widened to admit Dcyl: each is
# now load-bearing in two directions at once.

#: The shared helpers these predicates DO reuse, named as data so a laptop test
#: can assert every one still exists and a rename in a shared file fails at the
#: merge bar instead of silently dropping a clause.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_boundary_kinds", "_call", "_coefficient_reasons", "_inverse_epsilon_reasons",
    "_susceptibility_reasons", "_complex_layout_reasons")


def _shared_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses both predicates share, cylindrical geometry excluded."""
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED against coverage.py clause 2). Complex64 storage is REQUIRED.
    #    At |m| >= 1 the array path REFUSES real storage outright
    #    (stepping._cylindrical_imr_increment raises, and MEEP's change_m aborts
    #    on the same combination); at m = 0 both storages exist and the DECLARATION is
    #    what splits the two cylindrical products — a real m = 0 run is
    #    cylindrical_triton's (it refuses force_complex_fields=True by name), a
    #    complex one is this family's M_ZERO arm. The storage dtype itself is
    #    verified by the layout clause; this clause checks the declaration.
    if not getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields is not set: this family steps complex64 word "
            "pairs only. |m| >= 1 has no real-storage form (the array path "
            "raises, MEEP's change_m aborts), and a REAL m = 0 Dcyl run is "
            "cylindrical_triton's kernel, not this one")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "cylindrical split-field path only)")

    # 4. No conductivity anywhere on the curl targets. A missing or non-callable
    #    reader is refused OUTRIGHT — inferring "no conductivity" from the
    #    ABSENCE of condfac_for is admission by attribute absence, the exact
    #    reasoning coverage exists to refuse (complex_fields.py:875-884).
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; conductive complex "
                    f"cylindrical stepping is a separate, unbuilt product")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 5. No dispersion, refused OUTRIGHT: none of the sixteen demand rows carries
    #    a susceptibility, and complex ADE belongs to a future tranche. The shape
    #    clauses still run so an unreadable susceptibility is named, not shrugged.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche, and no cylindrical corpus row combines the two")
    reasons.extend(_susceptibility_reasons(fields))

    # 6. No instantaneous nonlinearity, no BFAST, no special_kz beta. Grid already
    #    refuses the last two on a cylindrical cell (grid.py:689-696 beta,
    #    :734-741 BFAST) — the clauses are written because "another module already
    #    guards it" is exactly the reasoning this file exists to refuse.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 7. NO BLOCH PHASE. Dcyl Bloch is z-only in MEEP (grid.py:654-658 refuses a k
    #    on r or phi), none of the sixteen rows carries one, and the kernel
    #    compiles no rotation at all. Admitting a z phase later is a predicate
    #    widening plus a gate row — not a silent extension.
    if getattr(grid, "has_bloch", False):
        reasons.append(
            f"nonzero k_point {getattr(grid, 'k_point', None)!r}: this kernel "
            f"compiles no Bloch rotation (a z-only Dcyl phase is a future widening)")
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {k_point!r} is not exactly zero")

    # 8. Stored E — the invariant behind differencing E while a pole is live, and
    #    the one an edit that ever makes stores_E optional under PML must be
    #    caught by (coverage.py:296-304's argument).
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 9. THE DIVISION'S r CUTS ARE GLOBAL. The radial sum restarts at each chunk's
    #    first row; a lifted division whose r cuts differ along z keys the array
    #    path's sum per cell, which the scan's per-row line cannot carry (DESIGN D8).
    layout = None
    try:
        layout = getattr(fields, "chunk_layout", None)
    except Exception:  # noqa: BLE001 - an unreadable attribute is not an installed layout
        layout = None
    if layout is not None and len(getattr(layout, "boxes", ())) > 1:
        try:
            cuts = layout.boundaries(0)
        except Exception as exc:  # noqa: BLE001 - an unreadable division is refused
            reasons.append(f"the chunk division's r cuts could not be read: {exc!r}")
        else:
            if cuts is None:
                reasons.append(
                    "the chunk division's r cuts differ along z (the radial sum is keyed "
                    "per cell there; the scan kernel carries global r cuts only)")

    return reasons


def _cylindrical_geometry_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The POSITIVE Dcyl + |m| >= 1 requirements."""
    reasons: List[str] = []

    if not getattr(grid, "cylindrical", False):
        reasons.append("grid is not cylindrical (this product steps Dcyl grids only)")

    # EVERY INTEGER m, since 2026-09-04. m = 0 used to be refused here by name as
    # cylindrical_triton's row; the two products are now split by STORAGE (clause
    # 2 of _shared_reasons requires force_complex_fields, cylindrical_triton
    # refuses it), so a complex m = 0 run — corpus row
    # examples:dipole_in_vacuum_cyl_off_axis.py — is this family's M_ZERO arm and
    # a real m = 0 run is still the other kernel's. An unreadable m is refused: it
    # selects the arm.
    m = getattr(grid, "m", None)
    if m is None:
        reasons.append("grid does not report m")
    else:
        try:
            int(m)
        except Exception:  # noqa: BLE001 - an unreadable m is not coverage
            reasons.append(f"grid.m = {m!r} is not an integer")

    flags = tuple(bool(_call(grid, "is_axis", axis, default=False)) for axis in range(3))
    if flags != CYLINDRICAL_AXIS_FLAGS:
        reasons.append(f"grid.is_axis {flags!r} is not {CYLINDRICAL_AXIS_FLAGS!r} "
                       "(r must be the leading axis)")

    kinds = _boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        if kinds[0] != REQUIRED_R_KIND:
            reasons.append(f"axis 0 boundary {kinds[0]!r} is not {REQUIRED_R_KIND!r}")
        if kinds[1] != REQUIRED_PHI_KIND:
            reasons.append(f"axis 1 boundary {kinds[1]!r} is not {REQUIRED_PHI_KIND!r}")
        if kinds[2] not in ADMITTED_Z_KINDS:
            reasons.append(f"axis 2 boundary {kinds[2]!r} is outside {ADMITTED_Z_KINDS!r}")

    # No fold anywhere. Grid already refuses a mirror on a Dcyl cell
    # (grid.py:648-653); the clause is written for the reason above.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
    else:
        if int(shape[1]) != 1:
            reasons.append(f"phi extent is {shape[1]} cells, not 1 (Dcyl is 2.5-D: "
                           "the exp(i*m*phi) dependence is analytic)")
        # AT LEAST ONE OFF-AXIS RADIAL ROW. Measured 2026-08-13 on a real
        # Grid(cell_size=(1.0, 0, 20), m=1, complex, z metallic) -> shape
        # (1, 1, 20): ``stepping.step_B`` RAISES ``IndexError: index 1 is out of
        # bounds for axis 0 with size 1`` from the |m| = 1 axis increment's
        # ``xp.take(Ez, 1, axis=0)`` (:642), while this predicate admitted the run
        # with no reasons at all. The kernel's matching load — ``off = nyz +
        # j*nz + k`` under ``mask=live`` (:753-755) — is true on every lane of
        # that grid, so it would read a full plane PAST THE END of Ez and store a
        # silently wrong axis row where the array path stops loudly.
        #
        # AND ON THE GPU BACKEND IT DOES NOT STOP LOUDLY AT ALL. Re-measured on
        # the GPU host 2026-08-13 (CuPy 13.5.1, RTX A6000): ``cupy.take`` DOES NOT
        # BOUNDS-CHECK — ``cp.take(a, 1, axis=0)`` on a size-1 axis returns row
        # 0's data instead of raising, so the SAME driver on the SAME grid steps
        # to completion with a finite state where NumPy refuses. The two
        # backends do not agree about whether the configuration is steppable,
        # which is a defect of the ARRAY PATH (``stepping.py``, not this file's
        # to edit) and is recorded here because it changes what this clause is
        # for: it is not a convenience that mirrors a loud array-path failure,
        # it is the only thing between a GPU run and a silently wrong axis row.
        # The evidence is the composition probe's ``nr>=2 refusal`` leg, which
        # measures the per-backend behaviour rather than asserting one of them.
        #
        # WHY THE CLAUSE IS UNCONDITIONAL rather than |m| = 1 only: at |m| >= 2
        # nr = 1 does not fault, but every row is inside ``ZERO_ROWS`` and the
        # whole volume is held at zero, so the configuration is degenerate in
        # both products; and a clause that has to re-derive the m class is a
        # clause that can disagree with :func:`m_class`.
        #
        # ``stepping._require_cylindrical_steppable`` (:513-522) is the hook that
        # would otherwise carry this and is a deliberate ``del grid`` no-op, so
        # the requirement is stated HERE — it may not be inferred from a helper
        # that checks nothing.
        if int(shape[0]) < MINIMUM_RADIAL_ROWS:
            reasons.append(
                f"radial extent is {shape[0]} cell(s), fewer than "
                f"{MINIMUM_RADIAL_ROWS}: the |m| = 1 axis increment reads the "
                f"FIRST OFF-AXIS row (stepping.py:671, where NumPy raises "
                f"IndexError and CuPy silently returns row 0 — both measured) "
                f"and the kernel's matching load would run past the end of the "
                f"source volume")

    # accurate_fields_near_cylorigin is READ, never assumed: it selects
    # ZERO_ROWS at |m| >= 2 (stepping._cylindrical_axis_rows, :601-622) and a
    # grid that cannot answer would compile the wrong constexpr — a plane of
    # wrong values, not a crash.
    if getattr(grid, "accurate_fields_near_cylorigin", None) is None:
        reasons.append("grid does not report accurate_fields_near_cylorigin; it "
                       "selects the |m| >= 2 near-axis row count and may not be assumed")

    # A configuration the array path itself refuses is not one a kernel may step.
    try:
        from ..stepping import _require_cylindrical_steppable  # noqa: PLC0415

        _require_cylindrical_steppable(grid)
    except Exception as exc:  # noqa: BLE001 - an unsteppable grid is refused
        reasons.append(f"stepping refuses this cylindrical grid: {exc!r}")

    return reasons


def cylindrical_complex_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May the complex cylindrical curl kernel step this (fields, pml, sub_step)?

    Every clause names its own requirement and appends its own reason; the scan
    continues after a failure so a refusal reports everything that disqualified
    the run rather than the first thing.

    An off-diagonal chi1inv row is ADMITTED here (constitutive-only, its whole
    effect is inside ``update_E``, stepping.py:1001-1008) and refused by the E-side
    constitutive predicate — the same per-sub-step split coverage.py makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + CURL_SOURCES)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # Both sub-lattices: the suffix the plan binds is the sub-step's own, and
        # a swap is a silent half-cell error in the absorber profile.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the CERTIFIED ``complex_fields.bloch_constitutive_step`` run here?

    IT NEEDS NO NEW KERNEL, and that is a MEASURED finding rather than an
    assumption: ``stepping.update_H`` (:907-924) and ``stepping.update_E``
    (:926-993) carry no cylindrical branch at all, and the identity leg in the
    module docstring reports **480/480 rows, 0 differing uint32 words** against a
    plain complex elementwise ``dsigw`` reference on real complex Dcyl grids.
    (An earlier revision of this line said 640, an m set the harness does not
    carry; the count is computed from the leg's own axes by
    ``constitutive_identity_row_count`` and a laptop test pins the two together.)
    What refuses them today is ``complex_fields._complex_grid_reasons`` clause 4,
    the blanket cylindrical refusal — which is shared, is right for that product,
    and must stay.

    The enumeration is :func:`cylindrical_complex_curl_coverage`'s minus the
    curl-specific clauses (no ghost rule, no ownership mask, no prefix, no axis
    rules), plus the E side's own refusals: a registered polarization (already
    refused module-wide), an off-diagonal chi1inv row (the row product reads
    neighbours and the sub-step stops being element-wise), and ``stores_E``.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        # inv_eps stays FLOAT32 under complex storage (stepping.py:37-38,
        # fields.py:1203-1204), so coverage.py's float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))




# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class CylindricalComplexCurlPlan:
    """A launchable complex cylindrical curl sub-step: the radial scan, then ONE kernel.

    Built two ways and launched ONE way, exactly as ``launch.PmlCurlPlan`` is:
    :func:`plan_cylindrical_complex_curl` is the engine route (reads a real
    ``Fields``/``PML`` and passes through the predicate),
    :func:`plan_cylindrical_complex_curl_from_arrays` is the harness route (bare device
    arrays, no predicate, the chunk record and the scan's starts line REQUIRED,
    ``kernel=`` / ``scan_kernel=`` for the mutation legs). Both produce this object and
    both go through :meth:`run`, so the bytes a gate certifies are the bytes the engine
    would launch.

    Allocation-free per step: the scan's output and bindings, the two i*m/r rows (grid
    invariants -- Yee shift, radial extent, m, Courant -- built once, as the array path's
    ``scratch.constant`` builds them once per run) and the Courant words are bound at
    plan time.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "courant", "m", "backward", "bcz",
                 "m_class", "zero_rows", "block", "num_warps", "chunk_bounds",
                 "scan", "_targets", "_aux", "_sources", "_coefficients", "_imr_rows",
                 "_cb", "_courant", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, courant: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int,
                 block: int, targets, auxiliaries, sources, coefficients, xp: Any, *,
                 chunk_bounds: Any, scan_starts: Any, kernel: Any = None,
                 scan_kernel: Any = None, num_warps: Optional[int] = None) -> None:
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        spec = SUB_STEPS[sub_step]
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "CylindricalComplexCurlPlan")
        # float(): the array path scales each part by a Python float, which it rounds
        # to the part's float32 (part_products.scale); Triton types a Python float
        # argument as fp32 -- so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        self.courant = float(courant)
        self._courant = CupyPointer(xp.asarray(courant_constants(self.courant)))
        self.m = int(m)
        self.backward = int(spec["backward"])
        self.bcz = int(bcz)
        self.m_class = m_class(m)
        self.zero_rows = zero_rows(m, accurate_fields_near_cylorigin)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        source_map = dict(zip(spec["sources"], sources))
        self.scan = RadialScan(sub_step, source_map[PREFIX[sub_step]["component"]],
                               scan_starts, xp, block=self.block, kernel=scan_kernel)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        rows = self.shape[0]
        self._imr_rows = tuple(
            CupyPointer(xp.asarray(imr_rinv_row(spec["targets"][index], sign, m,
                                                courant, rows)))
            for index, _register, sign in IMR_TERMS[sub_step])
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step: the scan, then the curl. In place; references stay valid.

        ``guard`` is not for callers: it exists so a gate can MEASURE the contraction
        guard's effect rather than assert it.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        self.scan.run(guard)
        kernel = self._kernel if self._kernel is not None else cyl_complex_pml_curl_step
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            CupyPointer(_word_view(self.scan.out)), *self._imr_rows,
            *self._coefficients,
            self._cb, self._courant,
            nx, ny, nz, self.n_elem, self.dtdx, float(self.m),
            BACKWARD=self.backward,
            BCZ=self.bcz,
            M_CLASS=self.m_class,
            ZERO_ROWS=self.zero_rows,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"CylindricalComplexCurlPlan({self.sub_step}, shape={self.shape}, "
                f"m_class={self.m_class}, zero_rows={self.zero_rows}, "
                f"bcz={self.bcz}, block={self.block})")


def plan_cylindrical_complex_curl(fields: Any, pml: Any, sub_step: str,
                                  block: Optional[int] = None,
                                  num_warps: Optional[int] = None, chunk_layout: Any = None
                                  ) -> Optional[CylindricalComplexCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    ``chunk_layout`` is the run's division (``None`` for the one-rank default); the chunk
    record and the scan's starts line both follow it. The Triton import stays BELOW the
    predicate, so a NumPy host can plan (to ``None``) without the optional dependency.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_complex_curl_coverage(fields, pml, sub_step).covered:
        return None

    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalComplexCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, float(grid.courant), int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        grid.xp, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        scan_starts=radial_scan_starts(fields, pml, sub_step, chunk_layout),
    )


def plan_cylindrical_complex_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        courant: float, m: int, accurate_fields_near_cylorigin: bool, bcz: int, xp: Any, *,
        chunk_bounds: Any, scan_starts: Any,
        block: Optional[int] = None, kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = None) -> CylindricalComplexCurlPlan:
    """Build a plan from bare device arrays -- the gate's and benchmark's route.

    ``arrays`` is keyed by component name (complex64 volumes), ``flat`` by
    ``kms_x``/``sinv_x``/... on the sub-lattice the caller ALREADY SELECTED (half-integer
    for B, integer for D). ``chunk_bounds`` and ``scan_starts`` are REQUIRED. No
    predicate runs: the caller is a harness that constructed the configuration
    deliberately, and ``kernel=`` / ``scan_kernel=`` carry the mutation overrides --
    dropping one silently disarms every mutation leg that targets it.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return CylindricalComplexCurlPlan(
        sub_step, shape, dtdx, courant, m, accurate_fields_near_cylorigin, bcz,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        scan_kernel=scan_kernel, num_warps=num_warps,
    )


def plan_cylindrical_complex_constitutive(fields: Any, pml: Any, side: str,
                                          block: Optional[int] = None,
                                          num_warps: Optional[int] = None, chunk_layout: Any = None
                                          ) -> Optional[ComplexConstitutivePlan]:
    """Build the complex constitutive plan for the admitted Dcyl slice.

    The compiled arithmetic is ``complex_fields.bloch_constitutive_step``; complex
    cylindrical changes neither its per-part forms nor its coefficient layout. This
    wrapper owns the distinct predicate, so the complex family cannot silently broaden
    its blanket Dcyl refusal. ``chunk_layout`` is the run's division, and the H side
    carries MEEP's start of H (``h_state``), as the complex engine route does.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not cylindrical_complex_constitutive_coverage(fields, pml, side).covered:
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )
