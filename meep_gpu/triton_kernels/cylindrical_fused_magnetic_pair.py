"""The CYLINDRICAL COMPLEX magnetic seam in one launch: Dcyl ``step_B`` welded into
``update_H`` under complex64 storage, at every m.

The complex-storage Dcyl arm's B->H product. The two halves are the shipped bodies --
:func:`.cylindrical_complex.cyl_complex_pml_curl_step` on ``step_B`` (per part: the
presence-selected curls, Bz's ``dtdx`` times the radial difference, the ownership mask,
the |m| = 1 axis-row increment, the chunk-keyed PML forms, the i*m/r ladder and the
per-|m| axis rules) and :mod:`.complex_fused_magnetic_pair`'s constitutive half
(:func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 0`` arm per part, keyed on
the chunk record, with MEEP's start of H) -- welded by the one substitution
:func:`.kernels.fused_curl_constitutive_B` makes on the real path: the constitutive
half's source load becomes the register pair the curl half computed.

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY of one of those bodies; the gate's
transcription leg checks it. No product here has a complex operand, so the kernel calls
no complex-multiply helper (:data:`MULTIPLY_HELPERS` is empty).

THE RADIAL SCAN IS UPSTREAM OF THE SEAM: the plan launches
:func:`.cylindrical_triton.cyl_radial_difference` (per part), then this kernel.

THE FIVE PASSES IN THE SEAM. ``step_B -> magnetic sources -> fill_symmetry_bc_B ->
zero_metal_B -> fill_folded_far_ghosts_B -> update_H``: magnetic sources are REFUSED BY
NAME (an undeclared source list is a refusal); the two symmetry fills cannot run on a
Dcyl cell (``Grid`` refuses a mirror there and the predicate refuses one on every axis);
``zero_metal_B`` is carried inline on the registers, and on an admitted Dcyl grid only
``ZM_Z`` can be true (:data:`FORBIDDEN_WALL_AXES`).

DEVICE STATUS. The weld released on 2026-08-20 measured the previous forms and does not
license this text; certification is owed at the combined device round.

Import contract: importable WITHOUT Triton -- the predicates and the plan builder (to
``None``) answer on a host without it.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _word_view,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
from .cylindrical_complex import (
    IMR_TERMS,
    PREFIX,
    cylindrical_complex_constitutive_coverage,
    cylindrical_complex_curl_coverage,
    imr_rinv_row,
    m_class,
    zero_rows,
)
from .cylindrical_triton import RadialScan, courant_constants, radial_scan_starts
from .launch import SUB_STEPS, CupyPointer, _flat
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

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3281-3288). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills cannot even
#: exist on a Dcyl cell — and the inert two are listed anyway: what a launch
#: REPLACES is what the driver would otherwise have called.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The complex-multiply device functions this kernel is allowed to call: none. Every
#: product has a real operand and is formed per part; a helper call appearing here
#: would be a complex operand, which the test scans for.
MULTIPLY_HELPERS: Tuple[str, ...] = ()

#: The axes ``zero_metal_axes`` may NEVER report walled on an admitted Dcyl grid.
#: See the module docstring — a consequence of ``Grid``'s ``metallic_axes``
#: construction, checked rather than assumed, because ``ZM_X`` would compile a clear
#: on the r = 0 row that the per-|m| axis rules own.
FORBIDDEN_WALL_AXES: Tuple[int, ...] = (0, 1)

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FORBIDDEN_WALL_AXES",
    "MULTIPLY_HELPERS", "REPLACES",
    "CylindricalFusedMagneticPairPlan",
    "cyl_complex_fused_curl_constitutive_B",
    "cyl_complex_fused_curl_constitutive_B_kernel",
    "cylindrical_fused_magnetic_pair_coverage",
    "plan_cylindrical_fused_magnetic_pair",
    "plan_cylindrical_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The forms, the record layout and the family's constexprs, read by the kernel.
    # Imported only when Triton is: forms.py and kernels.py import it at module scope.
    from .forms import (axis_present, constitutive_form, curl_form, curl_stencil,
                        record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE
    from .cylindrical_complex import (COURANT_FOUR, COURANT_MINUS, COURANT_PLUS,
                                      M_MANY, M_ONE, M_ZERO, METALLIC, PERIODIC)

    @triton.jit(do_not_specialize=["start_h"])
    def cyl_complex_fused_curl_constitutive_B(
        f0, f1, f2,                   # curl targets:     Bx,By,Bz (c8 as words)
        u0, u1, u2,                   # curl auxiliaries: fu_B*    (c8 as words)
        g0, g1, g2,                   # curl sources:     Ex,Ey,Ez (c8 as words)
        rad,                          # the radial difference (c8 as words)
        q0, q2,                       # i*m/r rinv rows for targets 0 and 2 (f32, nr long)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                   # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, INTEGER lattice
        cb,                           # chunk record: both halves' membership and presence
        courant,                      # float64 [4*Courant, Courant, -Courant]
        nx, ny, nz, n_elem, dtdx,     # n_elem = COMPLEX cells; dtdx pre-rounded
        mf,                           # float32(m): the |m| = 1 B-side partner's factor
        start_h,                      # the constitutive half's start of H
        BACKWARD: tl.constexpr,       # bound to 0 by every builder
        BCZ: tl.constexpr,            # PERIODIC or METALLIC; r is CYL_AXIS, phi periodic
        M_CLASS: tl.constexpr,        # M_ZERO, M_ONE or M_MANY
        ZERO_ROWS: tl.constexpr,      # |m| or 1 under M_MANY; unused otherwise
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Dcyl complex ``step_B`` + ``zero_metal_B`` + complex ``update_H``, one launch.

        The curl half is :func:`.cylindrical_complex.cyl_complex_pml_curl_step`'s body up
        to its stores; the seam and the constitutive half are
        :mod:`.complex_fused_magnetic_pair`'s. ``BACKWARD`` is carried and bound to 0 so
        the curl half stays a verbatim copy.
        """
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

        # The complex pair's names for the walled planes: on a Dcyl grid axis 0 IS r,
        # so `at_x` is the curl half's `at_r`; phi's plane is spelled out.
        at_x, at_y = at_r, j == 0
        # --- the seam: stepping.zero_metal_B, between the two sub-steps ---------
        # Applied to the REGISTERS, before both the store and the constitutive read,
        # so the two consumers see the one value the array path leaves in B. BOTH
        # parts: `array[_face(axis, 0)] = 0` on a complex64 volume writes complex
        # zero. On the B side the component on the walled axis is cleared (its Yee
        # shift there is 0).
        if ZM_X:
            v0_re = tl.where(at_x, 0.0, v0_re)
            v0_im = tl.where(at_x, 0.0, v0_im)
        if ZM_Y:
            v1_re = tl.where(at_y, 0.0, v1_re)
            v1_im = tl.where(at_y, 0.0, v1_im)
        if ZM_Z:
            v2_re = tl.where(at_z, 0.0, v2_re)
            v2_im = tl.where(at_z, 0.0, v2_im)

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

        # ==================== the constitutive half ===========================
        # complex_fields.bloch_constitutive_step's SCALE=0 arm, per part, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half just
        # produced -- the substitution kernels.fused_curl_constitutive_B
        # makes on the real path. The record's dsigw line on the integer lattice
        # picks the form; f_w is read and written on the accumulation's cells only,
        # `prev` before the store.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, 0 + 0, i)
        dw1 = record_in(cb, 0 + 4, j)
        dw2 = record_in(cb, 0 + 8, k)
        # MEEP's start of H: the registers v0..v2 ARE the B words MEEP's first
        # update_eh copies from.
        start = start_h != 0

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        src_re = v0_re
        src_im = v0_im
        tl.store(w0 + 2 * idx, src_re, mask=live & dw0)
        tl.store(w0 + 2 * idx + 1, src_im, mask=live & dw0)
        a_re = tl.load(h0 + 2 * idx, mask=live & dw0, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h0 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0),
                 mask=live)
        tl.store(h0 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0), mask=live)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=live & dw1, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        src_re = v1_re
        src_im = v1_im
        tl.store(w1 + 2 * idx, src_re, mask=live & dw1)
        tl.store(w1 + 2 * idx + 1, src_im, mask=live & dw1)
        a_re = tl.load(h1 + 2 * idx, mask=live & dw1, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h1 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1),
                 mask=live)
        tl.store(h1 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1), mask=live)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=live & dw2, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        src_re = v2_re
        src_im = v2_im
        tl.store(w2 + 2 * idx, src_re, mask=live & dw2)
        tl.store(w2 + 2 * idx + 1, src_im, mask=live & dw2)
        a_re = tl.load(h2 + 2 * idx, mask=live & dw2, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h2 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2),
                 mask=live)
        tl.store(h2 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2), mask=live)

else:  # pragma: no cover - laptop path
    cyl_complex_fused_curl_constitutive_B = None  # type: ignore[assignment]


def cyl_complex_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if cyl_complex_fused_curl_constitutive_B is None:
        raise ImportError(
            "the cylindrical complex fused magnetic B/H kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return cyl_complex_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None) -> Coverage:
    """May ONE launch span Dcyl ``step_B`` -> wall -> complex ``update_H``?

    A conjunction of the two halves' own predicates plus the seam clauses. Nothing
    is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it — the same
    construction :func:`.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage` use. The Dcyl geometry clauses, the
    storage clause (complex64 at every m; the real m = 0 run is cylindrical_triton's),
    and the nr >= 2 clause are all INHERITED from the halves and not restated.
    """
    reasons: List[str] = []

    curl = cylindrical_complex_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    constitutive = cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E seam and does not disqualify this
    # pair. MEASURED COST ON THE CORPUS: ZERO — all sixteen cylindrical rows
    # declare electric sources only, which is what makes this the only large cell
    # with no source attrition. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not
    # hold the source list, so a predicate that inferred "no magnetic source" from
    # not being told would be the over-covering refusal this clause prevents.
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

    # THE TWO SYMMETRY PASSES. `fill_symmetry_bc_B` (driver.py:3285) and
    # `fill_folded_far_ghosts_B` (:3287) both return at their first line unless
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566); `Grid` refuses a
    # mirror on a Dcyl cell outright (grid.py:648-653) and both halves restate
    # that. RE-CHECKED HERE ANYWAY, not inferred: this module's coverage may not be
    # read off another module's guard, and the cost is one attribute read.
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
    answerable = True
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            answerable = False
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    # THE WALL TABLE MUST BE THE Dcyl ONE. `Grid` cannot report r or phi walled on
    # a cylindrical cell (module docstring), and ZM_X would compile a clear on the
    # r = 0 row that the per-|m| axis rules own — a zero the array path never
    # writes there. Refused by name rather than compiled.
    if answerable:
        walls = zero_metal_axes(grid)
        for axis in FORBIDDEN_WALL_AXES:
            if walls[axis]:
                reasons.append(
                    f"zero_metal_axes reports axis {axis} walled on a grid this "
                    f"family admits; on a Dcyl cell neither r nor phi can be "
                    f"(grid.py:512-514), and the r = 0 clear ZM_X would compile is "
                    f"a row the per-|m| axis rules own")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedMagneticPairPlan:
    """The radial scan, then ONE launch for the driver's call sites in the seam.

    The complex volumes are bound as float32 WORD VIEWS at plan time and ``n_elem`` stays
    the COMPLEX cell count, as :class:`.cylindrical_complex.CylindricalComplexCurlPlan`
    does. Allocation-free per step: the scan's output and bindings, the two i*m/r rows,
    the chunk record and the Courant words are bound once. ``chunk_bounds`` is REQUIRED
    and ONE record serves both halves, each reading its own sub-lattice.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "courant", "m", "backward", "bcz", "m_class",
        "zero_rows", "zero_metal", "block", "num_warps", "chunk_bounds",
        "scan", "_targets", "_aux", "_sources", "_curl_coefficients", "_imr_rows",
        "_h_targets", "_h_aux", "_constitutive_coefficients", "_cb",
        "_courant", "_h_state", "_grid", "_kernel",
    )

    #: The driver call sites one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, courant: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
                 block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, constitutive_coefficients, xp: Any, *,
                 chunk_bounds: Any, scan_starts: Any, kernel: Any = None,
                 scan_kernel: Any = None, num_warps: Optional[int] = 1, h_state: Any = None) -> None:
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        spec = SUB_STEPS[CURL_SUB_STEP]
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "CylindricalFusedMagneticPairPlan")
        # float(): the array path scales each part by a Python float rounded to the
        # part's float32, and Triton types a Python float argument as fp32.
        self.dtdx = float(dtdx)
        self.courant = float(courant)
        self._courant = CupyPointer(xp.asarray(courant_constants(self.courant)))
        self.m = int(m)
        self.backward = BACKWARD
        self.bcz = int(bcz)
        self.m_class = m_class(m)
        self.zero_rows = zero_rows(m, accurate_fields_near_cylorigin)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        source_map = dict(zip(spec["sources"], sources))
        self.scan = RadialScan(CURL_SUB_STEP, source_map[PREFIX[CURL_SUB_STEP]["component"]],
                               scan_starts, xp, block=self.block, kernel=scan_kernel)
        self._curl_coefficients = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        self._imr_rows = tuple(
            CupyPointer(xp.asarray(imr_rinv_row(spec["targets"][index], sign, m,
                                                courant, self.shape[0])))
            for index, _register, sign in IMR_TERMS[CURL_SUB_STEP])
        self._h_targets = tuple(CupyPointer(_word_view(a)) for a in h_targets)
        self._h_aux = tuple(CupyPointer(_word_view(a)) for a in h_aux)
        self._constitutive_coefficients = tuple(
            CupyPointer(_flat(a)) for a in constitutive_coefficients)
        self._h_state = h_state
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32, so
        # the complex cell count must leave room for the doubling. Refused here as well
        # because the from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs. Dropping it is a
        # DISARMING -- every mutation leg would then launch the shipped kernel.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """The scan, then ONE launch, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        self.scan.run(guard)
        kernel = (self._kernel if self._kernel is not None
                  else cyl_complex_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        state = self._h_state
        start = 0 if state is None or state.h_separated else 1
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            CupyPointer(_word_view(self.scan.out)), *self._imr_rows,
            *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._constitutive_coefficients,
            self._cb, self._courant,
            nx, ny, nz, self.n_elem, self.dtdx, float(self.m), start,
            BACKWARD=self.backward,
            BCZ=self.bcz,
            M_CLASS=self.m_class,
            ZERO_ROWS=self.zero_rows,
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def __repr__(self) -> str:
        return (f"CylindricalFusedMagneticPairPlan(shape={self.shape}, "
                f"bcz={self.bcz}, m_class={self.m_class}, "
                f"zero_rows={self.zero_rows}, zero_metal={self.zero_metal}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_cylindrical_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        chunk_layout: Any = None,
        ) -> Optional[CylindricalFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``chunk_layout`` is the run's division (``None`` for the one-rank default); the chunk
    record and the scan's starts line both follow it.
    """
    if not cylindrical_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, float(grid.courant), int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0,
        zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl's sub-lattice is the sub-step's own; the constitutive side's is its
        # own. The kernel never asks which is which, so a swap here is a silent
        # half-cell error in the absorber profile; the gate carries a mutation for it.
        [getattr(pml, f"{stem}_{axis}{''}") for axis in "xyz"
         for stem in ("kps", "kms")],
        grid.xp, kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        scan_starts=radial_scan_starts(fields, pml, CURL_SUB_STEP, chunk_layout),
        h_state=fields,
    )


def plan_cylindrical_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, courant: float, m: int,
        accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
        xp: Any, *, chunk_bounds: Any, scan_starts: Any,
        block: Optional[int] = None, kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalFusedMagneticPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat`` and
    ``constitutive_flat`` are keyed on the sub-lattice the caller ALREADY SELECTED;
    ``chunk_bounds`` and ``scan_starts`` are REQUIRED; ``kernel=`` / ``scan_kernel=``
    carry the mutation overrides.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalFusedMagneticPairPlan(
        shape, dtdx, courant, m, accurate_fields_near_cylorigin, bcz, zero_metal,
        DEFAULT_BLOCK if block is None else block,
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
