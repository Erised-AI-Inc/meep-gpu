"""COMPLEX cylindrical (Dcyl) curl sub-steps, in hand CUDA -- complex64 storage, any m.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid with
complex64 storage (``abs(grid.m) >= 1``, and ``m = 0`` under ``force_complex_fields``)
and an active split-field PML: the whole sub-step, in two launches -- the radial scan
(:mod:`.cylindrical_prefix`, one column-serial kernel, run per stored part) and the curl
kernel below.

A THIRD CURL PAIR beside the real m = 0 one (``cylindrical_kernels``, which refuses
complex storage by name) and the Cartesian complex one (``complex_emitter``, which
refuses Dcyl by name). Both refusals stay.

=============================================================================
THE FORMS ARE THE ARRAY PATH'S (bit_parity_0_10 KERNELS.md section 1), each read from
``stepping`` and transcribed, per stored part
=============================================================================

1. **The certified complex prelude, imported**: ``complex_emitter``'s word-pair
   addressing, its part-by-part products, its ghost reads, ``curl_stencil``
   (``stepping._curl_from_operands`` with the record's partner-presence word; phi is
   invariant, so Bx/Dx take MEEP's one-partner form) and the chunk-form ``pml_apply``
   (``stepping._apply_curl_regions``, keyed on the chunk record ``cb``). No Bloch
   phase reaches this family (``Grid`` refuses a k on r or phi and the predicate one on
   z), so every operand is the stored word and the ghosts are plain.
2. **Bz's and Dz's whole curl is ``dtdx`` times the radial difference** of Ep's / Hp's
   per-chunk sum, ``part_products.scale(dtdx, radial)``, which the scan writes first.
3. **The i*m/r coupling is a PASS OF ITS OWN after the curl** (step_db.cpp:177-294,
   ``stepping._cylindrical_imr_increment`` through ``_apply_folded_increment``): per
   target ``Bx <- Ez (+1)``, ``Bz <- Ex (-1)``, ``Dx <- Hz (-1)``, ``Dz <- Hx (+1)``,
   the increment is the partner's quarter turn by the real-part ``rinv``,
   ``(-(rinv * g.im), rinv * g.re)`` -- ``rinv = the_m / r`` a realnum QUOTIENT built
   on the host the array path's way (:func:`imr_rinv_row`), so the kernel performs no
   division -- masked to ``+0`` on MEEP's unowned cells (the target's own mask), and
   SUBTRACTED through MEEP's additive ladder keyed on the target's dsig and dsigu
   layers: ``df = sinv * inc``; ``f -= df`` outside the dsigu chunks, ``fu -= df; f -=
   sinv_u * df`` inside them (``fold_apply``). Subtracting a masked ``+0`` keeps a stored
   ``-0``; adding one would not. Applied in register order after the target's curl
   form, which is the array path's order on every word the pass reads.
4. **The |m| = 1 axis-row increments REPLACE the masked curl row 0** before the PML
   form (``_cylindrical_axis_increment_B`` / ``_D``): B: ``Bx`` row 0 takes
   ``-realnum(double(-Courant) * double((Ep - Ep_zup) - (m*Ez1.im, -(m*Ez1.re))))`` with
   ``Ez1`` the first off-axis row; D: ``Dy`` row 0 takes
   ``-realnum(double(Courant) * double((Hr - Hr_zdn) - 2*Hz))``. Each is MEEP's double
   product with the double Courant, rounded once (step_db.cpp:377,
   ``stepping._realnum_double_product``; ``double_product``), the bracket in float32
   left to right. The Courant words are ``double`` kernel arguments
   (:func:`axis_courant_words`).
5. **The per-m axis rules** on the stored words after the pass
   (``_cylindrical_axis_zero_B`` / ``_D``): m = 0: ``Bx[0] = 0``; ``Dz[0] +=
   realnum(double(4*Courant) * double(Hp[0]))`` per part, then ``Dy[0] = 0``. |m| = 1:
   ``Dz[0] = 0`` (field only). |m| >= 2: every component of the family and its ``fu``
   held at ``(+0, +0)`` on rows ``[0, zero_rows)``.

The r_to_minus_r near ghost is not implemented and cannot be observed (the targets
taking a shift-down along r have r Yee shift 0 or a radial-difference curl, and the
mask owns row 0) -- the real pair's reason, restated in ``cylindrical_kernels``.

NOT CUPY AT IMPORT: the source is built from the imported prelude, so a host test
imports this module to read the text. ``cupy`` is imported at the compile and launch
sites. The device strings are pure ASCII (NVRTC reads them through the locale), and
``--fmad=false`` is correctness.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

# THE CERTIFIED COMPLEX STRINGS ARE IMPORTED, NOT COPIED.
from .complex_emitter import curl_bounds_block, prelude
from .cylindrical_prefix import radial_difference, radial_scan_device_tables, radial_scan_tables


#: The two entry points. The records that certified their 0.9.2 texts are re-cut in the
#: combined certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = (
    "cyl_step_B_pml_complex",
    "cyl_step_D_pml_complex",
)

#: This file's half of the partition ``step_curl_kernels.py`` maintains.
UNCERTIFIED_KERNELS: Tuple[str, ...] = ()
#: The compile options this module passes NVRTC; a gate substitutes the empty tuple as
#: a control.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane.
_COMPLEX_THREADS = 256

#: The two curl sub-steps, their kernel entry points and whether the side differences
#: DOWN (D) or UP (B).
KERNELS: Dict[str, Tuple[str, bool]] = {
    "step_B": ("cyl_step_B_pml_complex", False),
    "step_D": ("cyl_step_D_pml_complex", True),
}

CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Which Yee sub-lattice each sub-step's coefficient vectors (and chunk-record lines)
#: come from: the B curl the HALF-INTEGER positions, the D curl the INTEGER ones.
HALF_INTEGER: Dict[str, bool] = {"step_B": True, "step_D": False}

#: Which volumes each sub-step writes and reads, in the kernel's argument order
#: (``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``).
CURL_ARRAYS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "step_B": (("Bx", "By", "Bz"), ("Ex", "Ey", "Ez")),
    "step_D": (("Dx", "Dy", "Dz"), ("Hx", "Hy", "Hz")),
}

#: The i*m/r call sites -- ``(target index, target name, sign)`` per sub-step, read off
#: ``stepping._CYLINDRICAL_IMR_B`` / ``_D``. Target 1 takes none on either side; the
#: row is indexed by the TARGET's own radial Yee shift.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "Bx", +1.0), (2, "Bz", -1.0)),
    "step_D": ((0, "Dx", -1.0), (2, "Dz", +1.0)),
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0.
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}

#: ``m_class`` codes, matched by the device text: ``stepping``'s own partition
#: (``m == 0`` / ``abs(m) == 1`` / else).
M_ZERO = 0    # m = 0: Bx[0] = 0 on B; Dz[0] += 4*Courant*Hp[0], Dy[0] = 0 on D
M_ONE = 1     # |m| = 1: axis-row increments on Bx and Dy, plus Dz[0] = 0
M_MANY = 2    # |m| >= 2: near-axis zeroing of all six volumes


# ---------------------------------------------------------------------------
# Host-side constants, computed the array path's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``m_class`` for one azimuthal order -- ``M_ZERO``, ``M_ONE`` or ``M_MANY``."""
    if int(m) == 0:
        return M_ZERO
    return M_ONE if abs(int(m)) == 1 else M_MANY


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows``: ``slice(0, 1)`` on the ACCURATE branch and
    ``slice(0, |m|)`` on MEEP's default; zero below |m| = 2, where neither applies.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_rinv_row(grid: Any, target: str, sign: float, rows: int) -> Any:
    """The i*m/r pass's real-part ``rinv`` per r row, host float32, the array path's way.

    ``stepping._cylindrical_imr_increment``'s row, step for step: ``the_m`` is
    ``realnum(sign * 2.0 * m * Courant)`` (an integer product times the double Courant,
    rounded once), ``here_r`` the target's doubled r coordinate ``2*ir + iyee_r``
    clamped at 1 and cast to realnum, and ``rinv = the_m / here_r`` a realnum QUOTIENT.
    The clamp bites only on row 0 of a shift-0 target, whose increment the mask zeroes.
    Built on the host with NumPy (one correctly rounded division per row) and bound as
    a device row, so the kernel performs no division.
    """
    import numpy  # noqa: PLC0415 - host-side

    from ..fields import IYEE_SHIFTS  # noqa: PLC0415 - NumPy only

    real = numpy.float32
    iyee_r = IYEE_SHIFTS[target][0]
    the_m = real(sign * 2.0 * grid.m * float(grid.courant))
    r_doubled = 2 * numpy.arange(int(rows), dtype=numpy.int64) + iyee_r
    here_r = numpy.maximum(r_doubled, 1).astype(real)
    return numpy.ascontiguousarray(the_m / here_r, dtype=real)


def imr_rows_for(sub_step: str, grid: Any, rows: int) -> Tuple[Any, Any]:
    """The two bound i*m/r rows for one sub-step, in kernel argument order (host).

    GRID INVARIANTS -- Yee shift, radial extent, m, Courant -- built once per frozen
    configuration, as ``stepping``'s own ``scratch.constant`` cache builds them once.
    """
    return tuple(imr_rinv_row(grid, target, sign, rows)
                 for _index, target, sign in IMR_TERMS[sub_step])


def axis_courant_words(sub_step: str, grid: Any) -> Tuple[Any, ...]:
    """The axis rules' double factors, in kernel argument order, as ``np.float64``.

    B: ``-Courant`` (the |m| = 1 Br increment, step_db.cpp:377 with sd = -1). D:
    ``+Courant`` (the |m| = 1 Dp increment, sd = +1) and ``4*Courant`` (the m = 0 on-axis
    Dz rule, :324). Each is the array path's ``_realnum_double_product`` factor, exact in
    double, and travels as a ``double`` argument: a float32 factor is another word
    wherever Courant is not a float32 word.
    """
    import numpy  # noqa: PLC0415 - host-side

    courant = float(grid.courant)
    if sub_step == "step_B":
        return (numpy.float64(-1.0 * courant),)
    if sub_step == "step_D":
        return (numpy.float64(+1.0 * courant), numpy.float64(4.0 * courant))
    raise ValueError(f"sub_step must be one of {list(KERNELS)}, got {sub_step!r}")


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: The three helpers this family adds to the certified prelude.
_CYLINDRICAL_HELPERS = r'''
// stepping._cylindrical_imr_increment, per part: MEEP adds the partner's OTHER part
// times the signed realnum rinv (re += rinv*g.im, im += (-rinv)*g.re); returned
// negated, in curl sign convention, as a quarter turn of the partner
// (part_products.quarter_turn) -- two real products and a negation, never a
// complex product.
__device__ __forceinline__ cf imr_increment(float rinv, cf g) {
    cf o;
    o.re = (rinv * g.im) * -1.0f;
    o.im = rinv * g.re;
    return o;
}

// stepping._apply_folded_increment with no conductivity: MEEP's additive ladder on
// one pass's increment (in curl sign convention, so it is SUBTRACTED), keyed on the
// target's dsig and dsigu layers as its curl keys:
//   df = sinv * inc;  outside the dsigu chunks f -= df;
//   inside them fu -= df and f -= sinv_u * df.
// fu is read and written on the dsigu cells only. Subtracting a masked +0 keeps a
// stored -0, which adding it would not.
__device__ __forceinline__ void fold_apply(
    float* f, float* fu, int idx, cf inc, float sinv, float sinv_u, int dsigu
) {
    cf df = mul_coefficient_left(sinv, inc);
    if (dsigu) cf_store(fu, idx, cf_sub(cf_load(fu, idx), df));
    cf value = dsigu ? cf_sub(cf_load(f, idx), mul_coefficient_left(sinv_u, df))
                     : cf_sub(cf_load(f, idx), df);
    cf_store(f, idx, value);
}

// stepping._realnum_double_product, per part: realnum(double(c) * double(v)), the
// form of MEEP's axis rules where a realnum meets the double Courant
// (step_db.cpp:324, :377): one double product, rounded once.
__device__ __forceinline__ cf double_product(double c, cf v) {
    cf o;
    o.re = __double2float_rn(c * (double)(v.re));
    o.im = __double2float_rn(c * (double)(v.im));
    return o;
}

'''

#: The B side. Bx and By are the complex root's blocks on the Dcyl term table (no
#: phase); Bz's whole curl is dtdx times the radial difference; the |m| = 1 increment
#: replaces Bx's curl row 0; the i*m/r pass follows Bx's and Bz's forms; the axis rules
#: close.
_CYL_STEP_B_TEMPLATE = r'''
extern "C" __global__ void cyl_step_B_pml_complex(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    const float* __restrict__ rad,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    int nx, int ny, int nz, float dtdx, float m_word, double minus_courant,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int m_class, int zero_rows,
    const int* __restrict__ cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Strides in COMPLEX CELLS. cf_load does the word doubling.
    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

__BOUNDS__    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> cell 0 on x.
    {
        cf f_1 = cf_load(Ez, idx);
        cf sf = cshift_up(Ez, idx, j, ny, sy, bc_y);
        cf f_2 = cf_load(Ey, idx);
        cf ss = cshift_up(Ey, idx, k, nz, sz, bc_z);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_y, pr_z);
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        // stepping._cylindrical_axis_increment_B, |m| = 1: the masked axis row is
        // REPLACED by -realnum(double(-Courant) * double(bracket)), the bracket
        // (Ep - Ep_zup) - (m*Ez1.im, -(m*Ez1.re)) in float32 left to right, Ez1 the
        // first off-axis row (the predicate requires two radial rows).
        if (m_class == 1 && i == 0) {
            cf e1 = cf_load(Ez, idx + sx);
            cf turn;
            turn.re = m_word * e1.im;
            turn.im = (m_word * e1.re) * -1.0f;
            curl = cf_neg(double_product(minus_courant, cf_sub(cf_sub(f_2, ss), turn)));
        }
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], in_h_z);
        // The i*m/r pass, Bx <- +Ez: its own increment after the curl, masked on the
        // axis row (the target's own mask), through the ladder.
        if (m_class != 0) {
            cf inc = imr_increment(imr0[i], f_1);
            if (bc_x == BC_METALLIC && i == 0) inc = cf_zero();
            fold_apply(Bx, fu_Bx, idx, inc, sinv_y[j], sinv_z[k], in_h_z);
        }
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> cell 0 on y.
    {
        cf f_1 = cf_load(Ex, idx);
        cf sf = cshift_up(Ex, idx, k, nz, sz, bc_z);
        cf f_2 = cf_load(Ez, idx);
        cf ss = cshift_up(Ez, idx, i, nx, sx, bc_x);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_z, pr_x);
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], in_h_x);
    }

    // Bz: dsig=x, dsigu=y; iyee=(1,1,0) -> cell 0 on z. The whole curl is dtdx times
    // the radial difference of Ep's per-chunk sum; then the i*m/r pass, Bz <- -Ex.
    {
        cf curl = mul_coefficient_left(dtdx, cf_load(rad, idx));
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j], in_h_y);
        if (m_class != 0) {
            cf inc = imr_increment(imr2[i], cf_load(Ex, idx));
            if (bc_z == BC_METALLIC && k == 0) inc = cf_zero();
            fold_apply(Bz, fu_Bz, idx, inc, sinv_x[i], sinv_y[j], in_h_y);
        }
    }

    // stepping._cylindrical_axis_zero_B on the stored words, after the pass. |m| = 1:
    // nothing. |m| >= 2: the family and its fu at (+0, +0) on rows [0, zero_rows).
    // m = 0: Br on the axis row, the field only.
    if (m_class == 2 && i < zero_rows) {
        cf_store(Bx, idx, cf_zero());
        cf_store(By, idx, cf_zero());
        cf_store(Bz, idx, cf_zero());
        cf_store(fu_Bx, idx, cf_zero());
        cf_store(fu_By, idx, cf_zero());
        cf_store(fu_Bz, idx, cf_zero());
    }
    if (m_class == 0 && i == 0) {
        cf_store(Bx, idx, cf_zero());
    }
}
'''

#: The D side. Dx and Dy are the complex root's blocks; Dz's whole curl is dtdx times
#: the backward radial difference of Hp's per-chunk sum; the |m| = 1 increment replaces
#: Dy's curl row 0; the i*m/r pass follows Dx's and Dz's forms; the axis rules close.
_CYL_STEP_D_TEMPLATE = r'''
extern "C" __global__ void cyl_step_D_pml_complex(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ rad,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    int nx, int ny, int nz, float dtdx, double plus_courant, double four_courant,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int m_class, int zero_rows,
    const int* __restrict__ cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

__BOUNDS__    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0) -> cell 0 on y, z.
    {
        cf f_1 = cf_load(Hz, idx);
        cf sf = cshift_dn(Hz, idx, j, ny, sy, bc_y);
        cf f_2 = cf_load(Hy, idx);
        cf ss = cshift_dn(Hy, idx, k, nz, sz, bc_z);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_y, pr_z);
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], in_i_z);
        // The i*m/r pass, Dx <- -Hz.
        if (m_class != 0) {
            cf inc = imr_increment(imr0[i], f_1);
            if (bc_y == BC_METALLIC && j == 0) inc = cf_zero();
            if (bc_z == BC_METALLIC && k == 0) inc = cf_zero();
            fold_apply(Dx, fu_Dx, idx, inc, sinv_y[j], sinv_z[k], in_i_z);
        }
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0) -> cell 0 on x, z.
    {
        cf f_1 = cf_load(Hx, idx);
        cf sf = cshift_dn(Hx, idx, k, nz, sz, bc_z);
        cf f_2 = cf_load(Hz, idx);
        cf ss = cshift_dn(Hz, idx, i, nx, sx, bc_x);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_z, pr_x);
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        // stepping._cylindrical_axis_increment_D, |m| = 1: the masked axis row is
        // REPLACED by -realnum(double(Courant) * double((Hr - Hr_zdn) - 2*Hz)).
        if (m_class == 1 && i == 0) {
            cf bracket = cf_sub(cf_sub(f_1, sf), mul_coefficient_left(2.0f, f_2));
            curl = cf_neg(double_product(plus_courant, bracket));
        }
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], in_i_x);
    }

    // Dz: dsig=x, dsigu=y; iyee=(0,0,1) -> cell 0 on x, y. The whole curl is dtdx times
    // the backward radial difference; then the i*m/r pass, Dz <- +Hx.
    {
        cf curl = mul_coefficient_left(dtdx, cf_load(rad, idx));
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j], in_i_y);
        if (m_class != 0) {
            cf inc = imr_increment(imr2[i], cf_load(Hx, idx));
            if (bc_x == BC_METALLIC && i == 0) inc = cf_zero();
            if (bc_y == BC_METALLIC && j == 0) inc = cf_zero();
            fold_apply(Dz, fu_Dz, idx, inc, sinv_x[i], sinv_y[j], in_i_y);
        }
    }

    // stepping._cylindrical_axis_zero_D on the stored words, after the pass. |m| = 1:
    // Dz on the axis row, the field only. |m| >= 2: the family and its fu at (+0, +0)
    // on rows [0, zero_rows). m = 0: Dz on the axis takes realnum(double(4*Courant) *
    // double(Hp)) per part, added in float32 (Hy is the stored Hp), then Dp is +0.
    if (m_class == 1 && i == 0) {
        cf_store(Dz, idx, cf_zero());
    }
    if (m_class == 2 && i < zero_rows) {
        cf_store(Dx, idx, cf_zero());
        cf_store(Dy, idx, cf_zero());
        cf_store(Dz, idx, cf_zero());
        cf_store(fu_Dx, idx, cf_zero());
        cf_store(fu_Dy, idx, cf_zero());
        cf_store(fu_Dz, idx, cf_zero());
    }
    if (m_class == 0 && i == 0) {
        cf_store(Dz, idx, cf_add(cf_load(Dz, idx), double_product(four_courant, cf_load(Hy, idx))));
        cf_store(Dy, idx, cf_zero());
    }
}
'''

_TEMPLATES: Dict[str, str] = {"step_B": _CYL_STEP_B_TEMPLATE,
                              "step_D": _CYL_STEP_D_TEMPLATE}


def cylindrical_complex_source(sub_step: str) -> str:
    """The device source for one sub-step.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. The prelude is the certified complex
    family's (``complex_emitter.prelude``) and the chunk-record block its curl block for
    the side's lattice; only :data:`_CYLINDRICAL_HELPERS` and the per-side template are
    this family's own text.
    """
    if sub_step not in _TEMPLATES:
        raise ValueError(
            f"sub_step must be one of {sorted(_TEMPLATES)}, got {sub_step!r}")
    body = _TEMPLATES[sub_step].replace("__BOUNDS__", curl_bounds_block(sub_step))
    source = prelude() + _CYLINDRICAL_HELPERS + body
    # PURE ASCII IS A COMPILE REQUIREMENT: a mutation that inserts a non-ASCII
    # character is refused at emission, not at NVRTC three frames away.
    source.encode("ascii")
    return source


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered."""
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for sub_step in sorted(_TEMPLATES):
        digest.update(sub_step.encode("ascii"))
        digest.update(cylindrical_complex_source(sub_step).encode("ascii"))
    return digest.hexdigest()


def shipped_kernel_names(source: str):
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


# ---------------------------------------------------------------------------
# Compilation -- the CuPy half
# ---------------------------------------------------------------------------

def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache  # noqa: PLC0415

    return clear_kernel_cache()


def _get_kernel(sub_step: str):
    """Compile one sub-step, memoized on (name, options, policy, source).

    The source is emitted per call: it is part of the memo key, and a gate mutates this
    family by monkeypatching :func:`cylindrical_complex_source`.
    """
    import cupy as cp  # noqa: PLC0415 - see the module docstring on the lazy import

    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    name = KERNELS[sub_step][0]
    code = cylindrical_complex_source(sub_step)
    key = kernel_cache_key(f"cyl_complex_{sub_step}", True, _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# ---------------------------------------------------------------------------
# The bindings
# ---------------------------------------------------------------------------

def word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume -- the pointer the kernel gets.

    Refuses anything that is not a C-contiguous complex64 array: a strided view would be
    read in the wrong order by the flat word index.
    """
    import cupy as cp  # noqa: PLC0415

    if array is None:
        raise ValueError("expected a complex64 array, got None")
    if array.dtype != cp.complex64:
        raise ValueError(f"expected a complex64 array, got dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            "complex array is not C-contiguous; its float32 word view would not be "
            "either, and the kernel indexes it as a flat word array")
    return array.view(cp.float32)


def cylindrical_complex_curl_tables(pml: 'PML', half_integer: bool) -> Dict[str, Any]:
    """One sub-step's six kms/sinv coefficient vectors, as cached device views.

    THE COEFFICIENTS STAY FLOAT32 UNDER COMPLEX STORAGE: each meets a part as one real
    product (``part_products.scale``).
    """
    import cupy as cp  # noqa: PLC0415

    suffix = "_h" if half_integer else ""
    tables: Dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        for stem in ("kms", "sinv"):
            attribute = f"{stem}_{axis}{suffix}"
            vector = getattr(pml, attribute, None)
            if vector is None:
                raise ValueError(f"pml.{attribute} is missing")
            flat = vector.reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the complex cylindrical PML kernels "
                    f"index float32 coefficient vectors, one real product per part")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector")
            tables[f"{stem}_{axis}"] = flat
    return tables


def cylindrical_complex_boundary_codes(grid) -> Tuple[int, int, int]:
    """This grid's three kernel boundary codes (``'axis'`` -> ``BC_METALLIC`` on r only,
    the real pair's mapping)."""
    from .coverage import real_pml_boundary_kinds  # noqa: PLC0415
    from .cylindrical_coverage import cylindrical_boundary_codes  # noqa: PLC0415

    return tuple(int(code) for code in
                 cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid))))


def chunk_record_argument(chunk_bounds: Any) -> Any:
    """The ``cb`` argument, checked: one contiguous ``int32[25]`` vector on the device.

    The same contract as ``step_curl_kernels.chunk_record_argument``, stated without
    importing that CuPy module: there is no default, because a kernel launched without
    the run's division would step another division's PML forms.
    """
    from ..chunk_map import RECORD_WORDS  # noqa: PLC0415 - NumPy only

    if chunk_bounds is None:
        raise ValueError("the cylindrical complex kernels take the run's chunk record "
                         "(chunk_map.chunk_bounds_record) and were handed none")
    if (str(getattr(chunk_bounds, "dtype", "")) != "int32"
            or tuple(getattr(chunk_bounds, "shape", ())) != (RECORD_WORDS,)
            or not chunk_bounds.flags.c_contiguous):
        raise ValueError(
            f"the chunk record must be one contiguous int32[{RECORD_WORDS}] device vector, "
            f"got {getattr(chunk_bounds, 'dtype', type(chunk_bounds).__name__)} of shape "
            f"{tuple(getattr(chunk_bounds, 'shape', ()))}")
    return chunk_bounds


def device_rows(rows: Sequence[Any], xp: Any) -> Tuple[Any, ...]:
    """Host float32 rows moved onto ``xp``'s device, contiguous."""
    return tuple(xp.ascontiguousarray(xp.asarray(row, dtype=xp.float32)) for row in rows)


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def curl_arguments(sub_step: str, fields: 'Fields', radial: Any, imr_rows: Sequence[Any],
                   dtdx: float, m: int, courant_words: Sequence[Any],
                   tables: Dict[str, Any], boundary_codes: Sequence[Any],
                   m_class_code: int, zero_rows_count: int, chunk_bounds: Any) -> Tuple:
    """The kernel's argument tuple, in signature order, for one sub-step."""
    import numpy as np  # noqa: PLC0415

    targets, sources = CURL_ARRAYS[sub_step]
    shape = tuple(int(n) for n in getattr(fields, targets[0]).shape)
    if tuple(int(n) for n in radial.shape) != shape:
        raise ValueError(f"the radial difference is {tuple(radial.shape)}, the target "
                         f"{shape}; the curl reads it at the target's index")
    arguments = [word_view(getattr(fields, name)) for name in targets]
    arguments += [word_view(getattr(fields, "fu_" + name)) for name in targets]
    arguments += [word_view(getattr(fields, name)) for name in sources]
    arguments += [word_view(radial)]
    arguments += list(imr_rows)
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                  np.float32(dtdx)]
    if sub_step == "step_B":
        arguments += [np.float32(float(m))]
    arguments += [np.float64(word) for word in courant_words]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [np.int32(code) for code in boundary_codes]
    arguments += [np.int32(m_class_code), np.int32(zero_rows_count)]
    arguments += [chunk_bounds]
    return tuple(arguments)


def step_cylindrical_complex(sub_step: str, fields: 'Fields',
                             grid=None, pml: 'PML' = None, *,
                             chunk_bounds: Any,
                             tables: Dict[str, Any] = None,
                             boundary_codes: Sequence[Any] = None,
                             imr_rows: Sequence[Any] = None,
                             radial: Any = None,
                             scan: Dict[str, Any] = None,
                             scratch: Any = None,
                             kernel: Any = None) -> None:
    """``stepping.step_B`` / ``step_D`` for a complex Dcyl grid: the scan, then one launch.

    ``grid`` and ``pml`` are required: every derived quantity (the sub-lattice, the
    boundary codes, the m class, the rows, the Courant words, the scan's chunk starts)
    is derived HERE by the functions the predicate and the array path ask.
    ``chunk_bounds`` is the run's chunk record on the device (required, keyword-only:
    a launch handed another division would step another division's forms). The
    overrides (``tables``, ``boundary_codes``, ``imr_rows``, ``radial``, ``kernel``) are
    a gate's doors for deliberately wrong inputs.
    """
    if sub_step not in KERNELS:
        raise ValueError(f"sub_step must be one of {sorted(KERNELS)}, got {sub_step!r}")
    if grid is None or pml is None:
        raise ValueError("pass the grid and the layer: every quantity this launch binds "
                         "is derived from them")
    xp = grid.xp
    targets, _sources = CURL_ARRAYS[sub_step]
    rows = int(getattr(fields, targets[0]).shape[0])
    if tables is None:
        tables = cylindrical_complex_curl_tables(pml, HALF_INTEGER[sub_step])
    if boundary_codes is None:
        boundary_codes = cylindrical_complex_boundary_codes(grid)
    if imr_rows is None:
        imr_rows = device_rows(imr_rows_for(sub_step, grid, rows), xp)
    if radial is None:
        if scan is None:
            scan = radial_scan_device_tables(radial_scan_tables(fields, pml, sub_step), xp)
        radial = radial_difference(fields, pml, sub_step, tables=scan, scratch=scratch)
    m = int(grid.m)
    arguments = curl_arguments(
        sub_step, fields, radial, imr_rows, float(grid.dt / grid.dx), m,
        axis_courant_words(sub_step, grid), tables, boundary_codes, m_class(m),
        zero_rows(m, bool(grid.accurate_fields_near_cylorigin)),
        chunk_record_argument(chunk_bounds))
    cells = int(getattr(fields, targets[0]).size)
    blocks = (cells + _COMPLEX_THREADS - 1) // _COMPLEX_THREADS
    (kernel or _get_kernel(sub_step))((blocks,), (_COMPLEX_THREADS,), arguments)


__all__ = [
    "AXIS_INCREMENT_TARGET", "CERTIFIED_KERNELS", "CURL_ARRAYS", "CURL_SUB_STEPS",
    "chunk_record_argument",
    "HALF_INTEGER", "IMR_TERMS", "KERNELS", "M_MANY",
    "M_ONE", "M_ZERO", "UNCERTIFIED_KERNELS", "axis_courant_words", "corpus_digest",
    "curl_arguments", "cylindrical_complex_boundary_codes",
    "cylindrical_complex_curl_tables", "cylindrical_complex_source", "device_rows",
    "imr_rinv_row", "imr_rows_for", "m_class", "shipped_kernel_names",
    "step_cylindrical_complex", "word_view", "zero_rows",
]
