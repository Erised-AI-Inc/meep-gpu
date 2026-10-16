"""CYLINDRICAL (Dcyl) curl sub-steps at m = 0, in hand CUDA -- real float32 storage.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid with
``grid.m == 0``, real storage and an active split-field PML: the whole sub-step, in two
launches -- the radial scan (:mod:`.cylindrical_prefix`, one column-serial kernel) and
the curl kernel below.

These are a SECOND PAIR beside the Cartesian ``step_B_pml_real`` / ``step_D_pml_real``,
with their own signature and their own predicate (``cylindrical_coverage``): the
Cartesian pair refuses cylindrical by name, correctly, because it has no radial
difference, no axis tail and no double ``4*Courant``.

=============================================================================
THE FORMS ARE THE ARRAY PATH'S (bit_parity_0_10 KERNELS.md section 1, MEEP 1.33.0 single
precision without contraction), each read from ``stepping`` and transcribed
=============================================================================

1. **The curl stencil and the PML forms are the root's**, reached through the SAME
   ``step_curl_kernels._REAL_PML_PRELUDE`` (imported, never copied): ``curl_stencil``
   is ``stepping._curl_from_operands`` with the record's partner-presence word (phi is
   invariant on a Dcyl grid, so Bx and Dx take MEEP's one-partner form
   ``(-dtdx) * (g2s - g2)`` and By / Dy the left-to-right bracket), and ``pml_apply``
   is ``stepping._apply_curl_regions``: the split-field recurrence on the cells whose
   chunk carries the component's dsigu layer, the dsig-only form elsewhere, ``fu``
   read and written on the recurrence's cells only. The chunk record ``cb``
   (``chunk_map.chunk_bounds_record``) is decoded once after the index decode, the
   literal block ``bounds_block.bounds_block`` emits for the curl's entries.
2. **Bz's and Dz's WHOLE curl is ``dtdx`` times the radial difference** ``rad``:
   their phi partner is the derivative MEEP drops on a Dcyl grid (step_db.cpp:86-92),
   and the r derivative is the difference of MEEP's per-chunk ``f_rderiv_int``
   (``stepping.cylindrical_rderiv_difference``), which the scan kernel writes into a
   target-shaped volume before this launch. ``dtdx * rad`` is
   ``part_products.scale(dtdx, radial)``: one float32 product.
3. **The ownership mask** (``stepping._mask_non_owned_cells``): with the r axis
   compiled METALLIC, ``if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;`` masks
   exactly the axis row ``is_axis`` masks (MEEP's ``little_owned_corner0`` excludes
   r = 0, which the per-m rules own), and the z wall rows are the certified mask.
4. **The m = 0 axis rules** (``stepping._cylindrical_axis_zero_B`` / ``_D``), applied
   to the updated words after the recurrence:
   ``Br[0] = 0``; ``Dz[0] += realnum(double(4*Courant) * double(Hp[0]))`` -- a DOUBLE
   product with the double ``Courant`` rounded once (step_db.cpp:324,
   ``stepping._realnum_double_product``), then a float32 add -- and ``Dp[0] = 0``. The
   multiplicand is a ``double`` kernel argument, :func:`axis_courant`: a float32
   ``4*Courant`` is a different factor wherever Courant is not a float32 word (0.5
   is; 0.3 is not). ``Hy`` is the stored Hp the D side already binds.

WHAT NEEDS NO CODE. ``_shift_up``'s ``CYL_AXIS`` arm is the METALLIC arm (a hard zero
at r_max). ``_shift_down``'s ``CYL_AXIS`` arm (the r_to_minus_r image of row 0) is NOT
implemented, and cannot be observed: the only targets taking a shift-down along r are
Dy (partner Hz) and Dz (whose curl is the radial difference and reads no ghost), and
Dy's r Yee shift is 0, so the ownership mask drops its row-0 curl, the only row the
near ghost would reach. A mutation that put a faithful ghost back must come out NOT
caught; if it is ever caught, the ownership mask has broken, not the ghost.

``--fmad=false`` is correctness: ``(fprev * kms) - curl`` and ``(fcur * kms_u) +
fu_new`` are contraction candidates the array path rounds twice.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML

# THE PRELUDE IS IMPORTED, NOT COPIED. ``shift_up``, ``shift_dn``, ``curl_stencil`` and
# ``pml_apply`` are the root's spellings, and a second copy here would be a second
# thing to keep in step.
from .step_curl_kernels import _REAL_PML_PRELUDE, _REAL_PML_THREADS, chunk_record_argument
from .compile_cache import get_or_compile, kernel_cache_key, clear_kernel_cache
from .coverage import real_pml_boundary_kinds
from .cylindrical_coverage import (
    covers_real_pml_cylindrical_curl, cylindrical_boundary_codes)
from .cylindrical_prefix import radial_difference, radial_scan_device_tables, radial_scan_tables


#: The two entry points. The records that certified their 0.9.2 texts are re-cut in
#: the combined certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = (
    "cyl_step_B_pml_real",
    "cyl_step_D_pml_real",
)

#: This file's half of the certified / uncertified partition ``step_curl_kernels.py``
#: maintains.
UNCERTIFIED_KERNELS: Tuple[str, ...] = ()
#: The compile options this module passes NVRTC; a gate substitutes the empty tuple
#: as a control.
_COMPILE_OPTIONS = ('--fmad=false',)

# =============================================================================
# THE B SIDE
# =============================================================================
#
# Bx and By are the root's blocks on the Dcyl term table; Bz's whole curl is dtdx
# times the radial difference; the m = 0 axis tail zeroes Br on the axis row after
# the recurrence. Every B component reads the HALF-INTEGER record along its dsigu
# axis (bounds_block.curl_entries).

_cyl_step_B_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void cyl_step_B_pml_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    const float* __restrict__ rad,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
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

    // THE CHUNK RECORD (chunk_map.chunk_bounds_record): row r of this lattice and axis
    // lies in a chunk that carries the axis's layer iff (a0 <= r < a1) || (b0 <= r < b1).
    const int cbh_x_a0 = cb[12], cbh_x_a1 = cb[13], cbh_x_b0 = cb[14], cbh_x_b1 = cb[15];
    const int in_h_x = (i >= cbh_x_a0 && i < cbh_x_a1) || (i >= cbh_x_b0 && i < cbh_x_b1);
    const int cbh_y_a0 = cb[16], cbh_y_a1 = cb[17], cbh_y_b0 = cb[18], cbh_y_b1 = cb[19];
    const int in_h_y = (j >= cbh_y_a0 && j < cbh_y_a1) || (j >= cbh_y_b0 && j < cbh_y_b1);
    const int cbh_z_a0 = cb[20], cbh_z_a1 = cb[21], cbh_z_b0 = cb[22], cbh_z_b1 = cb[23];
    const int in_h_z = (k >= cbh_z_a0 && k < cbh_z_a1) || (k >= cbh_z_b0 && k < cbh_z_b1);
    const int cb_presence = cb[24];
    const int pr_x = (cb_presence >> 0) & 1;
    const int pr_y = (cb_presence >> 1) & 1;
    const int pr_z = (cb_presence >> 2) & 1;

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> cell 0 on x.
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_y, pr_z);
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], in_h_z);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> cell 0 on y.
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_z, pr_x);
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], in_h_x);
    }

    // Bz: dsig=x, dsigu=y; iyee=(1,1,0) -> cell 0 on z. CYLINDRICAL: the phi partner
    // is the derivative MEEP drops, so the WHOLE curl is dtdx times the radial
    // difference of Ep's per-chunk sum (stepping.step_B; the scan wrote rad).
    {
        float curl = dtdx * rad[idx];
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j], in_h_y);
    }

    // stepping._cylindrical_axis_zero_B, m = 0: Br on the axis row is +0, after the
    // recurrence. The auxiliary is not touched.
    if (i == 0) Bx[idx] = 0.0f;
}
'''


# =============================================================================
# THE D SIDE
# =============================================================================
#
# Dx and Dy are the root's blocks on the Dcyl term table; Dz's whole curl is dtdx times
# the radial difference of Hp's per-chunk sum; the axis tail carries the m = 0 on-axis
# Dz rule and Dp = 0. Every D component reads the INTEGER record along its dsigu axis.

_cyl_step_D_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void cyl_step_D_pml_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ rad,
    int nx, int ny, int nz, float dtdx, double axis_courant,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
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

    // THE CHUNK RECORD (chunk_map.chunk_bounds_record): row r of this lattice and axis
    // lies in a chunk that carries the axis's layer iff (a0 <= r < a1) || (b0 <= r < b1).
    const int cbi_x_a0 = cb[0], cbi_x_a1 = cb[1], cbi_x_b0 = cb[2], cbi_x_b1 = cb[3];
    const int in_i_x = (i >= cbi_x_a0 && i < cbi_x_a1) || (i >= cbi_x_b0 && i < cbi_x_b1);
    const int cbi_y_a0 = cb[4], cbi_y_a1 = cb[5], cbi_y_b0 = cb[6], cbi_y_b1 = cb[7];
    const int in_i_y = (j >= cbi_y_a0 && j < cbi_y_a1) || (j >= cbi_y_b0 && j < cbi_y_b1);
    const int cbi_z_a0 = cb[8], cbi_z_a1 = cb[9], cbi_z_b0 = cb[10], cbi_z_b1 = cb[11];
    const int in_i_z = (k >= cbi_z_a0 && k < cbi_z_a1) || (k >= cbi_z_b0 && k < cbi_z_b1);
    const int cb_presence = cb[24];
    const int pr_x = (cb_presence >> 0) & 1;
    const int pr_y = (cb_presence >> 1) & 1;
    const int pr_z = (cb_presence >> 2) & 1;

    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0) -> cell 0 on y, z.
    {
        float f1 = Hz[idx];
        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);
        float f2 = Hy[idx];
        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);
        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_y, pr_z);
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], in_i_z);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0) -> cell 0 on x, z.
    {
        float f1 = Hx[idx];
        float sf = shift_dn(Hx, idx, k, nz, sz, bc_z);
        float f2 = Hz[idx];
        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);
        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_z, pr_x);
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], in_i_x);
    }

    // Dz: dsig=x, dsigu=y; iyee=(0,0,1) -> cell 0 on x, y. CYLINDRICAL: the whole curl
    // is dtdx times the backward radial difference of Hp's per-chunk sum
    // (stepping.step_D; the scan wrote rad).
    {
        float curl = dtdx * rad[idx];
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j], in_i_y);
    }

    // stepping._cylindrical_axis_zero_D, m = 0, on the updated words: Dz on the axis
    // takes realnum(double(4*Courant) * double(Hp)) -- MEEP's double product, rounded
    // once (step_db.cpp:324) -- added in float32; then Dp on the axis is +0. Hy is the
    // stored Hp (update_H has not run yet this sub-step).
    if (i == 0) {
        Dz[idx] = Dz[idx] + __double2float_rn(axis_courant * (double)(Hy[idx]));
        Dy[idx] = 0.0f;
    }
}
'''


# =============================================================================
# COMPILATION
# =============================================================================

def _kernel_code_map() -> Dict[str, Tuple[str, str]]:
    """``sub_step -> (source, entry point)``, REBUILT PER CALL.

    Per-call is load-bearing: a gate mutates a kernel by assigning over the
    module-level source string, and a memoized map would hand back the pre-mutation
    string forever -- a leg reporting a pass for a mutation it never applied.
    """
    return {
        "step_B": (_cyl_step_B_pml_real_kernel_code, "cyl_step_B_pml_real"),
        "step_D": (_cyl_step_D_pml_real_kernel_code, "cyl_step_D_pml_real"),
    }


def _get_cylindrical_kernel(sub_step: str):
    """The compiled kernel for one sub-step, through the shared memo (the subnormal
    policy in force and the source actually compiled are both in the key)."""
    code, entry = _kernel_code_map()[sub_step]
    key = kernel_cache_key(f"cyl_{sub_step}_pml_real", False, _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, entry, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return clear_kernel_cache()


# =============================================================================
# THE LAUNCH
# =============================================================================

def cylindrical_curl_tables(pml: 'PML', half_integer: bool) -> dict:
    """Flatten one sub-step's six PML coefficient vectors into cached device views.

    Called ONCE per frozen configuration, not per launch. ``half_integer`` selects the
    sub-lattice: the B curl reads the half-integer positions (``kms_x_h``) and the D
    curl the integer ones. Backwards, it is a half-cell error in the absorber profile.
    """
    suffix = "_h" if half_integer else ""
    tables = {}
    for axis in ("x", "y", "z"):
        for name in ("kms", "sinv"):
            attribute = f"{name}_{axis}{suffix}"
            flat = getattr(pml, attribute).reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the cylindrical PML kernels index "
                    f"float32 coefficient vectors.")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector.")
            tables[f"{name}_{axis}"] = flat
    return tables


def axis_courant(grid) -> Any:
    """The m = 0 on-axis Dz factor, ``4 * Courant`` IN DOUBLE (``np.float64``).

    ``stepping._cylindrical_axis_zero_D`` multiplies the stored Hp by
    ``4.0 * float(grid.courant)`` in double and rounds the product once
    (``_realnum_double_product``, MEEP's step_db.cpp:324). The word travels as a
    ``double`` kernel argument; a float32 ``4*Courant`` is another factor wherever
    Courant is not a float32 word.
    """
    return np.float64(4.0 * float(grid.courant))


def cylindrical_boundary_codes_for(grid) -> Tuple[Any, Any, Any]:
    """This grid's three kernel boundary codes, as ``np.int32``.

    ``real_pml_boundary_kinds`` resolves the axes the way ``stepping._boundary_kinds``
    does; :func:`cylindrical_boundary_codes` maps ``'axis'`` onto ``BC_METALLIC`` on
    the r axis and REFUSES it anywhere else.
    """
    codes = cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    return tuple(np.int32(code) for code in codes)


def _launch(kernel, targets, auxiliaries, sources, radial, tables: dict,
            boundary_codes, dtdx: float, chunk_bounds, extra_scalars=()) -> None:
    """Shared launch for both sides: the argument list differs only in which arrays.

    The grid is sized from the TARGET volume, which the radial difference shares.
    """
    cb = chunk_record_argument(chunk_bounds)
    nx, ny, nz = targets[0].shape
    if tuple(radial.shape) != (nx, ny, nz):
        raise ValueError(f"the radial difference is {tuple(radial.shape)}, the target "
                         f"{(nx, ny, nz)}; the curl reads it at the target's index")
    blocks = (nx * ny * nz + _REAL_PML_THREADS - 1) // _REAL_PML_THREADS
    kernel((blocks,), (_REAL_PML_THREADS,), (
        targets[0], targets[1], targets[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        sources[0], sources[1], sources[2],
        radial,
        np.int32(nx), np.int32(ny), np.int32(nz),
        np.float32(dtdx), *extra_scalars,
        tables["kms_x"], tables["sinv_x"],
        tables["kms_y"], tables["sinv_y"],
        tables["kms_z"], tables["sinv_z"],
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        cb,
    ))


def _step_B_fused_pml_cylindrical(fields: 'Fields', tables: dict, boundary_codes,
                                  dtdx: float, radial=None, scratch: Any = None, *,
                                  chunk_bounds, pml: 'PML' = None, scan=None):
    """``stepping.step_B`` for a real m = 0 Dcyl grid under a PML: scan, then curl.

    ``tables`` are the HALF-INTEGER views (:func:`cylindrical_curl_tables`).
    ``radial`` may be handed in by a caller that already computed it (a gate does, so
    the same words go to both paths); otherwise the scan runs here over the current Ep,
    from ``scan`` (``cylindrical_prefix.radial_scan_device_tables``) or from ``pml``.
    """
    if radial is None:
        radial = radial_difference(fields, pml, "step_B", tables=scan, scratch=scratch)
    _launch(
        _get_cylindrical_kernel("step_B"),
        (fields.Bx, fields.By, fields.Bz),
        (fields.fu_Bx, fields.fu_By, fields.fu_Bz),
        (fields.Ex, fields.Ey, fields.Ez),
        radial, tables, boundary_codes, dtdx, chunk_bounds)


def _step_D_fused_pml_cylindrical(fields: 'Fields', tables: dict, boundary_codes,
                                  dtdx: float, radial=None, scratch: Any = None, *,
                                  chunk_bounds, pml: 'PML' = None, scan=None):
    """``stepping.step_D`` for a real m = 0 Dcyl grid under a PML: scan, then curl.

    ``tables`` are the INTEGER views (:func:`cylindrical_curl_tables`).
    """
    if radial is None:
        radial = radial_difference(fields, pml, "step_D", tables=scan, scratch=scratch)
    _launch(
        _get_cylindrical_kernel("step_D"),
        (fields.Dx, fields.Dy, fields.Dz),
        (fields.fu_Dx, fields.fu_Dy, fields.fu_Dz),
        (fields.Hx, fields.Hy, fields.Hz),
        radial, tables, boundary_codes, dtdx, chunk_bounds,
        extra_scalars=(axis_courant(fields.grid),))


#: Both entry points by sub-step name, so a caller does not spell an if.
CYLINDRICAL_LAUNCHERS = {"step_B": _step_B_fused_pml_cylindrical,
                         "step_D": _step_D_fused_pml_cylindrical}

#: Which PML sub-lattice each sub-step's tables come from -- ``step_B`` passes
#: ``half_integer=True`` and ``step_D`` False (``stepping._apply_curl``).
HALF_INTEGER: Dict[str, bool] = {"step_B": True, "step_D": False}


def device_chunk_record(fields: 'Fields', pml: 'PML') -> Any:
    """The record of the division the array path steps this run with, on the device.

    ``chunk_map.chunk_bounds_record`` of the region plan's layout -- the lifted
    simulation's own division when the driver installed one, else the one-rank one. A
    plan that already holds the composition's record (``arms.StepContext``) passes that.
    """
    from ..chunk_map import chunk_bounds_record  # noqa: PLC0415 - NumPy only
    from ..stepping import _region_plan  # noqa: PLC0415 - NumPy only

    record = chunk_bounds_record(_region_plan(fields, pml).layout)
    return cp.ascontiguousarray(cp.asarray(record, dtype=cp.int32))


def step_cylindrical(fields: 'Fields', pml: 'PML', sub_step: str,
                     tables: dict = None, boundary_codes=None, chunk_bounds=None) -> bool:
    """Step one Dcyl curl sub-step on the device, or return False if not covered.

    FALSE IS THE ONLY REFUSAL: a configuration this pair does not carry falls back to
    the array path, never raises into a caller that would otherwise have stepped
    correctly. ``chunk_bounds`` defaults to :func:`device_chunk_record`.
    """
    covered, _reason = covers_real_pml_cylindrical_curl(fields, pml, fields.grid,
                                                        sub_step)
    if not covered:
        return False
    if tables is None:
        tables = cylindrical_curl_tables(pml, HALF_INTEGER[sub_step])
    if boundary_codes is None:
        boundary_codes = cylindrical_boundary_codes_for(fields.grid)
    if chunk_bounds is None:
        chunk_bounds = device_chunk_record(fields, pml)
    grid = fields.grid
    scan = radial_scan_device_tables(radial_scan_tables(fields, pml, sub_step), grid.xp)
    CYLINDRICAL_LAUNCHERS[sub_step](fields, tables, boundary_codes,
                                    float(grid.dt / grid.dx),
                                    scratch=getattr(fields, "scratch", None),
                                    chunk_bounds=chunk_bounds, pml=pml, scan=scan)
    return True
