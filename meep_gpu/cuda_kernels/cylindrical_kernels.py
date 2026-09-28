"""CYLINDRICAL (Dcyl) curl sub-steps at m = 0, in hand CUDA — real float32 storage.

WHAT IT REPLACES. ``stepping.step_B`` (:261) and ``stepping.step_D`` (:379) on a
Dcyl grid with ``grid.m == 0``, real storage and an active split-field PML —
everything except the radial prefix scan, which stays on the array path
DELIBERATELY and by the sibling track's decisive measurement. See "THE SCAN".

CUDA HAS NEVER BEEN MEASURED ON A Dcyl RUN AT ANY SUB-STEP. The shipped pair
``step_B_pml_real`` / ``step_D_pml_real`` refuses cylindrical BY NAME in
``coverage.py:499-503``, and that refusal is CORRECT for those two kernels: they have
no prefix pointer, no axis tail and no ``4*Courant`` scalar, so a Dcyl grid handed to
them is a wrong answer on four of the six targets. These are a SECOND PAIR, with a
different signature and their own predicate in ``cylindrical_coverage.py``. Nothing
in ``step_curl_kernels.py`` or ``coverage.py`` changes.

=============================================================================
WHAT CYLINDRICAL ACTUALLY REQUIRES THAT THE CARTESIAN PAIR DOES NOT
=============================================================================

FOUR THINGS, and only four. Everything else — the ghost rule, the curl grouping on
targets 0 and 1, the ownership mask and the split-field recurrence — is the CERTIFIED
Cartesian body, character for character, reached through the SAME
``_REAL_PML_PRELUDE`` string (imported, not copied, so the two cannot drift and so
the shared mutation battery's needles arm here unchanged).

1. **A RADIAL PREFIX SUM.** MEEP's ``f_rderiv_int`` (step_db.cpp:99-119, transcribed
   at ``stepping.cylindrical_rderiv_prefix``, :1257-1305) turns ``(1/r)d(r*Fp)/dr``
   into a plain difference by building a running sum along r. That is a SCAN — a
   sequentially dependent reduction — and no amount of widening turns an elementwise
   curl kernel into one. It is the reason this is a new kernel rather than a flag.
2. **Bz's WHOLE CURL IS REPLACED** (stepping.py:371-375). ``step_B`` extends Ep
   (= Ey) by ONE ZERO WALL ROW to (nr + 1, nphi, nz) (:354-360), prefixes it at
   ``ir0 = 0.0``, and computes ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` — ONE
   subtract and ONE multiply, NOT the four-operand grouping, which is a different
   float32 number. The wall row is not decoration: without it the forward difference
   at the last row becomes minus the whole accumulated sum (:303-314 records the
   symptom that found it).
3. **Dz's ``first`` SOURCE IS SWAPPED** (stepping.py:442-455). Hp (= Hy) is prefixed
   at ``ir0 = 0.5`` — half of Hp's r-Yee shift — and substituted for the Dz term
   ONLY; the unmodified backward-difference machinery then produces the cylindrical
   derivative. ``Dx``'s Hy operands stay RAW, which is why this kernel binds both the
   prefix and the untouched ``Hy``. No wall row on this side: a backward difference
   reads rows i and i-1 and never looks past the top.
4. **THE m = 0 AXIS RULES**, applied AFTER the recurrence:
   ``Bx[r=0] = 0`` (``_cylindrical_axis_zero_B``, stepping.py:688-690);
   ``Dz[r=0] += (4*Courant)*Hp[r=0]`` then ``Dy[r=0] = 0``
   (``_cylindrical_axis_zero_D``, :583-587). The D-side increment is a POST-ADD and
   NOT folded into the curl, and that was MEASURED rather than chosen: the fold
   breaks m = 0 under PML (Er 2.7e-01 / Hp 4.5e-01 against 3.6e-07,
   stepping.py:574-580), because Dz's dsig is R, whose sigma is zero on the axis row,
   so the plain post-add is the exact one. ``_cylindrical_axis_increment_B`` and
   ``_D`` both return ``None`` at m = 0 (:531-535, :640-641), so there is no curl-row
   fold to carry at this azimuthal order — a fact READ OFF the array path, not
   assumed from the |m| = 1 code being nearby.

WHAT NEEDED NO NEW CODE AT ALL:

* **The r axis compiles as METALLIC.** ``_shift_up``'s ``CYL_AXIS`` arm IS the
  ``METALLIC`` arm, character for character (stepping.py:1828-1830) — a hard zero at
  the far r face, the PEC wall a Dcyl cell carries at ``r_max``.
  ``_shift_down``'s ``CYL_AXIS`` arm is NOT (:1830-1846): it images stored row 0 with
  a direction sign and the ``(-1)^m`` phase. It is UNOBSERVABLE at m = 0 — the only
  terms taking a shift-down along r are ``Dy`` (partner Hz, second operand) and
  ``Dz`` (partner Hp, first operand), both with r-Yee shift 0, so the ownership mask
  zeroes their curl at exactly the row the near ghost writes. BOTH sibling tracks
  measured that rather than arguing it (Triton's ``axis_ghost_sign`` 16/16 uncaught;
  Metal's array-path probe 0 of 33,012 words with 32 ghost substitutions fired), and
  the CUDA gate carries the same mutation ARMED AS A NULL. No byte gate can certify
  the choice from inside the kernel, so it is stated here: if a run ever reports that
  mutation CAUGHT, the ownership mask has broken, not the ghost.
* **The ownership mask is the certified body's.** With the r axis compiled METALLIC,
  ``if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;`` masks exactly the cells
  ``_mask_non_owned_cells``' ``is_axis`` clause masks (stepping.py:1945-1949). The
  ``_stored_past_owned`` far-face clause is not reproduced and does not need to be:
  it fires only on a FOLDED periodic axis, which the predicate refuses.

=============================================================================
THE SCAN STAYS ON THE ARRAY PATH — INHERITED AS A READING, RE-MEASURED HERE
=============================================================================

The oracle a CUDA kernel must reproduce on this backend is ``cupy.cumsum``'s float32
summation order, because ``cylindrical_rderiv_prefix`` calls ``xp.cumsum`` and ``xp``
is CuPy on every configuration this predicate admits (stepping.py:1315, :1333).
The Triton tranche measured that oracle to be unreproducible by any other scan:
``cupy.cumsum`` in float32 is deterministic across repeats but is NOT a sequential
accumulation (87,270 of 102,400 elements differ at (320,1,320)), ``numpy.cumsum`` IS
byte-equal to a sequential accumulation, and ``tl.cumsum`` in a single tile matches
NEITHER (57,172 of 102,400 differ from CuPy). A hand CUDA column-serial scan is in
the same position as ``tl.cumsum``: it computes the sequential sum, which is NumPy's
answer and not CuPy's.

THAT IS A HYPOTHESIS ABOUT THIS BACKEND UNTIL MEASURED ON IT, so
``gate_cuda_cylindrical_real.py`` carries a leg (``scan_order``) that compiles a
column-serial CUDA scan and compares it word for word against ``cupy.cumsum`` on the
gate's own shapes. Whatever it reports, the DESIGN does not change: this pair does
not own the scan. What the leg settles is whether "the scan cannot be fused
bit-identically on CuPy" is a measurement on this backend or a citation from another.

THE CEILING THIS IMPOSES IS REAL AND IS NOT HIDDEN. The curl pair cannot go below the
prefix's own array-path launches, so no throughput claim is made or possible here;
this module is a CORRECTNESS artifact only. The Metal port DOES fuse the scan, and it
can, because that backend's engine holds NumPy and the sequential sum IS its oracle
(``metal_kernels/cylindrical_real.py``, "THE SCAN RUNS ON THE DEVICE HERE"). The
premise genuinely inverts between backends; it is not a difference of taste.

=============================================================================
WHAT IS CUDA-SPECIFIC
=============================================================================

* ``_COMPILE_OPTIONS``' ``--fmad=false`` is CORRECTNESS here, not tuning, for the
  same two reasons the certified pair gives: ``(fu*kms) - curl`` and
  ``(f*kms_u) + fu_new`` are both FMA candidates the array path rounds twice. The
  gate scores an unguarded control and reports whether the guard actually bit.
* ``axis_coef`` IS HOST-ROUNDED and is NOT ``4.0f * dtdx`` recomputed in the kernel.
  The array path forms ``4.0 * (grid.dt / grid.dx)`` as a Python float and NEP-50
  casts it WEAKLY onto float32 storage (stepping.py:585), so the multiplicand is the
  float32 rounding of the float64 product. (The two happen to agree because scaling
  by four is exact; the gate carries the recomputation as a MEASURED NULL rather than
  leaving the agreement to luck.)
* The boundary kind stays an ARGUMENT, not a layout: ``bc_x``/``bc_y``/``bc_z`` are
  passed exactly as the certified pair passes them, with ``'axis'`` resolved to
  ``BC_METALLIC`` by :func:`cylindrical_boundary_codes` in the predicate module. That
  keeps the shared ``drop_metallic_mask`` mutation armed on this source unchanged.

=============================================================================
NOT CERTIFIED, NOT WIRED
=============================================================================

These two kernels are NOT in ``certification.json`` and must not be counted as
coverage until ``gate_cuda_cylindrical_real.py`` releases against them on hardware.
``fastpath.plan_fast_path`` is untouched and still returns ``None`` on every branch.
:data:`UNCERTIFIED_KERNELS` below is this file's half of the same partition
``step_curl_kernels.py`` maintains, so a kernel added here without a gate verdict is
visible as uncertified rather than silently counted.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML

# THE PRELUDE IS IMPORTED, NOT COPIED. ``shift_up``, ``shift_dn`` and ``pml_apply``
# are the certified spellings and a second copy of them here would be a second thing
# to keep in step — and would silently un-arm the shared mutation battery, whose
# needles are those exact strings (``probe_fused_kernel_bit_identity._swap_dsig_dsigu``
# and friends match on ``float fu_new = ((fprev * kms) - curl) * sinv;``).
from .step_curl_kernels import _REAL_PML_PRELUDE, _REAL_PML_THREADS
from .compile_cache import get_or_compile, kernel_cache_key, clear_kernel_cache
from .coverage import real_pml_boundary_kinds
from .cylindrical_coverage import (
    covers_real_pml_cylindrical_curl, cylindrical_boundary_codes)
# The prefix lives in a CUPY-FREE sibling and is imported back here, exactly as the
# predicate and the compile memo are: it is pure array work over ``grid.xp``, and the
# wall-row extension is the sort of off-by-one a laptop test must be able to pin.
from .cylindrical_prefix import cylindrical_prefix


#: RECORDED 2026-08-27. The gate was re-run on the GPU host and RELEASED under both
#: float32 subnormal policies; the block is ``cuda_cylindrical_real_2026-08-27`` in
#: ``certification.json``, and the two halves landed together as the partition
#: test requires. The subject module was checked UNCHANGED since that run before
#: the record was landed, so this names the bytes that ship.
CERTIFIED_KERNELS = (
    "cyl_step_B_pml_real",
    "cyl_step_D_pml_real",
)

#: NOT CERTIFIED. Neither kernel in this file has a gate verdict in
#: ``certification.json``; both are here to BE gated. This tuple is the file's half
#: of the partition ``step_curl_kernels.py`` maintains between certified and
#: uncertified kernels, restated so a reader who arrives at this file first is told
#: the status before reading the sources.
UNCERTIFIED_KERNELS: Tuple[str, ...] = ()
#: The compile options this module passes NVRTC. ``--fmad=false`` is correctness on
#: this sub-step; the gate substitutes the empty tuple as a control and reports
#: whether the guard changed an answer.
_COMPILE_OPTIONS = ('--fmad=false',)

# =============================================================================
# THE B SIDE
# =============================================================================
#
# Bx and By are the CERTIFIED Cartesian blocks, unchanged. Bz's whole curl is the
# radial forward difference of the wall-extended prefix, and the axis tail zeroes Br
# on the axis row after the recurrence.

_cyl_step_B_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void cyl_step_B_pml_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    const float* __restrict__ pfx,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> mask on x.
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> mask on y.
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: CYLINDRICAL SUBSTITUTION (stepping.py:343-347). The phi-derivative partner
    // is the invariant-axis exact zero, so the whole curl IS the radial forward
    // difference of the WALL-EXTENDED prefix: ONE subtract, ONE multiply. NOT the
    // four-operand grouping and NOT `dtdx*pfx_up - dtdx*pfx_here`, both of which are
    // different float32 numbers. `pfx` has nx + 1 rows at the same (phi, z) stride,
    // so row i + 1 is idx + sx and the read is in bounds at i = nx - 1.
    // iyee=(1,1,0) -> mask on z, exactly as in the Cartesian body.
    {
        float pfx_here = pfx[idx];
        float pfx_up   = pfx[idx + sx];
        float curl = dtdx * (pfx_up - pfx_here);
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }

    // stepping._cylindrical_axis_zero_B:659-661 -- Br on the axis is identically
    // zero (a radial vector at r = 0 points nowhere). AFTER the recurrence, exactly
    // as the array path runs it after the term loop (:375-376): the auxiliary is NOT
    // zeroed, only the field.
    if (i == 0) Bx[idx] = 0.0f;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 343-347->371-375


# =============================================================================
# THE D SIDE
# =============================================================================
#
# Dx and Dy are the CERTIFIED Cartesian blocks, unchanged — including Dx's RAW Hy
# operands, which is why this kernel binds both `Hy` and `pfx`. Dz's `first` operand
# pair comes from the prefix and its ghost rule is the same METALLIC rule the raw
# operand takes. The axis tail carries the m = 0 on-axis Dz post-add and Dp = 0.

_cyl_step_D_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void cyl_step_D_pml_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ pfx,
    int nx, int ny, int nz, float dtdx, float axis_coef,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0) -> mask y, z.
    // Hy IS RAW HERE. Only Dz's term takes the prefix (stepping.py:424-426); a
    // prefix leaking into Dx is a defect the gate arms as `prefix_into_dx`.
    {
        float f1 = Hz[idx];
        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);
        float f2 = Hy[idx];
        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0) -> mask x, z.
    {
        float f1 = Hx[idx];
        float sf = shift_dn(Hx, idx, k, nz, sz, bc_z);
        float f2 = Hz[idx];
        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Dz: CYLINDRICAL SUBSTITUTION (stepping.py:413-426). The `first` source is the
    // prefix of Hp at ir0 = 0.5 and the UNMODIFIED backward machinery then produces
    // the cylindrical derivative; the second operand pair (Hx along phi) is
    // untouched. iyee=(0,0,1) -> mask x, y, unchanged from the Cartesian body.
    {
        float f1 = pfx[idx];
        float sf = shift_dn(pfx, idx, i, nx, sx, bc_x);
        float f2 = Hx[idx];
        float ss = shift_dn(Hx, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }

    // stepping._cylindrical_axis_zero_D:583-587, in the array path's own order.
    // The Dz increment is a POST-ADD and deliberately NOT folded into the curl: the
    // fold was MEASURED to break m = 0 under PML (Er 2.7e-01 / Hp 4.5e-01 against
    // 3.6e-07, :545-551), because Dz's dsig is R, whose sigma is zero on the axis
    // row, so the plain post-add is the exact one. `axis_coef` is the HOST-ROUNDED
    // float32 of 4.0*(dt/dx) -- see the module docstring -- and `Hy` is the raw
    // stored Hp, which is what `fields.get_H("Hy")` returns under an active PML
    // (update_H has not run yet this sub-step). Two separate roundings, as
    // `A += s * B` performs them.
    if (i == 0) {
        Dz[idx] = Dz[idx] + (axis_coef * Hy[idx]);
        Dy[idx] = 0.0f;
    }
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 424-426->453-455, 413-426->442-455


# =============================================================================
# COMPILATION
# =============================================================================

def _kernel_code_map() -> Dict[str, Tuple[str, str]]:
    """``sub_step -> (source, entry point)``, REBUILT PER CALL.

    Per-call is load-bearing for the same reason ``step_curl_kernels._get_kernel``
    rebuilds its map: the gate mutates a kernel by assigning over the module-level
    source string, and a memoized map would hand back the pre-mutation string
    forever — a leg reporting a pass for a mutation it never applied.
    """
    return {
        "step_B": (_cyl_step_B_pml_real_kernel_code, "cyl_step_B_pml_real"),
        "step_D": (_cyl_step_D_pml_real_kernel_code, "cyl_step_D_pml_real"),
    }


def _get_cylindrical_kernel(sub_step: str):
    """The compiled kernel for one sub-step, through the shared memo.

    The memo key carries the subnormal policy in force AND the source string
    actually compiled, so neither a late policy install nor a mutated body can be
    served an earlier binary.
    """
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

    Called ONCE per frozen configuration, not per launch — the same discipline
    ``step_curl_kernels.real_pml_curl_tables`` applies, and for the same reason
    (port-reference defect 5.14: six device allocations per sub-step for tables that
    never change). ``half_integer`` selects the sub-lattice: the B curl reads the
    half-integer positions (``kms_x_h``) and the D curl the integer ones. Getting
    that backwards is a half-cell error in the absorber profile, not a crash.
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


def axis_coefficient(dtdx: float) -> Any:
    """The m = 0 on-axis Dz multiplicand, rounded ON THE HOST.

    ``4.0 * (grid.dt / grid.dx)`` is formed in float64 by the array path and NEP-50
    casts that one scalar to float32 before the multiply (stepping.py:585). Binding
    the already-rounded word is the literal transcription; recomputing
    ``4.0f * dtdx`` inside the kernel from an fp32 ``dtdx`` is a different number in
    general (it happens to agree here because scaling by four is exact, which the
    gate measures as a null rather than assuming).
    """
    return np.float32(4.0 * float(dtdx))


def cylindrical_boundary_codes_for(grid) -> Tuple[Any, Any, Any]:
    """This grid's three kernel boundary codes, as ``np.int32``.

    ``real_pml_boundary_kinds`` resolves the axes the way ``stepping._boundary_kinds``
    does; :func:`cylindrical_boundary_codes` maps ``'axis'`` onto ``BC_METALLIC`` on
    the r axis and REFUSES it anywhere else.
    """
    codes = cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    return tuple(np.int32(code) for code in codes)


def _launch(kernel, targets, auxiliaries, sources, prefix, tables: dict,
            boundary_codes, dtdx: float, extra_scalars: Sequence[Any] = ()) -> None:
    """Shared launch for both sides: the argument list differs only in which arrays.

    THE GRID IS SIZED FROM THE TARGET VOLUME, never from the prefix. The B side's
    prefix is one row taller and a dispatch sized from it would step cells the array
    path does not own.
    """
    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _REAL_PML_THREADS - 1) // _REAL_PML_THREADS
    kernel((blocks,), (_REAL_PML_THREADS,), (
        targets[0], targets[1], targets[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        sources[0], sources[1], sources[2],
        prefix,
        np.int32(nx), np.int32(ny), np.int32(nz),
        np.float32(dtdx), *extra_scalars,
        tables["kms_x"], tables["sinv_x"],
        tables["kms_y"], tables["sinv_y"],
        tables["kms_z"], tables["sinv_z"],
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ))


def _step_B_fused_pml_cylindrical(fields: 'Fields', tables: dict, boundary_codes,
                                  dtdx: float, prefix=None, scratch: Any = None):
    """``stepping.step_B`` for a real m = 0 Dcyl grid under a PML, in one launch.

    ``tables`` are the HALF-INTEGER coefficient views from
    :func:`cylindrical_curl_tables` (``half_integer=True``). ``prefix`` may be handed
    in by a caller that already computed it (the gate does, so the same words go to
    both paths); otherwise it is computed here from the current Ey.
    """
    if prefix is None:
        prefix = cylindrical_prefix(fields, "step_B", scratch=scratch)
    _launch(
        _get_cylindrical_kernel("step_B"),
        (fields.Bx, fields.By, fields.Bz),
        (fields.fu_Bx, fields.fu_By, fields.fu_Bz),
        (fields.Ex, fields.Ey, fields.Ez),
        prefix, tables, boundary_codes, dtdx)


def _step_D_fused_pml_cylindrical(fields: 'Fields', tables: dict, boundary_codes,
                                  dtdx: float, prefix=None, scratch: Any = None):
    """``stepping.step_D`` for a real m = 0 Dcyl grid under a PML, in one launch.

    ``tables`` are the INTEGER coefficient views from
    :func:`cylindrical_curl_tables` (``half_integer=False``).
    """
    if prefix is None:
        prefix = cylindrical_prefix(fields, "step_D", scratch=scratch)
    _launch(
        _get_cylindrical_kernel("step_D"),
        (fields.Dx, fields.Dy, fields.Dz),
        (fields.fu_Dx, fields.fu_Dy, fields.fu_Dz),
        (fields.Hx, fields.Hy, fields.Hz),
        prefix, tables, boundary_codes, dtdx,
        extra_scalars=(axis_coefficient(dtdx),))


#: Both entry points by sub-step name, so a caller does not spell an if.
CYLINDRICAL_LAUNCHERS = {"step_B": _step_B_fused_pml_cylindrical,
                         "step_D": _step_D_fused_pml_cylindrical}

#: Which PML sub-lattice each sub-step's tables come from — ``step_B`` passes
#: ``half_integer=True`` (stepping.py:404) and ``step_D`` False (:486).
HALF_INTEGER: Dict[str, bool] = {"step_B": True, "step_D": False}


def step_cylindrical(fields: 'Fields', pml: 'PML', sub_step: str,
                     tables: dict = None, boundary_codes=None) -> bool:
    """Step one Dcyl curl sub-step on the device, or return False if not covered.

    FALSE IS THE ONLY REFUSAL: a configuration this pair does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly. NOTHING CALLS THIS YET — ``fastpath.plan_fast_path`` still
    returns ``None`` on every branch, and these kernels are not certified.
    """
    covered, _reason = covers_real_pml_cylindrical_curl(fields, pml, fields.grid,
                                                        sub_step)
    if not covered:
        return False
    if tables is None:
        tables = cylindrical_curl_tables(pml, HALF_INTEGER[sub_step])
    if boundary_codes is None:
        boundary_codes = cylindrical_boundary_codes_for(fields.grid)
    grid = fields.grid
    CYLINDRICAL_LAUNCHERS[sub_step](fields, tables, boundary_codes,
                                    float(grid.dt / grid.dx),
                                    scratch=getattr(fields, "scratch", None))
    return True
