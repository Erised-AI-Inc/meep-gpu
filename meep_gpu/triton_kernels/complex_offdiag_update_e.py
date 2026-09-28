"""Complex64 off-diagonal electric constitutive update with two storage tails.

This family closes the tensor-row intersection left between the complex and
off-diagonal Triton products.  The numerical core is shared by both residual
shapes:

* a folded, active-PML tail stores ``f_w`` and accumulates ``E`` with the
  half-integer ``kps``/``kms`` vectors;
* an unfolded, inactive-PML tail stores the row product directly into ``E``.

The core transcribes ``stepping._offdiagonal_terms``.  It forms the partner-axis
DOWN pair, multiplies that pair by the real tensor coefficient, shifts the
already-multiplied product UP the row component's own axis, scales by 0.25, wall
masks the coupling, and adds it to ``D * chi1inv``.  Bloch phases are applied to
the shifted complex operand in the direction of each shift.  Mirror parity is
applied only to the partner-axis DOWN ghost; the own-axis UP ghost is zero, as
the array path calls ``_shift_up`` without a component or reflect row.

Dispatch is deliberately not wired here.  The module exposes predicates and
builders for a single later composition edit in ``launch.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_fields as _complex
from . import coverage as _coverage
from . import folded_complex as _folded
from . import folded_offdiag_update_e as _folded_offdiag
from . import offdiag_update_e as _offdiag


E_TERMS = _offdiag.E_TERMS
ROW_SLOTS = _offdiag.ROW_SLOTS
WALL_MASK_AXES = _offdiag.WALL_MASK_AXES
DEFAULT_BLOCK = _complex.DEFAULT_BLOCK
HALF_INTEGER = True

CODE_PERIODIC = _folded.CODE_PERIODIC
CODE_METALLIC = _folded.CODE_METALLIC
CODE_MIRROR_METALLIC = _folded.CODE_MIRROR_METALLIC
CODE_MIRROR_PERIODIC = _folded.CODE_MIRROR_PERIODIC
MIRROR_CODES = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
MIRROR_SOURCE_INDEX = _folded.MIRROR_SOURCE_INDEX

triton = _complex.triton
tl = _complex.tl

PERIODIC = tl.constexpr(CODE_PERIODIC)
METALLIC = tl.constexpr(CODE_METALLIC)
MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)
MIRROR_ROW = tl.constexpr(MIRROR_SOURCE_INDEX)

_mul_field_left = _complex._mul_field_left
_mul_coefficient_left = _complex._mul_coefficient_left
_rotate_field_left = _complex._rotate_field_left

__all__ = [
    "ComplexOffdiagUpdateEPlan",
    "complex_folded_offdiag_update_e_coverage",
    "complex_no_pml_offdiag_update_e_coverage",
    "complex_offdiag_update_e_step",
    "mirror_ghost_weights",
    "plan_complex_folded_offdiag_update_e",
    "plan_complex_no_pml_offdiag_update_e",
    "plan_complex_offdiag_update_e_from_arrays",
]


@triton.jit
def _complex_offdiag_term(
    g, u, o_c, o_d, o_u, o_ud, v_c, v_d, v_u, v_ud,
    down_boundary, up_boundary,
    dpr, dpi, upr, upi, ghost_weight,
    DPH: tl.constexpr, UPH: tl.constexpr, MG: tl.constexpr,
    EXPANSION: tl.constexpr,
):
    """One complex OFFDIAG term in MEEP's operand order.

    The coefficient multiply is between the shifts: both near and far pairs are
    multiplied at their own tensor-coefficient site.  Applying ``u[i]`` after
    the four-point average is algebraically similar for uniform ``u`` but not
    the Yee registration and not bitwise equivalent.
    """
    c_re = tl.load(g + 2 * o_c, mask=v_c, other=0.0)
    c_im = tl.load(g + 2 * o_c + 1, mask=v_c, other=0.0)
    d_re = tl.load(g + 2 * o_d, mask=v_d, other=0.0)
    d_im = tl.load(g + 2 * o_d + 1, mask=v_d, other=0.0)
    u_re = tl.load(g + 2 * o_u, mask=v_u, other=0.0)
    u_im = tl.load(g + 2 * o_u + 1, mask=v_u, other=0.0)
    ud_re = tl.load(g + 2 * o_ud, mask=v_ud, other=0.0)
    ud_im = tl.load(g + 2 * o_ud + 1, mask=v_ud, other=0.0)

    if DPH:
        r_re, r_im = _rotate_field_left(d_re, d_im, dpr, dpi, EXPANSION)
        d_re = tl.where(down_boundary, r_re, d_re)
        d_im = tl.where(down_boundary, r_im, d_im)
        r_re, r_im = _rotate_field_left(ud_re, ud_im, dpr, dpi, EXPANSION)
        ud_re = tl.where(down_boundary, r_re, ud_re)
        ud_im = tl.where(down_boundary, r_im, ud_im)
    if MG:
        r_re, r_im = _mul_coefficient_left(
            ghost_weight, d_re, d_im, EXPANSION)
        d_re = tl.where(down_boundary, r_re, d_re)
        d_im = tl.where(down_boundary, r_im, d_im)
        r_re, r_im = _mul_coefficient_left(
            ghost_weight, ud_re, ud_im, EXPANSION)
        ud_re = tl.where(down_boundary, r_re, ud_re)
        ud_im = tl.where(down_boundary, r_im, ud_im)

    near_re, near_im = _mul_field_left(
        c_re + d_re, c_im + d_im,
        tl.load(u + o_c, mask=v_c, other=0.0), EXPANSION)
    far_re, far_im = _mul_field_left(
        u_re + ud_re, u_im + ud_im,
        tl.load(u + o_u, mask=v_u, other=0.0), EXPANSION)

    # ``shift_up(product)`` phases the PRODUCT, not the pre-coefficient field.
    if UPH:
        r_re, r_im = _rotate_field_left(far_re, far_im, upr, upi, EXPANSION)
        far_re = tl.where(up_boundary, r_re, far_re)
        far_im = tl.where(up_boundary, r_im, far_im)

    return _mul_coefficient_left(
        0.25, near_re + far_re, near_im + far_im, EXPANSION)


@triton.jit
def _complex_masked_row_sum(diag_re, diag_im, total_re, total_im,
                            at_a, at_b,
                            WMA: tl.constexpr, WMB: tl.constexpr):
    """Mask coupling at metallic tangential-E walls, then add the diagonal."""
    if WMA:
        total_re = tl.where(at_a, 0.0, total_re)
        total_im = tl.where(at_a, 0.0, total_im)
    if WMB:
        total_re = tl.where(at_b, 0.0, total_re)
        total_im = tl.where(at_b, 0.0, total_im)
    return diag_re + total_re, diag_im + total_im


@triton.jit
def _complex_pml_tail(f, w, sr, si, kp, km, idx, live,
                      EXPANSION: tl.constexpr):
    """One component of ``_apply_constitutive_pml``, kept unrolled by callers."""
    prev_re = tl.load(w + 2 * idx, mask=live, other=0.0)
    prev_im = tl.load(w + 2 * idx + 1, mask=live, other=0.0)
    tl.store(w + 2 * idx, sr, mask=live)
    tl.store(w + 2 * idx + 1, si, mask=live)
    a_re = tl.load(f + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(f + 2 * idx + 1, mask=live, other=0.0)
    t_re, t_im = _mul_coefficient_left(kp, sr, si, EXPANSION)
    a_re, a_im = a_re + t_re, a_im + t_im
    t_re, t_im = _mul_coefficient_left(km, prev_re, prev_im, EXPANSION)
    tl.store(f + 2 * idx, a_re - t_re, mask=live)
    tl.store(f + 2 * idx + 1, a_im - t_im, mask=live)


@triton.jit
def complex_offdiag_update_e_step(
    f0, f1, f2,                     # Ex/Ey/Ez complex64 word views
    w0, w1, w2,                     # f_w_Ex/Ey/Ez, unused when PML=0
    g0, g1, g2,                     # Dx/Dy/Dz complex64 word views
    e0, e1, e2,                     # diagonal inverse epsilon, real f32
    u01, u02, u11, u12, u21, u22,  # six real off-diagonal row volumes
    kp0, km0, kp1, km1, kp2, km2,  # half-integer PML vectors, dead for PML=0
    gwx, gwy, gwz,                  # mirror DOWN weights, exact +/-1
    dpxr, dpxi, dpyr, dpyi, dpzr, dpzi,  # conjugate phases for DOWN
    upxr, upxi, upyr, upyi, upzr, upzi,  # forward phases for UP
    nx, ny, nz, n_elem,
    R01: tl.constexpr, R02: tl.constexpr,
    R11: tl.constexpr, R12: tl.constexpr,
    R21: tl.constexpr, R22: tl.constexpr,
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
    MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
    WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
    PML: tl.constexpr,
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Shared complex tensor-row core plus direct-store/PML constexpr tails."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1

    di, dj, dk = i - 1, j - 1, k - 1
    ui, uj, uk = i + 1, j + 1, k + 1
    dvx, dvy, dvz = live, live, live
    uvx, uvy, uvz = live, live, live
    if BCX == PERIODIC:
        di = tl.where(at_x, nx - 1, di)
        ui = tl.where(last_x, 0, ui)
    elif BCX == METALLIC:
        dvx = live & (~at_x)
        uvx = live & (~last_x)
    else:
        di = tl.where(at_x, MIRROR_ROW, di)
        uvx = live & (~last_x)
    if BCY == PERIODIC:
        dj = tl.where(at_y, ny - 1, dj)
        uj = tl.where(last_y, 0, uj)
    elif BCY == METALLIC:
        dvy = live & (~at_y)
        uvy = live & (~last_y)
    else:
        dj = tl.where(at_y, MIRROR_ROW, dj)
        uvy = live & (~last_y)
    if BCZ == PERIODIC:
        dk = tl.where(at_z, nz - 1, dk)
        uk = tl.where(last_z, 0, uk)
    elif BCZ == METALLIC:
        dvz = live & (~at_z)
        uvz = live & (~last_z)
    else:
        dk = tl.where(at_z, MIRROR_ROW, dk)
        uvz = live & (~last_z)

    # Ex row: own x; partners Dy/down-y then Dz/down-z.
    g_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    g_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    diag_re, diag_im = _mul_field_left(
        g_re, g_im, tl.load(e0 + idx, mask=live, other=0.0), EXPANSION)
    if R01:
        t0_re, t0_im = _complex_offdiag_term(
            g1, u01, idx, i * nyz + dj * nz + k,
            ui * nyz + j * nz + k, ui * nyz + dj * nz + k,
            live, dvy, uvx, uvx & dvy, at_y, last_x,
            dpyr, dpyi, upxr, upxi, gwy,
            DPH=PHY, UPH=PHX, MG=MG_Y, EXPANSION=EXPANSION)
        if R02:
            q_re, q_im = _complex_offdiag_term(
                g2, u02, idx, i * nyz + j * nz + dk,
                ui * nyz + j * nz + k, ui * nyz + j * nz + dk,
                live, dvz, uvx, uvx & dvz, at_z, last_x,
                dpzr, dpzi, upxr, upxi, gwz,
                DPH=PHZ, UPH=PHX, MG=MG_Z, EXPANSION=EXPANSION)
            t0_re, t0_im = t0_re + q_re, t0_im + q_im
        src0_re, src0_im = _complex_masked_row_sum(
            diag_re, diag_im, t0_re, t0_im, at_y, at_z, WM_Y, WM_Z)
    else:
        if R02:
            t0_re, t0_im = _complex_offdiag_term(
                g2, u02, idx, i * nyz + j * nz + dk,
                ui * nyz + j * nz + k, ui * nyz + j * nz + dk,
                live, dvz, uvx, uvx & dvz, at_z, last_x,
                dpzr, dpzi, upxr, upxi, gwz,
                DPH=PHZ, UPH=PHX, MG=MG_Z, EXPANSION=EXPANSION)
            src0_re, src0_im = _complex_masked_row_sum(
                diag_re, diag_im, t0_re, t0_im, at_y, at_z, WM_Y, WM_Z)
        else:
            src0_re, src0_im = diag_re, diag_im

    # Ey row: own y; partners Dz/down-z then Dx/down-x.
    g_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    g_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    diag_re, diag_im = _mul_field_left(
        g_re, g_im, tl.load(e1 + idx, mask=live, other=0.0), EXPANSION)
    if R11:
        t1_re, t1_im = _complex_offdiag_term(
            g2, u11, idx, i * nyz + j * nz + dk,
            i * nyz + uj * nz + k, i * nyz + uj * nz + dk,
            live, dvz, uvy, uvy & dvz, at_z, last_y,
            dpzr, dpzi, upyr, upyi, gwz,
            DPH=PHZ, UPH=PHY, MG=MG_Z, EXPANSION=EXPANSION)
        if R12:
            q_re, q_im = _complex_offdiag_term(
                g0, u12, idx, di * nyz + j * nz + k,
                i * nyz + uj * nz + k, di * nyz + uj * nz + k,
                live, dvx, uvy, uvy & dvx, at_x, last_y,
                dpxr, dpxi, upyr, upyi, gwx,
                DPH=PHX, UPH=PHY, MG=MG_X, EXPANSION=EXPANSION)
            t1_re, t1_im = t1_re + q_re, t1_im + q_im
        src1_re, src1_im = _complex_masked_row_sum(
            diag_re, diag_im, t1_re, t1_im, at_x, at_z, WM_X, WM_Z)
    else:
        if R12:
            t1_re, t1_im = _complex_offdiag_term(
                g0, u12, idx, di * nyz + j * nz + k,
                i * nyz + uj * nz + k, di * nyz + uj * nz + k,
                live, dvx, uvy, uvy & dvx, at_x, last_y,
                dpxr, dpxi, upyr, upyi, gwx,
                DPH=PHX, UPH=PHY, MG=MG_X, EXPANSION=EXPANSION)
            src1_re, src1_im = _complex_masked_row_sum(
                diag_re, diag_im, t1_re, t1_im, at_x, at_z, WM_X, WM_Z)
        else:
            src1_re, src1_im = diag_re, diag_im

    # Ez row: own z; partners Dx/down-x then Dy/down-y.
    g_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    g_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    diag_re, diag_im = _mul_field_left(
        g_re, g_im, tl.load(e2 + idx, mask=live, other=0.0), EXPANSION)
    if R21:
        t2_re, t2_im = _complex_offdiag_term(
            g0, u21, idx, di * nyz + j * nz + k,
            i * nyz + j * nz + uk, di * nyz + j * nz + uk,
            live, dvx, uvz, uvz & dvx, at_x, last_z,
            dpxr, dpxi, upzr, upzi, gwx,
            DPH=PHX, UPH=PHZ, MG=MG_X, EXPANSION=EXPANSION)
        if R22:
            q_re, q_im = _complex_offdiag_term(
                g1, u22, idx, i * nyz + dj * nz + k,
                i * nyz + j * nz + uk, i * nyz + dj * nz + uk,
                live, dvy, uvz, uvz & dvy, at_y, last_z,
                dpyr, dpyi, upzr, upzi, gwy,
                DPH=PHY, UPH=PHZ, MG=MG_Y, EXPANSION=EXPANSION)
            t2_re, t2_im = t2_re + q_re, t2_im + q_im
        src2_re, src2_im = _complex_masked_row_sum(
            diag_re, diag_im, t2_re, t2_im, at_x, at_y, WM_X, WM_Y)
    else:
        if R22:
            t2_re, t2_im = _complex_offdiag_term(
                g1, u22, idx, i * nyz + dj * nz + k,
                i * nyz + j * nz + uk, i * nyz + dj * nz + uk,
                live, dvy, uvz, uvz & dvy, at_y, last_z,
                dpyr, dpyi, upzr, upzi, gwy,
                DPH=PHY, UPH=PHZ, MG=MG_Y, EXPANSION=EXPANSION)
            src2_re, src2_im = _complex_masked_row_sum(
                diag_re, diag_im, t2_re, t2_im, at_x, at_y, WM_X, WM_Y)
        else:
            src2_re, src2_im = diag_re, diag_im

    if PML:
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        _complex_pml_tail(f0, w0, src0_re, src0_im, kp_0, km_0,
                          idx, live, EXPANSION)
        _complex_pml_tail(f1, w1, src1_re, src1_im, kp_1, km_1,
                          idx, live, EXPANSION)
        _complex_pml_tail(f2, w2, src2_re, src2_im, kp_2, km_2,
                          idx, live, EXPANSION)
    else:
        tl.store(f0 + 2 * idx, src0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, src0_im, mask=live)
        tl.store(f1 + 2 * idx, src1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, src1_im, mask=live)
        tl.store(f2 + 2 * idx, src2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, src2_im, mask=live)


def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """Mirror DOWN weights derived by the existing measured implementation."""
    return _folded_offdiag.mirror_ghost_weights(grid)


def _row_side_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    reasons: List[str] = []
    if not any(value is not None for value in _offdiag.row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: the element-wise "
            "complex constitutive family owns that configuration")
    reasons.extend(_offdiag._row_reasons(fields, shape))
    return reasons


def complex_no_pml_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> "_coverage.Coverage":
    """Coverage for Group J's complex/Bloch tensor row with no active PML."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    reasons = _complex._complex_grid_reasons(
        fields, pml, grid, probe, require_active_pml=False)
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_row_side_reasons(fields, shape))
    names = tuple(term[0] for term in E_TERMS) + tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_complex._complex_layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
    return _coverage.Coverage(not reasons, tuple(reasons))


def complex_folded_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> "_coverage.Coverage":
    """Coverage for Group D's folded complex tensor row under active PML."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    reasons, codes = _folded._complex_fold_offdiag_grid_reasons(
        fields, pml, grid, _coverage.CURL_TARGETS)
    if not _folded._has_real_fold(grid):
        reasons.append("no mirror plane is active: the unfolded PML run belongs "
                       "to a separate complex off-diagonal arm")
    elif codes is not None and not any(code in MIRROR_CODES for code in codes):
        reasons.append("no axis resolves to a mirror boundary")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append("special_kz beta with off-diagonal epsilon is not carried")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_row_side_reasons(fields, shape))
    names = (tuple(term[0] for term in E_TERMS)
             + tuple("f_w_" + term[0] for term in E_TERMS)
             + tuple(term[1] for term in E_TERMS))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_complex._complex_layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            reasons.extend(_coverage._coefficient_reasons(
                pml, shape, ("kps", "kms"), ("_h",)))
    # A folded row uses the +/-1 complex parity multiply on its DOWN ghost.
    reasons.extend(_folded._parity_expansion_reasons(probe))
    return _coverage.Coverage(not reasons, tuple(reasons))


def _resolve_expansion(probe: Any, folded: bool) -> Optional[int]:
    record = probe if probe is not None else _complex.load_expansion_probe()
    reasons = (_folded._parity_expansion_reasons(probe) if folded
               else _complex._expansion_reasons(probe))
    if reasons or record is None:
        return None
    if folded:
        return _folded.parity_expansion_license(record)["expansion"]
    return _complex.expansion_from_probe(record)


class ComplexOffdiagUpdateEPlan:
    """Allocation-free plan for the shared core and one of its two tails."""

    __slots__ = ("shape", "n_elem", "block", "pml", "expansion", "row_mask",
                 "boundary_codes", "phase_flags", "down_phases", "up_phases",
                 "mirror_axes", "ghost_weights", "wall_axes", "_targets",
                 "_aux", "_sources", "_inverse", "_rows", "_coefficients",
                 "_grid", "_kernel", "num_warps")

    def __init__(self, shape, *, pml: bool, expansion: int, targets, auxiliaries,
                 sources, inverse_epsilon, rows, coefficients, boundary_codes,
                 phases, ghost_weights, wall_axes, block=DEFAULT_BLOCK,
                 kernel=None, num_warps=None):
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.pml = bool(pml)
        self.expansion = int(expansion)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (0, 1, 2, 3) for code in self.boundary_codes):
            raise ValueError(f"boundary codes must be three values in 0..3, got "
                             f"{boundary_codes!r}")
        self.mirror_axes = tuple(int(code in MIRROR_CODES)
                                 for code in self.boundary_codes)
        self.ghost_weights = tuple(float(value) for value in ghost_weights)
        if len(self.ghost_weights) != 3 or any(
                self.mirror_axes[a] and self.ghost_weights[a] not in (-1.0, 1.0)
                for a in range(3)):
            raise ValueError("each mirrored axis needs an exact +/-1 ghost weight")
        self.wall_axes = tuple(int(value) for value in wall_axes)
        if len(self.wall_axes) != 3 or any(value not in (0, 1)
                                           for value in self.wall_axes):
            raise ValueError("wall axes must contain three 0/1 values")
        for axis in range(3):
            if self.mirror_axes[axis] and self.wall_axes[axis]:
                raise ValueError(f"axis {axis} is both mirrored and wall-masked")
            if self.mirror_axes[axis] and self.shape[axis] <= MIRROR_SOURCE_INDEX:
                raise ValueError(f"axis {axis} has {self.shape[axis]} cells but "
                                 f"the mirror ghost images row {MIRROR_SOURCE_INDEX}")

        targets, auxiliaries, sources = map(tuple, (targets, auxiliaries, sources))
        inverse_epsilon, rows = tuple(inverse_epsilon), tuple(rows)
        if not all(len(items) == 3 for items in
                   (targets, auxiliaries, sources, inverse_epsilon)):
            raise ValueError("targets, auxiliaries, sources and inverse epsilon "
                             "must each carry three components")
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(f"rows must carry {len(ROW_SLOTS)} slots")
        self.row_mask = tuple(int(value is not None) for value in rows)
        if not any(self.row_mask):
            raise ValueError("no off-diagonal row slot survives")

        output_values = targets + (auxiliaries if self.pml else ())
        input_values = sources + inverse_epsilon + tuple(
            value for value in rows if value is not None)
        output_addresses: Dict[int, str] = {}
        output_labels = tuple(f"E[{i}]" for i in range(3)) + (
            tuple(f"f_w_E[{i}]" for i in range(3)) if self.pml else ())
        for label, value in zip(output_labels, output_values):
            address = _offdiag._base_address(value)
            if address is not None and address in output_addresses:
                raise ValueError(f"output {label} aliases "
                                 f"{output_addresses[address]}")
            if address is not None:
                output_addresses[address] = label
        for index, value in enumerate(input_values):
            address = _offdiag._base_address(value)
            if address is not None and address in output_addresses:
                raise ValueError(f"input {index} aliases output "
                                 f"{output_addresses[address]}")

        # Dead row pointers remain typed but are compiled out.
        rows_bound = tuple(inverse_epsilon[i // 2] if value is None else value
                           for i, value in enumerate(rows))
        self._targets = tuple(CupyPointer(_complex._word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_complex._word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_complex._word_view(a)) for a in sources)
        self._inverse = tuple(CupyPointer(_flat(a)) for a in inverse_epsilon)
        self._rows = tuple(CupyPointer(_flat(a)) for a in rows_bound)
        coefficients = tuple(coefficients)
        if len(coefficients) != 6:
            raise ValueError("six kps/kms pointers are required (typed placeholders "
                             "are accepted for the no-PML constexpr tail)")
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)

        flags_u, self.up_phases = _complex._phase_arguments(phases, backward=False)
        flags_d, self.down_phases = _complex._phase_arguments(phases, backward=True)
        if flags_u != flags_d:
            raise ValueError("forward and backward phase flags disagree")
        self.phase_flags = flags_u
        for axis in range(3):
            if self.mirror_axes[axis] and self.phase_flags[axis]:
                raise ValueError(f"folded axis {axis} also carries a Bloch phase")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = complex_offdiag_update_e_step if kernel is None else kernel

    @property
    def launch_grid(self):
        return self._grid

    def run(self) -> None:
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        self._kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inverse,
            *self._rows, *self._coefficients, *self.ghost_weights,
            *self.down_phases, *self.up_phases, *self.shape, self.n_elem,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            PHX=self.phase_flags[0], PHY=self.phase_flags[1],
            PHZ=self.phase_flags[2],
            MG_X=self.mirror_axes[0], MG_Y=self.mirror_axes[1],
            MG_Z=self.mirror_axes[2],
            WM_X=self.wall_axes[0], WM_Y=self.wall_axes[1], WM_Z=self.wall_axes[2],
            PML=int(self.pml), EXPANSION=self.expansion, BLOCK=self.block, **extra)


def _plan(fields: Any, pml: Any, *, folded: bool, block: Optional[int],
          num_warps: Optional[int], probe: Any) -> Optional[ComplexOffdiagUpdateEPlan]:
    verdict = (complex_folded_offdiag_update_e_coverage(fields, pml, probe)
               if folded else
               complex_no_pml_offdiag_update_e_coverage(fields, pml, probe))
    if not verdict.covered:
        return None
    expansion = _resolve_expansion(probe, folded)
    if expansion is None:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    kinds = _boundary_kinds(grid, pml if folded else None)
    if folded:
        codes, reasons = _folded.folded_axis_kinds(grid, pml)
        if codes is None or reasons:  # pragma: no cover - predicate owns this
            return None
    else:
        codes = tuple(CODE_METALLIC if kind == "metallic" else CODE_PERIODIC
                      for kind in kinds)
    phases = _complex.bloch_phase_table(grid, kinds)
    targets = [getattr(fields, name) for name in ("Ex", "Ey", "Ez")]
    auxiliaries = ([getattr(fields, "f_w_" + name) for name in ("Ex", "Ey", "Ez")]
                   if folded else targets)
    sources = [getattr(fields, name) for name in ("Dx", "Dy", "Dz")]
    inverse = [fields.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez")]
    rows = _offdiag.row_volumes_for(fields)
    coefficients = ([getattr(pml, f"{stem}_{axis}_h")
                     for axis in "xyz" for stem in ("kps", "kms")]
                    if folded else [inverse[i // 2] for i in range(6)])
    return ComplexOffdiagUpdateEPlan(
        grid.shape, pml=folded, expansion=expansion,
        targets=targets, auxiliaries=auxiliaries, sources=sources,
        inverse_epsilon=inverse, rows=rows, coefficients=coefficients,
        boundary_codes=codes, phases=phases,
        ghost_weights=(mirror_ghost_weights(grid) if folded else (1.0, 1.0, 1.0)),
        wall_axes=_offdiag.wall_mask_axes(grid),
        block=DEFAULT_BLOCK if block is None else block, num_warps=num_warps)


def plan_complex_no_pml_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        num_warps: Optional[int] = None,
        probe: Any = None) -> Optional[ComplexOffdiagUpdateEPlan]:
    return _plan(fields, pml, folded=False, block=block,
                 num_warps=num_warps, probe=probe)


def plan_complex_folded_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        num_warps: Optional[int] = None,
        probe: Any = None) -> Optional[ComplexOffdiagUpdateEPlan]:
    return _plan(fields, pml, folded=True, block=block,
                 num_warps=num_warps, probe=probe)


def plan_complex_offdiag_update_e_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], rows: Dict[str, Dict[str, Any]],
        *, pml: bool, boundary_codes: Sequence[int],
        phases: Sequence[Optional[complex]], ghost_weights: Sequence[float],
        wall_axes: Sequence[int], expansion: int, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = None,
        ) -> ComplexOffdiagUpdateEPlan:
    """Bare-array builder used by the CUDA byte gate and its mutations."""
    shape = tuple(int(n) for n in arrays["Ex"].shape)
    row_values = tuple((rows.get(row, {}) or {}).get(partner)
                       for row, partner in ROW_SLOTS)
    inverse = [arrays["inv_eps_" + name] for name in ("Ex", "Ey", "Ez")]
    coefficients = ([flat[f"{stem}_{axis}"]
                     for axis in "xyz" for stem in ("kps", "kms")]
                    if pml else [inverse[i // 2] for i in range(6)])
    return ComplexOffdiagUpdateEPlan(
        shape, pml=pml, expansion=expansion,
        targets=[arrays[name] for name in ("Ex", "Ey", "Ez")],
        auxiliaries=([arrays["f_w_" + name] for name in ("Ex", "Ey", "Ez")]
                     if pml else [arrays[name] for name in ("Ex", "Ey", "Ez")]),
        sources=[arrays[name] for name in ("Dx", "Dy", "Dz")],
        inverse_epsilon=inverse, rows=row_values, coefficients=coefficients,
        boundary_codes=boundary_codes, phases=phases,
        ghost_weights=ghost_weights, wall_axes=wall_axes,
        block=DEFAULT_BLOCK if block is None else block,
        kernel=kernel, num_warps=num_warps)
