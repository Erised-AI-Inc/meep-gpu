"""Complex64 off-diagonal electric constitutive update with two storage tails.

This family closes the tensor-row intersection left between the complex and
off-diagonal Triton products.  The numerical core is shared by both residual
shapes:

* a folded, active-PML tail stores ``f_w`` and accumulates ``E`` with the
  half-integer ``kps``/``kms`` vectors;
* an unfolded, inactive-PML tail stores the row product directly into ``E``.

The core transcribes ``stepping._offdiagonal_terms`` and ``_rounded_row`` on each
stored part, as MEEP's ``DOCMP`` loop steps ``f[c][0]`` and ``f[c][1]`` with real
operands: per component, MEEP's row ``gs*us`` is widened to double; each surviving
partner term forms the partner-axis DOWN pair, multiplies it by the real tensor
coefficient, and adds the product shifted UP the row component's own axis, in
float32; the wall mask, MEEP's double ``0.25`` and the slot's per-chunk presence are
applied as the term is added in double (``offdiag_update_e._row_add``), offset 1 then
offset 2; and the row is rounded to float32 once. On a Bloch-periodic axis the
not-owned word is MEEP's connection of the stored window
(:func:`.complex_fields._connect`): ``eikna`` above it on the partner's axis, for the
pair, and ``conj(eikna)`` below it on the component's own axis, for the near product.
Mirror parity is applied only to the partner-axis DOWN ghost; the own-axis UP ghost
is zero, as the array path calls ``_shift_up`` without a component or reflect row.
The PML tail is :func:`.forms.constitutive_form` on the chunk record's half-integer
line of the component's own axis, with ``f_w`` read and written on the
accumulation's cells only.

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

# The forms, the record layout, the double row and the connection, read by the kernel
# below. Imported only when Triton is: the modules that define them import it at
# module scope, and this module's host route must answer without it.
if _complex._TRITON_IMPORT_ERROR is None:
    from .complex_fields import _connect  # noqa: E402
    from .forms import constitutive_form, record_in  # noqa: E402
    from .kernels import RECORD_HALF_INTEGER  # noqa: E402
    from .offdiag_update_e import PRESENCE_LINES, _row_add  # noqa: E402

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
    at_p, at_o, w_d,
    par, pai, obr, obi,
    CNP: tl.constexpr, CNO: tl.constexpr, MG: tl.constexpr,
):
    """One partner's OFFDIAG sum before its scale, per part, in float32.

    ``stepping._offdiagonal_terms`` on complex storage::

        pair    = connect_above(g) + shift_down(g)          # partner axis
        product = u * pair                                   # per part
        summed  = connect_below(product) + shift_up(product) # own axis

    ``connect_above`` is MEEP's not-owned word ABOVE the stored window on the
    partner's axis (``eikna``, :func:`.complex_fields._connect`) at stored partner
    index 0, for the near pair's ``g[i]`` and the far pair's ``g[i+s]`` alike (both
    sit on the same partner index); ``shift_down`` is a plain roll on a periodic axis,
    the mirror ghost ``w * g[MIRROR_ROW]`` on a folded one (``MG``; ``w_d`` is the
    per-lane weight, the plane's parity on the ghost lane and 1.0 elsewhere, so the
    multiply is exact) and a zero past a wall. The coefficient multiplies each part
    BETWEEN the shifts (the near pair takes ``u`` at its own node, the far pair at the
    next node up the component's own axis). ``connect_below`` is the not-owned word
    BELOW the window on the component's OWN axis (``conj(eikna)``) at stored own index
    0, applied to the near PRODUCT; the far product is a plain roll of the product,
    unconnected. The ``0.25`` is MEEP's double literal and is applied per part by
    ``offdiag_update_e._row_add``, in double.
    """
    c_re = tl.load(g + 2 * o_c, mask=v_c, other=0.0)
    c_im = tl.load(g + 2 * o_c + 1, mask=v_c, other=0.0)
    d_re = tl.load(g + 2 * o_d, mask=v_d, other=0.0)
    d_im = tl.load(g + 2 * o_d + 1, mask=v_d, other=0.0)
    p_re = tl.load(g + 2 * o_u, mask=v_u, other=0.0)
    p_im = tl.load(g + 2 * o_u + 1, mask=v_u, other=0.0)
    q_re = tl.load(g + 2 * o_ud, mask=v_ud, other=0.0)
    q_im = tl.load(g + 2 * o_ud + 1, mask=v_ud, other=0.0)
    if CNP:
        r_re, r_im = _connect(c_re, c_im, par, pai, CNP)
        c_re = tl.where(at_p, r_re, c_re)
        c_im = tl.where(at_p, r_im, c_im)
        r_re, r_im = _connect(p_re, p_im, par, pai, CNP)
        p_re = tl.where(at_p, r_re, p_re)
        p_im = tl.where(at_p, r_im, p_im)
    if MG:
        d_re = w_d * d_re
        d_im = w_d * d_im
        q_re = w_d * q_re
        q_im = w_d * q_im
    u_c = tl.load(u + o_c, mask=v_c, other=0.0)
    u_u = tl.load(u + o_u, mask=v_u, other=0.0)
    near_re = (c_re + d_re) * u_c
    near_im = (c_im + d_im) * u_c
    far_re = (p_re + q_re) * u_u
    far_im = (p_im + q_im) * u_u
    if CNO:
        r_re, r_im = _connect(near_re, near_im, obr, obi, CNO)
        near_re = tl.where(at_o, r_re, near_re)
        near_im = tl.where(at_o, r_im, near_im)
    return near_re + far_re, near_im + far_im


@triton.jit
def _complex_pml_tail(f, w, sr, si, kp, km, dw, idx, live):
    """One component of ``_apply_constitutive_regions``, per part.

    MEEP's accumulation on the cells whose chunk carries the component's own layer,
    ``E = src`` elsewhere (:func:`.forms.constitutive_form`); ``f_w`` read and written
    on the accumulation's cells only, ``prev`` before the store.
    """
    prev_re = tl.load(w + 2 * idx, mask=live & dw, other=0.0)
    prev_im = tl.load(w + 2 * idx + 1, mask=live & dw, other=0.0)
    tl.store(w + 2 * idx, sr, mask=live & dw)
    tl.store(w + 2 * idx + 1, si, mask=live & dw)
    a_re = tl.load(f + 2 * idx, mask=live & dw, other=0.0)
    a_im = tl.load(f + 2 * idx + 1, mask=live & dw, other=0.0)
    tl.store(f + 2 * idx, constitutive_form(a_re, prev_re, sr, kp, km, dw), mask=live)
    tl.store(f + 2 * idx + 1, constitutive_form(a_im, prev_im, si, kp, km, dw),
             mask=live)


@triton.jit
def complex_offdiag_update_e_step(
    f0, f1, f2,                     # Ex/Ey/Ez complex64 word views
    w0, w1, w2,                     # f_w_Ex/Ey/Ez, unused when PML=0
    g0, g1, g2,                     # Dx/Dy/Dz complex64 word views
    e0, e1, e2,                     # diagonal inverse epsilon, real f32
    u01, u02, u11, u12, u21, u22,  # six real off-diagonal row volumes
    kp0, km0, kp1, km1, kp2, km2,  # half-integer PML vectors, dead for PML=0
    gwx, gwy, gwz,                  # mirror DOWN weights, exactly +1.0 / -1.0
    caxr, caxi, cayr, cayi, cazr, cazi,  # connection ABOVE the window (eikna)
    cbxr, cbxi, cbyr, cbyi, cbzr, cbzi,  # connection BELOW the window (conj eikna)
    cb,                             # chunk record: each component's dsigw membership
    op,                             # off-diagonal presence: slot masks, segment lines
    nx, ny, nz, n_elem,
    R01: tl.constexpr, R02: tl.constexpr,
    R11: tl.constexpr, R12: tl.constexpr,
    R21: tl.constexpr, R22: tl.constexpr,
    P01: tl.constexpr, P02: tl.constexpr,
    P11: tl.constexpr, P12: tl.constexpr,
    P21: tl.constexpr, P22: tl.constexpr,
    PRESENCE: tl.constexpr,
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,
    MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
    WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
    PML: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``stepping.update_E`` with off-diagonal rows on complex storage, per part.

    :func:`.offdiag_update_e.offdiag_constitutive_step` (unfolded) and
    :func:`.folded_offdiag_update_e.folded_offdiag_constitutive_step` (folded) on
    each stored part, plus the Bloch connection of the stored window
    (:func:`_complex_offdiag_term`). Per component and part: MEEP's row ``gs*us``
    widened to double, each surviving OFFDIAG term added in double where its chunk
    keeps the slot's array (``_row_add``: the wall mask, the double ``0.25``, the
    presence), rounded once to float32 (``stepping._rounded_row``). The tail is the
    PML one (``PML=1``: :func:`.forms.constitutive_form` on the record's
    half-integer dsigw line, ``f_w`` read and written on the accumulation's cells
    only) or the absorber-free one (``E = src``).
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny
    at_x, at_y, at_z = i == 0, j == 0, k == 0

    # --- per-axis neighbour indices, both directions, with the ghost rule.
    # stepping._shift_down / _shift_up: PERIODIC wraps (a plain roll, phased or
    # not); METALLIC masks both faces; MIRROR redirects the DOWN ghost to stored row
    # MIRROR_ROW with the plane's weight and masks the UP ghost (``_shift_up`` is
    # called without a component, so both mirror terminations serve zero).
    di, dj, dk = i - 1, j - 1, k - 1
    ui, uj, uk = i + 1, j + 1, k + 1
    dvx, dvy, dvz = live, live, live
    uvx, uvy, uvz = live, live, live
    wx, wy, wz = 1.0, 1.0, 1.0
    if BCX == PERIODIC:
        di = tl.where(di < 0, nx - 1, di)
        ui = tl.where(ui == nx, 0, ui)
    elif BCX == METALLIC:
        dvx = live & (di >= 0)
        uvx = live & (ui < nx)
    else:  # MIRROR_METALLIC or MIRROR_PERIODIC
        di = tl.where(at_x, MIRROR_ROW, di)
        wx = tl.where(at_x, gwx, 1.0)
        uvx = live & (ui < nx)
    if BCY == PERIODIC:
        dj = tl.where(dj < 0, ny - 1, dj)
        uj = tl.where(uj == ny, 0, uj)
    elif BCY == METALLIC:
        dvy = live & (dj >= 0)
        uvy = live & (uj < ny)
    else:
        dj = tl.where(at_y, MIRROR_ROW, dj)
        wy = tl.where(at_y, gwy, 1.0)
        uvy = live & (uj < ny)
    if BCZ == PERIODIC:
        dk = tl.where(dk < 0, nz - 1, dk)
        uk = tl.where(uk == nz, 0, uk)
    elif BCZ == METALLIC:
        dvz = live & (dk >= 0)
        uvz = live & (uk < nz)
    else:
        dk = tl.where(at_z, MIRROR_ROW, dk)
        wz = tl.where(at_z, gwz, 1.0)
        uvz = live & (uk < nz)

    # Which chunk of the {low, interior, high} product owns the cell, per component
    # lattice (offdiag_update_e.presence_record's segment lines).
    c0, c1, c2 = i * 0, i * 0, i * 0
    if PRESENCE:
        sxi = tl.load(op + PRESENCE_LINES + i, mask=live, other=0)
        syi = tl.load(op + PRESENCE_LINES + nx + j, mask=live, other=0)
        szi = tl.load(op + PRESENCE_LINES + nx + ny + k, mask=live, other=0)
        half = PRESENCE_LINES + nx + ny + nz
        sxh = tl.load(op + half + i, mask=live, other=0)
        syh = tl.load(op + half + nx + j, mask=live, other=0)
        szh = tl.load(op + half + nx + ny + k, mask=live, other=0)
        c0 = (sxh * 9 + syi * 3) + szi     # Ex: half-integer on x
        c1 = (sxi * 9 + syh * 3) + szi     # Ey: half-integer on y
        c2 = (sxi * 9 + syi * 3) + szh     # Ez: half-integer on z

    # --- Ex row: own x; partners Dy (down y) then Dz (down z) -------------------
    gs_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    gs_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    us = tl.load(e0 + idx, mask=live, other=0.0)
    if R01:
        row_re = tl.cast(gs_re * us, tl.float64)
        row_im = tl.cast(gs_im * us, tl.float64)
        s_re, s_im = _complex_offdiag_term(
            g1, u01, idx, i * nyz + dj * nz + k,
            ui * nyz + j * nz + k, ui * nyz + dj * nz + k,
            live, dvy, uvx, uvx & dvy, at_y, at_x, wy,
            cayr, cayi, cbxr, cbxi, CNP=CNY, CNO=CNX, MG=MG_Y)
        row_re = _row_add(row_re, s_re, at_y, at_z, op, c0, 0, WM_Y, WM_Z, P01)
        row_im = _row_add(row_im, s_im, at_y, at_z, op, c0, 0, WM_Y, WM_Z, P01)
        if R02:
            s_re, s_im = _complex_offdiag_term(
                g2, u02, idx, i * nyz + j * nz + dk,
                ui * nyz + j * nz + k, ui * nyz + j * nz + dk,
                live, dvz, uvx, uvx & dvz, at_z, at_x, wz,
                cazr, cazi, cbxr, cbxi, CNP=CNZ, CNO=CNX, MG=MG_Z)
            row_re = _row_add(row_re, s_re, at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
            row_im = _row_add(row_im, s_im, at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
        src0_re = tl.cast(row_re, tl.float32)
        src0_im = tl.cast(row_im, tl.float32)
    else:
        if R02:
            row_re = tl.cast(gs_re * us, tl.float64)
            row_im = tl.cast(gs_im * us, tl.float64)
            s_re, s_im = _complex_offdiag_term(
                g2, u02, idx, i * nyz + j * nz + dk,
                ui * nyz + j * nz + k, ui * nyz + j * nz + dk,
                live, dvz, uvx, uvx & dvz, at_z, at_x, wz,
                cazr, cazi, cbxr, cbxi, CNP=CNZ, CNO=CNX, MG=MG_Z)
            row_re = _row_add(row_re, s_re, at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
            row_im = _row_add(row_im, s_im, at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
            src0_re = tl.cast(row_re, tl.float32)
            src0_im = tl.cast(row_im, tl.float32)
        else:
            src0_re = gs_re * us
            src0_im = gs_im * us

    # --- Ey row: own y; partners Dz (down z) then Dx (down x) -------------------
    gs_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    gs_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    us = tl.load(e1 + idx, mask=live, other=0.0)
    if R11:
        row_re = tl.cast(gs_re * us, tl.float64)
        row_im = tl.cast(gs_im * us, tl.float64)
        s_re, s_im = _complex_offdiag_term(
            g2, u11, idx, i * nyz + j * nz + dk,
            i * nyz + uj * nz + k, i * nyz + uj * nz + dk,
            live, dvz, uvy, uvy & dvz, at_z, at_y, wz,
            cazr, cazi, cbyr, cbyi, CNP=CNZ, CNO=CNY, MG=MG_Z)
        row_re = _row_add(row_re, s_re, at_x, at_z, op, c1, 2, WM_X, WM_Z, P11)
        row_im = _row_add(row_im, s_im, at_x, at_z, op, c1, 2, WM_X, WM_Z, P11)
        if R12:
            s_re, s_im = _complex_offdiag_term(
                g0, u12, idx, di * nyz + j * nz + k,
                i * nyz + uj * nz + k, di * nyz + uj * nz + k,
                live, dvx, uvy, uvy & dvx, at_x, at_y, wx,
                caxr, caxi, cbyr, cbyi, CNP=CNX, CNO=CNY, MG=MG_X)
            row_re = _row_add(row_re, s_re, at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
            row_im = _row_add(row_im, s_im, at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
        src1_re = tl.cast(row_re, tl.float32)
        src1_im = tl.cast(row_im, tl.float32)
    else:
        if R12:
            row_re = tl.cast(gs_re * us, tl.float64)
            row_im = tl.cast(gs_im * us, tl.float64)
            s_re, s_im = _complex_offdiag_term(
                g0, u12, idx, di * nyz + j * nz + k,
                i * nyz + uj * nz + k, di * nyz + uj * nz + k,
                live, dvx, uvy, uvy & dvx, at_x, at_y, wx,
                caxr, caxi, cbyr, cbyi, CNP=CNX, CNO=CNY, MG=MG_X)
            row_re = _row_add(row_re, s_re, at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
            row_im = _row_add(row_im, s_im, at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
            src1_re = tl.cast(row_re, tl.float32)
            src1_im = tl.cast(row_im, tl.float32)
        else:
            src1_re = gs_re * us
            src1_im = gs_im * us

    # --- Ez row: own z; partners Dx (down x) then Dy (down y) -------------------
    gs_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    gs_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    us = tl.load(e2 + idx, mask=live, other=0.0)
    if R21:
        row_re = tl.cast(gs_re * us, tl.float64)
        row_im = tl.cast(gs_im * us, tl.float64)
        s_re, s_im = _complex_offdiag_term(
            g0, u21, idx, di * nyz + j * nz + k,
            i * nyz + j * nz + uk, di * nyz + j * nz + uk,
            live, dvx, uvz, uvz & dvx, at_x, at_z, wx,
            caxr, caxi, cbzr, cbzi, CNP=CNX, CNO=CNZ, MG=MG_X)
        row_re = _row_add(row_re, s_re, at_x, at_y, op, c2, 4, WM_X, WM_Y, P21)
        row_im = _row_add(row_im, s_im, at_x, at_y, op, c2, 4, WM_X, WM_Y, P21)
        if R22:
            s_re, s_im = _complex_offdiag_term(
                g1, u22, idx, i * nyz + dj * nz + k,
                i * nyz + j * nz + uk, i * nyz + dj * nz + uk,
                live, dvy, uvz, uvz & dvy, at_y, at_z, wy,
                cayr, cayi, cbzr, cbzi, CNP=CNY, CNO=CNZ, MG=MG_Y)
            row_re = _row_add(row_re, s_re, at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
            row_im = _row_add(row_im, s_im, at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
        src2_re = tl.cast(row_re, tl.float32)
        src2_im = tl.cast(row_im, tl.float32)
    else:
        if R22:
            row_re = tl.cast(gs_re * us, tl.float64)
            row_im = tl.cast(gs_im * us, tl.float64)
            s_re, s_im = _complex_offdiag_term(
                g1, u22, idx, i * nyz + dj * nz + k,
                i * nyz + j * nz + uk, i * nyz + dj * nz + uk,
                live, dvy, uvz, uvz & dvy, at_y, at_z, wy,
                cayr, cayi, cbzr, cbzi, CNP=CNY, CNO=CNZ, MG=MG_Y)
            row_re = _row_add(row_re, s_re, at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
            row_im = _row_add(row_im, s_im, at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
            src2_re = tl.cast(row_re, tl.float32)
            src2_im = tl.cast(row_im, tl.float32)
        else:
            src2_re = gs_re * us
            src2_im = gs_im * us

    if PML:
        # Each component's own (dsigw) layer on the half-integer sub-lattice.
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        _complex_pml_tail(f0, w0, src0_re, src0_im, kp_0, km_0, dw0, idx, live)
        _complex_pml_tail(f1, w1, src1_re, src1_im, kp_1, km_1, dw1, idx, live)
        _complex_pml_tail(f2, w2, src2_re, src2_im, kp_2, km_2, dw2, idx, live)
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
        fields: Any, pml: Any) -> "_coverage.Coverage":
    """Coverage for Group J's complex/Bloch tensor row with no active PML."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    reasons = _complex._complex_grid_reasons(
        fields, pml, grid, require_active_pml=False)
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_row_side_reasons(fields, shape))
    names = tuple(term[0] for term in E_TERMS) + tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_complex._complex_layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
    # The per-chunk presence of every surviving row slot, packed for the kernel.
    if not reasons:
        reasons.extend(_offdiag._presence_reasons(fields, None))
    return _coverage.Coverage(not reasons, tuple(reasons))


def complex_folded_offdiag_update_e_coverage(
        fields: Any, pml: Any) -> "_coverage.Coverage":
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
    # The per-chunk presence of every surviving row slot, packed for the kernel.
    if not reasons:
        reasons.extend(_offdiag._presence_reasons(fields, pml))
    return _coverage.Coverage(not reasons, tuple(reasons))


class ComplexOffdiagUpdateEPlan:
    """Allocation-free plan for the shared core and one of its two tails.

    ``chunk_bounds`` (``launch.bind_chunk_bounds``) and ``presence``
    (``offdiag_update_e.bind_presence``) are required: the PML tail keys its form on
    the record, and every surviving term is added only where its chunk keeps the
    slot's array. Every product here is formed part by part.
    """

    __slots__ = ("shape", "n_elem", "block", "pml", "row_mask",
                 "boundary_codes", "connections", "above_values", "below_values",
                 "mirror_axes", "ghost_weights", "wall_axes", "chunk_bounds",
                 "presence", "presence_varies", "_targets", "_aux", "_sources",
                 "_inverse", "_rows", "_coefficients", "_cb", "_op", "_grid",
                 "_kernel", "num_warps")

    def __init__(self, shape, *, pml: bool, targets, auxiliaries,
                 sources, inverse_epsilon, rows, coefficients, boundary_codes,
                 phases, ghost_weights, wall_axes, chunk_bounds, presence,
                 block=DEFAULT_BLOCK, kernel=None, num_warps=None):
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.pml = bool(pml)
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

        # MEEP's connection of each axis's not-owned words: ABOVE the stored window
        # (eikna; the partner pair's same-index read) and BELOW it (conj(eikna); the
        # near product's own-axis read). The two tables classify every axis alike.
        kinds_above, self.above_values = _complex.connection_arguments(phases,
                                                                       backward=True)
        kinds_below, self.below_values = _complex.connection_arguments(phases,
                                                                       backward=False)
        if kinds_above != kinds_below:  # pragma: no cover - conj keeps the class
            raise ValueError("the two connection directions classify an axis apart")
        self.connections = kinds_above
        for axis in range(3):
            if self.mirror_axes[axis] and self.connections[axis]:
                raise ValueError(f"folded axis {axis} also carries a Bloch phase")
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexOffdiagUpdateEPlan")
        self.presence_varies, self.presence, self._op = _offdiag.bind_presence(
            presence, self.shape, self.row_mask, targets[0],
            "ComplexOffdiagUpdateEPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = complex_offdiag_update_e_step if kernel is None else kernel

    @property
    def launch_grid(self):
        return self._grid

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        varies = self.presence_varies
        self._kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inverse,
            *self._rows, *self._coefficients, *self.ghost_weights,
            *self.above_values, *self.below_values, self._cb, self._op,
            *self.shape, self.n_elem,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            P01=varies[0], P02=varies[1], P11=varies[2], P12=varies[3],
            P21=varies[4], P22=varies[5], PRESENCE=int(any(varies)),
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            MG_X=self.mirror_axes[0], MG_Y=self.mirror_axes[1],
            MG_Z=self.mirror_axes[2],
            WM_X=self.wall_axes[0], WM_Y=self.wall_axes[1], WM_Z=self.wall_axes[2],
            PML=int(self.pml), BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra)

    def __repr__(self) -> str:
        return (f"ComplexOffdiagUpdateEPlan(shape={self.shape}, pml={self.pml}, "
                f"rows={self.row_mask}, presence={self.presence_varies}, "
                f"bc={self.boundary_codes}, connections={self.connections}, "
                f"block={self.block})")


def _plan(fields: Any, pml: Any, *, folded: bool, block: Optional[int],
          num_warps: Optional[int],
          chunk_layout: Any) -> Optional[ComplexOffdiagUpdateEPlan]:
    verdict = (complex_folded_offdiag_update_e_coverage(fields, pml)
               if folded else
               complex_no_pml_offdiag_update_e_coverage(fields, pml))
    if not verdict.covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

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
        grid.shape, pml=folded,
        targets=targets, auxiliaries=auxiliaries, sources=sources,
        inverse_epsilon=inverse, rows=rows, coefficients=coefficients,
        boundary_codes=codes, phases=phases,
        ghost_weights=(mirror_ghost_weights(grid) if folded else (1.0, 1.0, 1.0)),
        wall_axes=_offdiag.wall_mask_axes(grid),
        chunk_bounds=chunk_bounds_for(fields, pml if folded else None, chunk_layout),
        presence=_offdiag.presence_for(fields, pml if folded else None),
        block=DEFAULT_BLOCK if block is None else block, num_warps=num_warps)


def plan_complex_no_pml_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        num_warps: Optional[int] = None,
        chunk_layout: Any = None) -> Optional[ComplexOffdiagUpdateEPlan]:
    """The absorber-free complex row plan on the run's chunk division (presence)."""
    return _plan(fields, pml, folded=False, block=block,
                 num_warps=num_warps, chunk_layout=chunk_layout)


def plan_complex_folded_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        num_warps: Optional[int] = None,
        chunk_layout: Any = None) -> Optional[ComplexOffdiagUpdateEPlan]:
    """The folded complex PML row plan on the run's chunk division."""
    return _plan(fields, pml, folded=True, block=block,
                 num_warps=num_warps, chunk_layout=chunk_layout)


def plan_complex_offdiag_update_e_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], rows: Dict[str, Dict[str, Any]],
        *, pml: bool, boundary_codes: Sequence[int],
        phases: Sequence[Optional[complex]], ghost_weights: Sequence[float],
        wall_axes: Sequence[int], chunk_bounds: Any, presence: Any,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = None,
        ) -> ComplexOffdiagUpdateEPlan:
    """Bare-array builder used by the byte gates and their mutations.

    ``chunk_bounds`` (``chunk_map.chunk_bounds_record``) and ``presence``
    (``offdiag_update_e.presence_record`` or ``offdiag_update_e.ALL_PRESENT``) have
    no default: a gate whose plan fell back to some default division would stop
    measuring the forms it exists to measure.
    """
    shape = tuple(int(n) for n in arrays["Ex"].shape)
    row_values = tuple((rows.get(row, {}) or {}).get(partner)
                       for row, partner in ROW_SLOTS)
    inverse = [arrays["inv_eps_" + name] for name in ("Ex", "Ey", "Ez")]
    coefficients = ([flat[f"{stem}_{axis}"]
                     for axis in "xyz" for stem in ("kps", "kms")]
                    if pml else [inverse[i // 2] for i in range(6)])
    return ComplexOffdiagUpdateEPlan(
        shape, pml=pml,
        targets=[arrays[name] for name in ("Ex", "Ey", "Ez")],
        auxiliaries=([arrays["f_w_" + name] for name in ("Ex", "Ey", "Ez")]
                     if pml else [arrays[name] for name in ("Ex", "Ey", "Ez")]),
        sources=[arrays[name] for name in ("Dx", "Dy", "Dz")],
        inverse_epsilon=inverse, rows=row_values, coefficients=coefficients,
        boundary_codes=boundary_codes, phases=phases,
        ghost_weights=ghost_weights, wall_axes=wall_axes,
        chunk_bounds=chunk_bounds, presence=presence,
        block=DEFAULT_BLOCK if block is None else block,
        kernel=kernel, num_warps=num_warps)
