"""The PML ``step_D`` welded into OFF-DIAGONAL ``update_E`` — one launch, scratch output.

THIS IS THE CELL THE 2026-08-20 ROUND REFUSED, BUILT A DIFFERENT WAY.

    ceiling 8  (of 16 rows)  D->E  PML -> off-diagonal

``results/triton_fused_offdiag_electric_2026-08-20`` measured, on an RTX A6000
under both float32 subnormal policies, that a ONE-ORDINARY-LAUNCH weld of the two
SHIPPED IN-PLACE halves disagrees with its two-launch reference on 42 of 60
subject cases at up to 6.58e-02, schedule-dependently. That measurement stands.
Its subject stepped D IN PLACE while the off-diagonal constitutive arm read its
neighbours' D out of the same volume, and every one of its four control legs
(pointwise C1, zero-coefficient C2, curl-is-identity C3, planted-pointwise P)
came back bit-identical — which is exactly the finding that the STENCIL, and
nothing else in the weld, was the defect.

This product removes the stencil's premise instead of arguing with it. The curl
half writes ``D_new``/``fu_new`` into LAUNCH-LOCAL SCRATCH, so ``Dx``/``fu_Dx``/
``Hx`` are pre-launch state for the whole dispatch; the constitutive half takes
its own cell's displacement from a register and RE-DERIVES each of its at most
twelve foreign taps from that pre-launch state through the same inlined
:func:`_step_cell`; and :class:`OffdiagFusedElectricPairPlan` rotates the D/fu
references after the launch returns. NOTHING WRITTEN IS EVER READ, so there is no
cross-program hazard left for a block schedule to expose — which is what the
gate's S2 leg (identical at every block size, where the 2026-08-20 subject went
from 4384 differing words at BLOCK=64 to 0 at BLOCK=1024) is there to measure.

WHAT ONE LAUNCH REPLACES. Three driver call sites, in driver order:
``step_D`` (driver.py:3302), ``zero_metal_D`` (:3310) and ``update_E`` (:3313).
``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` are REFUSED rather than
carried — this family is unfolded, both passes are no-ops on every row it serves,
and the folded family that does carry the first of them is
``folded_offdiag_fused_electric_pair``. The separate composition is two whole-grid
launches plus an array-path wall pass.

THE SEAM ARITHMETIC IS A CLOSED FORM, MEASURED BEFORE IT WAS BUILT. On this
family the form degenerates all the way to ``clear ? +0 : v``, because the eight
corpus rows this cell reaches declare ``zero_metal_D`` as their ONLY live in-seam
pass (measured per row off the census in
``results/fusion_matrix_triton_2026-09-02_residue``; seven of the eight are the
``TestCavityArraySlice`` family and ``examples:cavity_arrayslice.py``). The
general form — near mirror redirect, far reflect row, the clear inside the far
parity and outside the near one — was measured byte-identical to the driver's
pass order over ten fixtures and two complete steps in
``results/metal_scratch_weld_closed_form_2026-09-01/probe.json``; that is another
backend's artifact and certifies nothing here, but it is why this lane built the
degenerate case first and the folded one second.

WHY THE DEPOSIT REPAIR IS REFUSED BY NAME, AND WHY THAT IS NOT MERELY UNWIRED.
:data:`CARRIES_DEPOSIT_REPAIR` is False. ``deposit_repair`` inverts a
constitutive recurrence AT THE DEPOSIT CELL: it recomputes ``E[c]`` from the
injected ``D[c]``. That inversion is POINTWISE, and this constitutive half is
not — a deposit at cell ``c`` moves the off-diagonal coupling of up to four
cells (``c`` itself, ``c`` shifted up the partner's axis, ``c`` shifted down the
component's own axis, and the corner), so a per-cell repair would leave three of
them stepped from a pre-injection displacement. The refusal is therefore about
the SHAPE of the repair and not about wiring: making this cell serve its eight
in-seam-source rows needs a repair whose footprint is the stencil's, which does
not exist. Eight of the sixteen corpus rows on this cell declare an in-seam
electric source and are refused for exactly that reason; the other eight are what
this product serves.

NOT WIRED. ``fastpath`` is untouched; dispatch is a separate milestone. The
callers are the gate (``parity/meep_gpu/gate_triton_offdiag_stencil_welds.py``),
the probe (``parity/meep_gpu/probe_triton_offdiag_scratch_weld.py``) and the
laptop tests.

Import contract: importable WITHOUT Triton — the predicate and the plan builders
(to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .. import deposit_repair as _deposit_repair
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table
from .offdiag_update_e import (
    DEFAULT_BLOCK,
    E_TERMS,
    ROW_SLOTS,
    offdiag_constitutive_coverage,
    row_volumes_for,
    wall_mask_axes,
)

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "offdiag_fused_electric_pair"

#: Does this product bracket its fused launch with the deposit repair? NO, and
#: the reason is arithmetic rather than wiring — see the module docstring. The
#: flag is a claim about the PLAN this module builds and is only ever changed in
#: the same edit as that wiring.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: The curl sub-step this product starts at, and the constitutive side it ends on.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The seam this product spans, as ``coverage.FUSED_PAIRS`` keys it.
PAIR = "D"

#: The three volumes the curl half writes and the three it differences — READ
#: from the certified curl arm's own table (``launch.SUB_STEPS``), never spelled,
#: so a rename there reaches this product instead of drifting past it. This
#: family is the ABSORBER path, where H is stored and is what ``step_D``
#: differences; the no-PML arms' table says ``B`` and says why, and a test pins
#: this product against the right one of the two.
CURL_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
CURL_SOURCES: Tuple[str, ...] = ("Hx", "Hy", "Hz")

#: The two volumes per component the launch ROTATES: the stepped displacement and
#: its split-field auxiliary. Order is the launch's argument order and is not
#: negotiable — the plan binds writes then reads by exactly this sequence.
ROTATED: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

#: The driver call sites ONE launch of this plan performs, in driver order.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The in-seam passes this family REFUSES rather than carries, with the clause
#: that refuses each. Both are no-ops on every row this cell reaches (the fold is
#: refused by ``coverage._grid_reasons`` clause 5), and they are named here so a
#: reader can see the seam is accounted for in full rather than in part.
REFUSED_IN_SEAM_PASSES: Dict[str, str] = {
    "fill_symmetry_bc_D": "coverage._grid_reasons clause 5 (no mirror plane)",
    "fill_folded_far_ghosts_D": "coverage._grid_reasons clause 5 (no mirror plane)",
}

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins
#: against ``launch.SUB_STEPS`` rather than a literal buried in the builder.
BACKWARD = 1

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SOURCES",
    "CURL_SUB_STEP", "CURL_TARGETS", "FAMILY", "PAIR", "REFUSED_IN_SEAM_PASSES",
    "REPAIR_PATHS", "REPLACES", "ROTATED",
    "OffdiagFusedElectricPairPlan",
    "offdiag_fused_curl_constitutive_D",
    "offdiag_fused_curl_constitutive_D_kernel",
    "offdiag_fused_electric_pair_coverage",
    "plan_offdiag_fused_electric_pair",
    "plan_offdiag_fused_electric_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names — including the `tl.constexpr`
# annotations — through the defining module's `__globals__`, so a kernel defined
# inside a function compiles to `NameError('tl is not defined')` at first launch
# (dispersive_update_e.py:138-143 records the run that showed it, 320 of 320
# cases).

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    #: Boundary codes, identical to ``kernels``' and ``offdiag_update_e``'s own.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _step_cell(i, j, k, live,
                   f0, f1, f2, u0, u1, u2, g0, g1, g2,
                   kmx, sinvx, kmy, sinvy, kmz, sinvz,
                   nx, ny, nz, dtdx,
                   BACKWARD: tl.constexpr,
                   BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr):
        """``kernels.pml_curl_step`` at ONE named cell, returning instead of storing.

        THE BODY BELOW THE TWO INDEX LINES IS THE CERTIFIED CURL'S OWN, byte for
        byte between its decode anchor (``i = plane // ny``) and its store anchor
        (``tl.store(u0 + idx``). ``offdiag_scratch_weld.certified_tail`` and
        ``.lifted_tail`` cut both at those anchors and the probe's transcription
        leg asserts the two strings equal, so this is a LIFT and not a
        transcription: the only edits are the head (``idx`` and ``nyz`` rebuilt
        from the function's own ``(i, j, k)`` by integer arithmetic, which moves
        no bit) and the tail (six stores become one return).

        ``live`` is the caller's per-lane validity for THIS cell, not the
        dispatch's ``idx < n_elem``: a foreign tap masked off by the
        constitutive's own ghost rule passes ``live=False`` there, every load
        inside is masked, and no out-of-range coordinate is ever dereferenced.
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which `tl.load`'s
        # `other=` delivers without dereferencing anything.
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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Cell 0 of a metallic axis, for every target whose Yee shift there is 0. The
        # B targets have a single zero shift each (Bx:x, By:y, Bz:z); the D targets
        # have two (Dx:y,z  Dy:x,z  Dz:x,y). Masked, then the recurrence runs on zero.
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

        # --- split-field recurrence (stepping._apply_pml_update) -------------------
        # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
        # sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
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

        return v0, v1, v2, n0, n1, n2

    @triton.jit
    def _seam_resolve(v0, v1, v2, i, j, k,
                      ZM_X: tl.constexpr, ZM_Y: tl.constexpr,
                      ZM_Z: tl.constexpr):
        """THE CLOSED FORM: every in-seam pass this family carries, per cell.

        One pass survives here — ``stepping.zero_metal_D``, which writes an exact
        zero into stored cell 0 of every component whose Yee shift on that
        metallic axis is 0 (``_zero_metal``, stepping.py:2253-2277). The block is
        ``kernels.fused_curl_constitutive_D``'s own D-side ``ZM_*`` block, lifted
        whole and grouped BY AXIS exactly as that kernel groups it; the probe's
        transcription leg asserts it is a contiguous run of that certified body.

        It is total over the volume, which is what lets the constitutive half
        apply it to a FOREIGN tap: the driver's pass has already run everywhere
        by the time ``update_E`` reads a neighbour, so a tap resolved without it
        would read a wall cell the array path had cleared.
        """
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)
        return v0, v1, v2

    @triton.jit
    def _tap(COMP: tl.constexpr, i, j, k, valid,
             f0, f1, f2, u0, u1, u2, g0, g1, g2,
             kmx, sinvx, kmy, sinvy, kmz, sinvz,
             nx, ny, nz, dtdx,
             BACKWARD: tl.constexpr,
             BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
             ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr):
        """One FOREIGN displacement the constitutive half would have loaded.

        The whole substitution, in one place: step the cell from pre-launch
        state, resolve the seam over it, take the component the certified load
        named, and serve ``+0.0`` where that load's mask was False — which is
        what ``other=0.0`` served.
        """
        v0, v1, v2, n0, n1, n2 = _step_cell(
            i, j, k, valid, f0, f1, f2, u0, u1, u2, g0, g1, g2,
            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
            BACKWARD, BCX, BCY, BCZ)
        v0, v1, v2 = _seam_resolve(v0, v1, v2, i, j, k, ZM_X, ZM_Y, ZM_Z)
        value = v0
        if COMP == 1:
            value = v1
        if COMP == 2:
            value = v2
        return tl.where(valid, value, 0.0)

    @triton.jit
    def _welded_offdiag_term(near_c, near_d, far_u, far_ud, u, o_c, o_u, v_c, v_u):
        """``offdiag_update_e._offdiag_term`` with its four D loads REDIRECTED.

        The certified helper's four ``tl.load(g + o*, mask=v*, other=0.0)``
        expressions become the four values the caller resolved; everything else —
        the two sums, the coefficient multiply BETWEEN the shifts, the 0.25
        applied last to the sum — is that helper's, unchanged, because the array
        path's is unchanged. The probe checks the substitution by needle: four
        exact replacements, no other edit.
        """
        near = (near_c
                + near_d)
        far = (far_u
               + far_ud)
        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def _masked_row_sum(diag, total, at_a, at_b,
                        WMA: tl.constexpr, WMB: tl.constexpr):
        """Wall-mask the coupling, THEN form the row sum — the array path's
        order (``_mask_metallic_wall_coupling`` at stepping.py:1252 runs before
        the ``constitutive + coupling`` add at :1007-1008). ``at_a``/``at_b`` are
        the face-0 predicates of the component's two Yee-shift-0 axes, in
        ascending axis order (the mask's own loop order); only face 0 is
        zeroed — the high wall is the shift-up zero ghost."""
        if WMA:
            total = tl.where(at_a, 0.0, total)
        if WMB:
            total = tl.where(at_b, 0.0, total)
        return diag + total

    @triton.jit
    def offdiag_fused_curl_constitutive_D(
        d0s, d1s, d2s,                  # SCRATCH out: the stepped Dx, Dy, Dz
        a0s, a1s, a2s,                  # SCRATCH out: the stepped fu_Dx, fu_Dy, fu_Dz
        q0, q1, q2,                     # PRE-LAUNCH Dx, Dy, Dz            (read-only)
        r0, r1, r2,                     # PRE-LAUNCH fu_Dx, fu_Dy, fu_Dz   (read-only)
        g0, g1, g2,                     # curl sources: Hx, Hy, Hz         (read-only)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER sub-lattice
        f0, f1, f2,                     # targets:      Ex, Ey, Ez             (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        e0, e1, e2,                     # inverse-epsilon VOLUMES (three distinct or aliased)
        u01, u02,                       # row Ex: partner Ey (down y), partner Ez (down z)
        u11, u12,                       # row Ey: partner Ez (down z), partner Ex (down x)
        u21, u22,                       # row Ez: partner Ex (down x), partner Ey (down y)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``zero_metal_D`` + off-diagonal ``update_E``, one launch.

        ``g0``/``g1``/``g2`` are the MAGNETIC field here, not the displacement:
        every displacement read the certified constitutive body made has been
        redirected, and the probe asserts no ``g`` load survives below the seam
        for exactly that reason — one missed read would be a smooth, plausible,
        entirely wrong answer rather than a compile error.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- per-axis neighbour indices, both directions, with the ghost rule
        # (stepping._shift_down / _shift_up, plain PERIODIC/METALLIC branches).
        di, dj, dk = i - 1, j - 1, k - 1
        ui, uj, uk = i + 1, j + 1, k + 1
        dvx, dvy, dvz = live, live, live
        uvx, uvy, uvz = live, live, live
        if BCX == METALLIC:
            dvx = live & (di >= 0)
            uvx = live & (ui < nx)
        else:
            di = tl.where(di < 0, nx - 1, di)
            ui = tl.where(ui == nx, 0, ui)
        if BCY == METALLIC:
            dvy = live & (dj >= 0)
            uvy = live & (uj < ny)
        else:
            dj = tl.where(dj < 0, ny - 1, dj)
            uj = tl.where(uj == ny, 0, uj)
        if BCZ == METALLIC:
            dvz = live & (dk >= 0)
            uvz = live & (uk < nz)
        else:
            dk = tl.where(dk < 0, nz - 1, dk)
            uk = tl.where(uk == nz, 0, uk)

        # Wall-plane predicates for the coupling mask (face 0 only).
        at_x, at_y, at_z = i == 0, j == 0, k == 0

        # ============ THE SCRATCH WELD: step_D and the in-seam wall pass ========
        # Nothing written below is read. The constitutive half re-derives every
        # foreign tap from PRE-LAUNCH state through the same `_step_cell`, and
        # the plan rotates the buffers after the launch returns.
        own0, own1, own2, aux0, aux1, aux2 = _step_cell(
            i, j, k, live, q0, q1, q2, r0, r1, r2, g0, g1, g2,
            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
            BACKWARD, BCX, BCY, BCZ)
        tl.store(a0s + idx, aux0, mask=live)
        tl.store(a1s + idx, aux1, mask=live)
        tl.store(a2s + idx, aux2, mask=live)
        # The auxiliary takes the RAW stepped value: neither `_fill_symmetry_ghost_cells`
        # nor `_zero_metal` walks `fu_*` (both loop D_CURL_TERMS / D_COMPONENTS).
        dfin0, dfin1, dfin2 = _seam_resolve(own0, own1, own2, i, j, k,
                                            ZM_X, ZM_Y, ZM_Z)
        tl.store(d0s + idx, dfin0, mask=live)
        tl.store(d1s + idx, dfin1, mask=live)
        tl.store(d2s + idx, dfin2, mask=live)

        # The twelve foreign taps, one per index shape the certified body loads:
        # three DOWN faces, three UP faces, six corners. Each is the component the
        # certified load named, re-derived at that cell and seam-resolved. A dead
        # row leaves its taps unused and the compiler drops them; nothing here is
        # a coefficient, so an unused tap cannot change a live one.
        tap_xd_0 = _tap(0, di, j, k, dvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_yd_1 = _tap(1, i, dj, k, dvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_zd_2 = _tap(2, i, j, dk, dvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_xu_1 = _tap(1, ui, j, k, uvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_xu_2 = _tap(2, ui, j, k, uvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_yu_0 = _tap(0, i, uj, k, uvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_yu_2 = _tap(2, i, uj, k, uvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_zu_0 = _tap(0, i, j, uk, uvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_zu_1 = _tap(1, i, j, uk, uvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                        kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                        BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_xu_yd_1 = _tap(1, ui, dj, k, uvx & dvy,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_xu_zd_2 = _tap(2, ui, j, dk, uvx & dvz,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_yu_zd_2 = _tap(2, i, uj, dk, uvy & dvz,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_yu_xd_0 = _tap(0, di, uj, k, uvy & dvx,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_zu_xd_0 = _tap(0, di, j, uk, uvz & dvx,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)
        tap_zu_yd_1 = _tap(1, i, dj, uk, uvz & dvy,
                           q0, q1, q2, r0, r1, r2, g0, g1, g2,
                           kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                           BACKWARD, BCX, BCY, BCZ, ZM_X, ZM_Y, ZM_Z)

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z)
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        gs0 = dfin0
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            total0 = _welded_offdiag_term(
                dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,
                u01, idx,
                ui * nyz + j * nz + k,
                live, uvx)
            if R02:
                total0 = total0 + _welded_offdiag_term(
                    dfin2, tap_zd_2, tap_xu_2, tap_xu_zd_2,
                    u02, idx,
                    ui * nyz + j * nz + k,
                    live, uvx)
            src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
        else:
            if R02:
                total0 = _welded_offdiag_term(
                    dfin2, tap_zd_2, tap_xu_2, tap_xu_zd_2,
                    u02, idx,
                    ui * nyz + j * nz + k,
                    live, uvx)
                src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z,
                                       WM_Y, WM_Z)
            else:
                src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(f0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(f0 + idx, a0, mask=live)

        # --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x)
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        gs1 = dfin1
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            total1 = _welded_offdiag_term(
                dfin2, tap_zd_2, tap_yu_2, tap_yu_zd_2,
                u11, idx,
                i * nyz + uj * nz + k,
                live, uvy)
            if R12:
                total1 = total1 + _welded_offdiag_term(
                    dfin0, tap_xd_0, tap_yu_0, tap_yu_xd_0,
                    u12, idx,
                    i * nyz + uj * nz + k,
                    live, uvy)
            src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
        else:
            if R12:
                total1 = _welded_offdiag_term(
                    dfin0, tap_xd_0, tap_yu_0, tap_yu_xd_0,
                    u12, idx,
                    i * nyz + uj * nz + k,
                    live, uvy)
                src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z,
                                       WM_X, WM_Z)
            else:
                src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(f1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(f1 + idx, a1, mask=live)

        # --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y)
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        gs2 = dfin2
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            total2 = _welded_offdiag_term(
                dfin0, tap_xd_0, tap_zu_0, tap_zu_xd_0,
                u21, idx,
                i * nyz + j * nz + uk,
                live, uvz)
            if R22:
                total2 = total2 + _welded_offdiag_term(
                    dfin1, tap_yd_1, tap_zu_1, tap_zu_yd_1,
                    u22, idx,
                    i * nyz + j * nz + uk,
                    live, uvz)
            src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
        else:
            if R22:
                total2 = _welded_offdiag_term(
                    dfin1, tap_yd_1, tap_zu_1, tap_zu_yd_1,
                    u22, idx,
                    i * nyz + j * nz + uk,
                    live, uvz)
                src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y,
                                       WM_X, WM_Y)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(f2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(f2 + idx, a2, mask=live)

else:  # pragma: no cover - the laptop path
    offdiag_fused_curl_constitutive_D = None  # type: ignore[assignment]
    _step_cell = None  # type: ignore[assignment]
    _seam_resolve = None  # type: ignore[assignment]
    _tap = None  # type: ignore[assignment]
    _welded_offdiag_term = None  # type: ignore[assignment]
    _masked_row_sum = None  # type: ignore[assignment]


def offdiag_fused_curl_constitutive_D_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if offdiag_fused_curl_constitutive_D is None:
        raise ImportError(
            "the off-diagonal fused electric pair needs the optional `triton` "
            "package (pip install triton). The engine runs without it; only "
            f"this fast path is unavailable. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return offdiag_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def offdiag_fused_electric_pair_coverage(fields: Any, pml: Any,
                                         sources: Any = None
                                         ) -> "_coverage.Coverage":
    """May ONE launch step ``step_D`` + ``zero_metal_D`` + off-diagonal ``update_E``?

    Positive clauses only, and every one is a conjunction of predicates that
    already exist:

    1. :func:`coverage.pml_curl_coverage` on ``step_D`` — the curl half is the
       certified curl arm, lifted, so a configuration that arm refuses this
       product must refuse too. It is what refuses the fold, the complex
       storage, a conductivity, an absent absorber and the coordinate systems.
    2. :func:`offdiag_update_e.offdiag_constitutive_coverage` — the constitutive
       half is the certified off-diagonal arm, welded. It is what REQUIRES at
       least one surviving off-diagonal row (a zero-row run is the plain fused
       pair's and the two predicates must not overlap) and what refuses a
       registered polarization.
    3. **No electric source in the seam.** The driver injects between the two
       (driver.py:3303-3308), and this product does NOT carry the deposit
       repair: :data:`CARRIES_DEPOSIT_REPAIR` is False, so
       ``deposit_repair.seam_source_reasons`` returns one refusal per in-seam
       source. That is a refusal about the SHAPE of the repair, not about
       wiring — see the module docstring: the coupling is a stencil, so one
       deposited cell moves the constitutive result of up to four, and the
       repair's inversion is per cell.
    4. **The grid can answer the wall question.** ``zero_metal_D`` is carried
       INLINE by the closed form, from :func:`coverage.zero_metal_axes`; a grid
       that cannot answer compiles to ``ZM=False`` and silently skips a plane
       the array path clears.
    5. **The rotation has two distinct buffers per rotating volume**, checked as
       allocation rather than assumed: the six volumes must exist, because the
       plan twins them.

    ``sources`` MUST BE DECLARED for the same reason every fused predicate in
    this package requires it: ``Fields`` does not hold the source list, and
    inferring "no electric source" from not knowing is how a predicate
    over-covers.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    curl = _coverage.pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    constitutive = offdiag_constitutive_coverage(fields, pml)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)

    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, PAIR,
        undeclared=(
            "the source set was not declared: this predicate cannot read it off "
            "Fields, and inferring 'no electric source' from not knowing is how "
            "a predicate over-covers"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E, and this product cannot "
            f"carry the deposit — the off-diagonal coupling is a STENCIL, so a "
            f"deposit at one cell moves the constitutive result of up to four, "
            f"while deposit_repair inverts the recurrence AT the deposit cell "
            f"only"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried "
                f"inline by the closed form")

    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; the plan twins every rotating volume "
                f"and cannot rotate what does not exist")

    return _coverage.Coverage(not reasons, tuple(reasons))


def explain_offdiag_fused_electric_pair(fields: Any, pml: Any,
                                        sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads. One home for the verdict."""
    return offdiag_fused_electric_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class OffdiagFusedElectricPairPlan(ScratchWeldPairPlan):
    """One allocation-free launch for ``step_D``, ``zero_metal_D`` and ``update_E``.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns
    a twin of each, reads the engine's CURRENT attribute immediately before the
    launch to decide which of the pair is live, and moves the engine's references
    only after the launch returns. A pointer captured at plan time would be one
    rotation stale — and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w, and the six
    scratch buffers) and any input (the pre-launch D/fu/H, inverse epsilon, the
    row volumes and the coefficient vectors), and among the outputs themselves.
    The base class additionally refuses a scratch buffer that IS its own live
    volume, which would be the 2026-08-20 in-place weld wearing this class's
    name.
    """

    __slots__ = ("dtdx", "backward", "bc", "zero_metal", "wall_mask", "row_mask",
                 "_sources", "_pml", "_e_targets", "_aux", "_inv_eps", "_rows",
                 "_coeff", "_kernel", "_pointer")

    replaces = REPLACES

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 zero_metal: Sequence[bool], wall_mask: Sequence[int],
                 block: int, fields: Any, twins: Dict[str, Any],
                 sources: Sequence[Any], pml_columns: Sequence[Any],
                 e_targets: Sequence[Any], aux: Sequence[Any],
                 inverse_epsilon: Sequence[Any], rows: Sequence[Any],
                 coefficients: Sequence[Any], kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.wall_mask = tuple(int(flag) for flag in wall_mask)
        if len(self.bc) != 3 or len(self.zero_metal) != 3 or len(self.wall_mask) != 3:
            raise ValueError(
                "this plan needs three boundary codes, three wall flags and "
                "three coupling-mask flags")
        self._pointer = CupyPointer
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._pml = tuple(CupyPointer(_flat(a)) for a in pml_columns)
        self._e_targets = tuple(CupyPointer(a) for a in e_targets)
        self._aux = tuple(CupyPointer(a) for a in aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        rows = tuple(rows)
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(
                f"this plan needs {len(ROW_SLOTS)} off-diagonal row slots, "
                f"got {len(rows)}")
        if not any(row is not None for row in rows):
            raise ValueError(
                "no off-diagonal row slot survives; that configuration is the "
                "plain fused pair's and the predicate refuses it first")
        self.row_mask = tuple(1 if row is not None else 0 for row in rows)
        # A dead slot binds the component's own E pointer: correctly typed, and
        # the constexpr arm is what stops the read. The certified off-diagonal
        # plan's own convention, kept rather than re-invented.
        self._rows = tuple(
            CupyPointer(row) if row is not None
            else CupyPointer(tuple(e_targets)[index // 2])
            for index, row in enumerate(rows))
        self._coeff = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._check_aliases(twins, sources, e_targets, aux, inverse_epsilon,
                            rows, pml_columns, coefficients)
        self._kernel = kernel

    def _check_aliases(self, twins, sources, e_targets, aux, inverse_epsilon,
                       rows, pml_columns, coefficients) -> None:
        """No output may share an allocation with an input or another output.

        The coupling reads partner displacement at neighbour offsets while E and
        f_w are being written; an aliased pair would make the result depend on
        block schedule — which is the whole defect this design removes, and
        would remove it only to reintroduce it through a binding.
        """
        def address(array: Any) -> Optional[int]:
            data = getattr(array, "data", None)
            pointer = getattr(data, "ptr", None)
            if pointer is not None:
                return int(pointer)
            interface = getattr(array, "__array_interface__", None)
            if isinstance(interface, dict):
                return int(interface["data"][0])
            return None

        outputs: Dict[int, str] = {}
        named = list(zip(ROTATED, [twins[name] for name in ROTATED]))
        named += [(f"E{index}", array) for index, array in enumerate(e_targets)]
        named += [(f"f_w{index}", array) for index, array in enumerate(aux)]
        for label, array in named:
            key = address(array)
            if key is None:
                continue
            if key in outputs:
                raise ValueError(
                    f"outputs {outputs[key]} and {label} are the same "
                    f"allocation; one launch would write both")
            outputs[key] = label
        inputs: List[Tuple[str, Any]] = [
            (f"source{index}", array) for index, array in enumerate(sources)]
        inputs += [(f"inv_eps{index}", array)
                   for index, array in enumerate(inverse_epsilon)]
        inputs += [(f"row{index}", array) for index, array in enumerate(rows)
                   if array is not None]
        inputs += [(f"pml{index}", array)
                   for index, array in enumerate(pml_columns)]
        inputs += [(f"coefficient{index}", array)
                   for index, array in enumerate(coefficients)]
        for label, array in inputs:
            key = address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the launch "
                    f"would read a volume it is writing")

    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else offdiag_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        scratch = [self._pointer(array) for array in writes]
        prior = [self._pointer(array) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._sources, *self._pml,
            *self._e_targets, *self._aux, *self._inv_eps, *self._rows,
            *self._coeff,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            WM_X=self.wall_mask[0], WM_Y=self.wall_mask[1],
            WM_Z=self.wall_mask[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"OffdiagFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, row_mask={self.row_mask}, "
                f"wall_mask={self.wall_mask}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        ) -> Optional[OffdiagFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not offdiag_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    grid = fields.grid
    kinds = _coverage._boundary_kinds(grid, pml)
    return OffdiagFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        _coverage.zero_metal_axes(grid),
        wall_mask_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        fields,
        twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CURL_SOURCES],
        [getattr(pml, f"{stem}_{axis}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, component) for component, _target, _axis in E_TERMS],
        [getattr(fields, "f_w_" + component)
         for component, _target, _axis in E_TERMS],
        [fields.inverse_epsilon_for(component)
         for component, _target, _axis in E_TERMS],
        row_volumes_for(fields),
        # The HALF-INTEGER sub-lattice, in the certified off-diagonal plan's own
        # order (kps_x, kms_x, kps_y, ...): component 0 takes axis x, 1 y, 2 z,
        # which is what makes `for axis in "xyz" for stem in (...)` the kernel's
        # `kp0, km0, kp1, km1, kp2, km2`. Swapping the sub-lattice is a half-cell
        # error in the absorber profile, converged and smooth and wrong.
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


def plan_offdiag_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], constitutive_flat: Dict[str, Any],
        dtdx: float, codes: Sequence[int], zero_metal: Sequence[bool],
        wall_mask: Sequence[int], rows: Sequence[Any], fields: Any,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> OffdiagFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

    ``fields`` is the object whose attributes the ROTATION swaps. The gate passes
    a small namespace holding the six rotating volumes under the engine's own
    names, so the rotation the gate exercises is the rotation the engine would
    get — a plan that rotated a private dict instead would certify a launcher
    that does not ship. ``arrays`` supplies the twins under ``scratch_Dx`` ...;
    ``kernel=`` carries the mutation override, and dropping it silently disarms
    every mutation leg.
    """
    shape = tuple(int(n) for n in arrays[CURL_TARGETS[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return OffdiagFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, wall_mask,
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CURL_SOURCES],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[component] for component, _target, _axis in E_TERMS],
        [arrays["f_w_" + component] for component, _target, _axis in E_TERMS],
        [arrays["inv_eps_" + component] for component, _target, _axis in E_TERMS],
        rows,
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )
