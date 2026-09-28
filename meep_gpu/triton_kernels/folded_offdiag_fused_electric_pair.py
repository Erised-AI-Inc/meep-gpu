"""The FOLDED PML ``step_D`` welded into folded OFF-DIAGONAL ``update_E``.

    ceiling 9  (of 19 rows)  D->E  folded PML -> folded off-diagonal

The largest of the three stencil-blocked cells this lane can reach, and the one
the fusion-residue audit rated MEDIUM rather than HIGH because "the fold adds
read-side redirect-at-image-source machinery". It does, and the redirect is
per-cell pure — nothing written is read — so the runtime-cross-lane-read refusal
that stops the folded MAGNETIC pair (``folded_fused_magnetic_pair.py:602-608``)
does not arise here.

THE DESIGN IS ``offdiag_fused_electric_pair``'S, with a longer closed form; read
that module and ``offdiag_scratch_weld`` first. The curl half writes ``D_new``
and ``fu_new`` into launch-local scratch, the constitutive half re-derives each
of its foreign stencil taps from pre-launch state, and the plan rotates the D/fu
references after the launch. What is different is the seam:

    THIS FAMILY CARRIES TWO IN-SEAM PASSES, NOT ONE.

``fill_symmetry_bc_D`` (driver.py:3309) writes ``cell 0 = phase * cell 2`` on
every mirrored axis where a component's Yee shift is 0, in X, Y, Z order; then
``zero_metal_D`` (:3310) clears cell 0 on every metallic, UNMIRRORED axis. Both
are resolved per cell by :func:`_d_final`: the near fill becomes a READ-SIDE
REDIRECT (the cell's own coordinate on a folded axis moves from 0 to
``MIRROR_ROW``, carrying that axis's ``PH_*``), and the wall clear is applied
AFTER it, which is the driver's order — the fill runs first and the clear
overwrites whatever it wrote on a cell that is on both. The parity spelling is
the certified fill's own (``symmetry.mirror_ghost_fill``: ``PHASE * value``, a
multiply by exactly +/-1.0, exact in float32 for every input including
subnormals and signed zeros).

Measured per row off the census
(``results/fusion_matrix_triton_2026-09-02_residue``): of the nine reachable
rows, seven declare ``fill_D`` as their only live in-seam pass and two declare
``fill_D`` and ``zero_metal_D``. NONE declares ``fill_folded_far_ghosts_D``,
and this weld does not carry it — see the refusal below, which is by the pass's
own liveness condition rather than by boundary code, so a folded periodic axis
with no stored far slot (where that pass writes nothing) stays served.

WHAT IS REFUSED, AND WHY IT IS A REFUSAL AND NOT A GAP:

* ``fill_folded_far_ghosts_D`` — an axis that STORES MEEP's far ghost slot
  (``stepping._stored_past_owned``: mirrored and ``stored_cells >
  owned_cells``) has a top slot whose value is imaged from a RUNTIME reflect row
  after the sources land. The closed form for it exists and is measured on
  another backend, but this weld does not implement it, and a predicate that
  admitted the axis would be a predicate wider than its product. No corpus row
  on this cell needs it.
* An in-seam electric source, for exactly the reason
  ``offdiag_fused_electric_pair`` refuses one: the coupling is a stencil and
  ``deposit_repair`` inverts per cell. :data:`CARRIES_DEPOSIT_REPAIR` is False.

NOT WIRED. ``fastpath`` is untouched; dispatch is a separate milestone.

Import contract: importable WITHOUT Triton.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import symmetry as _symmetry
from .. import deposit_repair as _deposit_repair
from .folded_offdiag_update_e import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    MIRROR_SOURCE_INDEX,
    folded_offdiag_composition_coverage,
    mirror_ghost_axes,
    mirror_ghost_weights,
)
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table
from .offdiag_update_e import (
    DEFAULT_BLOCK,
    E_TERMS,
    ROW_SLOTS,
    row_volumes_for,
    wall_mask_axes,
)

FAMILY = "folded_offdiag_fused_electric_pair"

#: See the module docstring: the coupling is a STENCIL and ``deposit_repair``
#: inverts the constitutive recurrence AT the deposit cell, so a per-cell repair
#: would leave up to three neighbours stepped from a pre-injection displacement.
CARRIES_DEPOSIT_REPAIR = False
REPAIR_PATHS: Tuple[str, ...] = ()

CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"
PAIR = "D"

CURL_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
CURL_SOURCES: Tuple[str, ...] = ("Hx", "Hy", "Hz")
ROTATED: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

#: Three driver call sites, in driver order — one more than the unfolded family.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "update_E")

REFUSED_IN_SEAM_PASSES: Dict[str, str] = {
    "fill_folded_far_ghosts_D":
        "an axis that stores MEEP's far ghost slot is refused by name; the "
        "closed form for the far image is not implemented here",
}

BACKWARD = 1

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SOURCES",
    "CURL_SUB_STEP", "CURL_TARGETS", "FAMILY", "PAIR", "REFUSED_IN_SEAM_PASSES",
    "REPAIR_PATHS", "REPLACES", "ROTATED",
    "FoldedOffdiagFusedElectricPairPlan",
    "far_ghost_axes",
    "folded_offdiag_fused_curl_constitutive_D",
    "folded_offdiag_fused_curl_constitutive_D_kernel",
    "folded_offdiag_fused_electric_pair_coverage",
    "mirror_fill_phases",
    "plan_folded_offdiag_fused_electric_pair",
    "plan_folded_offdiag_fused_electric_pair_from_arrays",
]


def mirror_fill_phases(grid: Any) -> Tuple[int, int, int]:
    """The ``PH_X``/``PH_Y``/``PH_Z`` constexprs — one PLANE PHASE per axis.

    ``symmetry.mirror_ghost_fill`` takes exactly this: the plane's declared
    phase, and the near fill's parity for a Yee-shift-0 component is that phase
    unchanged (``fields.mirror_parity(c, axis, phase) == phase * (1 - 2 *
    iyee[c][axis])``, verified exhaustively by that kernel's own gate). The near
    fill only ever touches shift-0 components, so one signed integer per folded
    axis is the whole parity input.

    ``+1`` on an unfolded axis, where the redirect never fires and the value is
    never read. An unreadable plane returns ``0``, which is not a phase: the
    predicate refuses it by name through ``symmetry.folded_axis_kinds`` and the
    plan validates the triple, so a zero can never reach a launch.
    """
    out: List[int] = []
    for axis in range(3):
        if not bool(_coverage._call(grid, "is_mirrored", axis, default=False)):
            out.append(1)
            continue
        phase = _coverage._call(grid, "mirror_phase", axis, default=None)
        out.append(int(phase) if phase in (1, -1) else 0)
    return (out[0], out[1], out[2])


def far_ghost_axes(grid: Any) -> Tuple[bool, bool, bool]:
    """Per axis: would ``fill_folded_far_ghosts_D`` write anything here?

    ``stepping._stored_past_owned`` itself, read rather than restated. It is the
    pass's own liveness condition — mirrored AND storing one slot past MEEP's
    owned window — and asking it directly is what keeps a folded PERIODIC axis
    with no extra slot SERVED rather than refused for carrying a boundary code.
    An unreadable ``stepping`` returns True on every axis, which refuses: a
    predicate that cannot establish the pass is dead must not admit.
    """
    reader = _symmetry._stored_past_owned_reader()
    if reader is None:  # pragma: no cover - stepping is always importable
        return (True, True, True)
    out: List[bool] = []
    for axis in range(3):
        try:
            out.append(bool(reader(grid, axis)))
        except Exception:  # noqa: BLE001 - unanswerable means refused
            out.append(True)
    return (out[0], out[1], out[2])


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    #: The four-valued boundary code set, identical to ``symmetry``'s and
    #: ``folded_offdiag_update_e``'s own; a test pins all three equal.
    PERIODIC = tl.constexpr(_symmetry.CODE_PERIODIC)
    METALLIC = tl.constexpr(_symmetry.CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)
    MIRROR_ROW = tl.constexpr(MIRROR_SOURCE_INDEX)

    @triton.jit
    def _step_cell(i, j, k, live,
                   f0, f1, f2, u0, u1, u2, g0, g1, g2,
                   kmx, sinvx, kmy, sinvy, kmz, sinvz,
                   nx, ny, nz, dtdx,
                   BACKWARD: tl.constexpr,
                   BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr):
        """``symmetry.pml_curl_step_folded`` at ONE named cell, returning.

        THE BODY BELOW THE TWO INDEX LINES IS THE CERTIFIED FOLDED CURL'S OWN,
        byte for byte between its decode anchor and its store anchor; the probe's
        transcription leg asserts the two strings equal. The only edits are the
        head (``idx``/``nyz`` rebuilt from ``(i, j, k)`` by integer arithmetic)
        and the tail (six stores become one return).
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
        # PERIODIC wraps; every other rule here serves an exact 0.0 past the face,
        # which `tl.load`'s `other=` delivers without dereferencing anything. On a
        # folded axis that zero stands in for a value nothing reads (see 1 above).
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == PERIODIC:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        else:
            vx = live & (si >= 0) & (si < nx)
        if BCY == PERIODIC:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        else:
            vy = live & (sj >= 0) & (sj < ny)
        if BCZ == PERIODIC:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        else:
            vz = live & (sk >= 0) & (sk < nz)

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

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells) ------------
        # Every target whose Yee shift is 0 on a non-periodic axis. Byte-copied
        # from the shipped kernel with `== METALLIC` widened to `!= PERIODIC`.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY != PERIODIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ != PERIODIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX != PERIODIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ != PERIODIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX != PERIODIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY != PERIODIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX != PERIODIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY != PERIODIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ != PERIODIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        # --- ownership mask, the TOP plane of a folded PERIODIC axis ------------
        # The complement of the block above: every target whose Yee shift is 1
        # there. Written out per (side, target, axis) exactly as that block is —
        # no loop, no derived predicate — so a reader checks it against
        # `_mask_non_owned_cells`'s `if iyee[axis] != 0` arm by eye.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
            # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) — shift 1 on its OWN axis only.
            if BCX == MIRROR_PERIODIC:
                curl0 = tl.where(last_x, 0.0, curl0)
            if BCY == MIRROR_PERIODIC:
                curl1 = tl.where(last_y, 0.0, curl1)
            if BCZ == MIRROR_PERIODIC:
                curl2 = tl.where(last_z, 0.0, curl2)
        else:
            # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) — shift 1 on the two OTHER axes.
            if BCY == MIRROR_PERIODIC:
                curl0 = tl.where(last_y, 0.0, curl0)
            if BCZ == MIRROR_PERIODIC:
                curl0 = tl.where(last_z, 0.0, curl0)
            if BCX == MIRROR_PERIODIC:
                curl1 = tl.where(last_x, 0.0, curl1)
            if BCZ == MIRROR_PERIODIC:
                curl1 = tl.where(last_z, 0.0, curl1)
            if BCX == MIRROR_PERIODIC:
                curl2 = tl.where(last_x, 0.0, curl2)
            if BCY == MIRROR_PERIODIC:
                curl2 = tl.where(last_y, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ----------------
        # Byte-copied. The PML coefficient vectors are built at the STORED extent
        # on a folded axis (measured: kms_y.shape == (1, 22, 1) on a grid storing
        # 22), so the `n_a` indexing needs no fold-aware change.
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
    def _d_final(COMP: tl.constexpr, i, j, k, valid,
                 f0, f1, f2, u0, u1, u2, g0, g1, g2,
                 kmx, sinvx, kmy, sinvy, kmz, sinvz,
                 nx, ny, nz, dtdx,
                 BACKWARD: tl.constexpr,
                 BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
                 MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
                 PH_X: tl.constexpr, PH_Y: tl.constexpr, PH_Z: tl.constexpr,
                 ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr):
        """THE CLOSED FORM: one component's displacement AS THE SEAM LEAVES IT.

        Both in-seam passes, resolved per cell, in the driver's order:

            fill (near)   coordinate 0 on a folded axis where this component's
                          Yee shift is 0 redirects to MIRROR_ROW and the value
                          carries that axis's PH — ``symmetry.mirror_ghost_fill``'s
                          ``field[0] = PHASE * field[2]``, read from the other end;
            clear         cell 0 on a metallic UNMIRRORED axis where the shift is
                          0 becomes exactly +0.0 — ``stepping._zero_metal``.

        THE CLEAR IS OUTSIDE THE FILL AND NOT INSIDE IT. The driver runs
        ``fill_symmetry_bc_D`` then ``zero_metal_D`` (driver.py:3309-3310), so a
        cell on both a fold plane and a wall ends at zero; putting the clear
        first would leave it holding the imaged value. The two act on disjoint
        axes (``zero_metal_axes`` asks ``is_metallic and not is_mirrored``), so
        the ordering is visible only on a corner — which is exactly where a
        reversed order is invisible to everything but a word comparison.

        The axes are redirected in X, Y, Z order, the order
        ``_fill_symmetry_ghost_cells`` applies them in, so a corner unowned on
        two planes carries the product of both phases.
        """
        # --- the near fill, as a read-side redirect. D component c has Yee shift
        # 1 on axis c and 0 on the other two, so each component redirects on the
        # two axes that are not its own.
        near_x = i == 0
        near_y = j == 0
        near_z = k == 0
        if COMP != 0:
            if MG_X:
                i = tl.where(near_x, MIRROR_ROW, i)
        if COMP != 1:
            if MG_Y:
                j = tl.where(near_y, MIRROR_ROW, j)
        if COMP != 2:
            if MG_Z:
                k = tl.where(near_z, MIRROR_ROW, k)

        v0, v1, v2, n0, n1, n2 = _step_cell(
            i, j, k, valid, f0, f1, f2, u0, u1, u2, g0, g1, g2,
            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
            BACKWARD, BCX, BCY, BCZ)
        value = v0
        if COMP == 1:
            value = v1
        if COMP == 2:
            value = v2

        # The phase, one axis at a time and in the same order. `PH * value` is the
        # certified fill's own spelling — a multiply by exactly +/-1, exact in
        # float32 for every input including subnormals and signed zeros.
        if COMP != 0:
            if MG_X:
                value = tl.where(near_x, PH_X * value, value)
        if COMP != 1:
            if MG_Y:
                value = tl.where(near_y, PH_Y * value, value)
        if COMP != 2:
            if MG_Z:
                value = tl.where(near_z, PH_Z * value, value)

        # --- the wall clear, AFTER the fill (the driver's order).
        if COMP != 0:
            if ZM_X:
                value = tl.where(near_x, 0.0, value)
        if COMP != 1:
            if ZM_Y:
                value = tl.where(near_y, 0.0, value)
        if COMP != 2:
            if ZM_Z:
                value = tl.where(near_z, 0.0, value)
        return tl.where(valid, value, 0.0)

    @triton.jit
    def _welded_folded_offdiag_term(near_c, near_d, far_u, far_ud, u, o_c, o_u,
                                    v_c, v_u, w_d, MG: tl.constexpr):
        """``folded_offdiag_update_e._folded_offdiag_term`` with its four D loads
        REDIRECTED.

        The certified helper's four ``tl.load(g + o*, mask=v*, other=0.0)``
        expressions become the four values the caller resolved. Everything else
        is that helper's, unchanged: the runtime ghost weight ``w_d`` still
        multiplies the two ghost-lane addends and only those, the ``MG``
        constexpr still selects between the weighted and the certified-verbatim
        arms, the coefficient multiply still sits BETWEEN the shifts, and 0.25
        still scales the sum, applied last.
        """
        if MG:
            near = (near_c
                    + w_d * near_d)
            far = (far_u
                   + w_d * far_ud)
        else:
            near = (near_c
                    + near_d)
            far = (far_u
                   + far_ud)
        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def _masked_row_sum(diag, total, at_a, at_b,
                        WMA: tl.constexpr, WMB: tl.constexpr):
        """Wall-mask the coupling, THEN form the row sum — the array path's order
        (``_mask_metallic_wall_coupling`` at stepping.py:1252 runs before the
        ``constitutive + coupling`` add at :1007-1008).

        Byte-copied from :func:`offdiag_update_e._masked_row_sum`. On a folded
        axis ``WM`` is 0 by construction (``wall_mask_axes`` asks the mask's own
        ``is_metallic and not is_mirrored``, stepping.py:1282), which is what
        leaves the fold plane's coupling alive — the thing zeroing it costs
        2.0e-02 (stepping.py:1269-1274)."""
        if WMA:
            total = tl.where(at_a, 0.0, total)
        if WMB:
            total = tl.where(at_b, 0.0, total)
        return diag + total

    @triton.jit
    def folded_offdiag_fused_curl_constitutive_D(
        d0s, d1s, d2s,                  # SCRATCH out: the stepped Dx, Dy, Dz
        a0s, a1s, a2s,                  # SCRATCH out: the stepped fu_Dx, fu_Dy, fu_Dz
        q0, q1, q2,                     # PRE-LAUNCH Dx, Dy, Dz            (read-only)
        r0, r1, r2,                     # PRE-LAUNCH fu_Dx, fu_Dy, fu_Dz   (read-only)
        g0, g1, g2,                     # curl sources: Hx, Hy, Hz         (read-only)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER sub-lattice
        f0, f1, f2,                     # targets:      Ex, Ey, Ez             (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        e0, e1, e2,                     # inverse-epsilon VOLUMES
        u01, u02,                       # row Ex: partner Ey (down y), partner Ez (down z)
        u11, u12,                       # row Ey: partner Ez (down z), partner Ex (down x)
        u21, u22,                       # row Ez: partner Ex (down x), partner Ey (down y)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        gwx, gwy, gwz,                  # RUNTIME mirror ghost weights, exactly +1.0 / -1.0
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
        PH_X: tl.constexpr, PH_Y: tl.constexpr, PH_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``fill_symmetry_bc_D`` + ``zero_metal_D`` + folded
        off-diagonal ``update_E``, one launch.

        ``g0``/``g1``/``g2`` are the MAGNETIC field here, not the displacement.
        ``MG_*`` and ``gw*`` are the CONSTITUTIVE half's ghost machinery (the
        down-shift weight on a mirror partner axis); ``PH_*`` are the FILL's
        plane phases. The two are different facts about the same fold and are
        carried separately for the same reason ``ZM_*`` is carried beside
        ``BC*``: one is a boundary rule, the other a driver pass.
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
        # stepping._shift_down (:1787) / _shift_up (:1723): PERIODIC wraps;
        # METALLIC masks both faces; MIRROR redirects the DOWN ghost to stored
        # row MIRROR_ROW with the plane's weight and masks the UP ghost.
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

        # ============ THE SCRATCH WELD: step_D and the two in-seam passes ======
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
        # The auxiliary takes the RAW stepped value: neither
        # `_fill_symmetry_ghost_cells` nor `_zero_metal` walks `fu_*` (both loop
        # D_CURL_TERMS / D_COMPONENTS).
        dfin0 = _d_final(0, i, j, k, live, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                         kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                         BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                         PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        dfin1 = _d_final(1, i, j, k, live, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                         kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                         BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                         PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        dfin2 = _d_final(2, i, j, k, live, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                         kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                         BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                         PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tl.store(d0s + idx, dfin0, mask=live)
        tl.store(d1s + idx, dfin1, mask=live)
        tl.store(d2s + idx, dfin2, mask=live)

        # The fifteen foreign taps, one per (component, cell) the certified body
        # loads: three DOWN faces, six UP faces, six corners.
        tap_xd_0 = _d_final(0, di, j, k, dvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_yd_1 = _d_final(1, i, dj, k, dvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_zd_2 = _d_final(2, i, j, dk, dvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_xu_1 = _d_final(1, ui, j, k, uvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_xu_2 = _d_final(2, ui, j, k, uvx, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_yu_0 = _d_final(0, i, uj, k, uvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_yu_2 = _d_final(2, i, uj, k, uvy, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_zu_0 = _d_final(0, i, j, uk, uvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_zu_1 = _d_final(1, i, j, uk, uvz, q0, q1, q2, r0, r1, r2, g0, g1, g2,
                            kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                            BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                            PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_xu_yd_1 = _d_final(1, ui, dj, k, uvx & dvy,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_xu_zd_2 = _d_final(2, ui, j, dk, uvx & dvz,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_yu_zd_2 = _d_final(2, i, uj, dk, uvy & dvz,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_yu_xd_0 = _d_final(0, di, uj, k, uvy & dvx,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_zu_xd_0 = _d_final(0, di, j, uk, uvz & dvx,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)
        tap_zu_yd_1 = _d_final(1, i, dj, uk, uvz & dvy,
                               q0, q1, q2, r0, r1, r2, g0, g1, g2,
                               kmx, sinvx, kmy, sinvy, kmz, sinvz, nx, ny, nz, dtdx,
                               BACKWARD, BCX, BCY, BCZ, MG_X, MG_Y, MG_Z,
                               PH_X, PH_Y, PH_Z, ZM_X, ZM_Y, ZM_Z)

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
            total0 = _welded_folded_offdiag_term(
                dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,
                u01, idx,
                ui * nyz + j * nz + k,
                live, uvx, wy, MG_Y)
            if R02:
                total0 = total0 + _welded_folded_offdiag_term(
                    dfin2, tap_zd_2, tap_xu_2, tap_xu_zd_2,
                    u02, idx,
                    ui * nyz + j * nz + k,
                    live, uvx, wz, MG_Z)
            src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
        else:
            if R02:
                total0 = _welded_folded_offdiag_term(
                    dfin2, tap_zd_2, tap_xu_2, tap_xu_zd_2,
                    u02, idx,
                    ui * nyz + j * nz + k,
                    live, uvx, wz, MG_Z)
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
            total1 = _welded_folded_offdiag_term(
                dfin2, tap_zd_2, tap_yu_2, tap_yu_zd_2,
                u11, idx,
                i * nyz + uj * nz + k,
                live, uvy, wz, MG_Z)
            if R12:
                total1 = total1 + _welded_folded_offdiag_term(
                    dfin0, tap_xd_0, tap_yu_0, tap_yu_xd_0,
                    u12, idx,
                    i * nyz + uj * nz + k,
                    live, uvy, wx, MG_X)
            src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
        else:
            if R12:
                total1 = _welded_folded_offdiag_term(
                    dfin0, tap_xd_0, tap_yu_0, tap_yu_xd_0,
                    u12, idx,
                    i * nyz + uj * nz + k,
                    live, uvy, wx, MG_X)
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
            total2 = _welded_folded_offdiag_term(
                dfin0, tap_xd_0, tap_zu_0, tap_zu_xd_0,
                u21, idx,
                i * nyz + j * nz + uk,
                live, uvz, wx, MG_X)
            if R22:
                total2 = total2 + _welded_folded_offdiag_term(
                    dfin1, tap_yd_1, tap_zu_1, tap_zu_yd_1,
                    u22, idx,
                    i * nyz + j * nz + uk,
                    live, uvz, wy, MG_Y)
            src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
        else:
            if R22:
                total2 = _welded_folded_offdiag_term(
                    dfin1, tap_yd_1, tap_zu_1, tap_zu_yd_1,
                    u22, idx,
                    i * nyz + j * nz + uk,
                    live, uvz, wy, MG_Y)
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
    folded_offdiag_fused_curl_constitutive_D = None  # type: ignore[assignment]
    _step_cell = None  # type: ignore[assignment]
    _d_final = None  # type: ignore[assignment]
    _welded_folded_offdiag_term = None  # type: ignore[assignment]
    _masked_row_sum = None  # type: ignore[assignment]


def folded_offdiag_fused_curl_constitutive_D_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if folded_offdiag_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded off-diagonal fused electric pair needs the optional "
            "`triton` package (pip install triton). The engine runs without it; "
            f"only this fast path is unavailable. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_offdiag_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def folded_offdiag_fused_electric_pair_coverage(fields: Any, pml: Any,
                                                sources: Any = None
                                                ) -> "_coverage.Coverage":
    """May ONE launch step this whole folded D seam?

    1. :func:`symmetry.folded_composition_curl_coverage` on ``step_D`` — the
       curl half is the certified folded curl arm, lifted, and the COMPOSITION
       verdict is the one that matters. The standalone folded predicates
       deliberately admit an UNFOLDED grid so their own gates can prove
       reduction to the certified unfolded kernels; using those here would make
       this product and ``offdiag_fused_electric_pair`` both admit every
       unfolded off-diagonal run, and ``launch._select_slot`` empties a slot
       with more than one admitter. Two products admitting one configuration is
       not a routing question to settle by branch order — it is an ambiguity,
       and the composition split is where this package settles it.
    2. :func:`folded_offdiag_update_e.folded_offdiag_composition_coverage` —
       the constitutive half, under the same split and for the same reason.
    3. **No electric source in the seam** (:data:`CARRIES_DEPOSIT_REPAIR` is
       False, and the reason is the stencil's footprint, not the wiring).
    4. **No axis stores MEEP's far ghost slot.** That is
       ``fill_folded_far_ghosts_D``'s own liveness condition
       (``stepping._stored_past_owned``); this weld does not carry that pass, so
       an axis on which it would write anything is refused BY NAME. Asked as the
       pass's condition rather than as a boundary code, so a folded PERIODIC axis
       with no extra stored slot — where the pass writes nothing — stays served.
    5. **Every fold plane reports a readable phase**, because the fill's parity
       is a compile-time constant taken from it; a zero would silently install
       neither ``+1`` nor ``-1``.
    6. The grid answers the wall question, and the six rotating volumes exist.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    curl = _symmetry.folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    constitutive = folded_offdiag_composition_coverage(fields, pml)
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

    for axis, stores_far in enumerate(far_ghost_axes(grid)):
        if stores_far:
            reasons.append(
                f"axis {axis} stores MEEP's far ghost slot, so "
                f"fill_folded_far_ghosts_D writes inside this seam and this "
                f"weld does not carry that pass")

    for axis, phase in enumerate(mirror_fill_phases(grid)):
        if phase not in (1, -1):
            reasons.append(
                f"axis {axis} is folded but reports no readable mirror phase, "
                f"so the near fill's parity constant cannot be resolved")

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


def explain_folded_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads."""
    return folded_offdiag_fused_electric_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedOffdiagFusedElectricPairPlan(ScratchWeldPairPlan):
    """One allocation-free launch for the whole folded D seam.

    The unfolded plan's contract, plus two per-fold inputs: the RUNTIME ghost
    weights the constitutive half's down shift takes, and the COMPILE-TIME plane
    phases the fill's redirect takes. They are validated against each other and
    against the boundary codes, because a fold that carried one without the other
    is a plane of wrong values and not a crash.
    """

    __slots__ = ("dtdx", "backward", "bc", "zero_metal", "wall_mask", "row_mask",
                 "mirror_axes", "phases", "ghost_weights", "_sources", "_pml",
                 "_e_targets", "_aux", "_inv_eps", "_rows", "_coeff", "_kernel",
                 "_pointer")

    replaces = REPLACES

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 zero_metal: Sequence[bool], wall_mask: Sequence[int],
                 phases: Sequence[int], ghost_weights: Sequence[float],
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
        self.phases = tuple(int(phase) for phase in phases)
        self.ghost_weights = tuple(float(weight) for weight in ghost_weights)
        self.mirror_axes = mirror_ghost_axes(self.bc)
        if not (len(self.bc) == len(self.zero_metal) == len(self.wall_mask)
                == len(self.phases) == len(self.ghost_weights) == 3):
            raise ValueError("this plan needs three of every per-axis input")
        for axis in range(3):
            folded = bool(self.mirror_axes[axis])
            if folded and self.phases[axis] not in (1, -1):
                raise ValueError(
                    f"axis {axis} is folded (code {self.bc[axis]}) and its fill "
                    f"phase is {self.phases[axis]}; the near redirect would "
                    f"carry neither +1 nor -1")
            if folded and self.ghost_weights[axis] not in (1.0, -1.0):
                raise ValueError(
                    f"axis {axis} is folded and its ghost weight is "
                    f"{self.ghost_weights[axis]}; the down shift would weight "
                    f"the ghost lane by something that is not a parity")
            if self.zero_metal[axis] and folded:
                raise ValueError(
                    f"axis {axis} is both folded and declared for the wall "
                    f"clear; `_zero_metal` skips a folded axis for a reason "
                    f"worth 1.28e+00 relative L2 (stepping.py:2257-2277)")
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
                "plain folded fused pair's and the predicate refuses it first")
        self.row_mask = tuple(1 if row is not None else 0 for row in rows)
        self._rows = tuple(
            CupyPointer(row) if row is not None
            else CupyPointer(tuple(e_targets)[index // 2])
            for index, row in enumerate(rows))
        self._coeff = tuple(CupyPointer(_flat(a)) for a in coefficients)
        _refuse_aliases(FAMILY, twins, sources, e_targets, aux, inverse_epsilon,
                        rows, pml_columns, coefficients)
        self._kernel = kernel

    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_offdiag_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        scratch = [self._pointer(array) for array in writes]
        prior = [self._pointer(array) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._sources, *self._pml,
            *self._e_targets, *self._aux, *self._inv_eps, *self._rows,
            *self._coeff, *self.ghost_weights,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            MG_X=self.mirror_axes[0], MG_Y=self.mirror_axes[1],
            MG_Z=self.mirror_axes[2],
            PH_X=self.phases[0], PH_Y=self.phases[1], PH_Z=self.phases[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            WM_X=self.wall_mask[0], WM_Y=self.wall_mask[1],
            WM_Z=self.wall_mask[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedOffdiagFusedElectricPairPlan(shape={self.shape}, "
                f"bc={self.bc}, mirror={self.mirror_axes}, phases={self.phases}, "
                f"zero_metal={self.zero_metal}, row_mask={self.row_mask}, "
                f"block={self.block})")


def _refuse_aliases(family: str, twins, sources, e_targets, aux,
                    inverse_epsilon, rows, pml_columns, coefficients) -> None:
    """No output may share an allocation with an input or another output.

    Shared with the unfolded family through :mod:`offdiag_fused_electric_pair`'s
    own check being the same question; kept as a function here rather than
    imported so each family's slot list is the one it actually binds.
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
    named = [(name, twins[name]) for name in ROTATED]
    named += [(f"E{index}", array) for index, array in enumerate(e_targets)]
    named += [(f"f_w{index}", array) for index, array in enumerate(aux)]
    for label, array in named:
        key = address(array)
        if key is None:
            continue
        if key in outputs:
            raise ValueError(
                f"{family}: outputs {outputs[key]} and {label} are the same "
                f"allocation; one launch would write both")
        outputs[key] = label
    inputs: List[Tuple[str, Any]] = [
        (f"source{index}", array) for index, array in enumerate(sources)]
    inputs += [(f"inv_eps{index}", array)
               for index, array in enumerate(inverse_epsilon)]
    inputs += [(f"row{index}", array) for index, array in enumerate(rows)
               if array is not None]
    inputs += [(f"pml{index}", array) for index, array in enumerate(pml_columns)]
    inputs += [(f"coefficient{index}", array)
               for index, array in enumerate(coefficients)]
    for label, array in inputs:
        key = address(array)
        if key is not None and key in outputs:
            raise ValueError(
                f"{family}: input {label} aliases output {outputs[key]}: the "
                f"launch would read a volume it is writing")


def plan_folded_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        ) -> Optional[FoldedOffdiagFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused."""
    if not folded_offdiag_fused_electric_pair_coverage(
            fields, pml, sources).covered:
        return None
    grid = fields.grid
    codes, _reasons = _symmetry.folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    return FoldedOffdiagFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx, codes,
        _coverage.zero_metal_axes(grid),
        wall_mask_axes(grid),
        mirror_fill_phases(grid),
        mirror_ghost_weights(grid),
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
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_offdiag_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, codes: Sequence[int],
        zero_metal: Sequence[bool], wall_mask: Sequence[int],
        phases: Sequence[int], ghost_weights: Sequence[float],
        rows: Sequence[Any], fields: Any, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1
        ) -> FoldedOffdiagFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    ``fields`` is the object whose attributes the ROTATION swaps; the gate
    passes a namespace holding the six rotating volumes under the engine's own
    names, so the rotation the gate exercises is the rotation the engine gets.
    ``kernel=`` carries the mutation override and dropping it disarms every
    mutation leg.
    """
    shape = tuple(int(n) for n in arrays[CURL_TARGETS[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return FoldedOffdiagFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, wall_mask, phases, ghost_weights,
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
