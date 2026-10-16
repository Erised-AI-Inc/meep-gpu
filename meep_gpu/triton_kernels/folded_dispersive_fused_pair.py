"""The folded DISPERSIVE electric seam in one launch: ``step_D`` into ``update_E``.

THE CELL. ``(folded PML, folded dispersive)`` on ``D->E`` — four corpus rows,
``tests:TestLoadDump.test_load_dump_{structure,structure_sharded,chunk_layout_file,
chunk_layout_sim}_2d``, measured at ``D->E 4, ceiling 4`` in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-31_cells``, where it is the
LARGEST remaining "reachable; NO PRODUCT OCCUPIES THIS CELL — not built" cell on the
Triton board. No sibling on any backend performs this conjunction: it is a genuine
new weld rather than a port, and both halves were separately certified before it.

WHY NEITHER EXISTING PRODUCT REACHES IT, and neither refusal is wrong
====================================================================

* :func:`.folded_fused_pair.folded_fused_pair_coverage` refuses a registered
  susceptibility BY NAME (folded_fused_pair.py:1189-1193) — "this kernel bakes the
  plain constitutive product, whose source is D and not (D - sum P)". True of its
  kernel;
* :func:`.dispersive_fused_pair.dispersive_fused_pair_coverage` refuses the fold,
  inherited from ``coverage._grid_reasons`` clause 5, because its curl half is the
  unfolded ``pml_curl_step`` and it carries none of the four passes in a folded seam.

So the conjunction belongs to neither file, exactly as
:mod:`.folded_dispersive_update_e` belongs to neither ``symmetry.py`` nor
``dispersive_update_e.py``. This module is the SEAM twin of that one: that module
composed the two ADMISSIONS for the ``update_E`` slot alone; this one welds the
folded curl to it.

EVERYTHING HERE IS TRANSCRIBED, and every arithmetic line carries its source
====================================================================

* the folded ``step_D`` curl, the near fill, the wall clear and the far fill —
  :func:`.folded_fused_pair.folded_fused_curl_constitutive_D`, statement for
  statement from its opening index arithmetic through its last carry block. That
  body is itself the transcription of :func:`.symmetry.pml_curl_step_folded` plus
  ``stepping._write_mirror_ghost`` and ``stepping._fill_folded_far_ghosts``, all
  released under ``results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/``;
* the pole chain — :func:`.dispersive_update_e.constitutive_step_dispersive`, whose
  eight ``if NP > k:`` arms transcribe ``Fields.displacement_minus_polarization``
  (fields.py:1097) in ``fields.polarizations`` order.

A reader can diff this body against those two and see that nothing moved. If you
find yourself deriving here, you have taken a wrong turn.

WHAT ACTUALLY MOVES, AND IT IS ONE SUBSTITUTION IN TWO PLACES
=============================================================

The certified folded body computes the constitutive source as ``v * inv_eps`` at an
owned cell (folded_fused_pair.py:745) and ``ghost * inv_eps`` at a carried one
(:487, inside ``_carry_ghost_E``). The dispersive body computes it as
``(D - P0 - P1 - ...) * inv_eps`` (dispersive_update_e.py:206-224). The weld is that
one substitution, made in exactly those two places and nowhere else:

    owned cell   src = (v  - sum P[idx]) * inv_eps[idx]
    ghost cell   src = (gh - sum P[dst]) * inv_eps[dst]

**THE POLES ARE READ AT THE DESTINATION INDEX, NOT AT THE SOURCE LANE'S.** That is
the whole content of the ghost half and it is a fact about the array path rather
than a choice: ``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` write ``D``
only (stepping.py:1497-1498, :1529-1532) — they do not touch ``P`` — and
``update_E`` then runs over the WHOLE stored extent, so at a ghost cell it reads
that cell's own ``P``. Imaging the source lane's poles instead would be a smooth,
plausible, wrong field on every folded plane, and the gate arms a mutation for it.

THE ORDER IS BIT-LOAD-BEARING. The poles may not be pre-summed and may not be
reordered: either changes float32 rounding at two or more poles, which is why the
certified body spells eight separate arms and why the gate arms a mutation that
reverses them. ``dispersive_update_e``'s own recon caught pre-summing 20/20 at two
poles.

THE FIVE DRIVER PASSES, AND THE SEAM CLAUSES
============================================

Identical to the plain folded pair's, and for identical reasons — the driver runs
the same five call sites whatever the constitutive source is
(driver.py:3292-3304). ``step_D`` -> electric injection -> ``fill_symmetry_bc_D`` ->
``zero_metal_D`` -> ``fill_folded_far_ghosts_D`` -> ``update_E``. Three of the four
inner passes are CARRIED INLINE; the fourth is the injection, and it is carried by
the deposit repair.

**CARRIES_DEPOSIT_REPAIR IS TRUE AND THE CELL IS WORTH NOTHING WITHOUT IT.** All
four corpus rows declare ``source_field_types == ['D']`` — an ELECTRIC source,
injected between the two halves — so under the source clause as
:mod:`.complex_conductive_fused_pair` states it (repair False) this arm would admit
ZERO of the four. The flag is declared in the SAME change as the wiring that
brackets the launch (:class:`.deposit_repair.LeadingRepairPlan` /
:class:`.deposit_repair.TrailingRepairPlan` through :data:`REPLACES`), never before
it and never after.

THE REPAIR ALREADY INVERTS THIS CONSTITUTIVE, and that was read rather than
assumed: :func:`.deposit_repair.apply` recomputes the D seam as
``fields.displacement_minus_polarization(component) * inverse_epsilon_for(component)``
(deposit_repair.py:551-553) — the dispersive source, not ``D`` — so nothing about
the pole chain needs a new repair path. What it refuses stays refused by name: an
off-diagonal row, a nonlinear one, a cylindrical ``r = 0`` axis, a fold too short to
hold the near fill's source row, and an absorber whose split-field recurrence never
ran.

WHY THERE IS NO STENCIL PROBLEM HERE
====================================

The board refuses eighteen ``D->E`` seam-instances as STRUCTURALLY UNFUSABLE on any
backend: ``stepping._offdiagonal_terms`` (stepping.py:1235-1253) reads each partner
component's ``D`` volume at four indices while ``step_D`` writes those volumes in
place in the first half of the same launch. THIS CONSTITUTIVE IS NOT THAT ONE — it
is element-wise in ``D`` and element-wise in every ``P`` — so the value stays in a
register at an owned cell and comes from this lane's own composed ghost at a carried
one. The off-diagonal row is refused by the E half already
(``folded_dispersive_update_e``'s own clause, which cites the measured 13824/69120
divergence) and is restated here by name, because THIS is the fact that makes the
cell buildable at all.

**THE POLE ARRAYS ARE READ, NEVER WRITTEN, by either half.** ``update_P`` runs after
``update_E`` (driver.py:3306) and is not in this seam; the ``P`` volumes this kernel
loads are the ones the previous timestep left, exactly as the certified dispersive
body loads them. That is also why the D STORE IS KEPT: the array path leaves the
stepped displacement in the D volume and ``update_P`` reads it next.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans five driver call sites across two of
them; there is no slot it can claim without a composition rule nothing has measured.
Nothing in ``launch.py`` names this module, ``fastpath.plan_fast_path`` is
unchanged, and dispatch stays disabled — the same deferral
:mod:`.complex_fused_electric_pair`, :mod:`.cylindrical_fused_electric_pair` and
:mod:`.complex_conductive_fused_pair` ship under.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .folded_dispersive_update_e import folded_dispersive_constitutive_coverage
from .symmetry import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    _far_reflect_rows,
    _stored_past_owned_reader,
    folded_axis_kinds,
    folded_composition_curl_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between four seam-instances and zero: every one of
#: the four corpus rows declares an electric source, which the driver injects between
#: the two halves (driver.py:3294-3299). The flag is a claim about the PLAN this
#: module builds -- that the leading slot saves and the trailing slot restores -- and
#: is only ever changed in the same edit as that wiring. It reaches
#: ``deposit_repair.repairable``, which goes on refusing BY NAME every seam the
#: repair cannot invert.
CARRIES_DEPOSIT_REPAIR = True

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name, and identical
#: to the plain folded pair's: a susceptibility changes the constitutive SOURCE, not
#: which passes sit in the seam.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The two codes a folded axis can take. The NEAR fill runs on either
#: (``stepping._fill_symmetry_ghost_cells`` gates on the mirror phases alone); the
#: FAR fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``).
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding is
#: visible without importing :mod:`launch`; the host suite pins the two equal.
BACKWARD = 1

#: Which folded axes the near fill images for each D component, and which the FAR
#: fill does. Derived from ``TARGET_IYEE`` at import rather than written out, and
#: pinned by test against ``fields.IYEE_SHIFTS`` — the same derivation
#: :mod:`.folded_fused_pair` makes, from the same table, because the geometry is the
#: same geometry.
NEAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 0)
    for name in ("Dx", "Dy", "Dz")
)
FAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 1)
    for name in ("Dx", "Dy", "Dz")
)

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAR_FILL_AXES", "MIRROR_CODES", "NEAR_FILL_AXES", "REPLACES",
    "FoldedDispersiveFusedPairPlan",
    "folded_dispersive_fused_curl_constitutive_D",
    "folded_dispersive_fused_curl_constitutive_D_kernel",
    "folded_dispersive_fused_pair_coverage",
    "mirror_phases",
    "plan_folded_dispersive_fused_pair",
    "plan_folded_dispersive_fused_pair_from_arrays",
]


if triton is not None:

    # The forms of the root kernels (forms.py) and the chunk record's two word
    # offsets, bound under kernels.py's names, as the folded pair binds them.
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def _subtract_poles_at(base, p0, p1, p2, p3, p4, p5, p6, p7,
                           where, mask, NP: tl.constexpr):
        """``base`` minus each driving pole's ``P``, at ``where``, in exact order.

        :func:`.dispersive_update_e.constitutive_step_dispersive`'s eight
        ``if NP > k:`` arms with the opening ``tl.load(g + idx)`` replaced by the
        ``base`` argument and the load index made explicit — the one substitution
        this weld makes, written ONCE so the reader and the gate's transcription leg
        can compare it to the certified arms statement for statement rather than to
        four inlined copies of them.

        ``where`` IS THE CELL WHOSE POLES ARE READ, and it is the DESTINATION on
        every call: an owned lane passes its own ``idx``, a carried ghost passes
        ``dst``. The array path's fills write ``D`` and never ``P``
        (stepping.py:1497-1498, :1529-1532), so ``update_E`` at a ghost reads that
        ghost's own poles.

        THE ORDER IS BIT-LOAD-BEARING: the poles may not be pre-summed and may not be
        reordered — either changes float32 rounding at two or more poles. The gate
        arms a mutation that reverses them and one that pre-sums them.
        """
        if NP > 0:
            base = base - tl.load(p0 + where, mask=mask, other=0.0)
        if NP > 1:
            base = base - tl.load(p1 + where, mask=mask, other=0.0)
        if NP > 2:
            base = base - tl.load(p2 + where, mask=mask, other=0.0)
        if NP > 3:
            base = base - tl.load(p3 + where, mask=mask, other=0.0)
        if NP > 4:
            base = base - tl.load(p4 + where, mask=mask, other=0.0)
        if NP > 5:
            base = base - tl.load(p5 + where, mask=mask, other=0.0)
        if NP > 6:
            base = base - tl.load(p6 + where, mask=mask, other=0.0)
        if NP > 7:
            base = base - tl.load(p7 + where, mask=mask, other=0.0)
        return base

    @triton.jit
    def _carry_ghost_E_dispersive(f, w, e, ie,
                                  p0, p1, p2, p3, p4, p5, p6, p7,
                                  dst, ghost, kp_d, km_d, dw_d, mask,
                                  NP: tl.constexpr):
        """Write ONE imaged ghost cell of ``D`` and run dispersive ``update_E`` there.

        :func:`.folded_fused_pair._carry_ghost_E` with its ``src`` line split into
        the certified pole chain and the same inverse-epsilon multiply. Everything
        else is that function's statements, moved and not rewritten: the
        displacement store, the workspace read BEFORE the write and the workspace
        write on the destination's dsigw cells only, and the ``E`` form keyed on the
        destination's membership ``dw_d`` (:func:`.forms.constitutive_form`).

        ``ghost`` is ALREADY the composed value — parity applied, and the wall clear
        applied or not according to which fill wrote it. ``D`` IS ON THE LEFT of the
        inverse-epsilon multiply because the array path writes ``source *
        fields.inverse_epsilon_for(component)``.
        """
        tl.store(f + dst, ghost, mask=mask)
        prev = tl.load(w + dst, mask=mask & dw_d, other=0.0)
        s = _subtract_poles_at(ghost, p0, p1, p2, p3, p4, p5, p6, p7,
                               dst, mask, NP)
        src = s * tl.load(ie + dst, mask=mask, other=0.0)
        tl.store(w + dst, src, mask=mask & dw_d)
        acc = tl.load(e + dst, mask=mask & dw_d, other=0.0)
        acc = constitutive_form(acc, prev, src, kp_d, km_d, dw_d)
        tl.store(e + dst, acc, mask=mask)

    @triton.jit
    def folded_dispersive_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx, Dy, Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz
        g0, g1, g2,                       # curl sources: Hx, Hy, Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex, Ey, Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex, f_w_Ey, f_w_Ez
        ie0, ie1, ie2,                    # inverse epsilon, per target
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        a0, a1, a2, a3, a4, a5, a6, a7,   # Ex poles, registration order
        b0, b1, b2, b3, b4, b5, b6, b7,   # Ey poles, registration order
        c0, c1, c2, c3, c4, c5, c6, c7,   # Ez poles, registration order
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + the two fills + the wall + DISPERSIVE ``update_E``.

        :func:`.folded_fused_pair.folded_fused_curl_constitutive_D`, statement for
        statement, with ONE substitution in two places: the constitutive source is
        ``(v - P0 - P1 - ...) * inv_eps`` at an owned cell and ``(ghost - sum P[dst])
        * inv_eps[dst]`` at a carried one (:func:`_carry_ghost_E_dispersive`), the
        poles in ``fields.polarizations`` order (:func:`_subtract_poles_at`). Every
        PML form is that kernel's: the record ``cb`` picks the curl expression, the
        recurrence on the dsigu cells and the constitutive accumulation on the dsigw
        cells, each auxiliary read and written on its own form's cells only, every
        select evaluated where its cell is, a carried ghost included.

        ``BACKWARD`` IS CARRIED AND MUST BE 1, for the folded pair's reason.

        ``NP0``/``NP1``/``NP2`` are the LIVE pole count per component and are
        ``tl.constexpr``, so the subtraction chain is unrolled at compile time and a
        component with no pole compiles to EXACTLY the plain folded body. Unused pole
        slots are bound to the component's own ``D`` pointer by the plan and are
        never read.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from symmetry.pml_curl_step_folded; the only edits below its
        # stores are the ownership masks the near-fill carry needs.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # PERIODIC wraps; every other rule serves an exact 0.0 past the face. On a
        # folded axis that zero stands in for a value nothing reads: its only
        # consumer is the plane the cell-0 mask below zeroes.
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
        # kernels.pml_curl_step's three calls: MEEP's left-to-right association with
        # both partners, the one-partner form where the record says the other
        # partner's derivative axis is invariant, an exact +0 with neither.
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
        curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
        curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells :1865) -----
        # Every target whose Yee shift is 0 on a non-periodic axis. `!= PERIODIC`
        # rather than `== METALLIC`: _mask_non_owned_cells asks `is_mirrored or
        # is_metallic or is_axis`, and _boundary_kinds resolves exactly those to a
        # non-periodic kind.
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

        # --- ownership mask, the TOP plane of a folded PERIODIC axis -----------
        # symmetry.pml_curl_step_folded's second added block, BYTE-COPIED from its
        # BACKWARD == 1 arm (symmetry.py:294-301). It was absent while this family
        # refused MIRROR_PERIODIC; carrying fill_folded_far_ghosts_D brings the axis
        # in, so the block comes with it. Dx:(1,0,0) Dy:(0,1,0) Dz:(0,0,1) — shift 1
        # on its OWN axis only, which is exactly the set the far carry writes, and
        # three lines against the B arm's six.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BCX == MIRROR_PERIODIC:
            curl0 = tl.where(last_x, 0.0, curl0)
        if BCY == MIRROR_PERIODIC:
            curl1 = tl.where(last_y, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl2 = tl.where(last_z, 0.0, curl2)

        # --- ownership of the fill destinations --------------------------------
        # Component m is a NEAR destination at stored cell 0 of every folded axis
        # that is NOT its own and a FAR destination at the top plane of its own axis
        # when that axis is folded PERIODIC (IYEE_SHIFTS: Dx (1,0,0), Dy (0,1,0),
        # Dz (0,0,1) — shift 0 on the two others, 1 on its own). Those lanes do NOT
        # load or store D, E or f_w_E: the source lane writes all three for them.
        # The masks are what make the carry race-free — a lane that merely discarded
        # the value would still have READ a word another lane writes.
        own0, own1, own2 = live, live, live
        if NEAR_Y:
            own0 = own0 & (j != 0)
            own2 = own2 & (j != 0)
        if NEAR_Z:
            own0 = own0 & (k != 0)
            own1 = own1 & (k != 0)
        if NEAR_X:
            own1 = own1 & (i != 0)
            own2 = own2 & (i != 0)
        if FAR_X:
            own0 = own0 & (i != nx - 1)
        if FAR_Y:
            own1 = own1 & (j != ny - 1)
        if FAR_Z:
            own2 = own2 & (k != nz - 1)

        # --- the PML forms (stepping._apply_curl_regions) ---------------------
        # The PML coefficient vectors are built at the STORED extent on a folded
        # axis, so the n_a indexing needs no fold-aware change.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # Which cells take the recurrence: the component's dsigu layer on its own
        # sub-lattice, verbatim from the folded curl (integer here, BACKWARD = 1).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=own0, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)

        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=own1, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)

        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=own2, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Byte-copied from dispersive_fused_pair's ZM block — the same six rows
        # _zero_metal writes for the D family (Dy/Dz on an x wall, Dx/Dz on y,
        # Dx/Dy on z). Applied to the REGISTER, before both the store and the
        # constitutive read, so the two consumers see the one value the array path
        # leaves in D. A walled axis is never a folded axis (_zero_metal
        # :2237-2239), so a wall's cleared plane (cell 0 of a walled axis) is never
        # a fold's destination plane; the predicate CHECKS that rather than leaning
        # on it. It does NOT follow that the clear and the near fill are unrelated
        # on this family — a component folded on y is still cleared on a z wall, and
        # that is why each near-ghost block below re-applies these lines AFTER its
        # parity, in the driver's order.
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        # fu is written on the recurrence's cells ONLY (the auxiliary contract),
        # destinations included: the array path's step_D writes it on every such
        # cell, a ghost row reading its image row's membership, and neither fill
        # touches it (the fills write `field`, never `fu_field`).
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=own0)
        tl.store(f1 + idx, v1, mask=own1)
        tl.store(f2 + idx, v2, mask=own2)

        # ==================== the constitutive half ===========================
        # folded_fused_pair's owned-cell block with `v * inv_eps` replaced by
        # `(v - sum P) * inv_eps` (dispersive_update_e's eight arms, at this lane's
        # OWN index, through _subtract_poles_at). That block is, in turn,
        # verbatim from kernels.constitutive_step's SCALE=1 arm, with
        # `tl.load(g + idx)` replaced by the register the curl half just produced
        # and `live` by the component's ownership mask. The form is keyed on the
        # dsigw line of the component's own axis on the half-integer lattice.
        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z
        # (stepping.E_CONSTITUTIVE_TERMS :227) — the component's OWN axis, which
        # is exactly the axis the NEAR fill never images along and exactly the one
        # the FAR fill does. That is what lets a near-only destination reuse these
        # six registers unchanged and what makes the far half load its own pair.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=own0 & dw0, other=0.0)   # BEFORE the store.
        s0 = _subtract_poles_at(v0, a0, a1, a2, a3, a4, a5, a6, a7,
                                idx, own0, NP0)
        src0 = s0 * tl.load(ie0 + idx, mask=own0, other=0.0)
        tl.store(w0 + idx, src0, mask=own0 & dw0)
        a0v = tl.load(e0 + idx, mask=own0 & dw0, other=0.0)
        a0v = constitutive_form(a0v, prev0, src0, kp_0, km_0, dw0)
        tl.store(e0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1 & dw1, other=0.0)
        s1 = _subtract_poles_at(v1, b0, b1, b2, b3, b4, b5, b6, b7,
                                idx, own1, NP1)
        src1 = s1 * tl.load(ie1 + idx, mask=own1, other=0.0)
        tl.store(w1 + idx, src1, mask=own1 & dw1)
        a1v = tl.load(e1 + idx, mask=own1 & dw1, other=0.0)
        a1v = constitutive_form(a1v, prev1, src1, kp_1, km_1, dw1)
        tl.store(e1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2 & dw2, other=0.0)
        s2 = _subtract_poles_at(v2, c0, c1, c2, c3, c4, c5, c6, c7,
                                idx, own2, NP2)
        src2 = s2 * tl.load(ie2 + idx, mask=own2, other=0.0)
        tl.store(w2 + idx, src2, mask=own2 & dw2)
        a2v = tl.load(e2 + idx, mask=own2 & dw2, other=0.0)
        a2v = constitutive_form(a2v, prev2, src2, kp_2, km_2, dw2)
        tl.store(e2 + idx, a2v, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # stepping._write_mirror_ghost (:1451):  D[..0..]  = +phase * D[..2..]
        # stepping._fill_folded_far_ghosts (:1529-1532): D[..-1..] = -phase * D[..r..]
        # and every COMPOSITION of the two, which is what a cell that is a near
        # destination on one axis and a far one on another carries.
        #
        # THE COMPOSITION IS CHAINED, NOT A PRODUCT OF PARITIES, and on this family
        # that is a byte-level distinction rather than a stylistic one. The driver
        # runs fill_symmetry_bc_D (:3300), then zero_metal_D (:3301), then
        # fill_folded_far_ghosts_D (:3302), so
        #
        #     a NEAR ghost is  clear(parity * v)  — the clear applied AFTER the
        #                                           parity, which is why each block
        #                                           below re-applies the ZM lines;
        #     a FAR  ghost is  parity * (the value already standing at its source),
        #                                           with NO clear after it, because
        #                                           :3302 is the last pass before
        #                                           update_E;
        #     a FAR-over-NEAR cell is the far parity times THE NEAR GHOST THIS LANE
        #                                           JUST COMPOSED — not the raw
        #                                           product -PH_m * PH_a * v.
        #
        # MEASURED, NOT ARGUED. parity/meep_gpu/measure_d_side_composition_law.py
        # scores this rule and four transcription errors against the array path's
        # own three passes over 156 grid declarations: this rule 156/156, and the
        # four alternatives caught on 84, 36, 36 and 70 cases respectively, with no
        # vacuous case. Reversing either order, or flattening the composite, is one
        # of those four.
        #
        # THE PARITY IS +PHASE ON A SHIFT-0 COMPONENT AND -PHASE ON A SHIFT-1 ONE
        # (fields.mirror_parity :117 is `phase * (1 - 2 * iyee)`), so the near carry
        # takes `PH?` and the far carry `-PH?`, and +-1.0 is exact in float32.
        #
        # THE COEFFICIENT INDEX MOVES ONLY FOR THE FAR HALF. update_E indexes
        # component m on axis m (stepping.E_CONSTITUTIVE_TERMS :227); the near fill
        # images along an axis that is NOT m, so its destination sits at the SAME
        # coefficient index as the lane that owns it and reuses that lane's pair —
        # a reload there would reload the identical word. The far fill images along
        # exactly axis m, so its destination reads kps/kms at the TOP row rather
        # than reusing the source lane's at the reflect row.
        #
        # THE NEAR SOURCE INDEX IS THE LITERAL 2, as it is in
        # symmetry.mirror_ghost_fill (`base + 2 * stride`), rather than the named
        # MIRROR_SOURCE_INDEX: a jit body that closes over a module-level Python int
        # is a Triton-version question this file has no way to measure without a
        # device. The literal is pinned to the name by test, so the two cannot drift.
        #
        # The destination coefficient pair for the FAR half, per component, at the
        # top row of that component's own axis. A scalar `nx - 1` would not
        # broadcast against a masked vector load, so it is built as a per-lane
        # tensor the same way the magnetic twin builds its `origin`.
        top_x = idx * 0 + (nx - 1)
        top_y = idx * 0 + (ny - 1)
        top_z = idx * 0 + (nz - 1)
        kp_f0 = tl.load(kp0 + top_x, mask=live, other=0.0)
        km_f0 = tl.load(km0 + top_x, mask=live, other=0.0)
        kp_f1 = tl.load(kp1 + top_y, mask=live, other=0.0)
        km_f1 = tl.load(km1 + top_y, mask=live, other=0.0)
        kp_f2 = tl.load(kp2 + top_z, mask=live, other=0.0)
        km_f2 = tl.load(km2 + top_z, mask=live, other=0.0)
        # ... and its membership there, for the same reason: the FAR destination's
        # own-axis row is the top row, so its select is keyed on the top row's word.
        # A NEAR-only destination keeps the source lane's coordinate on that axis,
        # and so its flag (dw0, dw1, dw2 above).
        dw_f0 = record_in(cb, RECORD_HALF_INTEGER + 0, top_x)
        dw_f1 = record_in(cb, RECORD_HALF_INTEGER + 4, top_y)
        dw_f2 = record_in(cb, RECORD_HALF_INTEGER + 8, top_z)

        # The per-axis source lane and destination offset, once. Each is read only
        # under a mask its own axis's constexpr gates, so an axis that carries no
        # fill contributes no lane and no load.
        #
        # EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWN OWNERSHIP MASK, and
        # that is a DEFECT THE MAGNETIC TWIN'S DEVICE GATE FOUND rather than a
        # precaution (its results/run_farcarry2/, 2026-08-20). With two fills a lane
        # can be the SOURCE of one and the DESTINATION of the other: on a grid
        # folding x PERIODIC and y, the lane at (i = rx, j = 0) is the far source
        # for Dx on x and is itself the near fill's destination on y. It would then
        # write the composite cell with a `v` built from an `f0` load its own
        # ownership mask zeroed, racing the lane that legitimately owns it.
        near_i = live & (i == 2)
        near_j = live & (j == 2)
        near_k = live & (k == 2)
        far_i = live & (i == rx)
        far_j = live & (j == ry)
        far_k = live & (k == rz)
        dn_x = -2 * nyz
        dn_y = -2 * nz
        dn_z = -2
        df_x = (nx - 1 - rx) * nyz
        df_y = (ny - 1 - ry) * nz
        df_z = (nz - 1 - rz)

        # THE THREE-TERM GUARDS ARE PARENTHESISED, AND THAT IS A MEASURED PLATFORM
        # FACT rather than a style choice: Triton 3.1.0's frontend refuses a chained
        # boolean outright — "chained boolean operators (A or B or C) are not
        # supported; use parentheses to split the chain" — and it refuses at COMPILE
        # time on the device, so `if NEAR_Y and NEAR_Z and FAR_X:` is a kernel that
        # does not build rather than one that builds wrong.

        # --- component 0 (Dx): near on y and z, far on x -----------------------
        # The near ghosts are composed FIRST and kept in registers, because the
        # far-over-near cells below are the far parity applied to exactly these
        # values. The ZM lines re-applied to each are zero_metal_D's six rows for
        # this component (Dx clears on a y wall and on a z wall), in the driver's
        # parity-then-clear order.
        if NEAR_Y:
            gv_0y = PHY * v0
            if ZM_Y:
                gv_0y = tl.where(at_y, 0.0, gv_0y)
            if ZM_Z:
                gv_0y = tl.where(at_z, 0.0, gv_0y)
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + dn_y, gv_0y,
                                      kp_0, km_0, dw0, own0 & near_j, NP0)
        if NEAR_Z:
            gv_0z = PHZ * v0
            if ZM_Y:
                gv_0z = tl.where(at_y, 0.0, gv_0z)
            if ZM_Z:
                gv_0z = tl.where(at_z, 0.0, gv_0z)
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + dn_z, gv_0z,
                                      kp_0, km_0, dw0, own0 & near_k, NP0)
        if NEAR_Y and NEAR_Z:
            # The doubly-unowned corner. Y is applied before Z
            # (_fill_symmetry_ghost_cells :1441-1447), so the value is
            # phase_z * (phase_y * v) — two exact +-1.0 multiplies, in the array
            # path's own order, then the clear.
            gv_0yz = PHZ * (PHY * v0)
            if ZM_Y:
                gv_0yz = tl.where(at_y, 0.0, gv_0yz)
            if ZM_Z:
                gv_0yz = tl.where(at_z, 0.0, gv_0yz)
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + dn_y + dn_z, gv_0yz,
                                      kp_0, km_0, dw0, own0 & near_j & near_k, NP0)
        if FAR_X:
            # NO ZM AFTER THE PARITY: :3302 is the last pass before update_E, so
            # the far ghost is -phase times the value standing at its source, which
            # is `v0` read after the clear lines above.
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + df_x, -PHX * v0,
                                      kp_f0, km_f0, dw_f0, own0 & far_i, NP0)
        if FAR_X and NEAR_Y:
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + df_x + dn_y, -PHX * gv_0y,
                                      kp_f0, km_f0, dw_f0, own0 & far_i & near_j, NP0)
        if FAR_X and NEAR_Z:
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + df_x + dn_z, -PHX * gv_0z,
                                      kp_f0, km_f0, dw_f0, own0 & far_i & near_k, NP0)
        if FAR_X and (NEAR_Y and NEAR_Z):
            _carry_ghost_E_dispersive(f0, w0, e0, ie0,
                                      a0, a1, a2, a3, a4, a5, a6, a7,
                                      idx + df_x + dn_y + dn_z, -PHX * gv_0yz,
                                      kp_f0, km_f0, dw_f0, own0 & far_i & near_j & near_k, NP0)

        # --- component 1 (Dy): near on x and z, far on y -----------------------
        if NEAR_X:
            gv_1x = PHX * v1
            if ZM_X:
                gv_1x = tl.where(at_x, 0.0, gv_1x)
            if ZM_Z:
                gv_1x = tl.where(at_z, 0.0, gv_1x)
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + dn_x, gv_1x,
                                      kp_1, km_1, dw1, own1 & near_i, NP1)
        if NEAR_Z:
            gv_1z = PHZ * v1
            if ZM_X:
                gv_1z = tl.where(at_x, 0.0, gv_1z)
            if ZM_Z:
                gv_1z = tl.where(at_z, 0.0, gv_1z)
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + dn_z, gv_1z,
                                      kp_1, km_1, dw1, own1 & near_k, NP1)
        if NEAR_X and NEAR_Z:
            # X is applied before Z: phase_z * (phase_x * v).
            gv_1xz = PHZ * (PHX * v1)
            if ZM_X:
                gv_1xz = tl.where(at_x, 0.0, gv_1xz)
            if ZM_Z:
                gv_1xz = tl.where(at_z, 0.0, gv_1xz)
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + dn_x + dn_z, gv_1xz,
                                      kp_1, km_1, dw1, own1 & near_i & near_k, NP1)
        if FAR_Y:
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + df_y, -PHY * v1,
                                      kp_f1, km_f1, dw_f1, own1 & far_j, NP1)
        if FAR_Y and NEAR_X:
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + df_y + dn_x, -PHY * gv_1x,
                                      kp_f1, km_f1, dw_f1, own1 & far_j & near_i, NP1)
        if FAR_Y and NEAR_Z:
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + df_y + dn_z, -PHY * gv_1z,
                                      kp_f1, km_f1, dw_f1, own1 & far_j & near_k, NP1)
        if FAR_Y and (NEAR_X and NEAR_Z):
            _carry_ghost_E_dispersive(f1, w1, e1, ie1,
                                      b0, b1, b2, b3, b4, b5, b6, b7,
                                      idx + df_y + dn_x + dn_z, -PHY * gv_1xz,
                                      kp_f1, km_f1, dw_f1, own1 & far_j & near_i & near_k, NP1)

        # --- component 2 (Dz): near on x and y, far on z -----------------------
        if NEAR_X:
            gv_2x = PHX * v2
            if ZM_X:
                gv_2x = tl.where(at_x, 0.0, gv_2x)
            if ZM_Y:
                gv_2x = tl.where(at_y, 0.0, gv_2x)
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + dn_x, gv_2x,
                                      kp_2, km_2, dw2, own2 & near_i, NP2)
        if NEAR_Y:
            gv_2y = PHY * v2
            if ZM_X:
                gv_2y = tl.where(at_x, 0.0, gv_2y)
            if ZM_Y:
                gv_2y = tl.where(at_y, 0.0, gv_2y)
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + dn_y, gv_2y,
                                      kp_2, km_2, dw2, own2 & near_j, NP2)
        if NEAR_X and NEAR_Y:
            # X is applied before Y: phase_y * (phase_x * v).
            gv_2xy = PHY * (PHX * v2)
            if ZM_X:
                gv_2xy = tl.where(at_x, 0.0, gv_2xy)
            if ZM_Y:
                gv_2xy = tl.where(at_y, 0.0, gv_2xy)
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + dn_x + dn_y, gv_2xy,
                                      kp_2, km_2, dw2, own2 & near_i & near_j, NP2)
        if FAR_Z:
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + df_z, -PHZ * v2,
                                      kp_f2, km_f2, dw_f2, own2 & far_k, NP2)
        if FAR_Z and NEAR_X:
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + df_z + dn_x, -PHZ * gv_2x,
                                      kp_f2, km_f2, dw_f2, own2 & far_k & near_i, NP2)
        if FAR_Z and NEAR_Y:
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + df_z + dn_y, -PHZ * gv_2y,
                                      kp_f2, km_f2, dw_f2, own2 & far_k & near_j, NP2)
        if FAR_Z and (NEAR_X and NEAR_Y):
            _carry_ghost_E_dispersive(f2, w2, e2, ie2,
                                      c0, c1, c2, c3, c4, c5, c6, c7,
                                      idx + df_z + dn_x + dn_y, -PHZ * gv_2xy,
                                      kp_f2, km_f2, dw_f2, own2 & far_k & near_i & near_j, NP2)

else:  # pragma: no cover - laptop path
    folded_dispersive_fused_curl_constitutive_D = None  # type: ignore[assignment]


def folded_dispersive_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_dispersive_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded dispersive fused electric D/E kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_dispersive_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def mirror_phases(grid: Any) -> Tuple[int, int, int]:
    """``PHX``/``PHY``/``PHZ``: the declared plane phase per axis, 0 where unfolded.

    :func:`.folded_fused_pair.mirror_phases`, reused rather than re-derived would be
    the better shape and is not available: that module is not imported here, because
    importing it would pull its Triton kernel definition onto a laptop through the
    module-scope ``try: import triton`` it also runs. The two bodies are pinned equal
    by the host suite (``test_mirror_phases_matches_the_folded_pairs``), which is the
    check a shared helper would have made unnecessary.
    """
    phases: List[int] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            phases.append(0)
            continue
        value = _call(grid, "mirror_phase", axis, default=None)
        phases.append(int(value) if value in (1, -1) else 0)
    return (phases[0], phases[1], phases[2])


def folded_dispersive_fused_pair_coverage(fields: Any, pml: Any,
                                          sources: Any = None) -> Coverage:
    """May ONE launch span folded ``step_D`` -> fills -> wall -> dispersive ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    construction :func:`.folded_fused_pair.folded_fused_pair_coverage` and
    :func:`.coverage.fused_pair_coverage` use.

    DISJOINT FROM ``folded_fused_pair`` BY CONSTRUCTION, in both directions and
    without either predicate being touched: that product refuses a registered
    susceptibility by name, and this one's E half REQUIRES one
    (``folded_dispersive_update_e``'s inverted clause). Two predicates admitting one
    slot is how a wrong body gets chosen at random, and ``_select_slot`` fails closed
    on it — the slot is left UNSELECTED and falls to the array path, a silent
    coverage LOSS rather than an error.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    electric = folded_dispersive_constitutive_coverage(fields, pml)
    if not electric.covered:
        reasons.extend(f"folded dispersive constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # electric source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. All four corpus rows of this cell
    # declare exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
    # NOTHING; it is True, and the plan below brackets its launch with the repair.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so an
    # undeclared `sources` is itself a REFUSAL. A MAGNETIC source is injected in the
    # B/H half and does not disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR FILL IS CARRIED, under exactly the conditions the plain folded
        # pair's carry rests on — the pass images the top stored slot from
        # stepping._far_reflect_rows' row, and the lane that owns that destination is
        # the one AT the reflect row, so a row outside the allocation is an
        # out-of-range write from a lane that owns neither cell. The row checks below
        # are symmetry.mirror_ghost_fill_coverage's own, restated here because THIS
        # family writes the destination from a different lane.
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        reader = _stored_past_owned_reader()
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
            if reader is None:  # pragma: no cover - folded_axis_kinds refused first
                reasons.append(
                    "stepping._stored_past_owned is not importable; the far carry "
                    "cannot be checked against the array path's own predicate")
                break
            past_owned = bool(reader(grid, axis))
            if periodic != past_owned:
                reasons.append(
                    f"axis {axis} code {int(code)} and "
                    f"stepping._stored_past_owned {past_owned} disagree; the kernel "
                    f"would mask the top plane and image the far ghost on different "
                    f"axes than the array path")
            if not periodic or len(shape) != 3:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(
                    f"axis {axis} is a folded PERIODIC axis with no reflect row "
                    f"from stepping._far_reflect_rows; the far carry has nothing "
                    f"to image")
                continue
            row, extent = int(row), int(shape[axis])
            if not (0 <= row < extent - 1):
                reasons.append(
                    f"axis {axis} reflect row {row} is outside [0, {extent - 1}); "
                    f"the far carry would image the plane it writes, or read "
                    f"outside the allocation")
            if row == 0:
                reasons.append(
                    f"axis {axis} reflect row is 0, the plane the NEAR fill writes: "
                    f"the far carry would image a ghost rather than an owned cell")
            if extent - 1 == MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} stores {extent} cells, so the far carry's write "
                    f"plane IS the near fill's read plane "
                    f"(cell {MIRROR_SOURCE_INDEX})")

        # THE NEAR FILL IMAGES STORED CELL 2 FROM THAT CELL'S OWN LANE. BOTH mirror
        # codes: the near fill runs on either.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                        f"the near fill images stored cell {MIRROR_SOURCE_INDEX} and "
                        f"this kernel images it from that cell's own lane")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. CHECKED rather than inferred:
        # an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall there; "
                    f"stepping._zero_metal skips a folded axis, so the two passes "
                    f"have drifted and the carry's disjointness no longer holds")

        # The phases the kernel bakes must be exactly the ones the array path would
        # use, re-derived through the builder's own function.
        phases = mirror_phases(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
                reasons.append(
                    f"axis {axis} is folded but its mirror phase resolves to "
                    f"{phases[axis]!r}, not +1 or -1")

    # The wall clear is carried inline, so the grid must be able to answer which axes
    # are walled; an unanswerable one compiles to ZM=False and silently skips a plane
    # the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A SUSCEPTIBILITY IS REQUIRED, restated here as well as in the E half. Without
    # one the source is D and the configuration is folded_fused_pair's; two
    # predicates admitting one slot leaves it UNSELECTED and takes the certified
    # fusion off the device along with this one.
    if not tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "no susceptibility is registered: update_E's source is D, not "
            "(D - sum P), and that configuration is folded_fused_pair's")

    # THE STENCIL, restated. The eighteen structurally unfusable D->E seam-instances
    # on this board are the OFF-DIAGONAL constitutive reading partner D volumes at
    # four indices (stepping.py:1235-1253) while step_D writes them in place. The E
    # half refuses that row already — with the measured 13824/69120 divergence on
    # exactly the folded dispersive off-diagonal configuration — and it is named
    # again because THIS is what makes the present cell buildable at all.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D "
            "volumes step_D writes in place (stepping.py:1235-1253), and no "
            "grid-wide barrier exists inside one launch")

    # THE POLE COUNTS MUST FIT the kernel's compiled slots. The E half already
    # refuses more than MAX_POLES; evaluated here as well so an over-ceiling
    # configuration is a refusal BY NAME rather than a ValueError at plan time, and
    # so the from-arrays route's ceiling and this one cannot drift apart.
    from .dispersive_update_e import (  # noqa: PLC0415
        E_TERMS, MAX_POLES, poles_per_component)
    try:
        order = poles_per_component(fields)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))
    for component, _displacement, _axis in E_TERMS:
        count = len(tuple(order.get(component, ())))
        if count > MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedDispersiveFusedPairPlan:
    """One allocation-free launch for FIVE of the driver's electric passes.

    :class:`.folded_fused_pair.FoldedFusedPairPlan` plus the pole binding, and the
    binding is the reason this is a class rather than a subclass of that one: THE
    POLE POINTERS ARE RESOLVED PER LAUNCH, never cached.
    ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` every step
    (dispersion.py:687-691), so a pointer captured at plan time is one timestep stale
    and stale in a way that still computes. What IS cached is pole ORDER, which is
    bit-load-bearing and which
    :class:`.dispersive_update_e.LivePoleBinding` refuses to let change after
    planning.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "phases",
        "reflect", "near", "far", "counts",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_poles",
        "_grid", "_kernel", "chunk_bounds", "_cb",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, poles: Any,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any) -> None:
        from .dispersive_update_e import E_TERMS, MAX_POLES  # noqa: PLC0415
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phases = tuple(int(value) for value in phases)
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT — the folded pair's rule, and
        # for its reason: a kernel that masked the top plane on one axis set and
        # imaged the far ghost on another is a plane of wrong values, not a crash.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        if not any(self.near):
            raise ValueError(
                "no axis is folded: this product exists to carry the mirror fills "
                "inside the D seam, and an unfolded dispersive grid belongs to "
                "dispersive_fused_pair")
        for axis, code in enumerate(self.bc):
            folded = code in MIRROR_CODES
            if folded and self.phases[axis] not in (1, -1):
                raise ValueError(
                    f"axis {axis} is folded but its baked phase is "
                    f"{self.phases[axis]!r}, not +1 or -1")
            if folded and self.zero_metal[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a wall clear; "
                    f"stepping._zero_metal skips a folded axis")
            if code == CODE_MIRROR_PERIODIC and not (
                    0 <= self.reflect[axis] < int(shape[axis]) - 1):
                raise ValueError(
                    f"axis {axis} is folded PERIODIC with reflect row "
                    f"{self.reflect[axis]}, which is outside "
                    f"[0, {int(shape[axis]) - 1}); the far carry would image the "
                    f"plane it writes, or read outside the allocation")
            if code != CODE_MIRROR_PERIODIC and self.reflect[axis] != -1:
                raise ValueError(
                    f"axis {axis} is not folded PERIODIC but carries reflect row "
                    f"{self.reflect[axis]}; stepping._far_reflect_rows answers None "
                    f"there and the kernel would read a row nothing wrote")
        self.counts = tuple(int(value) for value in poles.counts)
        if len(self.counts) != len(E_TERMS):
            raise ValueError("a plan requires one pole count per E component")
        for (component, _displacement, _axis), count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{component} has {count} poles; MAX_POLES={MAX_POLES}")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        self._inverse_epsilon = tuple(CupyPointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
        self._poles = poles
        # The chunk record (launch.bind_chunk_bounds), required: ONE record serves
        # both halves, as in the folded pair, and is passed as the kernel's cb.
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedDispersiveFusedPairPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown,
        # it is a DISARMING — every mutation leg would then launch the shipped kernel
        # and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Resolve current P pointers and launch the fused pair, in place."""
        from .dispersive_update_e import MAX_POLES  # noqa: PLC0415
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415
        from .launch import CupyPointer  # noqa: PLC0415

        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"planned {self.counts[index]}")
            slots.extend(CupyPointer(array) for array in group)
            # The dead slots bind this component's D volume: correctly typed, and NP
            # compiles every load away. DispersiveConstitutivePlan's own convention.
            slots.extend([self._targets[index]] * (MAX_POLES - len(group)))

        kernel = (self._kernel if self._kernel is not None
                  else folded_dispersive_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients, *slots, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.reflect[0], self.reflect[1], self.reflect[2],
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phases[0], PHY=self.phases[1], PHZ=self.phases[2],
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedDispersiveFusedPairPlan(shape={self.shape}, bc={self.bc}, "
                f"phases={self.phases}, zero_metal={self.zero_metal}, "
                f"near={self.near}, far={self.far}, reflect={self.reflect}, "
                f"poles={self.counts}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_folded_dispersive_fused_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        chunk_layout: Any = None) -> Optional[FoldedDispersiveFusedPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``); its record is what both halves' forms key on.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not folded_dispersive_fused_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .dispersive_update_e import (  # noqa: PLC0415
        LivePoleBinding, poles_per_component)
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return FoldedDispersiveFusedPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        mirror_phases(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, poles_per_component(fields)),
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count and
        # `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_dispersive_fused_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], poles: Dict[str, Any], codes,
        zero_metal, phases, dtdx: float, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None), *,
        chunk_bounds: Any) -> FoldedDispersiveFusedPairPlan:
    """Build from bare device arrays — the gate's route.

    ``chunk_bounds`` is REQUIRED and has no default, as in
    ``launch.plan_from_arrays``.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``poles`` is
    keyed by E component and holds that component's ``P`` volumes in registration
    order, and ``kernel=`` carries the mutation override — dropping it silently
    disarms every mutation leg.
    """
    from .dispersive_update_e import E_TERMS, StaticPoleBinding  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedDispersiveFusedPairPlan(
        shape, dtdx, codes, zero_metal, phases,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        StaticPoleBinding([poles[component] for component, _d, _a in E_TERMS]),
        reflect=reflect,
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
