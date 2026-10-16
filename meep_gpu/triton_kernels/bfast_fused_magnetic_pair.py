"""The BFAST magnetic seam in one launch: BFAST ``step_B`` welded into ``update_H``.

The ``BFAST PML`` -> ``BFAST run`` cell's B->H product. Both halves are already
certified and neither is re-derived here:

* the curl half is :func:`.bfast_curl.bfast_pml_curl_step` — itself
  ``kernels.pml_curl_step``'s body plus ONE constexpr-gated pass;
* the constitutive half is ``kernels.constitutive_step``'s ``SCALE = 0`` arm,
  admitted on a BFAST run by the shipped
  :func:`.bfast_curl.bfast_run_constitutive_coverage` ("the constitutive sub-steps
  read nothing bfast-dependent — fields.py:227-230, the pass is on step_db only");
* and the WELD is the one substitution :func:`.kernels.fused_curl_constitutive_B`
  already makes between exactly those two bodies on a BFAST-inactive run.

SO THIS KERNEL IS ``fused_curl_constitutive_B``, STATEMENT FOR STATEMENT, PLUS THE
SAME ``if HAS_BFAST:`` BLOCK ``bfast_pml_curl_step`` ADDS TO ``pml_curl_step``. A
reader must be able to diff this body against those two and see nothing else moved;
the gate's transcription leg asserts exactly that, as two statement-list equalities
against both shipped sources.

===========================================================================
THE PASS, AND WHERE IT GOES
===========================================================================

``stepping._bfast_term`` is MEEP's BFAST pass, which runs per component right after
its curl (step_db.cpp:124-142), through the pass's own additive ladder
(``stepping._apply_folded_increment``). That is where it sits here: after the three
curl forms and before the stores. Transcribed from
:func:`.bfast_curl.bfast_pml_curl_step`, not re-derived:

* **the operands are the loads the curl already made.** BFAST SUMS the same
  shifted/center pairs the curl DIFFERENCES (the shared gather is the engine's
  deliberate advantage over MEEP's two loops);
* per target :func:`.forms.bfast_advance` on the record's presence and layer bits:
  with both partners the state takes MEEP's two roundings (``F = S - F_prev``
  stored, ``raw = F - F_prev``), with one the surviving term alone, and in MEEP's
  plain branch no ``- F_prev``;
* the increment ``raw * -1.0`` is masked to ``+0``, and the state kept, on the cells
  the curl's own ownership predicate drops; the state is stored in place, and
  :func:`.forms.fold_ladder` subtracts the increment — never a Triton unary minus;
* NO ``dtdx`` anywhere in the pass;
* ALL THREE targets run the pass whenever any k component is nonzero. With seeded
  state a zero-k target's ``F = -F_prev`` is byte-visible, so no arm skips it.

THE SIX ``k1``/``k2`` WORDS are host-computed and rounded ONCE by
:func:`.bfast_curl.bfast_curl_coefficients`, exactly as
:class:`.bfast_curl.BfastPmlCurlPlan` passes them.

THE THREE STATE POINTERS ARE THE ENGINE'S OWN ``f_bfast_B*`` ARRAYS, bound
pointer-identically. That is not an optimisation: the driver's flux backup/restore
mutates them in place (driver.py:4126-4135), so a copy would desynchronise the run.

``HAS_BFAST`` IS CARRIED AND MUST BE 1 on any configuration this product's
predicate admits — the BFAST family's clause 11 is INVERTED inside
``bfast_pml_curl_coverage`` (``grid.bfast_active`` required), so a BFAST-inactive
run is the ordinary :func:`.coverage.fused_pair_coverage`'s and is refused here by
name. It is carried rather than hard-coded for ``bfast_pml_curl_step``'s own
reason: it keeps the curl half a verbatim copy, and it gives the gate a
certified-kernel arm (``HAS_BFAST = 0`` must reproduce
``kernels.fused_curl_constitutive_B`` bit for bit).

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3282-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME** (driver.py:3283-3284). Ignorance is
  never an empty set: ``Fields`` does not hold the source list, so an undeclared
  ``sources`` is a REFUSAL and not an assumed ``()``.
* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` — PROVABLY NO-OPS
  HERE**. Both return at their first line unless ``grid.has_symmetry()``
  (stepping.py:1481-1482, :1565-1566), and ``_bfast_grid_reasons`` refuses every
  fold. THE PREDICATE RE-CHECKS IT ANYWAY rather than reading its own coverage off
  another module's guard.
* **``zero_metal_B`` — CARRIED INLINE**, through :func:`.coverage.zero_metal_axes`,
  which is IMPORTED rather than re-spelled so the predicate and the compile-time
  choice cannot disagree.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

ONE seam-instance, reachable, from ``results/fusion_matrix_triton_2026-08-20_closed/``
at the B->H cell (``BFAST PML``, ``BFAST run``):

    tests:TestReflectanceAngular.test_reflectance_angular_2_35_7   shape (1, 1, 1800)

It declares an ELECTRIC source only, so the magnetic seam is clear; it is
all-periodic with ``has_metallic == false``, so the inline ``zero_metal_B`` carry
compiles to three ``False`` flags ON THAT ROW — which is why the gate scores every
wall-clear mutation on a WALLED case instead. The D/E partner cell is worth ZERO on
this corpus (that row injects electrically inside that seam) and is not built.

THE OTHER BACKEND BUILT THIS FIRST: ``metal_kernels/bfast_fused_magnetic_pair.py``,
certified by ``parity/meep_gpu/gate_metal_below_the_cut_fused_pairs.py``, and worth
the same one row there.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites. Wired, it would
also contend for ``step_B`` with the ``BFAST PML`` arm, which IS wired and admits
the same configurations — and ``_select_slot`` (launch.py:1983-1996) leaves a slot
with two admitters UNSELECTED, taking the certified curl off the device with it.
Nothing in ``launch.py`` names this module; ``fastpath.plan_fast_path`` is
unchanged.

DEVICE STATUS: see ``parity/meep_gpu/probe_triton_bfast_fused_magnetic_pair.py``
and its artifact. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .bfast_curl import (
    BFAST_STATE_NAMES,
    bfast_curl_coefficients,
    bfast_pml_curl_coverage,
    bfast_run_constitutive_coverage,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
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

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3282-3289). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "REPLACES",
    "BfastFusedMagneticPairPlan",
    "bfast_fused_curl_constitutive_B",
    "bfast_fused_curl_constitutive_B_kernel",
    "bfast_fused_magnetic_pair_coverage",
    "plan_bfast_fused_magnetic_pair",
    "plan_bfast_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``kernels.METALLIC``, restated for the reason
    # every sibling restates it — importing a ``tl.constexpr`` wrapper and
    # re-wrapping it is not the same object, and the comparison in the kernel body
    # is against a literal code.
    METALLIC = tl.constexpr(1)

    # The forms, the BFAST pass and the chunk record's word offsets, under the names
    # kernels.py binds them: the lifted body reads them exactly as the root pair does.
    from .forms import (axis_present, bfast_advance, constitutive_form,  # noqa: E402
                        curl_form, curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    @triton.jit(do_not_specialize=["start_h"])
    def bfast_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx,By,Bz
        u0, u1, u2,                       # curl auxiliaries: fu_Bx,fu_By,fu_Bz
        g0, g1, g2,                       # curl sources: Ex,Ey,Ez
        s0, s1, s2,                       # f_bfast states, one per target
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx, start_h,  # start_h: MEEP's start of H
        k1_0, k2_0, k1_1, k2_1, k1_2, k2_2,   # f32 (k1,k2) per target, host-rounded
        BACKWARD: tl.constexpr,           # bound to 0 by every builder
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        HAS_BFAST: tl.constexpr,          # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """BFAST ``step_B`` + ``zero_metal_B`` + ``update_H``, one launch.

        ``kernels.fused_curl_constitutive_B``'s body, character for character, plus
        ``bfast_curl.bfast_pml_curl_step``'s BFAST pass after the three curl forms and
        before the wall clear -- the array path's order (each component's curl and its
        BFAST pass, then the seam, then update_H). The state words are stored on the
        cells MEEP's owned loop visits; the registers the pass leaves are the B words
        both the store and the H half read. ``start_h`` is the root pair's MEEP start
        of H. ``HAS_BFAST = 0`` builds ``kernels.fused_curl_constitutive_B`` exactly.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ====================================
        # Verbatim from pml_curl_step; the only edit below its stores is that nothing
        # reads f0/f1/f2 back.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
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
        # One call per target: MEEP's left-to-right association with both partners, the
        # one-partner form where the record says the other partner's derivative axis is
        # invariant, and an exact +0 with neither. The presence word is the record's
        # (stepping.curl_presence, MEEP's have_plus_deriv / have_minus_deriv).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
        curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
        curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
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
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # Which cells take the recurrence: the component's dsigu layer, on the component's
        # own sub-lattice. The dsigu axis is z for target 0, x for 1 and y for 2 (vec.hpp
        # cycle_direction); the sub-lattice is half-integer for every B target and integer
        # for every D one (the iyee table), so each target's record line is a compile-time
        # word index, chunk_map.record_index(half_integer, axis, 0).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)

        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)

        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the BFAST pass (stepping._bfast_term, then stepping._apply_folded_increment)
        # MEEP's STEP_BFAST on each component right after its STEP_CURL
        # (step_db.cpp:124-142), on the SAME gathered partners, applied to the registers
        # the forms returned (forms.bfast_advance: the state's two roundings, the
        # one-partner forms by the record's presence bits, MEEP's plain branch on cells
        # in neither the dsig nor the dsigu chunk). The increment is masked to +0, and
        # the state kept, on the cells the curl's own mask drops (MEEP writes F inside
        # its owned loop only); the BFAST ladder subtracts it (forms.fold_ladder). No
        # dtdx anywhere: MEEP passes it to step_bfast and never reads it.
        if HAS_BFAST:
            ds0 = record_in(cb, lattice + 4, j)
            ds1 = record_in(cb, lattice + 8, k)
            ds2 = record_in(cb, lattice + 0, i)
            st0 = tl.load(s0 + idx, mask=live, other=0.0)
            st1 = tl.load(s1 + idx, mask=live, other=0.0)
            st2 = tl.load(s2 + idx, mask=live, other=0.0)
            bf0, q0 = bfast_advance(st0, c_y, c, b_z, b, k1_0, k2_0, py, pz, ds0 | du0)
            bf1, q1 = bfast_advance(st1, a_z, a, c_x, c, k1_1, k2_1, pz, px, ds1 | du1)
            bf2, q2 = bfast_advance(st2, b_x, b, a_y, a, k1_2, k2_2, px, py, ds2 | du2)
            if BACKWARD:
                if BCY == METALLIC:
                    q0 = tl.where(at_y, 0.0, q0)
                    bf0 = tl.where(at_y, st0, bf0)
                if BCZ == METALLIC:
                    q0 = tl.where(at_z, 0.0, q0)
                    bf0 = tl.where(at_z, st0, bf0)
                if BCX == METALLIC:
                    q1 = tl.where(at_x, 0.0, q1)
                    bf1 = tl.where(at_x, st1, bf1)
                if BCZ == METALLIC:
                    q1 = tl.where(at_z, 0.0, q1)
                    bf1 = tl.where(at_z, st1, bf1)
                if BCX == METALLIC:
                    q2 = tl.where(at_x, 0.0, q2)
                    bf2 = tl.where(at_x, st2, bf2)
                if BCY == METALLIC:
                    q2 = tl.where(at_y, 0.0, q2)
                    bf2 = tl.where(at_y, st2, bf2)
            else:
                if BCX == METALLIC:
                    q0 = tl.where(at_x, 0.0, q0)
                    bf0 = tl.where(at_x, st0, bf0)
                if BCY == METALLIC:
                    q1 = tl.where(at_y, 0.0, q1)
                    bf1 = tl.where(at_y, st1, bf1)
                if BCZ == METALLIC:
                    q2 = tl.where(at_z, 0.0, q2)
                    bf2 = tl.where(at_z, st2, bf2)
            tl.store(s0 + idx, bf0, mask=live)
            tl.store(s1 + idx, bf1, mask=live)
            tl.store(s2 + idx, bf2, mask=live)
            v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
            v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)
            v2, n2 = fold_ladder(v2, n2, q2, si_x, si_y, du2)

        # --- the seam: stepping.zero_metal_B, between the two sub-steps ------------
        # Applied to the REGISTER, before both the store and the constitutive read, so
        # the two consumers see the one value the array path leaves in B.
        if ZM_X:
            v0 = tl.where(at_x, 0.0, v0)
        if ZM_Y:
            v1 = tl.where(at_y, 0.0, v1)
        if ZM_Z:
            v2 = tl.where(at_z, 0.0, v2)

        # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
        # reads and writes the field alone and fu keeps its word (the auxiliary contract).
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ================================
        # Verbatim from constitutive_step's SCALE=0 arm, with `src0 = tl.load(g0+idx)`
        # replaced by the register v0 (and v1/v2). The three source pointers and the
        # three inv_eps placeholders are gone with it.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        # Which cells take the accumulation: the component's own (dsigw) layer on the
        # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
        # (the constitutive term tables), so each target's record line is again a
        # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        dw0 = record_in(cb, 0, i)         # H: the integer sub-lattice
        dw1 = record_in(cb, 4, j)
        dw2 = record_in(cb, 8, k)
        # MEEP's start of H (see the docstring).
        start = start_h != 0
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        src0 = v0
        tl.store(w0 + idx, src0, mask=live & dw0)
        a0 = tl.load(h0 + idx, mask=live & dw0, other=0.0)
        prev0 = tl.where(start, src0, prev0)    # MEEP's start of H
        a0 = tl.where(start, src0, a0)
        a0 = constitutive_form(a0, prev0, src0, kp_0, km_0, dw0)
        tl.store(h0 + idx, a0, mask=live)

        # --- component 1 -----------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        src1 = v1
        tl.store(w1 + idx, src1, mask=live & dw1)
        a1 = tl.load(h1 + idx, mask=live & dw1, other=0.0)
        prev1 = tl.where(start, src1, prev1)    # MEEP's start of H
        a1 = tl.where(start, src1, a1)
        a1 = constitutive_form(a1, prev1, src1, kp_1, km_1, dw1)
        tl.store(h1 + idx, a1, mask=live)

        # --- component 2 -----------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        src2 = v2
        tl.store(w2 + idx, src2, mask=live & dw2)
        a2 = tl.load(h2 + idx, mask=live & dw2, other=0.0)
        prev2 = tl.where(start, src2, prev2)    # MEEP's start of H
        a2 = tl.where(start, src2, a2)
        a2 = constitutive_form(a2, prev2, src2, kp_2, km_2, dw2)
        tl.store(h2 + idx, a2, mask=live)

else:  # pragma: no cover - laptop path
    bfast_fused_curl_constitutive_B = None  # type: ignore[assignment]


def bfast_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if bfast_fused_curl_constitutive_B is None:
        raise ImportError(
            "the BFAST fused magnetic B/H kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return bfast_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def bfast_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                       sources: Any = None) -> Coverage:
    """May ONE launch span BFAST ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.

    The two halves are the arms ``launch.plan_step`` really selects on a BFAST row
    today (``BFAST PML`` on ``step_B``, ``BFAST run`` on ``update_H``), so this
    predicate is narrower than the pair the planner already puts on the device,
    never wider. The BFAST family's clause 11 is INVERTED inside
    ``bfast_pml_curl_coverage`` itself, so it needs no rung here: a BFAST-inactive
    row fails both halves together.
    """
    reasons: List[str] = []

    curl = bfast_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"BFAST curl half: {reason}" for reason in curl.reasons)
    constitutive = bfast_run_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"BFAST constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E seam and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET.
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

    # THE TWO SYMMETRY PASSES, re-checked rather than read off another module's
    # guard: if the BFAST tranche ever admits a fold this weld would silently
    # swallow two passes that had started doing work.
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
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    # THE INVARIANCE TABLE. The six k1/k2 words are gated on
    # ``grid.is_invariant`` (bfast_curl_coefficients, grid.py:1165-1183 — the
    # grid's OWN declared dimensionality, never a shape test), so a grid that
    # cannot answer it would silently be given the un-gated coefficients.
    if getattr(grid, "is_invariant", None) is None:
        reasons.append(
            "grid does not expose is_invariant; the BFAST k1/k2 gating "
            "(bfast_curl_coefficients, grid.py:1165-1183) cannot be resolved")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BfastFusedMagneticPairPlan:
    """One allocation-free launch for five of the driver's magnetic call sites.

    :class:`.launch.FusedPairPlan`'s ``pair='B'`` bindings plus the three state
    pointers, the six host-rounded scalars and the ``HAS_BFAST`` constexpr — which
    is exactly the delta :class:`.bfast_curl.BfastPmlCurlPlan` carries over
    :class:`.launch.PmlCurlPlan`. No inverse epsilon: that is the E side's.

    The state pointers wrap the engine's OWN ``f_bfast_B*`` arrays — the in-place,
    pointer-identical mutation the driver's flux backup/restore depends on
    (driver.py:4126-4135).
    """

    __slots__ = ("chunk_bounds", "_cb", "_h_state", "shape", "n_elem", "dtdx", "ks", "has_bfast", "backward", "bc",
                 "zero_metal", "block", "num_warps", "_targets", "_aux",
                 "_sources", "_states", "_curl_coefficients", "_h_targets",
                 "_h_aux", "_h_coefficients", "_grid", "_kernel")

    #: The five driver call sites one launch performs.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, ks, block: int,
                 targets, auxiliaries, sources, states, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_bfast: int = 1, *, chunk_bounds: Any,
                 h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "BfastFusedMagneticPairPlan")
        # MEEP's start of H (launch.FusedPairPlan's contract): the Fields the H
        # half starts from, or None for a state that has already started.
        self._h_state = h_state
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (bfast_curl.bfast_curl_coefficients);
        # float() keeps the bits, and Triton types a Python float argument as fp32.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.has_bfast = int(has_bfast)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.block = int(block)
        # ONE WARP, the measured default for a fused pair
        # (launch.FUSED_DEFAULT_NUM_WARPS). Explicit None remains "take Triton's
        # default".
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._states = tuple(CupyPointer(a) for a in states)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._h_targets = tuple(CupyPointer(a) for a in h_targets)
        self._h_aux = tuple(CupyPointer(a) for a in h_aux)
        self._h_coefficients = tuple(CupyPointer(_flat(a)) for a in h_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else bfast_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        start = self.start_h()
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._states,
            *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, start,
            *self.ks,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            HAS_BFAST=self.has_bfast,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def start_h(self) -> int:
        """1 on the one launch that starts H from B, 0 otherwise."""
        state = self._h_state
        return 0 if state is None or state.h_separated else 1

    def __repr__(self) -> str:
        return (f"BfastFusedMagneticPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, ks={self.ks!r}, "
                f"has_bfast={self.has_bfast}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_bfast_fused_magnetic_pair(fields: Any, pml: Any, sources: Any = None,
                                   block: Optional[int] = None,
                                   num_warps: Optional[int] = 1,
                                   kernel: Any = None,
                                   chunk_layout: Any = None,
                                   ) -> Optional[BfastFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not bfast_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE SAME FUNCTION THE CURL PLAN CALLS, with the grid's OWN declared
    # invariance flags (grid.py:1165-1183 — never a shape test) and the sub-step's
    # side. Identical arithmetic to the array path's per-call-site computation, and
    # not a second spelling of it.
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                 magnetic=(CURL_SUB_STEP == "step_B"))
    return BfastFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), ks,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        # The engine's OWN state arrays, pointer-identical.
        [getattr(fields, name) for name in BFAST_STATE_NAMES[CURL_SUB_STEP]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H. The kernel takes both and never asks which is
        # which, so a swap here is a silent half-cell error in the absorber
        # profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


def plan_bfast_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, ks,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_bfast: int = 1, *,
        chunk_bounds: Any,
        ) -> BfastFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``kernel=``
    carries the mutation override and ``has_bfast=0`` the identity leg's
    certified-kernel arm, which must reproduce
    ``kernels.fused_curl_constitutive_B`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return BfastFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, ks,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [arrays[name] for name in BFAST_STATE_NAMES[CURL_SUB_STEP]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, has_bfast=has_bfast,
        chunk_bounds=chunk_bounds,
    )
