"""The folded magnetic seam in one launch: ``step_B`` welded into ``update_H``.

WHY THE B/H HALF AND NOT THE D/E HALF — measured, from the 186-row census
``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired`` (Triton predicates,
lifted on NumPy, read on ``covered_modulo_backend`` because the lift is not CuPy).
The seam funnels, clause by clause:

    folded D->E   77 curl -> 53 constitutive -> 4 no-electric-source ->  2
    folded B->H   77 curl -> 77 constitutive -> 54 no-magnetic-source -> 45

The binding clause on the electric half is the SOURCE, not the fold and not the
fusion: 157 of the 186 rows declare an electric source and 58 declare a magnetic
one (``('D',) 128``, ``('B','D') 29``, ``('B',) 29``). Closing the same seam on the
magnetic half admits 45 rows where the electric one admits 2 — 22x — and the
counterfactual is the one the Metal folded D/E family recorded beside its own gate
(``parity/meep_gpu/results/metal_folded_fused_pair_2026-08-19/corpus_admission.json``,
``counterfactual_B_to_H.admitted = 45``), reproduced here against the Triton
predicates rather than the Metal ones.

THE SEAM IS NOT EMPTY, and that is worth saying plainly because the shorthand
"nothing injects a source between ``step_B`` and ``update_H``" is FALSE. The driver
runs FIVE passes there (driver.py:3281-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

Each is answered by name below. What is true is the weaker, measured statement:
the magnetic slot is EMPTY IN 54 of the 77 rows that clear both halves, where the
electric slot is empty in 4 of 53.

===========================================================================
THE FOUR PASSES IN THE SEAM
===========================================================================

* **the magnetic sources — REFUSED BY NAME.** ``driver.step`` injects them between
  the two halves (driver.py:3283-3284), so a fused pair would consume a
  pre-injection ``B``. Ignorance is never an empty set: ``Fields`` does not hold
  the source list, so an undeclared ``sources`` is a REFUSAL and not an assumed
  ``()``. This is the clause that caps the family, and it is 23 of the 77 rows.
* **``fill_symmetry_bc_B`` — CARRIED INLINE** (driver.py:3285;
  ``stepping.fill_symmetry_bc_B`` :1398, ``_fill_symmetry_ghost_cells`` :1409,
  ``_write_mirror_ghost`` :1451). It writes ``cell 0 = parity * cell 2`` on every
  folded axis for every component whose Yee shift there is 0, ``update_H`` then
  reads exactly those cells, and it is NOT dead: a pair that skipped it would feed
  the constitutive half a pre-fill ``B`` on one component of every folded axis.
* **``zero_metal_B`` — CARRIED INLINE** (driver.py:3286;
  ``stepping._zero_metal`` :2208-2247), through
  :func:`.coverage.zero_metal_axes`, which is IMPORTED rather than re-spelled.
* **``fill_folded_far_ghosts_B`` — CARRIED INLINE as of 2026-08-20**
  (driver.py:3287; ``stepping._stored_past_owned`` :1454-1469, consumed at
  ``_fill_folded_far_ghosts`` :1516). It was REFUSED BY NAME until then, and that
  refusal was this family's cap on a folded PERIODIC axis: the pass images the
  LAST stored slot from a runtime reflect row on an axis that is not the
  component's own. Carrying it brings the folded curl's top-plane mask with it
  (``symmetry.pml_curl_step_folded``'s second added block), which is now
  transcribed here, and turns a folded PERIODIC axis from a refusal into an
  admission — 9 more corpus rows at this cell, priced in
  ``results/fusion_matrix_triton_2026-08-20_farcarry/``.

===========================================================================
THE B GEOMETRY IS NOT THE D GEOMETRY — the whole of what is new here
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219): ``Bx (0,1,1)``, ``By (1,0,1)``,
``Bz (1,1,0)`` against ``Dx (1,0,0)``, ``Dy (0,1,0)``, ``Dz (0,0,1)``. The near
fill touches component ``m`` on axis ``a`` exactly when ``iyee[m][a] == 0``, so

    D family:  a != m   (two axes per component)
    B family:  a == m   (the component's OWN axis, and only it)

and the FAR fill is the exact complement — ``iyee[m][a] == 1`` — so on the B half
it touches the two axes that are NOT the component's own, up to two destination
planes per component with a corner where both fire.

THREE CONSEQUENCES, and each one moves in a different direction:

1. **THE COEFFICIENT INDEX MOVES.** ``update_H`` indexes ``kps``/``kms`` on the
   component's own axis (``stepping.H_CONSTITUTIVE_TERMS`` :226,
   ``update_H`` :919-923), and on the B half that is exactly the axis the fill
   images along — so the fill's SOURCE (stored 2) and DESTINATION (stored 0) take
   DIFFERENT coefficient entries. This is the property the Metal folded D/E pair
   relies on (its "the coefficient index does not move") and the reason that
   family refuses the far fill; it is FALSE here. It is not fatal — the source
   thread loads the destination's pair explicitly, two extra scalar loads — but it
   is transcribed rather than inherited, and the gate carries a mutation that
   reuses the source's pair instead.
2. **NO PARITY PRODUCTS FROM THE NEAR FILL — BUT THE FAR ONE MAKES THEM.** A B
   component is a NEAR destination on at most ONE axis, so
   ``_fill_symmetry_ghost_cells``' X/Y/Z application order (:1441-1447) is
   unobservable on that half. The FAR fill is the complement and reaches two axes
   per component, so a cell at the top of both carries the PRODUCT of two
   parities, and a cell that is a near destination on one axis and a far one on
   another carries the product of theirs. ONE LANE OWNS ALL OF THEM, and the rule
   is measured rather than argued: the seam leaves, at the cell sitting at ``last``
   on far-axis set ``T`` and at stored 0 on the near axis when ``z``,

       (prod over T of mirror_parity) * (z ? mirror_parity : 1) * v_m(that cell
       with every a in T moved to reflect_a and near moved to 2)

   independent of the order the array path applies its axes in — zero differing
   words over nine configurations against the array path's own three passes
   (``results/metal_folded_far_carry_2026-08-20/ownership.json``).
3. **THE WALL CLEAR AND THE NEAR FILL CANNOT MEET; THE FAR ONE INHERITS IT.**
   ``_zero_metal`` clears component ``m`` at stored cell 0 of axis ``a`` when
   ``iyee[m][a] == 0``, i.e. ``a == m`` again — the same cell set the NEAR fill
   writes — and it SKIPS a folded axis (:2237-2239: ``is_metallic(axis) and not
   is_mirrored(axis)``). So on that half the two passes are mutually exclusive per
   axis and the D side's "parity-then-clear" ordering question is VACUOUS. A FAR
   ghost moves along an axis that is NOT the component's own, so it sits at the
   same coordinate on every axis that could clear it as the lane that owns it: its
   clear status is that lane's, INHERITED by reading ``v`` after the clear lines,
   which is the driver's own order (:3286 before :3287). The predicate checks the
   disjointness rather than inferring it from ``zero_metal_axes``' construction.

===========================================================================
THE OWNERSHIP RESTRUCTURE — one launch, no barrier, no race
===========================================================================

The fill reads a cell THIS LAUNCH WRITES. Triton has no device-wide barrier, so
the naive carry — the program holding stored cell 0 reading stored cell 2 — races
on ``B``. Ownership moves instead. For component ``m`` on a folded axis ``m``:

* a lane on ANY destination plane — stored 0 of the near axis, the top plane of
  each far one — is a DESTINATION. It computes the curl and the
  split-field ``n`` and stores ``fu_B`` (the array path's ``step_B`` writes ``fu``
  at every cell and the fill does not touch it), and then STOPS: it never loads
  ``B``, never stores ``B``, and never touches ``H`` or ``f_w_H``. Its stepped
  displacement is discarded by the array path too — the fill overwrites it;
* the lane at stored 2 on the near axis and at the reflect row on each far one is
  the SOURCE. It does its own cell in full and then writes every destination that
  back-substitutes to it — up to seven — each with its composed parity, the
  displacement stored, and the constitutive update performed there. The NEAR half
  of a destination moves the indexed axis, so those take the destination's own
  coefficient pair at stored index 0; a FAR-only destination does not move it and
  takes the source lane's pair unchanged.

  EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWNERSHIP MASK, and that is a
  defect the device gate found rather than a precaution: with two fills a lane can
  be the SOURCE of one and the DESTINATION of the other, and it would then write a
  ghost from a ``v`` built on an ``f`` load its own mask zeroed.

Every destination word is written by exactly one lane, no lane reads a word
another lane writes — the destination's ``B``/``H``/``f_w_H`` loads are masked OFF
rather than merely unused, which is what makes that a statement about memory
traffic and not about dead registers — and no barrier is needed.

===========================================================================
WHAT IS DELIBERATELY *NOT* NEW
===========================================================================

* the folded curl half is :func:`.symmetry.pml_curl_step_folded` with
  ``BACKWARD = 0``, byte-copied — INCLUDING the ``MIRROR_PERIODIC`` top-plane
  block, which arrived with the far carry and which the gate's transcription leg
  checks line by line against that emitter's own body;
* the constitutive half is :func:`.kernels.constitutive_step`'s ``SCALE = 0`` arm
  on ``(Hx, Hy, Hz) <- (Bx, By, Bz)``, byte-copied, with ``src = tl.load(g + idx)``
  replaced by the register the curl half just produced — exactly the substitution
  :func:`.kernels.fused_curl_constitutive_B` makes;
* the parity is ``fields.mirror_parity(c, axis, phase) ==
  phase * (1 - 2 * iyee[c][axis])`` (fields.py:117), measured here again over all
  12 components x 3 axes x 2 phases: ``+phase`` on the NEAR fill, which only ever
  touches shift-0 components, and ``-phase`` on the FAR one, which only ever
  touches shift-1 ones. The multiply is by exactly +-1.0, exact in float32 for
  every input including subnormals and signed zeros.

ROUTED 2026-08-27, AND STILL NOT AN ARM — the two are different questions and this
paragraph used to answer only the first. ``launch.plan_step`` assigns at most one
ARM per ``STEP_ORDER`` slot and this product spans FOUR driver passes, so it can
still claim no slot through the arm table and registers none. What it now has is
the OTHER route: ``launch._install_folded_fused_pairs`` (launch.py:2189-2258) calls
this module's own ``folded_fused_magnetic_pair_coverage`` and
``plan_folded_fused_magnetic_pair`` under ``fuse=True`` on a folded grid, and on
admission ``_install_fused_pair`` puts the plan in ``step_B`` with a ``NoopPlan``
naming it in ``update_H``. The absorb clause in front of it
(``launch._pair_may_absorb``, table ``FOLDED_FUSED_PAIR_ARMS``) requires the
``folded PML`` curl arm and the ``folded`` constitutive arm to have already won
those two slots, which are this predicate's own two conjuncts.

DISPATCH IS STILL NOT REACHED, and that is a separate mechanism rather than a
restatement. ``fastpath.plan_fast_path`` asks for ``fuse=False`` (fastpath.py:2076)
and additionally refuses the WHOLE plan if any slot comes back carrying a label
``fastpath.arm_is_fused`` recognises — which is why the label this module's plan is
installed under leads with ``fused pair`` (launch.py:2145-2160). A default run
therefore still never reaches this plan.

DEVICE STATUS: **RELEASED 2026-08-20**, the GPU host RTX A6000 (physical GPU 6, pinned
    by UUID and verified empty), Triton 3.1.0, CuPy 13.5.1, under the 'keep'
    float32 subnormal policy (13128 PTX instructions audited, 0 carrying .ftz).
    RE-CUT for the ``fill_folded_far_ghosts_B`` carry: 10/10 device legs
    bit-identical to BOTH oracles — the CuPy array path and the separately
    certified Triton products — over TEN complete driver steps, 16/16 mutations
    matching their declared expectation (13 caught, 3 predicted nulls), 4/4
    refusal/carry rows
    (``parity/meep_gpu/results/triton_folded_far_carry_2026-08-20/run_farcarry6/``).
    The weld is ``triton_folded_fused_magnetic_pair_device_gate`` in
    ``fingerprints.json``.

    THE RUN FOUND A DEFECT, and it was in the KERNEL rather than the harness for
    the first time in this family's history: the carry masks were not ANDed with
    the component's ownership mask, so on a grid folding TWO axes a lane that is
    the source of one fill and the destination of the other raced the lane that
    legitimately owns the composite cell. One word of ``Hx`` and one of ``Hy``,
    1-2 ULP, on the two-folded-axis rows alone and invisible on every single-fold
    one (``run2.log`` beside the artifact). A gate whose case table stopped at one
    folded axis would have released it.

    RE-CUT 2026-08-21, AND THAT ARGUMENT CAME BACK ONE RUNG HIGHER. The paragraph
    above is correct and was also the shape of this family's own next gap: the
    2026-08-20 case table stopped at TWO folded axes, so the three-axis blocks
    below — ``if NEAR_a and (FAR_b and FAR_c):``, the triple composites — were
    SHIPPED CODE NO DEVICE LEG EVER EXECUTED. An adversarial verifier found it; the
    gate now counts it rather than arguing it. A new ``branch_reachability`` leg
    executes every live constexpr guard in this kernel against the constexprs each
    case compiles and reports **20 of 45 unreachable on the old table** — the whole
    z half of the carry, the z wall clear and the unfolded-periodic-Y ghost rule,
    not only the triples. Three cases close all twenty: three folded PERIODIC axes
    at mixed phases and an odd full count (SEVEN ghost cells owned by one source
    lane, the deepest composition this family can emit), the grid shape
    [16, 12, 213] of a corpus row this family's census claims, and a 3-D fold with
    a live z wall. Four mutations arrive with them, including the first that
    re-plants the ownership defect the 2026-08-20 run found.

    THE KERNEL IS UNCHANGED BY THAT ROUND. Nothing below moved; what moved is what
    is known to execute it.

    ROUTED BUT STILL NOT DISPATCHED. A weld licenses the claim, not a dispatch.
    Since 2026-08-27 ``launch.plan_step(..., fuse=True)`` on a folded grid does
    build this plan (see the routing paragraph above); ``fastpath`` asks for
    ``fuse=False`` and refuses any plan carrying a ``fused pair`` label, so nothing
    in a default run reaches it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
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
    folded_constitutive_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? Flipping it is a
#: claim about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring. See
#: ``deposit_repair`` for why.
#:
#: BECAME ``True`` ON 2026-08-30, and the thing that changed is not this module. While it
#: was False the source-presence clause below refused every in-seam deposit, because
#: ``deposit_repair.apply`` wrote only the deposit index itself and the driver's
#: post-injection ``fill_symmetry_bc_*`` / ``fill_folded_far_ghosts_*`` write that cell's
#: MIRROR IMAGES somewhere else -- so a repaired seam would have left the images holding a
#: pre-injection field. That premise is now false: ``deposit_repair._fill_image_rules``
#: (deposit_repair.py:217) and ``repair_cells`` (:256) extend the saved and restored set to
#: exactly the cells those two fills image a deposit into, which is the INVERSE of the
#: forward carry this module's kernel already performs. Metal made the identical flip on
#: 2026-08-28 for the identical reason.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``, which
#: goes on refusing BY NAME every seam the repair cannot invert -- an off-diagonal
#: constitutive (a stencil over the curl's own in-place output), a nonlinear one, a
#: cylindrical r = 0 axis, a fold too short to hold the near fill's source row, and an
#: absorber whose split-field recurrence never ran. Flipping this widens the predicate to
#: what the repair can carry and to nothing else.
CARRIES_DEPOSIT_REPAIR = True

#: The two codes a folded axis can take. The NEAR fill runs on either; the FAR fill
#: only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``:1454-1469), which is
#: the whole of what separates the two carries.
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

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

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289). Declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without importing :mod:`launch`; ``test_triton_folded_fused_magnetic_pair``
#: pins the two equal.
BACKWARD = 0

#: Which component the near fill images on which axis, for THIS family. Derived
#: from ``TARGET_IYEE`` at import rather than written out, and pinned by test
#: against ``fields.IYEE_SHIFTS``: axis ``a`` fills exactly the components whose
#: Yee shift there is 0, which for B is the single component ``a``.
NEAR_FILL_COMPONENTS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
          if TARGET_IYEE[name][axis] == 0)
    for axis in range(3)
)

#: Which component the FAR fill images on which axis. The exact complement of
#: :data:`NEAR_FILL_COMPONENTS` — ``stepping._fill_folded_far_ghosts`` (:1516-1534)
#: writes axis ``a`` for every component whose Yee shift there is ONE — and derived
#: from the same table for the same reason. For B that is the two components that
#: are NOT the axis, so a folded PERIODIC axis gives two components a far ghost and
#: one a near ghost, and the sets never overlap.
FAR_FILL_COMPONENTS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
          if TARGET_IYEE[name][axis] == 1)
    for axis in range(3)
)

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAR_FILL_COMPONENTS",
    "NEAR_FILL_COMPONENTS",
    "REPLACES", "FoldedFusedMagneticPairPlan",
    "folded_fused_curl_constitutive_B", "folded_fused_curl_constitutive_B_kernel",
    "folded_fused_magnetic_pair_coverage", "mirror_phases",
    "plan_folded_fused_magnetic_pair", "plan_folded_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The forms of the root kernels (forms.py) and the chunk record's two word offsets,
    # bound under kernels.py's names: both halves read the record and call the forms
    # exactly as symmetry.pml_curl_step_folded and kernels.constitutive_step do.
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def _carry_ghost(f, w, h, dst, ghost, kp_d, km_d, dw_d, start, mask):
        """Write ONE imaged ghost cell and run ``update_H`` there.

        THE EIGHT STATEMENTS ARE THE CERTIFIED ONES, MOVED AND NOT REWRITTEN: they
        are ``kernels.constitutive_step``'s ``SCALE=0`` arm with ``src`` bound to the
        carried value instead of a load, which is exactly what the owned cell above
        does. They became a device function when the far carry landed, because a B
        component can now be the source of up to SEVEN ghost cells (a near plane,
        two far planes, and every composition of them) and seven inlined copies of
        the same eight statements is seven places for one of them to drift.

        The ORDER is the array path's: workspace read, workspace write, H
        accumulate, then the flux store. ``prev`` is read BEFORE the write, which is
        the one ordering the constitutive cannot survive being wrong about.

        ``dw_d`` IS THE DESTINATION'S MEMBERSHIP, evaluated where the ghost cell is:
        the accumulation and its ``f_w`` traffic on the cells whose chunk carries the
        component's own (dsigw) layer, ``H = B`` elsewhere
        (:func:`.forms.constitutive_form`). The caller passes the row-0 flag for
        anything with the NEAR half in it, which moves along exactly the component's
        own axis, and the source lane's flag for a FAR-only destination, which keeps
        the source's coordinate on that axis. ``start`` is MEEP's start of H, as in
        ``kernels.constitutive_step``: on the run's first update H and ``f_w`` start
        as the cell's B, a ghost cell included.
        """
        prev = tl.load(w + dst, mask=mask & dw_d, other=0.0)
        tl.store(w + dst, ghost, mask=mask & dw_d)
        acc = tl.load(h + dst, mask=mask & dw_d, other=0.0)
        prev = tl.where(start, ghost, prev)
        acc = tl.where(start, ghost, acc)
        acc = constitutive_form(acc, prev, ghost, kp_d, km_d, dw_d)
        tl.store(h + dst, acc, mask=mask)
        tl.store(f + dst, ghost, mask=mask)

    @triton.jit(do_not_specialize=["start_h"])
    def folded_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx, By, Bz
        u0, u1, u2,                       # curl auxiliaries: fu_Bx, fu_By, fu_Bz
        g0, g1, g2,                       # curl sources: Ex, Ey, Ez
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx, Hy, Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx, f_w_Hy, f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        start_h,                          # RUNTIME: MEEP's start of H (see below)
        BACKWARD: tl.constexpr,           # bound to 0 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_B`` + ``fill_symmetry_bc_B`` + ``zero_metal_B`` + ``update_H``.

        ``BACKWARD`` IS CARRIED AND MUST BE 0. It keeps the curl half a verbatim
        copy of :func:`.symmetry.pml_curl_step_folded` rather than a hand-
        specialised one, which is the whole reason the transcription risk here is
        low. It cannot be 1: both carries below are transcribed for the B family's
        Yee shifts, where the NEAR fill touches component ``m`` on axis ``m`` alone
        and the FAR fill on the two that are not; the D family's shifts swap those
        two roles exactly, which is a different carry with a different coefficient
        question on each half.

        ``BCX``/``BCY``/``BCZ`` take ALL FOUR of :mod:`.symmetry`'s codes as of
        2026-08-20, ``MIRROR_PERIODIC`` included: this kernel now carries
        ``fill_folded_far_ghosts_B`` (driver.py:3287), which is the pass that used
        to make that code a refusal. The folded curl's top-plane mask block comes
        with it and is transcribed below.

        ``NEAR_a`` / ``FAR_a`` say which fill runs on axis ``a``: NEAR on either
        mirror code (``stepping._fill_symmetry_ghost_cells`` gates on the mirror
        phases alone) and FAR only on ``MIRROR_PERIODIC``
        (``stepping._stored_past_owned``:1454-1469). They are DERIVED FROM ``bc`` by
        the plan and are not independent inputs, so they cannot disagree with the
        codes the curl half branches on.

        ``rx``/``ry``/``rz`` are ``stepping._far_reflect_rows``' answer per axis,
        RUNTIME as they are in :func:`.symmetry.mirror_ghost_fill` and for its
        reason: the row is ``n_full - stored + 2``, which is ``stored - 2`` at an
        even full count and ``stored - 3`` at an odd one, so baking ``n - 2``
        reflects about the window top instead of about the second mirror and is a
        whole cell wrong on every odd-count run. ``-1`` on an axis with no far
        ghost, where no lane reads it.

        ``PHX``/``PHY``/``PHZ`` are the folded axes' declared mirror phases,
        ``grid.mirror_phase(axis)``, and 0 on an unfolded axis. Zero is chosen
        deliberately: if a carry block ever fired on an axis the host did not
        classify as folded, it would write an exact zero plane — loud — rather
        than a plausible field.

        ``cb`` IS THE RUN'S CHUNK RECORD, and ONE record serves both halves, as in
        ``kernels.fused_curl_constitutive_B``: the curl half reads its dsigu lines on
        the half-integer lattice and the presence word, the constitutive half its
        dsigw lines on the integer one. Every select is evaluated where its cell is,
        a carried ghost included.

        ``start_h`` IS MEEP'S START OF H (``stepping.separate_h_from_b``), as in
        ``kernels.fused_curl_constitutive_B``: nonzero on the run's first update, when
        H and ``f_w`` start as B on the accumulation's cells. The plan passes it from
        ``Fields.h_separated``; it is a runtime value Triton does not specialize on.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny
        # Index 0 on any axis, as a per-lane tensor: the near fill's destination
        # coefficient index. A scalar `+ 0` would not broadcast against a masked
        # vector load, and a literal is not an index.
        origin = idx * 0

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

        # --- ownership mask, the TOP plane of a folded PERIODIC axis ------------
        # symmetry.pml_curl_step_folded's second added block, BYTE-COPIED from its
        # BACKWARD == 0 arm. It was absent while this family refused
        # MIRROR_PERIODIC; carrying fill_folded_far_ghosts_B brings the axis in, so
        # the block comes with it. Bx:(0,1,1) By:(1,0,1) Bz:(1,1,0) — shift 1 on the
        # two OTHER axes, which is exactly the set the far carry writes.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
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

        # --- ownership of the fill destinations --------------------------------
        # Component m is a NEAR destination at stored cell 0 of axis m and a FAR
        # destination at the top plane of each OTHER folded periodic axis
        # (IYEE_SHIFTS: Bx (0,1,1), By (1,0,1), Bz (1,1,0) — shift 0 on its own
        # axis, 1 on the others). Those lanes do NOT load or store B, H or f_w_H:
        # the source lane writes all three for them. The masks are what make the
        # carry race-free — a lane that merely discarded the value would still have
        # READ a word another lane writes.
        own0, own1, own2 = live, live, live
        if NEAR_X:
            own0 = own0 & (i != 0)
        if FAR_Y:
            own0 = own0 & (j != ny - 1)
        if FAR_Z:
            own0 = own0 & (k != nz - 1)
        if NEAR_Y:
            own1 = own1 & (j != 0)
        if FAR_X:
            own1 = own1 & (i != nx - 1)
        if FAR_Z:
            own1 = own1 & (k != nz - 1)
        if NEAR_Z:
            own2 = own2 & (k != 0)
        if FAR_X:
            own2 = own2 & (i != nx - 1)
        if FAR_Y:
            own2 = own2 & (j != ny - 1)

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
        # sub-lattice, verbatim from the folded curl (half-integer here, BACKWARD = 0).
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

        # --- the seam: stepping.zero_metal_B (driver.py:3286) ------------------
        # Applied to the REGISTER, before both the store and the constitutive
        # read, so the two consumers see the one value the array path leaves in B.
        # A walled axis is never a folded axis (_zero_metal :2237-2239), so these
        # three lines cannot touch a fill destination; the predicate checks that
        # rather than leaning on it.
        if ZM_X:
            v0 = tl.where(at_x, 0.0, v0)
        if ZM_Y:
            v1 = tl.where(at_y, 0.0, v1)
        if ZM_Z:
            v2 = tl.where(at_z, 0.0, v2)

        # fu is written on the recurrence's cells ONLY (the auxiliary contract),
        # destinations included: the array path's step_B writes it on every such
        # cell, a ghost row reading its image row's membership, and neither fill
        # touches it.
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=own0)
        tl.store(f1 + idx, v1, mask=own1)
        tl.store(f2 + idx, v2, mask=own2)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.constitutive_step's SCALE=0 arm, with
        # `src = tl.load(g + idx)` replaced by the register the curl half just
        # produced and `live` by the component's ownership mask. Component 0 takes
        # its coefficient from axis x, 1 from y, 2 from z
        # (stepping.H_CONSTITUTIVE_TERMS :226) — the component's OWN axis, whose
        # dsigw line on the integer lattice keys the form.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, 0, i)
        dw1 = record_in(cb, 4, j)
        dw2 = record_in(cb, 8, k)
        # MEEP's start of H (see the docstring).
        start = start_h != 0

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=own0 & dw0, other=0.0)   # BEFORE the store.
        src0 = v0
        tl.store(w0 + idx, src0, mask=own0 & dw0)
        a0v = tl.load(h0 + idx, mask=own0 & dw0, other=0.0)
        prev0 = tl.where(start, src0, prev0)  # H starts as B (start_h)
        a0v = tl.where(start, src0, a0v)
        a0v = constitutive_form(a0v, prev0, src0, kp_0, km_0, dw0)
        tl.store(h0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1 & dw1, other=0.0)
        src1 = v1
        tl.store(w1 + idx, src1, mask=own1 & dw1)
        a1v = tl.load(h1 + idx, mask=own1 & dw1, other=0.0)
        prev1 = tl.where(start, src1, prev1)  # H starts as B (start_h)
        a1v = tl.where(start, src1, a1v)
        a1v = constitutive_form(a1v, prev1, src1, kp_1, km_1, dw1)
        tl.store(h1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2 & dw2, other=0.0)
        src2 = v2
        tl.store(w2 + idx, src2, mask=own2 & dw2)
        a2v = tl.load(h2 + idx, mask=own2 & dw2, other=0.0)
        prev2 = tl.where(start, src2, prev2)  # H starts as B (start_h)
        a2v = tl.where(start, src2, a2v)
        a2v = constitutive_form(a2v, prev2, src2, kp_2, km_2, dw2)
        tl.store(h2 + idx, a2v, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # stepping._write_mirror_ghost (:1451):  field[0]  = +phase * field[2]
        # stepping._fill_folded_far_ghosts (:1516): field[-1] = -phase * field[row]
        # and every COMPOSITION of the two, which is what a cell at the top of more
        # than one folded plane carries. The composed value is
        #
        #     (prod over the axes at `last` of mirror_parity)
        #         * (near ? mirror_parity : 1) * v_m(the fully back-substituted cell)
        #
        # independent of the order the array path applies its axes in — MEASURED
        # against the array path's own three passes over nine configurations,
        # results/metal_folded_far_carry_2026-08-20/ownership.json, zero differing
        # words. So ONE lane, the one at that back-substituted cell, owns every
        # ghost the fills leave, and no lane reads a word another lane writes.
        #
        # THE PARITY IS +PHASE ON A SHIFT-0 COMPONENT AND -PHASE ON A SHIFT-1 ONE
        # (fields.mirror_parity :117 is `phase * (1 - 2 * iyee)`), so the near carry
        # takes `PH?` and the far carry `-PH?`, and +-1.0 is exact in float32.
        #
        # THE COEFFICIENT INDEX MOVES ONLY FOR THE NEAR HALF. update_H indexes
        # component m on axis m (stepping.H_CONSTITUTIVE_TERMS :226); the near fill
        # images along that same axis, so its destination reads kps/kms at index 0
        # rather than reusing the source lane's at index 2. The far fill images
        # along an axis that is NOT m, so its destination sits at the SAME
        # coefficient index as the lane that owns it and reuses that lane's pair —
        # a reload there would reload the identical word.
        #
        # THE NEAR SOURCE INDEX IS THE LITERAL 2, as it is in
        # symmetry.mirror_ghost_fill (`base + 2 * stride`), rather than the named
        # MIRROR_SOURCE_INDEX: a jit body that closes over a module-level Python int
        # is a Triton-version question this file has no way to measure without a
        # device. The literal is pinned to the name by test.
        kp_d0 = tl.load(kp0 + origin, mask=live, other=0.0)
        km_d0 = tl.load(km0 + origin, mask=live, other=0.0)
        kp_d1 = tl.load(kp1 + origin, mask=live, other=0.0)
        km_d1 = tl.load(km1 + origin, mask=live, other=0.0)
        kp_d2 = tl.load(kp2 + origin, mask=live, other=0.0)
        km_d2 = tl.load(km2 + origin, mask=live, other=0.0)
        # ... and its membership there, for the same reason: a NEAR destination's
        # own-axis row is 0, so its select is keyed on row 0's word (a ghost row
        # reads its image row's line). A FAR-only destination keeps the source
        # lane's coordinate on that axis, and so its flag (dw0, dw1, dw2 above).
        dw_d0 = record_in(cb, 0, origin)
        dw_d1 = record_in(cb, 4, origin)
        dw_d2 = record_in(cb, 8, origin)

        # The per-axis source lane, destination offset and parity, once. Each is
        # read only under a mask its own axis's constexpr gates, so an axis that
        # carries no fill contributes no lane and no load.
        # EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWN OWNERSHIP MASK, and
        # that is a DEFECT THE DEVICE GATE FOUND rather than a precaution
        # (results/run_farcarry2/, 2026-08-20: one word of Hx at (i=0, j=ny-1) and
        # one of Hy at (i=nx-1, j=0), 1-2 ULP, on the two-folded-axis rows alone).
        #
        # The lane masks below say "this lane sits ON the source plane". Until the
        # far carry there was only one fill, and a lane on its source plane could
        # never be a destination — the near fill reads stored 2 and writes stored 0
        # of the SAME axis. With two fills a lane can be the source of one and the
        # DESTINATION of the other: on a grid folding x and y, the lane at
        # (i = rx, j = 0) is the far source for By on x and is itself the near
        # fill's destination on y. It would then write the composite cell with a `v`
        # built from an `f1` load its own ownership mask zeroed, racing the lane
        # that legitimately owns it. `own?` is exactly the Metal twin's outer
        # `if (!(j == 0 || last_x))`, which is why that board never saw this.
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
        # time on the device, so `if NEAR_X and FAR_Y and FAR_Z:` is a kernel that
        # does not build rather than one that builds wrong.
        #
        # --- component 0 (Bx): near on x, far on y and z -----------------------
        if FAR_Y:
            _carry_ghost(f0, w0, h0, idx + df_y, -PHY * v0,
                         kp_0, km_0, dw0, start, own0 & far_j)
        if FAR_Z:
            _carry_ghost(f0, w0, h0, idx + df_z, -PHZ * v0,
                         kp_0, km_0, dw0, start, own0 & far_k)
        if FAR_Y and FAR_Z:
            _carry_ghost(f0, w0, h0, idx + df_y + df_z, PHY * PHZ * v0,
                         kp_0, km_0, dw0, start, own0 & far_j & far_k)
        if NEAR_X:
            _carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0,
                         kp_d0, km_d0, dw_d0, start, own0 & near_i)
        if NEAR_X and FAR_Y:
            _carry_ghost(f0, w0, h0, idx + dn_x + df_y, -PHX * PHY * v0,
                         kp_d0, km_d0, dw_d0, start, own0 & near_i & far_j)
        if NEAR_X and FAR_Z:
            _carry_ghost(f0, w0, h0, idx + dn_x + df_z, -PHX * PHZ * v0,
                         kp_d0, km_d0, dw_d0, start, own0 & near_i & far_k)
        if NEAR_X and (FAR_Y and FAR_Z):
            _carry_ghost(f0, w0, h0, idx + dn_x + df_y + df_z,
                         PHX * PHY * PHZ * v0, kp_d0, km_d0, dw_d0, start,
                         own0 & near_i & far_j & far_k)

        # --- component 1 (By): near on y, far on x and z -----------------------
        if FAR_X:
            _carry_ghost(f1, w1, h1, idx + df_x, -PHX * v1,
                         kp_1, km_1, dw1, start, own1 & far_i)
        if FAR_Z:
            _carry_ghost(f1, w1, h1, idx + df_z, -PHZ * v1,
                         kp_1, km_1, dw1, start, own1 & far_k)
        if FAR_X and FAR_Z:
            _carry_ghost(f1, w1, h1, idx + df_x + df_z, PHX * PHZ * v1,
                         kp_1, km_1, dw1, start, own1 & far_i & far_k)
        if NEAR_Y:
            _carry_ghost(f1, w1, h1, idx + dn_y, PHY * v1,
                         kp_d1, km_d1, dw_d1, start, own1 & near_j)
        if NEAR_Y and FAR_X:
            _carry_ghost(f1, w1, h1, idx + dn_y + df_x, -PHY * PHX * v1,
                         kp_d1, km_d1, dw_d1, start, own1 & near_j & far_i)
        if NEAR_Y and FAR_Z:
            _carry_ghost(f1, w1, h1, idx + dn_y + df_z, -PHY * PHZ * v1,
                         kp_d1, km_d1, dw_d1, start, own1 & near_j & far_k)
        if NEAR_Y and (FAR_X and FAR_Z):
            _carry_ghost(f1, w1, h1, idx + dn_y + df_x + df_z,
                         PHY * PHX * PHZ * v1, kp_d1, km_d1, dw_d1, start,
                         own1 & near_j & far_i & far_k)

        # --- component 2 (Bz): near on z, far on x and y -----------------------
        if FAR_X:
            _carry_ghost(f2, w2, h2, idx + df_x, -PHX * v2,
                         kp_2, km_2, dw2, start, own2 & far_i)
        if FAR_Y:
            _carry_ghost(f2, w2, h2, idx + df_y, -PHY * v2,
                         kp_2, km_2, dw2, start, own2 & far_j)
        if FAR_X and FAR_Y:
            _carry_ghost(f2, w2, h2, idx + df_x + df_y, PHX * PHY * v2,
                         kp_2, km_2, dw2, start, own2 & far_i & far_j)
        if NEAR_Z:
            _carry_ghost(f2, w2, h2, idx + dn_z, PHZ * v2,
                         kp_d2, km_d2, dw_d2, start, own2 & near_k)
        if NEAR_Z and FAR_X:
            _carry_ghost(f2, w2, h2, idx + dn_z + df_x, -PHZ * PHX * v2,
                         kp_d2, km_d2, dw_d2, start, own2 & near_k & far_i)
        if NEAR_Z and FAR_Y:
            _carry_ghost(f2, w2, h2, idx + dn_z + df_y, -PHZ * PHY * v2,
                         kp_d2, km_d2, dw_d2, start, own2 & near_k & far_j)
        if NEAR_Z and (FAR_X and FAR_Y):
            _carry_ghost(f2, w2, h2, idx + dn_z + df_x + df_y,
                         PHZ * PHX * PHY * v2, kp_d2, km_d2, dw_d2, start,
                         own2 & near_k & far_i & far_j)

else:  # pragma: no cover - laptop path
    folded_fused_curl_constitutive_B = None  # type: ignore[assignment]


def folded_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_fused_curl_constitutive_B is None:
        raise ImportError(
            "the folded fused magnetic B/H kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return folded_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def mirror_phases(grid: Any) -> Tuple[int, int, int]:
    """``PHX``/``PHY``/``PHZ``: the declared plane phase per axis, 0 where unfolded.

    ``grid.mirror_phase`` is the ONE input to a component's parity
    (``stepping._mirror_phases`` :2291), and ``stepping._symmetry_phase``
    (:2402-2414) RAISES on a ``None`` phase rather than folding with the
    even-mirror default. This bakes the phase into a constexpr, so an axis that
    cannot answer must be refused before a compile — which
    :func:`folded_fused_magnetic_pair_coverage` does through
    :func:`.symmetry.folded_axis_kinds`, whose folded branch already requires
    ``+1`` or ``-1``.
    """
    phases: List[int] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            phases.append(0)
            continue
        value = _call(grid, "mirror_phase", axis, default=None)
        phases.append(int(value) if value in (1, -1) else 0)
    return (phases[0], phases[1], phases[2])


def folded_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                        sources: Any = None) -> Coverage:
    """May ONE launch span ``step_B`` -> near fill -> wall -> ``update_H``?

    A conjunction of the two halves' own predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. Same
    construction as :func:`.dispersive_fused_pair.dispersive_fused_pair_coverage`
    and :func:`.coverage.fused_pair_coverage`.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    magnetic = folded_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not magnetic.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in magnetic.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E half and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no magnetic source" from not being told would
    # be the over-covering refusal this clause exists to prevent.
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

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR FILL IS CARRIED as of 2026-08-20, so what used to be a blanket
        # refusal of MIRROR_PERIODIC is now the set of conditions the carry's
        # OWNERSHIP MOVE rests on. `fill_folded_far_ghosts_B` (driver.py:3287)
        # images the top stored slot from stepping._far_reflect_rows' row, and the
        # lane that owns that destination is the one AT the reflect row — so a row
        # outside the allocation is an out-of-range write from a lane that owns
        # neither cell, not a soft error on a whole-plane assignment. The three row
        # checks are symmetry.mirror_ghost_fill_coverage's own (symmetry.py:842-895).
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
            reader = _stored_past_owned_reader()
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

        # THE NEAR FILL IMAGES STORED CELL 2 FROM THAT CELL'S OWN LANE. The folded
        # curl predicate already refuses a folded axis storing two cells or fewer;
        # restated because THIS family writes the destination from a different
        # lane and a missing source plane is an out-of-range write, not a soft
        # error. BOTH mirror codes: the near fill runs on either.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                        f"the near fill images stored cell {MIRROR_SOURCE_INDEX} and "
                        f"this kernel images it from that cell's own lane")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. On this family both act on
        # component m at stored cell 0 of axis m, and `_zero_metal` skips a folded
        # axis (:2237-2239) so they are disjoint by construction. CHECKED rather
        # than inferred: this module's own rule forbids reading coverage off
        # another module's guard, and an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall there; "
                    f"stepping._zero_metal skips a folded axis, so the two passes "
                    f"have drifted and the carry's disjointness no longer holds")

        # The phases the kernel bakes must be exactly the ones the array path
        # would use. `folded_axis_kinds` already refuses a folded axis whose phase
        # is not +-1; this re-derives them through the builder's own function so a
        # constexpr that came back 0 on a folded axis is caught here and not on
        # the device.
        phases = mirror_phases(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
                reasons.append(
                    f"axis {axis} is folded but its mirror phase resolves to "
                    f"{phases[axis]!r}, not +1 or -1")

    # The wall clear is carried inline, so the grid must be able to answer which
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips
    # a plane the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedFusedMagneticPairPlan:
    """One allocation-free launch for four of the driver's magnetic passes.

    ``chunk_bounds`` is the chunk record, required, and ONE record serves both
    halves, as in ``launch.FusedPairPlan``.

    ``h_state`` IS THE ``Fields`` THE H SIDE STARTS FROM, or ``None``, exactly as in
    ``launch.FusedPairPlan``: :meth:`run` passes ``start_h = 1`` while
    ``h_state.h_separated`` is still false and sets the flag after the first launch
    that has programs (an empty grid, the warm pass, starts nothing). ``None`` (the
    from-arrays route, the gates) means a state that has already started.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "phases",
        "reflect", "near", "far", "chunk_bounds",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_h_targets", "_h_aux", "_h_coefficients", "_cb", "_grid", "_kernel",
        "_h_state",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any, h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phases = tuple(int(value) for value in phases)
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT. The kernel branches its
        # curl half on the four codes and its two carries on these booleans; if
        # they could be passed independently they could disagree, and a kernel that
        # masked the top plane on one axis set and imaged the far ghost on another
        # is a plane of wrong values rather than a crash.
        self.near = tuple(code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
                          for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        for axis, code in enumerate(self.bc):
            folded = code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
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
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedFusedMagneticPairPlan")
        self._h_state = h_state
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._h_targets = tuple(CupyPointer(array) for array in h_targets)
        self._h_aux = tuple(CupyPointer(array) for array in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(array)) for array in h_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of this kernel. Dropping it is
        # not a slowdown, it is a DISARMING — every mutation leg would then launch
        # the shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        start = self.start_h()
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.reflect[0], self.reflect[1], self.reflect[2], start,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phases[0], PHY=self.phases[1], PHZ=self.phases[2],
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
        return (f"FoldedFusedMagneticPairPlan(shape={self.shape}, bc={self.bc}, "
                f"phases={self.phases}, zero_metal={self.zero_metal}, "
                f"near={self.near}, far={self.far}, reflect={self.reflect}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_fused_magnetic_pair(fields: Any, pml: Any, sources: Any = None,
                                    block: Optional[int] = None,
                                    num_warps: Optional[int] = 1,
                                    kernel: Any = None,
                                    chunk_layout: Any = None,
                                    ) -> Optional[FoldedFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``); its record is what both halves' forms key on. The
    plan starts H from ``fields`` (``h_state``), as ``launch.plan_fused_pair`` does.
    """
    if not folded_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return FoldedFusedMagneticPairPlan(
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
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H (stepping.py:948 vs the B curl's half_integer=True).
        # The kernel takes both and never asks which is which, so a swap here is a
        # silent half-cell error in the absorber profile; the gate carries a
        # mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count
        # and `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


def plan_folded_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, phases,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None), *,
        chunk_bounds: Any) -> FoldedFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

    ``chunk_bounds`` is REQUIRED and has no default, as in
    ``launch.plan_from_arrays``: a plan without the record of the division the
    caller built is refused by name. The state is taken as already started
    (``start_h = 0``).
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, phases,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        reflect=reflect,
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
