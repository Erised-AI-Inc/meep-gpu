"""The folded ELECTRIC seam in one launch: ``step_D`` welded into ``update_E``.

THE TRITON TWIN of :mod:`meep_gpu.metal_kernels.folded_fused_pair`. Same seam, same
four driver passes, same ownership restructure; the arithmetic is transcribed from
the Triton arms this backend already certifies rather than from the Metal text, so
the two products are siblings and not a port.

WHAT IT IS WORTH, MEASURED BEFORE IT WAS BUILT, and the number is small on purpose.
From the 186-row Triton census
(``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired_convention``) the
electric seam funnels::

    folded D->E   77 curl -> 53 constitutive -> 4 no-electric-source -> 2 folded METALLIC

and BOTH of the last two steps have since been retired by work, in that order. This
paragraph is kept as the record of what the funnel looked like before, because the
two retirements are the product's whole history:

* the ``2 folded METALLIC`` step was the blanket refusal of a folded PERIODIC axis,
  which put ``fill_folded_far_ghosts_D`` inside the seam. RETIRED 2026-08-21, when
  the far fill was carried inline and the blanket refusal was replaced by the
  carry's own reflect-row conditions (:1046 below), under gate
  ``results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/``.
* the ``4 no-electric-source`` step was the claim that the 49 rows depositing an
  electric source between ``step_D`` and ``update_E`` (driver.py:3294-3299) "can
  never be fused by any product on any backend". THAT SENTENCE WAS WRONG, and this
  module said it. RETIRED 2026-08-30 with :data:`CARRIES_DEPOSIT_REPAIR`: the seam
  IS fusable when the deposit is saved before the launch and recomputed after, over
  the closure of the cells the post-injection fills image it into. Metal retired the
  same sentence on 2026-08-28.

So the funnel this module is measured on is no longer ``77 -> 53 -> 4 -> 2``, and a
reader who wants today's number must read it off the current census rather than off
this paragraph. The two rows the pre-carry product served were
``tests:TestLoadDump.test_load_dump_fields_3d`` and
``tests:TestLoadDump.test_load_dump_fields_sharded_3d``.

THE SAME TWO ROWS THE METAL PRODUCT SERVES, and that agreement is a measurement
rather than a coincidence: the Metal census
(``results/metal_coverage_tranche6_2026-08-19``) funnels 77 -> 53 -> 4 on its own
predicates, over the SAME four rows, and its shipped predicate admits 2. Two
independently written predicate stacks agreeing row-for-row on a clause ladder this
long is the cross-check the family was asked for; where they disagree, one of them
is wrong and finding which is worth more than the product.

===========================================================================
THE FOUR PASSES IN THE SEAM
===========================================================================

The driver runs four passes between ``step_D`` and ``update_E``
(driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — REFUSED BY NAME.** ``driver.step`` injects them between
  the halves (driver.py:3294-3299), so a fused pair would consume a pre-injection
  ``D``. Ignorance is never an empty set: ``Fields`` does not hold the source list,
  so an undeclared ``sources`` is a REFUSAL and not an assumed ``()``. This is the
  clause that caps the family — 49 of 53 rows — and it is a fact about the driver,
  not a scope this module chose.
* **``fill_symmetry_bc_D`` — CARRIED INLINE** (driver.py:3300;
  ``stepping.fill_symmetry_bc_D`` :1398, ``_fill_symmetry_ghost_cells`` :1409,
  ``_write_mirror_ghost`` :1451). It writes ``cell 0 = parity * cell 2`` on every
  folded axis for every component whose Yee shift there is 0, ``update_E`` then
  reads exactly those cells, and it is NOT dead: a pair that skipped it would feed
  the constitutive half a pre-fill ``D`` on TWO of the three components of every
  folded axis.
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301;
  ``stepping._zero_metal`` :2208-2247), through :func:`.coverage.zero_metal_axes`,
  which is IMPORTED rather than re-spelled.
* **``fill_folded_far_ghosts_D`` — CARRIED INLINE as of 2026-08-21.** It runs only
  on a folded PERIODIC axis (``stepping._stored_past_owned`` :1454-1469, consumed at
  ``_fill_folded_far_ghosts`` :1489) and images the LAST stored slot from a runtime
  reflect row on the component's OWN axis. It was REFUSED BY NAME until then, on
  the ground that its source and destination take DIFFERENT constitutive
  coefficient entries; that is TRUE and is not what made it impossible. The
  magnetic twin had already shown the way out — load the destination's pair
  explicitly, two extra scalar loads — and the refusal was retired only once the
  carry below was in, never before. The folded curl's top-plane mask
  (``symmetry.pml_curl_step_folded``'s second added block) comes back with it, and
  is transcribed here from that emitter's ``BACKWARD == 1`` arm.

===========================================================================
THE D GEOMETRY IS NOT THE B GEOMETRY — what differs from the magnetic twin
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219): ``Dx (1,0,0)``, ``Dy (0,1,0)``,
``Dz (0,0,1)`` against ``Bx (0,1,1)``, ``By (1,0,1)``, ``Bz (1,1,0)``. The near
fill touches component ``m`` on axis ``a`` exactly when ``iyee[m][a] == 0`` and the
far fill exactly when ``iyee[m][a] == 1``, so the two families swap BOTH roles:

    D family:  NEAR on a != m (two axes)      FAR on a == m (one, its own)
    B family:  NEAR on a == m (one, its own)  FAR on a != m (two axes)

FOUR CONSEQUENCES, each the mirror image of the one
:mod:`.folded_fused_magnetic_pair` records:

1. **THE COEFFICIENT INDEX MOVES ON THE FAR HALF AND NOT ON THE NEAR ONE — the
   OPPOSITE of the magnetic twin.** ``update_E`` indexes ``kps``/``kms`` on the
   component's own axis (``stepping.E_CONSTITUTIVE_TERMS`` :227, ``update_E``
   :983-987). The near fill images along an axis that is never the component's own,
   so its source and destination take the SAME entry and the carry reuses the
   registers already loaded. The FAR fill images along exactly the coefficient
   axis, so its destination at stored row ``n_m - 1`` takes a DIFFERENT entry than
   its source at the reflect row, and the source lane loads it explicitly
   (``kp_f0``/``km_f0`` below, two extra scalar loads per component — the same
   construction the magnetic twin uses at index 0, transposed to the top plane).
2. **PARITY PRODUCTS ARE REAL ON BOTH HALVES, AND THE COMPOSITION IS CHAINED.** A D
   component is a NEAR destination on up to two axes and a FAR destination on one,
   so one source lane owns up to SEVEN ghost cells — every nonempty subset of
   {near_a, near_b, far_m}. The composed value is NOT the product of the raw
   parities against the owned value. It is the driver's three passes evaluated in
   order, which for a far-over-near cell means the far parity times the ALREADY
   CLAMPED near ghost.
3. **THE WALL CLEAR CAN MEET THE NEAR FILL, AND ITS ORDER RELATIVE TO EACH FILL IS
   OPPOSITE.** ``_zero_metal`` SKIPS a folded axis (:2237-2239), so a folded axis is
   never walled — but on the D family the clear acts on the same axis set as the
   NEAR fill (``iyee == 0``, i.e. ``a != m``), and a component folded on ``y`` can
   still be walled on ``z``. The driver runs ``fill_symmetry_bc_D`` (:3300), then
   ``zero_metal_D`` (:3301), then ``fill_folded_far_ghosts_D`` (:3302), so

       a NEAR ghost is  clear(parity * v)   — parity first, clear second;
       a FAR  ghost is  parity * clear(v)   — clear first, parity second, and
                                              NOTHING clears it afterwards.

   Those differ in exactly the sign bit of a zero, which is a byte difference on
   the uint32 view every gate here compares. On the magnetic twin the question is
   VACUOUS (there the clear and the near fill act on the same axis, which the fold
   excludes); here it is not.
4. **THE FAR DESTINATION INHERITS ITS SOURCE'S WALL FLAGS.** The far ghost differs
   from its source only on axis ``m``, which is folded and therefore never walled,
   so it sits at the same coordinate on every axis that could clear it. That is why
   consequence 3's "nothing clears it afterwards" is expressible at all: the value
   the far carry images is the source lane's ``v`` read AFTER the clear lines, which
   is the driver's own order. The predicate CHECKS the fold/wall disjointness rather
   than inferring it from ``zero_metal_axes``' construction.

**THE COMPOSITION LAW WAS MEASURED BEFORE THIS KERNEL WAS WRITTEN.**
``parity/meep_gpu/measure_d_side_composition_law.py`` runs the driver's three
passes over random ``D`` on 156 grid declarations — one, two and three folded axes,
both mirror phases, EVEN and ODD full counts, with and without a wall on an
unfolded axis, over uniform, signed-zero-lattice and subnormal-band value classes —
and scores five candidate rules as uint32 words. The chained rule above reproduces
the array path on **156 of 156**; each of the four alternatives is a transcription
error this kernel could have made, and each is CAUGHT:

    frozen_near_source   72/156 match, caught on 84   (the corner loses a parity)
    clear_before_parity 120/156 match, caught on 36   (near ghost order swapped)
    far_reclears        120/156 match, caught on 36   (far ghost cleared after)
    far_flat_product     86/156 match, caught on 70   (composite unchained)

Zero cases were vacuous. ``results/d_side_composition_law_2026-08-21/``.

===========================================================================
THE OWNERSHIP RESTRUCTURE — one launch, no barrier, no race
===========================================================================

The fill reads a cell THIS LAUNCH WRITES. Triton has no device-wide barrier, so the
naive carry — the program holding stored cell 0 reading stored cell 2 — races on
``D``. Ownership moves instead. For component ``m``:

* a lane at stored cell 0 of a NEAR axis, or at the TOP plane of the FAR axis, is a
  DESTINATION. It computes the curl and the split-field ``n`` and stores ``fu_D``
  (the array path's ``step_D`` writes ``fu`` at every cell and neither fill touches
  it — :1449-1450 and :1529-1532 write ``field``, never ``fu_field``), and then
  STOPS: it never forms a stored ``v``, never loads or stores ``D``, and never
  touches ``E`` or ``f_w_E``. Its stepped displacement is discarded by the array
  path too — the fill overwrites it;
* the lane at stored cell 2 on each near axis and at the reflect row on the far one
  is the SOURCE. It does its own cell in full and then writes every destination
  that back-substitutes to it — up to SEVEN — each with its chained value, the
  displacement stored, and the constitutive update performed there. A NEAR-only
  destination does not move the indexed axis and takes the source lane's
  coefficient pair; any destination with the FAR half in it sits at stored row
  ``n_m - 1`` on the indexed axis and takes the pair loaded there.

EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWNERSHIP MASK. That is not a
precaution: it is the defect the magnetic twin's device gate found on 2026-08-20,
in the same shape this family would meet it. With two fills a lane can be the SOURCE
of one and the DESTINATION of the other — on a grid folding ``x`` PERIODIC and
``y``, the lane at ``(i = rx, j = 0)`` is the far source for ``Dx`` on ``x`` and is
itself the near fill's destination on ``y`` — and it would then write a ghost from a
``v`` built on an ``f`` load its own mask zeroed, racing the lane that legitimately
owns the composite cell.

Every destination word is written by exactly one lane, no lane reads a word another
lane writes — the destination's ``D``/``E``/``f_w_E`` loads are masked OFF rather
than merely unused, which is what makes that a statement about memory traffic and
not about dead registers — and no barrier is needed.

===========================================================================
WHAT IS DELIBERATELY *NOT* NEW
===========================================================================

* the folded curl half is :func:`.symmetry.pml_curl_step_folded` with
  ``BACKWARD = 1``, byte-copied — INCLUDING the ``MIRROR_PERIODIC`` top-plane block,
  which arrived with the far carry and is transcribed from that emitter's
  ``BACKWARD`` arm (:294-301: ``Dx (1,0,0) Dy (0,1,0) Dz (0,0,1)`` — shift 1 on its
  OWN axis only, three lines against the B arm's six);
* the wall clear is :mod:`.dispersive_fused_pair`'s ``ZM`` block, byte-copied — the
  same six rows ``stepping._zero_metal`` writes for the D family;
* the constitutive half is :func:`.kernels.constitutive_step`'s ``SCALE = 1`` arm on
  ``(Ex, Ey, Ez) <- (Dx, Dy, Dz) * inverse epsilon``, byte-copied, with
  ``src = tl.load(g + idx) * tl.load(e + idx)`` replaced by the register the curl
  half just produced times the same inverse-epsilon load — **D on the LEFT**,
  because the array path writes ``source * fields.inverse_epsilon_for(component)``
  (stepping.py:1011-1013) and float multiplication is bitwise commutative but a
  transcription is not a place to rely on that;
* the parity is ``fields.mirror_parity(c, axis, phase) ==
  phase * (1 - 2 * iyee[c][axis])`` (fields.py:117): ``+phase`` on the NEAR fill,
  which only ever touches shift-0 components, and ``-phase`` on the FAR one, which
  only ever touches shift-1 ones. The multiply is by exactly +-1.0, exact in float32
  for every input including subnormals and signed zeros — which is what lets the
  composed value be applied one parity at a time in the array path's own order
  instead of as a single product.

ROUTED 2026-08-27, AND STILL NOT AN ARM — the two are different questions and this
paragraph used to answer only the first. ``launch.plan_step`` assigns at most one
ARM per ``STEP_ORDER`` slot and this product spans FIVE driver passes, so it can
still claim no slot through the arm table and registers none. What it now has is
the OTHER route: ``launch._install_folded_fused_pairs`` (launch.py:2189-2258) calls
this module's own ``folded_fused_pair_coverage`` and ``plan_folded_fused_pair``
under ``fuse=True`` on a folded grid, and on admission ``_install_fused_pair`` puts
the plan in ``step_D`` with a ``NoopPlan`` naming it in ``update_E``. The absorb
clause in front of it (``launch._pair_may_absorb``, table
``FOLDED_FUSED_PAIR_ARMS``) requires the ``folded PML`` curl arm and the ``folded``
constitutive arm to have already won those two slots, which are this predicate's
own two conjuncts.

DISPATCH IS STILL NOT REACHED, and that is a separate mechanism rather than a
restatement. ``fastpath.plan_fast_path`` asks for ``fuse=False`` (fastpath.py:2076)
and additionally refuses the WHOLE plan if any slot comes back carrying a label
``fastpath.arm_is_fused`` recognises — which is why the label this module's plan is
installed under leads with ``fused pair`` (launch.py:2145-2160). A default run
therefore still never reaches this plan.

===========================================================================
DEVICE STATUS OF THE FAR CARRY — **RELEASED 2026-08-21**
===========================================================================

the GPU host RTX A6000, physical GPU 4 (pinned by UUID
``GPU-00000000-0000-0000-0000-000000000002``, verified physically empty before the
run: absent from every compute-apps line, 3 MiB), Triton 3.1.0 / CuPy 13.5.1, under
the ``keep`` float32 subnormal policy (``ieee_keep_ftz_stripped``).
``parity/meep_gpu/results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/``:

* **11/11 device cases, 10 complete driver steps each, bit-identical** on the uint32
  view of every allocated volume, to BOTH the CuPy array path and the three
  separately certified Triton products this launch replaces — one fused launch per
  step, 24 of 26 volumes moved (the two that did not are the read-only material
  inputs). FOUR of those cases fold a PERIODIC axis and TWO are 3-D, which is what
  makes the far carry measured rather than merely written;
* both ARMED HARNESS mutations refused as they must;
* **16/16 kernel mutations caught**, including all six the far carry brought with
  it; the seventeenth (a commuted multiply) is a declared null and was confirmed as
  one;
* 4/4 predicate rows, now including the ADMISSION of a folded PERIODIC grid.

THE RUN FOUND THREE DEFECTS, AND ALL THREE WERE IN THE GATE RATHER THAN THE KERNEL.
Each was a mutation that came back UNCAUGHT while measuring nothing, and each was
diagnosed by measurement rather than argued away:

* ``m12`` (the CHAINED composite) was scored on a case that does not fold Y at all,
  so the ``if FAR_X and NEAR_Y:`` block it rewrites was compile-time absent — a
  DEAD-BRANCH mutation. Worse, it is a structural null on ANY 2-D case: the only
  axis that can clamp ``gv_0y`` is z, and z is invariant in 2-D. And on an EVEN Y
  plane it is null again, because the kernel's ``v`` is already wall-clamped, so the
  re-clamp can only change the sign bit of a zero. It now runs on a 3-D case that
  folds X PERIODIC, folds Y at an ODD phase, and carries a z wall;
* ``m13`` (the moved coefficient index) was scored on an EVEN full count, where the
  reflect row is ``nx - 2``, the destination is ``nx - 1``, and the PML arrays hold
  the SAME value at both (``kps_x_h[18] == kps_x_h[19]`` exactly) because the
  not-owned ghost slot copies its neighbour. A coincidence of the absorber's
  grading, not a property of the carry. It now runs on the ODD-count case, and a
  MEASURED precondition builds the grid and refuses the pairing if the two entries
  are equal;
* ``m10`` (the ghost's own material) had been re-expressed as "another COMPONENT's
  inverse-epsilon", which ``FdtdDriver.set_epsilon`` makes a bitwise no-op — it is
  documented isotropic and ``inverse_epsilon_for`` returns one array for all three
  components. It is now expressed through the POINTER, reaching the source cell.

**THE 2-D CASE TABLE COULD NOT REACH A THIRD OF THIS KERNEL.** In a 2-D run z is
invariant, so ``FAR_Z``, ``NEAR_Z``, every ``gv_2*`` register and the
``kp_f2``/``km_f2`` pair are compile-time absent. That is the same defect an
adversarial verifier found in the magnetic twin on 2026-08-21, and it cost nothing
here only because the 3-D cases were added before the release rather than after it.

**WHAT THE CARRY IS WORTH, and it is small on purpose.** Two seam-instances on the
Triton board — ``examples:binary_grating_phasemap.py`` and
``examples:diffracted_planewave.py``, the only two reachable D->E instances whose
``live_in_seam_passes`` contain ``fill_folded_far_ghosts_D``. That takes this
product from 2 of its cell's 4 reachable instances to 4 of 4, and the Triton D->E
seam from 2 to 4 of its 29 reachable ones.

THE ORDER THIS LANDED IN. The carry was transcribed first, against a composition law
measured on 156 grid declarations; the predicate's blanket refusal STOOD while it
was unmeasured, so nothing could price it; the gate reached the ``FAR_a`` blocks
through ``plan_folded_fused_pair_from_arrays``, which documents itself as bypassing
the predicate; and the refusal was retired only once that gate came back ALL GREEN.
:class:`FoldedFusedPairPlan` accepts a folded PERIODIC code and validates its
reflect row, where it used to raise.

ROUTED BUT STILL NOT DISPATCHED. A weld licenses the claim, not a dispatch. Since
2026-08-27 ``launch.plan_step(..., fuse=True)`` on a folded grid does build this
plan (see the routing paragraph above); ``fastpath`` asks for ``fuse=False`` and
refuses any plan carrying a ``fused pair`` label, so nothing in a default run
reaches it.

DEVICE STATUS OF THE 2026-08-20 KERNEL (the near carry alone, superseded by the
    edit above): **RELEASED 2026-08-20**, the GPU host RTX A6000 GPU 4 (verified empty by
    UUID before and after), Triton 3.1.0 / CuPy 13.5.1, under the ``keep`` float32
    subnormal policy (``ieee_keep_ftz_stripped``).
    ``parity/meep_gpu/results/triton_folded_fused_pair_2026-08-20/``:

    * **5/5 device cases, 10 complete driver steps each, bit-identical** on the
      uint32 view of every allocated volume, to BOTH the CuPy array path and the
      three separately certified Triton products this launch replaces — one fused
      launch per step, 24 of 26 volumes moved (the two that did not are the
      read-only material inputs);
    * both ARMED HARNESS mutations refused as they must: the un-substituted route
      agrees bytewise and is caught by the launch counter alone, and the frozen
      seam agrees trivially and is caught by the moved-state census alone;
    * **10/11 kernel mutations caught**, including the corner carry, the parity
      PRODUCT, the ghost's own wall clear and the ghost's inverse-epsilon index;
      the eleventh (a commuted multiply) is a declared null and was confirmed as
      one over three steps;
    * 4/4 refusals on real CuPy drivers.

    A FIRST RUN FAILED, and the failure was in the harness rather than the kernel:
    the staged tree did not carry the 186-row censuses, so the corpus leg funnelled
    0 -> 0 and reported "worth zero seam-instances" — a measurement about the
    staging dressed as one about the corpus. That leg now refuses a census that
    does not yield 186 measured rows, by name.

    ROUTED BUT STILL NOT DISPATCHED, 2026-08-27 — see the routing paragraph above.
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
#: (driver.py:3292-3304). Declared, never inferred from the slot name.
#:
#: ``fill_folded_far_ghosts_D`` JOINED THIS TUPLE IN THE SAME EDIT that put the
#: carry in the kernel body. A module that claims a pass it does not carry — or
#: carries one it does not claim — is the defect this family has already paid for
#: once on the Metal board, and a passing test pinned the stale tuple and hid it.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The two codes a folded axis can take. The NEAR fill runs on either
#: (``stepping._fill_symmetry_ghost_cells`` gates on the mirror phases alone); the
#: FAR fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``).
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without importing :mod:`launch`; ``test_triton_folded_fused_pair``
#: pins the two equal.
BACKWARD = 1

#: Which folded axes the near fill images for each D component. Derived from
#: ``TARGET_IYEE`` at import rather than written out, and pinned by test against
#: ``fields.IYEE_SHIFTS``: axis ``a`` fills exactly the components whose Yee shift
#: there is 0, which for D is every component EXCEPT ``a``.
NEAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 0)
    for name in ("Dx", "Dy", "Dz")
)

#: Which folded PERIODIC axes the FAR fill images for each D component. The exact
#: complement of :data:`NEAR_FILL_AXES` — ``stepping._fill_folded_far_ghosts``
#: (:1489-1532) writes axis ``a`` for every component whose Yee shift there is ONE —
#: and derived from the same table for the same reason. For D that is the
#: component's OWN axis and only it, so the two sets never overlap and a folded
#: PERIODIC axis gives one component a far ghost and two a near one.
FAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 1)
    for name in ("Dx", "Dy", "Dz")
)

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAR_FILL_AXES",
    "MIRROR_CODES", "NEAR_FILL_AXES", "REPLACES",
    "FoldedFusedPairPlan", "folded_fused_curl_constitutive_D",
    "folded_fused_curl_constitutive_D_kernel", "folded_fused_pair_coverage",
    "mirror_phases", "plan_folded_fused_pair", "plan_folded_fused_pair_from_arrays",
]


if triton is not None:

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def _carry_ghost_E(f, w, e, ie, dst, ghost, kp_d, km_d, mask):
        """Write ONE imaged ghost cell of ``D`` and run ``update_E`` there.

        THE EIGHT STATEMENTS ARE THE CERTIFIED ONES, MOVED AND NOT REWRITTEN: they
        are the displacement store followed by ``kernels.constitutive_step``'s
        ``SCALE=1`` arm with ``src`` bound to ``ghost * inverse_epsilon`` instead of
        ``load(g) * load(e)``, which is exactly what the owned cell above does. They
        became a device function when the far carry landed, because a D component
        can now be the source of up to SEVEN ghost cells (two near planes, one far
        plane, and every composition of them) and seven inlined copies of the same
        eight statements is seven places for one of them to drift.

        ``ghost`` is ALREADY the composed value — parity applied, and the wall
        clear applied or not according to which fill wrote it (see consequence 3 in
        the module docstring). This function decides nothing about either; it
        stores what it is handed.

        The ORDER is the array path's: the displacement store, then the workspace
        read, the workspace write, and the ``E`` accumulate. ``prev`` is read BEFORE
        the write, which is the one ordering the constitutive cannot survive being
        wrong about. ``D`` IS ON THE LEFT of the inverse-epsilon multiply because
        the array path writes ``source * fields.inverse_epsilon_for(component)``
        (stepping.py:1011-1013); float multiplication is bitwise commutative but a
        transcription is not a place to rely on that.
        """
        tl.store(f + dst, ghost, mask=mask)
        prev = tl.load(w + dst, mask=mask, other=0.0)
        src = ghost * tl.load(ie + dst, mask=mask, other=0.0)
        tl.store(w + dst, src, mask=mask)
        acc = tl.load(e + dst, mask=mask, other=0.0)
        acc = acc + kp_d * src
        acc = acc - km_d * prev
        tl.store(e + dst, acc, mask=mask)

    @triton.jit
    def folded_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx, Dy, Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz
        g0, g1, g2,                       # curl sources: Hx, Hy, Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex, Ey, Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex, f_w_Ey, f_w_Ez
        ie0, ie1, ie2,                    # inverse epsilon, per target
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx,
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``fill_symmetry_bc_D`` + ``zero_metal_D`` + ``update_E``.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim
        copy of :func:`.symmetry.pml_curl_step_folded` rather than a hand-
        specialised one, which is the whole reason the transcription risk here is
        low. It cannot be 0: the near-fill carry below is transcribed for the D
        family's Yee shifts, where the fill touches component ``m`` on the two
        axes that are NOT its own; the B family's shifts put the fill on the
        component's own axis, which is a different carry, one destination per
        component, and a coefficient index that MOVES.

        ``BCX``/``BCY``/``BCZ`` take ALL FOUR of :mod:`.symmetry`'s codes as of
        2026-08-21, ``MIRROR_PERIODIC`` included: this kernel now carries
        ``fill_folded_far_ghosts_D`` (driver.py:3302), which is the pass that used
        to make that code a refusal. The folded curl's top-plane mask block comes
        with it and is transcribed below.

        ``NEAR_a`` / ``FAR_a`` say which fill runs on axis ``a``: NEAR on either
        mirror code (``stepping._fill_symmetry_ghost_cells`` gates on the mirror
        phases alone) and FAR only on ``MIRROR_PERIODIC``
        (``stepping._stored_past_owned`` :1454-1469). They are DERIVED FROM ``bc``
        by the plan and are not independent inputs, so they cannot disagree with
        the codes the curl half branches on.

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
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

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

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        # The PML coefficient vectors are built at the STORED extent on a folded
        # axis, so the n_a indexing needs no fold-aware change.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=own0, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=own1, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=own2, other=0.0) * km_y) + n2) - p2) * si_y

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

        # fu is written at EVERY cell, destinations included: the array path's
        # step_D writes it everywhere and fill_symmetry_bc_D does not touch it
        # (stepping.py:1497-1498 writes `field`, never `fu_field`).
        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=own0)
        tl.store(f1 + idx, v1, mask=own1)
        tl.store(f2 + idx, v2, mask=own2)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.constitutive_step's SCALE=1 arm, with
        # `tl.load(g + idx)` replaced by the register the curl half just produced.
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

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=own0, other=0.0)   # BEFORE the store.
        src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)
        tl.store(w0 + idx, src0, mask=own0)
        a0v = tl.load(e0 + idx, mask=own0, other=0.0)
        a0v = a0v + kp_0 * src0
        a0v = a0v - km_0 * prev0
        tl.store(e0 + idx, a0v, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=own1, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=own1, other=0.0)
        tl.store(w1 + idx, src1, mask=own1)
        a1v = tl.load(e1 + idx, mask=own1, other=0.0)
        a1v = a1v + kp_1 * src1
        a1v = a1v - km_1 * prev1
        tl.store(e1 + idx, a1v, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=own2, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=own2, other=0.0)
        tl.store(w2 + idx, src2, mask=own2)
        a2v = tl.load(e2 + idx, mask=own2, other=0.0)
        a2v = a2v + kp_2 * src2
        a2v = a2v - km_2 * prev2
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
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,
                           kp_0, km_0, own0 & near_j)
        if NEAR_Z:
            gv_0z = PHZ * v0
            if ZM_Y:
                gv_0z = tl.where(at_y, 0.0, gv_0z)
            if ZM_Z:
                gv_0z = tl.where(at_z, 0.0, gv_0z)
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_z, gv_0z,
                           kp_0, km_0, own0 & near_k)
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
            _carry_ghost_E(f0, w0, e0, ie0, idx + dn_y + dn_z, gv_0yz,
                           kp_0, km_0, own0 & near_j & near_k)
        if FAR_X:
            # NO ZM AFTER THE PARITY: :3302 is the last pass before update_E, so
            # the far ghost is -phase times the value standing at its source, which
            # is `v0` read after the clear lines above.
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x, -PHX * v0,
                           kp_f0, km_f0, own0 & far_i)
        if FAR_X and NEAR_Y:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y, -PHX * gv_0y,
                           kp_f0, km_f0, own0 & far_i & near_j)
        if FAR_X and NEAR_Z:
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_z, -PHX * gv_0z,
                           kp_f0, km_f0, own0 & far_i & near_k)
        if FAR_X and (NEAR_Y and NEAR_Z):
            _carry_ghost_E(f0, w0, e0, ie0, idx + df_x + dn_y + dn_z, -PHX * gv_0yz,
                           kp_f0, km_f0, own0 & far_i & near_j & near_k)

        # --- component 1 (Dy): near on x and z, far on y -----------------------
        if NEAR_X:
            gv_1x = PHX * v1
            if ZM_X:
                gv_1x = tl.where(at_x, 0.0, gv_1x)
            if ZM_Z:
                gv_1x = tl.where(at_z, 0.0, gv_1x)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x, gv_1x,
                           kp_1, km_1, own1 & near_i)
        if NEAR_Z:
            gv_1z = PHZ * v1
            if ZM_X:
                gv_1z = tl.where(at_x, 0.0, gv_1z)
            if ZM_Z:
                gv_1z = tl.where(at_z, 0.0, gv_1z)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_z, gv_1z,
                           kp_1, km_1, own1 & near_k)
        if NEAR_X and NEAR_Z:
            # X is applied before Z: phase_z * (phase_x * v).
            gv_1xz = PHZ * (PHX * v1)
            if ZM_X:
                gv_1xz = tl.where(at_x, 0.0, gv_1xz)
            if ZM_Z:
                gv_1xz = tl.where(at_z, 0.0, gv_1xz)
            _carry_ghost_E(f1, w1, e1, ie1, idx + dn_x + dn_z, gv_1xz,
                           kp_1, km_1, own1 & near_i & near_k)
        if FAR_Y:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y, -PHY * v1,
                           kp_f1, km_f1, own1 & far_j)
        if FAR_Y and NEAR_X:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x, -PHY * gv_1x,
                           kp_f1, km_f1, own1 & far_j & near_i)
        if FAR_Y and NEAR_Z:
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_z, -PHY * gv_1z,
                           kp_f1, km_f1, own1 & far_j & near_k)
        if FAR_Y and (NEAR_X and NEAR_Z):
            _carry_ghost_E(f1, w1, e1, ie1, idx + df_y + dn_x + dn_z, -PHY * gv_1xz,
                           kp_f1, km_f1, own1 & far_j & near_i & near_k)

        # --- component 2 (Dz): near on x and y, far on z -----------------------
        if NEAR_X:
            gv_2x = PHX * v2
            if ZM_X:
                gv_2x = tl.where(at_x, 0.0, gv_2x)
            if ZM_Y:
                gv_2x = tl.where(at_y, 0.0, gv_2x)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x, gv_2x,
                           kp_2, km_2, own2 & near_i)
        if NEAR_Y:
            gv_2y = PHY * v2
            if ZM_X:
                gv_2y = tl.where(at_x, 0.0, gv_2y)
            if ZM_Y:
                gv_2y = tl.where(at_y, 0.0, gv_2y)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_y, gv_2y,
                           kp_2, km_2, own2 & near_j)
        if NEAR_X and NEAR_Y:
            # X is applied before Y: phase_y * (phase_x * v).
            gv_2xy = PHY * (PHX * v2)
            if ZM_X:
                gv_2xy = tl.where(at_x, 0.0, gv_2xy)
            if ZM_Y:
                gv_2xy = tl.where(at_y, 0.0, gv_2xy)
            _carry_ghost_E(f2, w2, e2, ie2, idx + dn_x + dn_y, gv_2xy,
                           kp_2, km_2, own2 & near_i & near_j)
        if FAR_Z:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z, -PHZ * v2,
                           kp_f2, km_f2, own2 & far_k)
        if FAR_Z and NEAR_X:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x, -PHZ * gv_2x,
                           kp_f2, km_f2, own2 & far_k & near_i)
        if FAR_Z and NEAR_Y:
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_y, -PHZ * gv_2y,
                           kp_f2, km_f2, own2 & far_k & near_j)
        if FAR_Z and (NEAR_X and NEAR_Y):
            _carry_ghost_E(f2, w2, e2, ie2, idx + df_z + dn_x + dn_y, -PHZ * gv_2xy,
                           kp_f2, km_f2, own2 & far_k & near_i & near_j)

else:  # pragma: no cover - laptop path
    folded_fused_curl_constitutive_D = None  # type: ignore[assignment]


def folded_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return folded_fused_curl_constitutive_D


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
    :func:`folded_fused_pair_coverage` does through
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


def folded_fused_pair_coverage(fields: Any, pml: Any,
                               sources: Any = None) -> Coverage:
    """May ONE launch span ``step_D`` -> near fill -> wall -> ``update_E``?

    A conjunction of the two halves' own predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. Same
    construction as :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage`.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    electric = folded_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not electric.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D. A
    # MAGNETIC source is injected in the B/H half and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no electric source" from not being told would
    # be the over-covering refusal this clause exists to prevent. This is the
    # clause that caps the family at 4 of 53 rows on the measured corpus.
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
        # THE FAR FILL IS CARRIED, AND THE REFUSAL THAT STOOD IN FOR IT IS RETIRED —
        # IN THAT ORDER, AND WITH A DEVICE GATE BETWEEN THE TWO.
        #
        # `fill_folded_far_ghosts_D` (driver.py:3302) was refused by name until
        # 2026-08-21. It is now transcribed into the kernel body, and the blanket
        # refusal is replaced by the conditions the carry's OWNERSHIP MOVE actually
        # rests on: the pass images the top stored slot from
        # stepping._far_reflect_rows' row, and the lane that owns that destination
        # is the one AT the reflect row — so a row outside the allocation is an
        # out-of-range write from a lane that owns neither cell, not a soft error on
        # a whole-plane assignment.
        #
        # THE HAZARD WAS CARRIED FIRST, THEN MEASURED, THEN THE REFUSAL RETIRED.
        # The clause did NOT lift when the carry was written: it stood, saying what
        # it was waiting for, while `plan_folded_fused_pair_from_arrays` (which
        # documents itself as bypassing this predicate) gave the gate a route to the
        # `FAR_a` blocks. It lifted only after
        # `results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/` came back
        # ALL GREEN. Lifting it earlier would have let the fusion matrix price two
        # more seam-instances against bytes nothing had measured.
        #
        # The row checks below are symmetry.mirror_ghost_fill_coverage's own, which
        # the magnetic twin already carries for its released far fill.
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

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. `_zero_metal` skips a folded
        # axis (:2237-2239), so no fold destination plane is a wall's cleared
        # plane and the imaged ghost's wall flags are its source's.
        # CHECKED rather than inferred: this module's own rule forbids reading
        # coverage off another module's guard, and an overlap would be a plane of
        # wrong values.
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
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A susceptibility makes the constitutive source (D - sum P) rather than D.
    # `folded_constitutive_coverage` already refuses it on the E side; restated
    # because this kernel bakes the plain product and a reader should not have to
    # chase the other predicate to learn that.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: this kernel bakes the plain "
            "constitutive product, whose source is D and not (D - sum P)")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedFusedPairPlan:
    """One allocation-free launch for FIVE of the driver's electric passes."""

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "phases",
        "reflect", "near", "far",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_grid",
        "_kernel",
    )

    #: The driver passes one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phases, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

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
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        if not any(self.near):
            raise ValueError(
                "no axis is folded: this product exists to carry the mirror fills "
                "inside the D seam, and an unfolded grid belongs to the certified "
                "curl and constitutive pair")
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
                  else folded_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.reflect[0], self.reflect[1], self.reflect[2],
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

    def __repr__(self) -> str:
        return (f"FoldedFusedPairPlan(shape={self.shape}, bc={self.bc}, "
                f"phases={self.phases}, zero_metal={self.zero_metal}, "
                f"near={self.near}, far={self.far}, reflect={self.reflect}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_folded_fused_pair(fields: Any, pml: Any, sources: Any = None,
                           block: Optional[int] = None,
                           num_warps: Optional[int] = 1,
                           kernel: Any = None) -> Optional[FoldedFusedPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not folded_fused_pair_coverage(fields, pml, sources).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    return FoldedFusedPairPlan(
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
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both
        # and never asks which is which, so a swap here is a silent half-cell
        # error in the absorber profile; the gate carries a mutation for exactly
        # it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count
        # and `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_fused_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, phases,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None)) -> FoldedFusedPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FoldedFusedPairPlan(
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
        reflect=reflect,
        kernel=kernel, num_warps=num_warps,
    )
