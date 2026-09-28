"""The H->D weld: ``update_H`` welded into ``step_D``, one dispatch, scratch output.

THE FOURTH FUSION SEAM ON THIS BACKEND, and the first one whose two halves sit in
the order constitutive-then-curl. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and exactly one statement sits between the two consults: ``for source in electric:
getattr(source, "withdraw", _no_withdraw)(self.fields)``. Nothing is INJECTED here --
the electric injection is one seam later (:3317-3322) and the magnetic one is one seam
earlier (:3283-3284) -- so :mod:`..deposit_repair` has nothing to say about this seam
and :mod:`..withdraw_hoist` is the module that does.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

``step_D``'s curl reads ``H`` at the thread's own cell AND at its three BACKWARD
neighbours, and ``update_H`` writes ``H``. Welded in place, thread ``idx`` would read
``H[idx - stride]`` at a moment decided by the schedule -- the hazard every board on
this track files as structurally unweldable -- and it would read ``f_w_H`` the same
way, which is easy to miss: on this side ``f_w_H_new[ii] == B[ii]`` exactly, so an
in-place write hands a racing neighbour ``B`` where it needs ``B_prev``.

Both premises are removed the way :mod:`..cuda_kernels.offdiag_stencil_weld` and the
four Metal families built on :mod:`.offdiag_weld_common` remove them on the MIRROR-IMAGE
D->E seam:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go to
  LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  dispatch and no thread can observe another thread's store;
* **the foreign read is not a read of another thread's output.** It is a RECOMPUTE from
  that same unwritten state, through :func:`h_cell_function` -- the certified
  ``constitutive_step`` body for side ``H``, whole, evaluated at an arbitrary cell. A
  pure function of unwritten memory has no schedule to depend on.

The launcher then ROTATES the ``H``/``f_w_H`` bindings, which is
:class:`.offdiag_weld_common.ScratchWeldPairPlan`'s certified choreography.

WHY THE RECOMPUTE IS EXACT RATHER THAN CLOSE. ``update_H`` is POINTWISE: per component
it reads ``H``, ``f_w_H`` and ``B`` at ONE cell and writes the same cell
(stepping.py:907-923 through ``_apply_constitutive_pml``, stepping.py:2130-2143). So the
stepped value at any cell is a function of state this launch does not write, and
evaluating it twice gives the same bits by construction rather than by a tolerance.
That is CHEAPER than the D->E weld's recompute, whose recomputed half is itself a curl.

WHY THE COEFFICIENT INDEX NEEDS NO THEOREM. :func:`h_cell_function` takes ``(i, j, k)``
and indexes ``kps``/``kms`` at THOSE coordinates, exactly as the certified body indexes
them at the thread's own -- so the recompute is the certified arithmetic at the
recomputed cell and not an approximation of it. (A curl never differences a component
along that component's own axis, so those coefficient values happen to equal the own
cell's; that is a fact about the emitted code, not a premise this weld rests on. It is
asserted in ``test_metal_fused_hd_pair`` rather than relied on here.)

=============================================================================
THE SIGNATURE -- 30 POINTERS PLUS ONE PACKED STRUCT, AND THE CEILING IS AN EQUALITY
=============================================================================

    3 H_out + 3 f_w_H_out          (scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (pre-launch, const)
  + 3 B                            (const, the constitutive source)
  + 3 D     + 3 fu_D               (in place: the curl reads and writes its OWN cell
                                    only, so nothing here races)
  + 6 curl coefficients (kms/sinv per axis)
  + 3 constitutive kps                                                        = 30

plus one ``constant Params&`` is **31 of the 31 bindings the platform allows**
(:data:`.device.MAX_BUFFER_BINDINGS`; buffer attribute indices run 0..30 and a 32nd is
a COMPILE ERROR). There is NO headroom, so :data:`PACKED_BINDINGS` is pinned as an
EQUALITY and the gate compiles one pointer past the shipped shape and requires the
failure -- the standing :mod:`.fused_electric_pair` set.

WHAT MAKES 30 POSSIBLE IS A SHARED COEFFICIENT GROUP, and dropping the sharing does not
merely cost performance, it does not compile. Both halves sit on the same Yee
sub-lattice -- ``launch.SUB_STEPS['step_D']['suffix'] == ''`` and
``CONSTITUTIVE_SIDES['H']['half_integer'] is False`` -- so the curl's ``kms_x/_y/_z``
and the constitutive's ``kms_x/_y/_z`` ARE THE SAME THREE VOLUMES and are bound once.
Unshared the signature is 33 pointers plus ``Params`` = 34, which
:func:`refuted_unshared_kms_source` compiles and the gate requires to FAIL. Bind the
HALF-INTEGER set by mistake instead and nothing fails: you get a smooth, converged,
half-cell-wrong absorber profile, which is why the host mutation for it is a gate leg
and not a comment.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the two consults, so there is no deposit to bracket. The
pass that IS there is the electric withdraw, and :mod:`..withdraw_hoist` is its
contract -- one slot, not two; a hoist, not a bracket.

:data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND, and the flag and the wiring move
together exactly as ``CARRIES_DEPOSIT_REPAIR``'s do. The only wiring that performs the
hoist is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which wraps
the built plan in a :class:`..withdraw_hoist.LeadingWithdrawPlan`; this product declares
:data:`INSTALLABLE` False, so that branch is unreachable for it and NOTHING would
perform the withdraw before its launch.
A product that declared True without that wrapper would let the fused launch read a D
still holding the previous step's standing dipole and report success -- the exact
failure ``withdraw_hoist`` exists to prevent. So the predicate refuses, BY NAME, every
row whose electric withdraw does work, which is what the three boards already file
under ``withdraw_seam``.

WHAT THAT COSTS, MEASURED rather than estimated
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, joined on the
Metal arms): the ``(update_H ordinary -> step_D PML)`` cell is **49 rows**, of which
**47** are ``buildable_not_built`` and **2** are ``withdraw_seam`` --
``examples:differential_cross_section.py`` and
``tests:TestIntegratedSource.test_integrated_source``. This predicate therefore reaches
**47 of the cell's 49**, and what would raise it to 49 is a package of two things
landed together: :data:`INSTALLABLE` True (the ``FUSED_PAIR_ARMS`` row landed
2026-09-05), and a device gate leg that drives a complete step on a row with a live
INTEGRATED electric source with the hoist in place and requires both ``not_hoisted``
and ``after_step_D`` to diverge.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS A MEASURED VERDICT RATHER THAN A MISSING GATE
=============================================================================

:data:`INSTALLABLE` is False, and :data:`INSTALLABLE_REASON` carries the measurement
rather than a policy. Over the driver's ``step_B - update_H - step_D - update_E`` slot
path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour. Driven through the real
composer on 2026-09-04 over the 179 buildable Metal rows of the 194-row basis, at
composer level (predicate AND absorb row): on **133** rows both neighbours install and
this product would be a LOSS (2 pairs -> 1); on **22** it is a TIE (only the B->H pair
serves, 3 launches / 1 seam either way) and the released precedent keeps the slot with
the incumbent; on **22** only the D->E pair serves and the composer's later half refuses
this product by name; and on **2** rows -- the nonlinear cell, ``examples:3rd-harm-1d.py``
and ``tests:Test3rdHarm1d.test_3rd_harm_1d`` -- NEITHER neighbour installs and an H->D
product is the one genuine gain (4 launches / 0 seams -> 3 / 1). That cell is outside
this product's primary ``(ordinary -> PML)`` arm and is reached only through the
``FUSED_PAIR_EXTRA_ARMS`` row below, which no device gate has driven yet.

WHAT KEEPS THE RELEASED B->H PAIR IN ITS SLOT IS NOT THIS FLAG. It is
``launch._neighbouring_seam_claimant``'s end-edge guard (a B->H span is never asked
whether a neighbour claims it) together with ``_pair_may_absorb`` reading the live
``selected`` after that pair installs first -- measured on 2026-09-04, when a
registered, admitting stand-in with NO absorb row cost the released pair its seam on
every row it reached and only this flag stood in the way. The flag is kept as belt and
braces; the composition does not rely on it.

What this product IS worth is stated plainly, because no artifact here may imply
otherwise: the priced predicate gap closed under the owner rule *close all fusion gaps
regardless*, and a certified half of the only span that could ever pay -- the four-slot
``step_B -> update_H -> step_D -> update_E`` weld, which is NOT built here. **No timing
exists for this shape and none is licensed by anything in this module.** The fused route
does MORE memory traffic than the two singles for the same step-level launch count;
whether the saved launch and the saved ``H`` write-then-read round trip pay for three
extra pointwise constitutive evaluations per component per thread is a hypothesis.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

ONE ARM PAIR ONLY: ``(ordinary -> PML)``. The ``(folded -> folded)`` cell is 78 rows and
the largest on this backend, and it is a SECOND PRODUCT rather than a widening of this
one -- Metal expresses the fold as a separate arm with its own ghost map in
:mod:`.symmetry` / :mod:`.folded_complex`, where the CUDA track carries it as a runtime
boundary code inside one kernel. THAT PRODUCT IS :mod:`.folded_fused_hd_pair`, BUILT
2026-09-07, and it reuses this module's lift whole: the constitutive half is
:func:`h_cell_function` unchanged (the fold adds no kernel on ``update_H``) and the curl
half is this module's own tables applied over ``symmetry.folded_curl_source``. The halo
question this paragraph left open is ANSWERED and was answered by measurement, not by a
guess: a folded ``step_D`` reads no cell beyond the same three backward neighbours,
because Metal's folded ghost is the certified METALLIC literal rather than a mirror read
and every mirror-image plane the curl touches is an ``update_H`` output over a ``B`` the
driver filled before the seam opened. Every other
configuration -- complex storage, cylindrical, beta, BFAST, conductive, an inactive
absorber -- is refused through the two halves' own certified predicates, which this one
conjoins without weakening.

THE NONLINEAR CELL IS THE ONE WIDENING, 2026-09-05, AND IT IS A TEXT IDENTITY RATHER
THAN A THIRD KERNEL. A chi2/chi3 run selects ``nonlinear_update_e``'s spine arms on
these two slots -- ``nonlinear PML magnetic`` and ``nonlinear PML curl`` -- and those
arms ARE the ordinary kernels under a scope view (``plan_nonlinear_run_constitutive``
calls ``launch.plan_constitutive``, ``plan_nonlinear_run_pml_curl`` calls
``launch.plan_pml_curl``; the source each hands to ``compile_source`` was measured
byte-identical to the ordinary arm's on ``metal_composition_matrix``'s ``cart_pml_real``
against ``nonlinear_real``). The Pade factor enters ``update_E`` only, one seam later.
So the predicate admits a nonlinear run through those two arms' own predicates, and
``launch.FUSED_PAIR_EXTRA_ARMS`` carries the matching absorb row. It is the 2-row cell
on which NEITHER neighbouring pair installs -- the one measured gain.

**THAT GATE RAN, 2026-09-07, and the widening is now measured rather than argued.**
``parity/meep_gpu/gate_metal_fused_hd_pair_nonlinear.py`` drives this kernel through
the driver's own consult order on six chi2/chi3 fixtures -- both coefficient arms, both
boundary extremes, and TWO 1-D fixtures with the corpus cell's own ``[1, 1, n]`` shape
-- byte-identical as uint32 against the array path, the certified nonlinear spine
singles, and both compositions, with the in-place and rotation-skipped null controls
diverging and the shared mutation table re-driven here. Both corpus rows
(``examples:3rd-harm-1d.py``, ``tests:Test3rdHarm1d.test_3rd_harm_1d``) were re-lifted
and driven to zero differing words. Its ``arbitration`` leg measured **zero installed
pairs on all six fixtures**, which is this paragraph's "neither neighbour installs" in
launches. Record: ``results/metal_fused_hd_pair_nonlinear_2026-09-07/``.

**IT DOES NOT FLIP THE FLAG, and does not license flipping it.** A gain is a reason to
CONSIDER installing; installing is a composition decision with its own release. What
the gate settles is the arithmetic and the cell's credit -- and one more thing worth
saying plainly: a SECOND product for this cell would be a defect, not a widening. Two
fused products admitting one ``update_H`` slot leaves it UNSELECTED naming both
(``arms._select_slot``), so the gate's ``no_second_product`` leg asks every registered
weld on that slot and requires exactly one -- this one -- to admit.

CYLINDRICAL IS REFUSED PERMANENTLY, and not merely by this family's arm pairing: the
cylindrical ``step_D`` differences ``cylindrical_rderiv_prefix(Hy)``, which ends in a
column-serial radial scan that is deliberately NOT a kernel on this track (a
hand-written scan is a different float32 number and this family's whole claim is
bytewise identity). A fused H->D on those rows is the two certified singles with a
host-side array pass between them and ZERO launches saved.

=============================================================================
WELDED 2026-09-06, AND WHAT THE WELD COVERS
=============================================================================

:data:`WELD_OWED` is empty: the gate released. It had run to completion four times and
refused each time on one leg, ``lift/every_corpus_row_in_the_cell``, and what settled it
was three separate findings about the LEG rather than one about this kernel -- no leg
ever found the fused launch computing anything the two certified singles do not.

* **A flushed INTERMEDIATE is the same platform fact as a flushed stored word, and the
  precondition was only looking at the stored ones.** On ``examples:gaussian-beam.py``
  and ``tests:TestEigenmodeSource.test_waveguide_flux`` every dispatched arrangement --
  the certified singles included -- was byte-identical to every other and all of them
  disagreed with the NumPy array path by 1 to 2 units in the last place at magnitude
  ~1e-32. Bisected on ``gaussian-beam.py``: ``step_D`` alone reproduces it and
  ``update_H`` alone is clean, and at cell (52, 101, 0) and its mirror (648, 101, 0) the
  curl's own operand difference ``shifted_first - first`` is ``-6.64085431592474e-39``,
  word ``0x80485000``, a float32 SUBNORMAL. Flushing that one intermediate to zero and
  running the rest of the certified float32 chain in the shipped order reproduces the
  device's words bit for bit -- ``0x8a8014f5`` -> ``0x8a8014f7`` in ``Dz`` and
  ``fu_Dz``. Neither of the gate's two standing instruments could see it: a stored-state
  census sees a subnormal only once one is written, and IEEE signals underflow only when
  a result is tiny AND INEXACT, while the difference of two nearby normals into the
  subnormal range is exact. The gate's ``IntermediateCensus`` now censuses the
  arithmetic as well as the state, and on six rows measured it fires on the same step
  the stored census does on five of them and one step earlier on the sixth.
* **A row whose source waveform is not a function of time cannot carry a comparison
  ACROSS arrangements.** ``examples:stochastic_emitter.py:80`` drives ten dipoles from
  ``mp.CustomSource(src_func=lambda t: np.random.randn())``, so each of the five
  arrangements consumes draws the one before it did not. The gate measures that with
  the engine's own accessor rather than by name, and refuses the row as undefined by
  construction -- inside the cell's denominator, never as a divergence.
* **Private scratch is not state**, and the exclusion twelve sibling probes already make
  is now made here too, with the premise driven rather than asserted.

What released it is ``parity/meep_gpu/results/metal_fused_hd_pair_2026-09-06_hdland``:
every leg PASS, the cell's 49 rows measured -- 47 admitted, 46 driven and byte-identical
over every step their own precondition held for, 1 refused as undefined by construction
and 2 refused BY NAME for a standing in-seam electric withdraw -- beside the synthetic
fixture's eight boundary specialisations at the full step budget against four
references, 21 shader mutations, four host mutations, a byte-neutral control and a
disarmed shipped-bytes leg.

WHAT THE WELD DOES NOT SAY. It does not say this product runs anywhere, and the four
facts a reader needs are stated here in the words a search will find:

* the board's 47 credited instances are **served by predicate admission** -- the
  board's own definition of *served* is that the predicate admits the row, nothing
  more;
* the product **installs on zero rows** by measured arbitration
  (:data:`INSTALLABLE_REASON`: 0 rows on which displacing the released B->H pair is a
  gain, so the composer refuses it by name on every configuration);
* it therefore **executes nowhere** -- no corpus row, no fixture, no user run reaches
  this kernel outside its own gate and tests;
* the weld **licenses no timing claim and no dispatch claim**: no timing exists for
  this shape, launch counts are not time, and the Metal backend is not planned
  against by the fast path at all.

It also does not license the ``(folded -> folded)`` cell, cylindrical storage, or a
keep-resolved run.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import offdiag_weld_common as _weld
from . import shaders
from .coverage import constitutive_coverage, pml_curl_coverage
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and there is only one thing there to carry: the
#: electric withdraw (:3313-3314), which :data:`HOISTS_THE_WITHDRAW` declines in this
#: round. It is not in this tuple because the product does not perform it.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, and the module that
#: owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather than as a
#: literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult (driver.py:3311) and the ``step_D`` consult (:3315), so there is no deposit
#: in this seam to carry across the launch. The electric injection is one seam later
#: (:3317-3322) and the magnetic one is one seam earlier (:3283-3284); this predicate
#: therefore does NOT consult ``deposit_repair.seam_source_reasons`` at all, and a
#: source of either polarity is not this seam's business.
CARRIES_DEPOSIT_REPAIR = False

#: Does this product perform the seam's electric withdraw before its launch? NO in this
#: round. See the module docstring: the only wiring that performs it is
#: ``launch._install_fused_pair``'s :data:`..withdraw_hoist.SEAM` branch, which this
#: product never reaches (:data:`INSTALLABLE` False), so True here would be a claim
#: about wiring that cannot fire. The predicate consequently
#: refuses every row with a standing in-seam withdraw BY NAME -- 2 of the cell's 49.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration, for a MEASURED
#: VERDICT rather than a policy. ``launch._declared_uninstallable`` reads this and
#: reports :data:`INSTALLABLE_REASON` on every run. It is belt and braces: the
#: composition that keeps the released B->H pair in its slot is
#: ``launch._neighbouring_seam_claimant``'s end-edge guard plus ``_pair_may_absorb``,
#: and neither reads this flag.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "measured 2026-09-04 through the Metal composer over the 179 buildable rows of "
    "parity/meep_gpu/results/h_to_d_seam_2026-09-04 (194-row basis), at composer "
    "level: over the driver's step_B - update_H - step_D - update_E slot path launches "
    "are 4 - (installed pairs), and this product takes one slot from EACH neighbouring "
    "seam. On 133 rows both neighbours install and installing it is a LOSS (2 pairs -> "
    "1, one more launch and one fewer seam served); on 22 rows only the B->H pair "
    "serves and it is a TIE (3 launches / 1 seam either way), which the released "
    "precedent resolves for the incumbent -- the B->H pair installs first and holds "
    "update_H, so _pair_may_absorb refuses this product by name; on 22 rows only the "
    "D->E pair serves and the composer's later half refuses this product naming that "
    "claimant; the 2 remaining rows are the nonlinear cell (3rd-harm-1d), where "
    "neither neighbour installs and an H->D product would be a gain of one seam and "
    "one launch -- outside this product's (ordinary -> PML) arm, reached only through "
    "its FUSED_PAIR_EXTRA_ARMS row, and not yet driven by a device gate. 0 rows on "
    "which displacing the released B->H pair is a gain. This is slot arbitration, not "
    "a verdict about the arithmetic, which the family's own gate certifies on its "
    "synthetic fixture AND over the corpus cell "
    "(results/metal_fused_hd_pair_2026-09-06_hdland, released). What that leaves: the "
    "board's credited instances are served by predicate admission only; the product "
    "installs on zero rows by this measured arbitration; it executes nowhere outside "
    "its own gate and tests; and the weld licenses no timing claim and no dispatch "
    "claim")

#: WELDED, AND THIS IS THE DECLARATION THAT SAYS SO. Empty means "this family holds a
#: released device gate"; non-empty means the gate has run and REFUSED, and the string is
#: what a release still owes. ``test_metal_weld_contract.test_every_metal_family_is_welded``
#: partitions the ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires
#: the two sets to be disjoint and to exhaust it, so this declaration is the only way a
#: gated family may stand unwelded -- exactly the shape the CUDA track uses for a kernel
#: whose gate is still owed (``cuda_kernels`` ``UNCERTIFIED_KERNELS`` and its partition
#: test). It is EMPTY from 2026-09-06: the gate released
#: (``parity/meep_gpu/results/metal_fused_hd_pair_2026-09-06_hdland``) and
#: ``fingerprints.json`` carries ``metal_fused_hd_pair_device_gate``, which is the
#: partition this constant is the complement of. Emptying it without the weld entry, or
#: adding the weld entry without emptying it, fails that test by name.
WELD_OWED: str = ""

#: How many bindings the shipped shape needs, and it is the platform ceiling EXACTLY.
#: There is no headroom, so the gate pins this as an equality: 31 must COMPILE and 32
#: must FAIL.
PACKED_BINDINGS = 31

#: The same signature with the five scalars bound SEPARATELY rather than packed --
#: 30 pointers + 5 scalars. Over the ceiling by four; the gate compiles it and requires
#: the failure.
SEPARATE_SCALAR_BINDINGS = 35

#: The same signature with the three shared ``kms`` volumes bound TWICE, once for each
#: half -- 33 pointers + ``Params``. Over the ceiling by three, which is why the sharing
#: is load-bearing rather than tidy.
UNSHARED_KMS_BINDINGS = 34

#: The shipped signature plus ONE more pointer. Over the ceiling by exactly one, which
#: is what turns :data:`PACKED_BINDINGS` from an inequality into an equality.
ONE_MORE_POINTER_BINDINGS = 32

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_LIFT_EDITS", "FAMILY",
    "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_CELL_POINTERS", "INSTALLABLE",
    "INSTALLABLE_REASON", "ONE_MORE_POINTER_BINDINGS", "OWN_LOAD_EDITS",
    "PACKED_BINDINGS", "REPLACES", "ROTATED_NAMES", "SEAM",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNSHARED_KMS_BINDINGS", "WELD_OWED",
    "certified_constitutive_tail", "compile_fused_hd_pair", "fused_hd_pair_source",
    "h_cell_function", "metal_fused_hd_pair_coverage", "offset_coordinates",
    "plan_metal_fused_hd_pair", "refuted_one_more_pointer_source",
    "refuted_separate_scalar_source", "refuted_unshared_kms_source", "register_arms",
    "shipped_signature_bindings", "welded_curl_tail",
]


# ---------------------------------------------------------------------------
# The lift: the certified constitutive body as a function of an arbitrary cell
# ---------------------------------------------------------------------------

#: Where both certified bodies end their index decode. ONE anchor serves both, which is
#: what lets the curl's prologue stand as the fused kernel's and the constitutive's be
#: dropped (it would redeclare ``ii``/``i``/``j``/``k``).
DECODE_END = _weld.DECODE_END

#: The marker that separates a certified shader's signature from its body. Both
#: templates end their parameter list with this exact line, so ONE anchor lifts either
#: body and a template that stopped carrying it raises rather than splicing a
#: truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: ``h_cell``'s pointer parameters, under the CERTIFIED CONSTITUTIVE BODY'S OWN NAMES
#: wherever the lifted text indexes them (``kp0``/``kp1``/``kp2``) and under the fused
#: kernel's names for the volumes the lift renames. That pairing is what lets the
#: lifted arithmetic stand character for character.
H_CELL_POINTERS: Tuple[Tuple[str, str], ...] = tuple(
    ("device const float*", name) for name in
    ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2",
     "kp0", "kp1", "kp2", "kmx", "kmy", "kmz"))

#: What one call site forwards after the three coordinates. Built from the table above
#: so the declaration and every call cannot drift apart.
H_CELL_TAIL_ARGS = ", ".join(("nxi", "nyi", "nzi")
                             + tuple(name for _kind, name in H_CELL_POINTERS))

#: Every line of certified CONSTITUTIVE text this weld does not lift verbatim, with the
#: reason. DATA, not prose, so the host suite and the gate can assert the list and a
#: line that stopped matching RAISES instead of silently leaving a stale read standing.
#:
#: THERE IS NO ARITHMETIC IN THIS TABLE. Every entry is a pointer spelling or a store
#: that moves to the caller; not one right-hand side, operand order or paren is touched.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float kp_0 = kp0[i], km_0 = km0[i];",
     "became": "    float kp_0 = kp0[i], km_0 = kmx[i];",
     "why": "ONE POINTER RENAME, NO ARITHMETIC. Both halves sit on the same Yee "
            "sub-lattice (SUB_STEPS['step_D']['suffix'] == '' and "
            "CONSTITUTIVE_SIDES['H']['half_integer'] is False), so the curl's kms_x "
            "and the constitutive's km0 ARE THE SAME VOLUME and the fused signature "
            "binds it once. Binding the half-integer set instead compiles and is a "
            "half-cell error in the absorber profile."},
    {"line": "    float prevN = wN[ii];",
     "became": "    float prevN = wiN[ii];",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which nothing "
            "in this launch writes. On this side f_w_H_new[ii] == B[ii] exactly, so an "
            "in-place write would hand a racing neighbour B where it needs B_prev."},
    {"line": "    float srcN = gN[ii];",
     "became": "    float srcN = bN[ii];",
     "why": "pointer spelling only: fused, `gN` in the curl half is the MAGNETIC field, "
            "so the constitutive source keeps its own name. The value is the same "
            "flux-density read the certified body makes."},
    {"line": "    wN[ii] = srcN;",
     "became": "(removed -- the caller stores own.srcN to the SCRATCH volume)",
     "why": "the weld writes a scratch volume, so the helper must not store. The value "
            "and the expression that produced it are untouched; only the destination "
            "moves, and only for the thread's OWN cell (a foreign recompute stores "
            "nothing at all)."},
    {"line": "    float aN = fN[ii];",
     "became": "    float aN = hiN[ii];",
     "why": "the accumulator is read from the PRE-LAUNCH H, which is const for the "
            "whole dispatch. This is what makes the foreign recompute a pure function "
            "of unwritten memory."},
    {"line": "    fN[ii] = aN;",
     "became": "(removed -- the value is RETURNED in the struct)",
     "why": "same as the split-field store: the destination moves to the caller, which "
            "addresses the scratch for its own cell and consumes the register for the "
            "curl half. The two accumulations, their order and their parens are "
            "untouched."},
    {"line": "    int k   = ii % nzi; ... int i   = plane / nyi;",
     "became": "(removed -- i, j and k are parameters; ii is composed from them)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a function "
            "evaluated at an ARBITRARY cell rather than at the thread's own, so the "
            "cell arrives as three coordinates and the flat index is COMPOSED by the "
            "same layout the decode inverts. The n_elem guard leaves with the thread "
            "index: every caller is either the kernel's own guarded thread or a tap "
            "the certified ghost rule already proved valid."},
)


def certified_constitutive_tail(contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``update_H`` constitutive body BELOW the decode prologue, stores removed.

    Not a transcription: this is :func:`.shaders.constitutive_source`'s own output for
    side ``H`` with the preamble, the decode prologue and the closing brace cut, and
    with exactly the substitutions :data:`CONSTITUTIVE_LIFT_EDITS` names. Each is
    applied through :func:`.offdiag_weld_common.needle`, so a spelling that drifted
    raises here rather than emitting a kernel that reads the wrong volume.
    """
    source = shaders.constitutive_source("H", contract)
    edits: List[Tuple[str, str]] = [
        ("    float kp_0 = kp0[i], km_0 = km0[i];\n",
         "    float kp_0 = kp0[i], km_0 = kmx[i];\n"),
        ("    float kp_1 = kp1[j], km_1 = km1[j];\n",
         "    float kp_1 = kp1[j], km_1 = kmy[j];\n"),
        ("    float kp_2 = kp2[k], km_2 = km2[k];\n",
         "    float kp_2 = kp2[k], km_2 = kmz[k];\n"),
    ]
    for target in range(3):
        edits.extend((
            (f"    float prev{target} = w{target}[ii];\n",
             f"    float prev{target} = wi{target}[ii];\n"),
            (f"    float src{target} = g{target}[ii];\n",
             f"    float src{target} = b{target}[ii];\n"),
            (f"    w{target}[ii] = src{target};\n", ""),
            (f"    float a{target} = f{target}[ii];\n",
             f"    float a{target} = hi{target}[ii];\n"),
            (f"    f{target}[ii] = a{target};\n", ""),
        ))
    tail = _weld.lift_curl_tail(source, store_edits=tuple(edits),
                                decode_end=DECODE_END)
    # THE CHECK THAT MATTERS. In the fused signature `f*`, `w*` and `g*` are the
    # DISPLACEMENT, its split-field auxiliary and the magnetic field; one missed
    # rename is a smooth, plausible, entirely wrong answer rather than a compile
    # error, because those names all exist in the enclosing kernel.
    for stem in ("f", "w", "g"):
        for target in range(3):
            if f"{stem}{target}[" in tail:
                raise AssertionError(
                    f"the lifted update_H body still indexes {stem}{target}, which in "
                    f"the fused kernel is not the volume the certified body meant")
    return tail


def h_cell_function(contract: str = shaders.CONTRACT_OFF) -> str:
    """``update_H`` at an arbitrary cell, as one inline function.

    ONE code path serves the thread's own cell and every foreign recompute, so the two
    cannot disagree. Emitted through
    :func:`.offdiag_weld_common.step_cell_function` -- the same helper the four D->E
    scratch welds use -- so the index composition ``ii = i*nyz + j*nzi + k`` is written
    once for the whole package.
    """
    return _weld.step_cell_function(
        "h_cell", certified_constitutive_tail(contract), value_type="float",
        pointer_parameters=H_CELL_POINTERS, scalar_parameters=(),
        returns=("a0", "a1", "a2", "src0", "src1", "src2"))


# ---------------------------------------------------------------------------
# The lift: the certified step_D curl with every H read redirected
# ---------------------------------------------------------------------------

#: The curl half's three OWN-cell magnetic loads and the register each becomes. The
#: thread has already computed its own ``H`` into ``own``, and that register is the
#: post-``update_H`` value the array path would have loaded.
OWN_LOAD_EDITS: Tuple[Tuple[str, str], ...] = tuple(
    (f"    float {var}   = g{target}[ii];\n", f"    float {var}   = own.a{target};\n")
    for target, var in enumerate(("a", "b", "c")))

#: The curl half's six SHIFTED magnetic loads, as ``(register, component)``. WHICH CELL
#: each one reads is NOT written here: it is parsed out of the certified body's own
#: ``int ox = si * nyz + j * nzi + k;`` lines by :func:`offset_coordinates`, so the
#: recompute lands on exactly the cell the emitter's index composed and a change to how
#: that index is spelled RAISES instead of quietly redirecting a tap.
#:
#: THE BRANCH AND THE SHIFT ARITHMETIC ARE NOT TOUCHED. Only the leaf load moves, and
#: the validity flag (``vx``/``vy``/``vz``) is PARSED OUT of the certified line rather
#: than retyped, so a metallic ghost that read an exact ``0.0f`` still reads an exact
#: ``0.0f`` and the periodic wrap is still the emitter's own integer expression. The
#: ternary is also what keeps an out-of-range recompute from being evaluated at all,
#: which is the same C guarantee the certified load already relies on for ``g0[oy]``
#: with a negative ``oy``.
HALO_TAPS: Tuple[Tuple[str, int], ...] = (
    ("a_y", 0), ("a_z", 0), ("b_x", 1), ("b_z", 1), ("c_x", 2), ("c_y", 2))


def offset_coordinates(tail: str) -> Dict[str, Tuple[str, str, str]]:
    """``{offset name: (i, j, k) expressions}``, read off the certified index lines.

    The emitter writes one line per shifted axis, each of the shape
    ``int ox = si * nyz + j * nzi + k;`` -- the SAME layout
    :func:`.offdiag_weld_common.step_cell_function` composes ``ii`` from. Parsing it
    rather than tabulating it is what makes "the index expressions are the emitter's"
    a property of the construction.
    """
    out: Dict[str, Tuple[str, str, str]] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped.startswith("int o") or " = " not in stripped:
            continue
        name, expression = stripped[len("int "):].rstrip(";").split(" = ", 1)
        parts = [piece.strip() for piece in expression.split(" + ")]
        if len(parts) != 3 or not parts[0].endswith(" * nyz") \
                or not parts[1].endswith(" * nzi"):
            raise AssertionError(
                f"the certified curl composes {name} as {expression!r}, which is not "
                f"the `a * nyz + b * nzi + c` layout this weld recomposes from "
                f"coordinates")
        out[name] = (parts[0][: -len(" * nyz")], parts[1][: -len(" * nzi")], parts[2])
    if set(out) != {"ox", "oy", "oz"}:
        raise AssertionError(
            f"the certified curl composes {sorted(out)} rather than the three shifted "
            f"offsets this weld redirects")
    return out


def welded_curl_tail(codes: Sequence[int],
                     contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``step_D`` curl body below the decode prologue, every H read redirected.

    ``codes`` is the per-axis PERIODIC/METALLIC triple; ``backward`` is read from
    :data:`.launch.SUB_STEPS` rather than written as a literal, so this family and the
    shipped table cannot drift.
    """
    source = shaders.curl_source(codes, bool(SUB_STEPS["step_D"]["backward"]),
                                 contract)
    if DECODE_END not in source:
        raise AssertionError(
            "the certified curl source no longer carries the decode anchor; the fused "
            "kernel would splice a truncated body")
    tail = source.split(DECODE_END, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError("the certified curl source does not end with '}'")
    tail = tail[: -len("}\n")]
    offsets = offset_coordinates(tail)
    for old, new in OWN_LOAD_EDITS:
        tail = _weld.needle(tail, old, new)
    for var, target in HALO_TAPS:
        prefix = f"    float {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the certified curl declares {var} {len(matches)} times; this weld "
                f"redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE CERTIFIED LINE, never
        # retyped: everything left of " ? " is the emitter's own validity expression,
        # and the offset name inside the brackets is what selects the coordinates.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(" : 0.0f;\n"):
            raise AssertionError(
                f"the certified curl no longer reads {var} as `{expected}...]` served "
                f"an exact 0.0f past the wall; the ghost rule this weld preserves has "
                f"changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = f"h_cell({', '.join(coordinates)}, {H_CELL_TAIL_ARGS}).a{target}"
        tail = _weld.needle(tail, old, f"{head} ? {call} : 0.0f;\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded curl half still reads g{target}; in this signature that "
                f"pointer does not exist and every magnetic read must be the register "
                f"or a recompute")
    return tail


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: The rotating volumes, in SIGNATURE ORDER: the three stored magnetic fields then the
#: three split-field histories. :meth:`.offdiag_weld_common.ScratchWeldPairPlan.run`
#: binds every SCRATCH first and every PRE-LAUNCH buffer second, in this order, then
#: the static arguments -- so this order IS the signature's and a kernel that
#: interleaved a static pointer between two rotating ones would bind the wrong buffer
#: to every argument after it.
ROTATED_NAMES: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };

__H_CELL__

kernel void fused_hd_pair_step(
    device float*       ho0     [[buffer(0)]],
    device float*       ho1     [[buffer(1)]],
    device float*       ho2     [[buffer(2)]],
    device float*       wo0     [[buffer(3)]],
    device float*       wo1     [[buffer(4)]],
    device float*       wo2     [[buffer(5)]],
    device const float* hi0     [[buffer(6)]],
    device const float* hi1     [[buffer(7)]],
    device const float* hi2     [[buffer(8)]],
    device const float* wi0     [[buffer(9)]],
    device const float* wi1     [[buffer(10)]],
    device const float* wi2     [[buffer(11)]],
    device const float* b0      [[buffer(12)]],
    device const float* b1      [[buffer(13)]],
    device const float* b2      [[buffer(14)]],
    device float*       f0      [[buffer(15)]],
    device float*       f1      [[buffer(16)]],
    device float*       f2      [[buffer(17)]],
    device float*       u0      [[buffer(18)]],
    device float*       u1      [[buffer(19)]],
    device float*       u2      [[buffer(20)]],
    device const float* kmx     [[buffer(21)]],
    device const float* sinvx   [[buffer(22)]],
    device const float* kmy     [[buffer(23)]],
    device const float* sinvy   [[buffer(24)]],
    device const float* kmz     [[buffer(25)]],
    device const float* sinvz   [[buffer(26)]],
    device const float* kp0     [[buffer(27)]],
    device const float* kp1     [[buffer(28)]],
    device const float* kp2     [[buffer(29)]],
    constant Params&    prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIVE SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any lifted text runs. Everything below this line is then
    // character-for-character what `shaders.curl_source` and
    // `shaders.constitutive_source` emit, with the declared lift edits and the
    // redirected magnetic reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
__PROLOGUE__
    // --- THE WELD: update_H, computed into registers and stored to SCRATCH ------
    // Nothing written here is read by this launch. The curl half below takes its
    // OWN cell's magnetic field from these registers and recomputes every foreign
    // tap from PRE-LAUNCH state through the same `h_cell`, so no thread observes
    // another thread's store; the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- step_D (stepping.step_D / _apply_pml_update:1905) ----------------------
    // D and fu_D update IN PLACE and that is safe by construction: the curl reads
    // and writes them at the THREAD'S OWN CELL only (`f0[ii]`, `u0[ii]`), so no
    // thread reads a displacement another thread wrote.
__CURL__
}
"""


def fused_hd_pair_source(codes: Sequence[int],
                         contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, contraction mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.

    THE WALL CLEAR IS NOT CARRIED and must not be. ``zero_metal_B`` runs one seam
    EARLIER (driver.py:3306, before the ``update_H`` consult) and ``zero_metal_D`` one
    seam LATER (:3324, after ``step_D``); neither is inside this seam, so a mask here
    would be a pass the driver runs again.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")

    curl = shaders.curl_source(codes, bool(SUB_STEPS["step_D"]["backward"]), contract)
    prologue = curl.split(_BODY_ANCHOR, 1)[1].split(DECODE_END, 1)[0] + DECODE_END
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__H_CELL__": h_cell_function(contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": H_CELL_TAIL_ARGS,
        "__CURL__": welded_curl_tail(codes, contract),
    })


def compile_fused_hd_pair(codes: Sequence[int],
                          contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, mode)."""
    return compile_source(fused_hd_pair_source(codes, contract)).fused_hd_pair_step


# ---------------------------------------------------------------------------
# The three refuted signatures — measurements, not arguments
# ---------------------------------------------------------------------------

_WRITTEN: Tuple[str, ...] = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2",
                             "f0", "f1", "f2", "u0", "u1", "u2")
_READ: Tuple[str, ...] = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2",
                          "b0", "b1", "b2",
                          "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                          "kp0", "kp1", "kp2")


def _touch_kernel(name: str, written: Sequence[str], read: Sequence[str],
                  contract: str, packed: bool = True) -> Tuple[int, str]:
    """A signature with a body that touches every buffer, and nothing else.

    What is being measured is the SIGNATURE; a body the compiler could drop would let
    dead-code elimination decide the answer, which is why every buffer is read and
    every written one is stored.

    ``packed`` binds the five scalars as ONE ``constant Params&``, exactly as the
    shipped kernel does, so the returned binding count is the number this family
    argues from rather than a number one binding-model away from it.
    """
    lines: List[str] = []
    slot = 0
    for label in written:
        lines.append(f"    device float*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in read:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    if packed:
        lines.append(f"    constant Params&    prm     [[buffer({slot})]],")
        slot += 1
        unpack = ["    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, "
                  "n_elem = prm.n_elem;",
                  "    float dtdx = prm.dtdx;"]
    else:
        for kind, label in (("constant uint&", "nx"), ("constant uint&", "ny"),
                            ("constant uint&", "nz"), ("constant uint&", "n_elem"),
                            ("constant float&", "dtdx")):
            lines.append(f"    {kind:<19} {label:<8}[[buffer({slot})]],")
            slot += 1
        unpack = []
    touch = " + ".join(f"{label}[0]" for label in read)
    return slot, "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };",
        "",
        f"kernel void {name}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *unpack,
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        f"    {written[0]}[idx] = {written[0]}[idx] + touch * float(nx + ny + nz);",
        *[f"    {label}[idx] = {label}[idx] * dtdx;" for label in written[1:]],
        "}",
        "",
    ))


def shipped_signature_bindings() -> int:
    """How many bindings the SHIPPED kernel declares, counted off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify.
    """
    source = fused_hd_pair_source((0, 0, 0))
    signature = source.split("kernel void fused_hd_pair_step(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 35-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and require
    the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an argument
    into a measurement.
    """
    slots, source = _touch_kernel("refuted_separate_scalar", _WRITTEN, _READ,
                                  contract, packed=False)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


def refuted_unshared_kms_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding signature that binds the three ``kms`` volumes TWICE.

    The measurement that makes the shared coefficient group a NECESSITY rather than a
    tidiness: with the constitutive half given its own ``km0``/``km1``/``km2`` the
    signature is 33 pointers plus one packed ``Params`` and does not compile at all.
    """
    slots, source = _touch_kernel("refuted_unshared_kms", _WRITTEN,
                                  _READ + ("km0", "km1", "km2"), contract)
    assert slots == UNSHARED_KMS_BINDINGS, (slots, UNSHARED_KMS_BINDINGS)
    return source


def refuted_one_more_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The shipped signature plus ONE pointer -- 32 bindings, over the ceiling by one.

    A family with no headroom has to pin the ceiling as an EQUALITY, so the gate
    compiles this and requires the failure beside the shipped 31 that must succeed.
    """
    slots, source = _touch_kernel("refuted_one_more_pointer", _WRITTEN,
                                  _READ + ("one_too_many",), contract)
    assert slots == ONE_MORE_POINTER_BINDINGS, (slots, ONE_MORE_POINTER_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                 residency: Any = None) -> Coverage:
    """May ONE dispatch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first -- and on this seam that matters, since the null
    ``update_H`` (an inactive absorber returns before its first statement) is the
    single largest non-fusion reason on the board and belongs to the constitutive side.
    """
    reasons: List[str] = []

    # THE NONLINEAR WIDENING, 2026-09-05, ON BOTH HALVES. A chi2/chi3 run is refused by
    # both ordinary predicates (coverage.py clause 10) and served on these two slots by
    # `nonlinear_update_e`'s spine arms, which build the SAME two kernels under a scope
    # view -- byte-identical text, measured; the Pade factor lives in update_E, one
    # seam later. So each half is admitted by EITHER its ordinary predicate OR the
    # spine arm's, and the two are disjoint on `has_nonlinearity`, so exactly one can
    # answer. A double refusal reports both, prefixed, the way the CUDA precedent
    # (`cuda_kernels/fused_magnetic_pair.py`'s nonlinear H widening) reports it. The
    # import is lazy for the reason every family's import of `launch` is: the
    # registry owns import order, and a module-scope import here would register the
    # nonlinear arms from inside this module's import.
    from . import nonlinear_update_e as _nonlinear  # noqa: PLC0415

    magnetic = constitutive_coverage(fields, pml, "H", residency)
    if not magnetic.covered:
        spine = _nonlinear.nonlinear_run_constitutive_coverage(fields, pml, "H",
                                                               residency)
        if not spine.covered:
            reasons.extend(f"constitutive half: {reason}"
                           for reason in magnetic.reasons)
            reasons.extend(f"constitutive half, nonlinear-run widening: {reason}"
                           for reason in spine.reasons)
    curl = pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        spine = _nonlinear.nonlinear_run_pml_curl_coverage(fields, pml, "step_D",
                                                           residency)
        if not spine.covered:
            reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
            reasons.extend(f"curl half, nonlinear-run widening: {reason}"
                           for reason in spine.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier
    # (driver.py:3283-3284) and the electric one is one seam later (:3317-3322). What
    # IS between them is the electric integrated-source withdraw (:3313-3314), and
    # `withdraw_hoist` owns it. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold
    # the source list, so a predicate that inferred "no withdraw stands" from not being
    # told would be the over-covering refusal this clause exists to prevent.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, "
            f"so _install_fused_pair's withdraw-hoist branch is unreachable for it "
            f"and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane (coverage.py
    # clause 5), and on THIS seam the reason is not the one the D->E and B->H pairs
    # give: neither fill runs between these two consults (fill_B closes one seam earlier
    # at driver.py:3306-3309 and fill_D opens one seam later at :3324-3327), so the
    # H->D seam is fill-FREE on a folded grid. What refuses a fold here is that this
    # product implements the `ordinary` and `PML` arms and a folded run selects the
    # `folded` arm on both slots -- the 78-row cell, and a second product.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `ordinary` "
                f"update_H and `PML` step_D arms (and their nonlinear-run spine "
                f"twins), and a folded run selects the `folded` arm on both slots; "
                f"the (folded -> folded) cell needs its own product and its own "
                f"ghost-map measurement")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES after each launch, so those
    # attributes must be settable and must be the arrays the residency mirrored.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _params_tensor(shape: Sequence[int], dtdx: float, device: str) -> Any:
    """The five scalars as one 20-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused pair's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4"), ("dtdx", "<f4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives: a
    configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_fused_hd_pair_coverage(fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ROTATING GROUP FIRST, in ROTATED_NAMES order: every scratch, then every
    # pre-launch buffer. `ScratchWeldPairPlan.run` splices exactly those two groups
    # ahead of `static_args`, so this order IS the signature's.
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host)
        twins[name] = twin

    flux = [bind(name, getattr(fields, name))
            for name in SUB_STEPS["step_B"]["targets"]]
    displacement = [bind(name, getattr(fields, name))
                    for name in SUB_STEPS["step_D"]["targets"]]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in SUB_STEPS["step_D"]["targets"]]

    # ONE SUB-LATTICE, TWO HALVES. `SUB_STEPS['step_D']['suffix']` is '' and
    # `CONSTITUTIVE_SIDES['H']['half_integer']` is False, so the curl's kms and the
    # constitutive's kms are THE SAME THREE VOLUMES and are bound once. Both suffixes
    # are READ FROM THE SHIPPED TABLES rather than written as literals, and their
    # equality is ASSERTED rather than assumed: it is the whole premise of the 30-
    # pointer signature, and if either table moved, sharing the group would bind one
    # half's coefficients to the other half's lattice -- a converged, smooth,
    # half-cell-wrong absorber profile rather than a failure.
    suffix = SUB_STEPS["step_D"]["suffix"]
    constitutive_suffix = "_h" if CONSTITUTIVE_SIDES["H"]["half_integer"] else ""
    if suffix != constitutive_suffix:
        raise AssertionError(
            f"step_D reads the {suffix or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive_suffix or 'integer'!r} one; this weld binds ONE kms "
            f"group for both halves and that is only correct while they agree")
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}{suffix}", getattr(pml, f"{stem}_{axis}{suffix}"),
             constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:kps_{axis}{constitutive_suffix}",
             getattr(pml, f"kps_{axis}{constitutive_suffix}"), constant=True)
        for axis in "xyz"]

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_fused_hd_pair(codes, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_params_tensor(grid.shape, dtdx, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS == MAX_BUFFER_BINDINGS, (
        PACKED_BINDINGS, MAX_BUFFER_BINDINGS)
    return _weld.plan_scratch_weld(
        FAMILY, residency, fields, rotated_names=ROTATED_NAMES,
        static_args=static, functions=selected, volumes=volumes,
        shape=grid.shape, codes=codes, zero_metal=(0, 0, 0), row_mask=(),
        replaces=REPLACES, twins=twins)


# ---------------------------------------------------------------------------
# Registration — wired=False, and refused by the composer twice over
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"fused H/D pair cannot fill {slot}",))
    return metal_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips an
    unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it -- which is what lets the composition sweep measure disjointness against
    this predicate instead of assuming it, and what lets
    ``launch._neighbouring_seam_claimant`` see the row when it asks who claims the seam
    ``update_H`` opens.

    THAT LAST READER USED TO BE WHY :data:`INSTALLABLE` WAS LOAD-BEARING FOR A
    RELEASED PRODUCT'S SLOTS, AND IS NOT ANY MORE. ``_neighbouring_seam_claimant``
    walks ``arms.registered("update_H")`` for welds that also replace ``step_D`` when
    an INTERIOR span asks who claims the seam it opens; the B->H span is an END EDGE
    of ``launch.FUSED_PAIR_SEAMS`` by the table's own degrees and is never asked
    (measured 2026-09-04: without that guard, registering this row -- flag or no flag
    -- refused ``fused_magnetic_pair``, a released, gate-passing product, on every
    row it reached, and only the flag stood in the way).

    It holds a ``launch.FUSED_PAIR_ARMS`` row (``("ordinary", "PML")``, read off the
    two predicates above) since 2026-09-05, so the seam loop ASKS it and refuses it on
    the flag; drop the flag and it is still refused by ``_pair_may_absorb`` wherever
    the B->H pair installed first, which is every corpus row its primary arm reaches.
    Two independent brakes still, and the gate's arbitration leg drives both.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="fused H/D pair: ",
                          noun="fused stored-H/PML D-curl pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
