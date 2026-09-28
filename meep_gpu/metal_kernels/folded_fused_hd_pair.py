"""The FOLDED H->D weld: the folded ``update_H`` welded into the folded ``step_D``.

THE SECOND PRODUCT ON THE FOURTH SEAM, and the largest single cell on this backend:
``h_to_d_seam.instances`` files **78** rows under the Metal arm pair
``(folded -> folded)`` -- more than the plain product's 49 -- and 75 of them as
``buildable_not_built`` (the other 3 as ``withdraw_seam``). :mod:`.fused_hd_pair`
refuses every one of them BY NAME, because it implements the ``ordinary`` update_H
and ``PML`` step_D arms and a folded run selects the ``folded`` arm on both slots.
This module is that second product; the two predicates are disjoint on ONE clause,
inverted (see :func:`metal_folded_fused_hd_pair_coverage` clause 4).

=============================================================================
THE SEAM IS FILL-FREE ON A FOLDED GRID, AND THAT IS THE WHOLE REASON THIS IS A
SPLICE RATHER THAN NEW ARITHMETIC
=============================================================================

``FdtdDriver.step`` (driver.py:3187-3339) runs, in this order::

    3291  step_B                        (consult)
    3293  magnetic inject
    3305  fill_B -> fill_symmetry_bc_B  (consult)
    3307  zero_metal_B                  (no consult)
    3309  fill_folded_far_ghosts_B      (consult)
    3311  update_H                      (consult)   <-- the seam opens
    3313  for source in electric: withdraw(...)     <-- the ONLY statement inside
    3315  step_D                        (consult)   <-- the seam closes
    3317  electric inject
    3325  fill_D -> fill_symmetry_bc_D  (consult)
    3327  zero_metal_D                  (no consult)
    3329  fill_folded_far_ghosts_D      (consult)
    3331  update_E

EVERY B-SIDE FILL, THE WALL CLEAR AND THE FAR PASS RUN BEFORE THE ``update_H``
CONSULT; EVERY D-SIDE ONE RUNS AFTER THE ``step_D`` CONSULT. So on a folded grid
this seam carries exactly what the unfolded one carries -- the electric withdraw --
and nothing else. The ``complex_fill_carry`` / ``carried_destinations`` machinery
the D->E and B->H folded pairs need (:mod:`.folded_fused_pair`,
:func:`.offdiag_weld_common.d_final_function`) IS NOT NEEDED HERE and must not be
imported: a fill carried across this launch would be a pass the driver runs again.

=============================================================================
WHAT THE FOLD CHANGES IN EACH HALF
=============================================================================

**The constitutive half: nothing.** ``stepping.update_H`` (stepping.py:907-923) is
three ``_apply_constitutive_pml`` calls (:2112-2143) over integer-position
coefficients, with no shift helper, no ghost and no ownership mask; it reads ``H``,
``f_w_H`` and ``B`` at ONE cell and writes the same cell, over the whole STORED
extent. Metal's folded ``update_H`` arm adds no kernel at all --
``symmetry.plan_folded_constitutive`` (symmetry.py:1662-1702) builds the CERTIFIED
:class:`.launch.ConstitutivePlan` through ``_constitutive_functions``, and "the fold
reaches this sub-step through exactly one thing: the stored extent"
(symmetry.py:1072-1079). So :func:`.fused_hd_pair.h_cell_function` -- the certified
``update_H`` body as a function of ``(i, j, k)``, with the seven declared
:data:`.fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- IS the folded arm's arithmetic,
character for character, and this module imports it rather than owning a copy.

**The curl half: one dead-value branch and one mask block.** The folded curl
(``symmetry._FOLDED_CURL_TEMPLATE``, symmetry.py:333-453) is
``shaders._CURL_TEMPLATE`` with ONE added slot, ``__TOP_MASK__``. Its ghost gather
and its cell-0 mask are emitted by the CERTIFIED emitters ``templates.ghost`` /
``templates.ownership_mask`` over ``symmetry._reduced_codes`` (:456-480), which maps
both mirror codes to METALLIC -- so on a folded axis the backward ghost is
``vN = (sN >= 0) && (sN < nN);`` and the load is ``bN = vN ? gN[oN] : 0.0f;``, an
exact ``0.0f`` past the mirror plane, NOT the array path's ``parity * field[2]``
(``stepping._shift_down``, stepping.py:1870-1873). That substitution is legal
because both ghost values are DEAD IN THE CURL (symmetry.py:30-43): their only
consumer is a plane one of the two ownership masks zeroes.
``symmetry.folded_top_plane_mask`` (:528-550) adds ``curlN = last_a ? 0.0f : curlN;``
for every D target with Yee shift 1 on a MIRROR_PERIODIC axis, and a comment only on
a MIRROR_METALLIC one.

**No cell beyond the three backward neighbours is read, on a fold or off it.**
``step_D`` differences ``H`` one cell DOWN (stepping.py:379-457,
``_curl_operands(..., backward=True)`` at :453-454), and the Metal kernel reads
``g[ii]`` and ``g[ox|oy|oz]`` with ``sN = N - 1`` (shaders.py:143, loads :154-162).
The halo question the plan left open is therefore answered from the source, and the
gate's ``purity`` and ``codes_source`` legs measure it rather than assert it:

* **stored cell 0 of a folded axis IS read** (by the thread at index 1) and it is a
  REAL stored cell whose ``H`` is ``update_H``'s own output over a ``B`` the
  driver's NEAR fill wrote before the seam opened (``fill_symmetry_bc_B`` ->
  ``_fill_symmetry_ghost_cells``, stepping.py:1399-1451, ``MIRROR_SOURCE_INDEX = 2``
  at :160). **There is no H fill anywhere in the engine** -- the fills exist for B
  and D only;
* **the last stored slot of a folded PERIODIC axis IS read** (as its own thread's
  own cell) and its ``H`` likewise comes from ``update_H`` over a ``B`` the FAR fill
  wrote (``fill_folded_far_ghosts_B``, stepping.py:1489-1532);
* **index -1 past a folded face is NOT an H cell at all**: it is the literal
  ``0.0f`` of the METALLIC ternary, which the lift preserves untouched.

So every H cell the folded ``step_D`` reads is an ``update_H`` output; none is a
filled H ghost; every fill-provided value is a ``B`` cell written before the seam
opened and is therefore a CONST input to this launch.

=============================================================================
THE SHAPE, AND THE BINDING CEILING IS THE PLAIN PRODUCT'S EXACTLY
=============================================================================

One dispatch. ``REPLACES = ("update_H", "step_D")``; ``SLOT = "update_H"``;
:class:`.offdiag_weld_common.ScratchWeldPairPlan` with the plain product's own
:data:`.fused_hd_pair.ROTATED_NAMES`.

    3 H_out + 3 f_w_H_out   (scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in    (pre-launch, const)
  + 3 B                     (const)
  + 3 D + 3 fu_D            (in place: the curl reads and writes its OWN cell only)
  + 6 curl coefficients (kms/sinv per axis, ONE shared group)
  + 3 constitutive kps                                                        = 30

plus one ``constant Params&`` = **31 = :data:`.device.MAX_BUFFER_BINDINGS`**, no
headroom. THE FOLD ADDS NO KERNEL ARGUMENT ON THIS BACKEND and there are four
separate reasons, each a fact rather than a preference:

* the mirror codes are a COMPILE-TIME source specialisation
  (``symmetry.folded_curl_source``, symmetry.py:553-580), exactly as the plain
  product bakes PERIODIC/METALLIC;
* **the parity is not needed at all.** This kernel performs no fill and its ghost is
  a literal zero, so the product is PHASE-BLIND: the emitted source for a ``+1``
  plane and a ``-1`` plane is byte-identical. Parity enters only the fill kernels
  (``symmetry.mirror_ghost_fill_source``:687-750), which are outside this seam;
* the reflect row is not needed: the far image is a B fill before the seam and a D
  fill after it, and the top-plane mask uses ``last_a = (a == na - 1)``, a decode
  fact (symmetry.py:428);
* no ghost-source row is needed: the near ghost's source cell is read only by the
  fill, which is not carried.

MEASURED 2026-09-07 on this host, torch MPS: the emitted source is **31 bindings on
all 13 boundary triples tried** -- every triple the corpus cell carries plus
``(P,P,P)``, ``(MM,P,P)``, ``(MP,P,P)``, ``(P,MP,MM)``, ``(MM,MM,M)`` -- and all 13
COMPILE through ``device.compile_source``. On the unfolded triple ``(P,P,P)`` the
emission differs from :func:`.fused_hd_pair.fused_hd_pair_source`'s only in comments
plus one dead line ``bool last_x = ..., last_y = ..., last_z = ...;``.

The three refuted signatures are the plain product's, IMPORTED rather than
re-spelled, because the signature IS the same one: 35 separate scalars, 34 unshared
``kms``, 32 one-more-pointer. The gate compiles each and requires the failure.

=============================================================================
WHERE THE CODES COME FROM, AND WHY THAT IS THE ONE THING THIS MODULE CAN GET
WRONG SILENTLY
=============================================================================

``codes`` MUST come from ``symmetry.folded_axis_kinds(grid, pml)`` and from nowhere
else. The plain product builds its triple as ``1 if kind == "metallic" else 0`` over
``stepping._boundary_kinds`` (fused_hd_pair.py:875-879); on a folded grid
``_boundary_kinds`` reports ``"mirror"`` (stepping.py:2191-2196), which that
expression maps to 0 = PERIODIC -- so the ghost would become a WRAP to the far plane
AND the cell-0 mask would not be emitted. That is a smooth wrong answer on every
folded axis, not a crash, and ``folded_curl_source`` cannot catch it either (0 and 1
are valid codes there). It is therefore a GATE MUTATION (``codes_from_boundary_kinds``)
rather than a comment.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver rather than
a choice: nothing is INJECTED between the two consults (the electric injection is one
seam later at driver.py:3317-3322, the magnetic one one seam earlier at :3283-3284),
so :mod:`..deposit_repair` has nothing to say here. What IS there is the electric
integrated-source withdraw, and :mod:`..withdraw_hoist` owns it.

:data:`HOISTS_THE_WITHDRAW` is False, for the plain product's measured reason
re-stated on this cell: the only wiring that performs the hoist is
``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch (launch.py:1010-1016),
reached only from the seam loop, which this product never reaches (no
``FUSED_PAIR_ARMS`` row, :data:`INSTALLABLE` False). True here would be a claim about
wiring that cannot fire, and a product that declared it without the wrapper would let
the fused launch read a ``D`` still holding the previous step's standing dipole and
report success.

WHAT THAT COSTS, MEASURED rather than estimated
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, joined on the
Metal arms 2026-09-07): the ``(folded -> folded)`` cell is **78 rows**, of which
**75** are ``buildable_not_built`` and **3** are ``withdraw_seam`` --
``examples:absorbed_power_density.py``, ``examples:finite_grating.py`` and
``examples:mie_scattering.py``, each with
``n_electric_withdraws_that_do_work == 1``. This predicate therefore reaches **75 of
the cell's 78**, and what would raise it to 78 is the rest of the same three-part
package the plain module names. One part LANDED on 2026-09-07 -- the
``launch.FUSED_PAIR_ARMS`` row ``("folded", "folded")`` -- and the other two did not:
:data:`INSTALLABLE` True, and a device leg driving a complete step on a row with a
live INTEGRATED electric source with the hoist in place, requiring both
``not_hoisted`` and ``after_step_D`` to diverge.

THE CELL'S SHAPE, measured the same way (the seam census joined against
``results/metal_coverage_2026-09-04_m0complex``, 78 of 78 rows joined). Boundary
triples, where MM is MIRROR_METALLIC and MP is MIRROR_PERIODIC::

    (MM, MM, P)  36 rows   2-D, two folded metallic axes
    (M,  MM, P)  26 rows   2-D, an x wall beside a folded metallic y
    (P,  MP, P)  10 rows   2-D, a folded PERIODIC y  (2 withdraw rows)
    (MP, MP, P)   2 rows   two folded periodic axes  (one 3-D)
    (M,  MM, MM)  2 rows   3-D
    (P,  MP, MP)  1 row    3-D, odd full count       (1 withdraw row)
    (MM, MM, MM)  1 row    3-D, three folded metallic axes

73 rows are 2-D and 5 are 3-D; 28 carry a live NON-folded metallic wall; 13 carry a
MIRROR_PERIODIC axis and therefore a live far pass and a top-plane mask. EVERY row is
buildable by the same kernel shape -- the partition is by fixture, not by
weldability -- and all 3 withdraw rows sit in the MP partition.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS A MEASURED VERDICT
=============================================================================

:data:`INSTALLABLE` is False. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``
(``launch._neighbouring_seam_claimant``, launch.py:1085-1190), and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour. Joined against the Metal
board (``results/fusion_matrix_metal_2026-09-03_complete``): on **78 of 78** rows of
this cell the ``B_to_H`` seam is already SERVED, by the released
``folded_fused_magnetic_pair``, on every single one; ``D_to_E`` is served on 67 and
unserved on 11. So this product TIES on 11 rows and LOSES on 67, and on ZERO rows is
displacing the released B->H pair a gain. The released precedent resolves a tie for
the incumbent: the B->H pair installs first and holds ``update_H``, so
``_pair_may_absorb`` refuses this product by name. Belt and braces:
``launch._declared_uninstallable`` (launch.py:1053-1082) reads this flag on every run
and refuses on it BEFORE the gate or the predicate is reached, so the
``FUSED_PAIR_ARMS`` row this family gained on 2026-09-07 lets the seam loop ASK the
product without letting it install: the flag stops it first, and with the flag out of
the way ``_pair_may_absorb`` stops it again on the incumbent.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY, AND WHAT IT DOES NOT CLAIM
=============================================================================

ONE ARM PAIR ONLY: ``(folded -> folded)``, REAL float32 storage. Complex storage
under a fold is the folded-complex family's; ``beta`` under a fold is
``folded_beta``'s; a registered susceptibility or an off-diagonal chi1inv row on
``update_E`` are one seam later and do not reach this pair, but an off-diagonal row
and a conductivity on the D targets ARE refused here through the curl half's own
certified predicate. Cylindrical is refused TWICE, by both halves, and permanently:
``stepping._boundary_kinds`` puts ``is_axis`` AHEAD of ``is_mirrored``
(stepping.py:2191-2196), so a folded r axis would report CYL_AXIS and the fold would
vanish silently, and ``_mirror_phases`` (:2346-2366) puts ``(-1)**grid.m`` into the
same slot the mirror phase occupies.

NO TIMING EXISTS FOR THIS SHAPE AND NONE IS LICENSED BY ANYTHING IN THIS MODULE. The
fused route does MORE memory traffic than the two singles at the same step-level
launch count -- three extra pointwise constitutive evaluations per component per
thread -- and whether the saved launch and the saved ``H`` write-then-read round trip
pay for that is a hypothesis nobody has measured.

WHAT THE FAMILY'S CREDITED INSTANCES MEAN, stated in the words a search will find:
the board's 75 instances are **served by predicate admission** and nothing more; the
product **installs on zero rows** by the measured arbitration above; it therefore
**executes nowhere** outside its own gate and tests; and it licenses **no timing
claim and no dispatch claim**.

=============================================================================
REGISTERED AND WELDED, 2026-09-07
=============================================================================

This module IS in ``registry.FAMILY_MODULES`` and ``launch.FUSED_PAIR_ARMS`` carries
its absorb row, so a normal composition imports it and the seam loop can ASK it --
and then refuses it on :data:`INSTALLABLE`, which is where the arbitration above is
enforced. Its arm is registered ``wired=False``, so ``plan_step`` cannot select it
either. :data:`WELD_OWED` is EMPTY because ``fingerprints.json`` now carries
``metal_folded_fused_hd_pair_device_gate``, minted from the released artifact by
``mint_metal_weld.py``; the partition
``test_metal_weld_contract.test_every_metal_family_is_welded`` is what makes the
empty string a claim rather than an omission.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import fused_hd_pair as _plain
from . import offdiag_weld_common as _weld
from . import shaders, symmetry
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "folded_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and on a folded grid there is still only one
#: thing there to carry: the electric withdraw (:3313-3314), which
#: :data:`HOISTS_THE_WITHDRAW` declines in this round. Both mirror fills, the wall
#: clear and both far passes sit OUTSIDE this span -- three before ``update_H`` and
#: three after ``step_D`` -- which is why this weld carries no fill and must not.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, and the module
#: that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather than as
#: a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult (driver.py:3311) and the ``step_D`` consult (:3315).
CARRIES_DEPOSIT_REPAIR = False

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round -- see the module docstring. The predicate consequently refuses every
#: row with a standing in-seam withdraw BY NAME: 3 of the cell's 78.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration, for a MEASURED
#: VERDICT rather than a policy. ``launch._declared_uninstallable`` reads this and
#: reports :data:`INSTALLABLE_REASON` on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "measured 2026-09-07 by joining the 78 rows of the (folded -> folded) cell in "
    "parity/meep_gpu/results/h_to_d_seam_2026-09-04 against the Metal board "
    "results/fusion_matrix_metal_2026-09-03_complete: over the driver's step_B - "
    "update_H - step_D - update_E slot path launches are 4 - (installed pairs), and "
    "this product takes one slot from EACH neighbouring seam. The B_to_H seam is "
    "already SERVED on 78 of 78 rows of this cell, by the released "
    "folded_fused_magnetic_pair on every one of them; D_to_E is served on 67 (54 "
    "folded_fused_pair, 9 folded_offdiag_fused_electric_pair, 4 "
    "folded_fused_dispersive_pair) and unserved on 11. So on 67 rows installing this "
    "product is a LOSS (2 pairs -> 1: one more launch and one fewer seam served) and "
    "on 11 it is a TIE (3 launches / 1 seam either way), which the released "
    "precedent resolves for the incumbent -- the B->H pair installs first and holds "
    "update_H, so _pair_may_absorb refuses this product by name. On ZERO rows is "
    "there a configuration in which neither neighbour installs, so there is no row "
    "on which displacing the released B->H pair is a gain. This is slot arbitration, "
    "not a verdict about the arithmetic, which this family's own gate certifies. "
    "What that leaves: the board's credited instances are served by predicate "
    "admission only; the product installs on zero rows by this measured arbitration; "
    "it executes nowhere outside its own gate and tests; and it licenses no timing "
    "claim and no dispatch claim")

#: WHAT A RELEASE STILL OWES. Empty means "this family holds a released device gate";
#: non-empty means it does not, and the string is the debt.
#: ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the
#: ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires the declaring
#: set and the welded set to be DISJOINT and to EXHAUST it -- so this declaration is
#: the only way a gated family may stand unwelded, and emptying it without the weld
#: entry (or adding the entry without emptying it) fails that test by name.
#:
#: EMPTY SINCE 2026-09-07: ``fingerprints.json`` carries
#: ``metal_folded_fused_hd_pair_device_gate``, minted from the RELEASED artifact by
#: ``mint_metal_weld.py --family folded_fused_hd_pair``, and the wiring change that
#: landed the ``registry.FAMILY_MODULES`` row and the ``launch.FUSED_PAIR_ARMS``
#: absorb row emptied this string in the same pass.
WELD_OWED: str = ""

#: WHAT THE GATE MEASURED, KEPT. The pre-weld declaration is not deleted with the
#: weld: it is the one-line record of which artifact this family's release rests on,
#: and a reader who finds an empty ``WELD_OWED`` should still be able to find that
#: without opening the board. The stamp named here is the RELEASED pre-wire run
#: (``release.released`` true); the sibling ``_2026-09-07_folded`` directory is the
#: WITHHELD first attempt and binds nothing.
_RETIRED_WELD_OWED = (
    "the gate has run and passed on this host "
    "(results/metal_folded_fused_hd_pair_2026-09-07_folded2/gate.json, every leg "
    "PASS); what is owed is the LEDGER, not a measurement -- the first weld entry "
    "through mint_metal_weld.py, the fingerprints.json re-cut that adds this "
    "module's host digest and the gate's artifact digest, and the "
    "test_metal_weld_contract floor bumps that follow. Those edit files a parallel "
    "campaign's gate artifacts pin, so they land as one wiring change together with "
    "the registry.FAMILY_MODULES row this module is also missing")

#: The binding shape is the plain product's EXACTLY -- imported rather than
#: re-spelled, because it is the same signature (the fold adds no argument on this
#: backend: symmetry.py:328-332). The gate re-measures each of the four on this host
#: rather than inheriting the number.
PACKED_BINDINGS = _plain.PACKED_BINDINGS
SEPARATE_SCALAR_BINDINGS = _plain.SEPARATE_SCALAR_BINDINGS
UNSHARED_KMS_BINDINGS = _plain.UNSHARED_KMS_BINDINGS
ONE_MORE_POINTER_BINDINGS = _plain.ONE_MORE_POINTER_BINDINGS

#: The rotating volumes, in SIGNATURE ORDER. The plain product's, imported: this weld
#: has the same six and the same order, and two spellings of one order is how a
#: launch binds the wrong buffer to every argument after the first mismatch.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

__all__ = [
    "ARMS", "CARRIES_DEPOSIT_REPAIR", "FAMILY", "HOISTS_THE_WITHDRAW",
    "INSTALLABLE", "INSTALLABLE_REASON", "ONE_MORE_POINTER_BINDINGS",
    "PACKED_BINDINGS", "REPLACES", "ROTATED_NAMES", "SEAM",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNSHARED_KMS_BINDINGS", "WELD_OWED",
    "compile_folded_fused_hd_pair", "folded_fused_hd_pair_source",
    "folded_welded_curl_tail", "metal_folded_fused_hd_pair_coverage",
    "plan_metal_folded_fused_hd_pair", "refuted_one_more_pointer_source",
    "refuted_separate_scalar_source", "refuted_unshared_kms_source",
    "register_arms", "shipped_signature_bindings",
]

#: The three refuted signatures, IMPORTED. They measure the SIGNATURE, and this
#: product's signature is the plain one's -- 30 pointers plus one packed ``Params``
#: -- so re-spelling them here would be a second place for the same measurement to
#: go stale. The sibling ``folded_fused_magnetic_pair`` imports its own twin's the
#: same way (folded_fused_magnetic_pair.py:44-50).
refuted_separate_scalar_source = _plain.refuted_separate_scalar_source
refuted_unshared_kms_source = _plain.refuted_unshared_kms_source
refuted_one_more_pointer_source = _plain.refuted_one_more_pointer_source


# ---------------------------------------------------------------------------
# The lift: the FOLDED step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def folded_welded_curl_tail(codes: Sequence[int],
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The FOLDED ``step_D`` curl body below the decode prologue, H reads redirected.

    THE BODY IS :func:`.fused_hd_pair.welded_curl_tail`'S, WITH ONE SUBSTITUTION:
    the source is ``symmetry.folded_curl_source`` rather than ``shaders.curl_source``.
    Everything else -- the decode anchor, the offset parse, the three own-cell load
    edits, the six halo redirects, the final "no ``gN[`` survives" check -- is the
    plain module's own tables and helpers, imported and applied here, because the
    folded template's load lines ARE the certified template's own (symmetry.py:394-402
    is character-for-character shaders.py:154-162). Every table this walks
    (:data:`.fused_hd_pair.OWN_LOAD_EDITS`, :data:`.fused_hd_pair.HALO_TAPS`,
    :func:`.fused_hd_pair.offset_coordinates`, :data:`.fused_hd_pair.H_CELL_TAIL_ARGS`)
    is the plain product's, so an edit to how the certified curl spells a neighbour
    index reaches both welds without a second edit -- and a needle that stopped
    matching RAISES here rather than emitting a kernel that reads the wrong volume.

    THE TWO FOLD-SPECIFIC BLOCKS SURVIVE THE LIFT UNTOUCHED and that is a property of
    what they touch rather than an exemption. ``folded_top_plane_mask`` writes
    ``curlN = last_a ? 0.0f : curlN;`` -- no ``gN[`` -- and the widened cell-0 mask
    writes ``curlN = at_a ? 0.0f : curlN;``, likewise. Only the six shifted MAGNETIC
    loads are redirected, and the ghost ternary's guard and offset are PARSED OUT of
    the emitter's own line rather than retyped, so a folded axis's exact ``0.0f``
    stays an exact ``0.0f`` and no recompute is evaluated past the face at all.

    ``codes`` is the four-code folded quadruple :func:`.symmetry.folded_axis_kinds`
    resolves, NEVER a hand-built PERIODIC/METALLIC triple -- see the module docstring.
    """
    source = symmetry.folded_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), contract)
    if _plain.DECODE_END not in source:
        raise AssertionError(
            "the folded curl source no longer carries the decode anchor; the fused "
            "kernel would splice a truncated body")
    tail = source.split(_plain.DECODE_END, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError("the folded curl source does not end with '}'")
    tail = tail[: -len("}\n")]
    offsets = _plain.offset_coordinates(tail)
    for old, new in _plain.OWN_LOAD_EDITS:
        tail = _weld.needle(tail, old, new)
    for var, target in _plain.HALO_TAPS:
        prefix = f"    float {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the folded curl declares {var} {len(matches)} times; this weld "
                f"redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE EMITTER'S OWN LINE.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(" : 0.0f;\n"):
            raise AssertionError(
                f"the folded curl no longer reads {var} as `{expected}...]` served "
                f"an exact 0.0f past the face; the ghost rule this weld preserves "
                f"has changed, and on a fold that rule is what makes the ghost DEAD")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = _weld.needle(tail, old, f"{head} ? {call} : 0.0f;\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded folded curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read "
                f"must be the register or a recompute")
    return tail


def folded_fused_hd_pair_source(codes: Sequence[int],
                                contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised folded fused source: (folded boundary quadruple, mode).

    The plain product's :data:`.fused_hd_pair._TEMPLATE` with the FOLDED curl's
    prologue and the FOLDED welded tail. The kernel entry point keeps the plain
    product's name ``fused_hd_pair_step`` because the template is the plain
    product's; :func:`compile_folded_fused_hd_pair` is the only caller that names it.

    NO FILL AND NO WALL CLEAR IS CARRIED, and on a folded grid that is a stronger
    statement than on an unfolded one: ``fill_symmetry_bc_B``, ``zero_metal_B`` and
    ``fill_folded_far_ghosts_B`` all close BEFORE the ``update_H`` consult
    (driver.py:3305-3310), and their D twins all open AFTER the ``step_D`` consult
    (:3325-3330). A fill carried here would be a pass the driver runs again.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")

    curl = symmetry.folded_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), contract)
    prologue = (curl.split(_plain._BODY_ANCHOR, 1)[1]
                .split(_plain.DECODE_END, 1)[0] + _plain.DECODE_END)
    return shaders.substitute(_plain._TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__H_CELL__": _plain.h_cell_function(contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": folded_welded_curl_tail(codes, contract),
    })


def compile_folded_fused_hd_pair(codes: Sequence[int],
                                 contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (folded boundaries, mode)."""
    return compile_source(
        folded_fused_hd_pair_source(codes, contract)).fused_hd_pair_step


def shipped_signature_bindings() -> int:
    """How many bindings the SHIPPED folded kernel declares, off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify -- and read on a FOLDED triple, so a
    fold that had grown an argument would be caught here rather than on the
    degenerate unfolded emission.
    """
    source = folded_fused_hd_pair_source(
        (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
         symmetry.CODE_PERIODIC))
    signature = source.split("kernel void fused_hd_pair_step(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                        residency: Any = None) -> Coverage:
    """May ONE dispatch span the FOLDED ``update_H`` -> withdraw -> ``step_D``?

    A conjunction of the two FOLDED halves' own certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first -- and on this seam that matters, since the null
    ``update_H`` (an inactive absorber returns before its first statement) is the
    single largest non-fusion reason on the board and belongs to the constitutive
    side.

    THE TWO HALVES ARE DELIBERATELY DIFFERENT PREDICATES rather than one. The folded
    CURL's contract refuses a conductivity on the D targets, because a conductivity
    routes to a different curl recurrence; the folded CONSTITUTIVE's deliberately
    does NOT, because a conductivity does not change ``update_H``
    (symmetry.py:1062-1070). The conjunction is what refuses the conductive rows, and
    inheriting one half's clause list for the other would silently narrow an
    independent sub-step.
    """
    reasons: List[str] = []

    magnetic = symmetry.folded_constitutive_coverage(fields, pml, "H", residency)
    if not magnetic.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in magnetic.reasons)
    curl = symmetry.folded_composition_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business. What IS between them is the electric integrated-source
    # withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it. IGNORANCE IS
    # NEVER AN EMPTY SET: `Fields` does not hold the source list, so a predicate that
    # inferred "no withdraw stands" from not being told would be the over-covering
    # refusal this clause exists to prevent.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from "
            "Fields that no electric withdraw stands between update_H and step_D"),
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
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE FOLD, REQUIRED, AND RESTATED AT PRODUCT LEVEL. Both halves already require
    # a mirror plane (`symmetry._requires_a_fold`, asked of the grid AND of the
    # resolved codes). The restatement is so the reason reads AT THE SEAM rather than
    # only at the arms, and it is the exact mirror image of `fused_hd_pair`'s clause,
    # which refuses a fold naming this product: the two predicates can never both
    # admit, and neither can be widened into the other without making every folded
    # row ambiguous.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if not folded_axes:
        reasons.append(
            "no axis is folded by a mirror plane: this product implements the "
            "`folded` update_H and `folded` step_D arms, and an unfolded run "
            "selects the `ordinary` update_H and `PML` step_D arms, which is "
            "fused_hd_pair's (ordinary -> PML) cell and its product")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES after each launch, so those
    # attributes must be settable and must be the arrays the residency mirrored.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a "
                f"plan-owned scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan_metal_folded_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        codes: Optional[Sequence[int]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the folded fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path,
    never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the SHADER mutation seam and ``codes`` the STRUCTURAL one. The
    gate compiles a deliberately broken copy of the shipped source and hands it
    through the first; through the second it hands the PERIODIC/METALLIC triple the
    plain product builds from ``_boundary_kinds``, which on a folded grid is a smooth
    wrong answer rather than a crash and is the one structural defect this family can
    make silently. Dropping either argument is not a slowdown, it is a silent
    DISARMING: every mutation leg would launch the shipped kernel and report the
    defect as uncaught.
    """
    if not metal_folded_fused_hd_pair_coverage(fields, pml, sources,
                                               residency).covered:
        return None

    grid = fields.grid
    if codes is None:
        # FROM `folded_axis_kinds` AND FROM NOWHERE ELSE. It routes the ghost rule
        # through `stepping._boundary_kinds` (where a fold outranks the outer
        # declaration), splits MIRROR through `stepping._stored_past_owned`, and
        # cross-checks that split against `grid.is_metallic` -- two independent
        # routes to the same fact, with a disagreement a refusal rather than a coin
        # toss. The plain product's `1 if kind == "metallic" else 0` expression maps
        # a fold to PERIODIC: the ghost becomes a wrap and the cell-0 mask vanishes.
        codes, _reasons = symmetry.folded_axis_kinds(grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    codes = tuple(int(code) for code in codes)

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

    # ONE SUB-LATTICE, TWO HALVES -- and the fold does not touch it. The PML
    # coefficient vectors are built at the FOLDED STORED EXTENT already
    # (symmetry.py:328-332 and the layout clauses in
    # `folded_pml_curl_coverage`/`folded_constitutive_coverage`), so the sharing
    # argument is exactly the plain product's: `SUB_STEPS['step_D']['suffix']` is ''
    # and `CONSTITUTIVE_SIDES['H']['half_integer']` is False, so the curl's kms and
    # the constitutive's kms are THE SAME THREE VOLUMES. Both suffixes are READ FROM
    # THE SHIPPED TABLES and their equality is ASSERTED rather than assumed: it is
    # the whole premise of the 30-pointer signature, and if either table moved,
    # sharing the group would bind one half's coefficients to the other half's
    # lattice -- a converged, smooth, half-cell-wrong absorber profile.
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
            selected[mode] = compile_folded_fused_hd_pair(codes, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_plain._params_tensor(grid.shape, dtdx, residency.device)])
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
# Registration — wired=False, gated on the fold, and refused by the composer twice
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"folded fused H/D pair cannot fill {slot}",))
    return metal_folded_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_folded_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``, GATED on the fold.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips
    an unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it -- which is what lets a composition sweep measure disjointness against
    this predicate instead of assuming it, and what lets
    ``launch._neighbouring_seam_claimant`` see the row when it asks who claims the
    seam ``update_H`` opens.

    THE GATE IS ``symmetry._has_fold`` AND IT IS LOAD-BEARING IN A WAY THE PLAIN
    PRODUCT'S ROW IS NOT. A gated-out arm contributes NO reason, which is right: on
    an UNFOLDED run ``update_H`` carries four other arms that will speak, and this
    one would only add a refusal about a fold nobody asked for. That is the folded
    family's own convention (symmetry.py:1794-1802) and this row follows it.

    THIS MODULE IS IN ``registry.FAMILY_MODULES`` SINCE 2026-09-07, so a normal
    composition imports it and this function runs -- the arm reaches the table on
    every run, still ``wired=False`` and still gated on the fold. The registry row
    landed in the same wiring change as the ``launch.FUSED_PAIR_ARMS`` row and the
    ledger entry.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "fused H/D pair (folded)",
                          _arm_coverage, _arm_plan,
                          prefix="fused H/D pair (folded): ",
                          noun="fused folded-H/folded-D-curl pair",
                          gate=symmetry._has_fold,
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
