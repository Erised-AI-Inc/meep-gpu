"""The hand-CUDA fusion block: a fused product's two slots, and the deposit it carries.

WHAT WAS MISSING, AND WHY IT BLOCKED 58 SEAM-INSTANCES. ``arms.plan_step`` walks
``STEP_ORDER`` and gives AT MOST ONE arm each slot, so a product whose one launch
spans ``step_B -> [the three in-seam passes] -> update_H`` had nowhere to be: it is
not a slot arm
(``registry.NOT_REGISTERED`` says why) and this track had no block that fills two
slots at once. Without two slots there is no LEADING slot that saves and no TRAILING
slot that restores, so ``fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR`` had to stay
``False`` and the seam clause refused every in-seam magnetic deposit -- 58 of the 186
``B_to_H`` seam-instances the corpus drives
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-28b/board.log``, "THE
SOURCE-SEAM CEILING"). This module is the two slots.

IT IS A PORT, NOT AN INVENTION. ``metal_kernels/launch._install_fused_pair`` and
``triton_kernels/launch._install_fused_pair`` are the same eleven lines against the
same shared module, and :func:`_install_fused_pair` below is the third copy, one per
composer. What differs is only how a CANDIDATE is found, and that difference is
forced (see THE ONE DIVERGENCE below).

=============================================================================
THE TWO SLOTS ARE THE MECHANISM, NOT AN ACCOUNTING DETAIL
=============================================================================

``step_B`` and ``update_H`` are separate driver consults (``driver.py:3291`` and
:3298) with the source injection, the symmetry fill and the wall clear in between
(:3293-:3296), and ``FastPath.dispatch`` contracts that True means the WHOLE sub-step
ran. A pair that owns both runs the kernel at the first consult and spends the second
on either:

* a ``NoopPlan`` -- the seam carried no deposit and there is nothing left to do; or
* ``deposit_repair.TrailingRepairPlan`` -- the only point in the step at which the
  injected field is FINAL and the fused launch's pre-injection accumulation is still
  known, because ``LeadingRepairPlan`` captured it at the first consult.

NO DRIVER CHANGE, AND NO DEVICE-WIDE BARRIER. The repair is host-side and sits
BETWEEN two launches; it adds nothing inside one. What it exploits is that the
deposit is a sparse scatter (``sources.py:1305``) and the constitutive half has no
neighbour reads, so the launch may run over the whole grid against an uninjected
field and a median 0.016% of the cells may be recomputed afterwards
(``deposit_repair.py:9-21``).

=============================================================================
THE ONE DIVERGENCE FROM THE METAL BLOCK, AND WHY IT IS FORCED
=============================================================================

Metal finds its candidates by walking ``arms.registered(curl_name)`` for specs whose
``replaces`` spans the seam -- its family modules register themselves at import.
THIS TRACK CANNOT DO THAT, for the reason ``registry.py:12-22`` already gives about
the arm table: the hand-CUDA family modules ``import cupy`` at module scope, and the
census that establishes what this composition covers is BACKEND-FREE. A table that
eagerly imported ``fused_magnetic_pair`` would be unbuildable on the very host the
measurement is checked on.

So :data:`FUSED_PRODUCTS` is a small table of its own and every predicate and builder
is resolved LAZILY, inside the guarded call. Where a product's module cannot be
imported at all, the seam gets a NAMED REFUSAL -- not an exception, and not a silent
skip -- which is the same fail-closed answer a refusing predicate gives.

TODAY THE ONE PRODUCT DOES IMPORT ON A CUPY-FREE HOST, and that was the enabling edit
rather than a happy accident: ``fused_magnetic_pair`` now takes ``cupy`` and its three
certified halves defensively, so its PREDICATE -- the part whose failure mode is a
silent wrong answer -- can be asked, tested and composed on a laptop, while the
emitter and ``_get_kernel`` refuse by name. So this block plans a real fused seam here;
what it cannot do here is LAUNCH one, and no test on this host claims otherwise.

=============================================================================
WHAT THIS BLOCK MAY NOT DO
=============================================================================

* **It may not claim a slot the arm table refused.** :func:`_pair_may_absorb` is
  imported from the shared composer, not re-implemented: a slot left UNSELECTED
  because two arms admitted it, or because its builder refused, may not then be
  taken by the fused product -- which is exactly the product an ambiguity declined
  to pick. A slot won by a DIFFERENT arm may not be substituted either.
* **It may not run unless asked.** ``arms.plan_step`` takes ``fuse=False`` by
  default, so the composition the coverage census measured is unchanged and this
  block is opt-in.
* **It may not raise.** ``plan_step`` never raises; a predicate or builder that
  raises here is a named refusal (:func:`_guarded_verdict`, :func:`_guarded_plan`),
  because an opt-in optimisation that CRASHES the plan is strictly worse than one
  that refuses it.
* **It may not check the repair flag itself.** The flag is checked in the family's
  own coverage predicate, which passes ``carries_repair=CARRIES_DEPOSIT_REPAIR`` to
  ``deposit_repair.seam_source_reasons``. Checking it twice would let the two
  answers disagree; a product that has not declared it refuses every in-seam source
  and never reaches :func:`_install_fused_pair` at all.

NOTHING HERE IS DISPATCH, corrected 2026-09-02.
``meep_gpu.fastpath.plan_fast_path`` no longer returns ``None`` on every branch: it
returns a ``FastPathPlan``, the driver consults it at seven slots, and an opted-in
run dispatches five cross-sub-step fused products -- through the TRITON seam loop
this file's :func:`install_fused_pairs` was ported into
(``triton_kernels/launch.py``'s ``_install_certified_fused_products``). This
package is still unreachable from that function, which imports ``triton_kernels``
and nothing else, so :meth:`CudaFusedPairPlan.run` -- the one function in this file
that would touch a device -- is called by no code path a plan build reaches.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence, Tuple

from .. import deposit_repair as _deposit_repair
from .. import withdraw_hoist as _withdraw_hoist
from . import compile_cache as _compile_cache
from ..triton_kernels.coverage import Coverage
from ..triton_kernels.launch import (NoopPlan, _guarded_plan, _guarded_verdict,
                                     _pair_may_absorb)

__all__ = ["FUSED_PAIR_ARMS", "FUSED_PAIR_EXTRA_ARMS", "FUSED_PAIR_SEAMS",
           "FUSED_PRODUCTS", "CudaFusedPairPlan", "CudaFusedTriplePlan",
           "declaring_plan", "install_fused_pairs", "replaced_sub_steps",
           "span_of"]


#: curl slot -> (constitutive slot, the ``deposit_repair`` seam name).
#:
#: THE D/E ROW LANDED WITH ``fused_electric_pair`` AND NOT BEFORE IT, which is the
#: rule this table has followed since it was written: an empty row would report "no
#: candidate" on every configuration for a seam nobody had built anything for, and a
#: reason that is true of every run reads as a claim about the run. It is the same
#: pair of slots ``metal_kernels.launch.FUSED_PAIR_SEAMS`` has carried since its own
#: electric product landed, and the seam name is ``deposit_repair``'s own: ``'D'``
#: selects the ELECTRIC list (``deposit_repair._in_seam_indexed``, where ``pair ==
#: "B"`` is the magnetic test and everything else is electric).
#:
#: THE E->P ROW LANDED 2026-09-01 WITH ``fused_polarization_pair``, under the same
#: landed-with-its-product rule, and its deposit-list name is ``None`` BECAUSE THE
#: SEAM HAS NO DEPOSIT LIST TO SELECT: the driver advances the polarizations
#: immediately after ``update_E`` (driver.py:3313 then :3315) with NOTHING between
#: the two consults -- no injection, no fill, no wall pass. Until that product
#: landed this table said the row "must not" exist, for exactly that reason read
#: the other way: with nothing to carry, a row would have given the seam a
#: deposit-protocol refusal that reads as a claim about a pair. What changed is
#: that a product now occupies the seam, so the row records the driver fact
#: instead: ``None`` means :func:`_install_fused_pair` reads the seam as EMPTY
#: unconditionally and always installs the ``NoopPlan`` branch. Handing
#: ``_in_seam_indexed`` any string here would silently select the electric or the
#: magnetic list -- both injected OUTSIDE this seam -- which is the misclassification
#: the ``None`` exists to make impossible.
#: THE H->D ROW LANDED 2026-09-04 AND IT IS THE ONE ROW ON THIS TABLE THAT LANDED
#: AHEAD OF ITS PRODUCT. The landed-with-its-product rule above was written against
#: a defect that this loop no longer has: :func:`install_fused_pairs` opens each seam
#: with ``if not candidates: continue`` and records NOTHING, so a row with no product
#: is SILENT on every configuration rather than reporting "no candidate" on every
#: one. What the row buys before its product exists is the arbitration
#: (:func:`_neighbouring_seam_claimant`) and the KEY ORDER, both of which have to be
#: in the tree and pinned by a test before a product can be offered a slot a released
#: product holds.
#:
#: THE ORDER IS LOAD-BEARING AND THE ROW IS APPENDED LAST.
#: :func:`install_fused_pairs` iterates ``.items()`` in dict order and reads the LIVE
#: ``selected``. Placed first, this seam would be offered ``update_H`` before
#: ``cuda_fused_magnetic_pair`` was, would find the slot carrying an ARM label rather
#: than a fused one, and :func:`_pair_may_absorb` would let it take both slots --
#: displacing a released, gate-passing product on every row that reaches it.
#: ``test_the_h_to_d_seam_row_is_last`` pins the order with that reason.
#:
#: ITS SEAM NAME IS ``withdraw_hoist.SEAM`` AND IT IS NEITHER A FIELD LETTER NOR
#: ``None``, which is a three-way distinction rather than a spelling. ``'B'``/``'D'``
#: select ``deposit_repair``'s magnetic/electric INJECTION lists; ``None`` (the E->P
#: row) means the seam is empty, which is the driver's own answer there. This seam is
#: neither: nothing is injected between ``update_H`` and ``step_D``, but the driver's
#: electric ``withdraw`` loop DOES sit between the consults, and a product spanning
#: the seam has to perform it before its launch (:mod:`..withdraw_hoist`). ``None``
#: could not say that, and a field letter would say something false --
#: ``deposit_repair._in_seam_indexed`` classifies ``pair != "B"`` as the ELECTRIC
#: list, so ``None`` handed to the two SIBLING copies (which do not guard it) would
#: silently bracket this seam with the electric deposit repair. The name is read by
#: :func:`_install_fused_pair`, which routes it to the withdraw hoist instead of to
#: ``deposit_repair.in_seam_sources``.
FUSED_PAIR_SEAMS: Dict[str, Tuple[str, Optional[str]]] = {
    "step_B": ("update_H", "B"),
    "step_D": ("update_E", "D"),
    "update_E": ("update_P", None),
    "update_H": ("step_D", _withdraw_hoist.SEAM),
}

#: Which ARM each fused product's kernel implements, per slot it absorbs. READ FROM
#: THE PREDICATE, not assigned: ``covers_fused_magnetic_pair`` opens with
#: ``covers_real_pml_curl(fields, pml, grid, "step_B")`` and
#: ``covers_real_pml_constitutive(fields, pml, grid, "H")``
#: (``fused_magnetic_pair.py``'s coverage section), and those two functions ARE the
#: ``PML`` curl arm's and the ``ordinary`` constitutive arm's own predicates
#: (``registry._TABLE``'s first two rows). The board's rank-1 cell is the same pair
#: of labels: ``B_to_H (cuda_curl/PML, cuda_constitutive/ordinary)``.
#:
#: A family with no row here is REFUSED BY NAME rather than absorbing a slot on a
#: mapping nobody established -- the same bound Metal's ``FUSED_PAIR_ARMS`` puts in
#: front of its seam loop.
#:
#: THE TWO COMPLEX ROWS LANDED 2026-08-30 with the products themselves, and they are
#: read from those predicates the same way. ``covers_complex_fused_magnetic_pair``
#: opens with ``covers_real_pml_complex_curl(..., "step_B")`` and
#: ``covers_real_pml_complex_constitutive(..., "H")``, which are BOTH the
#: ``cuda_complex`` family's ``complex`` arm (``registry._TABLE``'s two complex
#: rows), matching the board's rank-2 cell ``B_to_H (cuda_complex/complex,
#: cuda_complex/complex)``. ``covers_cylindrical_fused_magnetic_pair`` opens with
#: ``covers_pml_cylindrical_complex_curl`` -- the ``cuda_cyl_complex`` family's
#: ``cylindrical complex`` arm -- and the same complex constitutive, matching the
#: rank-1 cell ``B_to_H (cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``.
#:
#: THE ELECTRIC ROW LANDED 2026-08-30 and is read the same way:
#: ``covers_fused_electric_pair`` opens with ``covers_real_pml_curl(..., "step_D")``
#: and ``covers_real_pml_constitutive(..., "E")``, which ARE the ``PML`` curl arm's
#: and the ``ordinary`` constitutive arm's own predicates (``registry._TABLE``'s first
#: two rows, both of which register on ``step_D``/``update_E`` as well as on the
#: magnetic slots). The board's ``D_to_E (cuda_curl/PML, cuda_constitutive/ordinary)``
#: cell is the same pair of labels. IT SHARES ITS TWO LABELS WITH THE REAL MAGNETIC
#: PAIR AND THAT IS NOT AN AMBIGUITY: the two occupy DIFFERENT seams, and
#: :func:`install_fused_pairs` decides one seam at a time.
#:
#: THE TWO COMPLEX ELECTRIC ROWS LANDED 2026-08-30 with their products, and are read
#: off those predicates the same way. ``covers_complex_fused_electric_pair`` opens with
#: ``covers_real_pml_complex_curl(..., "step_D")`` and
#: ``covers_real_pml_complex_constitutive(..., "E")``, which are BOTH the
#: ``cuda_complex`` family's ``complex`` arm, matching the board's ``D_to_E
#: (cuda_complex/complex, cuda_complex/complex)`` cell.
#: ``covers_cylindrical_fused_electric_pair`` opens with
#: ``covers_pml_cylindrical_complex_curl(..., "step_D")`` -- the ``cuda_cyl_complex``
#: family's ``cylindrical complex`` arm -- and the same complex constitutive, matching
#: ``D_to_E (cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``. Each SHARES
#: its label pair with its magnetic twin, and that is not an ambiguity for the same
#: reason the real electric row is not: the twins occupy DIFFERENT seams and
#: :func:`install_fused_pairs` decides one seam at a time.
#: THE NO-ABSORBER ELECTRIC ROW LANDED 2026-08-31 and is read the same way, off the
#: predicate it conjoins: ``covers_no_pml_complex_fused_electric_pair`` opens with
#: ``complex_no_pml_kernels.covers_complex_no_pml_curl(..., "step_D")`` and
#: ``covers_complex_no_pml_stored_e``, which ARE the ``cuda_complex_no_pml`` family's
#: ``complex no-PML curl`` and ``complex no-PML stored E`` arms (``registry._TABLE``'s
#: two rows of that family on these slots). The board's ``D_to_E
#: (cuda_complex_no_pml/complex no-PML curl, cuda_complex_no_pml/complex no-PML stored
#: E)`` cell is the same pair of labels.
#:
#: IT SHARES ITS LABEL PAIR WITH NO OTHER ROW, and unlike the three PML rows above it
#: does not even need the seam loop's one-seam-at-a-time rule to stay disjoint from
#: them: the three PML products refuse a run with no ACTIVE absorber by name and this
#: one requires exactly that, so the four candidates on ``step_D`` partition on a
#: single boolean.
FUSED_PAIR_ARMS: Dict[str, Tuple[str, str]] = {
    "cuda_fused_magnetic_pair": ("PML", "ordinary"),
    "cuda_complex_fused_magnetic_pair": ("complex", "complex"),
    "cuda_cylindrical_fused_magnetic_pair": ("cylindrical complex", "complex"),
    # THE CARTESIAN COMPLEX H->D ROW, 2026-09-07, read off its predicate like every
    # row here: ``covers_complex_fused_hd_pair`` opens with
    # ``covers_real_pml_complex_constitutive(..., "H")`` -- the ``cuda_complex``
    # family's ``complex`` arm at ``update_H`` -- and then asks the certified curl
    # predicate its VARIANT selects at ``step_D``. On an unfolded grid that is
    # ``covers_real_pml_complex_curl``, the same ``cuda_complex``/``complex`` arm, and
    # the pair below is the board's cell ``H_to_D (cuda_complex/complex,
    # cuda_complex/complex)``. On a FOLDED grid it is
    # ``complex_folded_kernels.covers_complex_folded_curl`` and the board cell is
    # ``H_to_D (cuda_complex/complex, cuda_complex_folded/folded complex)`` -- ONE
    # product over TWO cells, which is why the arm row cannot name both and the
    # extra-arm row below carries the folded half.
    "cuda_complex_fused_hd_pair": ("complex", "complex"),
    "cuda_fused_electric_pair": ("PML", "ordinary"),
    "cuda_complex_fused_electric_pair": ("complex", "complex"),
    "cuda_cylindrical_fused_electric_pair": ("cylindrical complex", "complex"),
    "cuda_no_pml_complex_fused_electric_pair":
        ("complex no-PML curl", "complex no-PML stored E"),
    # THE TWO E->P ROWS, 2026-09-01, read off the predicates they conjoin like
    # every row above: ``covers_fused_polarization_pair`` opens with
    # ``covers_real_pml_dispersive_constitutive`` -- the ``cuda_dispersive``
    # family's ``dispersive`` arm at ``update_E`` -- and
    # ``covers_real_pml_ade_update_p`` -- the ``cuda_ade`` family's ``ADE`` arm
    # at ``update_P``; its no-absorber twin opens with the ``cuda_no_pml_dispersive``
    # and ``cuda_no_pml_ade`` families' own two predicates. The first slot of
    # this seam is a CONSTITUTIVE sub-step rather than a curl, and the tuple
    # order is (first slot's arm, second slot's arm) as everywhere else.
    "cuda_fused_polarization_pair": ("dispersive", "ADE"),
    "cuda_no_pml_fused_polarization_pair":
        ("no-PML dispersive store", "no-PML ADE"),
    # THE FIVE RESIDUAL MAGNETIC ROWS, 2026-09-02, read off the predicates they
    # conjoin like every row above. ``covers_complex_folded_fused_magnetic_pair``
    # opens with ``covers_complex_folded_curl(..., "step_B")`` -- the
    # ``cuda_complex_folded`` family's ``folded complex`` arm -- and
    # ``covers_real_pml_complex_constitutive(..., "H")`` -- the ``cuda_complex``
    # family's ``complex`` arm, which the fold admits directly. The beta weld
    # conjoins the ``cuda_complex_beta`` family's own two; the Dcyl real weld the
    # ``cuda_cylindrical`` curl arm and the ``ordinary`` constitutive; the
    # special_kz and BFAST welds each conjoin their family's own curl AND
    # constitutive admissions, which is why both slots carry the same label.
    "cuda_complex_folded_fused_magnetic_pair": ("folded complex", "complex"),
    "cuda_complex_beta_fused_magnetic_pair": ("complex beta", "complex beta"),
    "cuda_cylindrical_real_fused_magnetic_pair": ("cylindrical", "ordinary"),
    "cuda_special_kz_fused_magnetic_pair": ("real beta", "real beta"),
    "cuda_bfast_fused_magnetic_pair": ("BFAST", "BFAST"),
    # THE FIVE ELECTRIC TWINS, 2026-09-02 residue round (fusion-residue audit
    # §1.3: 12 electric-twin instances), read off the predicates they conjoin
    # like every row above -- each is its magnetic twin's label pair on the
    # OTHER seam, which is not an ambiguity because install_fused_pairs decides
    # one seam at a time. ``step_D`` becomes a NINE-candidate seam and the
    # partition is the same measured one ``step_B`` already carries, plus the
    # absorber boolean that separates the no-PML complex product from the rest.
    "cuda_complex_folded_fused_electric_pair": ("folded complex", "complex"),
    "cuda_complex_beta_fused_electric_pair": ("complex beta", "complex beta"),
    "cuda_cylindrical_real_fused_electric_pair": ("cylindrical", "ordinary"),
    "cuda_special_kz_fused_electric_pair": ("real beta", "real beta"),
    "cuda_bfast_fused_electric_pair": ("BFAST", "BFAST"),
    # THE THIRD E->P ROW, 2026-09-02: the complex no-absorber polarization pair,
    # read off the ``cuda_complex_no_pml`` family's own two arms at update_E and
    # update_P. Disjoint from the two REAL E->P rows on storage alone: their
    # halves refuse complex64 by name and these require it.
    "cuda_complex_no_pml_fused_polarization_pair":
        ("complex no-PML stored E", "complex no-PML ADE"),
    # THE TWO STENCIL WELDS, 2026-09-02, and the first rows on this table whose
    # cells the board had recorded UNBUILDABLE. Read off the predicates they
    # conjoin like every row above: ``covers_offdiag_fused_electric_pair`` opens
    # with ``covers_real_pml_curl(..., "step_D")`` -- the ``cuda_curl`` family's
    # ``PML`` arm -- and ``covers_real_pml_offdiag_constitutive``, which IS the
    # ``cuda_offdiag`` family's ``off-diagonal`` arm (``registry._TABLE``'s row on
    # ``update_E``); the folded one conjoins the same curl arm with
    # ``covers_folded_offdiag_composition``, the ``cuda_folded_offdiag`` family's
    # ``folded off-diagonal`` arm. The board's two STENCIL-BLOCKED D_to_E cells are
    # those two label pairs.
    #
    # THEY PARTITION AGAINST EACH OTHER AND AGAINST THE SEVEN OTHER step_D ROWS ON
    # ONE MEASURED BOOLEAN: the unfolded constitutive predicate refuses a fold
    # twice (coverage.py:1755-1791) and the composition predicate REQUIRES one, so
    # no row can be claimed by both -- and both refuse complex storage, beta,
    # BFAST, Dcyl and a nonzero Bloch k, which is how they stay disjoint from the
    # rest. The census's overlap check measures it rather than trusting this note.
    "cuda_offdiag_fused_electric_pair": ("PML", "off-diagonal"),
    "cuda_folded_offdiag_fused_electric_pair": ("PML", "folded off-diagonal"),
    # THE FOURTH E->P ROW, 2026-09-02, and the last unserved cell on that seam.
    # Read off the predicate it conjoins like every row above:
    # ``covers_dispersive_offdiag_fused_polarization_pair`` opens with
    # ``covers_real_pml_dispersive_offdiag_constitutive`` -- the
    # ``cuda_dispersive_offdiag`` family's ``dispersive off-diagonal`` arm at
    # update_E -- and ``covers_real_pml_ade_update_p``, the ``cuda_ade`` family's
    # ``ADE`` arm at update_P. It shares its SECOND label with
    # ``cuda_fused_polarization_pair`` and partitions against it on the FIRST:
    # that product's update_E half refuses an off-diagonal row by name and this
    # one's requires one.
    "cuda_dispersive_offdiag_fused_polarization_pair":
        ("dispersive off-diagonal", "ADE"),
    # THE DISPERSIVE ELECTRIC ROW, 2026-09-02, and the largest single cell left
    # unoccupied on any backend when it landed: ``D_to_E (cuda_curl/PML,
    # cuda_dispersive/dispersive)``, 7 corpus rows. Read off the predicate it
    # conjoins like every row above: ``covers_dispersive_fused_electric_pair``
    # opens with ``coverage.covers_real_pml_curl(..., "step_D")`` -- the
    # ``cuda_curl`` family's ``PML`` arm -- and
    # ``dispersive_kernels.covers_real_pml_dispersive_constitutive``, which IS the
    # ``cuda_dispersive`` family's ``dispersive`` arm at ``update_E``
    # (``registry._TABLE``'s row, and the FIRST label of the
    # ``cuda_fused_polarization_pair`` row above, which holds the SAME arm one seam
    # later).
    #
    # IT PARTITIONS AGAINST ``cuda_fused_electric_pair`` ON ONE MEASURED BOOLEAN and
    # that partition is load-bearing rather than tidy: ``install_fused_pairs`` leaves
    # a seam UNFUSED when two products admit it, so an overlap here would COST the 79
    # rows the real twin serves rather than adding seven. The ordinary constitutive
    # predicate refuses ``fields.polarizations`` truthy
    # (``coverage.covers_real_pml_constitutive``) and the dispersive one REQUIRES it
    # (its own stated disjointness clause), so no run can be claimed by both. Against
    # the other nine ``step_D`` rows it is disjoint for the reasons they are disjoint
    # from the real twin -- both refuse complex storage, beta, BFAST, Dcyl, an
    # off-diagonal row and an inert absorber by name.
    "cuda_dispersive_fused_electric_pair": ("PML", "dispersive"),
    # THE NO-ABSORBER DISPERSIVE-STORE ROW, 2026-09-02. Its PRIMARY cell is
    # ``D_to_E (cuda_conductive/conductive, cuda_no_pml_dispersive/no-PML
    # dispersive store)`` -- 2 corpus rows, ``absorber-1d.py`` and
    # ``TestAbsorber.test_absorber`` -- and the SECOND cell it occupies is
    # declared in ``FUSED_PAIR_EXTRA_ARMS`` below. Read off the predicate like
    # every row above: ``covers_no_pml_dispersive_fused_electric_pair`` conjoins
    # ``dispersive_kernels.covers_no_pml_dispersive_constitutive`` -- which IS
    # the ``cuda_no_pml_dispersive`` family's only arm -- with a DISJUNCTION over
    # the two ``step_D`` curl predicates, of which the conductive one is this
    # row's first label.
    #
    # IT PARTITIONS AGAINST EVERY OTHER ``step_D`` ROW ON THE LAYER. This is the
    # only electric product whose constitutive half REQUIRES an inert absorber
    # (``covers_no_pml_dispersive_constitutive`` refuses an active one by name and
    # every other row's constitutive half requires one), so no run can be claimed
    # by two.
    "cuda_no_pml_dispersive_fused_electric_pair":
        ("conductive", "no-PML dispersive store"),
    # THE CONDUCTIVE x ORDINARY ROW, 2026-09-02, and the cell this campaign has
    # called THE SIGNED-ZERO ROW: ``D_to_E (cuda_conductive/conductive,
    # cuda_constitutive/ordinary)``, 1 corpus row
    # (``tests:TestAdjointSolver.test_damping``). Read off the predicate:
    # ``covers_conductive_fused_electric_pair`` conjoins
    # ``conductive_kernels.covers_conductive_curl`` at ``step_D`` -- the
    # ``cuda_conductive`` family's arm -- with
    # ``coverage.covers_real_pml_constitutive(side='E')``, which IS the
    # ``cuda_constitutive`` family's ``ordinary`` arm one slot later.
    #
    # IT PARTITIONS AGAINST ``cuda_fused_electric_pair`` ON THE CONDUCTIVITY, and
    # that partition is load-bearing rather than tidy: an overlap would leave the
    # seam UNFUSED and cost the 79 rows the real twin serves.
    # ``coverage.covers_real_pml_curl`` refuses a target carrying a sigma by name
    # and ``covers_conductive_curl`` REQUIRES one. Against the dispersive electric
    # row it is disjoint on ``fields.polarizations``, which that row's
    # constitutive half requires and this one's refuses.
    "cuda_conductive_fused_electric_pair": ("conductive", "ordinary"),
    # THE TWO COMPLEX STENCIL WELDS, 2026-09-02, and the LAST two cells this board
    # recorded UNBUILDABLE. Read off the predicates they conjoin like every row
    # above: ``covers_folded_complex_offdiag_fused_electric_pair`` opens with
    # ``complex_folded_kernels.covers_complex_folded_curl(..., "step_D")`` -- the
    # ``cuda_complex_folded`` family's ``folded complex`` arm -- and
    # ``complex_offdiag_update_e.covers_complex_offdiag_pml_update_e``, which IS the
    # ``cuda_complex_offdiag`` family's ``complex off-diagonal PML`` arm; the
    # no-absorber one conjoins ``covers_complex_no_pml_curl(..., "step_D")`` with
    # ``covers_complex_no_pml_offdiag_update_e``.
    #
    # THEY PARTITION AGAINST EACH OTHER ON THE ABSORBER, which is the same measured
    # switch the array path branches on (stepping.py:1015 against :1022): the PML
    # constitutive arm requires an active layer and the store arm requires an inert
    # one, so no run can be claimed by both. Against the ELEVEN other ``step_D``
    # rows they are disjoint on TWO clauses at once -- every one of those either
    # refuses complex64 storage by name or refuses an off-diagonal chi1inv row by
    # name, and these two require both. The census's overlap check measures it
    # rather than trusting this note.
    "cuda_folded_complex_offdiag_fused_electric_pair":
        ("folded complex", "complex off-diagonal PML"),
    "cuda_complex_no_pml_offdiag_fused_electric_pair":
        ("complex no-PML curl", "complex off-diagonal no-PML"),
    # THE TWO THREE-SLOT ROWS, 2026-09-02, and the first values on this table of
    # length THREE. The tuple is still "the arm each slot the product absorbs
    # implements", in the product's own slot order -- there is simply one more slot.
    # Read off the predicates each conjoins exactly like every row above:
    # ``covers_three_slot_dispersive_weld`` opens with
    # ``covers_dispersive_fused_electric_pair`` -- itself the ``cuda_curl/PML`` and
    # ``cuda_dispersive/dispersive`` pair -- and continues into
    # ``covers_fused_polarization_pair``, whose second half is the ``cuda_ade/ADE``
    # arm at ``update_P``. So the three labels are the three the arm table must
    # already have selected, and :func:`install_fused_pairs` asks
    # ``_pair_may_absorb`` TWICE for a triple, once per seam it spans, rather than
    # growing a second copy of that clause.
    #
    # THE NO-ABSORBER ROW'S FIRST LABEL IS ``conductive`` and its SECOND cell --
    # ``no-PML curl`` at ``step_D`` -- is declared in ``FUSED_PAIR_EXTRA_ARMS``
    # below, exactly as its D->E half declares the same pair of cells.
    "cuda_three_slot_dispersive_weld": ("PML", "dispersive", "ADE"),
    "cuda_three_slot_no_pml_dispersive_weld":
        ("conductive", "no-PML dispersive store", "no-PML ADE"),
    # THE COMPLEX NO-ABSORBER THREE-SLOT ROW, 2026-09-04, read off the predicates it
    # conjoins like every row above: ``covers_three_slot_complex_no_pml_dispersive_
    # weld`` opens with ``covers_no_pml_complex_fused_electric_pair`` -- itself the
    # ``cuda_complex_no_pml`` family's ``complex no-PML curl`` and ``complex no-PML
    # stored E`` arms -- and continues into ``covers_complex_no_pml_fused_
    # polarization_pair``, whose second half is that family's ``complex no-PML ADE``
    # arm at ``update_P``. Disjoint from the two REAL three-slot rows on storage
    # alone, and from every other ``step_D`` row on the absorber and the storage
    # together.
    "cuda_three_slot_complex_no_pml_dispersive_weld":
        ("complex no-PML curl", "complex no-PML stored E", "complex no-PML ADE"),
    # THE FIRST ROW ON THE H->D SEAM, 2026-09-06, and the first whose FIRST slot is a
    # CONSTITUTIVE sub-step and second a curl -- the tuple order is (first slot's arm,
    # second slot's arm) as everywhere else, so it reads (update_H's arm, step_D's).
    # Read off the predicates it conjoins like every row above:
    # ``covers_fused_hd_pair`` opens with ``covers_real_pml_constitutive(..., "H")``
    # -- the ``cuda_constitutive`` family's ``ordinary`` arm -- and continues into
    # ``covers_real_pml_curl(..., "step_D")``, the ``cuda_curl`` family's ``PML`` arm.
    # The board's cell is the same pair of labels: ``H_to_D (cuda_constitutive/
    # ordinary, cuda_curl/PML)``, 127 seam-instances -- nearly three times the sibling
    # Metal backend's 49, because this backend carries a mirrored periodic axis as a
    # RUNTIME boundary code (``BC_MIRROR_PERIODIC``) inside one kernel where Metal and
    # Triton need a separate folded arm.
    #
    # IT SHARES ITS LABEL PAIR WITH NOTHING, because no other row on this table sits
    # on the ``update_H`` seam at all: ``install_fused_pairs`` decides one seam at a
    # time and ``update_H`` has exactly one candidate. What it DOES collide with is
    # the released B->H pair's second slot, and that is arbitrated by
    # ``_neighbouring_seam_claimant`` and by the product's own ``INSTALLABLE = False``,
    # not here.
    "cuda_fused_hd_pair": ("ordinary", "PML"),
    # THE TWO CYLINDRICAL H->D ROWS, 2026-09-07, read off the predicates they
    # conjoin like every row above: ``covers_cylindrical_real_fused_hd_pair``
    # opens with ``covers_real_pml_constitutive(..., "H")`` -- the
    # ``cuda_constitutive`` family's ``ordinary`` arm -- and
    # ``covers_real_pml_cylindrical_curl(..., "step_D")`` -- the ``cuda_cylindrical``
    # family's ``cylindrical`` arm; the complex twin conjoins the ``cuda_complex``
    # family's ``complex`` constitutive arm with the ``cuda_cyl_complex`` family's
    # ``cylindrical complex`` curl arm. Both declare INSTALLABLE = False (the
    # cylindrical launch algebra FAVOURS them, +2 launches per step per row, and the
    # ``4 - pairs`` rule below does not know it; the flip is a measured composition
    # gate plus a per-family launches_saved weight). ``update_H`` becomes a
    # THREE-candidate seam, partitioned on the Dcyl grid (both refuse a Cartesian
    # one, the Cartesian product refuses Dcyl) and on storage between the two.
    "cuda_cylindrical_real_fused_hd_pair": ("ordinary", "cylindrical"),
    "cuda_cylindrical_fused_hd_pair": ("complex", "cylindrical complex"),
    # THE TWO BETA H->D ROWS, 2026-09-07, read off their predicates like every row
    # here. ``covers_special_kz_fused_hd_pair`` opens with
    # ``special_kz_curl.covers_special_kz_constitutive(..., "H")`` and continues into
    # ``covers_special_kz_curl(..., "step_D")`` -- BOTH the ``cuda_special_kz``
    # family's ``real beta`` arm, which is why the pair below repeats one label. The
    # complex twin conjoins ``complex_beta_kernels``' two ``complex beta`` predicates
    # the same way. The board's cells are the same label pairs: ``H_to_D
    # (cuda_special_kz/real beta, cuda_special_kz/real beta)`` -- 2 seam-instances --
    # and ``H_to_D (cuda_complex_beta/complex beta, cuda_complex_beta/complex beta)``
    # -- 4. Both declare INSTALLABLE = False for the Cartesian launch algebra
    # ``fused_hd_pair.INSTALLABLE_REASON`` prices, so ``_declared_uninstallable``
    # refuses them on every configuration before the predicate is asked.
    #
    # ``update_H`` becomes a FIVE-candidate seam and the partition is a measured one:
    # every other candidate refuses ``grid.beta != 0`` by name (the certified real,
    # complex and folded predicates all carry the clause), these two require it, and
    # they split from each other on STORAGE -- ``special_kz_curl._beta_reasons``
    # refuses complex64 by name and ``complex_beta_kernels._beta_reasons`` refuses
    # real float32 by name, each naming the other family.
    "cuda_special_kz_fused_hd_pair": ("real beta", "real beta"),
    "cuda_complex_beta_fused_hd_pair": ("complex beta", "complex beta"),
    # THE TWO REMAINING H->D CELLS, 2026-09-07 -- ONE PRODUCT, TWO ROWS, because a
    # row here is a (first slot's arm, second slot's arm) PAIR and this product's two
    # variants differ in exactly the second. Read off the predicates it conjoins like
    # every row above: on the conductive variant it opens with
    # ``covers_real_pml_constitutive(..., "H")`` -- the ``cuda_constitutive`` family's
    # ``ordinary`` arm -- and continues into
    # ``conductive_kernels.covers_conductive_curl(..., "step_D")``, the
    # ``cuda_conductive`` family's ``conductive`` arm; on the BFAST variant it opens
    # with ``bfast_curl.covers_bfast_constitutive(..., "H")`` and continues into
    # ``bfast_curl.covers_bfast_curl(..., "step_D")``, which are the two arms the
    # ``cuda_bfast`` family registers.
    #
    # THE TWO VARIANTS ARE DISJOINT AND IT IS THE SHIPPED PREDICATES THAT MAKE THEM SO,
    # not a rule here: ``covers_conductive_curl`` delegates every non-conductivity
    # clause to the certified curl predicate on a LOSSLESS proxy, which refuses a
    # BFAST grid by name, and ``covers_bfast_curl`` delegates to the same predicate
    # with the REAL fields, which refuses a conductive target by name -- so a run
    # carrying both is refused by BOTH halves and ``variant_for`` returns a refusal
    # rather than a tail. ``update_H`` becomes a FIVE-candidate seam; the partition
    # against the four rows above is on the conductivity, on ``grid.bfast_active`` and
    # on the Dcyl grid together, and the board's overlap check measures it.
    #
    # BOTH DECLARE INSTALLABLE = False, on the same Cartesian ``4 - pairs`` algebra:
    # on each cell BOTH neighbours are released products (the certified magnetic pair
    # and the conductive electric pair on one, the two BFAST pairs on the other), so a
    # span taking one slot from each can only tie or lose.
    "cuda_conductive_bfast_fused_hd_pair": ("ordinary", "conductive"),
}

#: ADDITIONAL (curl arm, constitutive arm) pairs a product may absorb BESIDE the
#: primary row above -- a separate table rather than a widened value shape, so
#: every existing pin on a primary row stays a pin. ONE ROW, 2026-09-01, and it is
#: read off the predicate like every primary row: on a run carrying an
#: instantaneous chi2/chi3 the composer gives ``update_H`` to the
#: ``cuda_nonlinear`` family's ``nonlinear`` arm, whose kernel there IS
#: ``fused_update_H_pml_real`` -- the ordinary certified constitutive under a
#: nonlinear-only spine arm, because ``stepping.update_H`` (:907-923) reads
#: nothing nonlinear (``registry._TABLE``'s ``cuda_nonlinear`` row spells the
#: per-slot dict). ``covers_fused_magnetic_pair`` consults
#: ``covers_real_pml_nonlinear_constitutive(side="H")`` for exactly these rows,
#: so the absorb declaration and the predicate move together;
#: ``gate_cuda_fused_magnetic_pair.py``'s nonlinear fixtures are the measurement.
#: The board's cell is ``B_to_H (cuda_curl/PML, cuda_nonlinear/nonlinear)`` -- 2
#: seam-instances, ``3rd-harm-1d.py`` and ``Test3rdHarm1d.test_3rd_harm_1d``.
#:
#: THE SECOND ROW, 2026-09-02, and it is the table's original purpose used for the
#: first time: ONE product occupying TWO board cells. The no-absorber
#: dispersive-store weld's constitutive half is the same certified kernel on both,
#: and its curl half is ONE emitter -- ``conductive_kernels`` bakes the
#: per-component conductivity into three ``COND`` defines and takes the certified
#: lossless tail behind ``#else``, so the ``(False, False, False)`` build IS
#: ``no_pml_curl``'s ``step_D_no_pml_real``, character for character (that module
#: builds its template by transforming those very bytes). The two PREDICATES stay
#: disjoint and are both asked whole: ``covers_conductive_curl`` requires a target
#: carrying a sigma and ``covers_no_pml_curl`` refuses one, so the disjunction in
#: ``covers_no_pml_dispersive_fused_electric_pair`` is a lookup between two arms
#: that cannot both hold rather than a widening.
#:
#: The board's second cell is ``D_to_E (cuda_no_pml_curl/no-PML curl,
#: cuda_no_pml_dispersive/no-PML dispersive store)`` -- 1 seam-instance,
#: ``examples:material-dispersion.py``, which is lossless and two-pole.
#:
#: THE THIRD ROW, 2026-09-02, is the no-absorber THREE-SLOT weld's, and it is the
#: second row's fact one slot longer: its group-1 launch IS the no-absorber
#: dispersive-store D->E kernel, whose ``(False, False, False)`` build is
#: ``no_pml_curl``'s certified lossless tail character for character, so the same one
#: product occupies both ``step_D`` cells. The trailing label is unchanged because
#: the ADE arm does not know about the curl.
#: THE FOURTH ROW, 2026-09-06, is the H->D weld's nonlinear cell -- 2 seam-instances,
#: ``examples:3rd-harm-1d.py`` and ``tests:Test3rdHarm1d.test_3rd_harm_1d``. On this
#: backend that widening is not a byte comparison between two emitted texts, because
#: there is only ONE text: ``nonlinear_constitutive`` ships no ``update_H`` kernel at
#: all and its arm H is a NULL WIDENING that admits the shipped
#: ``constitutive_kernels.update_H_pml_real`` unchanged (that module's own docstring,
#: "Arm H (NULL WIDENING). No kernel.", with the premise armed as a defect in
#: ``gate_cuda_nonlinear.py``). The census agrees from the other side: both cells
#: record ``update_H_kernel: fused_update_H_pml_real``
#: (``results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``). The trailing label is
#: unchanged because the curl arm does not know about the nonlinearity -- the Pade
#: factor enters ``update_E``, one seam later.
FUSED_PAIR_EXTRA_ARMS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "cuda_fused_magnetic_pair": (("PML", "nonlinear"),),
    "cuda_no_pml_dispersive_fused_electric_pair":
        (("no-PML curl", "no-PML dispersive store"),),
    "cuda_three_slot_no_pml_dispersive_weld":
        (("no-PML curl", "no-PML dispersive store", "no-PML ADE"),),
    "cuda_fused_hd_pair": (("nonlinear", "PML"),),
    # THE BFAST VARIANT'S OWN CELL, 2026-09-07. It is an EXTRA row rather than a second
    # primary because a primary row is one pair per family and this product's two
    # variants are two pairs; the shape is the same one the nonlinear widening above
    # uses, and for the same reason -- one predicate, two label pairs the composer can
    # arrive at. Both labels are ``BFAST`` because the ``cuda_bfast`` family registers
    # BOTH a curl arm and a constitutive arm under that one label, and the
    # constitutive one ships NO kernel of its own: its entry point is the shipped
    # ``fused_update_H_pml_real`` (registry._TABLE's cuda_bfast constitutive row, and
    # h_to_d_seam_2026-09-04 records the same kernel on the row from the other side).
    "cuda_conductive_bfast_fused_hd_pair": (("BFAST", "BFAST"),),
}


# =============================================================================
# THE PLAN-HELD SOURCE, AND WHY THE COMPILED KERNEL IS NOT WHAT IS HELD
# =============================================================================
#
# WHAT THIS REMOVES, 2026-09-19. The first same-table CUDA timing put ``pml_2d``
# fused at 3.636 ms/step against the certified singles' 0.325 ms/step (the device
# figures are that round's, not re-measured here), and two host costs paid on EVERY
# launch are this file's.
#
# (1) Every plan builder below called its product's launcher without the launcher's
# own ``kernel=`` door (``fused_magnetic_pair.py:1584``,
# ``fused_electric_pair.py:1625``), so the launcher's ``(kernel or _get_kernel())``
# ran per launch (``fused_magnetic_pair.py:1622``, ``fused_electric_pair.py:1665``)
# and each getter re-spliced the whole certified source (29 133 and 43 293
# characters) before keying the memo on the fresh string. (2)
# :meth:`CudaFusedPairPlan.run` re-resolved the launch arguments per launch, which
# the tables' own docstrings say is done ONCE per frozen configuration
# (``fused_magnetic_pair.py:1480``, and the dispersive builder below).
#
# MEASURED on the laptop with a stub CuPy and NumPy fields: the magnetic getter
# 274.5 us per call, the electric 385.7 us; ``CudaFusedPairPlan.run`` end to end
# 313 us and 415 us per launch before this change, 15.3 us and 17.1 us after, of
# which the held acquisition is 2.9 us and 3.7 us and the certified launcher's own
# argument packing about 11 us.
#
# WHY THE SOURCE AND NOT THE KERNEL OBJECT. Both CUDA launch witnesses sit on the
# ``compile_cache`` path: ``CudaLaunchCounter.install`` rebinds
# ``compile_cache.get_or_compile`` to a proxy that counts every call of what it
# returns (``parity/meep_gpu/gate_dispatch_end_to_end.py:1049-1089``), and
# ``wrap_memo_store`` replaces the memo's VALUES after the first chunk has run
# (:1091-1117). A plan that held the object it was handed at its first launch would
# hold the UNWRAPPED kernel, every later launch would bypass both witnesses, and the
# route gate would read its two counters as disagreeing (:1140). So what is held is
# the emitted STRING; the key is rebuilt from it and ``get_or_compile`` is asked on
# every launch, through the module attribute rather than an import-time alias, so a
# proxy installed after this module loaded is still the one called.
#
# THE POLICY STAYS ON THE PER-LAUNCH PATH. ``kernel_cache_key`` reads
# ``compile_policy_token()`` whenever no token is passed (``compile_cache.py:178``),
# and none is passed here, so a policy installed after the first launch misses the
# memo and compiles the held text under the new policy -- the same answer the
# getter gave. The dispatch seam refuses to launch at all once the policy the
# freeze was gated on has lapsed (``fastpath.py:3845-3878``), so this is the second
# of two guards, not the only one.
#
# WHY A PLAN-HELD STRING IS NOT A STALE STRING. The getters emit per call so that a
# probe that rewrites a certified half is seen by the fused text too
# (``fused_magnetic_pair.py:1238-1250``). A plan lives for ONE freeze: the driver
# rebuilds it at every freeze (``driver.py:3266-3271``, re-armed by every
# ``invalidate_fast_path()``), the gates' wiring legs build a fresh plan per case
# (``gate_cuda_fused_magnetic_pair.py:1514``), and the mutation legs hand their
# mutated kernel to the launcher's ``kernel=`` door directly
# (``gate_cuda_fused_hd_pair.py:2016``) rather than through a plan. The string is
# emitted at the FIRST LAUNCH, not at plan build, so a probe that patches an
# emitter before a plan first runs is still honoured and a device-free plan build
# still emits nothing.
#
# THE KEY IS THE GETTER'S, COPIED AND PINNED. Each call below passes the name, the
# storage flag and the emitter exactly as that product's own ``_get_kernel`` does --
# including ``fused_electric_pair``'s ``True`` on a real-storage kernel
# (``fused_electric_pair.py:1265``) -- because a different key would split the memo
# from the entry ``warm()`` compiled and every first launch would recompile.
# ``test_fused_pair_held_source.py`` pins each covered product's key against its
# getter's.
#
# WHAT IS COVERED AND WHAT IS LEFT, 2026-09-19. Every builder that calls a
# ``_held_*`` helper below -- the argument-free getters, the arm-, mask- and
# arity-keyed ones held per shape, the four off-diagonal welds' text (their
# resolution is never held), both three-slot welds' LEADING group, and the H->D
# welds' launch 1. Left, each for a reason in the certified module rather than here,
# with the laptop getter cost per call (stub CuPy, 2026-09-19): both real
# polarization pairs, whose component launcher has no ``kernel=`` door
# (``fused_polarization_pair.py:925``; 1560 us per component); the three-slot
# welds' TRAILING group and ``complex_no_pml_three_slot_dispersive_weld``, which
# compile one specialization per component inside a ``run_`` whose single
# ``kernel=`` door would hand one text to all three (1748 us and 553 us per
# component); ``dispersive_offdiag_fused_polarization_pair`` and
# ``complex_no_pml_fused_polarization_pair``, per component the same way (1702 us,
# 46 us); and ``complex_fused_hd_pair`` / ``complex_beta_fused_hd_pair``, whose
# getters already memoize their emission (1.7 us, 1.5 us).


def _held_kernel(held: Dict[str, Any], module: Any, shape: Tuple[Any, ...],
                 emit: Callable[[], str], key_name: str, is_complex: bool,
                 name: str) -> Any:
    """The product's compiled kernel from a source emitted ONCE per plan and shape.

    ``held`` is the plan's own dict; ``shape`` is the hashable form of whatever the
    product's getter takes besides the source (``()`` for an argument-free getter, the
    arm, the mask or the canonical pole arity otherwise), so a plan whose getter
    argument can change between launches gets one emission per distinct value rather
    than a stale one. ``emit`` is the product's own emitter, read off its module by
    the launch that calls this, and ``key_name``/``is_complex``/``name`` are its
    getter's, verbatim.

    Returns whatever ``compile_cache.get_or_compile`` returns, which is the object a
    gate's proxy or memo wrap has substituted when one is installed.
    """
    sources = held.setdefault("sources", {})
    source = sources.get(shape)
    if source is None:
        source = sources[shape] = emit()
    options = module._COMPILE_OPTIONS  # noqa: SLF001 - the getter's own tuple
    key = _compile_cache.kernel_cache_key(key_name, is_complex, options, source)

    def compile_held() -> Any:
        # REACHED ONLY ON A MEMO MISS -- after ``warm()`` that is a policy change or a
        # first launch with no warm. The module's own ``cp`` where it has one, so a
        # harness that substituted it is honoured; the H/D families import CuPy inside
        # their getter and have no module attribute, and neither does this.
        cp = getattr(module, "cp", None)
        if cp is None:
            if hasattr(module, "cp"):
                raise RuntimeError(
                    f"{module.__name__} cannot compile {name!r} here: CuPy is not "
                    f"importable on this host")
            import cupy as cp  # noqa: PLC0415 - device-only, reached on a miss
        return cp.RawKernel(source, name, options=options)

    return _compile_cache.get_or_compile(key, compile_held)


def _held_arm_kernel(held: Dict[str, Any], module: Any,
                     emit: Callable[[Any], str], arm: Any) -> Any:
    """:func:`_held_kernel` under the ARM-KEYED getters' key, copied verbatim.

    The complex, complex-folded, complex-beta and complex-Dcyl pairs all key
    ``kernel_cache_key(f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
    _COMPILE_OPTIONS, <product>_source(arm))`` and compile the bare ``KERNEL_NAME``
    (``complex_fused_magnetic_pair.py:622-626`` and its seven siblings). The held
    shape is the RAW arm the getter would have been handed, so a second spelling of
    one arm is a second emission rather than one string standing in for another.
    """
    key_name = (f"{module.KERNEL_NAME}_arm"
                f"{module.complex_emitter.normalized_expansion(arm)}")
    return _held_kernel(held, module, (arm,), lambda: emit(arm), key_name, True,
                        module.KERNEL_NAME)


def _held_dispersive_kernel(held: Dict[str, Any], module: Any, counts: Any) -> Any:
    """:func:`_held_kernel` under ``dispersive_fused_electric_pair._get_kernel``'s key.

    The launcher canonicalizes the pole arity BEFORE its getter sees it
    (``dispersive_fused_electric_pair.py:883``,
    ``counts = dispersive_kernels.normalized_pole_counts(counts)``), so the held shape
    is that same call's answer and the emitter is handed it -- one canonical triple,
    one string, exactly the input the getter's own emission would have had
    (``dispersive_fused_electric_pair.py:690-691``, ``True`` on the key).
    """
    shape = module.dispersive_kernels.normalized_pole_counts(counts)
    return _held_kernel(held, module, shape,
                        lambda: module.dispersive_fused_electric_pair_source(shape),
                        module.KERNEL_NAME, True, module.KERNEL_NAME)


def _held_no_pml_dispersive_kernel(held: Dict[str, Any], module: Any, cond: Any,
                                   counts: Any) -> Any:
    """:func:`_held_kernel` under the no-absorber dispersive pair's getter's key.

    The mask is handed to the getter unchanged and the arity canonicalized first
    (``no_pml_dispersive_fused_electric_pair.py:1038``), so the held shape is the
    pair ``(mask, canonical arity)`` and the key is ``(KERNEL_NAME, True, ...)`` over
    ``no_pml_dispersive_fused_electric_pair_source(cond, counts)`` (:779-780).
    """
    canonical = module.dispersive_kernels.normalized_pole_counts(counts)
    return _held_kernel(
        held, module, (tuple(cond), canonical),
        lambda: module.no_pml_dispersive_fused_electric_pair_source(cond, canonical),
        module.KERNEL_NAME, True, module.KERNEL_NAME)


def _held_offdiag_kernel(held: Dict[str, Any], module: Any, row_mask: Any,
                         expansion: Any = None) -> Any:
    """:func:`_held_kernel` under an off-diagonal weld's getter's key.

    ONLY THE TEXT IS HELD, NEVER THE RESOLUTION: these resolvers hand out the scratch
    the launch rotates and read the row volumes fresh (see :func:`_resolved_once`).
    The row mask is resolved per launch, as before, and handed to the getter
    unchanged, so the held shape is that mask (and the arm, on the complex two).
    Keys copied from the four getters: the real ones ``(KERNEL_NAME, False,
    _COMPILE_OPTIONS, kernel_source(row_mask))`` (``offdiag_fused_electric_pair.py
    :572-573``, ``folded_offdiag_fused_electric_pair.py:575-576``), the complex ones
    ``(f"{KERNEL_NAME}_arm{normalized_expansion(expansion)}", True, ...,
    kernel_source(row_mask, expansion))`` (``folded_complex_offdiag_fused_electric_pair
    .py:402-405``, ``complex_no_pml_offdiag_fused_electric_pair.py:359-362``).
    """
    if expansion is None:
        return _held_kernel(held, module, (tuple(row_mask),),
                            lambda: module.kernel_source(row_mask),
                            module.KERNEL_NAME, False, module.KERNEL_NAME)
    key_name = (f"{module.KERNEL_NAME}_arm"
                f"{module.complex_emitter.normalized_expansion(expansion)}")
    return _held_kernel(held, module, (tuple(row_mask), expansion),
                        lambda: module.kernel_source(row_mask, expansion),
                        key_name, True, module.KERNEL_NAME)


def _resolved_once(held: Dict[str, Any],
                   resolve: Callable[[Any], Dict[str, Any]]
                   ) -> Callable[[Any], Dict[str, Any]]:
    """``resolve`` asked at the first launch and its answer held for the plan's life.

    FOR A RESOLVER WHOSE EVERY ENTRY IS A PURE READING OF THE FROZEN CONFIGURATION, and
    only for one: the tables, the boundary codes, the walls, the fills, the licence's
    arm and ``dt/dx`` are all fixed at the freeze that built this plan
    (``driver.py:3266-3271``), which is what the tables' own docstrings already
    promised ("Called ONCE per frozen configuration", ``fused_magnetic_pair.py:1480``).
    Measured 2026-09-19 on the laptop with a stub CuPy and NumPy fields: the plain
    pairs' resolver body costs 17.8 us per call and the held answer 0.08 us, and before
    this it ran on every launch for arguments that never change.

    NOT A PROPERTY OF :meth:`CudaFusedPairPlan.run`, deliberately. A resolver that
    hands out state which ROTATES between launches -- the off-diagonal welds' scratch
    (:func:`_offdiag_fused_electric_pair_plan` and its three siblings, which also read
    ``offdiag_row_volumes(ctx.fields)`` fresh) and ``special_kz_fused_hd_pair``'s
    ``dict(held)`` -- would hand its second launch the first launch's buffers, which
    the rotation has since made live fields. So each builder opts in by name and
    ``run`` still asks the resolver on every launch.
    """
    def once(ctx: Any) -> Dict[str, Any]:
        arguments = held.get("arguments")
        if arguments is None:
            arguments = held["arguments"] = resolve(ctx)
        return arguments

    return once


def _fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_fused_magnetic_pair`` as a ``Coverage``, imported at call time.

    THE REASON IS WRAPPED IN A 1-TUPLE for the reason ``arms.coverage_adapter``
    spells out: the shared normaliser iterates ``.reasons``, and a bare string is
    iterable, so a single-string refusal would arrive as one "reason" PER CHARACTER.
    Measured on this tree: ``"no PML"`` came back as six prefixed one-character
    reasons.
    """
    from .fused_magnetic_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        covers_fused_magnetic_pair,
    )

    covered, reason = covers_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the certified B/H product, built without a device.

    ``resolve_launch_args`` is a CLOSURE and is not called here, exactly as
    ``arms.CudaSlotPlan`` documents for the slot arms: resolving the tables imports
    the kernel module and needs CuPy, and the composition has to be buildable on the
    host that measured it.
    """
    from .fused_magnetic_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .fused_magnetic_pair import (  # noqa: PLC0415
            fused_magnetic_pair_fills, fused_magnetic_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        # THREE READINGS OF THE GRID, AND NO ONE OF THEM IS THE OTHERS. The
        # boundary codes are the curl's ghost-rule resolution (where a folded
        # METALLIC axis is indistinguishable from a plain wall); the walls are
        # ``_zero_metal``'s question, with a folded axis EXCLUDED; the fills are
        # the two mirror passes', where a folded metallic axis DOES carry the near
        # fill and does NOT carry the far one.
        return {"tables": fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": fused_magnetic_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import fused_magnetic_pair as family  # noqa: PLC0415

        # ``fused_magnetic_pair._get_kernel``'s key: (KERNEL_NAME, False, ...).
        kernel = _held_kernel(held, family, (), family.fused_magnetic_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["fills"], arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_fused_hd_pair`` as a ``Coverage``, imported at call time.

    THE ONLY PREDICATE IN THIS FILE ASKED ABOUT THE H->D SEAM, and the difference from
    every sibling above is what sits between its two consults. Nothing is INJECTED
    there, so it routes its source question through
    ``withdraw_hoist.seam_withdraw_reasons`` rather than through
    ``deposit_repair.seam_source_reasons``: what the driver runs between ``update_H``
    and ``step_D`` is the electric integrated-source WITHDRAW loop
    (driver.py:3313-3314). A copy of an electric closure with the module name changed
    would have asked ``deposit_repair`` about a seam with no deposit in it.
    """
    from .fused_hd_pair import covers_fused_hd_pair  # noqa: PLC0415

    covered, reason = covers_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the H->D weld, built without a device.

    THE SCRATCH IS RESOLVED IN THE CLOSURE, NOT HERE, and that is the same rule every
    plan builder on this table follows: allocating six device volumes needs CuPy and
    the composition has to be buildable on the host that measured it. The rotation
    happens inside :func:`.fused_hd_pair.run_fused_hd_pair`'s own launcher, which is
    reached through the ``launch`` closure below, so the plan carries the scratch
    between launches rather than re-allocating it.
    """
    from .fused_hd_pair import FAMILY, KERNEL_NAME, REPLACES  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .fused_hd_pair import (fused_hd_pair_codes,  # noqa: PLC0415
                                    fused_hd_pair_scratch, fused_hd_pair_tables)

        if "tables" not in held:
            held["tables"] = fused_hd_pair_tables(ctx.pml)
            held["codes"] = fused_hd_pair_codes(ctx.grid)
            held["scratch"] = fused_hd_pair_scratch(ctx.fields)
        return {"tables": held["tables"], "boundary_codes": held["codes"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import fused_hd_pair as family  # noqa: PLC0415

        # ``fused_hd_pair._get_kernel``'s key: (KERNEL_NAME, False, ...) over
        # ``kernel_source()``. The resolver above already holds its tables and scratch
        # and is left as it was: the scratch it allocates is read from ``held``
        # directly, never through the resolved dict, so nothing there can go stale.
        family.assert_scratch_is_disjoint(fields, held["scratch"], arguments["tables"])
        kernel = _held_kernel(held, family, (), family.kernel_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        record = family.launch_fused_hd_pair(
            fields, held["scratch"], arguments["tables"], arguments["boundary_codes"],
            arguments["dtdx"], kernel=kernel)
        held["scratch"] = family.rotate_into_fields(fields, held["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY, label="fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _cylindrical_real_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_real_fused_hd_pair`` as a ``Coverage``, at call time.

    Routes its source question through ``withdraw_hoist.seam_withdraw_reasons`` like
    ``_fused_hd_pair_coverage`` (nothing is INJECTED in the H->D seam).
    """
    from .cylindrical_real_fused_hd_pair import (  # noqa: PLC0415
        covers_cylindrical_real_fused_hd_pair,
    )

    covered, reason = covers_cylindrical_real_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_real_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the real m = 0 Dcyl H->D product, built without a device.

    THE STATE IS RESOLVED IN THE CLOSURE (six scratch twins, the increment and prefix
    buffers, the array path's own cached row vectors) and carried between launches;
    ``run_cylindrical_real_fused_hd_pair`` performs launch 1, the ``xp.cumsum`` on the
    array path, the rotation and launch 2 -- three launches per run, counted.
    """
    from .cylindrical_real_fused_hd_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .cylindrical_real_fused_hd_pair import resolve as resolve_state  # noqa: PLC0415

        if "state" not in held:
            held["state"] = resolve_state(ctx.fields, ctx.grid, ctx.pml)
        return {"state": held["state"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_real_fused_hd_pair as family  # noqa: PLC0415

        # HALF OF THIS PRODUCT'S PER-LAUNCH HOST WORK IS REACHABLE FROM HERE. The held
        # text feeds launch 1 (``run_cylindrical_real_fused_hd_pair`` threads
        # ``kernel=`` to ``launch_constitutive_and_increment`` only); launch 2 is the
        # certified ``cyl_step_D_pml_real`` through its own launcher, and ``run_``
        # re-asks the coverage predicate on every call. Both are the certified
        # module's and are left as they are.
        kernel = _held_kernel(held, family, (), family.kernel_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.run_cylindrical_real_fused_hd_pair(
            fields, fields.grid, context.pml, sources=context.sources,
            state=arguments["state"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="cylindrical real fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _cylindrical_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_fused_hd_pair`` as a ``Coverage``, at call time."""
    from .cylindrical_fused_hd_pair import (  # noqa: PLC0415
        covers_cylindrical_fused_hd_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_cylindrical_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex Dcyl H->D product, built without a device."""
    from .cylindrical_fused_hd_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .cylindrical_fused_hd_pair import resolve as resolve_state  # noqa: PLC0415

        if "state" not in held:
            held["state"] = resolve_state(ctx.fields, ctx.grid, ctx.pml,
                                          ctx.license_for(LICENSE_COMPLEX)["arm"])
        return {"state": held["state"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_fused_hd_pair as family  # noqa: PLC0415

        # Launch 1's text is held under its getter's key, ``(f"{KERNEL_NAME}_arm{arm}",
        # True, _COMPILE_OPTIONS, kernel_source(arm))`` with ``arm = _arm(expansion)``
        # (cylindrical_fused_hd_pair.py:531-538); launch 2 is the certified curl
        # through its own launcher, as in the real twin.
        expansion = arguments["state"]["arm"]
        arm = family._arm(expansion)  # noqa: SLF001 - the getter's own normalization
        kernel = _held_kernel(held, family, (expansion,),
                              lambda: family.kernel_source(arm),
                              f"{family.KERNEL_NAME}_arm{arm}", True,
                              family.KERNEL_NAME)
        return family.run_cylindrical_fused_hd_pair(
            fields, fields.grid, context.pml, expansion,
            sources=context.sources, license=context.license_for(LICENSE_COMPLEX),
            subnormal_policy=context.subnormal_policy, state=arguments["state"],
            kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="cylindrical complex fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _special_kz_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_special_kz_fused_hd_pair`` as a ``Coverage``, imported at call time.

    NO LICENCE ARGUMENT AND THAT IS THE STORAGE CLASS SPEAKING, not an omission: this
    family is REAL float32 throughout, so there is no complex expansion arm to be
    licensed and its predicate takes none. Its source question goes through
    ``withdraw_hoist.seam_withdraw_reasons`` for the reason
    ``_fused_hd_pair_coverage``'s does: nothing is INJECTED between the ``update_H``
    and ``step_D`` consults, so what sits there is the electric integrated-source
    WITHDRAW loop (driver.py:3313-3314) and ``deposit_repair`` has nothing to say
    about it.
    """
    from .special_kz_fused_hd_pair import (  # noqa: PLC0415
        covers_special_kz_fused_hd_pair,
    )

    covered, reason = covers_special_kz_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _special_kz_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the REAL beta H->D weld, built without a device.

    THE SCRATCH IS RESOLVED IN THE CLOSURE, not here: allocating six device volumes
    needs CuPy and the composition has to be buildable on the host that measured it.
    THE ROTATION HAPPENS INSIDE THE LAUNCH and the retired pair is kept as the next
    launch's scratch, which is the choreography every scratch-output weld on this
    track ships.
    """
    from .special_kz_fused_hd_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}
    #: The held text, in a dict of its own because ``resolve`` copies ``held`` whole.
    sources_held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .special_kz_fused_hd_pair import (  # noqa: PLC0415
            beta_scalars, special_kz_fused_hd_pair_codes,
            special_kz_fused_hd_pair_scratch, special_kz_fused_hd_pair_tables,
        )

        if "tables" not in held:
            held["tables"] = special_kz_fused_hd_pair_tables(ctx.pml)
            held["codes"] = special_kz_fused_hd_pair_codes(ctx.grid)
            held["beta"] = beta_scalars(ctx.grid)
            held["scratch"] = special_kz_fused_hd_pair_scratch(ctx.fields)
        return dict(held)

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import special_kz_fused_hd_pair as family  # noqa: PLC0415

        # THE SOURCE IS HELD AND THE RESOLUTION IS NOT: ``resolve`` returns
        # ``dict(held)`` INCLUDING the scratch this launch rotates, so a resolution held
        # across launches would hand the second launch the FIRST launch's scratch --
        # which the rotation below has made live fields. The held text sits beside it
        # under its own key and never enters that dict.
        # ``special_kz_fused_hd_pair._get_kernel``'s key: (KERNEL_NAME, False,
        # _COMPILE_OPTIONS, kernel_source()) -- its ``options`` door left at default.
        dtdx = float(context.grid.dt / context.grid.dx)
        family.assert_scratch_is_disjoint(fields, arguments["scratch"],
                                          arguments["tables"])
        kernel = _held_kernel(sources_held, family, (), family.kernel_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        record = family.launch_special_kz_fused_hd_pair(
            fields, arguments["scratch"], arguments["tables"], arguments["codes"],
            arguments["beta"], dtdx, kernel=kernel)
        held["scratch"] = family.rotate_into_fields(fields, arguments["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY, label="special_kz fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _complex_beta_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_beta_fused_hd_pair`` as a ``Coverage``, at call time."""
    from .complex_beta_fused_hd_pair import (  # noqa: PLC0415
        covers_complex_beta_fused_hd_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_beta_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _conductive_bfast_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_conductive_bfast_fused_hd_pair`` as a ``Coverage``, at call time.

    Routes its source question through ``withdraw_hoist.seam_withdraw_reasons`` like
    ``_fused_hd_pair_coverage`` (nothing is INJECTED in the H->D seam), and resolves
    its VARIANT from the run before asking either half -- so the reason a refusal
    carries names the tail the run would have taken.
    """
    from .conductive_bfast_fused_hd_pair import (  # noqa: PLC0415
        covers_conductive_bfast_fused_hd_pair,
    )

    covered, reason = covers_conductive_bfast_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _conductive_bfast_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the conductive/BFAST H->D product, built without a device.

    ONE LAUNCH, and the STATE IS RESOLVED IN THE CLOSURE (the shared coefficient
    group, the boundary triple, the BFAST scalars on that variant, and the six scratch
    twins) and carried between launches, because the rotation hands the retired pair
    forward as the next launch's scratch.

    ``KERNEL_NAME`` is the CONDUCTIVE variant's entry point and the label is the
    family's; which of the two device strings a run compiles is decided inside
    ``run_conductive_bfast_fused_hd_pair`` from the run itself, never here.
    """
    from .conductive_bfast_fused_hd_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .conductive_bfast_fused_hd_pair import resolve as resolve_state  # noqa: PLC0415

        if "state" not in held:
            held["state"] = resolve_state(ctx.fields, ctx.grid, ctx.pml)
        return {"state": held["state"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import conductive_bfast_fused_hd_pair as family  # noqa: PLC0415

        # THE COSTLIEST EMISSION ON THIS TABLE: 4601 us per getter call, measured
        # 2026-09-19 (laptop, stub CuPy, 200 calls). The getter's key is
        # ``(kernel_name(variant), False, _COMPILE_OPTIONS, current_source(variant,
        # cond))`` (conductive_bfast_fused_hd_pair.py:995-997), and ``current_source``
        # honours an override standing at the first launch. ``run_`` threads
        # ``kernel=`` to the one launch (:1435-1436) and still re-asks its predicate.
        state = arguments["state"]
        variant, cond = state["variant"], state["cond"]
        kernel = _held_kernel(held, family, (variant, cond),
                              lambda: family.current_source(variant, cond),
                              family.kernel_name(variant), False,
                              family.kernel_name(variant))
        return family.run_conductive_bfast_fused_hd_pair(
            fields, fields.grid, context.pml, sources=context.sources,
            state=state, kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="conductive/BFAST fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _complex_beta_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the COMPLEX beta H->D weld, built without a device.

    ONE TEXT AND NO VARIANT, unlike the Cartesian complex sibling: the beta family's
    device source IS the folded complex curl with the insert, so a fold is a runtime
    boundary code and ``resolve`` needs nothing from the grid but its codes, phases
    and beta words.
    """
    from .complex_beta_fused_hd_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_beta_fused_hd_pair import resolve as resolve_state  # noqa: PLC0415

        if "state" not in held:
            held["state"] = resolve_state(
                ctx.fields, ctx.grid, ctx.pml,
                ctx.license_for(LICENSE_COMPLEX)["arm"])
        return {"state": held["state"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .complex_beta_fused_hd_pair import (  # noqa: PLC0415
            run_complex_beta_fused_hd_pair,
        )

        return run_complex_beta_fused_hd_pair(
            fields, context.grid, context.pml, arguments["state"]["arm"],
            sources=context.sources, license=context.license_for(LICENSE_COMPLEX),
            subnormal_policy=context.subnormal_policy, state=arguments["state"],
            check=False)

    return CudaFusedPairPlan(family=FAMILY, label="complex beta fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _complex_fused_hd_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_fused_hd_pair`` as a ``Coverage``, imported at call time.

    THE SECOND PREDICATE IN THIS FILE ASKED ABOUT THE CARTESIAN H->D SEAM, and it
    routes its source question through ``withdraw_hoist.seam_withdraw_reasons`` for
    the reason ``_fused_hd_pair_coverage`` does: nothing is INJECTED between the
    ``update_H`` and ``step_D`` consults, so what sits there is the electric
    integrated-source WITHDRAW loop (driver.py:3313-3314) and ``deposit_repair`` has
    nothing to say about it.

    ONE PRODUCT, TWO BOARD CELLS. The predicate reads the fold off the grid and asks
    the certified curl predicate the variant selects -- ``covers_real_pml_complex_curl``
    on an unfolded grid, ``complex_folded_kernels.covers_complex_folded_curl`` on a
    folded one -- so the arm row below carries the constitutive label on both slots
    and the board resolves the two cells through the predicate rather than through a
    second column.
    """
    from .complex_fused_hd_pair import covers_complex_fused_hd_pair  # noqa: PLC0415
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_fused_hd_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_fused_hd_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex Cartesian H->D weld, built without a device.

    THE SCRATCH AND THE VARIANT ARE RESOLVED IN THE CLOSURE, not here: allocating six
    device volumes needs CuPy and the composition has to be buildable on the host that
    measured it. ``complex_fused_hd_pair.resolve`` reads the variant off the grid, so
    the text that would be compiled and the predicate that admitted it are decided by
    the same reading.
    """
    from .complex_fused_hd_pair import FAMILY, KERNEL_NAME, REPLACES  # noqa: PLC0415
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_fused_hd_pair import resolve as resolve_state  # noqa: PLC0415

        if "state" not in held:
            held["state"] = resolve_state(
                ctx.fields, ctx.grid, ctx.pml,
                ctx.license_for(LICENSE_COMPLEX)["arm"])
        return {"state": held["state"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .complex_fused_hd_pair import (  # noqa: PLC0415
            run_complex_fused_hd_pair,
        )

        return run_complex_fused_hd_pair(
            fields, fields.grid, context.pml, arguments["state"]["arm"],
            sources=context.sources, license=context.license_for(LICENSE_COMPLEX),
            subnormal_policy=context.subnormal_policy, state=arguments["state"])

    return CudaFusedPairPlan(family=FAMILY, label="complex fused H/D pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_H", "step_D"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_fused_electric_pair`` as a ``Coverage``, imported at call time.

    THE ONLY PREDICATE IN THIS FILE ASKED ABOUT THE ELECTRIC SEAM, and the difference
    is not cosmetic: it routes its source question through
    ``deposit_repair.seam_source_reasons(..., 'D')``, which selects the sources the
    driver injects between ``step_D`` and ``update_E``. A copy of the magnetic
    closure with the module name changed would have asked about ``'B'`` and admitted
    every electric deposit -- the exact defect the shared clause was introduced to
    prevent, recorded in ``fused_magnetic_pair``'s own preamble.
    """
    from .fused_electric_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        covers_fused_electric_pair,
    )

    covered, reason = covers_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the real D/E product, built without a device.

    ``resolve_launch_args`` is a CLOSURE and is not called here, exactly as
    ``arms.CudaSlotPlan`` documents for the slot arms: resolving the tables imports
    the kernel modules and needs CuPy, and the composition has to be buildable on the
    host that measured it.
    """
    from .fused_electric_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .fused_electric_pair import (  # noqa: PLC0415
            fused_electric_pair_fills, fused_electric_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        # THREE READINGS OF THE GRID, AND NO ONE OF THEM IS THE OTHERS. The boundary
        # codes are the curl's ghost-rule resolution (where a folded METALLIC axis is
        # indistinguishable from a plain wall); the walls are ``_zero_metal``'s
        # question, with a folded axis EXCLUDED; the fills are the two mirror passes',
        # where a folded metallic axis DOES carry the near fill and does NOT carry the
        # far one. The fill reading joined on 2026-08-31 with the carry.
        return {"tables": fused_electric_pair_tables(ctx.pml),
                "boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": fused_electric_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import fused_electric_pair as family  # noqa: PLC0415

        # ``fused_electric_pair._get_kernel``'s key passes ``True`` on this
        # real-storage kernel (fused_electric_pair.py:1265), and so does this one.
        kernel = _held_kernel(held, family, (), family.fused_electric_pair_source,
                              family.KERNEL_NAME, True, family.KERNEL_NAME)
        return family.launch_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["fills"], arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _dispersive_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_dispersive_fused_electric_pair`` as a ``Coverage``, at call time.

    THE DISPERSIVE TWIN of ``_fused_electric_pair_coverage`` and asked the same way:
    its source question routes through ``deposit_repair.seam_source_reasons(..., 'D')``
    (inside the predicate it delegates to), which selects the sources the driver
    injects between ``step_D`` and ``update_E``. Every one of the seven corpus rows
    this product exists for carries such a deposit, so the routing is the product.
    """
    from .dispersive_fused_electric_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        covers_dispersive_fused_electric_pair,
    )

    covered, reason = covers_dispersive_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _dispersive_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the dispersive D/E product, built without a device.

    ``resolve_launch_args`` is a CLOSURE and is not called here, exactly as
    ``arms.CudaSlotPlan`` documents for the slot arms: resolving the tables imports
    the kernel modules and needs CuPy, and the composition has to be buildable on the
    host that measured it.

    THE POLE PLAN IS RESOLVED INSIDE ``launch``, NOT IN ``resolve``, and that is the
    one place this closure differs from the real twin's.
    ``PolarizationState.update`` rotates ``P``/``P_prev``/scratch every step
    (dispersion.py:689-691), so a chain resolved once at composition time would name
    the first step's arrays forever -- stale in a way that still computes, which is
    the worst kind. ``resolve`` is called once per frozen configuration; ``launch``
    is called every step.
    """
    from .dispersive_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .fused_electric_pair import (  # noqa: PLC0415
            fused_electric_pair_fills, fused_electric_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        # THE SAME THREE READINGS OF THE GRID the real twin takes, from the same
        # module: the boundary codes are the curl's ghost-rule resolution (where a
        # folded METALLIC axis is indistinguishable from a plain wall); the walls are
        # ``_zero_metal``'s question, with a folded axis EXCLUDED; the fills are the
        # two mirror passes'. Sharing them is deliberate -- the carry is one object.
        return {"tables": fused_electric_pair_tables(ctx.pml),
                "boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": fused_electric_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import dispersive_fused_electric_pair as family  # noqa: PLC0415

        # THE ARITY IS READ PER LAUNCH, AS BEFORE, and only the text it selects is held
        # -- one emission per distinct canonical arity, never one arity standing in for
        # another.
        poles, counts = family.pole_bindings(fields)
        kernel = _held_dispersive_kernel(held, family, counts)
        return family.launch_dispersive_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            poles, counts, kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="dispersive fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _no_pml_dispersive_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_no_pml_dispersive_fused_electric_pair`` as a ``Coverage``.

    Its source question routes through ``deposit_repair.seam_source_reasons(..., 'D')``
    carrying ``repair_paths=REPAIR_PATHS`` -- the PLAIN repair, not the default -- so
    the recurrence the clause asks about is the one ``update_E`` actually runs on an
    inert layer. All three corpus rows this product exists for carry an electric
    deposit, so the routing IS the product.
    """
    from .no_pml_dispersive_fused_electric_pair import (  # noqa: PLC0415
        covers_no_pml_dispersive_fused_electric_pair,
    )

    covered, reason = covers_no_pml_dispersive_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _no_pml_dispersive_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the no-absorber dispersive D/E product.

    ``resolve_launch_args`` is a CLOSURE and is not called here, exactly as
    ``arms.CudaSlotPlan`` documents: resolving the boundary codes imports the kernel
    modules and needs CuPy, and the composition has to be buildable on the host that
    measured it.

    THE POLE PLAN AND THE CONDUCTIVITY MASK ARE RESOLVED INSIDE ``launch``, NOT IN
    ``resolve``. ``PolarizationState.update`` rotates ``P``/``P_prev``/scratch every
    step (dispersion.py:689-691), so a chain resolved once at composition time would
    name the first step's arrays forever -- stale in a way that still computes. The
    mask travels with it because the two are read from the same ``fields`` and a
    kernel compiled for one mask cannot be handed another's bindings.
    """
    from .no_pml_dispersive_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        # THE SAME TWO READINGS OF THE GRID the real twin takes, from the same
        # modules: the boundary codes are the curl's ghost-rule resolution and the
        # walls are ``_zero_metal``'s question. There is no fill plan here because
        # this family refuses a mirror plane.
        return {"boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import no_pml_dispersive_fused_electric_pair as family  # noqa: PLC0415

        # The mask and the arity are read per launch, as before; the text they select
        # is held per distinct pair of them.
        poles, counts = family.pole_bindings(fields)
        conductivity, cond = family.conductive_bindings(fields)
        kernel = _held_no_pml_dispersive_kernel(held, family, cond, counts)
        return family.launch_no_pml_dispersive_fused_electric_pair(
            fields, arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], poles, counts, conductivity, cond, kernel=kernel)

    return CudaFusedPairPlan(
        family=FAMILY, label="no-PML dispersive fused electric pair",
        kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
        slots=("step_D", "update_E"), context=context,
        # Pure readings of the frozen grid: held.
        resolve_launch_args=_resolved_once(held, resolve), launch=launch)


def _conductive_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_conductive_fused_electric_pair`` as a ``Coverage``, at call time."""
    from .conductive_fused_electric_pair import (  # noqa: PLC0415
        covers_conductive_fused_electric_pair,
    )

    covered, reason = covers_conductive_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _conductive_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the conductive D/E product, built without a device.

    THE CONDUCTIVITY MASK IS RESOLVED INSIDE ``launch``, and that is the one place
    this closure differs from the real twin's: the mask decides which device string
    was compiled, and the three ``f_cond`` histories it selects are ``Fields``
    attributes that a configuration freeze does not pin.
    """
    from .conductive_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .conductive_fused_electric_pair import (  # noqa: PLC0415
            conductive_fused_electric_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        return {"tables": conductive_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import conductive_fused_electric_pair as family  # noqa: PLC0415

        # The mask is read per launch, as before, and handed to the getter unchanged
        # (conductive_fused_electric_pair.py:975); the text it selects is held per
        # distinct mask under the getter's key, ``(KERNEL_NAME, True, ...)`` (:718).
        conductivity, histories, cond = family.conductive_bindings(fields)
        kernel = _held_kernel(
            held, family, tuple(cond),
            lambda: family.conductive_fused_electric_pair_source(cond),
            family.KERNEL_NAME, True, family.KERNEL_NAME)
        return family.launch_conductive_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["dtdx"], conductivity, histories, cond,
            kernel=kernel)

    return CudaFusedPairPlan(
        family=FAMILY, label="conductive fused electric pair",
        kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
        slots=("step_D", "update_E"), context=context,
        # Pure readings of the frozen pml/grid: held. The mask is not among them.
        resolve_launch_args=_resolved_once(held, resolve), launch=launch)


def _complex_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_fused_magnetic_pair`` as a ``Coverage``, imported at call time.

    THE LICENCE IS READ OFF THE CONTEXT, not defaulted. ``StepContext.license_for``
    is what the licensed ARMS are bound through (``arms._bind_licensed_curl``), and a
    fused product that asked its halves with ``license=None`` would get that family's
    own named refusal on every configuration -- which reads as a claim about the run
    rather than about the caller.
    """
    from .complex_fused_magnetic_pair import (  # noqa: PLC0415
        covers_complex_fused_magnetic_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex B/H product, built without a device."""
    from .complex_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_fused_magnetic_pair import (  # noqa: PLC0415
            complex_fused_magnetic_pair_tables,
        )
        from .complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments, complex_boundary_codes,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        # THE ARM IS THE LICENCE'S, and it is resolved HERE rather than defaulted:
        # both arms compile, both run, and they differ in the last bits of about a
        # quarter of the words, so a wrong arm is a wrong ANSWER rather than a crash.
        flags, values = bloch_phase_arguments(ctx.grid, False)
        return {"tables": complex_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": complex_boundary_codes(ctx.grid),
                "phase_flags": flags, "phase_values": values,
                "walls": zero_metal_axes(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_fused_magnetic_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(held, family, family.complex_fused_magnetic_pair_source,
                                  arguments["arm"])
        return family.launch_complex_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="complex fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _cylindrical_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_fused_magnetic_pair`` as a ``Coverage``, at call time."""
    from .cylindrical_fused_magnetic_pair import (  # noqa: PLC0415
        covers_cylindrical_fused_magnetic_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_cylindrical_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the Dcyl complex B/H product, built without a device.

    THE RADIAL PREFIX IS RESOLVED PER LAUNCH AND NOT PER CONFIGURATION, which is the
    one place this plan diverges from its two siblings: ``cylindrical_prefix`` is a
    cumulative sum over the CURRENT Ep, so a value cached beside the coefficient
    tables would be a stale field one step later -- the same class of defect as a
    fused pair reading a pre-injection B, arriving through a cache instead of a seam.
    """
    from .cylindrical_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .cylindrical_complex_kernels import (  # noqa: PLC0415
            axis_increment_scalars, cylindrical_complex_boundary_codes,
            imr_rows_for, m_class, zero_rows,
        )
        from .cylindrical_fused_magnetic_pair import (  # noqa: PLC0415
            cylindrical_fused_magnetic_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        dtdx = float(ctx.grid.dt / ctx.grid.dx)
        m = int(ctx.grid.m)
        return {"tables": cylindrical_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": cylindrical_complex_boundary_codes(ctx.grid),
                "imr_rows": imr_rows_for("step_B", ctx.grid.xp, m, dtdx,
                                         int(ctx.fields.Bx.shape[0]),
                                         ctx.fields.Bx.dtype),
                "increment_scalars": axis_increment_scalars(m, dtdx),
                "m_class": m_class(m),
                "zero_rows": zero_rows(
                    m, bool(ctx.grid.accurate_fields_near_cylorigin)),
                "walls": zero_metal_axes(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": dtdx}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_fused_magnetic_pair as family  # noqa: PLC0415
        from .cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.cylindrical_fused_magnetic_pair_source, arguments["arm"])
        return family.launch_cylindrical_fused_magnetic_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_B"), arguments["imr_rows"],
            arguments["boundary_codes"], arguments["increment_scalars"],
            arguments["m_class"], arguments["zero_rows"], arguments["walls"],
            arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="cylindrical fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _complex_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_fused_electric_pair`` as a ``Coverage``, at call time.

    THE SEAM ARGUMENT IS ``'D'`` INSIDE THAT PREDICATE and that is the whole reason
    this closure is not the complex MAGNETIC one with a module name changed: the
    magnetic closure's predicate routes its source question through
    ``deposit_repair.seam_source_reasons(..., 'B')``, which would have admitted every
    electric deposit -- the exact defect the shared clause was introduced to prevent.
    """
    from .complex_fused_electric_pair import (  # noqa: PLC0415
        covers_complex_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex D/E product, built without a device."""
    from .complex_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_fused_electric_pair import (  # noqa: PLC0415
            complex_fused_electric_pair_tables,
        )
        from .complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments, complex_boundary_codes,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        # ``True`` IS THE BACKWARD FLAG and it is the one argument here a copy of the
        # magnetic resolver would get silently wrong: ``step_D`` takes ``cshift_dn``
        # and the CONJUGATED Bloch factor (``complex_emitter.KERNELS['step_D'][1]``,
        # ``stepping._shift_down``:1818-1822). The forward table compiles, runs, and is
        # the wrong phase on every phased row.
        flags, values = bloch_phase_arguments(ctx.grid, True)
        return {"tables": complex_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": complex_boundary_codes(ctx.grid),
                "phase_flags": flags, "phase_values": values,
                "walls": zero_metal_axes(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_fused_electric_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(held, family, family.complex_fused_electric_pair_source,
                                  arguments["arm"])
        return family.launch_complex_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="complex fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _no_pml_complex_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_no_pml_complex_fused_electric_pair`` as a ``Coverage``, at call time.

    THE ONLY PRODUCT IN THIS FILE THAT REFUSES AN ACTIVE ABSORBER, which is exactly
    what keeps ``step_D`` a partitioned seam now that it carries four candidates: the
    other three each refuse an INACTIVE layer by name (``coverage.py:1141``,
    ``coverage._complex_grid_refusal``:2785, ``cylindrical_coverage``'s own copy) and
    this one requires it. Two admitters leave the seam UNFUSED naming both, so the
    disjointness is what buys the slots.

    THE LICENCE IS ``LICENSE_COMPLEX_NO_PML``, not ``LICENSE_COMPLEX``. Its halves are
    the ``cuda_complex_no_pml`` family's, and that family's own arms are bound through
    that licence (``registry._TABLE``); asking with the PML complex one would be asking
    a different verdict about a different arm.
    """
    from .no_pml_complex_fused_electric_pair import (  # noqa: PLC0415
        covers_no_pml_complex_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    covered, reason = covers_no_pml_complex_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX_NO_PML), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _no_pml_complex_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex NO-ABSORBER D/E product, built without a device.

    THE CURL ARM IS RESOLVED HERE, and this is the one resolver in this file that has
    to. The certified no-absorber curl ships TWO kernels -- a plain tail and a
    conductive one -- and which a run takes is ``fields.condfac_for``'s answer, read by
    the certified family's own classifier. Every corpus row on this product's cell
    takes the CONDUCTIVE one, so a resolver that defaulted to the plain arm would serve
    the whole cell with the wrong arithmetic rather than serving nothing.

    NO ``tables`` AND NO ``walls`` KEY: this pair reads no coefficient vector (there is
    no absorber to have one) and carries no wall pass (its predicate refuses every
    walled run). A resolver that carried an empty one anyway would suggest the launch
    binds something it does not.
    """
    from .no_pml_complex_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments, complex_boundary_codes,
        )
        from .no_pml_complex_fused_electric_pair import (  # noqa: PLC0415
            no_pml_complex_curl_arm,
        )

        # ``True`` IS THE BACKWARD FLAG and it is the one argument here a copy of a
        # magnetic resolver would get silently wrong: ``step_D`` takes ``cshift_dn``
        # and the CONJUGATED Bloch factor (``complex_emitter.KERNELS['step_D'][1]``,
        # ``stepping._shift_down``:1818-1822). The forward table compiles, runs, and is
        # the wrong phase on every phased row.
        flags, values = bloch_phase_arguments(ctx.grid, True)
        return {"boundary_codes": complex_boundary_codes(ctx.grid),
                "phase_flags": flags, "phase_values": values,
                "curl_arm": no_pml_complex_curl_arm(ctx.fields, "step_D"),
                "expansion": ctx.license_for(LICENSE_COMPLEX_NO_PML)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import no_pml_complex_fused_electric_pair as family  # noqa: PLC0415

        # THE RESOLUTION IS NOT HELD HERE: the curl arm is ``fields.condfac_for``'s
        # answer, read per launch as before. Only the text is held, per distinct
        # (expansion, tail), under the getter's key: the launcher derives
        # ``conductive = arm == "conductive"`` (no_pml_complex_fused_electric_pair.py
        # :1313) and its getter names ``kernel_name("conductive" if conductive else
        # "plain")`` and keys ``f"{name}_arm{normalized_expansion(arm)}"`` (:912-915).
        expansion = arguments["expansion"]
        conductive = arguments["curl_arm"] == "conductive"
        name = family.kernel_name("conductive" if conductive else "plain")
        kernel = _held_kernel(
            held, family, (expansion, conductive),
            lambda: family.no_pml_complex_fused_electric_pair_source(expansion,
                                                                     conductive),
            f"{name}_arm{family.complex_emitter.normalized_expansion(expansion)}",
            True, name)
        return family.launch_no_pml_complex_fused_electric_pair(
            fields, arguments["boundary_codes"], arguments["phase_flags"],
            arguments["phase_values"], arguments["dtdx"], arguments["curl_arm"],
            expansion, kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="complex no-absorber fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _cylindrical_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_fused_electric_pair`` as a ``Coverage``, at call time."""
    from .cylindrical_fused_electric_pair import (  # noqa: PLC0415
        covers_cylindrical_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_cylindrical_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the Dcyl complex D/E product, built without a device.

    THE RADIAL PREFIX IS RESOLVED PER LAUNCH AND NOT PER CONFIGURATION, as in the
    magnetic twin: ``cylindrical_prefix`` is a cumulative sum over the CURRENT Hp, so a
    value cached beside the coefficient tables would be a stale field one step later --
    the same class of defect as a fused pair reading a pre-injection D, arriving through
    a cache instead of a seam. EVERY CYLINDRICAL QUANTITY HERE IS THE ``step_D`` ONE:
    the prefix sums Hp rather than Ep, and ``imr_rows_for`` binds the D-side coefficient
    rows, whose signs are the B side's negated (``IMR_TERMS``).
    """
    from .cylindrical_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .cylindrical_complex_kernels import (  # noqa: PLC0415
            axis_increment_scalars, cylindrical_complex_boundary_codes,
            imr_rows_for, m_class, zero_rows,
        )
        from .cylindrical_fused_electric_pair import (  # noqa: PLC0415
            cylindrical_fused_electric_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        dtdx = float(ctx.grid.dt / ctx.grid.dx)
        m = int(ctx.grid.m)
        return {"tables": cylindrical_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": cylindrical_complex_boundary_codes(ctx.grid),
                "imr_rows": imr_rows_for("step_D", ctx.grid.xp, m, dtdx,
                                         int(ctx.fields.Dx.shape[0]),
                                         ctx.fields.Dx.dtype),
                "increment_scalars": axis_increment_scalars(m, dtdx),
                "m_class": m_class(m),
                "zero_rows": zero_rows(
                    m, bool(ctx.grid.accurate_fields_near_cylorigin)),
                "walls": zero_metal_axes(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": dtdx}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_fused_electric_pair as family  # noqa: PLC0415
        from .cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.cylindrical_fused_electric_pair_source, arguments["arm"])
        return family.launch_cylindrical_fused_electric_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_D"), arguments["imr_rows"],
            arguments["boundary_codes"], arguments["increment_scalars"],
            arguments["m_class"], arguments["zero_rows"], arguments["walls"],
            arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="cylindrical fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _fused_polarization_pair_coverage(context: Any) -> Coverage:
    """``covers_fused_polarization_pair`` as a ``Coverage``, imported at call time.

    THE FIRST PREDICATE IN THIS FILE ASKED ABOUT THE E->P SEAM. ``sources`` is
    passed through for uniformity and is INERT inside the predicate -- the driver
    injects nothing between ``update_E`` and ``update_P`` (driver.py:3313/:3315),
    so no source list can put a deposit in this seam; the predicate's docstring
    carries the argument and ``test_fused_polarization_pair.py`` measures the
    inertness in both directions.
    """
    from .fused_polarization_pair import (  # noqa: PLC0415 - see THE ONE DIVERGENCE
        covers_fused_polarization_pair,
    )

    covered, reason = covers_fused_polarization_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _fused_polarization_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the PML E->P product, built without a device.

    THE POLE PLAN IS NOT RESOLVED HERE AND MUST NEVER BE: the P buffers rotate
    per component between this product's own launches, so the launcher resolves
    every pole pointer immediately before each launch
    (``fused_polarization_pair.component_specs``) -- a plan that cached them
    would be stale from its own second launch. What IS resolved per frozen
    configuration is the half-integer coefficient tables, which never move.
    """
    from .fused_polarization_pair import (  # noqa: PLC0415
        FAMILY_PML, KERNEL_NAME, REPLACES,
    )

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .dispersive_kernels import dispersive_tables  # noqa: PLC0415

        # THE HALF-INTEGER SUB-LATTICE, decided by the certified family's own
        # resolver (stepping.py:1015): update_E reads it, and the integer views
        # are a half-cell error in the absorber profile.
        return {"tables": dispersive_tables(ctx.pml)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .fused_polarization_pair import (  # noqa: PLC0415 - a device import
            run_fused_polarization_pair,
        )

        return run_fused_polarization_pair(fields, None, "pml",
                                           tables=arguments["tables"])

    return CudaFusedPairPlan(family=FAMILY_PML, label="fused polarization pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_E", "update_P"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _no_pml_fused_polarization_pair_coverage(context: Any) -> Coverage:
    """``covers_no_pml_fused_polarization_pair`` as a ``Coverage``, at call time.

    Disjoint from the PML closure above on the absorber alone: both of this
    product's halves refuse an active layer by name and both of that one's
    require it, so the two candidates on ``update_E`` partition on a single
    boolean -- the same proof the four ``step_D`` products rest on.
    """
    from .fused_polarization_pair import (  # noqa: PLC0415
        covers_no_pml_fused_polarization_pair,
    )

    covered, reason = covers_no_pml_fused_polarization_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _no_pml_fused_polarization_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the no-absorber E->P product, built without a device.

    NO ``tables`` KEY: arm 2's store reads no coefficient vector (there is no
    absorber to have one), and a resolver that carried an empty one anyway would
    suggest the launch binds something it does not. The pole pointers are
    resolved per launch inside the launcher, as in the PML twin.
    """
    from .fused_polarization_pair import (  # noqa: PLC0415
        FAMILY_NO_PML, KERNEL_NAME, REPLACES,
    )

    def resolve(_ctx: Any) -> Dict[str, Any]:
        return {}

    def launch(fields: Any, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .fused_polarization_pair import (  # noqa: PLC0415 - a device import
            run_fused_polarization_pair,
        )

        return run_fused_polarization_pair(fields, None, "no_pml")

    return CudaFusedPairPlan(family=FAMILY_NO_PML,
                             label="no-absorber fused polarization pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_E", "update_P"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _complex_folded_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_folded_fused_magnetic_pair`` as a ``Coverage``, at call time.

    THE LICENCE IS READ OFF THE CONTEXT like every complex closure's: the folded
    family's arms bind through ``LICENSE_COMPLEX`` (``registry._TABLE``), and a
    fused product that asked with ``license=None`` would get that family's own
    named refusal on every configuration.
    """
    from .complex_folded_fused_magnetic_pair import (  # noqa: PLC0415
        covers_complex_folded_fused_magnetic_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_folded_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_folded_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the FOLDED complex B/H product, built without a device.

    THE BOUNDARY CODES ARE THE FOLD-AWARE RESOLVER'S
    (``complex_folded_kernels.folded_complex_boundary_codes``), not the plain
    complex one's -- that one refuses a fold by name, which is correct for a
    kernel with no fold branch and wrong for this one. The fills reading is the
    shared carry's own (``complex_fill_carry.fills_plan``).
    """
    from .complex_folded_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_fill_carry import fills_plan  # noqa: PLC0415
        from .complex_folded_fused_magnetic_pair import (  # noqa: PLC0415
            complex_folded_fused_magnetic_pair_tables,
        )
        from .complex_folded_kernels import (  # noqa: PLC0415
            folded_complex_boundary_codes,
        )
        from .complex_pml_kernels import bloch_phase_arguments  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = folded_complex_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(
                f"this grid has no folded-complex boundary codes: {refusal}")
        flags, values = bloch_phase_arguments(ctx.grid, False)
        return {"tables": complex_folded_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "walls": zero_metal_axes(ctx.grid),
                "fills": fills_plan(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_folded_fused_magnetic_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.complex_folded_fused_magnetic_pair_source,
            arguments["arm"])
        return family.launch_complex_folded_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="folded complex fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _complex_beta_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_beta_fused_magnetic_pair`` as a ``Coverage``, at call time."""
    from .complex_beta_fused_magnetic_pair import (  # noqa: PLC0415
        covers_complex_beta_fused_magnetic_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_beta_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_beta_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex-BETA B/H product, built without a device.

    THE BETA WORDS ARE RESOLVED BY THE CERTIFIED FAMILY'S OWN HELPER
    (host-rounded once, the real word's signed zero passed through), so the plan
    and the certified single cannot round differently.
    """
    from .complex_beta_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_beta_fused_magnetic_pair import (  # noqa: PLC0415
            beta_coefficients, complex_beta_fused_magnetic_pair_tables,
        )
        from .complex_fill_carry import fills_plan  # noqa: PLC0415
        from .complex_folded_kernels import (  # noqa: PLC0415
            folded_complex_boundary_codes,
        )
        from .complex_pml_kernels import bloch_phase_arguments  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = folded_complex_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(
                f"this grid has no fold-aware boundary codes: {refusal}")
        flags, values = bloch_phase_arguments(ctx.grid, False)
        return {"tables": complex_beta_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "beta_words": beta_coefficients(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": fills_plan(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_beta_fused_magnetic_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.complex_beta_fused_magnetic_pair_source,
            arguments["arm"])
        return family.launch_complex_beta_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["beta_words"], arguments["walls"], arguments["fills"],
            arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="complex beta fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _cylindrical_real_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_real_fused_magnetic_pair`` as a ``Coverage``.

    NO LICENCE: both halves are REAL-storage families and neither binds an
    expansion arm; asking with one would be asking a different question than
    the predicates answer.
    """
    from .cylindrical_real_fused_magnetic_pair import (  # noqa: PLC0415
        covers_cylindrical_real_fused_magnetic_pair,
    )

    covered, reason = covers_cylindrical_real_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_real_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the REAL Dcyl B/H product, built without a device.

    THE RADIAL PREFIX IS RESOLVED PER LAUNCH AND NOT PER CONFIGURATION, the same
    rule as the complex Dcyl twins: ``cylindrical_prefix`` is a cumulative sum
    over the CURRENT Ep, so a value cached beside the coefficient tables would
    be a stale field one step later.
    """
    from .cylindrical_real_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        import numpy as _np  # noqa: PLC0415
        from .coverage import real_pml_boundary_kinds  # noqa: PLC0415
        from .cylindrical_coverage import cylindrical_boundary_codes  # noqa: PLC0415
        from .cylindrical_real_fused_magnetic_pair import (  # noqa: PLC0415
            cylindrical_real_fused_magnetic_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes = cylindrical_boundary_codes(
            tuple(real_pml_boundary_kinds(ctx.grid)))
        return {"tables": cylindrical_real_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": tuple(_np.int32(code) for code in codes),
                "walls": zero_metal_axes(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_real_fused_magnetic_pair as family  # noqa: PLC0415
        from .cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

        kernel = _held_kernel(held, family, (),
                              family.cylindrical_real_fused_magnetic_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_cylindrical_real_fused_magnetic_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_B",
                               scratch=getattr(fields, "scratch", None)),
            arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="cylindrical real fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _special_kz_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_special_kz_fused_magnetic_pair`` as a ``Coverage``. No licence:
    real-storage halves bind no expansion arm."""
    from .special_kz_fused_magnetic_pair import (  # noqa: PLC0415
        covers_special_kz_fused_magnetic_pair,
    )

    covered, reason = covers_special_kz_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _special_kz_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the special_kz B/H product, built without a device."""
    from .special_kz_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        import numpy as _np  # noqa: PLC0415
        from .coverage import real_curl_boundary_codes  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .special_kz_fused_magnetic_pair import (  # noqa: PLC0415
            beta_scalars, special_kz_fused_magnetic_pair_fills,
            special_kz_fused_magnetic_pair_tables,
        )

        codes, refusal = real_curl_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        # The fills joined 2026-09-02 with the ported carry (the residue round).
        return {"tables": special_kz_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": tuple(_np.int32(code) for code in codes),
                "walls": zero_metal_axes(ctx.grid),
                "beta": beta_scalars(ctx.grid),
                "fills": special_kz_fused_magnetic_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import special_kz_fused_magnetic_pair as family  # noqa: PLC0415

        kernel = _held_kernel(held, family, (),
                              family.special_kz_fused_magnetic_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_special_kz_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["beta"], arguments["fills"],
            arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="special_kz fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _bfast_fused_magnetic_pair_coverage(context: Any) -> Coverage:
    """``covers_bfast_fused_magnetic_pair`` as a ``Coverage``. No licence:
    real-storage halves bind no expansion arm."""
    from .bfast_fused_magnetic_pair import (  # noqa: PLC0415
        covers_bfast_fused_magnetic_pair,
    )

    covered, reason = covers_bfast_fused_magnetic_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _bfast_fused_magnetic_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the BFAST B/H product, built without a device."""
    from .bfast_fused_magnetic_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        import numpy as _np  # noqa: PLC0415
        from .bfast_fused_magnetic_pair import (  # noqa: PLC0415
            bfast_fused_magnetic_pair_tables, bfast_scalars,
        )
        from .coverage import real_curl_boundary_codes  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = real_curl_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        return {"tables": bfast_fused_magnetic_pair_tables(ctx.pml),
                "boundary_codes": tuple(_np.int32(code) for code in codes),
                "walls": zero_metal_axes(ctx.grid),
                "coefficients": bfast_scalars(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import bfast_fused_magnetic_pair as family  # noqa: PLC0415

        kernel = _held_kernel(held, family, (), family.bfast_fused_magnetic_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_bfast_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["coefficients"], arguments["dtdx"],
            kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="BFAST fused magnetic pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_B", "update_H"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _cylindrical_real_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_cylindrical_real_fused_electric_pair`` as a ``Coverage``. No
    licence: real-storage halves bind no expansion arm."""
    from .cylindrical_real_fused_electric_pair import (  # noqa: PLC0415
        covers_cylindrical_real_fused_electric_pair,
    )

    covered, reason = covers_cylindrical_real_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _cylindrical_real_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the REAL Dcyl D/E product, built without a device.

    THE RADIAL PREFIX IS RESOLVED PER LAUNCH AND NOT PER CONFIGURATION -- the
    ``step_D`` prefix (a cumulative sum over the CURRENT Hp), never the
    ``step_B`` one.
    """
    from .cylindrical_real_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .cylindrical_real_fused_electric_pair import (  # noqa: PLC0415
            cylindrical_real_fused_electric_pair_tables,
        )
        from .cylindrical_kernels import (  # noqa: PLC0415
            cylindrical_boundary_codes_for,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        return {"tables": cylindrical_real_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": cylindrical_boundary_codes_for(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import cylindrical_real_fused_electric_pair as family  # noqa: PLC0415
        from .cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

        kernel = _held_kernel(held, family, (),
                              family.cylindrical_real_fused_electric_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_cylindrical_real_fused_electric_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_D",
                               scratch=getattr(fields, "scratch", None)),
            arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="cylindrical real fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _special_kz_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_special_kz_fused_electric_pair`` as a ``Coverage``. No licence:
    real-storage halves bind no expansion arm."""
    from .special_kz_fused_electric_pair import (  # noqa: PLC0415
        covers_special_kz_fused_electric_pair,
    )

    covered, reason = covers_special_kz_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _special_kz_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the special_kz D/E product, built without a device."""
    from .special_kz_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        import numpy as _np  # noqa: PLC0415
        from .coverage import real_curl_boundary_codes  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .special_kz_fused_electric_pair import (  # noqa: PLC0415
            beta_scalars, special_kz_fused_electric_pair_fills,
            special_kz_fused_electric_pair_tables,
        )

        codes, refusal = real_curl_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        return {"tables": special_kz_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": tuple(_np.int32(code) for code in codes),
                "walls": zero_metal_axes(ctx.grid),
                "beta": beta_scalars(ctx.grid),
                "fills": special_kz_fused_electric_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import special_kz_fused_electric_pair as family  # noqa: PLC0415

        kernel = _held_kernel(held, family, (),
                              family.special_kz_fused_electric_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_special_kz_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["beta"], arguments["fills"],
            arguments["dtdx"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="special_kz fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _bfast_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_bfast_fused_electric_pair`` as a ``Coverage``. No licence:
    real-storage halves bind no expansion arm."""
    from .bfast_fused_electric_pair import (  # noqa: PLC0415
        covers_bfast_fused_electric_pair,
    )

    covered, reason = covers_bfast_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _bfast_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the BFAST D/E product, built without a device."""
    from .bfast_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        import numpy as _np  # noqa: PLC0415
        from .bfast_curl import bfast_curl_coefficients  # noqa: PLC0415
        from .bfast_fused_electric_pair import (  # noqa: PLC0415
            bfast_fused_electric_pair_tables,
        )
        from .coverage import real_curl_boundary_codes  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = real_curl_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        return {"tables": bfast_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": tuple(_np.int32(code) for code in codes),
                "walls": zero_metal_axes(ctx.grid),
                "coefficients": bfast_curl_coefficients(ctx.grid, "step_D"),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import bfast_fused_electric_pair as family  # noqa: PLC0415

        kernel = _held_kernel(held, family, (), family.bfast_fused_electric_pair_source,
                              family.KERNEL_NAME, False, family.KERNEL_NAME)
        return family.launch_bfast_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["coefficients"], arguments["dtdx"],
            kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY, label="BFAST fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _complex_folded_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_folded_fused_electric_pair`` as a ``Coverage``."""
    from .complex_folded_fused_electric_pair import (  # noqa: PLC0415
        covers_complex_folded_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_folded_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_folded_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the folded-complex D/E product, built without a device."""
    from .complex_folded_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_folded_fused_electric_pair import (  # noqa: PLC0415
            complex_folded_fused_electric_pair_fills,
            complex_folded_fused_electric_pair_tables,
        )
        from .complex_folded_kernels import (  # noqa: PLC0415
            folded_complex_boundary_codes,
        )
        from .complex_pml_kernels import bloch_phase_arguments  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = folded_complex_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        # THE BACKWARD PHASE TABLE: `True` is the step_D sub-step's own
        # conjugation, the one argument a copy of the magnetic sibling would
        # get silently wrong.
        flags, values = bloch_phase_arguments(ctx.grid, True)
        return {"tables": complex_folded_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "walls": zero_metal_axes(ctx.grid),
                "fills": complex_folded_fused_electric_pair_fills(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_folded_fused_electric_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.complex_folded_fused_electric_pair_source,
            arguments["arm"])
        return family.launch_complex_folded_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="folded complex fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _complex_beta_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_beta_fused_electric_pair`` as a ``Coverage``."""
    from .complex_beta_fused_electric_pair import (  # noqa: PLC0415
        covers_complex_beta_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    covered, reason = covers_complex_beta_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_beta_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex-beta D/E product, built without a device."""
    from .complex_beta_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_beta_fused_electric_pair import (  # noqa: PLC0415
            beta_coefficients, complex_beta_fused_electric_pair_fills,
            complex_beta_fused_electric_pair_tables,
        )
        from .complex_folded_kernels import (  # noqa: PLC0415
            folded_complex_boundary_codes,
        )
        from .complex_pml_kernels import bloch_phase_arguments  # noqa: PLC0415
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

        codes, refusal = folded_complex_boundary_codes(ctx.grid)
        if refusal is not None:
            raise ValueError(f"this grid has no boundary codes: {refusal}")
        flags, values = bloch_phase_arguments(ctx.grid, True)
        return {"tables": complex_beta_fused_electric_pair_tables(ctx.pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "beta_words": beta_coefficients(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": complex_beta_fused_electric_pair_fills(ctx.grid),
                "arm": ctx.license_for(LICENSE_COMPLEX)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_beta_fused_electric_pair as family  # noqa: PLC0415

        kernel = _held_arm_kernel(
            held, family, family.complex_beta_fused_electric_pair_source,
            arguments["arm"])
        return family.launch_complex_beta_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["beta_words"], arguments["walls"], arguments["fills"],
            arguments["dtdx"], arguments["arm"], kernel=kernel)

    return CudaFusedPairPlan(family=FAMILY,
                             label="complex beta fused electric pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             # Pure readings of the frozen pml/grid/licence: held.
                             resolve_launch_args=_resolved_once(held, resolve),
                             launch=launch)


def _complex_no_pml_fused_polarization_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_no_pml_fused_polarization_pair`` as a ``Coverage``, asked
    with the ``cuda_complex_no_pml`` family's own licence -- the seven-pattern
    record, the same one its slot arms bind through."""
    from .complex_no_pml_fused_polarization_pair import (  # noqa: PLC0415
        covers_complex_no_pml_fused_polarization_pair,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    covered, reason = covers_complex_no_pml_fused_polarization_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX_NO_PML), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_no_pml_fused_polarization_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the complex no-absorber E->P product."""
    from .complex_no_pml_fused_polarization_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    def resolve(ctx: Any) -> Dict[str, Any]:
        return {"arm": ctx.license_for(LICENSE_COMPLEX_NO_PML)["arm"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .complex_no_pml_fused_polarization_pair import (  # noqa: PLC0415
            run_complex_no_pml_fused_polarization_pair,
        )

        return run_complex_no_pml_fused_polarization_pair(
            fields, None, arguments["arm"])

    return CudaFusedPairPlan(family=FAMILY,
                             label="complex no-PML fused polarization pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_E", "update_P"), context=context,
                             resolve_launch_args=resolve, launch=launch)


#: family -> (the curl slot it starts at, its coverage closure, its plan builder).
#: THREE ROWS SINCE 2026-08-30, and the two new ones are the board's rank-1 and
#: rank-2 unserved cells rather than variants of the first: the real pair emits
#: ``step_curl_kernels``/``constitutive_kernels`` text against float32 storage, and
#: neither complex product shares a byte of it. They register beside it rather than
#: inside it because ``install_fused_pairs`` picks at most ONE candidate per seam and
#: leaves the seam unfused on an ambiguity -- so what makes three rows legal is that
#: the three predicates are DISJOINT: the real curl refuses complex64 storage by
#: name, the complex curl refuses a run that is neither ``force_complex_fields`` nor
#: Bloch-phased AND refuses a Dcyl grid, and the cylindrical complex curl REQUIRES a
#: Dcyl grid at |m| >= 1.
#:
#: A row for a product with no module would report "the builder raised" on every
#: configuration instead of saying nothing, which reads as a claim about the run.
def _offdiag_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_offdiag_fused_electric_pair`` as a ``Coverage``, imported at call time."""
    from .offdiag_fused_electric_pair import (  # noqa: PLC0415
        covers_offdiag_fused_electric_pair,
    )

    covered, reason = covers_offdiag_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _offdiag_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the UNFOLDED stencil weld, built without a device.

    THE SCRATCH IS RESOLVED ONCE AND HELD IN THIS CLOSURE, which is the one thing
    this builder does that no earlier product's does. ``resolve_launch_args`` runs on
    every ``run()``, and allocating six full volumes per timestep would be the exact
    cost ``StepScratch``'s own measurement exists to avoid; holding them here makes
    "once per frozen configuration" a property of the code rather than a convention.
    The ROTATION then ping-pongs the same two sets forever: what the launcher hands
    back is the retired pair, which becomes the next launch's scratch.
    """
    from .offdiag_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from . import offdiag_stencil_weld as weld  # noqa: PLC0415
        from .coverage import (  # noqa: PLC0415
            offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags,
        )
        from .offdiag_emitter import normalized_row_mask  # noqa: PLC0415
        from .offdiag_fused_electric_pair import (  # noqa: PLC0415
            offdiag_fused_electric_pair_codes, offdiag_fused_electric_pair_scratch,
            offdiag_fused_electric_pair_tables,
        )

        if "tables" not in held:
            held["tables"] = offdiag_fused_electric_pair_tables(ctx.pml)
            held["scratch"] = offdiag_fused_electric_pair_scratch(ctx.fields)
        # THREE READINGS OF THE GRID, AND NO ONE OF THEM IS THE OTHERS: the boundary
        # codes are the ghost-rule resolution (cross-checked against the
        # constitutive's own reading, which this family binds as one vector), the
        # COUPLING wall mask is _mask_metallic_wall_coupling's question, and the fill
        # plan is the three in-seam passes' -- of which only zero_metal_D can be live
        # here, because the constitutive arm refuses a fold.
        return {"tables": held["tables"], "scratch": held["scratch"],
                "codes": offdiag_fused_electric_pair_codes(ctx.grid, ctx.pml),
                "wall_mask": offdiag_wall_mask_flags(ctx.grid),
                "fills": weld.fill_plan(ctx.grid),
                "row_mask": normalized_row_mask(offdiag_row_mask(ctx.fields)),
                "rows": [volume for volume in offdiag_row_volumes(ctx.fields)
                         if volume is not None],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import offdiag_fused_electric_pair as family  # noqa: PLC0415
        from .offdiag_fused_electric_pair import (  # noqa: PLC0415 - a device import
            assert_scratch_is_disjoint, launch_offdiag_fused_electric_pair,
            rotate_into_fields,
        )

        assert_scratch_is_disjoint(fields, arguments["scratch"], arguments["tables"])
        kernel = _held_offdiag_kernel(held, family, arguments["row_mask"])
        record = launch_offdiag_fused_electric_pair(
            fields, arguments["scratch"], arguments["tables"], arguments["codes"],
            arguments["wall_mask"], arguments["fills"], arguments["dtdx"],
            arguments["row_mask"], arguments["rows"], kernel=kernel)
        # THE ROTATION IS PART OF THE LAUNCH, not of the caller. A scratch-output
        # weld that did not rotate would have computed the step and thrown it away,
        # which is precisely the ``rotation_skipped`` null the gate arms.
        held["scratch"] = rotate_into_fields(fields, arguments["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY, label="off-diagonal stencil weld",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _folded_offdiag_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_folded_offdiag_fused_electric_pair`` as a ``Coverage``."""
    from .folded_offdiag_fused_electric_pair import (  # noqa: PLC0415
        covers_folded_offdiag_fused_electric_pair,
    )

    covered, reason = covers_folded_offdiag_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _folded_offdiag_fused_electric_pair_plan(context: Any) -> "CudaFusedPairPlan":
    """The plan object for the FOLDED stencil weld, built without a device.

    The scratch is held exactly as its unfolded twin holds it. What differs is that
    BOTH boundary readings are resolved and kept apart, and the mirror ghost weights
    join the plan: on a fold the curl and the constitutive read the same axis
    differently by design, and swapping the two triples is a wrong ghost rule on a
    whole plane rather than a crash.
    """
    from .folded_offdiag_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from . import folded_offdiag_kernels as folded  # noqa: PLC0415
        from . import offdiag_stencil_weld as weld  # noqa: PLC0415
        from .coverage import (  # noqa: PLC0415
            offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags,
        )
        from .folded_offdiag_fused_electric_pair import (  # noqa: PLC0415
            folded_offdiag_fused_electric_pair_codes,
            folded_offdiag_fused_electric_pair_scratch,
            folded_offdiag_fused_electric_pair_tables,
        )
        from .offdiag_emitter import normalized_row_mask  # noqa: PLC0415

        if "tables" not in held:
            held["tables"] = folded_offdiag_fused_electric_pair_tables(ctx.pml)
            held["scratch"] = folded_offdiag_fused_electric_pair_scratch(ctx.fields)
        return {"tables": held["tables"], "scratch": held["scratch"],
                "codes": folded_offdiag_fused_electric_pair_codes(ctx.grid, ctx.pml),
                "wall_mask": offdiag_wall_mask_flags(ctx.grid),
                "weights": folded.mirror_ghost_weights(ctx.grid),
                "fills": weld.fill_plan(ctx.grid),
                "row_mask": normalized_row_mask(offdiag_row_mask(ctx.fields)),
                "rows": [volume for volume in offdiag_row_volumes(ctx.fields)
                         if volume is not None],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import folded_offdiag_fused_electric_pair as family  # noqa: PLC0415
        from .folded_offdiag_fused_electric_pair import (  # noqa: PLC0415
            _validate_launch_arguments, assert_scratch_is_disjoint,
            launch_folded_offdiag_fused_electric_pair, rotate_into_fields,
        )

        assert_scratch_is_disjoint(fields, arguments["scratch"], arguments["tables"])
        _validate_launch_arguments(
            tuple(int(n) for n in fields.Dx.shape), arguments["codes"],
            arguments["wall_mask"], arguments["weights"], arguments["fills"])
        kernel = _held_offdiag_kernel(held, family, arguments["row_mask"])
        record = launch_folded_offdiag_fused_electric_pair(
            fields, arguments["scratch"], arguments["tables"], arguments["codes"],
            arguments["wall_mask"], arguments["weights"], arguments["fills"],
            arguments["dtdx"], arguments["row_mask"], arguments["rows"],
            kernel=kernel)
        held["scratch"] = rotate_into_fields(fields, arguments["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY,
                             label="folded off-diagonal stencil weld",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _folded_complex_offdiag_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_folded_complex_offdiag_fused_electric_pair`` as a ``Coverage``.

    TWO LICENCES, ONE PER HALF, and this is the first product on this table that
    needs two: the folded complex curl binds ``LICENSE_COMPLEX`` and the
    off-diagonal ``update_E`` binds ``LICENSE_COMPLEX_OFFDIAG``, whose arbiter
    classifies a FIFTH multiply orientation the base four do not carry. Handing one
    verdict to both halves would declare that one record's pattern set answered for
    the other's.
    """
    from .folded_complex_offdiag_fused_electric_pair import (  # noqa: PLC0415
        covers_folded_complex_offdiag_fused_electric_pair,
    )
    from .registry import LICENSE_COMPLEX, LICENSE_COMPLEX_OFFDIAG  # noqa: PLC0415

    covered, reason = covers_folded_complex_offdiag_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX_OFFDIAG), context.subnormal_policy,
        curl_license=context.license_for(LICENSE_COMPLEX))
    return Coverage(bool(covered), (str(reason),))


def _folded_complex_offdiag_fused_electric_pair_plan(context: Any
                                                     ) -> "CudaFusedPairPlan":
    """The plan object for the FOLDED COMPLEX stencil weld, built without a device.

    The scratch is held exactly as the real-storage welds hold it, and the two
    boundary readings are resolved and kept APART for the folded twin's reason. The
    Bloch table is resolved ONCE and cross-checked between the two resolvers inside
    the family's own helper, because the curl's conjugate factor and the
    constitutive's DOWN factor are the same number by two routes.
    """
    from .folded_complex_offdiag_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX_OFFDIAG  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from . import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
        from . import complex_offdiag_update_e as offdiag  # noqa: PLC0415
        from .coverage import (  # noqa: PLC0415
            offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags,
        )
        from .folded_complex_offdiag_fused_electric_pair import (  # noqa: PLC0415
            folded_complex_offdiag_fused_electric_pair_codes,
            folded_complex_offdiag_fused_electric_pair_phases,
            folded_complex_offdiag_fused_electric_pair_scratch,
            folded_complex_offdiag_fused_electric_pair_tables,
        )

        if "tables" not in held:
            held["tables"] = folded_complex_offdiag_fused_electric_pair_tables(
                ctx.pml)
            held["scratch"] = folded_complex_offdiag_fused_electric_pair_scratch(
                ctx.fields)
        return {"tables": held["tables"], "scratch": held["scratch"],
                "codes": folded_complex_offdiag_fused_electric_pair_codes(ctx.grid),
                "phases": folded_complex_offdiag_fused_electric_pair_phases(
                    ctx.grid),
                "wall_mask": offdiag_wall_mask_flags(ctx.grid),
                "weights": offdiag.mirror_ghost_weights(ctx.grid),
                "fills": weld.fill_plan(ctx.grid),
                "row_mask": offdiag.normalized_row_mask(
                    offdiag_row_mask(ctx.fields)),
                "rows": [volume for volume in offdiag_row_volumes(ctx.fields)
                         if volume is not None],
                "arm": ctx.license_for(LICENSE_COMPLEX_OFFDIAG)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import folded_complex_offdiag_fused_electric_pair as family  # noqa: PLC0415
        from .folded_complex_offdiag_fused_electric_pair import (  # noqa: PLC0415
            _validate_launch_arguments, assert_scratch_is_disjoint,
            launch_folded_complex_offdiag_fused_electric_pair, rotate_into_fields,
        )

        assert_scratch_is_disjoint(fields, arguments["scratch"],
                                   arguments["tables"])
        _validate_launch_arguments(
            tuple(int(n) for n in fields.Dx.shape), arguments["codes"],
            arguments["wall_mask"], arguments["weights"], arguments["fills"])
        kernel = _held_offdiag_kernel(held, family, arguments["row_mask"],
                                      arguments["arm"])
        record = launch_folded_complex_offdiag_fused_electric_pair(
            fields, arguments["scratch"], arguments["tables"], arguments["codes"],
            arguments["wall_mask"], arguments["phases"], arguments["weights"],
            arguments["fills"], arguments["dtdx"], arguments["row_mask"],
            arguments["rows"], arguments["arm"], kernel=kernel)
        held["scratch"] = rotate_into_fields(fields, arguments["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY,
                             label="folded complex off-diagonal stencil weld",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _complex_no_pml_offdiag_fused_electric_pair_coverage(context: Any) -> Coverage:
    """``covers_complex_no_pml_offdiag_fused_electric_pair`` as a ``Coverage``.

    TWO LICENCES, ONE PER HALF -- see the folded twin above. The complex
    no-absorber curl binds ``LICENSE_COMPLEX_NO_PML``, whose arbiter reads a
    SEVEN-pattern record, and the off-diagonal ``update_E`` binds
    ``LICENSE_COMPLEX_OFFDIAG``.
    """
    from .complex_no_pml_offdiag_fused_electric_pair import (  # noqa: PLC0415
        covers_complex_no_pml_offdiag_fused_electric_pair,
    )
    from .registry import (  # noqa: PLC0415
        LICENSE_COMPLEX_NO_PML, LICENSE_COMPLEX_OFFDIAG,
    )

    covered, reason = covers_complex_no_pml_offdiag_fused_electric_pair(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX_OFFDIAG), context.subnormal_policy,
        curl_license=context.license_for(LICENSE_COMPLEX_NO_PML))
    return Coverage(bool(covered), (str(reason),))


def _complex_no_pml_offdiag_fused_electric_pair_plan(context: Any
                                                     ) -> "CudaFusedPairPlan":
    """The plan object for the NO-ABSORBER COMPLEX stencil weld, without a device.

    THREE scratch volumes rather than six and no coefficient tables at all: this
    arm has neither a split-field auxiliary nor a constitutive recurrence.
    """
    from .complex_no_pml_offdiag_fused_electric_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )
    from .registry import LICENSE_COMPLEX_OFFDIAG  # noqa: PLC0415

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from . import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
        from . import complex_offdiag_update_e as offdiag  # noqa: PLC0415
        from .complex_no_pml_offdiag_fused_electric_pair import (  # noqa: PLC0415
            complex_no_pml_offdiag_fused_electric_pair_codes,
            complex_no_pml_offdiag_fused_electric_pair_phases,
            complex_no_pml_offdiag_fused_electric_pair_scratch,
        )
        from .coverage import (  # noqa: PLC0415
            offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags,
        )

        if "scratch" not in held:
            held["scratch"] = complex_no_pml_offdiag_fused_electric_pair_scratch(
                ctx.fields)
        return {"scratch": held["scratch"],
                "codes": complex_no_pml_offdiag_fused_electric_pair_codes(ctx.grid),
                "phases": complex_no_pml_offdiag_fused_electric_pair_phases(
                    ctx.grid),
                "wall_mask": offdiag_wall_mask_flags(ctx.grid),
                "weights": offdiag.mirror_ghost_weights(ctx.grid),
                "fills": weld.fill_plan(ctx.grid),
                "row_mask": offdiag.normalized_row_mask(
                    offdiag_row_mask(ctx.fields)),
                "rows": [volume for volume in offdiag_row_volumes(ctx.fields)
                         if volume is not None],
                "arm": ctx.license_for(LICENSE_COMPLEX_OFFDIAG)["arm"],
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import complex_no_pml_offdiag_fused_electric_pair as family  # noqa: PLC0415
        from .complex_no_pml_offdiag_fused_electric_pair import (  # noqa: PLC0415
            _validate_launch_arguments, assert_scratch_is_disjoint,
            launch_complex_no_pml_offdiag_fused_electric_pair, rotate_into_fields,
        )

        assert_scratch_is_disjoint(fields, arguments["scratch"])
        _validate_launch_arguments(arguments["codes"], arguments["wall_mask"],
                                   arguments["fills"])
        kernel = _held_offdiag_kernel(held, family, arguments["row_mask"],
                                      arguments["arm"])
        record = launch_complex_no_pml_offdiag_fused_electric_pair(
            fields, arguments["scratch"], arguments["codes"],
            arguments["wall_mask"], arguments["phases"], arguments["weights"],
            arguments["fills"], arguments["dtdx"], arguments["row_mask"],
            arguments["rows"], arguments["arm"], kernel=kernel)
        held["scratch"] = rotate_into_fields(fields, arguments["scratch"])
        record["rotated"] = True
        return record

    return CudaFusedPairPlan(family=FAMILY,
                             label="complex no-absorber off-diagonal stencil weld",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("step_D", "update_E"), context=context,
                             resolve_launch_args=resolve, launch=launch)


def _dispersive_offdiag_fused_polarization_pair_coverage(context: Any) -> Coverage:
    """``covers_dispersive_offdiag_fused_polarization_pair`` as a ``Coverage``."""
    from .dispersive_offdiag_fused_polarization_pair import (  # noqa: PLC0415
        covers_dispersive_offdiag_fused_polarization_pair,
    )

    covered, reason = covers_dispersive_offdiag_fused_polarization_pair(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _dispersive_offdiag_fused_polarization_pair_plan(context: Any
                                                     ) -> "CudaFusedPairPlan":
    """The plan object for the off-diagonal E->P weld, built without a device.

    THE TABLES ARE HELD, THE BUFFERS ARE NOT. Coefficient views never move, so
    they are resolved once; the polarization buffers ROTATE per component on the
    host and are re-read inside the launch on every call, which is the certified
    ADE launcher's own standing hazard restated -- a plan that cached them "would
    be stale from the second component of the first step, and stale in a way that
    still computes".
    """
    from .dispersive_offdiag_fused_polarization_pair import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from . import dispersive_offdiag_update_e as offdiag_e  # noqa: PLC0415

        if "tables" not in held:
            held["tables"] = offdiag_e.dispersive_offdiag_constitutive_tables(ctx.pml)
        return {"tables": held["tables"]}

    def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .dispersive_offdiag_fused_polarization_pair import (  # noqa: PLC0415
            run_dispersive_offdiag_fused_polarization_pair,
        )

        # THE WHOLE SEAM IN ONE CALL, because the tail launches and the rotations
        # between them are part of what REPLACES declares. A composition that
        # launched only the fused kernel would leave two components unadvanced.
        return run_dispersive_offdiag_fused_polarization_pair(
            fields, fields.grid, None, tables=arguments["tables"])

    return CudaFusedPairPlan(family=FAMILY,
                             label="dispersive off-diagonal polarization pair",
                             kernel_label=KERNEL_NAME, replaces=tuple(REPLACES),
                             slots=("update_E", "update_P"), context=context,
                             resolve_launch_args=resolve, launch=launch)


# ---------------------------------------------------------------------------
# THE THREE-SLOT WELDS, 2026-09-02 -- the first products on this table that own
# THREE driver slots rather than two
# ---------------------------------------------------------------------------
#
# WHY THEY EXIST is :func:`_later_seam_claimant`'s own measured collision: ``update_E``
# is the SECOND consult of ``D_to_E`` and the FIRST of ``E_to_P``, so a D->E pair and
# an E->P pair compete for one slot and the trade is exactly net zero on all ten rows
# that reach it. That function's docstring names the resolution -- "a THREE-SLOT
# product spanning ``step_D`` -> ``update_E`` -> ``update_P``, which serves BOTH seams
# instead of trading one for the other" -- and these two are it.
#
# THEIR SHAPE IS DIFFERENT FROM EVERY PAIR ABOVE IN ONE WAY THAT MATTERS: the product
# performs TWO device groups with its OWN host work between them. Group 1 is the
# shipped D->E launch; then ``deposit_repair.apply`` runs at the ``update_E`` consult,
# reading ``state.P`` as it stands; then group 2 advances the polarizations. So the
# plan carries TWO launchers rather than one, and :class:`CudaFusedTriplePlan` holds
# them as two half-plans the installer puts in different slots. Advancing P inside
# group 1, or before the repair, both compute and are wrong only at the deposit cells
# (``parity/meep_gpu/probe_cuda_three_slot_weld.py``, ``p_inside_the_launch`` and
# ``repair_after_p``, diverging on 6 of 7 configurations each).


def _three_slot_dispersive_weld_coverage(context: Any) -> Coverage:
    """``covers_three_slot_dispersive_weld`` as a ``Coverage``, at call time.

    ITS SOURCE QUESTION IS THE D->E HALF'S, asked through the conjunction: that half
    routes through ``deposit_repair.seam_source_reasons(..., 'D')`` carrying its own
    ``CARRIES_DEPOSIT_REPAIR`` and the split-field path, which is the declaration this
    weld's module repeats. All ten rows the two three-slot welds exist for carry an
    electric deposit inside the span, so the routing IS the product.
    """
    from .three_slot_dispersive_weld import (  # noqa: PLC0415
        covers_three_slot_dispersive_weld,
    )

    covered, reason = covers_three_slot_dispersive_weld(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _three_slot_dispersive_weld_plan(context: Any) -> "CudaFusedTriplePlan":
    """The plan object for the PML three-slot weld, built without a device.

    TWO LAUNCHERS, NOT ONE, and which slot each lands in is the product: the leading
    one is the shipped D->E launch and the trailing one is the polarization advance,
    and ``_install_fused_triple`` puts the deposit repair BETWEEN them.

    NOTHING THAT ROTATES IS RESOLVED HERE. The pole plan and the per-component state
    lists are read inside each launcher, immediately before the launch that binds
    them: ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` every
    step AND between this product's own three trailing launches, so a plan that cached
    them would be stale from its own second launch -- stale in a way that still
    computes. What IS resolved per frozen configuration is the grid reading and the
    half-integer coefficient tables, which never move.
    """
    from .three_slot_dispersive_weld import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES, SLOTS,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .fused_electric_pair import (  # noqa: PLC0415
            fused_electric_pair_fills, fused_electric_pair_tables,
        )
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        # THE SAME FOUR READINGS OF THE GRID the D->E half's own plan builder takes,
        # from the same modules: sharing them is deliberate, because group 1 IS that
        # product's launch and a second reading is a second thing to drift.
        return {"tables": fused_electric_pair_tables(ctx.pml),
                "boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "fills": fused_electric_pair_fills(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch_leading(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import dispersive_fused_electric_pair as leading  # noqa: PLC0415

        # GROUP 1 IS THE D->E PAIR'S OWN LAUNCH, so its held text is that product's,
        # under that product's key; the trailing group's per-component texts are left
        # to their own getter.
        poles, counts = leading.pole_bindings(fields)
        kernel = _held_dispersive_kernel(held, leading, counts)
        return leading.launch_dispersive_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            poles, counts, kernel=kernel)

    def launch_trailing(fields: Any, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .three_slot_dispersive_weld import (  # noqa: PLC0415 - a device import
            run_three_slot_polarization,
        )

        return run_three_slot_polarization(fields, "pml")

    return CudaFusedTriplePlan(
        family=FAMILY, label="three-slot dispersive weld",
        kernel_label=KERNEL_NAME, replaces=tuple(REPLACES), slots=tuple(SLOTS),
        # Pure readings of the frozen pml/grid: held, per the docstring above.
        context=context, resolve_launch_args=_resolved_once(held, resolve),
        launch_leading=launch_leading, launch_trailing=launch_trailing)


def _no_pml_three_slot_dispersive_weld_coverage(context: Any) -> Coverage:
    """``covers_no_pml_three_slot_dispersive_weld`` as a ``Coverage``, at call time.

    Disjoint from the PML closure above on the absorber alone: all four halves this
    one conjoins refuse an active layer by name and all four of that one's require
    it. Its source question routes through the no-absorber D->E half, which carries
    ``deposit_repair.PLAIN_PATH`` -- the recurrence ``update_E`` actually runs on an
    inert layer, and the declaration this weld's module repeats.
    """
    from .no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
        covers_no_pml_three_slot_dispersive_weld,
    )

    covered, reason = covers_no_pml_three_slot_dispersive_weld(
        context.fields, context.pml, context.grid, context.sources)
    return Coverage(bool(covered), (str(reason),))


def _no_pml_three_slot_dispersive_weld_plan(context: Any) -> "CudaFusedTriplePlan":
    """The plan object for the no-absorber three-slot weld, built without a device.

    NO ``tables`` AND NO ``fills`` KEY: the no-absorber D->E half reads no coefficient
    vector (there is no absorber to have one) and REFUSES a mirror plane, so a
    resolver carrying either would suggest the launch binds something it does not.
    The conductivity mask travels with the pole plan INSIDE the launcher, because the
    two are read from the same ``fields`` and a kernel compiled for one mask cannot be
    handed another's bindings.
    """
    from .no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES, SLOTS,
    )

    held: Dict[str, Any] = {}

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
        from .step_curl_kernels import real_curl_boundary_codes  # noqa: PLC0415

        return {"boundary_codes": real_curl_boundary_codes(ctx.grid),
                "walls": zero_metal_axes(ctx.grid),
                "dtdx": float(ctx.grid.dt / ctx.grid.dx)}

    def launch_leading(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from . import no_pml_dispersive_fused_electric_pair as leading  # noqa: PLC0415

        poles, counts = leading.pole_bindings(fields)
        conductivity, cond = leading.conductive_bindings(fields)
        kernel = _held_no_pml_dispersive_kernel(held, leading, cond, counts)
        return leading.launch_no_pml_dispersive_fused_electric_pair(
            fields, arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], poles, counts, conductivity, cond, kernel=kernel)

    def launch_trailing(fields: Any, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
            run_three_slot_polarization,
        )

        return run_three_slot_polarization(fields)

    return CudaFusedTriplePlan(
        family=FAMILY, label="no-absorber three-slot dispersive weld",
        kernel_label=KERNEL_NAME, replaces=tuple(REPLACES), slots=tuple(SLOTS),
        # Pure readings of the frozen grid: held.
        context=context, resolve_launch_args=_resolved_once(held, resolve),
        launch_leading=launch_leading, launch_trailing=launch_trailing)


def _complex_no_pml_three_slot_weld_coverage(context: Any) -> Coverage:
    """``covers_three_slot_complex_no_pml_dispersive_weld`` as a ``Coverage``.

    Asked with the ``cuda_complex_no_pml`` family's own licence -- the seven-pattern
    record its slot arms bind through -- exactly as the two products it conjoins
    are. Its SOURCE question is the D->E half's, which carries ``carries_repair=False``
    and refuses every electric in-seam source by name: that refusal is what licenses
    the single per-component launch, so it is not weakened here.
    """
    from .complex_no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
        covers_three_slot_complex_no_pml_dispersive_weld,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    covered, reason = covers_three_slot_complex_no_pml_dispersive_weld(
        context.fields, context.pml, context.grid, context.sources,
        context.license_for(LICENSE_COMPLEX_NO_PML), context.subnormal_policy)
    return Coverage(bool(covered), (str(reason),))


def _complex_no_pml_three_slot_weld_plan(context: Any) -> "CudaFusedTriplePlan":
    """The plan object for the complex no-absorber three-slot weld.

    ONE DEVICE GROUP, NOT TWO, and that is the product's whole claim: the leading half
    performs all three sub-steps (three per-component launches, host rotation
    between) and the trailing half is a NO-OP that still answers the ``update_P``
    consult -- a ``False`` there would make the driver advance the polarizations a
    second time on the array path. The seam is empty by predicate (no electric
    in-seam source is admitted), so ``_install_fused_triple`` installs no bracket and
    nothing sits between the leading consult and the trailing one that reaches these
    volumes; the gate measures both orderings identical
    (``p_at_update_P_consult``).

    NOTHING THAT ROTATES IS RESOLVED HERE: the pole buffers are read inside each
    launch, after the previous component's rotation.
    """
    from .complex_no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
        FAMILY, KERNEL_NAME, REPLACES, SLOTS,
    )
    from .registry import LICENSE_COMPLEX_NO_PML  # noqa: PLC0415

    def resolve(ctx: Any) -> Dict[str, Any]:
        from .complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments, complex_boundary_codes,
        )
        from .no_pml_complex_fused_electric_pair import (  # noqa: PLC0415
            no_pml_complex_curl_arm,
        )

        # ``True`` IS THE BACKWARD FLAG: step_D takes the conjugated Bloch factor.
        flags, values = bloch_phase_arguments(ctx.grid, True)
        return {"boundary_codes": complex_boundary_codes(ctx.grid),
                "phase_flags": flags, "phase_values": values,
                "curl_arm": no_pml_complex_curl_arm(ctx.fields, "step_D"),
                "expansion": ctx.license_for(LICENSE_COMPLEX_NO_PML)["arm"],
                "grid": ctx.grid}

    def launch_leading(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .complex_no_pml_three_slot_dispersive_weld import (  # noqa: PLC0415
            run_three_slot_no_pml_complex,
        )

        return run_three_slot_no_pml_complex(
            fields, arguments["grid"], arguments["expansion"],
            curl_arm=arguments["curl_arm"],
            boundary_codes=arguments["boundary_codes"],
            phase_flags=arguments["phase_flags"],
            phase_values=arguments["phase_values"])

    def launch_trailing(_fields: Any, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        # THE CONSULT IS ANSWERED, NOT LAUNCHED: update_P was advanced inside the
        # leading group. Returning a report rather than raising is what keeps the
        # driver from running the array path's update_P on top.
        return {"launched": False, "launches": 0, "replaces": ("update_P",),
                "advanced_in": "leading",
                "why": "the polarizations were advanced inside the leading group's "
                       "per-component launches; nothing sits between the two "
                       "consults on an admitted row"}

    return CudaFusedTriplePlan(
        family=FAMILY, label="complex no-absorber three-slot weld",
        kernel_label=KERNEL_NAME, replaces=tuple(REPLACES), slots=tuple(SLOTS),
        context=context, resolve_launch_args=resolve,
        launch_leading=launch_leading, launch_trailing=launch_trailing)


FUSED_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "cuda_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "fused magnetic pair",
        "coverage": _fused_magnetic_pair_coverage,
        "plan": _fused_magnetic_pair_plan,
        "module": "fused_magnetic_pair",
    },
    "cuda_complex_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "complex fused magnetic pair",
        "coverage": _complex_fused_magnetic_pair_coverage,
        "plan": _complex_fused_magnetic_pair_plan,
        "module": "complex_fused_magnetic_pair",
    },
    "cuda_cylindrical_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "cylindrical fused magnetic pair",
        "coverage": _cylindrical_fused_magnetic_pair_coverage,
        "plan": _cylindrical_fused_magnetic_pair_plan,
        "module": "cylindrical_fused_magnetic_pair",
    },
    # THE FIVE RESIDUAL MAGNETIC-SEAM ROWS, 2026-09-02, landed with their
    # products -- the last buildable cells on this board. ``step_B`` becomes an
    # EIGHT-candidate seam, and the disjointness proof is the same measured
    # partition the census's overlap check walks: the real pair refuses
    # complex64, beta and BFAST; the plain complex pair refuses folds, beta and
    # Dcyl; the FOLDED pair REQUIRES a fold and inherits the beta refusal; the
    # BETA pair REQUIRES grid.beta under complex storage (both complex siblings
    # refuse beta by name); the two Dcyl pairs split on storage and REQUIRE a
    # Dcyl grid every Cartesian product refuses; special_kz REQUIRES beta under
    # REAL storage; BFAST REQUIRES bfast_active, which every other product
    # refuses through the certified real predicate's own clause.
    "cuda_complex_folded_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "folded complex fused magnetic pair",
        "coverage": _complex_folded_fused_magnetic_pair_coverage,
        "plan": _complex_folded_fused_magnetic_pair_plan,
        "module": "complex_folded_fused_magnetic_pair",
    },
    "cuda_complex_beta_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "complex beta fused magnetic pair",
        "coverage": _complex_beta_fused_magnetic_pair_coverage,
        "plan": _complex_beta_fused_magnetic_pair_plan,
        "module": "complex_beta_fused_magnetic_pair",
    },
    "cuda_cylindrical_real_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "cylindrical real fused magnetic pair",
        "coverage": _cylindrical_real_fused_magnetic_pair_coverage,
        "plan": _cylindrical_real_fused_magnetic_pair_plan,
        "module": "cylindrical_real_fused_magnetic_pair",
    },
    "cuda_special_kz_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "special_kz fused magnetic pair",
        "coverage": _special_kz_fused_magnetic_pair_coverage,
        "plan": _special_kz_fused_magnetic_pair_plan,
        "module": "special_kz_fused_magnetic_pair",
    },
    "cuda_bfast_fused_magnetic_pair": {
        "curl_slot": "step_B",
        "label": "BFAST fused magnetic pair",
        "coverage": _bfast_fused_magnetic_pair_coverage,
        "plan": _bfast_fused_magnetic_pair_plan,
        "module": "bfast_fused_magnetic_pair",
    },
    # THE FIRST ROW ON ``step_D``. Its ``curl_slot`` is what keeps it out of the
    # magnetic loop: the three rows above are skipped when ``install_fused_pairs``
    # reaches ``step_D`` and this one when it reaches ``step_B``, so the two labels it
    # shares with ``cuda_fused_magnetic_pair`` can never make a seam ambiguous.
    "cuda_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "fused electric pair",
        "coverage": _fused_electric_pair_coverage,
        "plan": _fused_electric_pair_plan,
        "module": "fused_electric_pair",
    },
    # THE DISPERSIVE TWIN OF THE ROW ABOVE, 2026-09-02, and the two are the closest
    # pair on this table: one product with one half exchanged, sharing the whole
    # ownership-inversion carry through direct imports rather than a copy. It is
    # ALSO the pair with the tightest disjointness margin, which is why the margin is
    # a single boolean nobody has to reason about at the seam: the ordinary
    # constitutive predicate refuses ``fields.polarizations`` truthy and the
    # dispersive one requires it. An overlap would not add coverage -- it would leave
    # the seam UNFUSED and cost the 79 rows the row above serves.
    "cuda_dispersive_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "dispersive fused electric pair",
        "coverage": _dispersive_fused_electric_pair_coverage,
        "plan": _dispersive_fused_electric_pair_plan,
        "module": "dispersive_fused_electric_pair",
    },
    # THE TWO NO-ABSORBER / CONDUCTIVE ROWS, 2026-09-02, which close the last
    # POINTWISE-BUILDABLE cells this board had on the D->E seam. Both are the first
    # rows on this table whose curl half is the THREE-HISTORY conductive recurrence,
    # and the first whose predicates were written against a driver that rescales its
    # condinv injection SPARSELY -- the change that made a conductive row servable at
    # all (see each module's docstring).
    #
    # THE FIRST IS ALSO THE FIRST ROW ANYWHERE ON THIS TABLE TO DECLARE A NON-DEFAULT
    # ``REPAIR_PATHS``. Its constitutive half runs ``update_E``'s PURE OVERWRITE
    # (stepping.py:1019-1022) rather than the split-field accumulation, so the repair
    # that inverts it is ``deposit_repair.PLAIN_PATH``; ``install_fused_pairs`` reads
    # that declaration off the module and hands it to the bracket, and a product that
    # declared the wrong one would be REFUSED by ``deposit_repair.repairable`` rather
    # than mis-repair.
    "cuda_no_pml_dispersive_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "no-PML dispersive fused electric pair",
        "coverage": _no_pml_dispersive_fused_electric_pair_coverage,
        "plan": _no_pml_dispersive_fused_electric_pair_plan,
        "module": "no_pml_dispersive_fused_electric_pair",
    },
    "cuda_conductive_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "conductive fused electric pair",
        "coverage": _conductive_fused_electric_pair_coverage,
        "plan": _conductive_fused_electric_pair_plan,
        "module": "conductive_fused_electric_pair",
    },
    # THE TWO COMPLEX ELECTRIC ROWS, which make ``step_D`` a three-candidate seam the
    # way ``step_B`` already is. WHAT MAKES THREE ROWS LEGAL IS THE SAME DISJOINTNESS
    # PROOF the magnetic seam rests on, restated for this seam because it is a
    # different set of predicates: ``covers_real_pml_curl(..., "step_D")`` refuses
    # complex64 storage by name, ``covers_real_pml_complex_curl`` refuses a run that is
    # neither ``force_complex_fields`` nor Bloch-phased AND refuses a Dcyl grid, and
    # ``covers_pml_cylindrical_complex_curl`` REQUIRES a Dcyl grid at |m| >= 1. Two
    # admitters on one seam leaves it UNFUSED naming both, so an overlap here would
    # cost slots rather than gain them.
    "cuda_complex_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "complex fused electric pair",
        "coverage": _complex_fused_electric_pair_coverage,
        "plan": _complex_fused_electric_pair_plan,
        "module": "complex_fused_electric_pair",
    },
    "cuda_cylindrical_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "cylindrical fused electric pair",
        "coverage": _cylindrical_fused_electric_pair_coverage,
        "plan": _cylindrical_fused_electric_pair_plan,
        "module": "cylindrical_fused_electric_pair",
    },
    # THE FOURTH ``step_D`` ROW, 2026-08-31, and the FIRST product on this track that
    # refuses an absorber rather than requiring one. THE DISJOINTNESS PROOF IS THE
    # CHEAPEST ON THE BOARD and it is stated here because it is a different shape from
    # the three above: those three partition each other on STORAGE and on the grid
    # (``covers_real_pml_curl`` refuses complex64 by name; ``covers_real_pml_complex_curl``
    # refuses a run that is neither ``force_complex_fields`` nor Bloch-phased AND
    # refuses Dcyl; ``covers_pml_cylindrical_complex_curl`` REQUIRES Dcyl at |m| >= 1),
    # while all three ALSO refuse a run with no active layer by name. This one requires
    # exactly that, so it cannot co-admit with any of them whatever the storage or the
    # grid. Two admitters would leave the seam UNFUSED naming both, so the partition is
    # what buys the four slots rather than a tidiness.
    #
    # IT IS ALSO THE FIRST ROW WHOSE MODULE DECLARES ``CARRIES_DEPOSIT_REPAIR = False``
    # -- measured, not defaulted: all four rows of its board cell declare a MAGNETIC
    # source only, so the D seam is empty on every one of them and
    # ``_install_fused_pair`` always takes its ``NoopPlan`` branch here.
    "cuda_no_pml_complex_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "complex no-absorber fused electric pair",
        "coverage": _no_pml_complex_fused_electric_pair_coverage,
        "plan": _no_pml_complex_fused_electric_pair_plan,
        "module": "no_pml_complex_fused_electric_pair",
    },
    # THE TWO ROWS ON ``update_E`` -- the first products on the E->P seam, 2026-09-01,
    # and the first whose leading slot is a constitutive sub-step rather than a
    # curl (the table's ``curl_slot`` key keeps its name as the SEAM-LEADING
    # slot). THE DISJOINTNESS PROOF IS THE ABSORBER BOOLEAN, stated at the two
    # coverage closures: the PML product's halves require an active layer and the
    # no-absorber product's refuse one by name, so no configuration can admit
    # both, and the D_to_E products never collide with either because they bid on
    # a DIFFERENT seam (``install_fused_pairs`` decides one seam at a time).
    "cuda_fused_polarization_pair": {
        "curl_slot": "update_E",
        "label": "fused polarization pair",
        "coverage": _fused_polarization_pair_coverage,
        "plan": _fused_polarization_pair_plan,
        "module": "fused_polarization_pair",
    },
    "cuda_no_pml_fused_polarization_pair": {
        "curl_slot": "update_E",
        "label": "no-absorber fused polarization pair",
        "coverage": _no_pml_fused_polarization_pair_coverage,
        "plan": _no_pml_fused_polarization_pair_plan,
        "module": "fused_polarization_pair",
    },
    # THE FIVE ELECTRIC TWINS, 2026-09-02 residue round -- each the D-side twin
    # of a residual magnetic weld, keyed to `step_D` so the label pair it shares
    # with its magnetic twin can never make one seam ambiguous. On EVERY one the
    # flag is the product: every row of every twin's cell declares an ELECTRIC
    # source inside the seam (measured off the residualwelds census), so any of
    # the five with CARRIES_DEPOSIT_REPAIR at False would compile, gate green on
    # every arithmetic leg and serve nothing.
    "cuda_complex_folded_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "folded complex fused electric pair",
        "coverage": _complex_folded_fused_electric_pair_coverage,
        "plan": _complex_folded_fused_electric_pair_plan,
        "module": "complex_folded_fused_electric_pair",
    },
    "cuda_complex_beta_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "complex beta fused electric pair",
        "coverage": _complex_beta_fused_electric_pair_coverage,
        "plan": _complex_beta_fused_electric_pair_plan,
        "module": "complex_beta_fused_electric_pair",
    },
    "cuda_cylindrical_real_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "cylindrical real fused electric pair",
        "coverage": _cylindrical_real_fused_electric_pair_coverage,
        "plan": _cylindrical_real_fused_electric_pair_plan,
        "module": "cylindrical_real_fused_electric_pair",
    },
    "cuda_special_kz_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "special_kz fused electric pair",
        "coverage": _special_kz_fused_electric_pair_coverage,
        "plan": _special_kz_fused_electric_pair_plan,
        "module": "special_kz_fused_electric_pair",
    },
    "cuda_bfast_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "BFAST fused electric pair",
        "coverage": _bfast_fused_electric_pair_coverage,
        "plan": _bfast_fused_electric_pair_plan,
        "module": "bfast_fused_electric_pair",
    },
    # THE THIRD E->P ROW, 2026-09-02: the complex no-absorber polarization pair
    # over the cells the residue round's fitness entry determined POINTWISE.
    # Like its two REAL siblings its seam carries nothing (FUSED_PAIR_SEAMS'
    # update_E row is None), so _install_fused_pair always takes the NoopPlan
    # branch here and the module declares CARRIES_DEPOSIT_REPAIR = False.
    "cuda_complex_no_pml_fused_polarization_pair": {
        "curl_slot": "update_E",
        "label": "complex no-PML fused polarization pair",
        "coverage": _complex_no_pml_fused_polarization_pair_coverage,
        "plan": _complex_no_pml_fused_polarization_pair_plan,
        "module": "complex_no_pml_fused_polarization_pair",
    },
    # THE TWO STENCIL WELDS, 2026-09-02, landed with their products -- the first
    # products on this table whose board cells were recorded UNBUILDABLE rather
    # than merely unbuilt. ``step_D`` becomes an ELEVEN-candidate seam, and the
    # partition is the measured one the census's overlap check walks: the seven
    # products already there all take the DIAGONAL constitutive arm, which
    # ``covers_real_pml_constitutive(side='E')`` refuses on every off-diagonal run
    # by its own clause, and these two require exactly that; between themselves
    # they split on the fold, which one refuses twice and the other requires.
    "cuda_offdiag_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "off-diagonal stencil weld",
        "coverage": _offdiag_fused_electric_pair_coverage,
        "plan": _offdiag_fused_electric_pair_plan,
        "module": "offdiag_fused_electric_pair",
    },
    "cuda_folded_offdiag_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "folded off-diagonal stencil weld",
        "coverage": _folded_offdiag_fused_electric_pair_coverage,
        "plan": _folded_offdiag_fused_electric_pair_plan,
        "module": "folded_offdiag_fused_electric_pair",
    },
    # THE LAST UNSERVED E->P CELL, 2026-09-02, landed with its product. The
    # ``update_E`` seam becomes a FOUR-candidate one and the partition is a
    # single measured clause: the three products already there conjoin an
    # update_E half that refuses an off-diagonal chi1inv row BY NAME, and this
    # one REQUIRES one.
    "cuda_dispersive_offdiag_fused_polarization_pair": {
        "curl_slot": "update_E",
        "label": "dispersive off-diagonal polarization pair",
        "coverage": _dispersive_offdiag_fused_polarization_pair_coverage,
        "plan": _dispersive_offdiag_fused_polarization_pair_plan,
        "module": "dispersive_offdiag_fused_polarization_pair",
    },
    # THE TWO COMPLEX STENCIL WELDS, 2026-09-02, and the LAST two cells this board
    # recorded UNBUILDABLE. ``step_D`` becomes a THIRTEEN-candidate seam and the
    # partition is the measured one the census's overlap check walks: every one of
    # the eleven products already there either refuses complex64 storage BY NAME or
    # refuses an off-diagonal chi1inv row BY NAME, and these two require both --
    # so no run can be claimed by one of them and one of these. Between themselves
    # they split on the ABSORBER, which one constitutive arm requires active and
    # the other requires inert.
    "cuda_folded_complex_offdiag_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "folded complex off-diagonal stencil weld",
        "coverage": _folded_complex_offdiag_fused_electric_pair_coverage,
        "plan": _folded_complex_offdiag_fused_electric_pair_plan,
        "module": "folded_complex_offdiag_fused_electric_pair",
    },
    "cuda_complex_no_pml_offdiag_fused_electric_pair": {
        "curl_slot": "step_D",
        "label": "complex no-absorber off-diagonal stencil weld",
        "coverage": _complex_no_pml_offdiag_fused_electric_pair_coverage,
        "plan": _complex_no_pml_offdiag_fused_electric_pair_plan,
        "module": "complex_no_pml_offdiag_fused_electric_pair",
    },
    # THE TWO THREE-SLOT WELDS, 2026-09-02, and the first rows on this table
    # carrying a ``slots`` key: every other product's span is exactly
    # ``(curl_slot, FUSED_PAIR_SEAMS[curl_slot][0])``, and these two continue past
    # that into ``update_P``. The key is spelled HERE rather than imported from the
    # module for the reason every other declaration on this table is: the composer
    # must read it on the merge-bar host, before any product module is imported.
    # ``test_three_slot_dispersive_weld.py`` pins each row equal to its module's own
    # ``SLOTS``, so the two cannot drift.
    #
    # THEY BID AT ``step_D`` LIKE THE ELEVEN D->E ROWS ABOVE, and their predicates
    # are CONJUNCTIONS of a D->E predicate and an E->P one -- so each one's
    # admission set is a SUBSET of the D->E product it supersedes. That would be
    # two admitters on one seam, which :func:`install_fused_pairs` leaves UNFUSED
    # naming both; :func:`_superseded_by_a_longer_span` is what resolves it, by
    # refusing the SHORTER span by name. Non-lossy in both directions: where a
    # three-slot weld admits, its two-slot half does too (so nothing is left
    # unserved), and where it refuses, the pair installs exactly as it does today.
    "cuda_three_slot_dispersive_weld": {
        "curl_slot": "step_D",
        "label": "three-slot dispersive weld",
        "slots": ("step_D", "update_E", "update_P"),
        "coverage": _three_slot_dispersive_weld_coverage,
        "plan": _three_slot_dispersive_weld_plan,
        "module": "three_slot_dispersive_weld",
    },
    "cuda_three_slot_no_pml_dispersive_weld": {
        "curl_slot": "step_D",
        "label": "no-absorber three-slot dispersive weld",
        "slots": ("step_D", "update_E", "update_P"),
        "coverage": _no_pml_three_slot_dispersive_weld_coverage,
        "plan": _no_pml_three_slot_dispersive_weld_plan,
        "module": "no_pml_three_slot_dispersive_weld",
    },
    # THE THIRD THREE-SLOT ROW, 2026-09-04: the COMPLEX no-absorber weld over the
    # four TestLoadDump 3-D rows the real welds' record named as a different shape.
    # It supersedes ``cuda_no_pml_complex_fused_electric_pair`` by the same
    # strict-containment rule and takes ``update_E`` from
    # ``cuda_complex_no_pml_fused_polarization_pair`` through the existing
    # ``_pair_may_absorb`` clause; nothing new is arbitrated. ONE device group
    # (the trailing half answers the consult and launches nothing), because the
    # seam is empty by predicate.
    "cuda_three_slot_complex_no_pml_dispersive_weld": {
        "curl_slot": "step_D",
        "label": "complex no-absorber three-slot weld",
        "slots": ("step_D", "update_E", "update_P"),
        "coverage": _complex_no_pml_three_slot_weld_coverage,
        "plan": _complex_no_pml_three_slot_weld_plan,
        "module": "complex_no_pml_three_slot_dispersive_weld",
    },
    # THE FIRST ROW ON THE ``update_H`` SEAM, 2026-09-06, and the only one: the H->D
    # weld is the sole candidate its seam has, so ``install_fused_pairs``'s ambiguity
    # rule can never fire there. ``curl_slot`` is a misnomer on this row and is kept
    # because it is the table's key for "the FIRST slot of the seam this product
    # opens" -- here a CONSTITUTIVE sub-step, which is why the seam's second half is
    # a curl and the tuple order in ``FUSED_PAIR_ARMS`` reads (constitutive, curl).
    #
    # IT IS REFUSED ON EVERY CONFIGURATION AND THE ROW IS STILL WHAT MAKES THAT TRUE.
    # ``_declared_uninstallable`` reads ``fused_hd_pair.INSTALLABLE`` before the
    # predicate is asked and reports ``INSTALLABLE_REASON`` on every run, so the seam
    # loop names the refusal rather than leaving the product invisible -- and the
    # refusal it names is a measured verdict about the COMPOSITION (a span taking one
    # slot from each neighbour can only tie or lose over the four-slot path), never
    # about the arithmetic, which the family's own device gate measures.
    "cuda_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "fused H/D pair",
        "coverage": _fused_hd_pair_coverage,
        "plan": _fused_hd_pair_plan,
        "module": "fused_hd_pair",
    },
    # THE TWO CYLINDRICAL H->D PRODUCTS, 2026-09-07: two launches each (the
    # certified constitutive recompute plus the fused pre-cumsum increment in the
    # update_H slot, xp.cumsum untouched on the array path, the certified
    # cylindrical curl in step_D). Both declare INSTALLABLE = False, so
    # ``_declared_uninstallable`` refuses them on every configuration before the
    # predicate is asked; the composer measured with these rows patched in-process
    # (each gate's arbitration leg) selects the same composition as without them.
    "cuda_cylindrical_real_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "cylindrical real fused H/D pair",
        "coverage": _cylindrical_real_fused_hd_pair_coverage,
        "plan": _cylindrical_real_fused_hd_pair_plan,
        "module": "cylindrical_real_fused_hd_pair",
    },
    "cuda_cylindrical_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "cylindrical complex fused H/D pair",
        "coverage": _cylindrical_fused_hd_pair_coverage,
        "plan": _cylindrical_fused_hd_pair_plan,
        "module": "cylindrical_fused_hd_pair",
    },
    # THE CARTESIAN COMPLEX H->D PRODUCT, 2026-09-07: one launch (the certified
    # complex constitutive recompute at the thread's own cell and at every backward
    # neighbour the curl taps, into launch-local scratch, then the certified complex
    # curl, then the H/f_w_H rotation), in TWO variants -- the plain complex curl and
    # the folded one, selected from the grid. It declares INSTALLABLE = False, so
    # ``_declared_uninstallable`` refuses it on every configuration before the
    # predicate is asked; the composer measured with this row patched in-process (the
    # gate's arbitration leg) selects the same composition as without it.
    "cuda_complex_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "complex fused H/D pair",
        "coverage": _complex_fused_hd_pair_coverage,
        "plan": _complex_fused_hd_pair_plan,
        "module": "complex_fused_hd_pair",
    },
    # THE TWO BETA H->D PRODUCTS, 2026-09-07: one launch each (the certified
    # constitutive at the thread's own cell and at every backward neighbour the curl
    # taps, into launch-local scratch, then the certified BETA step_D body with its
    # twelve magnetic reads redirected, then the H/f_w_H rotation). Both declare
    # INSTALLABLE = False, so ``_declared_uninstallable`` refuses them on every
    # configuration before the predicate is asked; each gate's arbitration leg
    # measured the composer WITH these rows patched in-process and it selects the
    # same composition as without them.
    #
    # WHAT THEY ADD THAT NO OTHER ROW ON THIS SEAM HAS is the BETA PARTNER: the
    # analytic d/dz term consumes the SAME-CELL magnetic snapshot, which the array
    # path takes AFTER update_H, so welded it must be the launch's own register.
    # Both modules assert that pairing on every emit and both gates arm it.
    "cuda_special_kz_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "special_kz fused H/D pair",
        "coverage": _special_kz_fused_hd_pair_coverage,
        "plan": _special_kz_fused_hd_pair_plan,
        "module": "special_kz_fused_hd_pair",
    },
    "cuda_complex_beta_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "complex beta fused H/D pair",
        "coverage": _complex_beta_fused_hd_pair_coverage,
        "plan": _complex_beta_fused_hd_pair_plan,
        "module": "complex_beta_fused_hd_pair",
    },
    # THE LAST TWO H->D CELLS, 2026-09-07: ONE launch, TWO variants (the conductive
    # four-case PML recurrence and the BFAST insert), the variant read off the run by
    # ``variant_for`` rather than taken from a caller. It declares INSTALLABLE = False,
    # so ``_declared_uninstallable`` refuses it on every configuration before the
    # predicate is asked; the composer measured with this row patched in-process (the
    # gate's arbitration leg, results/cuda_conductive_bfast_fused_hd_pair_2026-09-07)
    # selects the same composition as without it, and on BOTH cells both released
    # neighbours keep the slots they had.
    "cuda_conductive_bfast_fused_hd_pair": {
        "curl_slot": "update_H",
        "label": "conductive/BFAST fused H/D pair",
        "coverage": _conductive_bfast_fused_hd_pair_coverage,
        "plan": _conductive_bfast_fused_hd_pair_plan,
        "module": "conductive_bfast_fused_hd_pair",
    },
}


class CudaFusedPairPlan:
    """One fused product, holding both of its slots.

    ``replaces`` is the DRIVER PASSES the single launch performs, declared by the
    product itself (``fused_magnetic_pair.REPLACES``) and not inferred from the two
    slot names -- ``fill_symmetry_bc_B``, ``zero_metal_B`` and
    ``fill_folded_far_ghosts_B`` are all carried in registers and appear in neither.
    A composition that reported only the slots would say those three ran on the
    array path when the kernel performed them.

    ``launchable`` follows ``arms.CudaSlotPlan``'s meaning exactly: is there an
    established resolution from this composition to the launch arguments. It says
    nothing about whether a device is present, and :meth:`run` is the only thing
    here that needs one.
    """

    __slots__ = ("family", "label", "kernel_label", "replaces", "slots",
                 "resolve_launch_args", "launch_kernel", "_context", "launches",
                 "last_launch", "warm_module")

    def __init__(self, family: str, label: str, kernel_label: str,
                 replaces: Tuple[str, ...], slots: Tuple[str, str],
                 context: Any,
                 resolve_launch_args: Optional[Callable[..., Dict[str, Any]]] = None,
                 launch: Optional[Callable[..., Dict[str, Any]]] = None,
                 ) -> None:
        self.family = family
        self.label = label
        self.kernel_label = kernel_label
        self.replaces = tuple(replaces)
        self.slots = tuple(slots)
        self._context = context
        self.resolve_launch_args = resolve_launch_args
        #: The product's OWN launcher, supplied by its plan builder and never
        #: defaulted. It became a parameter on 2026-08-30, when the second and third
        #: products landed: while this class named one module's launcher directly,
        #: every new product would have launched the REAL pair's kernel against its
        #: own resolved arguments -- which is not a crash on the first two arguments
        #: and is a wrong answer on the rest. A product with no launcher cannot run,
        #: and :meth:`run` says so by name rather than falling back to anyone's.
        self.launch_kernel = launch
        self.launches = 0
        #: The launcher's OWN report of the last launch it performed -- every
        #: launcher on this track returns ``{"launched", "blocks", "threads", ...}``.
        #: Kept so :attr:`launch_grid` can answer with a MEASURED grid rather than a
        #: predicted one; ``None`` before the first launch, which is the honest
        #: answer at plan time and is what the dispatch record's
        #: ``programs_per_dispatch`` reports until a step has run.
        self.last_launch: Optional[Dict[str, Any]] = None
        #: The family module :meth:`warm` compiles through, written by the installer
        #: off ``FUSED_PRODUCTS[family]["module"]``. Not read at plan build: the
        #: import is inside :meth:`warm`, so a host with no cupy can still compose.
        self.warm_module: Optional[str] = None

    #: The alias the shared protocol reads. ``metal_kernels.launch.declaring_plan``
    #: and its Triton twin read ``replaces_sub_steps`` off whatever occupies a slot,
    #: so a wrapper installed over this plan reports the pair's whole seam rather
    #: than its own one slot.
    @property
    def replaces_sub_steps(self) -> Tuple[str, ...]:
        return self.replaces

    @property
    def launchable(self) -> bool:
        return self.resolve_launch_args is not None and self.launch_kernel is not None

    def run(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """The single launch. NEEDS A DEVICE, and nothing in a plan build calls it.

        Held here rather than in the installer so the two repair plans wrap a
        LAUNCH -- ``LeadingRepairPlan.run`` saves and then calls ``inner.run`` --
        which is what makes the bracket a bracket rather than a label.

        BOTH HALVES OF THE PAIR ARE THE PRODUCT'S OWN. The argument resolution and
        the launcher come from the plan builder together, because they are two ends
        of one signature: a resolution paired with somebody else's launcher binds the
        right values to the wrong parameters.

        THE RESOLVER IS ASKED ON EVERY LAUNCH AND DECIDES WHAT IT RECOMPUTES,
        2026-09-19. A builder whose resolver reads only the frozen configuration
        wraps it in :func:`_resolved_once`; one whose resolver hands out rotating
        scratch or reads the fields does not, and this method treats the two alike,
        so a resolve-once here could not be applied to the second kind by accident.
        """
        if self.resolve_launch_args is None:
            raise RuntimeError(
                f"{self.family} has no established launch-argument resolution; a "
                f"guessed one is a half-cell error away from converged, smooth and "
                f"wrong")
        if self.launch_kernel is None:
            raise RuntimeError(
                f"{self.family} has no launcher; a fused product may not borrow "
                f"another product's, which would bind this configuration's values "
                f"to a different kernel's parameters")
        arguments = self.resolve_launch_args(self._context)
        result = self.launch_kernel(self._context.fields, arguments)
        self.launches += 1
        if isinstance(result, dict):
            self.last_launch = result
        return result

    @property
    def launch_grid(self) -> Optional[Tuple[int, ...]]:
        """The single-axis grid the LAST launch actually asked for, or ``None``.

        READ BACK OFF THE LAUNCHER'S REPORT rather than recomputed here. The
        dispatch seam reports it as ``programs_per_dispatch`` and uses it to tell "a
        kernel launched" from "a kernel computed over data" -- a distinction a
        predicted grid cannot draw, because the number that matters is the one the
        launch was given. ``None`` before the first launch means UNKNOWN, which is
        exactly what it is.
        """
        blocks = (self.last_launch or {}).get("blocks")
        try:
            return (int(blocks),)
        except (TypeError, ValueError):
            return None

    def warm(self) -> Optional[str]:
        """Compile this product's kernel at PLAN TIME, launching nothing.

        WHY A COMPILE AND NOT AN EMPTY-GRID LAUNCH. The Triton mechanism empties a
        launch grid so the JIT compiles at the subscript and enqueues zero programs;
        these kernels are NVRTC sources behind ``compile_cache``, and the compile is
        a separate call that touches no field array at all. So the warm pass here is
        strictly weaker in effect and strictly stronger in safety: nothing is
        enqueued, no argument is resolved, and no per-step behaviour changes.

        RETURNS A REASON instead of raising when the family module exposes no
        ``_get_kernel``. A raise unfills the slot, which would take a working
        product to the array path over the absence of a warm mechanism; a reason is
        RECORDED as unwarmed, which is the same answer the Triton composer's
        no-settable-attribute plans get.
        """
        if not self.warm_module:
            return (f"{self.family} declares no family module to compile through; "
                    f"the kernel compiles at its first launch instead")
        try:
            module = importlib.import_module(f".{self.warm_module}", __package__)
        except Exception as exc:  # noqa: BLE001 - an unimportable family is not a failure here
            return (f"{self.warm_module} is not importable in this process "
                    f"({exc!r}); the kernel compiles at its first launch instead")
        getter = getattr(module, "_get_kernel", None)
        if not callable(getter):
            return (f"{self.warm_module} exposes no _get_kernel, so there is no "
                    f"compile this pass can perform ahead of the first launch")
        # A GETTER THAT NEEDS A SHAPE HAS NO ARGUMENT-FREE COMPILE, and that is a
        # REASON, not a failure -- the same answer as a family with no ``_get_kernel``
        # at all, for the same cause. Several families emit their source PER SHAPE
        # (``three_slot_dispersive_weld._get_kernel(component_index, count,
        # sigma_kinds)``; the conductive and complex pairs take an arm), so there is
        # no body to compile until the launch resolves one.
        #
        # WHAT THIS COST BEFORE IT WAS CHECKED. ``getter()`` raised ``TypeError`` on
        # the missing arguments, the exception left ``warm_plan``, rung (9) removed
        # the slot from the dispatch set, and ``_split_pairs`` then refused the WHOLE
        # plan because a three-slot weld holding two of its three slots would skip or
        # double-apply a sub-step. The case fell to the array path and reported "the
        # fused product occupying step_D, update_E, update_P would dispatch only
        # step_D, update_E" -- a sentence about the composition, for a defect in the
        # warm pass. Tested by signature rather than by catching TypeError, so a
        # TypeError raised INSIDE a zero-argument getter is still a real failure.
        try:
            inspect.signature(getter).bind()
        except TypeError:
            return (f"{self.warm_module}._get_kernel"
                    f"{inspect.signature(getter)} takes required arguments: this "
                    f"family emits its source per shape, so there is no "
                    f"argument-free compile this pass can perform and the kernel "
                    f"compiles at its first launch instead")
        except (ValueError, AttributeError):  # noqa: PERF203 - unreadable signature
            return (f"{self.warm_module}._get_kernel exposes no readable signature, "
                    f"so this pass cannot tell an argument-free compile from one "
                    f"that needs a shape; the kernel compiles at its first launch")
        getter().compile()
        return None

    def __repr__(self) -> str:
        return (f"CudaFusedPairPlan({self.family}/{self.kernel_label}, "
                f"slots={list(self.slots)}, replaces={list(self.replaces)}, "
                f"launchable={self.launchable})")


class _TripleHalfPlan:
    """ONE device group of a three-slot product, as the object a slot holds.

    A three-slot weld performs its work in TWO groups with the deposit repair
    between them, so the thing in ``step_D`` and the thing in ``update_P`` cannot be
    one object with one ``run``. This is the adapter: it holds the triple and which
    of its two launchers to call.

    IT DECLARES ``replaces_sub_steps`` AND ``launchable`` ITSELF, delegating to the
    triple, and that is not decoration. ``declaring_plan`` follows ``absorbed_by``
    exactly ONE level, so for ``plans['step_D'] = LeadingRepairPlan(half, ...)`` the
    declaring plan IS this object -- read bare it would fall back to "replaces only
    my own slot" and the composition would report five driver passes as running on
    the array path when the kernel performed them.
    """

    __slots__ = ("triple", "which", "absorbed_by", "launches", "last_launch")

    def __init__(self, triple: "CudaFusedTriplePlan", which: str) -> None:
        self.triple = triple
        self.which = which
        #: The product that did the work, for a reader inspecting the composition.
        self.absorbed_by = triple
        self.launches = 0
        #: THIS HALF'S own last launch report. Per half rather than per triple
        #: because the two groups ask for different grids, and reporting the
        #: leading group's count for the trailing slot would be a claim about a
        #: launch that did not happen there.
        self.last_launch: Optional[Dict[str, Any]] = None

    @property
    def replaces_sub_steps(self) -> Tuple[str, ...]:
        return self.triple.replaces

    @property
    def launchable(self) -> bool:
        return self.triple.launchable

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.triple.run_half(self.which, *args, **kwargs)
        self.launches += 1
        if isinstance(result, dict):
            self.last_launch = result
        return result

    @property
    def launch_grid(self) -> Optional[Tuple[int, ...]]:
        """This half's measured grid. See :attr:`CudaFusedPairPlan.launch_grid`."""
        blocks = (self.last_launch or {}).get("blocks")
        try:
            return (int(blocks),)
        except (TypeError, ValueError):
            return None

    def warm(self) -> Optional[str]:
        """Compile the triple's kernel once, through the product it belongs to."""
        return self.triple.warm()

    def __repr__(self) -> str:
        return f"_TripleHalfPlan({self.triple.family}/{self.which})"


class CudaFusedTriplePlan:
    """One fused product holding THREE slots and TWO device groups.

    THE SHAPE THAT IS NEW. Every :class:`CudaFusedPairPlan` is one launch bracketed
    by the repair: save, launch, (the driver's passes), apply -- and nothing of the
    product's runs afterwards. A three-slot weld has a SECOND device group after the
    apply, and the order is load-bearing rather than tidy:
    ``deposit_repair.apply`` recomputes the constitutive product through
    ``Fields.displacement_minus_polarization`` on both of its paths
    (deposit_repair.py:781-782, :922-923), which subtracts ``state.P[c]`` AS IT
    STANDS (fields.py:1096-1105). So the repair is only correct while P is still
    P^n, and the polarization advance may not run before it.

    ``replaces`` is the DRIVER PASSES the two groups perform together, declared by
    the product itself and not inferred from the three slot names -- ``zero_metal_D``
    and the two mirror fills are carried in registers and appear in none of them.

    ``launchable`` follows ``arms.CudaSlotPlan``'s meaning exactly and requires BOTH
    launchers: a product that could run group 1 and not group 2 would leave the
    polarizations unadvanced while ``replaces`` said ``update_P`` had run.
    """

    __slots__ = ("family", "label", "kernel_label", "replaces", "slots", "_context",
                 "resolve_launch_args", "launch_leading", "launch_trailing",
                 "leading", "trailing", "launches", "warm_module")

    def __init__(self, family: str, label: str, kernel_label: str,
                 replaces: Tuple[str, ...], slots: Tuple[str, str, str],
                 context: Any,
                 resolve_launch_args: Optional[Callable[..., Dict[str, Any]]] = None,
                 launch_leading: Optional[Callable[..., Dict[str, Any]]] = None,
                 launch_trailing: Optional[Callable[..., Dict[str, Any]]] = None,
                 ) -> None:
        if len(tuple(slots)) != 3:
            raise ValueError(
                f"a fused triple owns three slots; {family} declared "
                f"{tuple(slots)!r}")
        self.family = family
        self.label = label
        self.kernel_label = kernel_label
        self.replaces = tuple(replaces)
        self.slots = tuple(slots)
        self._context = context
        self.resolve_launch_args = resolve_launch_args
        self.launch_leading = launch_leading
        self.launch_trailing = launch_trailing
        self.leading = _TripleHalfPlan(self, "leading")
        self.trailing = _TripleHalfPlan(self, "trailing")
        self.launches = 0
        #: See :attr:`CudaFusedPairPlan.warm_module`. One module for both groups:
        #: a three-slot weld is ONE kernel source launched twice.
        self.warm_module: Optional[str] = None

    @property
    def replaces_sub_steps(self) -> Tuple[str, ...]:
        return self.replaces

    @property
    def launchable(self) -> bool:
        return (self.resolve_launch_args is not None
                and self.launch_leading is not None
                and self.launch_trailing is not None)

    def run_half(self, which: str, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """One of the two device groups. NEEDS A DEVICE; no plan build calls it.

        THE RESOLVER IS ASKED PER GROUP, exactly as :meth:`CudaFusedPairPlan.run`
        asks it per launch, and what it recomputes is its builder's decision. Nothing
        that rotates is in these arguments -- the pole buffers are read inside each
        launcher, immediately before the launch that binds them -- so the two
        dispersive welds hold theirs for the plan's life (:func:`_resolved_once`,
        2026-09-19): a configuration change re-freezes, and a re-freeze builds a new
        plan (``driver.py:3266-3271``), so nothing is carried across one.
        """
        launcher = {"leading": self.launch_leading,
                    "trailing": self.launch_trailing}.get(which)
        if which not in ("leading", "trailing"):
            raise ValueError(
                f"a fused triple has a 'leading' and a 'trailing' group, not "
                f"{which!r}")
        if self.resolve_launch_args is None:
            raise RuntimeError(
                f"{self.family} has no established launch-argument resolution; a "
                f"guessed one is a half-cell error away from converged, smooth and "
                f"wrong")
        if launcher is None:
            raise RuntimeError(
                f"{self.family} has no {which} launcher; a fused product may not "
                f"borrow another product's, which would bind this configuration's "
                f"values to a different kernel's parameters")
        arguments = self.resolve_launch_args(self._context)
        result = launcher(self._context.fields, arguments)
        self.launches += 1
        return result

    def warm(self) -> Optional[str]:
        """Compile this weld's kernel at plan time. See the pair plan's twin."""
        return CudaFusedPairPlan.warm(self)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (f"CudaFusedTriplePlan({self.family}/{self.kernel_label}, "
                f"slots={list(self.slots)}, replaces={list(self.replaces)}, "
                f"launchable={self.launchable})")


def _install_fused_pair(plans: Dict[str, Any], selected: Dict[str, str],
                        fields: Any, pml: Any, sources: Any,
                        pair_name: Optional[str], curl_name: str,
                        update_name: str, label: str, pair: Any,
                        repair_paths: Sequence[str] = (
                            _deposit_repair.SPLIT_FIELD_PATH,)) -> None:
    """Put a built fused pair into its two slots, with the deposit repair if it needs one.

    A MIRROR of ``metal_kernels.launch._install_fused_pair`` and
    ``triton_kernels.launch._install_fused_pair`` -- one copy per composer, three
    composers -- so a reader who has understood the protocol on one track has
    understood it on all three, and so ``deposit_repair`` is reached the same way
    from each. ONE FORCED DIVERGENCE since 2026-09-01: this composer carries the
    E->P seam, which neither sibling has a product on, and that seam's
    ``pair_name`` is ``None`` because the driver injects NOTHING between its two
    consults (driver.py:3313/:3315) -- there is no deposit list for
    ``deposit_repair.in_seam_sources`` to select, and handing it any string would
    select a list injected OUTSIDE the seam. ``None`` therefore reads as an
    unconditionally empty seam here, which is the driver's own answer.

    THE FLAG IS NOT CHECKED HERE, deliberately, and both siblings say the same: it
    is checked in the family's own coverage predicate, which passes
    ``carries_repair=CARRIES_DEPOSIT_REPAIR`` to
    ``deposit_repair.seam_source_reasons``. A product that has not declared it
    refuses every in-seam source, so a configuration with a deposit never arrives
    here. Checking it twice would let the two answers disagree.

    THE H->D SEAM IS ROUTED TO THE WITHDRAW HOIST, 2026-09-04, AND THE BRANCH IS
    HERE RATHER THAN IN A FAMILY. Its ``pair_name`` is ``withdraw_hoist.SEAM``,
    which is not a ``deposit_repair`` field letter, so there is no deposit list to
    select and no bracket to install; what sits between the two consults is the
    driver's electric ``withdraw`` loop, which a spanning product has to perform
    before its launch. Both tracks therefore read ONE protocol -- the installer
    decides which of the two seam mechanisms a seam name names -- rather than each
    family deciding for itself, which is the same argument that keeps the deposit
    bracket out of the families. There is no trailing plan: nothing is injected
    here, so the second slot holds the ``NoopPlan`` unconditionally, exactly as the
    E->P row's ``None`` does.

    ``repair_paths`` IS THE INSTALLING PRODUCT'S DECLARATION of which recurrence its
    repair inverts, threaded here on 2026-09-02 to match
    ``triton_kernels.launch._install_fused_pair``'s signature. It defaults to the
    split-field one -- the only repair that existed before
    ``deposit_repair.PLAIN_PATH``, and the one every product on this table shipped
    with until the no-absorber dispersive weld landed -- so every caller written
    before this parameter keeps its exact behaviour. A product whose halves refuse an
    active absorber runs ``update_E``'s plain overwrite instead and must say so:
    ``deposit_repair.repairable`` refuses by name any configuration whose recurrence
    is not among the paths its caller declares, so a wrong declaration is a REFUSAL
    rather than a silent mis-repair.
    """
    if pair_name == _withdraw_hoist.SEAM:
        plans[curl_name] = _withdraw_hoist.LeadingWithdrawPlan(
            pair, fields, sources, span=(curl_name, update_name))
        plans[update_name] = NoopPlan(update_name, pair)
        selected[curl_name] = label
        selected[update_name] = label
        return
    seam = (() if pair_name is None
            else _deposit_repair.in_seam_sources(sources, pair_name))
    if not seam:
        plans[curl_name] = pair
        plans[update_name] = NoopPlan(update_name, pair)
    else:
        leading = _deposit_repair.LeadingRepairPlan(pair, fields, pml, seam, pair_name,
                                                    repair_paths)
        plans[curl_name] = leading
        plans[update_name] = _deposit_repair.TrailingRepairPlan(
            update_name, leading, fields, pml)
    selected[curl_name] = label
    selected[update_name] = label


def _install_fused_triple(plans: Dict[str, Any], selected: Dict[str, str],
                          fields: Any, pml: Any, sources: Any,
                          pair_name: Optional[str], slots: Sequence[str],
                          label: str, triple: Any,
                          repair_paths: Sequence[str]) -> None:
    """Put a built THREE-slot product into its three slots, with the repair between.

    THE BRACKET IS :func:`_install_fused_pair`'S, UNCHANGED, over the LEADING half:
    the save at the first consult, the apply at the second, the driver's
    inject/fill/clear untouched in between. What is added is the third slot, and
    WHERE IT SITS RELATIVE TO THE BRACKET IS THE PRODUCT: the trailing half goes in
    ``update_P``, which the driver consults AFTER ``update_E``, so the polarization
    advance runs after ``deposit_repair.apply`` has read ``state.P`` as P^n.

    Advancing P earlier -- inside the leading launch, or at the ``update_E`` consult
    instead of the repair -- computes, converges and is wrong only at the deposit
    cells (``parity/meep_gpu/probe_cuda_three_slot_weld.py``: ``p_inside_the_launch``
    and ``repair_after_p``, each diverging on 6 of 7 configurations by 7 to 60 words
    over 6 complete driver steps). That is why the third slot is written HERE, beside
    the bracket, rather than by a caller free to order the two differently.

    ``pair_name`` IS THE LEADING SEAM'S, not the trailing one's. A three-slot product
    spans two seams and only the first has an injection in it: the driver puts
    NOTHING between ``update_E`` and ``update_P`` (driver.py:3332/:3334), which is the
    ``None`` in ``FUSED_PAIR_SEAMS``' ``update_E`` row. So exactly one bracket is
    installed, on the seam that has a deposit to carry.
    """
    curl_name, update_name, tail_name = (str(name) for name in slots)
    seam = (() if pair_name is None
            else _deposit_repair.in_seam_sources(sources, pair_name))
    if not seam:
        plans[curl_name] = triple.leading
        plans[update_name] = NoopPlan(update_name, triple.leading)
    else:
        leading = _deposit_repair.LeadingRepairPlan(
            triple.leading, fields, pml, seam, pair_name, repair_paths)
        plans[curl_name] = leading
        plans[update_name] = _deposit_repair.TrailingRepairPlan(
            update_name, leading, fields, pml)
    plans[tail_name] = triple.trailing
    for name in (curl_name, update_name, tail_name):
        selected[name] = label


def declaring_plan(plan: Any) -> Any:
    """The plan whose DECLARATIONS describe the device work sitting in a slot.

    After a pair is installed, two of the occupants are WRAPPERS -- the
    ``NoopPlan``/``TrailingRepairPlan`` in the absorbed slot and the
    ``LeadingRepairPlan`` around the launch. None of the three declares
    ``replaces_sub_steps``, so read bare they would each fall back to "replaces only
    my own slot" and the composition would report the three in-seam passes as running on the
    array path when the kernel performed it. Every wrapper names the plan that did
    the work in ``absorbed_by``, so the declarations are read from THERE.
    """
    return getattr(plan, "absorbed_by", None) or plan


def replaced_sub_steps(plans: Dict[str, Any]) -> Tuple[str, ...]:
    """Every driver pass the installed plans perform, in ``STEP_ORDER``-free union.

    A UNION and not a concatenation: both slots of an installed pair report the same
    ``replaces_sub_steps``, so the pair's passes are counted once.
    """
    from ..triton_kernels.launch import STEP_ORDER  # noqa: PLC0415 - shared order

    found = []
    for slot in plans:
        declaring = declaring_plan(plans[slot])
        for name in getattr(declaring, "replaces_sub_steps", (slot,)):
            if name not in found:
                found.append(name)
    ordered = [name for name in STEP_ORDER if name in found]
    return tuple(ordered + [name for name in found if name not in ordered])


def _repair_paths_of(product: Dict[str, Any]) -> Tuple[str, ...]:
    """One product's ``REPAIR_PATHS``, imported at call time.

    THE IMPORT IS DEFERRED for the reason every other product import in this file is:
    the modules pull in kernel families that import CuPy, and ``install_fused_pairs``
    has to run on the merge-bar host. A module that cannot be imported at all is a
    fact the coverage predicate has already reported by the time this is reached --
    that verdict is taken before a plan is built -- so the fallback here is the
    DEFAULT declaration and not a way to paper over an unreadable module.
    """
    import importlib  # noqa: PLC0415

    try:
        module = importlib.import_module(f".{product['module']}", __package__)
    except Exception:  # pragma: no cover - the predicate refused a rung earlier
        return (_deposit_repair.SPLIT_FIELD_PATH,)
    return tuple(getattr(module, "REPAIR_PATHS",
                         (_deposit_repair.SPLIT_FIELD_PATH,)))


def _declared_uninstallable(family: str, product: Dict[str, Any]) -> Optional[str]:
    """``None`` when the product may be installed, else the reason it may not.

    ``INSTALLABLE`` IS A DECLARATION ABOUT THE PRODUCT, NOT ABOUT A RUN, which is why
    it is read here and reported on every configuration rather than folded into a
    coverage predicate. A predicate answers "can this weld serve this run" -- a
    question about arithmetic, and the one its gate measures -- and that answer must
    not change because of where the product sits in the driver's slot path. A module
    that sets ``INSTALLABLE = False`` is saying the opposite kind of thing: the
    arithmetic is certified and the COMPOSITION is refused, on every row, for a
    measured reason of its own.

    IT IS BELT AND BRACES, NOT THE RULE. :func:`_neighbouring_seam_claimant` is what
    a future product spanning more slots needs, and it is what decides the H->D
    seam's refusal from the run. This flag is what keeps a careless edit -- a new
    seam row, a reordered table, a widened arm declaration -- from installing a
    product whose own module says it must not be, and it is what makes the two new
    :data:`_SLOT_OPENS_A_LATER_SEAM` rows safe for the released pairs they neighbour.

    THE IMPORT IS DEFERRED AND GUARDED for the reason :func:`_repair_paths_of`'s is:
    these modules pull in kernel families that import CuPy and this loop has to run
    on the merge-bar host. A module that cannot be read is INSTALLABLE by default --
    fail-open here is correct because the coverage predicate has already refused an
    unreadable module a rung earlier, and fail-closed would turn an import problem
    into a silent, permanent refusal that reads like a decision.
    """
    import importlib  # noqa: PLC0415

    try:
        module = importlib.import_module(f".{product['module']}", __package__)
    except Exception:  # pragma: no cover - the predicate refused a rung earlier
        return None
    if getattr(module, "INSTALLABLE", True):
        return None
    declared = getattr(module, "INSTALLABLE_REASON", "")
    return (f"{family} declares INSTALLABLE = False: "
            f"{declared or 'its own module refuses the composition'}")


#: Which LATER seam each slot is the FIRST consult of. A product owning the slot on
#: the left closes a seam this table's right-hand seam would have opened, so the two
#: cannot both be installed on one run.
#:
#: THREE ROWS SINCE 2026-09-04, and the sentence this comment used to carry --
#: "``step_D``, ``step_B`` and ``update_H`` appear in one seam each" -- became FALSE
#: the day the H->D seam was priced onto the boards and given a row in
#: :data:`FUSED_PAIR_SEAMS`. It is corrected here rather than left standing, because
#: a table whose comment describes a different table is how an arbitration rule goes
#: quietly wrong.
#:
#: * ``update_E`` -- the electric constitutive is the SECOND slot of a D->E pair and
#:   the FIRST slot of an E->P pair. The original row, measured 2026-09-02.
#: * ``update_H`` -- the magnetic constitutive is the SECOND slot of a B->H pair and
#:   the FIRST slot of an H->D pair.
#: * ``step_D`` -- the electric curl is the SECOND slot of an H->D pair and the FIRST
#:   slot of a D->E pair.
#:
#: ``step_B`` is nobody's second slot and ``update_P`` is nobody's first, so those two
#: are the only slots on the four-slot path that stay out of this table.
#:
#: THE TWO NEW ROWS ARE WHY :func:`_declared_uninstallable` EXISTS. Read bare, the
#: ``update_H`` row makes :func:`_later_seam_claimant` withhold ``update_H`` from the
#: RELEASED B->H pair the moment an H->D product admits -- the exact displacement the
#: seam's key order is pinned to prevent. A product that declares itself
#: uninstallable claims nothing, so it is not a claimant, and the released pair keeps
#: its slot. SINCE 2026-09-05 THAT IS BELT AND BRACES RATHER THAN THE GUARD: the B->H
#: span is an END EDGE of :data:`FUSED_PAIR_SEAMS` by the table's own degrees, so
#: :func:`_neighbouring_seam_claimant` never asks the question of it at all, whatever
#: any H->D product declares (measured through the Metal composer on 2026-09-04: a
#: registered, admitting stand-in with no absorb declaration cost the released pair
#: its seam on every row it reached; the flag alone was what stood in the way).
_SLOT_OPENS_A_LATER_SEAM: Dict[str, str] = {
    "update_E": "update_E",
    "update_H": "update_H",
    "step_D": "step_D",
}


def _label_of(family: str, product: Dict[str, Any]) -> str:
    """The literal this product's plan builder passes as ``label=``.

    READ OFF THE TABLE ROW, not off a built plan: the offer is asked BEFORE the
    builder runs (that is the whole point of asking it before the predicate), so
    there is no plan to read it from yet. The row and the builder are pinned equal
    by ``test_cuda_certified_fused_products``, which walks the AST of this file, so
    the two spellings cannot drift.
    """
    return str(product.get("label") or family)


def _label_not_offered(label: str,
                       offered: Optional[FrozenSet[str]]) -> Optional[str]:
    """``None`` when this caller will run ``label``, else why the seam stays unfused.

    THE MIRROR OF ``triton_kernels.launch._label_not_offered``, down to the wording,
    and the wording is load-bearing: the driver-route gate's envelope leg greps for
    "offered to run" to tell a seam left unfused BY THE OFFER from a seam left
    unfused by a predicate, and a third spelling would make that classifier report a
    changed refusal rather than the narrowing.

    WHY A CALLER MAY OFFER A SUBSET AT ALL. ``fuse`` says WHETHER to attempt fusion;
    a caller that will only RUN some of the products this table can build has no
    other way to say so, and a fused product occupies EVERY slot of its seam -- so an
    un-offered product taking those slots forces the caller to refuse the whole step,
    taking the separate certified arms underneath it down as well.

    ``None`` MEANS NO RESTRICTION, which is the right default for every caller that
    composes in order to INSPECT -- the gates, the probes, the coverage census. They
    ask what the table can build, and an offer they did not make must not narrow it.
    """
    if offered is None or label in offered:
        return None
    return (f"{label} is not among the fused labels this caller offered to run "
            f"({', '.join(sorted(offered)) or 'none'}); the seam keeps its separate "
            f"sub-step plans")


def offered_labels(fuse_labels: Any) -> Optional[FrozenSet[str]]:
    """Normalise the caller's offer, refusing to read a non-collection as empty.

    A caller that passes something unreadable is treated as having made NO offer
    rather than an empty one: an empty offer silently disables every product, which
    is a behaviour change wearing the shape of a typo.
    """
    if fuse_labels is None:
        return None
    if isinstance(fuse_labels, str):
        return frozenset({fuse_labels})
    try:
        return frozenset(str(label) for label in fuse_labels)
    except TypeError:
        return None


def span_of(family: str, product: Dict[str, Any]) -> Tuple[str, ...]:
    """Every slot one product owns, in driver order.

    READ OFF THE TABLE, NEVER INFERRED FROM THE PRODUCT'S CLASS. A pair's span is its
    seam's two slots, which is what the ``slots`` key defaults to; a THREE-slot weld
    declares its own and continues past the seam into ``update_P``. The two
    arbitration rules below both ask this question, and a second way of answering it
    is a second thing to drift.
    """
    declared = product.get("slots")
    if declared:
        return tuple(str(name) for name in declared)
    curl_name = product["curl_slot"]
    return (curl_name, FUSED_PAIR_SEAMS[curl_name][0])


def _superseded_by_a_longer_span(
        candidates: Sequence[Tuple[str, Dict[str, Any], Tuple[str, ...]]]
) -> Dict[str, str]:
    """``{family: refusal}`` for every candidate a LONGER candidate strictly contains.

    WHY THIS EXISTS, 2026-09-02, WITH THE THREE-SLOT WELDS. Their predicates are
    CONJUNCTIONS of a D->E predicate and an E->P one, so each one's admission set is
    a SUBSET of the D->E product it supersedes -- and two admitters on one seam is
    what :func:`install_fused_pairs` refuses by leaving the seam UNFUSED naming both.
    Without this rule, landing a three-slot weld would COST the seven and three rows
    its D->E half already serves rather than adding the second seam to them.

    IT IS SLOT ARBITRATION AND NOT A VERDICT ABOUT ARITHMETIC, exactly as
    :func:`_later_seam_claimant` is: both predicates admit these runs and both gates
    certify them; what is decided here is who gets the slots.

    IT IS NON-LOSSY IN BOTH DIRECTIONS, and that is a property of the CONJUNCTION
    rather than a hope. Where the longer product admits, the shorter one does too, so
    refusing the shorter costs no row; where the longer refuses, it is not a candidate
    at all and the shorter installs exactly as it does today.

    STRICT CONTAINMENT, NOT LENGTH. Two products of different lengths whose spans
    merely overlap are still an AMBIGUITY -- the composer's own refusal, reached
    below -- because neither can serve what the other does.
    """
    spans = {family: span_of(family, product)
             for family, product, _arms in candidates}
    refusals: Dict[str, str] = {}
    for family, span in spans.items():
        for other, other_span in spans.items():
            if other == family or not set(span) < set(other_span):
                continue
            refusals[family] = (
                f"{other} admits this run and spans "
                f"{'/'.join(other_span)}, which strictly contains this product's "
                f"{'/'.join(span)}; the longer span serves every slot this one would "
                f"and one more, so the slots go to it. See "
                f"_superseded_by_a_longer_span -- this is slot arbitration, not a "
                f"verdict about {family}'s arithmetic, which its own gate certifies")
            break
    return refusals


def _spans_may_absorb(selected: Dict[str, str], span: Sequence[str],
                      arms: Sequence[str]) -> Optional[str]:
    """``None`` when a product may take EVERY slot of its span, else why it may not.

    ONE CALL PER SEAM THE SPAN CROSSES, through the shared
    ``triton_kernels.launch._pair_may_absorb`` -- never a second copy of its clause.
    A two-slot span crosses one seam and this is exactly that function; a THREE-slot
    span crosses two, so it is asked twice, on consecutive slot/arm pairs. The first
    refusal is returned, so the reason names the slot that actually went elsewhere.

    THE ARM TUPLE IS AS LONG AS THE SPAN and that is checked rather than assumed: a
    row of the wrong length would silently ask about a different seam than the one
    the product spans.
    """
    span = tuple(str(name) for name in span)
    arms = tuple(str(arm) for arm in arms)
    if len(arms) != len(span):
        return (f"the absorb declaration names {len(arms)} arms for a "
                f"{len(span)}-slot span {'/'.join(span)}; a fused product may not "
                f"claim a slot no declaration covers")
    for index in range(len(span) - 1):
        refusal = _pair_may_absorb(selected, span[index], span[index + 1],
                                   (arms[index], arms[index + 1]))
        if refusal is not None:
            return refusal
    return None


def _later_seam_claimant(plans: Dict[str, Any], reasons: Dict[str, Tuple[str, ...]],
                         selected: Dict[str, str], context: Any,
                         update_name: str,
                         span: Sequence[str] = (),
                         offered: Optional[FrozenSet[str]] = None) -> Optional[str]:
    """The E->P product that would lose ``update_name`` if a D->E pair took it.

    WHY THIS ARBITRATION EXISTS, MEASURED 2026-09-02. A D->E product owns ``step_D``
    AND ``update_E``; an E->P product owns ``update_E`` AND ``update_P``.
    :func:`install_fused_pairs` walks ``FUSED_PAIR_SEAMS`` in order and D_to_E comes
    first, so on any run both admit, the D->E product takes the slot and the E->P one
    is refused with an "arm table gave it elsewhere" reason -- silently trading one
    seam-instance for another.

    ON THIS CORPUS THAT TRADE IS EXACTLY NET ZERO ON BOTH CELLS THAT REACH IT:

    * ``cuda_dispersive_fused_electric_pair``'s seven rows (three
      ``stochastic_emitter*``, four ``TestLoadDump.*_2d``) are EXACTLY the seven
      ``cuda_fused_polarization_pair`` serves at E->P;
    * ``cuda_no_pml_dispersive_fused_electric_pair``'s three rows
      (``absorber-1d.py``, ``TestAbsorber.test_absorber``,
      ``material-dispersion.py``) are EXACTLY the three
      ``cuda_no_pml_fused_polarization_pair`` serves there.

    Read off the stamped census (each row's ``cuda_ade`` / ``cuda_no_pml_ade``
    covering ``update_P``) and off the pre-existing board's own served ledger
    (``results/fusion_matrix_cuda_2026-09-02_tierfix``: those two pairs serving 7 and
    3). So the D->E product gains what the E->P product loses, and pays for it by
    displacing a RELEASED, certified product.

    THE SEAM IS THEREFORE LEFT WITH THE PRODUCT THAT ALREADY HAD IT. That is a
    decision about SLOT ARBITRATION and it belongs here rather than in either
    product's predicate: a predicate answers "can this weld serve this run", which is
    a question about arithmetic and is what its gate measures, and the answer must not
    change because a different product exists. Both D->E predicates admit these runs
    and their gates certify them; what this function decides is who gets the slot.

    IT IS NOT A RANKING AND DOES NOT PRETEND TO BE. Where the trade is not net zero --
    a D->E cell with rows no E->P product serves -- this returns ``None`` and the D->E
    product installs as before, which is how
    ``cuda_conductive_fused_electric_pair``'s row (no susceptibility, so no E->P
    instance at all) is served. What would make the D->E weld strictly better is a
    THREE-SLOT product spanning ``step_D`` -> ``update_E`` -> ``update_P``, which
    serves BOTH seams instead of trading one for the other.

    THAT PRODUCT LANDED 2026-09-02, AND ``span`` IS WHAT THIS FUNCTION DOES ABOUT IT.
    The withholding above models a TRADE: one seam-instance gained, one lost. A
    product whose OWN span already reaches ``update_P`` makes no trade -- it serves
    the later seam itself -- so there is nothing to withhold and this returns
    ``None``. Asked here rather than at the call site because the trade is what this
    function is about, and a caller free to skip it could skip it for a product that
    really does trade.

    ``span`` DEFAULTS TO EMPTY, which reads as "a product that spans only this seam"
    and preserves every caller written before three slots existed.

    A PRODUCT THAT DECLARES ITSELF UNINSTALLABLE IS NOT A CLAIMANT, 2026-09-04, and
    that clause is load-bearing rather than tidy. :data:`_SLOT_OPENS_A_LATER_SEAM`
    gained an ``update_H`` row when the H->D seam was priced, which without this
    clause would make the RELEASED B->H pair lose ``update_H`` to a product that can
    never take it. Nothing on this table declares the flag today, so the answer this
    function gives on every corpus row is unchanged by the clause; what it changes is
    what the two new rows can do.
    """
    later = FUSED_PAIR_SEAMS.get(_SLOT_OPENS_A_LATER_SEAM.get(update_name, ""))
    if later is not None and later[0] in tuple(span):
        return None
    for family, product in FUSED_PRODUCTS.items():
        if product["curl_slot"] != update_name:
            continue
        if _declared_uninstallable(family, product) is not None:
            continue
        # AN UN-OFFERED PRODUCT IS NOT A CLAIMANT, and that is the same clause as
        # the uninstallable one above, applied to the same failure. A product this
        # caller will not run can never take the later seam, so withholding a slot
        # in its favour trades a seam-instance the caller HAS for one it cannot
        # have: net -1 rather than net zero, which is the one thing the trade model
        # this function implements rules out.
        if _label_not_offered(_label_of(family, product), offered) is not None:
            continue
        verdict = _guarded_verdict(
            lambda product=product: product["coverage"](context),
            f"the {family} pair predicate")
        if verdict.covered:
            return family
    _ = (plans, reasons, selected)
    return None


def _slot_degrees(seams: Dict[str, Tuple[str, Optional[str]]]) -> Dict[str, int]:
    """How many rows of ``seams`` name each slot -- its degree on the seam path.

    COMPUTED FROM THE TABLE, NEVER SPELLED. :func:`_neighbouring_seam_claimant` asks
    its question only of a span whose two END slots both have degree >= 2 -- an
    INTERIOR edge -- and the end/interior split has to follow the table so that a
    row added later moves it without anyone having to remember to.
    """
    degrees: Dict[str, int] = {}
    for curl_name, (update_name, _seam_name) in seams.items():
        for slot in (curl_name, update_name):
            degrees[slot] = degrees.get(slot, 0) + 1
    return degrees


def _is_end_edge_span(span: Sequence[str],
                      seams: Dict[str, Tuple[str, Optional[str]]]) -> bool:
    """True when either END slot of ``span`` sits in fewer than two rows of ``seams``."""
    span = tuple(span)
    degrees = _slot_degrees(seams)
    return min(degrees.get(span[0], 0), degrees.get(span[-1], 0)) < 2


def _neighbouring_seam_claimant(plans: Dict[str, Any],
                                reasons: Dict[str, Tuple[str, ...]],
                                selected: Dict[str, str], context: Any,
                                family: str, curl_name: str, update_name: str,
                                span: Sequence[str] = (),
                                offered: Optional[FrozenSet[str]] = None
                                ) -> Optional[Tuple[str, str]]:
    """``(claimant, refusal)`` for the product that would lose a slot to this one.

    THE RULE, SINCE 2026-09-05: the question is asked ONLY of a span that is an
    INTERIOR edge of the seam path -- both of its END slots sit in two or more rows
    of :data:`FUSED_PAIR_SEAMS` -- and such a product is refused where another
    product claims the seam its LAST slot opens. An end-edge span is never asked and
    never yields. The degrees are COMPUTED from the table (:func:`_slot_degrees`),
    never spelled, so a seam row added later moves the split without anyone
    remembering to. :func:`_later_seam_claimant` is the later-seam question, and it
    is still called here rather than copied, so its released wording and its trade
    model stay exactly what its own measurement licensed.

    WHY THAT IS THE RIGHT SHAPE IS A MATCHING ARGUMENT, not a corpus accident. Over a
    path of four slots the maximum matching has size 2 and it is always the two END
    edges; an interior edge belongs to no maximum matching unless both of its
    neighbours are absent. So an end-edge span must never yield, and an interior span
    must yield to a neighbour that serves. The other direction -- the seam a span's
    FIRST slot closes -- is answered exactly and cheaply by
    ``triton_kernels.launch._pair_may_absorb`` reading the live ``selected``: a fact
    about what installed rather than a prediction about a predicate, and a fact
    cannot annihilate with another fact. That is also what resolves the H->D TIE
    below in the released incumbent's favour -- the B->H pair installs first and
    holds ``update_H`` -- so there is no tie-break helper and must not be.

    Degrees on THIS table today (four rows): ``step_B`` 1, ``update_H`` 2,
    ``step_D`` 2, ``update_E`` 2, ``update_P`` 1. So ``B_to_H`` (1, 2) and
    ``E_to_P`` (2, 1) are end edges and never yield; ``H_to_D`` (2, 2) is interior
    and yields; and ``D_to_E`` (2, 2) is interior here -- unlike on the three-row
    Metal and Triton tables -- and yields to ``E_to_P``, which is EXACTLY the released
    2026-09-02 trade ruling, reproduced from the table's degrees rather than
    re-litigated.

    WHAT THE GUARD CLOSED was measured 2026-09-04 through the METAL composer: with
    the ``update_H`` row in the seam table, the later half ALONE refused the RELEASED
    B->H pair on every row an admitting H->D product reached -- even a stand-in with
    no absorb declaration, which can never install (3 launches / 1 seam against the
    shipped 2 / 2 on the 155-row control cell). That tree was safe only because its
    H->D product declares ``INSTALLABLE = False``; the flag protects one product,
    this guard protects the seam. No H->D product is in :data:`FUSED_PRODUCTS` yet,
    so the guard changes no plan this composer produces today -- and it cannot
    reproduce the 2026-09-04 annihilation, because ``E_to_P`` is an end edge by the
    table's own degrees and is never asked at all.

    An earlier-neighbour half was added and then removed on 2026-09-04; the block
    below records what it did and why measurement retired it.

    WHY THE SECOND DIRECTION HAD TO EXIST, AND IT IS ARITHMETIC RATHER THAN A CORPUS
    ACCIDENT. The driver's slot path here is ``step_B -- update_H -- step_D --
    update_E``: four slots, three seams, and an installed pair is an edge of a
    matching on that path. Launches over those four slots are therefore

        launches = 4 - (number of installed pairs)

    so a product is worth installing only when it RAISES the matching size. The H->D
    product takes one slot from each neighbour, which can never do that:

    * where both neighbours serve, installing it takes the matching from 2 to 1 --
      one MORE launch per step and one FEWER seam served;
    * where exactly one neighbour serves, it ties: 3 launches and one seam either way;
    * where neither serves it would be a gain, and there is no such row.

    GAIN, TIE OR LOSS FOR AN H->D PRODUCT, MEASURED 2026-09-04 at COMPOSER level
    (predicate AND absorb row) on Metal, over the 179 buildable rows of the 194-row
    ``h_to_d_seam_2026-09-04`` basis (this board's predicate-level join on the same
    basis: ``update_H`` held by a released B->H product on 179 of 179 rows):

    ====  =========================================================  ====================
    rows  installing an H->D product there is                         under this rule
    ====  =========================================================  ====================
     133  a LOSS: both neighbours install, 2 pairs -> 1               B->H installs first;
                                                                      H->D refused by name
                                                                      (``_pair_may_absorb``)
      22  a TIE: only B->H serves, 3 launches / 1 seam either way     the same; the released
                                                                      incumbent keeps the slot
      22  a LOSS: only D->E serves                                    the later half names
                                                                      the D->E claimant
       2  a GAIN: NEITHER serves (the nonlinear cell, 4 / 0 -> 3 / 1) it installs, where its
                                                                      arm reaches the cell
    ====  =========================================================  ====================

    0 rows on which displacing the released B->H pair is a gain. The predicate-level
    join that preceded it said "there is no row where neither neighbour serves"; that
    was true of the predicates and false of the composer, which additionally requires
    an absorb row, and the 2 "neither" rows are why this rule must not be written as
    a blanket veto. The refusal this function's caller writes is therefore a
    RUN-SPECIFIC measured reason in the existing "who gets the slot" wording, not a
    hard-coded veto: on a run where no neighbour's product admits, this returns
    ``None`` and the product installs.

    IT IS SLOT ARBITRATION AND NOT A VERDICT ABOUT ARITHMETIC, exactly as its two
    siblings are. Every predicate that reaches this point admitted and every gate
    that released one certified it; what is decided here is who gets the slots.

    THE ONLY SHAPE THAT ESCAPES IT is a span that reaches the neighbouring seam
    ITSELF rather than colliding with it -- a four-slot ``step_B -> update_H ->
    step_D -> update_E`` weld in ONE launch, which serves three seams in one launch
    where today's two pairs serve two in two. ``span`` is how that is expressed here,
    the same way :func:`_later_seam_claimant` takes it: a product whose own span
    already covers the neighbouring seam's slots makes no trade and is not withheld.

    THE TWO HALVES CARRY DIFFERENT REASONS AND MUST, because they report different
    measurements. The later half's is a TRADE, measured net zero, and its wording is
    the released 2026-09-02 one, unchanged down to the slot it names -- the only edit
    is that the later seam's second slot is now READ off :data:`FUSED_PAIR_SEAMS`
    instead of spelled ``update_P``, which is the same string on the ``update_E`` row
    it was written for and the true one on the two rows added since. The earlier
    half's is a DISPLACEMENT, measured net -1, and says so.
    """
    span = tuple(span) or (curl_name, update_name)
    # R1: AN END-EDGE SPAN IS NEVER ASKED. The matching argument is in the docstring;
    # the degrees are read off the table so the answer moves with it.
    if _is_end_edge_span(span, FUSED_PAIR_SEAMS):
        return None
    if update_name in _SLOT_OPENS_A_LATER_SEAM:
        claimant = _later_seam_claimant(plans, reasons, selected, context,
                                        update_name, span, offered)
        if claimant is not None and claimant != family:
            later = FUSED_PAIR_SEAMS[_SLOT_OPENS_A_LATER_SEAM[update_name]]
            return claimant, (
                f"{claimant} admits this run's {update_name}/{later[0]} seam and "
                f"would lose {update_name} to this pair; the two seams are traded "
                f"one for one on every corpus row that reaches this, so the slot "
                f"is left with the product that already serves it. See "
                f"_later_seam_claimant -- this is slot arbitration, not a verdict "
                f"about {family}'s arithmetic, which its own gate certifies")
    # THE EARLIER-NEIGHBOUR HALF WAS REMOVED 2026-09-04, MEASURED, NOT ARGUED. It
    # asked the same question in the other direction -- "does a product claim the seam
    # my FIRST slot closes?" -- and it did two things, both wrong:
    #
    # * it was REDUNDANT where it was meant to help. An H->D span's last slot is
    #   ``step_D``, which opens the D->E seam, so the later half above already names a
    #   D->E claimant and returns before the earlier half is reached.
    # * it REVERSED A RELEASED RULING where it did fire. The only product class whose
    #   last slot opens no seam is E->P (``update_P`` is terminal), so the earlier half
    #   fired there and there alone -- refusing the polarization pair because a D->E
    #   product admits, while the later half was already refusing that D->E product
    #   because the polarization pair claims ``update_E``. Both neighbours annihilated
    #   and the seam went unserved. Measured on a dispersive PML run (poles=2): the
    #   later half names ``cuda_fused_polarization_pair`` for the D->E product, and the
    #   earlier half named ``cuda_dispersive_fused_electric_pair`` for the polarization
    #   pair. The 2026-09-02 trade ruling gives ``update_E`` to the E->P incumbent, and
    #   this function may not re-litigate it from the other side.
    #
    # What the earlier half was FOR -- an H->D product installing on a row where only
    # the B->H neighbour serves, which the launch algebra makes a TIE rather than a
    # gain -- was settled on 2026-09-04/05 with the product in hand (on Metal): it is
    # a TIE, the released incumbent keeps the slot, and the mechanism is not a half of
    # this function at all but `_pair_may_absorb` reading the live `selected` after
    # the B->H pair installs first. What this function gained instead is the end-edge
    # guard above, which is what keeps the later half from refusing that incumbent.
    return None


def install_fused_pairs(plans: Dict[str, Any],
                        reasons: Dict[str, Tuple[str, ...]],
                        selected: Dict[str, str], context: Any,
                        offered: Optional[FrozenSet[str]] = None) -> None:
    """The opt-in fusion block: at most one product per seam, refusing on every doubt.

    Guarded exactly as an arm is. ``arms.plan_step`` never raises, so a product
    predicate that raises is a refusal and a builder that raises or returns ``None``
    leaves the seam unfused with a NAMED reason under ``fused_pair_<family>``.

    TWO ADMITTERS ON ONE SEAM IS AN AMBIGUITY, not a pick, and the wording
    deliberately avoids ``_select_slot``'s phrase "admit this configuration" so a
    probe scanning ``reasons`` for an ambiguous SLOT does not report a seam as one.

    ``offered`` IS APPLIED AT TWO POINTS AND THE SECOND IS NOT A DUPLICATE. The first
    is HERE, in the candidate loop, BEFORE the predicate -- which is a stronger
    placement than the Triton composer's because this composer has clauses that one
    does not. An un-offered product skipped before the predicate can neither
    SUPERSEDE a shorter offered span by strict containment nor make an offered
    product's seam ambiguous, both of which would leave the seam unfused for a
    product the caller cannot run anyway. The second is inside
    :func:`_later_seam_claimant`, where an un-offered product must not WITHHOLD a
    slot from an offered one: the withholding models a net-zero trade between two
    seams, and a trade with a product that cannot install is net -1.

    ``None`` MEANS NO RESTRICTION and composes exactly what this function composed
    before the offer existed, byte for byte -- which is what the coverage census
    asks for and what every gate and probe in the tree passes.
    """
    for curl_name, (update_name, pair_name) in FUSED_PAIR_SEAMS.items():
        candidates = []
        for family, product in FUSED_PRODUCTS.items():
            if product["curl_slot"] != curl_name:
                continue
            key = f"fused_pair_{family}"
            # THE OFFER IS ASKED FIRST, BEFORE EVERY OTHER CLAUSE, and the position
            # is the point -- see this function's docstring for what the two clauses
            # below would otherwise do with a product the caller cannot run.
            not_offered = _label_not_offered(_label_of(family, product), offered)
            if not_offered is not None:
                reasons[key] = (not_offered,)
                continue
            # THE DECLARATION IS ASKED FIRST, BEFORE THE PREDICATE, and the order
            # is the point: a missing absorb declaration is a fact about the TABLE,
            # true of every configuration, so it is reported on every configuration.
            # Asking the predicate first would make the reason appear and disappear
            # with the grid, which reads as a claim about the run.
            pair_arms = FUSED_PAIR_ARMS.get(family)
            if pair_arms is None:
                reasons[key] = (
                    f"{family} has no absorb declaration: which arm each of its two "
                    f"slots implements has not been established on this track, and "
                    f"a fused product may not substitute a numerical product no arm "
                    f"admitted",)
                continue
            # THE PRODUCT'S OWN INSTALLATION DECLARATION, ASKED BESIDE THE ABSORB
            # ONE AND FOR THE SAME REASON: it is a fact about the PRODUCT, true of
            # every configuration, so it is reported on every configuration. See
            # :func:`_declared_uninstallable`.
            uninstallable = _declared_uninstallable(family, product)
            if uninstallable is not None:
                reasons[key] = (uninstallable,)
                continue
            verdict = _guarded_verdict(
                lambda product=product: product["coverage"](context),
                f"the {family} pair predicate")
            if not verdict.covered:
                reasons[key] = verdict.reasons
                continue
            candidates.append((family, product, pair_arms))
        if not candidates:
            continue
        # A LONGER SPAN THAT STRICTLY CONTAINS ANOTHER'S IS NOT AN AMBIGUITY, and it
        # is settled BEFORE the ambiguity check for the reason that check exists: two
        # admitters would leave the seam UNFUSED naming both, which on these cells
        # would cost the ten rows the shorter products already serve. See
        # :func:`_superseded_by_a_longer_span` for why refusing the shorter one is
        # non-lossy in both directions.
        superseded = _superseded_by_a_longer_span(candidates)
        if superseded:
            for family, refusal in superseded.items():
                reasons[f"fused_pair_{family}"] = (refusal,)
            candidates = [row for row in candidates if row[0] not in superseded]
        if len(candidates) > 1:
            named = ", ".join(sorted(family for family, _p, _a in candidates))
            for family, _product, _arms in candidates:
                reasons[f"fused_pair_{family}"] = (
                    f"{named} all claim the {curl_name}/{update_name} seam; it is "
                    f"left unfused rather than assigned by table order",)
            continue
        family, product, pair_arms = candidates[0]
        key = f"fused_pair_{family}"
        span = span_of(family, product)
        # THE PRIMARY ROW FIRST, THEN EVERY DECLARED EXTRA, and the pair absorbs
        # on the FIRST arm pair the table already gave both slots to. At most one
        # can ever match -- the slots hold one selection each -- so the loop is a
        # lookup, not a preference order. A product with no matching declaration
        # is refused with the PRIMARY row's reason, which names the arm the slots
        # actually went to.
        #
        # A THREE-SLOT PRODUCT IS ASKED THE SAME QUESTION TWICE, once per seam it
        # spans, rather than through a second copy of the clause: a triple absorbs
        # ``step_D``/``update_E`` and ``update_E``/``update_P``, and it may absorb
        # each only where the arm table already gave both slots to the arms its
        # kernel implements. ``_pair_may_absorb`` is the shared helper both siblings
        # use, and consecutive pairs of the span/arm tuples ARE its two arguments.
        refusal = _spans_may_absorb(selected, span, pair_arms)
        if refusal is not None:
            for extra_arms in FUSED_PAIR_EXTRA_ARMS.get(family, ()):
                if _spans_may_absorb(selected, span, extra_arms) is None:
                    refusal = None
                    break
        if refusal is not None:
            reasons[key] = (refusal,)
            continue
        # A NEIGHBOURING SEAM'S CLAIM, ASKED BEFORE THIS PAIR IS BUILT. See
        # :func:`_neighbouring_seam_claimant`: a product owning a slot that another
        # seam's product also needs cannot install without displacing it, and over
        # the four-slot path ``step_B..update_E`` launches are ``4 - #pairs``, so
        # taking a slot from a neighbour can only tie or lose. The later-seam half
        # is the 2026-09-02 E->P trade, measured NET ZERO; the earlier-seam half is
        # the 2026-09-04 H->D collision, measured NET -1.
        neighbour = _neighbouring_seam_claimant(plans, reasons, selected, context,
                                                family, curl_name, update_name, span,
                                                offered)
        if neighbour is not None:
            reasons[key] = (neighbour[1],)
            continue

        pair, builder_refusal = _guarded_plan(
            lambda product=product: product["plan"](context),
            f"the {family} pair builder")
        if builder_refusal is not None:
            reasons[key] = (builder_refusal,)
        elif pair is not None:
            # THE MODULE'S OWN DECLARATION of which recurrence its repair inverts,
            # read HERE and handed to the installer rather than defaulted there --
            # exactly as ``triton_kernels.launch._certified_fused_product_entry``
            # reads it. A module written before ``deposit_repair.PLAIN_PATH`` existed
            # carries no such attribute and implicitly declares the split-field one,
            # which is what the fallback spells. THE FLAG
            # (``CARRIES_DEPOSIT_REPAIR``) IS NOT READ HERE, deliberately and as on
            # the other two composers: it is enforced inside the product's own
            # coverage predicate, so checking it twice would let the two answers
            # disagree.
            #
            # WHICH INSTALLER IS DECIDED BY THE SPAN, not by the plan's class: the
            # table is what declares how many slots a product owns, and reading the
            # object instead would let a builder that returned the wrong shape write
            # two slots for a three-slot product and leave ``update_P`` on the array
            # path while ``replaces`` said the kernel had advanced it.
            # THE FAMILY MODULE THE WARM PASS COMPILES THROUGH, stamped from the
            # TABLE ROW rather than asked of the builder: the row already declares
            # it, and a second declaration inside 38 builders would be 38 chances
            # for the two to disagree.
            try:
                pair.warm_module = product.get("module")
            except AttributeError:  # pragma: no cover - a builder returning a stub
                pass
            if len(span) == 3:
                _install_fused_triple(plans, selected, context.fields, context.pml,
                                      context.sources, pair_name, span,
                                      getattr(pair, "label", family), pair,
                                      _repair_paths_of(product))
            else:
                _install_fused_pair(plans, selected, context.fields, context.pml,
                                    context.sources, pair_name, curl_name,
                                    update_name, getattr(pair, "label", family),
                                    pair, _repair_paths_of(product))
        else:
            reasons[key] = ("the fused pair was refused",)
