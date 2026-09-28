"""The COMPLEX/Bloch H->D weld on this backend: complex ``update_H`` welded into
complex ``step_D``, ONE launch.

The complex-storage twin of :mod:`.fused_hd_pair`, on the second largest cell this
backend's fusion board leaves unbuilt on the H->D seam: **17 corpus instances** at
``(update_H complex -> step_D complex PML)``
(``parity/meep_gpu/results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``,
``aggregate.h_to_d_seam.instances`` filtered to that arm pair -- all seventeen
``buildable_not_built``, and NOT ONE of them carrying a standing in-seam electric
withdraw). The predicate below therefore reaches **17 of the cell's 17**, which is the
one thing that separates this cell from the real Cartesian one, where two rows are
refused by name for a live integrated electric withdraw.

Read :mod:`.fused_hd_pair` first. The seam, the hazard, the scratch-output shape, the
foreign-cell recompute, the withdraw clause and the arbitration are IDENTICAL and are
not re-argued here; what this file adds is the complex arithmetic and the two clauses
that arithmetic brings with it.

=============================================================================
THE SEAM, IN ONE PARAGRAPH
=============================================================================

The driver runs ``update_H`` (driver.py:3311) -> the electric integrated-source
withdraw (:3313-3314) -> ``step_D`` (:3315), and exactly one statement sits between
the two consults. Nothing is INJECTED here -- the electric injection is one seam later
and the magnetic one is one seam earlier -- so :mod:`..deposit_repair` has nothing to
say about this seam and :mod:`..withdraw_hoist` is the module that does.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE, ON WORD PAIRS
=============================================================================

Complex ``step_D``'s curl reads ``H`` at the program's own cell AND at its three
BACKWARD neighbours, and complex ``update_H`` writes ``H``. Welded in place a program
would read ``H`` at a neighbour at a moment decided by the block schedule -- the hazard
``results/triton_fused_offdiag_electric_2026-08-20`` measured on the mirror-image D->E
seam (42 of 60 subject cases divergent; the differing-word count moving from 4384 at
BLOCK=64 to 0 at BLOCK=1024) -- and it would read ``f_w_H`` the same way, which is easy
to miss: on this side the newly written ``f_w_H`` IS ``B`` exactly, so an in-place
write hands a racing neighbour ``B`` where it needs ``B_prev``.

Both premises are removed exactly as :mod:`.fused_hd_pair` removes them:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go to
  LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are pre-launch state for the whole
  dispatch and no program can observe another program's store;
* **the foreign read is not a read of another program's output.** It is a RECOMPUTE
  from that same unwritten state, through :func:`_h_cell_complex` -- the certified
  :func:`.complex_fields.bloch_constitutive_step` body for side ``H``, whole, evaluated
  at an arbitrary cell. A pure function of unwritten memory has no schedule to depend
  on.

:class:`ComplexFusedHdPairPlan` then ROTATES the ``H``/``f_w_H`` references, which is
:class:`.offdiag_scratch_weld.ScratchWeldPairPlan`'s certified choreography.

WHY THE RECOMPUTE IS EXACT RATHER THAN CLOSE. Complex ``update_H`` is POINTWISE: per
component it reads ``H``, ``f_w_H`` and ``B`` at ONE cell, on both float32 word planes,
and writes the same cell -- no neighbour read, no ghost rule, no ownership mask and NO
PHASE (``bloch_constitutive_step``'s own docstring: the constitutive sub-step never
sees the wrap, S:907-993). The stepped value at any cell is therefore a function of
state this launch does not write, and evaluating it twice gives the same bits by
construction rather than by a tolerance.

THE BLOCH PHASE IS NOT IN THE RECOMPUTE AND MUST NOT BE. The certified complex curl
applies :func:`.complex_fields._rotate_field_left` to the LOADED REGISTER after the
gather, on the wrapped lane only, through ``tl.where`` (a bitwise select). This weld
redirects the LOAD and leaves that phase block standing character for character, so a
wrapped neighbour is still ``<the neighbour's stepped H> * p_axis`` in that order --
the field-left orientation the array path uses (S:1862) and the one the expansion arm
was probed for. A weld that rotated inside the recompute would apply the phase to the
own cell too and would be wrong on every unwrapped lane.

=============================================================================
THE ARITHMETIC: FOUR MEASURED SPELLINGS, ARMED RATHER THAN REDISCOVERED
=============================================================================

Every real-coefficient multiply on the complex path is a FULL complex product with a
zero-imaginary operand. This module does not re-spell one: it imports
:func:`.complex_fields._mul_coefficient_left` as a ``triton.jit`` device function and
the curl half's own multiplies ride in with the lifted text. Four facts are load-bearing
and each was MEASURED rather than reasoned; each is a mutation on THIS kernel in the
gate rather than a verdict inherited from another family:

1. **the zero cross terms are spelled literally.** ``z_im * 0.0`` / ``0.0 * z_im``
   carry the sign of the field's words into the result exactly as the full complex
   multiply does; folding them to a literal ``0.0`` is byte-wrong on signed zeros
   (:func:`.complex_fields._mul_field_left`'s own measurement).
2. **negation is ``* -1.0``, never unary ``-``.** Triton lowers ``-x`` as ``0.0 - x``
   (3.1.0, ``language/semantic.py:386-391``) and under round-to-nearest
   ``0.0 - (+0.0)`` is ``+0.0``, so the addend's zero SIGN is lost.
3. **which factor the FMA fuses is per orientation.** ``_mul_field_left`` fuses the
   field's product and ``_mul_coefficient_left`` the coefficient's; the call-site
   orientation table is normative and both are probed.
4. **the arm is a MEASURED platform fact, never a default.** ``EXPANSION`` is bound
   from the complex expansion probe and an absent or non-discriminating artifact is a
   REFUSAL, inherited from both halves' own predicates.

**A COMPLEX-BY-REAL DIVIDE DOES NOT OCCUR IN THIS KERNEL, AND THAT IS A MEASUREMENT
RATHER THAN AN OMISSION.** The neighbouring Dcyl product had to settle the spelling of
one, and its verdict does not travel: CuPy's ``complex64 / float32`` is the SCALED
complex-by-complex algorithm (``cupy/_core/include/cupy/complex/arithmetic.h:96-110``,
every zero-valued term kept), not numpy's reciprocal multiply, which is wrong there on
5 to 25 per cent of words; and Triton's ``/`` lowers to ``div.full.f32`` (~2 ulp) so a
divide would have to be :func:`tl.math.div_rn`. **This seam divides nowhere.** Every
PML reciprocal is precomputed on the HOST into ``sinv_*`` (``pml.py``) and both halves
MULTIPLY by it. The fact is carried three ways rather than cited: a static count over
the emitted text (:func:`divide_free_evidence`), a laptop test, and an armed gate
mutation that respells one ``_mul_field_left(z, si)`` as a componentwise divide by
``1 / si`` and requires the divergence. The divide spellings the Dcyl round measured
are therefore RECORDED here as inapplicable-and-armed, not repeated.

**THE PLANTED ROW-ZERO TINY-NORMAL CASE IS PART OF THE GATE, NOT AN EXTRA.** The Dcyl
round measured that the wrong multiply arrangement is byte-visible only where a product
underflows or is a signed zero, and that a random battery did NOT reach that class
(0 of 14,387,104 words with the naive spelling). This gate therefore PLANTS a fixture
whose row 0 holds tiny normals of both signs beside exact zeros, and scores the
multiply mutations there as well as on the random battery.

=============================================================================
ONE SUB-LATTICE, ONE COEFFICIENT GROUP
=============================================================================

Both halves sit on the same Yee sub-lattice -- ``launch.SUB_STEPS['step_D']['suffix']``
is ``''`` and ``coverage.CONSTITUTIVE_SIDES['H']['half_integer']`` is False -- so the
curl's ``kms_x/_y/_z`` and the constitutive's ``km0/km1/km2`` ARE THE SAME THREE
VOLUMES and the signature binds them once. Both suffixes are READ from the shipped
tables and their equality is ASSERTED in :func:`plan_complex_fused_hd_pair`: binding
the half-integer set instead compiles, launches, converges, and is a half-cell error in
the absorber profile -- smooth, plausible and entirely wrong, which is why the gate
arms it as a host mutation rather than leaving it a comment.

**THE COEFFICIENTS STAY float32 UNDER COMPLEX STORAGE** (stepping.py:41-50), so the
nine coefficient vectors here are the same nine the real twin binds; only the twenty-one
FIELD volumes are word pairs, and they are bound as float32 WORD VIEWS
(:func:`.complex_fields._word_view`) with ``n_elem`` staying the COMPLEX cell count.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS CELL LOSES NO ROW TO IT
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the two consults. The pass that IS there is the electric
withdraw, and :mod:`..withdraw_hoist` is its contract -- one slot, not two; a hoist,
not a bracket.

:data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND, and the flag and the wiring move
together exactly as ``CARRIES_DEPOSIT_REPAIR``'s do: the only wiring that performs the
hoist is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, and this
product declares :data:`INSTALLABLE` False, so that branch is unreachable for it. The
predicate therefore refuses BY NAME every row whose electric withdraw does work.

WHAT THAT COSTS ON THIS CELL, MEASURED: **nothing**. All 17 instances carry
``integrated_electric_sources: 0`` and ``withdraw_in_seam: false`` in
``results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, and the board files none of them
under ``withdraw_seam``. The clause is still evaluated on every row rather than assumed
away -- ignorance is never an empty set, so an undeclared source list is a refusal.

=============================================================================
THE FOLDED COMPLEX NEIGHBOUR IS NOT A VARIANT OF THIS WELD ON THIS BACKEND
=============================================================================

``cuda_kernels/complex_fused_hd_pair`` covers its folded neighbour as a SECOND VARIANT
of one weld -- 17 + 5 instances, one transform, two emitted kernels -- because on that
backend ``complex_folded_kernels`` ships NO constitutive kernel and the census records
the PLAIN complex arm at ``update_H`` on all five folded rows, so the two cells differ
in the curl half alone and that half is the plain curl put through three anchored text
deltas.

**NEITHER HALF OF THAT HOLDS HERE, and the census says so.** On this backend the folded
complex rows select the arm ``folded complex`` at ``update_H`` (a distinct predicate,
:func:`.folded_complex.folded_complex_constitutive_coverage`, even though it launches
the same ``bloch_constitutive_step``), so the join that gives CUDA one shared H half
gives this backend two different cells; and the folded curl is
:func:`.folded_complex.folded_bloch_pml_curl_step`, a SEPARATE kernel with its own
mirror ghost map and its own boundary codes, not a text delta over
``bloch_pml_curl_step``. The five folded-complex instances also do not sit in one cell
here: they are ``(folded complex -> folded complex PML)`` = **2** rows
(``tests:TestEigCoeffs.test_binary_grating_special_kz_2_21_2``,
``tests:TestModeDecomposition.test_triangular_lattice_oblique``) and
``(folded complex -> folded complex beta PML)`` = **3** rows. So this module ships ONE
arm pair, ``(complex, complex PML)``, and whether a folded H->D halo needs cells beyond
the three backward neighbours stays what :mod:`.fused_hd_pair` calls it: a probe, not a
guess.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON HAS TWO HALVES
=============================================================================

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries both halves --
the label the composer would need and the ARBITRATION, which is MEASURED PER ROW rather
than asserted. Over the driver's ``step_B - update_H - step_D - update_E`` slot path
launches are ``4 - (installed pairs)`` and a product spanning ``update_H``/``step_D``
takes one slot from EACH neighbour, so installing it is a LOSS where both neighbouring
pairs install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of
the three each row is comes from that row's OWN composer slot table, is not uniform
over this cell, and is recorded row by row in the gate's lift leg under
``arbitration_over_the_driven_rows``. The counts live in the artifact and are not
copied here.

**NO TIMING EXISTS FOR THIS SHAPE AND NONE IS LICENSED BY ANYTHING IN THIS MODULE.**
The fused route does MORE memory traffic than the two singles for the same step-level
launch count; whether the saved launch and the saved ``H`` write-then-read round trip
pay for three extra pointwise complex constitutive evaluations per component per program
is a hypothesis.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. A
released product credits its admitted seam-instances while executing NOWHERE, and the
record and the board prose say exactly that: this product is in no
``fastpath.RELEASED_FUSED_ARMS`` envelope and ``fastpath`` never plans it. **The
composition-installed reference awaits wiring**: while this module is unregistered the
gate builds the product itself and drives the array path as the oracle.

THE EXPANSION LICENCE IS POLICY-CONDITIONAL and it is the certified family's, not this
product's: every complex arm in this package was certified under ``keep``
(:data:`.complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`) and the predicates this
module conjoins refuse under any other policy in force. Under ``flush`` the composer
installs nothing complex on these rows and this product's predicate refuses by that
name; what a gate can still measure there is the arithmetic from arrays with the arm
forced, and it must say so in the record.

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builder (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_fields as _cf
from . import coverage as _coverage
from . import fused_hd_pair as _real
from .. import withdraw_hoist as _withdraw_hoist
from .complex_fields import _resolve_expansion, _word_view
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "complex_fused_hd_pair"

#: The sub-step slot this product STARTS at -- the first half of the seam, in the
#: driver's own order, so a refusal is named on the slot the driver reaches first.
SLOT = "update_H"

#: The driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3311, :3315). NOTHING BETWEEN THEM IS CARRIED, and there is only one
#: thing there to carry: the electric withdraw (:3313-3314), which
#: :data:`HOISTS_THE_WITHDRAW` declines in this round.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under, and the
#: module that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather
#: than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements, on ``(update_H, step_D)``, spelled as the
#: census and the board's ``h_to_d_seam`` join spell them.
ARMS: Tuple[str, str] = ("complex", "complex PML")

#: The six volumes one launch ROTATES, the three it steps IN PLACE with their
#: auxiliaries, and the constitutive half's sources. READ from the real twin rather
#: than re-spelled: the volume names do not change with the storage dtype, and two
#: spellings of one list is one too many.
ROTATED: Tuple[str, ...] = _real.ROTATED
IN_PLACE: Tuple[str, ...] = _real.IN_PLACE
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _real.CONSTITUTIVE_SOURCES

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins against
#: ``launch.SUB_STEPS`` rather than a literal buried in the builder.
BACKWARD = 1

#: Elements per program. Restated from :data:`.complex_fields.DEFAULT_BLOCK` (which
#: needs Triton to import, and the predicate and the builders must answer on the
#: laptop that is this package's merge bar) and PINNED against it by a laptop test.
DEFAULT_BLOCK = 256

#: One launch per run, and the plan counts its own. A second launch appearing here
#: without a gate is the arbitration argument silently changing.
LAUNCHES_PER_RUN = 1

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult (driver.py:3311) and the ``step_D`` consult (:3315). This predicate
#: therefore does NOT consult ``deposit_repair.seam_source_reasons`` at all.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round. The predicate consequently refuses every row with a standing in-seam
#: withdraw BY NAME -- which on THIS cell is 0 of 17.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration.
#: ``launch._declared_uninstallable`` reads this and reports the reason on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "THE COMPOSER IS NOT OFFERED THIS PRODUCT AT ALL, and that is the first half of "
    "the reason: routing it through launch._install_certified_fused_products costs a "
    "label, and on this backend a label plan_step can write must also be declared in "
    "the DISPATCHER'S own tables -- outside this package by the one-way rule it keeps, "
    "and outside the round that built this product. The flag is what keeps a later "
    "edit that lands those lines from installing the product before a composition "
    "gate has driven it end to end; while it stands, the gate's reference is the "
    "ARRAY PATH and the composition-installed reference AWAITS WIRING. "
    "THE SECOND HALF IS THE ARBITRATION, and it is MEASURED PER ROW rather than "
    "asserted: over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs) and a TWO-SLOT H->D product takes one slot "
    "from EACH neighbour, so installing it is a LOSS where both neighbouring pairs "
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of "
    "the three each corpus row is comes from that row's OWN composer slot table and "
    "is recorded, row by row, in the lift leg of "
    "parity/meep_gpu/gate_triton_complex_fused_hd_pair.py under "
    "`arbitration_over_the_driven_rows`. WHAT IT MEASURED, stated here rather than "
    "left to be discovered by opening an artifact: ALL TWELVE driven corpus rows are "
    "a LOSS -- 12 of 12, no TIE, no GAIN -- in the arbitration block of "
    "results/triton_complex_fused_hd_pair_2026-09-07/keep/gate.json, which counts "
    "PAIRS (one label held by both slots of a seam) rather than slots. Installing "
    "this product would RAISE the per-step launch count on every row it claims, "
    "which is why INSTALLABLE is False and would remain False even with the label "
    "wall down. A THIRD FACT IS SPECIFIC TO THIS ARM and is measured with the "
    "others rather than assumed: the complex arms are certified under the 'keep' "
    "float32 subnormal policy only, so under 'flush' the composer selects nothing "
    "complex on these rows at all and the arbitration question does not arise. The "
    "only span that is strictly additive everywhere is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

#: The operand orientations this kernel launches, as a claim about its text rather
#: than a default. The curl half rotates a field by the phase (``c8_mul_c8``), scales
#: by ``dtdx`` (``python_float_left``) and applies the split-field coefficients
#: field-left; the H-side constitutive applies ``kps``/``kms`` coefficient-left; the
#: ``SCALE`` arm that would bring in ``inv_eps`` belongs to the E side and is not
#: carried; and the seam adds ``tl.where`` selects, which are not multiplies. So the
#: BASE FOUR is exactly the set, and a laptop test pins it by scanning the shipped
#: body for any multiply helper outside the three this module can reach.
PROBE_PATTERNS: Tuple[str, ...] = _cf.PROBE_PATTERNS

#: The policy the complex tranche's expansion licence was cut under. Restated from
#: the certified module so the two cannot drift.
CERTIFIED_UNDER_SUBNORMAL_POLICY: str = _cf.CERTIFIED_UNDER_SUBNORMAL_POLICY

#: The three multiply helpers this kernel may reach, and NO OTHER. A laptop test
#: scans the shipped kernel text for ``_mul_``/``_rotate_``/``_div_`` call sites and
#: fails on anything outside this tuple -- which is how "this product launches no
#: orientation its two halves do not already launch" stays a measurement.
MULTIPLY_HELPERS: Tuple[str, ...] = (
    "_rotate_field_left", "_mul_field_left", "_mul_coefficient_left")

#: THE DIVIDE FACTS THIS SEAM DOES NOT USE, recorded with their measured provenance so
#: a later reader reaches for the armed mutation instead of rediscovering them. Each
#: entry is a spelling that is CORRECT SOMEWHERE ELSE in this package and would be a
#: silent defect here; the last one is the fact that makes them moot on this seam.
DIVIDE_FACTS: Tuple[Dict[str, str], ...] = (
    {"claim": "CuPy's complex64 / float32 is the SCALED complex-by-complex algorithm "
              "(cupy/_core/include/cupy/complex/arithmetic.h:96-110) with every "
              "zero-valued term kept, NOT numpy's reciprocal multiply",
     "where_it_is_true": "cylindrical_fused_hd_pair's radial prefix divide",
     "why_it_is_moot_here": "this seam performs no division at all"},
    {"claim": "numpy's a * (float32(1)/d) -- the spelling the Metal complex verdict "
              "prescribes, and correct there because that backend's engine is NumPy "
              "-- is wrong on CuPy (793,105 / 820,455 of 14,387,104 words, keep / "
              "flush, NVRTC record)",
     "where_it_is_true": "metal_kernels' complex families",
     "why_it_is_moot_here": "this seam performs no division at all"},
    {"claim": "Triton's f32 `/` lowers to div.full.f32 (~2 ulp), so a correctly "
              "rounded quotient needs tl.math.div_rn",
     "where_it_is_true": "every Triton kernel that divides",
     "why_it_is_moot_here": "this seam performs no division at all"},
    {"claim": "every PML reciprocal is precomputed on the HOST into sinv_* (pml.py) "
              "and both halves MULTIPLY by it through the zero-imaginary complex "
              "product; the emitted text contains no `/` and no div_rn",
     "where_it_is_true": "here",
     "why_it_is_moot_here": "it IS the reason -- measured by divide_free_evidence() "
                            "and armed as a gate mutation rather than cited"},
)

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_UNDER_SUBNORMAL_POLICY",
    "CONSTITUTIVE_LIFT_EDITS", "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES",
    "CURL_SUB_STEP", "DEFAULT_BLOCK", "DIVIDE_FACTS", "FAMILY", "HALO_TAPS",
    "HOISTS_THE_WITHDRAW", "H_CELL_TAIL_ARGS", "INSTALLABLE", "INSTALLABLE_REASON",
    "IN_PLACE", "LAUNCHES_PER_RUN", "MULTIPLY_HELPERS", "OWN_LOAD_EDITS",
    "PROBE_PATTERNS", "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT",
    "ComplexFusedHdPairPlan",
    "certified_constitutive_tail", "certified_curl_tail", "complex_fused_hd_pair_coverage",
    "curl_lift_edits", "divide_free_evidence", "explain_complex_fused_hd_pair",
    "function_source", "fused_complex_constitutive_curl_H_to_D_kernel", "halo_taps",
    "lifted_constitutive_tail", "lifted_curl_tail", "offset_coordinates",
    "plan_complex_fused_hd_pair", "plan_complex_fused_hd_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Where BOTH certified bodies end their index decode. One anchor serves both, which
#: is what lets the fused kernel carry one decode prologue.
DECODE_END = "    i = plane // ny\n"

#: Where :func:`_h_cell_complex` re-composes the flat index from its own coordinates.
H_CELL_DECODE_END = "        idx = i * nyz + j * nz + k\n"

#: Where :func:`_h_cell_complex` stops being lifted text and returns its registers.
H_CELL_RETURN_START = ("        return (out0_re, out0_im, out1_re, out1_im, "
                       "out2_re, out2_im,\n")

#: Where the fused kernel's lifted CURL body begins. A marker rather than the shared
#: decode anchor, because the weld block sits between the decode and the curl.
CURL_BODY_ANCHOR = "        # === the certified complex step_D curl body begins here ===\n"

#: What one :func:`_h_tap_complex` call site forwards after the five leading
#: arguments. Built once so the declaration, the generated lift edits and every call
#: cannot drift apart.
H_CELL_TAIL_ARGS = ("hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,\n"
                    "                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, "
                    "EXPANSION")


def function_source(function_name: str, path: Optional[Path] = None) -> str:
    """One function's source TEXT, by name, off a file rather than by import.

    TEXT AND NOT ``inspect``: :mod:`.complex_fields` imports Triton at module scope,
    so on the laptop that is this package's merge bar it cannot be imported at all --
    and the lift checks are exactly the checks that must run there.
    """
    return _real._source_of(  # noqa: SLF001 - the shared lift helper
        function_name,
        Path(__file__).with_name("complex_fields.py") if path is None else path)


#: Every line of certified CONSTITUTIVE text this weld does not lift verbatim, with
#: the reason. DATA, not prose, so the host suite and the gate can assert the list and
#: a line that stopped matching RAISES instead of silently leaving a stale read.
#:
#: THERE IS NO ARITHMETIC IN THIS TABLE. Every entry is a pointer spelling, a
#: constexpr branch resolved to the side this product serves, or a store that moves to
#: the caller; not one right-hand side, operand order or paren is touched, and every
#: complex product still goes through the certified helper under the run's own
#: ``EXPANSION``.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "km_N = tl.load(kmN + <axis>, ...)",
     "became": "km_N = tl.load(kmx|kmy|kmz + <axis>, ...)",
     "why": "ONE POINTER RENAME PER AXIS, NO ARITHMETIC. Both halves sit on the same "
            "Yee sub-lattice (SUB_STEPS['step_D']['suffix'] == '' and "
            "CONSTITUTIVE_SIDES['H']['half_integer'] is False), so the curl's kms_x "
            "and the constitutive's km0 ARE THE SAME VOLUME and the fused signature "
            "binds it once. Binding the half-integer set instead compiles and is a "
            "half-cell error in the absorber profile, converged and smooth and "
            "wrong; plan_complex_fused_hd_pair ASSERTS the two suffixes agree."},
    {"line": "if SCALE: ie = tl.load(eN + idx, ...); "
             "src = _mul_field_left(src, ie, EXPANSION)",
     "became": "(removed -- the constexpr branch is RESOLVED to SCALE = 0)",
     "why": "the H side reads B directly (mu = 1 baked in exactly as the array path "
            "bakes it at stepping.py:907-923), so the SCALE = 1 arm is text no launch "
            "of this product could reach and the three inverse-epsilon pointers leave "
            "with it. It also removes the ONE `_mul_field_left` the constitutive half "
            "would have launched, which is why the H side's orientation set is the "
            "coefficient-left product alone."},
    {"line": "prev_re/prev_im = tl.load(wN + 2 * idx ...)",
     "became": "prev_re/prev_im = tl.load(wiN + 2 * idx ...)",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which "
            "nothing in this launch writes. On this side the newly written f_w_H IS B "
            "exactly, so an in-place write would hand a racing neighbour B where it "
            "needs B_prev -- the second, easily missed half of the hazard."},
    {"line": "src_re/src_im = tl.load(gN + 2 * idx ...)",
     "became": "src_re/src_im = tl.load(bN + 2 * idx ...)",
     "why": "one pointer rename: the magnetic flux density under its own name. In the "
            "fused signature `gN` is the curl half's magnetic FIELD, so the "
            "constitutive source cannot keep that name."},
    {"line": "tl.store(wN + 2 * idx, src_re, ...); "
             "tl.store(wN + 2 * idx + 1, src_im, ...)",
     "became": "sN_re = src_re; sN_im = src_im",
     "why": "the weld writes a scratch volume, so the helper must not store. The "
            "value and the expression that produced it are untouched; only the "
            "destination moves to the caller, which addresses the scratch for its OWN "
            "cell and stores nothing at all for a foreign recompute. The capture also "
            "gives each component its own register, which the certified body did not "
            "need because it stored immediately."},
    {"line": "a_re/a_im = tl.load(fN + 2 * idx ...)",
     "became": "a_re/a_im = tl.load(hiN + 2 * idx ...)",
     "why": "the accumulator is read from the PRE-LAUNCH H, const for the whole "
            "dispatch. This is what makes the foreign recompute a pure function of "
            "unwritten memory. In the fused signature `fN` is the DISPLACEMENT."},
    {"line": "tl.store(fN + 2 * idx, a_re, ...); tl.store(fN + 2 * idx + 1, a_im, ...)",
     "became": "outN_re = a_re; outN_im = a_im",
     "why": "same as the split-field store: the destination moves to the caller. The "
            "two accumulations, their order, their operands and their coefficient-left "
            "complex products are untouched."},
    {"line": "idx = tl.program_id(0) * BLOCK + ... ; i = plane // ny",
     "became": "(removed -- i, j and k are parameters; idx is composed from them)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a function "
            "evaluated at an ARBITRARY cell rather than at the program's own, so the "
            "cell arrives as three coordinates and the flat index is COMPOSED by the "
            "same layout the decode inverts. `live` arrives as the caller's per-lane "
            "validity: a tap the certified ghost rule masked off passes False, every "
            "load inside is masked, and no out-of-range coordinate is dereferenced."},
)


def certified_constitutive_tail() -> str:
    """``bloch_constitutive_step``'s body BELOW the decode, stores captured, dedented.

    Not a transcription: this is the certified body's own text with exactly the
    substitutions :data:`CONSTITUTIVE_LIFT_EDITS` names, each applied through
    :func:`.fused_hd_pair.needle`, so a spelling that drifted raises here rather than
    leaving the fused kernel reading the wrong volume.
    """
    needle = _real.needle
    tail = _real._cut(function_source("bloch_constitutive_step"), DECODE_END)  # noqa: SLF001
    axes = ("i", "j", "k")
    pointers = ("kmx", "kmy", "kmz")
    for target in range(3):
        tail = needle(
            tail,
            f"km_{target} = tl.load(km{target} + {axes[target]}, "
            f"mask=live, other=0.0)",
            f"km_{target} = tl.load({pointers[target]} + {axes[target]}, "
            f"mask=live, other=0.0)")
    for target in range(3):
        trailer = "  # D LEFT (S:982-984)\n" if target == 0 else "\n"
        tail = needle(
            tail,
            f"if SCALE:\n"
            f"    ie = tl.load(e{target} + idx, mask=live, other=0.0)\n"
            f"    src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)"
            + trailer,
            "")
        tail = needle(tail, f"prev_re = tl.load(w{target} + 2 * idx",
                      f"prev_re = tl.load(wi{target} + 2 * idx")
        tail = needle(tail, f"prev_im = tl.load(w{target} + 2 * idx",
                      f"prev_im = tl.load(wi{target} + 2 * idx")
        tail = needle(tail, f"src_re = tl.load(g{target} + 2 * idx",
                      f"src_re = tl.load(b{target} + 2 * idx")
        tail = needle(tail, f"src_im = tl.load(g{target} + 2 * idx",
                      f"src_im = tl.load(b{target} + 2 * idx")
        tail = needle(
            tail,
            f"tl.store(w{target} + 2 * idx, src_re, mask=live)\n"
            f"tl.store(w{target} + 2 * idx + 1, src_im, mask=live)\n",
            f"s{target}_re = src_re\ns{target}_im = src_im\n")
        tail = needle(tail, f"a_re = tl.load(f{target} + 2 * idx",
                      f"a_re = tl.load(hi{target} + 2 * idx")
        tail = needle(tail, f"a_im = tl.load(f{target} + 2 * idx",
                      f"a_im = tl.load(hi{target} + 2 * idx")
        tail = needle(
            tail,
            f"tl.store(f{target} + 2 * idx, a_re, mask=live)\n"
            f"tl.store(f{target} + 2 * idx + 1, a_im, mask=live)\n",
            f"out{target}_re = a_re\nout{target}_im = a_im\n")
    # THE CHECK THAT MATTERS. In the fused signature `f*`, `w*` and `g*` are the
    # DISPLACEMENT, its split-field auxiliary and the magnetic field; one missed
    # rename is a smooth, plausible, entirely wrong answer rather than a compile
    # error, because those names all exist in the enclosing kernel.
    for stem in ("f", "w", "g", "e", "km"):
        for target in range(3):
            if f"{stem}{target} +" in tail:
                raise AssertionError(
                    f"the lifted complex update_H body still indexes {stem}{target}, "
                    f"which in the fused kernel is not the volume the certified body "
                    f"meant")
    if "SCALE" in tail or "tl.store(" in tail:
        raise AssertionError(
            "the lifted complex update_H body still stores or branches on SCALE; the "
            "weld's helper must return registers and carry no constexpr arm")
    return tail.rstrip("\n") + "\n"


def lifted_constitutive_tail() -> str:
    """:func:`_h_cell_complex`'s own body between its two anchors, dedented."""
    source = function_source("_h_cell_complex", Path(__file__))
    return _real._cut(source, H_CELL_DECODE_END,  # noqa: SLF001
                      stop=H_CELL_RETURN_START).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# The curl lift: complex step_D with every magnetic read redirected
# ---------------------------------------------------------------------------

#: The curl half's six OWN-cell magnetic loads and the register each becomes -- two
#: word planes per component. The program has already computed its own ``H`` into
#: ``own*_re``/``own*_im``, and those registers are the post-``update_H`` values the
#: array path would have loaded.
OWN_LOAD_EDITS: Tuple[Tuple[str, str], ...] = tuple(
    (f"{var}_{plane} = tl.load(g{target} + 2 * idx{'' if plane == 're' else ' + 1'}, "
     f"mask=live, other=0.0)\n",
     f"{var}_{plane} = own{target}_{plane}\n")
    for target, var in enumerate(("a", "b", "c"))
    for plane in ("re", "im"))

#: The curl half's six SHIFTED magnetic operands, as ``(register stem, component)``.
#: WHICH CELL each one reads is NOT written here: it is parsed out of the certified
#: body's own ``ox = si * nyz + j * nz + k`` lines by :func:`offset_coordinates`, and
#: which MASK guards it is parsed out of the load line itself by :func:`halo_taps`. So
#: the recompute lands on exactly the cell the emitter's index composed, and a change
#: to how that index or that guard is spelled RAISES instead of quietly redirecting a
#: tap.
#:
#: THE BRANCH, THE SHIFT ARITHMETIC AND THE BLOCH ROTATION ARE NOT TOUCHED. Only the
#: two leaf loads move, and they move TOGETHER into one call that returns both planes
#: -- so a wrapped lane is still rotated afterwards by the emitter's own ``tl.where``
#: block, in the emitter's own order.
HALO_TAPS: Tuple[Tuple[str, int], ...] = (
    ("a_y", 0), ("a_z", 0), ("b_x", 1), ("b_z", 1), ("c_x", 2), ("c_y", 2))


def offset_coordinates(tail: str) -> Dict[str, Tuple[str, str, str]]:
    """``{offset name: (i, j, k) expressions}``, read off the certified index lines.

    The certified complex curl composes its three shifted offsets with exactly the
    layout the real one does, so the parse is the real module's own -- shared rather
    than re-spelled, because two parsers is two chances to read a moved index wrong.
    """
    return _real.offset_coordinates(tail)


def halo_taps(tail: str) -> Dict[str, Dict[str, Tuple[int, str, str]]]:
    """``{stem: {plane: (component, offset name, mask name)}}``, off the load lines.

    The emitter writes each shifted magnetic operand as a PAIR::

        a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
        a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)

    Everything this weld needs to redirect them -- which component, which cell, which
    guard -- is in those lines, so it is PARSED rather than transcribed. The two
    planes must agree on all three or the pair is not one operand, and that is
    checked here rather than assumed.
    """
    out: Dict[str, Dict[str, Tuple[int, str, str]]] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if " = tl.load(g" not in stripped:
            continue
        register, rest = stripped.split(" = tl.load(g", 1)
        component = int(rest[0])
        body = rest[1:]
        if not body.startswith(" + 2 * ") or not body.endswith(", other=0.0)"):
            raise AssertionError(
                f"the certified complex curl no longer reads {register.strip()} as "
                f"`tl.load(gN + 2 * offset[ + 1], mask=..., other=0.0)`; the word-pair "
                f"addressing this weld preserves has changed")
        address, mask = body[len(" + 2 * "):-len(", other=0.0)")].split(", mask=", 1)
        address = address.strip()
        plane = "im" if address.endswith(" + 1") else "re"
        offset = address[:-len(" + 1")].strip() if plane == "im" else address
        if offset == "idx":
            continue
        stem = register.strip()
        if not stem.endswith("_" + plane):
            raise AssertionError(
                f"the shifted operand {stem!r} does not name its word plane; this "
                f"weld pairs the two loads by that suffix")
        stem = stem[: -len("_" + plane)]
        out.setdefault(stem, {})[plane] = (component, offset, mask.strip())
    declared = {name: component for name, component in HALO_TAPS}
    if set(out) != set(declared):
        raise AssertionError(
            f"the certified complex curl's shifted magnetic operands are "
            f"{sorted(out)}, not the declared {sorted(declared)}")
    for stem, planes in out.items():
        if set(planes) != {"re", "im"}:
            raise AssertionError(
                f"the shifted operand {stem} is loaded on {sorted(planes)}, not on "
                f"both word planes; this weld redirects a complex value, not a word")
        if planes["re"] != planes["im"]:
            raise AssertionError(
                f"the two word planes of {stem} read {planes['re']} and "
                f"{planes['im']}: a pair that disagrees on component, cell or guard "
                f"is not one complex operand")
        if planes["re"][0] != declared[stem]:
            raise AssertionError(
                f"{stem} reads component {planes['re'][0]}, not the declared "
                f"{declared[stem]}")
    return out


def curl_lift_edits() -> Tuple[Tuple[str, str], ...]:
    """The twelve ``(old, new)`` replacements the curl half takes, GENERATED.

    Six own-cell word loads become registers, and six shifted PAIRS become a single
    recompute at the cell the emitter's own index line composed, under the emitter's
    own guard, returning both planes. Nothing in the returned pairs is transcribed
    from this file except the call's shape: the component, the coordinates and the
    mask all come from the parse.
    """
    tail = _real._cut(function_source("bloch_pml_curl_step"), DECODE_END)  # noqa: SLF001
    offsets = offset_coordinates(tail)
    taps = halo_taps(tail)
    edits: List[Tuple[str, str]] = list(OWN_LOAD_EDITS)
    for stem, component in HALO_TAPS:
        _component, offset, mask = taps[stem]["re"]
        coordinates = ", ".join(offsets[offset])
        edits.append((
            f"{stem}_re = tl.load(g{component} + 2 * {offset}, mask={mask}, "
            f"other=0.0)\n"
            f"{stem}_im = tl.load(g{component} + 2 * {offset} + 1, mask={mask}, "
            f"other=0.0)\n",
            f"{stem}_re, {stem}_im = _h_tap_complex({component}, {coordinates}, "
            f"{mask},\n"
            f"                     {H_CELL_TAIL_ARGS})\n"))
    return tuple(edits)


def certified_curl_tail() -> str:
    """``bloch_pml_curl_step``'s body below the decode, every H read redirected."""
    tail = _real._cut(function_source("bloch_pml_curl_step"), DECODE_END)  # noqa: SLF001
    for old, new in curl_lift_edits():
        tail = _real.needle(tail, old, new)
    for target in range(3):
        if f"g{target} +" in tail:
            raise AssertionError(
                f"the welded complex curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register pair or a recompute")
    return tail


def lifted_curl_tail() -> str:
    """The fused kernel's own curl body, from its marker to the end, dedented."""
    source = function_source("fused_complex_constitutive_curl_H_to_D", Path(__file__))
    return _real._cut(source, CURL_BODY_ANCHOR)  # noqa: SLF001


#: The three functions the divide-free count walks -- the whole of this product's
#: device text. Declared so the count's own denominator is visible.
DEVICE_FUNCTIONS: Tuple[str, ...] = (
    "fused_complex_constitutive_curl_H_to_D", "_h_cell_complex", "_h_tap_complex")


def divide_free_evidence(path: Optional[Path] = None) -> Dict[str, Any]:
    """A STATIC COUNT over this product's device text: this seam divides nowhere.

    The claim :data:`DIVIDE_FACTS` rests on, MEASURED off the source rather than
    argued -- and measured over the PARSE TREE rather than the characters. A
    text-level slash count reads the two floor divisions of the certified index
    decode (``plane = idx // nz``) and every slash inside a docstring as divisions,
    which made the first cut of this function report ``divide_free: False`` on a
    kernel that divides nowhere. Walking :mod:`ast` counts operators, so a prose
    slash cannot enter the number and a real ``/`` cannot hide from it.

    ``path`` reads the three functions off ANOTHER file, which is what makes the zero
    falsifiable: the laptop suite plants a divide into a copy of this module and
    requires the count to move.
    """
    import ast  # noqa: PLC0415

    subject = Path(__file__) if path is None else Path(path)
    counts = {"true_divisions": 0, "floor_divisions": 0, "div_rn_calls": 0,
              "reciprocal_calls": 0, "functions_walked": list(DEVICE_FUNCTIONS),
              "source": subject.name}
    reciprocal_names = ("rcp", "rcp_rn", "reciprocal")
    for name in DEVICE_FUNCTIONS:
        tree = ast.parse(function_source(name, subject))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                counts["true_divisions"] += 1
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                counts["floor_divisions"] += 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
                counts["true_divisions"] += 1
            elif isinstance(node, ast.Call):
                attribute = node.func
                leaf = getattr(attribute, "attr", None)
                if leaf in ("div_rn", "div_rz", "div_ru", "div_rd"):
                    counts["div_rn_calls"] += 1
                elif leaf in reciprocal_names:
                    counts["reciprocal_calls"] += 1
    counts["divide_free"] = (counts["true_divisions"] == 0
                             and counts["div_rn_calls"] == 0
                             and counts["reciprocal_calls"] == 0)
    counts["floor_divisions_are_the_index_decode"] = (
        "the floor divisions are the flat-index decode (`plane = idx // nz`, "
        "`i = plane // ny`, once per function that decodes), which is integer and is "
        "the certified bodies' own")
    return counts


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names -- including the `tl.constexpr` annotations --
# through the defining module's `__globals__`, so a kernel defined inside a function
# compiles to `NameError('tl is not defined')` at first launch.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    from .complex_fields import (  # noqa: PLC0415
        _mul_coefficient_left, _mul_field_left, _rotate_field_left,
    )

    #: Boundary codes, identical to :mod:`.complex_fields`' own.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _h_cell_complex(i, j, k, live,
                        hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                        kp0, kp1, kp2, kmx, kmy, kmz, ny, nz,
                        EXPANSION: tl.constexpr):
        """``complex_fields.bloch_constitutive_step`` for side H at ONE named cell.

        ONE code path serves the program's own cell and every foreign recompute, so
        the two cannot disagree. The body below the index line is the certified
        body's own text with exactly :data:`CONSTITUTIVE_LIFT_EDITS`;
        :func:`certified_constitutive_tail` and :func:`lifted_constitutive_tail` cut
        both at their declared anchors and the laptop test asserts the two strings
        equal, so this is a LIFT and not a transcription. Returns the three stepped
        H word pairs, then the three f_w_H word pairs.

        ``live`` is the caller's per-lane validity for THIS cell, not the dispatch's
        ``idx < n_elem``: a foreign tap masked off by the certified ghost rule passes
        ``live=False``, every load inside is masked, and no out-of-range coordinate
        is ever dereferenced.
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z (dsigw).
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(kmx + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(kmy + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(kmz + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        prev_re = tl.load(wi0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.
        prev_im = tl.load(wi0 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b0 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b0 + 2 * idx + 1, mask=live, other=0.0)
        s0_re = src_re
        s0_im = src_im
        a_re = tl.load(hi0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi0 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out0_re = a_re
        out0_im = a_im

        # --- component 1 -----------------------------------------------------------
        prev_re = tl.load(wi1 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(wi1 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b1 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b1 + 2 * idx + 1, mask=live, other=0.0)
        s1_re = src_re
        s1_im = src_im
        a_re = tl.load(hi1 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi1 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out1_re = a_re
        out1_im = a_im

        # --- component 2 -----------------------------------------------------------
        prev_re = tl.load(wi2 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(wi2 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b2 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b2 + 2 * idx + 1, mask=live, other=0.0)
        s2_re = src_re
        s2_im = src_im
        a_re = tl.load(hi2 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi2 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out2_re = a_re
        out2_im = a_im
        return (out0_re, out0_im, out1_re, out1_im, out2_re, out2_im,
                s0_re, s0_im, s1_re, s1_im, s2_re, s2_im)

    @triton.jit
    def _h_tap_complex(COMP: tl.constexpr, i, j, k, valid,
                       hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                       kp0, kp1, kp2, kmx, kmy, kmz, ny, nz,
                       EXPANSION: tl.constexpr):
        """One FOREIGN magnetic word pair the curl half would have loaded.

        The whole substitution, in one place: step the cell from PRE-LAUNCH state,
        take the component the certified load named, and serve an exact
        ``(+0.0, +0.0)`` where that load's mask was False -- which is what the two
        ``other=0.0`` word loads served. NO BLOCH PHASE HERE: the emitter applies it
        afterwards, to the wrapped lane only, and applying it inside would rotate
        every unwrapped lane too.
        """
        (o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,
         _s0_re, _s0_im, _s1_re, _s1_im, _s2_re, _s2_im) = _h_cell_complex(
            i, j, k, valid, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        v_re = o0_re
        v_im = o0_im
        if COMP == 1:
            v_re = o1_re
            v_im = o1_im
        if COMP == 2:
            v_re = o2_re
            v_im = o2_im
        return tl.where(valid, v_re, 0.0), tl.where(valid, v_im, 0.0)

    @triton.jit
    def fused_complex_constitutive_curl_H_to_D(
        ho0, ho1, ho2,                  # SCRATCH out: stepped Hx,Hy,Hz    (c8 as words)
        wo0, wo1, wo2,                  # SCRATCH out: stepped f_w_H*      (c8 as words)
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz              (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx, f_w_Hy, f_w_Hz (read-only)
        b0, b1, b2,                     # Bx, By, Bz                         (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz               (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER sub-lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        nx, ny, nz, n_elem, dtdx,       # n_elem = COMPLEX cells
        pxr, pxi, pyr, pyi, pzr, pzi,   # per-axis complex64-rounded phase (conj here)
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """complex ``update_H`` + complex ``step_D``, one launch, scratch output.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u`` step IN PLACE and that is safe by construction,
        because the curl reads and writes them at the program's OWN cell only.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE WELD: update_H, computed into registers, stored to SCRATCH
        # Nothing stored here is read by this launch. The curl half below takes its
        # OWN cell's magnetic field from these registers and recomputes every foreign
        # tap from PRE-LAUNCH state through the same `_h_cell_complex`, so no program
        # observes another program's store; the plan rotates H/f_w_H after the launch
        # returns.
        (own0_re, own0_im, own1_re, own1_im, own2_re, own2_im,
         src0_re, src0_im, src1_re, src1_im, src2_re, src2_im) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        tl.store(ho0 + 2 * idx, own0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)
        tl.store(ho1 + 2 * idx, own1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, own1_im, mask=live)
        tl.store(ho2 + 2 * idx, own2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, own2_im, mask=live)
        tl.store(wo0 + 2 * idx, src0_re, mask=live)
        tl.store(wo0 + 2 * idx + 1, src0_im, mask=live)
        tl.store(wo1 + 2 * idx, src1_re, mask=live)
        tl.store(wo1 + 2 * idx + 1, src1_im, mask=live)
        tl.store(wo2 + 2 * idx, src2_re, mask=live)
        tl.store(wo2 + 2 * idx + 1, src2_im, mask=live)

        # === the certified complex step_D curl body begins here ===

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall — a (+0.0, +0.0)
        # word pair from `other=` IS the complex metallic ghost (S:1781-1783, S:1827-1829).
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

        # --- wrapped-lane predicates, one plane per axis ----------------------------
        # Up-shift wraps where the unwrapped neighbour index was n (i == n-1); the
        # down-shift wraps where it was -1 (i == 0). One plane per phased axis,
        # matching the single-plane multiply of S:1862. On a collapsed (n = 1) axis
        # every lane is the wrap lane, which is exactly xp.roll's behaviour there.
        if BACKWARD:
            wx, wy, wz = i == 0, j == 0, k == 0
        else:
            wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

        # --- loads: two words per operand ------------------------------------------
        a_re = own0_re
        a_im = own0_im
        b_re = own1_re
        b_im = own1_im
        c_re = own2_re
        c_im = own2_im
        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        a_z_re, a_z_im = _h_tap_complex(0, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        b_x_re, b_x_im = _h_tap_complex(1, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        b_z_re, b_z_im = _h_tap_complex(1, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        c_x_re, c_x_im = _h_tap_complex(2, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)
        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, EXPANSION)

        # --- Bloch phase on the wrapped lane, BEFORE the difference -----------------
        # Which operand crossed which face follows the stencil: b_x/c_x crossed x,
        # a_y/c_y crossed y, a_z/b_z crossed z. Field LEFT (S:1862); the host passed
        # the CONJUGATE for BACKWARD (S:1818-1822). `tl.where` is a bitwise select,
        # so unwrapped lanes keep the loaded words untouched.
        if PHX:
            rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
            b_x_re = tl.where(wx, rot_re, b_x_re)
            b_x_im = tl.where(wx, rot_im, b_x_im)
            rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
            c_x_re = tl.where(wx, rot_re, c_x_re)
            c_x_im = tl.where(wx, rot_im, c_x_im)
        if PHY:
            rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
            a_y_re = tl.where(wy, rot_re, a_y_re)
            a_y_im = tl.where(wy, rot_im, a_y_im)
            rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
            c_y_re = tl.where(wy, rot_re, c_y_re)
            c_y_im = tl.where(wy, rot_im, c_y_im)
        if PHZ:
            rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
            a_z_re = tl.where(wz, rot_re, a_z_re)
            a_z_im = tl.where(wz, rot_im, a_z_im)
            rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
            b_z_re = tl.where(wz, rot_re, b_z_re)
            b_z_im = tl.where(wz, rot_im, b_z_im)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
        # Complex add/sub is component-wise (measured 0/2^17 word mismatches), so the
        # grouping is kernels.py's per plane; the dtdx multiply is the zero-imag
        # complex product with the SCALAR on the left (S:1635).
        t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
        t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
        t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
        t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
        t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
        t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
        curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
        curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Writes +0.0 to BOTH planes — the array path assigns complex zero (S:1896, S:1902).
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY == METALLIC:
                curl0_re = tl.where(at_y, 0.0, curl0_re)
                curl0_im = tl.where(at_y, 0.0, curl0_im)
            if BCZ == METALLIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
            if BCX == METALLIC:
                curl1_re = tl.where(at_x, 0.0, curl1_re)
                curl1_im = tl.where(at_x, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            if BCX == METALLIC:
                curl2_re = tl.where(at_x, 0.0, curl2_re)
                curl2_im = tl.where(at_x, 0.0, curl2_im)
            if BCY == METALLIC:
                curl2_re = tl.where(at_y, 0.0, curl2_re)
                curl2_im = tl.where(at_y, 0.0, curl2_im)
        else:
            if BCX == METALLIC:
                curl0_re = tl.where(at_x, 0.0, curl0_re)
                curl0_im = tl.where(at_x, 0.0, curl0_im)
            if BCY == METALLIC:
                curl1_re = tl.where(at_y, 0.0, curl1_re)
                curl1_im = tl.where(at_y, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

        # --- split-field recurrence (stepping._apply_pml_update) -------------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). Every
        # multiply is the zero-imag product with the FIELD on the left (S:1929-1935);
        # the subtraction of the curl and of fprev is plane-wise between them. The
        # loaded p registers ARE S:1928's fprev copy.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
        p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
        q_re = q_re - curl0_re
        q_im = q_im - curl0_im
        n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)
        e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
        r_re = (r_re + n0_re) - p0_re
        r_im = (r_im + n0_im) - p0_im
        v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

        p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
        p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
        q_re = q_re - curl1_re
        q_im = q_im - curl1_im
        n1_re, n1_im = _mul_field_left(q_re, q_im, si_z, EXPANSION)
        e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
        r_re = (r_re + n1_re) - p1_re
        r_im = (r_im + n1_im) - p1_im
        v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

        p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
        p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
        q_re = q_re - curl2_re
        q_im = q_im - curl2_im
        n2_re, n2_im = _mul_field_left(q_re, q_im, si_x, EXPANSION)
        e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
        r_re = (r_re + n2_re) - p2_re
        r_im = (r_im + n2_im) - p2_im
        v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

        # --- stores: u then f (kernels.py:193-198 order), both planes ---------------
        tl.store(u0 + 2 * idx, n0_re, mask=live)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
        tl.store(u1 + 2 * idx, n1_re, mask=live)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
        tl.store(u2 + 2 * idx, n2_re, mask=live)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)

else:  # pragma: no cover - the laptop path
    fused_complex_constitutive_curl_H_to_D = None  # type: ignore[assignment]


def fused_complex_constitutive_curl_H_to_D_kernel() -> Any:
    """The shipped kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the complex fused H->D pair needs Triton to launch; the predicate, the "
            f"lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return fused_complex_constitutive_curl_H_to_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                   probe: Any = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified complex predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it -- and
    that carries the expansion licence, the subnormal-policy clause and the complex
    layout checks without this file restating any of them.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _cf.complex_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE,
                                                     probe=probe)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _cf.complex_pml_curl_coverage(fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"complex curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier
    # (driver.py:3283-3284) and the electric one is one seam later (:3317-3322). What
    # IS between them is the electric integrated-source withdraw (:3313-3314), and
    # `withdraw_hoist` owns it. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not
    # hold the source list, so a predicate that inferred "no withdraw stands" from not
    # being told would be the over-covering refusal this clause exists to prevent.
    # ON THIS CELL the clause costs nothing -- all 17 corpus instances carry
    # `integrated_electric_sources: 0` -- but it is EVALUATED, never assumed away.
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

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane
    # (`complex_fields._complex_grid_reasons`), and on THIS seam the reason is not the
    # one the D->E and B->H pairs give: neither fill runs between these two consults
    # (fill_B closes one seam earlier at driver.py:3304-3309 and fill_D opens one seam
    # later at :3324-3327), so the H->D seam is fill-FREE on a folded grid. What
    # refuses a fold here is that this product implements the `complex` and
    # `complex PML` arms and a folded complex run selects the `folded complex` arms on
    # both slots -- two further cells (2 rows and 3 rows on this backend's census),
    # each needing its own product and its own ghost-map measurement.
    mirrored = getattr(grid, "is_mirrored", None)
    for axis in range(3):
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `complex` "
                f"update_H and `complex PML` step_D arms, and a folded complex run "
                f"selects the `folded complex` arms on both slots; those cells need "
                f"their own product and their own ghost-map measurement")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED after each launch, so those attributes
    # must be allocated.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_complex_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                                  probe: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads. One home for the verdict."""
    return complex_fused_hd_pair_coverage(fields, pml, sources, probe=probe)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ComplexFusedHdPairPlan(ScratchWeldPairPlan):
    """ONE allocation-free launch for complex ``update_H`` and complex ``step_D``.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns a
    twin of each, reads the engine's CURRENT attribute immediately before the launch
    to decide which of the pair is live, word-views it there, and moves the engine's
    references only after the launch returns. A pointer captured at plan time would be
    one rotation stale -- and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between anything this launch writes (the six twins,
    ``D`` and ``fu_D``) and anything it reads (``B``, the coefficient vectors), and
    among the written volumes themselves. The base class additionally refuses a
    scratch buffer that IS its own live volume, which would be the in-place weld this
    design exists to avoid, wearing this class's name.
    """

    __slots__ = ("dtdx", "backward", "bc", "phased", "phase_values", "expansion",
                 "_b", "_targets", "_aux", "_curl_coeff", "_kps", "_kernel",
                 "_pointer", "_dtype")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 phased: Sequence[int], phase_values: Sequence[float],
                 expansion: int, block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        if len(self.phased) != 3 or len(self.phase_values) != 6:
            raise ValueError(
                "this plan binds three PH flags and six float32 phase components "
                "(re, im per axis), exactly as ComplexPmlCurlPlan does")
        self.expansion = int(expansion)
        self._dtype = getattr(targets[0], "dtype", None)
        if str(self._dtype) != "complex64":
            raise ValueError(
                f"this plan steps complex64 word pairs, not {self._dtype}")
        # THE WORD INDEX IS int32 IN THE KERNEL. A grid whose doubled cell count
        # overflows it would address the wrong words rather than fail, so it is
        # refused here -- the Dcyl complex product's own check, and the same reason.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows this kernel's 2 * idx addressing")
        self._pointer = CupyPointer
        self._b = tuple(CupyPointer(_word_view(a)) for a in flux)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._curl_coeff = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        self._kps = tuple(CupyPointer(_flat(a)) for a in constitutive_coefficients)
        if len(self._curl_coeff) != 6 or len(self._kps) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "three constitutive kps; the constitutive kms are the curl's own "
                "three, shared because both halves sit on the integer sub-lattice")
        self._check_aliases(twins, flux, targets, auxiliaries,
                            curl_coefficients, constitutive_coefficients)
        self._kernel = kernel

    def _check_aliases(self, twins, flux, targets, auxiliaries,
                       curl_coefficients, constitutive_coefficients) -> None:
        """No written volume may share an allocation with a read one, or another.

        The whole design is that nothing written by the constitutive half is read;
        an aliased pair would put the schedule back into the answer, and would do it
        through a binding rather than through the kernel text.
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
        named += [(IN_PLACE[index], array) for index, array in enumerate(targets)]
        named += [(IN_PLACE[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        for label, array in named:
            key = address(array)
            if key is None:
                continue
            if key in outputs:
                raise ValueError(
                    f"outputs {outputs[key]} and {label} are the same allocation; "
                    f"one launch would write both")
            outputs[key] = label
        inputs: List[Tuple[str, Any]] = [
            (CONSTITUTIVE_SOURCES[index], array)
            for index, array in enumerate(flux)]
        inputs += [(f"curl_coefficient{index}", array)
                   for index, array in enumerate(curl_coefficients)]
        inputs += [(f"kps{index}", array)
                   for index, array in enumerate(constitutive_coefficients)]
        for label, array in inputs:
            key = address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the launch would "
                    f"read a volume it is writing")

    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else fused_complex_constitutive_curl_H_to_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        # WORD-VIEWED HERE, not at plan time: the rotation moves which allocation is
        # live, so the view has to be taken of whatever the engine names now.
        scratch = [self._pointer(_word_view(array)) for array in writes]
        prior = [self._pointer(_word_view(array)) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._targets, *self._aux,
            *self._curl_coeff, *self._kps,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ComplexFusedHdPairPlan(shape={self.shape}, bc={self.bc}, "
                f"phased={self.phased}, expansion={self.expansion}, "
                f"block={self.block}, num_warps={self.num_warps})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)``, READ from the shipped tables.

    Their equality is the whole premise of the shared ``kms`` group, so it is
    ASSERTED rather than assumed: if either table moved, sharing the group would bind
    one half's coefficients to the other half's lattice -- a converged, smooth,
    half-cell-wrong absorber profile rather than a failure.
    """
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl = SUB_STEPS[CURL_SUB_STEP]["suffix"]
    constitutive = ("_h" if _coverage.CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
                    ["half_integer"] else "")
    if curl != constitutive:
        raise AssertionError(
            f"step_D reads the {curl or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive or 'integer'!r} one; this weld binds ONE kms group "
            f"for both halves and that is only correct while they agree")
    if int(SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD:
        raise AssertionError(
            f"launch.SUB_STEPS says step_D differences "
            f"{SUB_STEPS[CURL_SUB_STEP]['backward']}, and this product declares "
            f"BACKWARD = {BACKWARD}")
    return curl, constitutive


def plan_complex_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                               block: Optional[int] = None,
                               num_warps: Optional[int] = 1, kernel: Any = None,
                               probe: Any = None) -> Optional[ComplexFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``kernel`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped kernel and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not complex_fused_hd_pair_coverage(fields, pml, sources, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE PHASE TABLE AND ITS CONJUGATION ARE THE CERTIFIED MODULE'S, not this one's:
    # `_phase_arguments` rounds to complex64 BEFORE splitting and negates the
    # imaginary part for a BACKWARD sub-step, which is what makes the wrap
    # bit-identical to `_shift_down`'s.
    phases = _cf.bloch_phase_table(grid, kinds)
    phased, values = _cf._phase_arguments(phases, backward=bool(BACKWARD))  # noqa: SLF001
    return ComplexFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phased, values, expansion,
        DEFAULT_BLOCK if block is None else block,
        fields,
        twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )


def plan_complex_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], phases: Sequence[Optional[complex]], expansion: int,
        fields: Any, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> ComplexFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps, so a gate hands a small namespace
    holding the six rotating volumes under the engine's own names and exercises the
    rotation the engine would get. ``arrays`` supplies the twins under
    ``scratch_Hx`` ...; ``phases`` is the per-axis ``Optional[complex]`` table and the
    conjugation for ``step_D`` is applied HERE, by the certified module's own
    encoder; ``expansion`` arrives as the constexpr the caller resolved OR FORCED.
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    phased, values = _cf._phase_arguments(tuple(phases),  # noqa: SLF001
                                          backward=bool(BACKWARD))
    return ComplexFusedHdPairPlan(
        shape, dtdx, codes, phased, values, int(expansion),
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )
