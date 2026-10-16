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

THE BLOCH CONNECTION IS NOT IN THE RECOMPUTE AND MUST NOT BE. The certified complex
curl connects the UNSHIFTED partners: on a phased axis the shifted operand is a plain
roll of the stored window, and plane 0 of the unshifted partner along that axis is
replaced by MEEP's not-owned word, :func:`.complex_fields._connect` of the loaded
register through ``tl.where`` (a bitwise select). This weld redirects the LOADS -- the
own-cell registers and the six foreign recomputes -- and leaves that connection block
standing character for character, so a connected word is still MEEP's connection of
the program's own stepped ``H``. A weld that connected inside the recompute would
connect foreign taps the certified curl never connects.

=============================================================================
THE ARITHMETIC: THE REAL KERNELS' FORMS, ONCE PER STORED PART
=============================================================================

Every product on the complex path has a REAL operand (the coefficients stay float32
under complex storage, stepping.py:41-50), so each stored part takes the real kernels'
forms (:mod:`.forms`) with the same coefficient words and the same record selects --
MEEP's ``DOCMP`` loop, which steps ``f[c][0]`` and ``f[c][1]`` with real operands alone
(update_eh.cpp:149, step_db.cpp:47). No complex product is formed and no FMA is
emitted; the one multiply helper the kernel reaches is the connection
(:data:`MULTIPLY_HELPERS`), whose product with a phase's float32 parts is MEEP's
unfused one and whose negation is spelled ``* -1.0`` (Triton lowers unary ``-x`` as
``0.0 - x``, which loses the sign of ``+0``). The constitutive half is
:func:`.complex_fields.bloch_constitutive_step`'s per-part accumulation keyed on the
record's dsigw lines, with MEEP's start of H; the curl half is
:func:`.complex_fields.bloch_pml_curl_step`'s per-part stencil and forms keyed on the
record's presence word and dsigu lines, the auxiliary contract on ``fu_D`` included.

**A COMPLEX-BY-REAL DIVIDE DOES NOT OCCUR IN THIS KERNEL, AND THAT IS A MEASUREMENT
RATHER THAN AN OMISSION.** Every PML reciprocal is precomputed on the HOST into
``sinv_*`` (``pml.py``) and both halves MULTIPLY by it. The fact is carried three ways
rather than cited: a static count over the emitted text (:func:`divide_free_evidence`),
a laptop test, and an armed gate mutation that respells one multiply by ``si`` as a
divide by ``1 / si`` and requires the divergence. The divide spellings the Dcyl round
measured are RECORDED in :data:`DIVIDE_FACTS` as inapplicable-and-armed, not repeated.

**THE PLANTED ROW-ZERO TINY-NORMAL CASE IS PART OF THE GATE, NOT AN EXTRA.** A wrong
spelling of a signed-zero or underflowing word is byte-visible only on that class, and
a random battery does not reach it. This gate therefore PLANTS a fixture whose row 0
holds tiny normals of both signs beside exact zeros, and scores the arithmetic
mutations there as well as on the random battery.

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
from .complex_fields import _word_view
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

#: The one multiply helper this kernel may reach, and NO OTHER: the Bloch connection.
#: A laptop test scans the shipped kernel text for ``_mul_``/``_rotate_``/``_div_``/
#: ``_connect`` call sites and fails on anything outside this tuple -- which is how
#: "this product forms no complex product its certified halves do not form" stays a
#: measurement.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect",)

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
              "and both halves MULTIPLY each stored part by it; the emitted text "
              "contains no `/` and no div_rn",
     "where_it_is_true": "here",
     "why_it_is_moot_here": "it IS the reason -- measured by divide_free_evidence() "
                            "and armed as a gate mutation rather than cited"},
)

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR",
    "CONSTITUTIVE_LIFT_EDITS", "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES",
    "CURL_SUB_STEP", "DEFAULT_BLOCK", "DIVIDE_FACTS", "FAMILY", "HALO_TAPS",
    "HOISTS_THE_WITHDRAW", "H_CELL_TAIL_ARGS", "INSTALLABLE", "INSTALLABLE_REASON",
    "IN_PLACE", "LAUNCHES_PER_RUN", "MULTIPLY_HELPERS", "OWN_LOAD_EDITS",
    "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT",
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
                    "                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, "
                    "nz")


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
#: complex product still goes through the certified helper.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "if SCALE: lattice = RECORD_HALF_INTEGER else: lattice = 0",
     "became": "lattice = 0",
     "why": "the constexpr branch is RESOLVED to the side this product serves. "
            "update_H is SCALE 0, so the dsigw membership lines are the record's "
            "INTEGER lines, exactly the sub-lattice the shared kms group sits on. The "
            "three record_in calls are the certified calls, unedited."},
    {"line": "km_N = tl.load(kmN + <axis>, ...)",
     "became": "km_N = tl.load(kmx|kmy|kmz + <axis>, ...)",
     "why": "ONE POINTER RENAME PER AXIS, NO ARITHMETIC. Both halves sit on the same "
            "Yee sub-lattice (SUB_STEPS['step_D']['suffix'] == '' and "
            "CONSTITUTIVE_SIDES['H']['half_integer'] is False), so the curl's kms_x "
            "and the constitutive's km0 ARE THE SAME VOLUME and the fused signature "
            "binds it once. Binding the half-integer set instead compiles and is a "
            "half-cell error in the absorber profile, converged and smooth and "
            "wrong; plan_complex_fused_hd_pair ASSERTS the two suffixes agree."},
    {"line": "if SCALE: ie = tl.load(eN + idx, ...); src_re = src_re * ie; "
             "src_im = src_im * ie",
     "became": "(removed -- the constexpr branch is RESOLVED to SCALE = 0)",
     "why": "the H side reads B directly (mu = 1 baked in exactly as the array path "
            "bakes it), so the SCALE = 1 arm is text no launch of this product could "
            "reach and the three inverse-epsilon pointers leave with it."},
    {"line": "prev_re/prev_im = tl.load(wN + 2 * idx, mask=live & dwN, ...)",
     "became": "prev_re/prev_im = tl.load(wiN + 2 * idx, mask=live & dwN, ...)",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which "
            "nothing in this launch writes, under the certified dsigw guard. On this "
            "side the newly written f_w_H IS B exactly, so an in-place write would "
            "hand a racing neighbour B where it needs B_prev -- the second, easily "
            "missed half of the hazard."},
    {"line": "src_re/src_im = tl.load(gN + 2 * idx ...)",
     "became": "src_re/src_im = tl.load(bN + 2 * idx ...)",
     "why": "one pointer rename: the magnetic flux density under its own name. In the "
            "fused signature `gN` is the curl half's magnetic FIELD, so the "
            "constitutive source cannot keep that name."},
    {"line": "tl.store(wN + 2 * idx, src_re, mask=live & dwN); "
             "tl.store(wN + 2 * idx + 1, src_im, mask=live & dwN)",
     "became": "sN_re = src_re; sN_im = src_im",
     "why": "the weld writes a scratch volume, so the helper must not store. The "
            "value and the expression that produced it are untouched; only the "
            "destination moves to the caller, which writes the pair on the "
            "accumulation's cells and the pre-launch pair everywhere else (the "
            "auxiliary contract on a twin the rotation publishes) for its OWN cell, "
            "and stores nothing at all for a foreign recompute. The helper returns "
            "the dsigw flags for that select."},
    {"line": "a_re/a_im = tl.load(fN + 2 * idx, mask=live & dwN, ...)",
     "became": "a_re/a_im = tl.load(hiN + 2 * idx, mask=live & dwN, ...)",
     "why": "the accumulator is read from the PRE-LAUNCH H, const for the whole "
            "dispatch. This is what makes the foreign recompute a pure function of "
            "unwritten memory. In the fused signature `fN` is the DISPLACEMENT."},
    {"line": "if SCALE == 0: prev/a = tl.where(start, src, ...)",
     "became": "prev/a = tl.where(start, src, ...)",
     "why": "the constexpr branch is RESOLVED to the H side: MEEP's start of H "
            "(stepping.separate_h_from_b) starts H and its history as copies of B on "
            "the run's first update_H, per part, and the four selects are the "
            "certified ones. `start_h` is a launch scalar, the same at the own cell "
            "and at every foreign recompute."},
    {"line": "tl.store(fN + 2 * idx, constitutive_form(a_re, ...), mask=live); "
             "tl.store(fN + 2 * idx + 1, constitutive_form(a_im, ...), mask=live)",
     "became": "outN_re = constitutive_form(a_re, ...); "
               "outN_im = constitutive_form(a_im, ...)",
     "why": "same as the split-field store: the destination moves to the caller. The "
            "two form calls, their operands and their order are the certified ones, "
            "character for character."},
    {"line": "idx = tl.program_id(0) * BLOCK + ... ; i = plane // ny",
     "became": "(removed -- i, j and k are parameters; idx is composed from them)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a function "
            "evaluated at an ARBITRARY cell rather than at the program's own, so the "
            "cell arrives as three coordinates and the flat index is COMPOSED by the "
            "same layout the decode inverts. `live` arrives as the caller's per-lane "
            "validity: a tap the certified ghost rule masked off passes False, every "
            "load inside is masked, and no out-of-range coordinate is dereferenced. "
            "The record `cb` is read at the cell the helper is handed."},
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
    tail = needle(tail,
                  "if SCALE:\n"
                  "    lattice = RECORD_HALF_INTEGER\n"
                  "else:\n"
                  "    lattice = 0\n",
                  "lattice = 0\n")
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
        tail = needle(
            tail,
            f"if SCALE:\n"
            f"    ie = tl.load(e{target} + idx, mask=live, other=0.0)\n"
            f"    src_re = src_re * ie\n"
            f"    src_im = src_im * ie\n",
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
            f"tl.store(w{target} + 2 * idx, src_re, mask=live & dw{target})\n"
            f"tl.store(w{target} + 2 * idx + 1, src_im, mask=live & dw{target})\n",
            f"s{target}_re = src_re\ns{target}_im = src_im\n")
        tail = needle(tail, f"a_re = tl.load(f{target} + 2 * idx",
                      f"a_re = tl.load(hi{target} + 2 * idx")
        tail = needle(tail, f"a_im = tl.load(f{target} + 2 * idx",
                      f"a_im = tl.load(hi{target} + 2 * idx")
        # The branch is keyed on the line before it, which names its component:
        # components 1 and 2 spell the branch identically.
        loaded = (f"a_im = tl.load(hi{target} + 2 * idx + 1, mask=live & "
                  f"dw{target}, other=0.0)\n")
        branch = loaded + (
            "if SCALE == 0:                    # H starts as a copy of B on its "
            "own cells\n" if target == 0 else "if SCALE == 0:\n")
        tail = needle(
            tail,
            branch
            + "    prev_re = tl.where(start, src_re, prev_re)\n"
              "    prev_im = tl.where(start, src_im, prev_im)\n"
              "    a_re = tl.where(start, src_re, a_re)\n"
              "    a_im = tl.where(start, src_im, a_im)\n",
            loaded
            + ("# H starts as a copy of B on its own cells (MEEP's start of H)\n"
               if target == 0 else "")
            + "prev_re = tl.where(start, src_re, prev_re)\n"
              "prev_im = tl.where(start, src_im, prev_im)\n"
              "a_re = tl.where(start, src_re, a_re)\n"
              "a_im = tl.where(start, src_im, a_im)\n")
        form = f"kp_{target}, km_{target}, dw{target})"
        tail = needle(
            tail,
            f"tl.store(f{target} + 2 * idx, constitutive_form(a_re, prev_re, src_re, "
            f"{form},\n"
            f"         mask=live)\n"
            f"tl.store(f{target} + 2 * idx + 1, constitutive_form(a_im, prev_im, "
            f"src_im, {form},\n"
            f"         mask=live)\n",
            f"out{target}_re = constitutive_form(a_re, prev_re, src_re, {form}\n"
            f"out{target}_im = constitutive_form(a_im, prev_im, src_im, {form}\n")
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

    # The certified complex bodies' own names: the Bloch connection, the shared forms
    # and the record's word offsets, the very objects the certified kernels call.
    from .complex_fields import _connect  # noqa: PLC0415
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: PLC0415
                        curl_stencil, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: PLC0415

    #: Boundary codes, identical to :mod:`.complex_fields`' own.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _h_cell_complex(i, j, k, live,
                        hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                        kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz):
        """``complex_fields.bloch_constitutive_step`` for side H at ONE named cell.

        ONE code path serves the program's own cell and every foreign recompute, so
        the two cannot disagree. The body below the index line is the certified
        body's own text with exactly :data:`CONSTITUTIVE_LIFT_EDITS`;
        :func:`certified_constitutive_tail` and :func:`lifted_constitutive_tail` cut
        both at their declared anchors and the laptop test asserts the two strings
        equal, so this is a LIFT and not a transcription. Returns the three stepped
        H word pairs, the three B word pairs the split-field history takes, and the
        three dsigw membership flags the caller keys that history's scratch store on.

        ``live`` is the caller's per-lane validity for THIS cell, not the dispatch's
        ``idx < n_elem``: a foreign tap masked off by the certified ghost rule passes
        ``live=False``, every load inside is masked, and no out-of-range coordinate
        is ever dereferenced. The record ``cb`` is read at the cell handed in, so a
        recomputed neighbour takes its own chunk's form; ``start_h`` is the launch's
        MEEP start of H, the same at every cell.
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        lattice = 0
        dw0 = record_in(cb, lattice + 0, i)
        dw1 = record_in(cb, lattice + 4, j)
        dw2 = record_in(cb, lattice + 8, k)
        # MEEP's start of H; always 0 on the E side.
        start = start_h != 0

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z (dsigw).
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(kmx + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(kmy + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(kmz + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        prev_re = tl.load(wi0 + 2 * idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(wi0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        src_re = tl.load(b0 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b0 + 2 * idx + 1, mask=live, other=0.0)
        s0_re = src_re
        s0_im = src_im
        a_re = tl.load(hi0 + 2 * idx, mask=live & dw0, other=0.0)
        a_im = tl.load(hi0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        # H starts as a copy of B on its own cells (MEEP's start of H)
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        out0_re = constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0)
        out0_im = constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0)

        # --- component 1 -----------------------------------------------------------
        prev_re = tl.load(wi1 + 2 * idx, mask=live & dw1, other=0.0)
        prev_im = tl.load(wi1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        src_re = tl.load(b1 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b1 + 2 * idx + 1, mask=live, other=0.0)
        s1_re = src_re
        s1_im = src_im
        a_re = tl.load(hi1 + 2 * idx, mask=live & dw1, other=0.0)
        a_im = tl.load(hi1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        out1_re = constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1)
        out1_im = constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1)

        # --- component 2 -----------------------------------------------------------
        prev_re = tl.load(wi2 + 2 * idx, mask=live & dw2, other=0.0)
        prev_im = tl.load(wi2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        src_re = tl.load(b2 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b2 + 2 * idx + 1, mask=live, other=0.0)
        s2_re = src_re
        s2_im = src_im
        a_re = tl.load(hi2 + 2 * idx, mask=live & dw2, other=0.0)
        a_im = tl.load(hi2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        out2_re = constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2)
        out2_im = constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2)
        return (out0_re, out0_im, out1_re, out1_im, out2_re, out2_im,
                s0_re, s0_im, s1_re, s1_im, s2_re, s2_im, dw0, dw1, dw2)

    @triton.jit
    def _h_tap_complex(COMP: tl.constexpr, i, j, k, valid,
                       hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                       kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz):
        """One FOREIGN magnetic word pair the curl half would have loaded.

        The whole substitution, in one place: step the cell from PRE-LAUNCH state,
        take the component the certified load named, and serve an exact
        ``(+0.0, +0.0)`` where that load's mask was False -- which is what the two
        ``other=0.0`` word loads served. The Bloch connection is not applied here:
        the certified curl connects the UNSHIFTED partners at plane 0, never a tap.
        """
        (o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,
         _s0_re, _s0_im, _s1_re, _s1_im, _s2_re, _s2_im,
         _d0, _d1, _d2) = _h_cell_complex(
            i, j, k, valid, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        v_re = o0_re
        v_im = o0_im
        if COMP == 1:
            v_re = o1_re
            v_im = o1_im
        if COMP == 2:
            v_re = o2_re
            v_im = o2_im
        return tl.where(valid, v_re, 0.0), tl.where(valid, v_im, 0.0)

    @triton.jit(do_not_specialize=["start_h"])
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
        cb,                             # chunk record: both halves' forms and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # n_elem = COMPLEX cells; MEEP's start of H
        cxr, cxi, cyr, cyi, czr, czi,   # per-axis connection factor, float32 parts
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kinds
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
         src0_re, src0_im, src1_re, src1_im, src2_re, src2_im,
         dw0, dw1, dw2) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + 2 * idx, own0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)
        tl.store(ho1 + 2 * idx, own1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, own1_im, mask=live)
        tl.store(ho2 + 2 * idx, own2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, own2_im, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word pair, so
        # the rotation hands the engine an f_w_H that kept its words there and the
        # twin is written at every cell.
        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(
            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo0 + 2 * idx + 1, tl.where(dw0, src0_im, tl.load(
            wi0 + 2 * idx + 1, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx, tl.where(dw1, src1_re, tl.load(
            wi1 + 2 * idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx + 1, tl.where(dw1, src1_im, tl.load(
            wi1 + 2 * idx + 1, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx, tl.where(dw2, src2_re, tl.load(
            wi2 + 2 * idx, mask=live & (dw2 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx + 1, tl.where(dw2, src2_im, tl.load(
            wi2 + 2 * idx + 1, mask=live & (dw2 == 0), other=0.0)), mask=live)

        # === the certified complex step_D curl body begins here ===

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
        # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

        # --- loads: two words per operand ------------------------------------------
        a_re = own0_re
        a_im = own0_im
        b_re = own1_re
        b_im = own1_im
        c_re = own2_re
        c_im = own2_im
        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        a_z_re, a_z_im = _h_tap_complex(0, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_x_re, b_x_im = _h_tap_complex(1, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_z_re, b_z_im = _h_tap_complex(1, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_x_re, c_x_im = _h_tap_complex(2, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)

        # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
        # Target 0 reads c along y and b along z, target 1 a along z and c along x,
        # target 2 b along x and a along y (stepping's term table). `tl.where` is a
        # bitwise select, so every cell off plane 0 keeps its loaded words.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
        b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
        a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
        c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
        b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
        a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
        if CNX:
            r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
            c1_re = tl.where(at_x, r_re, c_re)
            c1_im = tl.where(at_x, r_im, c_im)
            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
            b2_re = tl.where(at_x, r_re, b_re)
            b2_im = tl.where(at_x, r_im, b_im)
        if CNY:
            r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
            c0_re = tl.where(at_y, r_re, c_re)
            c0_im = tl.where(at_y, r_im, c_im)
            r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
            a2_re = tl.where(at_y, r_re, a_re)
            a2_im = tl.where(at_y, r_im, a_im)
        if CNZ:
            r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
            b0_re = tl.where(at_z, r_re, b_re)
            b0_im = tl.where(at_z, r_im, b_im)
            r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
            a1_re = tl.where(at_z, r_re, a_re)
            a1_im = tl.where(at_z, r_im, a_im)

        # --- the curl (stepping._curl_from_operands), per part ---------------------
        # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
        # association with both partners, the one-partner form where the record says the
        # other partner's derivative axis is invariant, an exact +0 with neither. The
        # dtdx scale is the real scalar times each part (part_products.scale).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
        curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
        curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
        curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
        curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
        curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
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

        # --- the PML forms (stepping._apply_curl_regions), per part ----------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
        # select per target serves both parts; fu is read and written on the
        # recurrence's cells only (the auxiliary contract), both words.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
        p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl2_im, km_x, si_x, km_y, si_y, du2)

        # --- stores: fu (recurrence cells only), then the fields, both parts --------
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
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

def complex_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None
                                   ) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified complex predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it -- and
    that carries the complex layout checks without this file restating any of them.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _cf.complex_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _cf.complex_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
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


def explain_complex_fused_hd_pair(fields: Any, pml: Any, sources: Any = None
                                  ) -> "_coverage.Coverage":
    """The predicate under the name a report reads. One home for the verdict."""
    return complex_fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ComplexFusedHdPairPlan(_real.StartsH, ScratchWeldPairPlan):
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

    __slots__ = ("dtdx", "backward", "bc", "connections", "connection_values",
                 "_b", "_targets", "_aux", "_curl_coeff", "_kps", "_kernel",
                 "_pointer", "_dtype", "_h_state")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 connections: Sequence[int], connection_values: Sequence[float],
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], kernel: Any = None,
                 num_warps: Optional[int] = 1, *, chunk_bounds: Any,
                 h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        # THE CONNECTION IS THE CERTIFIED CURL PLAN'S: three kinds and six float32
        # parts from complex_fields.connection_arguments for step_D's direction, and
        # a connection only on a wrapping axis, exactly as ComplexPmlCurlPlan checks.
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        if len(self.connections) != 3 or len(self.connection_values) != 6 or any(
                kind not in _cf.CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(
                "this plan binds three connection kinds and six float32 parts (re, im "
                "per axis), exactly as ComplexPmlCurlPlan does")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            if kind and code != 0:
                raise ValueError(f"axis {axis} carries a connection but its boundary "
                                 f"code {code} does not wrap")
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
        # MEEP's start of H, on the real product's terms (fused_hd_pair.StartsH).
        self._h_state = h_state

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
            *self._curl_coeff, *self._kps, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, self.start_h(),
            *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ComplexFusedHdPairPlan(shape={self.shape}, bc={self.bc}, "
                f"connections={self.connections}, "
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
                               num_warps: Optional[int] = 1, kernel: Any = None
                               ) -> Optional[ComplexFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``kernel`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped kernel and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    if not complex_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE CONNECTION IS THE CERTIFIED MODULE'S, not this one's:
    # `connection_arguments` classifies each axis's not-owned word for step_D's
    # direction (eikna above the window under its backward differences) and rounds a
    # product's parts to float32, as the certified curl plan binds it.
    phases = _cf.bloch_phase_table(grid, kinds)
    connections, values = _cf.connection_arguments(phases, backward=bool(BACKWARD))
    return ComplexFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        connections, values,
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
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        h_state=fields,
    )


def plan_complex_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], phases: Sequence[Optional[complex]],
        fields: Any, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, *, chunk_bounds: Any) -> ComplexFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps, so a gate hands a small namespace
    holding the six rotating volumes under the engine's own names and exercises the
    rotation the engine would get. ``arrays`` supplies the twins under
    ``scratch_Hx`` ...; ``phases`` is the per-axis ``Optional[complex]`` table and
    step_D's connection is classified HERE, by the certified module's own encoder.
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    connections, values = _cf.connection_arguments(tuple(phases),
                                                   backward=bool(BACKWARD))
    return ComplexFusedHdPairPlan(
        shape, dtdx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
