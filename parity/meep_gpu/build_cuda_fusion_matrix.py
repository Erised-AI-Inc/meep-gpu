#!/usr/bin/env python3
"""THE CUDA FUSION BOARD — which (curl, constitutive) pair each corpus row drives at
each fusable seam on the HAND-CUDA track, and whether that pair could be welded into
one launch at all.

WHY THIS IS NOT ``build_fusion_matrix.py`` WITH A DIFFERENT CENSUS
==================================================================

The Metal board (``build_fusion_matrix.py``, beside this file) asks three questions
per cell: which arm the COMPOSER selected, whether a shipped FUSED PRODUCT covers that
pair, and whether the fused signature FITS the platform's binding ceiling. Since
2026-08-28 the hand-CUDA track answers the FIRST of those the same way — the composer
shipped, and this board was re-pointed at it. The other two still change, and the
third dissolves:

1. **THERE IS A COMPOSER NOW, AND ITS SELECTION IS THE CELL.** Until 2026-08-28
   ``meep_gpu/cuda_kernels/`` had no ``arms.py`` and no ``registry``, so this board
   derived a cell from which shipped PREDICATE admitted the row at a slot and proved
   that well defined by measuring the predicates disjoint. Both modules now ship
   (``arms.py`` walks ``STEP_ORDER`` and hands each slot's admitted arms to the same
   ``_select_slot`` the Triton and Metal tracks use), and the earlier version of this
   file refused to run for exactly that reason: *"with a composer present the
   authoritative cell is the arm it selects and this walk would report a shape the
   engine does not run."* This is that re-point.

   A cell is now **the ARM the composer selects at that slot**, read from the CUDA
   census's ``plan_step.selected`` / ``selected_family`` / ``selected_kernel``, which
   is the same column the Metal board reads on its own track. The refusal directions
   invert with it: the script now REFUSES if the composer modules are ABSENT (the
   record it reads could not have been produced) and refuses if the census carries no
   composer column. Predicate disjointness is no longer what makes a cell well
   defined — the composer is fail-closed and refuses an ambiguous slot rather than
   picking by table order — so what is measured here instead is the composer's own
   REFUSAL census, and an ``ambiguous`` refusal at any slot a seam needs still raises.

   Two facts follow from reading the selection rather than the admission, and both are
   improvements rather than restatements. A family may ship a DIFFERENT body at
   different slots — ``cuda_nonlinear`` plans ``update_E_pml_real_nonlinear`` on the
   electric side and reuses the ordinary ``update_H_pml_real`` on the magnetic one —
   so the fitness verdict is keyed on the SELECTED KERNEL, not on the family. And
   families the old predicate walk never had a column for (``cuda_folded_offdiag``,
   ``cuda_dispersive``, ``cuda_complex_beta``, …) now appear in the cells, because the
   composer selects them.

2. **THERE IS ONE FUSED PRODUCT, AND THIS BOARD NOW COUNTS IT.**
   ``fused_magnetic_pair.py`` welds ``step_B_pml_real``, the three in-seam passes and
   ``update_H_pml_real`` into one launch. Two device releases stand behind it, both
   under both float32 subnormal policies: ``cuda_fused_magnetic_pair_2026-08-27``
   (the wall wipe carried, both mirror fills refused) and
   ``cuda_fused_magnetic_pair_2026-08-29_fillcarry`` (all three carried, through the
   ownership inversion).

   The "served by a fused product" column was a HARD-CODED ZERO until 2026-08-29,
   for two different reasons in succession, and neither survives: first there was no
   such product, and then there was one whose module imported CuPy at scope, so the
   backend-free census recorded ``askable: false`` and this board went on printing 0
   beside a note saying no module emitted such a kernel — which had stopped being
   true. The module now takes CuPy defensively, the census asks
   ``covers_fused_magnetic_pair`` on every census row, and the column is that answer
   COUNTED (:data:`FUSED_PRODUCT_COLUMN`). It is the shipped predicate's own verdict
   and never a restatement of its clauses: a board that re-derived them could admit a
   row the launch cannot serve, which is a silent wrong answer rather than a wrong
   number. A census that could not ask is a REFUSAL here, not a zero.

   ``dispatch`` still reads ``wired: false``, so SERVED means "this product's
   predicate admits the row", never "anything runs it". The board's other question is
   unchanged: **which seams a product COULD be built for, ranked by how many corpus
   rows would use it.**

3. **THE FITNESS VERDICT IS NOT A BINDING COUNT.** Metal's ceiling is
   ``MAX_BUFFER_BINDINGS = 31`` and its per-cell verdict is arithmetic on pointer
   counts. CUDA has no such ceiling that binds: the kernel parameter space is 4KB
   (32,764 bytes on sm_70+), and the widest kernel in this package is nowhere near it.
   The CUDA verdict is **POINTWISE vs STENCIL**:

       Fusing a curl into the constitutive that follows it puts both programs in ONE
       launch. The curl writes B (or D) IN PLACE. If the constitutive half then reads
       that same volume ONLY at the cell the thread owns, every read is of a value
       that thread itself just wrote, and the weld is a register hand-off — POINTWISE,
       and buildable. If the constitutive half reads a NEIGHBOUR of that volume, it is
       reading a cell ANOTHER BLOCK writes in the same launch, and CUDA has no
       grid-wide barrier inside an ordinary launch. The answer would depend on block
       schedule. STENCIL, and not buildable byte-identically without cooperative
       groups (which caps the grid at the resident-block count).

   That verdict is MEASURED from the emitted device code, not read off a kernel name:
   :func:`analyze_kernel_reads` strips C comments, finds the thread's own flat
   index, walks every array subscript in the kernel text and through every
   ``__device__`` helper reachable from it — resolving helper index parameters to
   their call sites by fixpoint — and reports each read that lands anywhere other
   than the thread's own cell. A second detector,
   :func:`_shifted_coordinate_helpers`, catches the case a subscript walk cannot see:
   an emitter that builds the neighbour index in PYTHON and interpolates it whole.
   :data:`DETECTOR_CONTROLS` runs four independently-settled cases through the same
   function before any verdict is reported, and the run REFUSES if one does not
   reproduce. A family whose source this script cannot select is UNDETERMINED and is
   said so — never guessed.

THE SEAM, NOT THE SLOT
======================

Unchanged from the Metal board, and for the same driver reason. A fused product
serves a ROW at a SEAM. The seams and the passes between their halves are DERIVED
from ``driver.py`` at run time by locating the call sites, rather than transcribed —
the Metal board's transcribed line numbers (3282-3306) are already nine lines stale
against this tree, which is exactly the failure deriving them prevents.

The denominator is

    rows x 2 curl->constitutive seams
  + (rows carrying a susceptibility) x 1 E->P seam
                                                   -----

counted the same way the 759-slot denominator adds the polarization sub-step only
where it exists: 194 x 2 + 15 = 403 on the 2026-09-03 basis (186 x 2 + 15 = 387 on
every board before it). Both counts are read from the census, not spelled.

WHAT THIS BOARD READS — ONE CENSUS, NO JOIN
============================================

The CUDA census (``results/cuda_predicate_coverage_2026-08-28b/`` and later) lifted
all engine-accepted corpus rows — 186 until 2026-09-03, 194 from
``cuda_predicate_coverage_2026-09-03_extended`` on, the eight added being example
scripts the 2026-08-09 corpus campaign accepted in its _blocked/_gdsii/_sigma legs —
called every live hand-CUDA predicate on each,
AND asked the shipped composer what it selects. Every fact this board needs is in
that one record:

  * ``plan_step.selected`` / ``selected_family`` / ``selected_kernel`` — the cell.
  * ``plan_step.refusals`` — why a slot has no arm, in the composer's own words.
  * ``configuration.source_field_types`` — whether a row's sources are magnetic or
    electric, which decides the SOURCE-SEAM CEILING. A property of the LIFTED MEEP
    SCRIPT, not of a backend.
  * ``plan_step.live`` — which in-seam driver passes actually run on the row. The
    Metal board MEASURED that deriving this from the boundary triple is wrong on 32
    and 85 rows respectively, so it is READ rather than re-derived.

UNTIL 2026-08-28 THE LAST TWO WERE JOINED OUT OF ``metal_coverage_tranche6_2026-08-19``
because the CUDA census recorded neither. The promoted battery
(``cuda_predicate_battery.py``) now records both on every CUDA row, so the join is
GONE — and its removal is checked rather than assumed: this script refuses unless
every measured row carries ``configuration.source_field_types``, a resolved
``plan_step.live``, and the composer column. A partial record would silently price the
unrecorded rows as source-free and pass-free, which is the optimistic direction.

The composer is asked through the census's CuPy-backend shim, not on a real device —
every shipped CUDA predicate short-circuits on a non-CuPy array module, so a composer
asked with the real NumPy grid selects almost nothing. That is the same factoring the
predicate columns use (``covered_modulo_backend``), and the census records a per-row
``plan_step.shim`` soundness probe beside it. The subject digest the census recorded
is RE-COMPUTED here against the live tree and the run refuses on drift: this board
reads DEVICE CODE from the working tree while its cells come from a recorded
selection, and those two must describe the same package.

Usage (from anywhere)::

    MEEP_GPU_CUDA_CENSUS=cuda_predicate_coverage_2026-08-28b \
    /path/to/python -u parity/meep_gpu/build_cuda_fusion_matrix.py \
        --out parity/meep_gpu/results/fusion_matrix_cuda_2026-08-28b
"""

from __future__ import annotations

import argparse
import ast
import collections
import os
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

#: The results tree this cut reads its censuses from. Named explicitly, and the
#: package root put on ``sys.path`` explicitly, for the reason the Metal board
#: states: a canonical script that only works from one working directory is the
#: same trap as one that silently reads a stale ancestor's clauses.
RESULTS = HERE / "results"
#: ``the repository root`` — this file sits at ``parity/meep_gpu/``.
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# SERVED IN DISPATCH, measured rather than read off ``certification.json``'s
# ``dispatch`` block. That block records whether anyone wired the product into a
# planner; the question here is whether the DISPATCHER can select it, and the two
# stopped being the same question on 2026-08-29. See ``dispatch_reachability``.
import dispatch_reachability as _reach  # noqa: E402

# THE RECONCILED BUCKET NAMES AND THE THREE TIERS, shared with the Metal and Triton
# boards (release decision R1/R2, 2026-09-02). One name list in one file is what stops a
# fourth vocabulary growing back; the module also carries the floors that make the
# decomposition and the served total REFUSABLE rather than merely printed.
import dispatch_agreement  # FLOOR 8; see its docstring for why it is not in fusion_taxonomy
import fusion_taxonomy  # noqa: E402
# THE FOURTH SEAM, update_H -> step_D (2026-09-04): one pricing rule for the three
# boards, off the probe artifact and the driver's own text. See h_to_d_seam.py.
import h_to_d_seam  # noqa: E402

# THE NEAR FILL'S SOURCE ROW IS READ, NOT SPELLED, exactly as the two sibling boards
# read it: `_deposit_is_repairable_like_for_like` mirrors `deposit_repair.repairable`'s
# too-short-fold refusal, and `deposit_repair._folded_seam_reasons` compares against
# `stepping.MIRROR_SOURCE_INDEX`. A board that spelled the number would keep answering
# the old threshold after the engine moved and would ADMIT folds the shipped predicate
# refuses -- over-counting, the one direction that reads as success.
try:
    from meep_gpu.stepping import MIRROR_SOURCE_INDEX as NEAR_SOURCE_INDEX
except Exception:  # pragma: no cover - the row the near fill images is not guessable
    raise SystemExit("meep_gpu.stepping.MIRROR_SOURCE_INDEX is not importable; the "
                     "row the near fill images -- and so which folds the deposit "
                     "repair refuses -- cannot be read and this script refuses to "
                     "assume it")

PACKAGE = REPO_ROOT / "meep_gpu"
CUDA_KERNELS = PACKAGE / "cuda_kernels"
DRIVER = PACKAGE / "driver.py"

#: The standing hand-CUDA census: every engine-accepted row (194 since 2026-09-03),
#: every live CUDA predicate called on each
#: AND the composer asked, on a NumPy host behind the CuPy-backend shim.
#: Overridable so a fresh census can be measured against without editing this file.
#:
#: MOVED 2026-08-29 to the census the tree actually matches. The previous default,
#: ``cuda_predicate_coverage_2026-08-28b``, predates the fill-carry round, so
#: :func:`subject_pin` refused every argument-free run by name — fail-closed and
#: correct, but it meant the board could not be built without knowing which env var
#: to set, and the last three cuts all set it. The pin still guards: if this default
#: goes stale the same way, the run refuses rather than pairing a stale selection
#: with a fresh read of the device code.
#: MOVED AGAIN 2026-08-30, for the same reason and by the same rule. The two hand-CUDA
#: complex B/H products landed as `meep_gpu/cuda_kernels/complex_fused_magnetic_pair.py`
#: and `.../cylindrical_fused_magnetic_pair.py`, and `SUBJECT_GLOB` is
#: `meep_gpu/cuda_kernels/*.py` -- so the subject manifest moved the moment those two
#: files existed, and `subject_pin` refused every argument-free run by name. The census
#: cut with them, `cuda_predicate_coverage_2026-08-30_complexfuse`, is the one this tree
#: hashes to: 211 rows across the same three legs as its predecessor (134 tests, 52
#: examples, 25 parametrised), and its recorded `subject_manifest_sha256` was checked
#: against a live recomputation over all 69 subject files before this default moved.
#: The pin still guards -- a stale default refuses rather than pairing a stale selection
#: with a fresh read of the device code.
#: MOVED A THIRD TIME LATER THE SAME DAY, and the refusal is what moved it:
#: `meep_gpu/cuda_kernels/fused_electric_pair.py` and its suite landed, the subject
#: manifest went from 69 files to 71, and the argument-free run refused by name rather
#: than pricing a fresh read of the device code against the previous selection. The
#: census cut with them is `cuda_predicate_coverage_2026-08-30_electric2`: the same
#: three legs and the same 211 rows, its `subject_manifest_sha256`
#: (`9451c431b4c9…`) checked against a live recomputation over all 71 subject files
#: before this default moved.
#:
#: ITS DRIVER IS THE TRANCHE COPY BESIDE IT, NOT `parity/meep_gpu/measure_predicate_coverage.py`,
#: and that is a fact worth writing down rather than rediscovering: the promoted driver
#: has gained a `--battery` switch and a by-name root resolver but LOST the
#: `subject_manifest_sha256` stamp, so a census cut with it records `None` on every row
#: and `subject_pin` below refuses it -- correctly, and for a reason that reads as
#: drift. Every CUDA census this board can consume is cut with the copy that stamps.
#: SO IT STOOD UNTIL 2026-09-03: the promoted driver now computes the subject digest
#: once before any leg and stamps every row with it beside ``battery_sha256`` (an
#: unpinned census is refused), and ``_extended`` below was cut with it -- no driver
#: copy sits beside that record, and its example rows name the interpreter each ran
#: under, which only the promoted driver records.
#: MOVED A FOURTH TIME ON 2026-08-31, and by the same rule as the three before it: two
#: more products landed in ``meep_gpu/cuda_kernels/`` (``complex_fused_electric_pair.py``
#: and ``cylindrical_fused_electric_pair.py``, plus their suite), ``SUBJECT_GLOB`` is
#: ``meep_gpu/cuda_kernels/*.py``, so the subject manifest went from 71 files to 74 and
#: the argument-free run refused BY NAME rather than pricing a fresh read of the device
#: code against the previous selection. The census cut with them is
#: ``cuda_predicate_coverage_2026-08-31_complexelectric``: the same three legs and the
#: same 211 rows, cut with the same tranche-copy driver (the one that STAMPS
#: ``subject_manifest_sha256``) and reading the live battery beside it, which gained the
#: two new fused-product columns in the same edit.
#: MOVED A FIFTH TIME ON 2026-08-31, by the same rule as the four before it: one more
#: product landed in ``meep_gpu/cuda_kernels/`` (``no_pml_complex_fused_electric_pair.py``
#: plus its suite), ``SUBJECT_GLOB`` is ``meep_gpu/cuda_kernels/*.py``, so the subject
#: manifest moved again and the previous selection is not a reading of THIS device
#: code. The census cut with it is ``cuda_predicate_coverage_2026-09-01_eppair2``:
#: the same three legs and the same 211 rows, cut with the same tranche-copy driver
#: (the one that STAMPS ``subject_manifest_sha256``) and reading the live battery
#: beside it, which gained the eighth and ninth fused-product columns -- the two
#: E->P products -- in the same campaign that certified them.
#: MOVED A SIXTH TIME ON 2026-09-01 (the certification-close round), same rule:
#: the folded-complex and complex-beta curl pairs moved into CERTIFIED_KERNELS,
#: ``covers_fused_magnetic_pair`` gained the measured nonlinear-H widening, and
#: the ``cuda_cylindrical`` registry row's kernel label was corrected to the
#: entry point its module ships (``fused_cyl_{slot}_pml_real``), so the subject
#: manifest and the selected-kernel labels both moved. The census cut with it is
#: ``cuda_predicate_coverage_2026-09-01_certclose``: same three legs, same
#: tranche-copy stamping driver, same live battery (no new product columns --
#: this round certified halves and widened one predicate; it shipped no new
#: fused product).
#: MOVED A SEVENTH TIME ON 2026-09-02 (the residual-welds round), same rule as
#: every move before it: the five residual fused magnetic products landed in
#: ``meep_gpu/cuda_kernels/`` (six modules -- the shared cf fill carry and the
#: five welds -- plus their suite), ``SUBJECT_GLOB`` is
#: ``meep_gpu/cuda_kernels/*.py``, so the subject manifest moved and the
#: argument-free run refused BY NAME rather than pricing a fresh read of the
#: device code against the previous selection. The census cut with them is
#: ``cuda_predicate_coverage_2026-09-02_residualwelds``: the same three legs and
#: the same 211 rows, cut with the same tranche-copy stamping driver, reading
#: the live battery beside it, which gained the five new fused-product columns
#: in the same campaign that certified them (10 released legs on the GPU host,
#: results/cuda_fused_residual_*_2026-09-02/).
#: MOVED AN EIGHTH TIME ON 2026-09-02 (the residue round), by the rule that has
#: moved it every time before: SEVEN more modules landed in
#: ``meep_gpu/cuda_kernels/`` -- the shared D-side complex carry, the five
#: ELECTRIC twins and the complex no-absorber E->P weld -- plus their suite, so
#: ``SUBJECT_GLOB`` went from 85 files to 93 and the argument-free run refused
#: BY NAME rather than pricing a fresh read of this device code against the
#: previous selection. Two of the standing modules also moved in the same round
#: (``complex_beta_`` and ``special_kz_fused_magnetic_pair``, whose
#: ``CARRIES_DEPOSIT_REPAIR`` was re-measured TRUE and whose blocked rows the
#: board therefore now credits), which is a second, independent reason the old
#: selection could not be paired with this tree. The census cut with them is
#: ``cuda_predicate_coverage_2026-09-02_electrictwins``: the same three legs and
#: the same 211 rows, cut with the same tranche-copy stamping driver, reading
#: the live battery beside it, which gained the six new fused-product columns in
#: the same campaign that certified them (results/cuda_residue_*_2026-09-02/ and
#: the re-gate tree results/cuda_regate_2026-09-02_residue/).
#: MOVED A NINTH TIME LATER THE SAME DAY, and what moved it is worth recording
#: because the first cut MEASURED it: the E->P weld's
#: ``_drive_identity_problem`` established "update_P reads what update_E wrote"
#: only through ``array.data.ptr``, a CuPy attribute, so on the backend-free
#: census all four rows of its cell refused with a reason about the drive
#: whatever the drive was -- and the ``_electrictwins`` board printed a
#: certified product's whole cell as served by nothing. The clause now settles
#: the fact by OBJECT identity first (the DEVICE TEXT is unchanged; the weld was
#: re-gated in its own right as ``..._2026-09-02b``), the module's bytes moved
#: with it, and the census cut against those bytes is
#: ``cuda_predicate_coverage_2026-09-02_electrictwins_c``.
#:
#: (``_electrictwins_b`` is on disk and this board REFUSES it, correctly and by
#: name: a subject file was edited while that cut was running, so its rows carry
#: TWO manifests and neither is a reading of one tree. The refusal is the
#: mechanism working -- the remedy is a cut over a quiescent tree, which
#: ``_c`` is, and never a relaxed pin.)
#: 2026-09-02, THE STENCIL ROUND: ``_stencilwelds`` re-cuts the same 186 rows over a
#: tree carrying the two off-diagonal stencil welds and the off-diagonal E->P weld,
#: with the three new fused columns the battery now asks. It is cut over a QUIESCENT
#: tree -- the modules' CERTIFIED_KERNELS declarations were their final bytes before
#: both the gate campaign and this census -- so its rows carry ONE subject manifest,
#: which is what the pin below requires and what ``_electrictwins_b`` failed.
#: MOVED A TENTH TIME ON 2026-09-03, and for the first time by the CORPUS as well as
#: the subject. The corpus: eight example scripts the 2026-08-09 campaign accepted in
#: its _blocked/_gdsii/_sigma legs had gone unpriced on every board because the
#: census driver named only the stock leg's lift record;
#: ``cuda_predicate_coverage_2026-09-03_extended`` prices them -- 194 rows (60
#: examples + 134 tests, 24 of them the parameterised leg's matched rows), 403 PRICED
#: / 384 FUSABLE / 359 ATTAINABLE where every earlier cut had 387 / 368 / 344. The
#: board cut over it (``fusion_matrix_cuda_2026-09-03_extended``) reads 353/403
#: served against ``fusion_matrix_cuda_2026-09-03_land``'s 340/387 (328/387 before
#: the three-slot weld landed), with 0 status changes on the 186 shared labels: 13 of
#: the 16 new instances served, the 3 unserved being coupler.py's D->E seam
#: (source_seam_unbracketable, the 25th) and dipole_in_vacuum_cyl_off_axis.py's two
#: seams (missing_half: no arm admits complex64 cylindrical storage with m != 0).
#: The subject: the three-slot weld's partition flip (UNCERTIFIED -> CERTIFIED)
#: moved the manifest, so ``_stencilwelds`` (8dc04a1a...) refuses by name and
#: ``_extended`` (3c865730...) is the one this tree hashes to; the shipped bytes were
#: re-gated as ``results/cuda_regate_2026-09-03_three_slot_shipped`` (4/4 legs).
#:
#: MOVED 2026-09-06 TO ``cuda_predicate_coverage_2026-09-06_hd``, the census the
#: standing board (``fusion_matrix_cuda_2026-09-06_hd``, 483/597) was cut over. Every
#: board from ``_2026-09-04_arbfix`` on was cut over a census this default did not
#: name, and on this backend the stale default does not even answer a different
#: question — :func:`subject_pin` refuses it by name, because the subject moved with
#: every landing since. ``_hd`` is the one whose recorded manifest (67c8b062...) this
#: tree hashes to. Against its predecessor ``_2026-09-05_postr1`` (5654d61a...) the
#: two censuses were diffed row by row: identical 194 rows, identical configurations,
#: identical ``plan_step.selected`` at every slot, 0 predicate columns moved on any
#: row; the only difference is the new ``cuda_fused_hd_pair`` verdict column and the
#: manifest. So the +124 between the two boards is the H->D product alone — 124
#: H_to_D instances buildable_not_built -> served, 0 instances moved by the census.
#: An argument-free cut over ``_hd`` reproduces the standing board with zero
#: instances changed (``fusion_matrix_cuda_2026-09-06_consolidated``).
#:
#: MOVED 2026-09-07 TO ``cuda_predicate_coverage_2026-09-07_cyl2``, the census cut
#: on the cylindrical H->D wiring: the subject grew 116 -> 118 modules when
#: ``cuda_kernels/cylindrical_{real_,}fused_hd_pair.py`` appeared and moved again
#: when ``fused_pairs.py`` / ``registry.py`` / ``cuda_kernels/test_fused_hd_pair.py``
#: took the wiring, so :func:`subject_pin` refused ``_hd`` (67c8b062...) by name;
#: ``_cyl2`` records 0f499c2a... over 118 files, which this tree hashes to. It is the
#: first census whose battery carries the two cylindrical H->D verdict columns
#: (``cuda_predicate_battery.py``, same edit), which is what lets the H_to_D block
#: credit the two cells. Against ``_hd`` the expected move is +19 H_to_D instances,
#: all cylindrical (3 real m = 0 rows + 16 complex rows; the 2 withdraw rows stay
#: refused by name), 0 instances moved by the census elsewhere. A partial
#: ``_2026-09-07_cyl`` directory beside it was ABANDONED mid-cut (its battery lacked
#: the two columns) and is not a census.
#: MOVED AGAIN 2026-09-07 TO ``cuda_predicate_coverage_2026-09-07_wired2``, the census
#: cut on the CARTESIAN COMPLEX H->D wiring. The subject grew when
#: ``cuda_kernels/complex_fused_hd_pair.py`` joined the tree and moved again when the
#: wiring took ``fused_pairs.py``, ``registry.py`` and that module's own partition
#: declaration, so :func:`subject_pin` refuses ``_cyl2`` by name. It is the first census
#: whose battery carries the ``cuda_complex_fused_hd_pair`` verdict column
#: (``cuda_predicate_battery.py``, same edit), which is what lets the H_to_D block credit
#: the plain and folded complex cells. Against ``_cyl2`` the expected move is +22 H_to_D
#: instances -- 17 on ``H_to_D (cuda_complex/complex, cuda_complex/complex)`` and 5 on
#: ``H_to_D (cuda_complex/complex, cuda_complex_folded/folded complex)``, ONE product
#: over both -- and 0 instances moved by the census anywhere else.
#:
#: ``_wired`` (no digit) IS A COMPLETE CENSUS AND NOT THIS ONE. It was cut on the same
#: battery two hours earlier, before ``cuda_kernels/test_fused_hd_pair.py`` took the
#: fourth-candidate edit the wiring makes true; that file is inside ``SUBJECT_GLOB``, so
#: the subject moved (0d263ec4... -> f295d0d4...) and :func:`subject_pin` refuses
#: ``_wired`` BY NAME. It is kept as the record of the tree it was cut against rather
#: than overwritten -- the census script refuses to write over one -- exactly as
#: ``_cyl`` sits beside ``_cyl2``.
#:
#: MOVED AGAIN 2026-09-08 TO ``cuda_predicate_coverage_2026-09-08_merge``, the census
#: cut after the H->D wiring landed. Nothing about the CUDA weld ledger moved -- both
#: its tiers measure zero drift (197 raw pins and 197 ``code_identity`` pins, all live)
#: -- so this is a SUBJECT re-cut and not a re-gate: no CUDA gate was re-run for it.
#: ``SUBJECT_GLOB`` went from 119 files to 125. Six arrived
#: (``complex_beta_fused_hd_pair.py``, ``conductive_bfast_fused_hd_pair.py`` and
#: ``special_kz_fused_hd_pair.py``, each with its own test module) and three moved in
#: place (``fused_pairs.py``, ``registry.py``, ``test_fused_hd_pair.py``), taking the
#: manifest f295d0d4... -> 9aaef043... and making :func:`subject_pin` refuse ``_wired2``
#: BY NAME. The re-cut carries 219 rows on the same three legs (60 examples, 134 tests,
#: 25 tests_param, 24 of them matched by facts with 0 unbalanced groups) and the SAME
#: battery bytes ``_wired2`` used (c95f1cb6...), so the predicate answers are asked by
#: the same instrument and only the tree underneath them changed.
#:
#: THE SUBJECT PIN IS NOT THE ONLY GATE, AND CLEARING IT EXPOSES THE OTHER ONE. This
#: board's :data:`FUSED_PRODUCT_COLUMNS` declares 36 columns; a census cut with the
#: battery above answers 35. ``cuda_conductive_bfast_fused_hd_pair`` -- the module and
#: the column here both landed with the wiring -- has no column in
#: ``cuda_predicate_battery.py``, which last moved the day before, so
#: :func:`_fused_verdicts` refuses every argument-free run by name. NO CENSUS CAN FIX
#: THAT: the battery has to gain the column first, and the census after it. The refusal
#: is the correct one to be left standing meanwhile -- a board that inferred the missing
#: column would be reporting an answer nobody measured.
#:
#: MOVED AGAIN 2026-09-08 TO ``cuda_predicate_coverage_2026-09-08_board``, and the move
#: is the block above being CLEARED THE ONLY WAY IT SAID IT COULD BE. The battery gained
#: the ``cuda_conductive_bfast_fused_hd_pair`` column -- asked without a licence and off
#: the seam string ``update_H -> step_D``, exactly as its sibling ``cuda_fused_hd_pair``
#: is -- and this census was cut AFTER it, so the 36th answer is measured rather than
#: inferred. Nothing else moved: ``SUBJECT_GLOB`` is the same 125 files and the same
#: manifest ``9aaef043...`` the ``_merge`` cut recorded, so :func:`subject_pin` resolves
#: on the same bytes; what changed is the INSTRUMENT, and the battery digest moves with
#: it (``c95f1cb6...`` -> the digest this cut records), which is why a new census is
#: required rather than a re-read of ``_merge``. ``_merge`` stays selectable by name as
#: the record of the 35-column instrument.
#: MOVED AGAIN 2026-09-11, by the same rule and for the same cause. The dispatch
#: round edited `cuda_kernels/fused_pairs.py` (the warm pass) and `fastpath_cuda.py`
#: (the release table), both of which are in the 125-file subject set, so
#: `subject_pin` refused `_2026-09-08_board` BY NAME rather than pairing that cut's
#: recorded selection with a fresh read of this tree's device code.
#: MOVED A THIRD TIME the same day, to `_certified`. The five kernels the 09-11
#: campaign released were moved from `UNCERTIFIED_KERNELS` into `CERTIFIED_KERNELS`
#: in four `cuda_kernels` modules AFTER `_tables` was cut, and the partition test
#: beside them was inverted to match -- five of the 125 subject files, which
#: `subject_pin` named one by one. Those five are the whole of the drift: the census
#: itself measures which corpus rows drive which slots and is unchanged by a
#: certification list, but pairing a recorded selection with a fresh POINTWISE /
#: STENCIL read of a tree it was not cut against is exactly the pairing the pin
#: exists to refuse. MOVED A FOURTH TIME to `_shipping`, for the recursive reason
#: this subject set makes unavoidable: `cuda_kernels/test_ade_update_p.py` is BOTH a
#: census subject and the test that names the ADE record block, so pointing it at the
#: re-cut moved a subject of the census that had just been cut. The census is cut LAST
#: in a round, after every `cuda_kernels/*.py` edit has landed.
#: MOVED A FIFTH TIME, to `_2026-09-14_weldgrid`, cut after the complex no-absorber
#: three-slot weld's launch-grid repair moved a `cuda_kernels` subject. Every cut since
#: (`fusion_matrix_cuda_2026-09-14_weldgrid`, `_2026-09-15_weldgrid`, `_2026-09-15_offdiag`)
#: already named it through MEEP_GPU_CUDA_CENSUS while this default still named
#: `_shipping`, whose recorded subject no longer matches the tree.
#: MOVED A SIXTH TIME, to `_2026-09-15_epsfix`, and this one RECOVERS A ROW rather than
#: following an edit. `_2026-09-14_weldgrid` carries
#: `test_simulation.py::TestSimulation.test_epsilon_input_file` as `measured: false,
#: note: child died`, which is why this table's denominator read 594 against the other
#: two backends' 597. The child died for a reason that was never about the row: it reads
#: `cyl-ellipsoid-eps-ref.h5` from the corpus's SIBLING `tests/` directory
#: (test_simulation.py:311-318, `dirname(__file__)/../../tests`), and the local corpus
#: copy held only the `python/` subtree, so MEEP's reader crashed in the child. With the
#: fixture restored the row measures in 5.8 s. The re-measure was a `--resume` into a
#: COPY of the census, so `_2026-09-14_weldgrid` is untouched as the provenance of the
#: 346-of-594 cut; exactly one row's `measured` flag moved and both subject-digest
#: manifests are byte-identical.
#:
#: THE COPY IS SELECTIVE, AND THE FIRST ONE THAT WAS NOT DOUBLE-COUNTED. `_epsfix` (kept
#: beside this one as the record of the mistake) was a straight copy, and the resume ran
#: `test_simulation.py` as ONE child, writing a module-level `per_row_tests/
#: test_simulation.json` holding all 21 rows BESIDE the 20 per-case records the original
#: crash-recovery had written one child at a time. The board loads both, so those 21 rows
#: were counted twice: 193 rows -> 214, denominator 594 -> 657, dispatch 346 -> 385. None
#: of that was real. `_epsfix2` copies everything EXCEPT `per_row_tests/test_simulation__*`,
#: so the module-level record is the only one for that module and the row basis reads the
#: 194 the other two backends carry.
#:
#: A CRASH-RECOVERED MODULE IS THE SHAPE TO WATCH: `measure_predicate_coverage.py`
#: retries one case per child when a module child dies, and a later `--resume` of the same
#: module writes the module-level record without removing the per-case ones.
#:
#: AND `--resume` APPENDS TO THE LEG JSONL RATHER THAN REPLACING ROWS. That, not the
#: per-case records, is what actually double-counted: `tests.jsonl` went 134 -> 155 lines
#: and 20 rows appeared twice, so `_legs` returned 214 entries over 194 distinct labels.
#: `_epsfix` and `_epsfix2` are both wrong for this reason and are kept only as the record
#: of it. `_epsfix3` is the cut this default names: the 21 `test_simulation.py` rows were
#: dropped from the copied `tests.jsonl` BEFORE the resume, so the module re-ran whole and
#: appended exactly one row per case -- 194 entries, 194 distinct, no duplicates.
#:
#: `_2026-09-19_singles` REPLACES it as the default, cut fresh (no resume, no copy) over
#: the same 194-row basis after the singles round moved `arms.py` and `registry.py` and
#: the allpaths round moved the rest of the subject: 60 examples + 134 tests, 179
#: measured, the same 15 parameterised rows recovered by `match_param_rows.py` (24 of 25,
#: as `_epsfix3`). The corpus is MEEP 1.33.0 WITH its sibling `tests/` directory, pulled
#: to a path outside `/tmp`. `_epsfix3` stays as the provenance of the 348 / 597 board.
#: `_2026-09-19_overhead` REPLACES `_singles` the same day, cut by the same driver on
#: the same basis (60 + 134 rows, 179 measured, 24 of 25 param rows matched) after the
#: overhead round moved `cuda_kernels/fused_pairs.py` inside the subject.
#: `_2026-09-19_adeparse` REPLACES `_overhead` the same day, same driver and basis (60 +
#: 134 rows, 179 measured, 24 of 25 param rows matched), after the certified-ADE reader
#: in `cuda_kernels/fused_polarization_pair.py` stopped re-parsing `ade_kernels.py` per
#: emission and `cuda_kernels/test_ade_strings_parse_cache.py` joined the subject. The
#: board it cuts (`fusion_matrix_cuda_2026-09-19_adeparse`) differs from `_overhead`'s
#: only in `census` and `subject_pin`: 352 / 597 in dispatch, 532 served.
#: THE DEFAULT MOVED AGAIN 2026-09-23, to ``cuda_predicate_coverage_2026-09-23_sparse``:
#: the sparse-transport pass moved `stepping.py` and the two CUDA no-PML modules, seven
#: of the 127 subjects, so the 09-19 census no longer hashes to this tree. Same corpus,
#: same 194 rows / 179 measured / 15 unmeasured / 0 duplicate labels; the board it cuts
#: (`fusion_matrix_cuda_2026-09-23_sparse_b`) reads 352 / 597 in dispatch, 532 served
#: -- unchanged, which is the expected reading of a change that moves transport and
#: not what dispatches.
#: THE DEFAULT MOVED AGAIN 2026-09-26, to ``cuda_predicate_coverage_2026-09-25_roundb``,
#: the census Round B was certified on. Against ``_2026-09-23_sparse`` 89 of its 127
#: subjects changed and four were added (`cuda_kernels/own_cell_hoist.py` and the three
#: hoist tests beside it), so this tree's 131 subject files hash to ``239fd3ed...`` and
#: :func:`subject_pin` refused the old default by name. ``_2026-09-24_hoist`` was never
#: the default -- its board was cut through MEEP_GPU_CUDA_CENSUS, as Round B's was --
#: and no longer hashes to this tree either: 88 of its 127 subjects changed, the same
#: four were added. The battery digest moved with it (``750bc6e4...`` ->
#: ``71fb17c5...``) on line-number citations in its prose only. Same corpus and basis:
#: 60 + 134 rows, 179 measured, the 15 unmeasured rows recovered by 24 of 25 matched
#: parameterised rows, 194 distinct labels; the board it cuts
#: (`fusion_matrix_cuda_2026-09-25_roundb`) reads 352 / 597 in dispatch, 532 served --
#: unchanged from the ``_2026-09-23_sparse_b`` and ``_2026-09-24_hoist`` boards.
#: THE DEFAULT MOVED AGAIN 2026-10-06, to ``cuda_predicate_coverage_2026-10-06_092``, the
#: first CUDA census cut on release bytes. Against ``_2026-09-25_roundb`` 60 of its 131 subjects
#: changed and one was added (`cuda_kernels/test_single_arm_launches.py`): the 2026-09-27
#: composer edits (`arms.py`, `registry.py`: the mirror-fill family on `fill_B`/`fill_D` and
#: the off-diagonal `update_E` launchers), the neutral-wording pass, and test edits since,
#: `test_complex_no_pml.py`'s release-host rule among them, so this tree's 132 subject files
#: hash to ``9945e4ff...`` and :func:`subject_pin` refused the old default by name. The
#: battery digest moved with it (``71fb17c5...`` -> ``ac59bd40...``). Same corpus and basis:
#: 60 + 134 rows, 179 measured, the 15 unmeasured rows recovered by 24 of 25 matched
#: parameterised rows, 194 distinct labels. Row for row against ``_2026-09-25_roundb``:
#: every predicate answer and every selection on the slots both carry is unchanged; 79 of
#: the 194 rows gain `fill_B` and `fill_D`, selected to the mirror-fill family. The board it
#: cuts (`fusion_matrix_cuda_2026-10-06_092`) reads 352 / 597 in dispatch, 532 served, 29
#: products bound and 3 withheld -- unchanged from the ``_2026-09-25_roundb`` board.
CENSUS = RESULTS / (os.environ.get("MEEP_GPU_CUDA_CENSUS")
                    or "cuda_predicate_coverage_2026-10-06_092")

#: The composer modules the cells come from. Their PRESENCE is what makes a selection
#: readable, so it is asserted rather than assumed: without them the census's
#: ``plan_step.selected`` could not have been produced by this tree, and the board
#: would be reporting a shape nothing composes. The old direction of this check
#: (refuse if they APPEAR) is what stopped the predicate walk on 2026-08-28.
COMPOSER_MODULES = ("arms.py", "registry.py")

#: The composer's own arm table, read by ``ast`` rather than imported — ``registry``
#: imports every family module and several of those import the device toolchain,
#: which is not installed on the authoring host. Used ONLY as a floor: the census's
#: recorded ``selected_kernel`` is cross-checked against it, so a census cut against a
#: different registry cannot be read as if it described this one.
REGISTRY = CUDA_KERNELS / "registry.py"


# ---------------------------------------------------------------------------
# THE RECORD
# ---------------------------------------------------------------------------

def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _legs(directory: Path) -> List[dict]:
    record = load(directory / "examples.jsonl") + load(directory / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load(directory / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def rows() -> Tuple[List[dict], Dict[str, dict]]:
    """The CUDA census rows, and the engine facts read OUT OF THE SAME ROWS.

    NO JOIN. Until 2026-08-28 ``source_field_types`` and ``plan_step.live`` were
    joined out of ``metal_coverage_tranche6_2026-08-19`` because the CUDA census
    recorded neither. The promoted battery records both, so the join is gone — and
    the removal is CHECKED here rather than assumed: a row missing either fact, or
    missing the composer column, stops the run. Reading a missing
    ``source_field_types`` as an empty tuple would clear the B->H source-seam clause
    for that row and inflate the ceiling; reading a missing ``live`` as empty would
    report a fused launch with nothing to carry. Both errors point the same
    optimistic way, which is why neither is allowed a default.
    """
    cuda = _legs(CENSUS)
    facts: Dict[str, dict] = {}
    no_sources: List[str] = []
    no_live: List[str] = []
    no_selection: List[str] = []
    for row in cuda:
        label = f"{row['leg']}:{row['row']}"
        configuration = row.get("configuration") or {}
        plan = row.get("plan_step") or {}
        if "source_field_types" not in configuration:
            no_sources.append(label)
        if not plan.get("live_resolved") or plan.get("live") is None:
            no_live.append(label)
        if plan.get("selected") is None or plan.get("selected_family") is None \
                or plan.get("selected_kernel") is None:
            no_selection.append(label)
        facts[label] = {
            "source_field_types": tuple(
                str(k) for k in (configuration.get("source_field_types") or ())),
            "live_driver_passes": tuple(plan.get("live") or ()),
        }
    if no_selection:
        raise SystemExit(
            f"{len(no_selection)} of {len(cuda)} rows in {CENSUS.name} carry no "
            f"composer column (first: {no_selection[:3]}). A cell on this board IS "
            f"the arm the composer selects; without plan_step.selected / "
            f"selected_family / selected_kernel there is nothing to report. Re-cut "
            f"the census with a battery that asks arms.plan_step.")
    if no_sources or no_live:
        raise SystemExit(
            f"{CENSUS.name} is missing engine facts this board no longer joins from "
            f"the Metal census: {len(no_sources)} rows without "
            f"configuration.source_field_types (first: {no_sources[:3]}), "
            f"{len(no_live)} without a resolved plan_step.live (first: "
            f"{no_live[:3]}). Both defaults are optimistic — a source-free row "
            f"clears the source seam and a pass-free row carries nothing — so the "
            f"run stops instead of taking them.")
    return cuda, facts


def subject_pin(record: List[dict]) -> Dict[str, Any]:
    """The census's subject digest, RE-COMPUTED against the live tree.

    WHY A BOARD NEEDS THIS AND THE CENSUS DID NOT. The census measured predicates
    and asked the composer, both against the tree it ran on, and recorded
    ``subject_manifest_sha256`` so its own numbers could be pinned to it. This board
    mixes two clocks: the CELLS come from that recorded selection, while the FITNESS
    verdict is measured from device code read out of the WORKING TREE right now. If
    the package moved between them the board would pair a stale selection with a
    fresh read-pattern and report the difference as a fact. The manifest is the same
    one ``measure_predicate_coverage.subject_digest`` builds — sha256 over sorted
    ``"<relpath>  <sha256>"`` lines — so it moves if a file changes, is added, or is
    removed.
    """
    import hashlib  # noqa: PLC0415 - local: nothing else in this board hashes

    entries = {}
    for path in sorted(REPO_ROOT.glob(SUBJECT_GLOB)):
        entries[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    for name in SUBJECT_EXTRA:
        path = REPO_ROOT / name
        if path.exists():
            entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = "".join(f"{name}  {digest}\n"
                       for name, digest in sorted(entries.items()))
    live = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    recorded = sorted({row.get("subject_manifest_sha256") for row in record})
    if recorded != [live]:
        raise SystemExit(
            f"the census subject and the working tree disagree. The census recorded "
            f"{recorded} over {len(entries)} subject files; this tree hashes to "
            f"{live}. The cells below come from the recorded selection and the "
            f"POINTWISE/STENCIL verdict is measured from THIS tree's device code — "
            f"pairing them would report the drift as a finding. Re-cut the census, "
            f"or check out the tree it was cut against.")
    return {"manifest_sha256": live, "subject_files": len(entries),
            "scope": [SUBJECT_GLOB, *SUBJECT_EXTRA],
            "agrees_with_census": True}


#: The subject, spelled exactly as ``measure_predicate_coverage.py`` spells it. Kept
#: as two constants rather than imported because the census harness is COPIED into
#: each tranche directory and importing one tranche's copy would pin this board to
#: that tranche.
#: The census columns carrying each SHIPPED fused product's own per-row verdict.
#: THREE SINCE 2026-08-30, and it became a tuple rather than a widened string when
#: the second and third landed: a board that read one column and reported it as "what
#: the shipped products serve" would report the two complex products' cells as
#: unserved forever, which is a wrong number about a cell somebody built.
#:
#: EACH IS ITS OWN PREDICATE'S ANSWER, asked by the census on every row and counted
#: here -- never a restatement of its clauses, which could admit a row the launch
#: cannot serve. A census that could not ask one is a REFUSAL here, not a zero.
#:
#: FOUR SINCE 2026-08-30, and the fourth is the first on a DIFFERENT SEAM. The three
#: magnetic products span ``["step_B", "update_H"]``; ``cuda_fused_electric_pair``
#: spans ``["step_D", "update_E"]``, and ``fused_seam_of`` reads that off each
#: column's own recorded ``spans`` -- which is why adding it needs no change there.
#:
#: SIX SINCE 2026-08-31, and the two new ones complete the electric seam's three-way
#: partition: the real pair takes float32 storage, ``cuda_complex_fused_electric_pair``
#: takes complex64 on a Cartesian grid and ``cuda_cylindrical_fused_electric_pair``
#: takes complex64 on a Dcyl grid at |m| >= 1. The overlap check below walks every
#: (row, seam) pair and RAISES on a double admission, so the partition is measured on
#: this census rather than inherited from the predicates' prose.
#: THE SEVENTH COLUMN LANDED 2026-08-31 with the complex NO-ABSORBER electric weld,
#: and it is the first product on this board that refuses an absorber rather than
#: requiring one -- which is what makes ``step_D`` a FOUR-candidate seam without
#: making any configuration ambiguous. The overlap check below walks every (row, seam)
#: pair and RAISES on a double admission, so that partition is measured on this census
#: rather than inherited from the four predicates' prose.
#: NINE SINCE 2026-09-01, and the two new ones are the first on a THIRD seam:
#: ``cuda_fused_polarization_pair`` and its no-absorber twin span
#: ``["update_E", "update_P"]``, which ``fused_seam_of`` reads off each column's
#: own recorded ``spans`` exactly as it did when the electric seam arrived. The
#: E->P seam has no ``SOURCE_SEAM_CLAUSE`` entry -- the driver injects nothing
#: between its two consults -- so these columns move ``served_by`` and nothing
#: about the ceiling. The two partition on the absorber boolean alone, and the
#: overlap check below measures that on this census rather than trusting it.
#: FOURTEEN SINCE 2026-09-02: the five RESIDUAL magnetic welds landed -- the
#: board's last buildable cells (folded-complex, complex-beta, Dcyl real,
#: special_kz, BFAST, all on the B_to_H seam). ``step_B`` becomes an
#: EIGHT-candidate seam and the overlap check below is what measures the
#: partition on this census rather than trusting the eight predicates' prose.
FUSED_PRODUCT_COLUMNS: Tuple[str, ...] = (
    "cuda_fused_magnetic_pair",
    "cuda_complex_fused_magnetic_pair",
    "cuda_cylindrical_fused_magnetic_pair",
    "cuda_fused_electric_pair",
    "cuda_complex_fused_electric_pair",
    "cuda_cylindrical_fused_electric_pair",
    "cuda_no_pml_complex_fused_electric_pair",
    "cuda_fused_polarization_pair",
    "cuda_no_pml_fused_polarization_pair",
    "cuda_complex_folded_fused_magnetic_pair",
    "cuda_complex_beta_fused_magnetic_pair",
    "cuda_cylindrical_real_fused_magnetic_pair",
    "cuda_special_kz_fused_magnetic_pair",
    "cuda_bfast_fused_magnetic_pair",
    # TWENTY SINCE 2026-09-02 (the residue round): the five ELECTRIC TWINS of
    # the residual magnetic welds -- `step_D` becomes a NINE-candidate seam,
    # partitioned the way `step_B`'s eight already are plus the absorber
    # boolean -- and the THIRD E->P product, the complex no-absorber
    # polarization pair, disjoint from the two real ones on storage alone.
    # The overlap check below measures both partitions on this census rather
    # than trusting the predicates' prose.
    "cuda_complex_folded_fused_electric_pair",
    "cuda_complex_beta_fused_electric_pair",
    "cuda_cylindrical_real_fused_electric_pair",
    "cuda_special_kz_fused_electric_pair",
    "cuda_bfast_fused_electric_pair",
    "cuda_complex_no_pml_fused_polarization_pair",
    # TWENTY-TWO SINCE 2026-09-02 (the stencil round): the two SCRATCH-OUTPUT
    # welds on the two off-diagonal D_to_E cells this board had recorded
    # STENCIL-BLOCKED. They are the first products here whose cells carried an
    # UNBUILDABLE verdict rather than merely no product, and the verdict is not
    # relaxed -- it was a measurement of the IN-PLACE weld, and these two write a
    # launch-local scratch, recompute every foreign sample from pre-launch state
    # and rotate the D bindings afterwards. The overlap check below measures
    # their partition against the other nine step_D candidates on this census.
    "cuda_offdiag_fused_electric_pair",
    "cuda_folded_offdiag_fused_electric_pair",
    # TWENTY-THREE: the off-diagonal E->P weld, which takes the last unserved
    # cell on that seam. The board recorded it POINTWISE-BUILDABLE with a
    # ROTATION HAZARD; the hazard was of the per-component split, and this
    # product keeps update_E monolithic so no rotation happens inside the launch.
    "cuda_dispersive_offdiag_fused_polarization_pair",
    # TWENTY-FOUR: the DISPERSIVE electric weld, which takes the largest cell any
    # backend had left unoccupied -- `D_to_E (cuda_curl/PML,
    # cuda_dispersive/dispersive)`, 7 corpus rows, POINTWISE-BUILDABLE and empty. It
    # makes `step_D` a TEN-candidate seam, and the partition against the nine already
    # there is one measured boolean: the ordinary constitutive predicate refuses
    # `fields.polarizations` truthy and the dispersive one requires it. The overlap
    # check below measures it on this census rather than trusting that prose -- and
    # an overlap here would not be a duplicate count, it would leave the seam
    # UNFUSED and cost the 79 rows `cuda_fused_electric_pair` serves.
    "cuda_dispersive_fused_electric_pair",
    # TWENTY-FIVE AND TWENTY-SIX: the two CONDUCTIVE-CURL electric welds, which take
    # the last three POINTWISE-BUILDABLE `D_to_E` seam-instances this board had. The
    # first occupies TWO cells with one product -- `conductive_kernels` bakes the
    # per-component conductivity into three COND defines and the (0,0,0) build IS the
    # certified lossless curl -- and declares the second in
    # `fused_pairs.FUSED_PAIR_EXTRA_ARMS`. The second is the SIGNED-ZERO row, priced
    # here for the first time because the driver's condinv injection now replays its
    # rescale sparsely at the published deposit indices instead of over three whole
    # volumes; before that change no bracket could have made the cell byte-identical
    # and the cell was correctly worth zero.
    #
    # They make `step_D` a TWELVE-candidate seam. The partition against the ten
    # already there is measured by the overlap check below, not trusted to this
    # prose: the no-absorber one is the only electric product whose constitutive half
    # REQUIRES an inert layer, and the conductive x ordinary one is separated from the
    # real twin by the conductivity clause (`covers_real_pml_curl` refuses a target
    # carrying a sigma BY NAME) and from the dispersive one by `fields.polarizations`.
    "cuda_no_pml_dispersive_fused_electric_pair",
    "cuda_conductive_fused_electric_pair",
    # TWENTY-NINE AND THIRTY, 2026-09-02: the two THREE-SLOT welds, and the first
    # columns on this board whose `spans` names three slots. They are the only
    # products that occupy BOTH sides of the shared `update_E` slot -- which is
    # exactly the collision `_loses_the_shared_slot` was added to model, and the
    # ten rows it withholds are the ten these serve at both seams:
    #
    #   cuda_three_slot_dispersive_weld           7 rows (the stochastic_emitter
    #                                             and TestLoadDump.*_2d cells)
    #   cuda_three_slot_no_pml_dispersive_weld    3 rows (absorber-1d,
    #                                             TestAbsorber, material-dispersion)
    #
    # THEY DO NOT COLLIDE WITH THE D->E PRODUCTS THEY SUPERSEDE, and that is read
    # from the composer rather than decided here: each weld's predicate is a
    # CONJUNCTION of a D->E predicate and an E->P one, so its admission set is a
    # subset of each, and `fused_pairs._superseded_by_a_longer_span` refuses the
    # shorter span by name. `_serving_products` below applies the same rule to the
    # same declared spans, so the overlap check measures what the composer installs.
    "cuda_three_slot_dispersive_weld",
    "cuda_three_slot_no_pml_dispersive_weld",
    # THIRTY-ONE, 2026-09-04: the COMPLEX no-absorber THREE-SLOT weld, over the
    # four `TestLoadDump.*_3d` rows the two real three-slot welds' record named as
    # a different shape (magnetic source only, complex64 storage, no bracket, a
    # register hand-off across the E->P seam). It is the third column whose
    # `spans` names three slots, and it takes the board's last four
    # `buildable_not_built` D_to_E instances the way the real welds took their ten:
    # one product on both sides of the shared slot, refused nowhere by
    # `_loses_the_shared_slot` because the same column serves both seams.
    "cuda_three_slot_complex_no_pml_dispersive_weld",
    # TWENTY-SEVEN AND TWENTY-EIGHT: the two COMPLEX SCRATCH-OUTPUT welds, which take
    # the LAST two cells this board recorded UNBUILDABLE -- `D_to_E
    # (cuda_complex_folded, cuda_complex_offdiag/complex off-diagonal PML)` and
    # `D_to_E (cuda_complex_no_pml, cuda_complex_offdiag/complex off-diagonal
    # no-PML)`. The STENCIL verdict is NOT relaxed and `fitness_rule` is NOT edited:
    # STENCIL stays the true reading of both constitutive kernels' device text. What
    # changes is that a product now occupies each cell whose weld does not require
    # those reads to come from the curl half's in-place output.
    #
    # They make `step_D` a FOURTEEN-candidate seam, and the partition against the
    # twelve already there is DOUBLE rather than single: every one of those either
    # refuses complex64 storage BY NAME or refuses an off-diagonal chi1inv row BY
    # NAME, and these two require both. Between themselves they split on the
    # absorber. The overlap check below measures all of it on this census.
    "cuda_folded_complex_offdiag_fused_electric_pair",
    "cuda_complex_no_pml_offdiag_fused_electric_pair",
    # THE FIRST COLUMN ON THE FOURTH SEAM, 2026-09-06. ``cuda_fused_hd_pair`` spans
    # ``update_H`` -> ``step_D`` -- constitutive then curl, the other way round from
    # every column above -- so ``_seams_of_span`` resolves it to ``H_to_D`` through
    # :data:`SPANNABLE_SEAM_SLOTS` and it is the column that lets this board answer
    # ``served_by`` on the seam ``h_to_d_seam.price`` already priced. It makes NO seam
    # ambiguous: it is the only product on ``update_H`` -> ``step_D`` and every column
    # above spans a different pair, so the overlap check below has nothing to resolve
    # there and measures that rather than assuming it.
    #
    # CREDITING IT IS NOT SAYING IT RUNS. Its module declares INSTALLABLE = False and
    # ``fused_pairs._declared_uninstallable`` refuses it on every configuration; the
    # H_to_D block records that declaration beside the count, because SERVED on this
    # board is PREDICATE ADMISSION and a reader is owed both facts in one place.
    "cuda_fused_hd_pair",
    # THE SECOND AND THIRD COLUMNS ON THE FOURTH SEAM, 2026-09-07: the two
    # cylindrical H->D products, spanning ``update_H`` -> ``step_D`` on the two
    # cells the board had recorded buildable_not_built (3 real m = 0 rows, 16
    # complex rows) plus the 2 withdraw rows they refuse by name. ``update_H``
    # becomes a three-candidate seam partitioned on the Dcyl grid and on storage;
    # the overlap check below measures that on the census rather than assuming it.
    # CREDITING THEM IS NOT SAYING THEY RUN: both declare INSTALLABLE = False, and
    # the H_to_D block records the declaration beside the count.
    "cuda_cylindrical_real_fused_hd_pair",
    "cuda_cylindrical_fused_hd_pair",
    # THE FOURTH COLUMN ON THE FOURTH SEAM, 2026-09-07: the CARTESIAN COMPLEX H->D
    # product, over the two cells this board had left ``buildable_not_built`` there
    # -- ``H_to_D (cuda_complex/complex, cuda_complex/complex)`` and ``H_to_D
    # (cuda_complex/complex, cuda_complex_folded/folded complex)``. ONE COLUMN COVERS
    # BOTH, and that is read off the predicate rather than declared here: it reads the
    # fold off the grid and asks the certified curl predicate its VARIANT selects
    # (``covers_real_pml_complex_curl`` unfolded, ``complex_folded_kernels.
    # covers_complex_folded_curl`` folded), so the two cells resolve through the
    # composer's own selection at ``step_D`` and never through a second column.
    # ``update_H`` becomes a FOUR-candidate seam, partitioned on storage and on the
    # Dcyl grid; the overlap check below measures that on this census rather than
    # assuming it. CREDITING IT IS NOT SAYING IT RUNS: the module declares
    # INSTALLABLE = False and the H_to_D block records the declaration beside the
    # count.
    "cuda_complex_fused_hd_pair",
    # THE FIFTH COLUMN ON THE FOURTH SEAM, 2026-09-07, and the one that takes this
    # board's buildable_not_built H->D bucket to zero outside the beta cluster:
    # the conductive and BFAST curl tails, ONE product over the two cells
    # H_to_D (cuda_constitutive/ordinary, cuda_conductive/conductive) and
    # H_to_D (cuda_bfast/BFAST, cuda_bfast/BFAST) -- 1 seam-instance each,
    # tests:TestAdjointSolver.test_damping and
    # tests:TestReflectanceAngular.test_reflectance_angular_2_35_7.
    #
    # ONE COLUMN COVERS BOTH, read off the predicate rather than declared here: it
    # resolves the VARIANT from the run (grid.bfast_active and fields.condfac_for,
    # the two readers the array path itself uses) and then asks the pair of certified
    # predicates that variant selects, so the two cells resolve through the composer's
    # own selection at step_D and never through a second column. update_H
    # becomes a FIVE-candidate seam; the partition against the four columns above is
    # on the conductivity, on BFAST and on the Dcyl grid together, and the overlap
    # check below measures it on this census rather than assuming it.
    #
    # CREDITING IT IS NOT SAYING IT RUNS: the module declares INSTALLABLE = False and
    # the H_to_D block records the declaration beside the count.
    "cuda_conductive_bfast_fused_hd_pair",
    # THIRTY-SEVEN AND THIRTY-EIGHT, 2026-09-08: the two BETA H->D welds, which take
    # the last six seam-instances this board carried as buildable_not_built. Both
    # released on 2026-09-07 under both policies with every corpus row of their cells
    # driven bit-identical (4 of 4 and 2 of 2); they were uncounted only because
    # neither this tuple nor cuda_predicate_battery.py named them.
    "cuda_complex_beta_fused_hd_pair",
    "cuda_special_kz_fused_hd_pair",
)

SUBJECT_GLOB = "meep_gpu/cuda_kernels/*.py"
SUBJECT_EXTRA = ("meep_gpu/stepping.py", "meep_gpu/fields.py",
                 "meep_gpu/dispersion.py", "meep_gpu/pml.py", "meep_gpu/grid.py")


def selection(row: dict, slot: str) -> Optional[dict]:
    """THE CELL: the arm the composer selected at ``slot``, or ``None`` if it refused.

    THE COMPOSER WAS ASKED BEHIND THE CENSUS'S CuPy-BACKEND SHIM, AND THAT IS NOT A
    CHOICE. Every shipped hand-CUDA predicate opens with a clause refusing a non-CuPy
    array module and the census ran on a NumPy host, so a composer asked with the real
    grid selects almost nothing — the census records the real-grid composition beside
    this one under ``plan_step.unfactored`` and it is 740 of 759 slots refused on the
    backend clause alone. The shimmed answer is what the composer would return on a
    CuPy host with the same lifted object, and the census's per-row
    ``plan_step.shim`` probe is what licenses it: it compares grid reads
    proxy-vs-real and records ``sound`` on every row.

    ``None`` is a REFUSAL, never a gap to be filled — the composer's own reasons for
    it are carried through to the cell from ``plan_step.refusals``.
    """
    plan = row.get("plan_step") or {}
    family = (plan.get("selected_family") or {}).get(slot)
    if family is None:
        return None
    return {"family": family,
            "arm": (plan.get("selected") or {}).get(slot),
            "kernel": (plan.get("selected_kernel") or {}).get(slot),
            "launchable": slot in (plan.get("launchable") or ())}


def refusal(row: dict, slot: str) -> dict:
    """The composer's own refusal at an unselected slot, quoted rather than described."""
    plan = row.get("plan_step") or {}
    entry = ((plan.get("refusals") or {}).get(slot)) or {}
    return {"kind": entry.get("kind", "unreported"),
            "admitters": list(entry.get("admitters") or ()),
            "reasons": list(entry.get("reasons") or ())}


# ---------------------------------------------------------------------------
# THE SEAMS — DERIVED FROM driver.py, NOT TRANSCRIBED
# ---------------------------------------------------------------------------

#: Each seam: name, the curl call this board looks for, the constitutive call that
#: closes it, and the driver calls that sit BETWEEN them. The line numbers are found
#: in the file; only the ORDER is spelled here, because the order is the physics.
SEAM_SHAPE: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("B_to_H", "step_B", "update_H",
     ("magnetic source.inject", "fill_symmetry_bc_B", "zero_metal_B",
      "fill_folded_far_ghosts_B")),
    ("D_to_E", "step_D", "update_E",
     ("electric source.inject", "fill_symmetry_bc_D", "zero_metal_D",
      "fill_folded_far_ghosts_D")),
    ("E_to_P", "update_E", "update_P", ()),
)

#: The in-seam driver passes, in ``plan_step.live``'s spelling — the CUDA census's own
#: names for the same three driver calls, which the battery took from the Metal
#: census's vocabulary when it grew the column. ``E_to_P`` has none — that is why it
#: is absent rather than empty.
IN_SEAM_PASSES: Dict[str, Tuple[str, str, str]] = {
    "B_to_H": ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B"),
    "D_to_E": ("fill_D", "zero_metal_D", "fill_folded_far_ghosts_D"),
}

#: Which CUDA in-seam kernel each of those passes would need. ``in_seam_passes.py``
#: ships six, and the pass names differ from the driver's — mapped here once.
IN_SEAM_CUDA_KERNEL: Dict[str, str] = {
    "fill_B": "fill_symmetry_B", "zero_metal_B": "zero_metal_B",
    "fill_folded_far_ghosts_B": "fill_folded_far_B",
    "fill_D": "fill_symmetry_D", "zero_metal_D": "zero_metal_D",
    "fill_folded_far_ghosts_D": "fill_folded_far_D",
}


def locate_the_seams() -> Dict[str, dict]:
    """Find each seam's call sites in ``driver.py`` and report the span.

    DERIVED, BECAUSE THE TRANSCRIPTION IS ALREADY STALE. The Metal board spells
    ``driver.py:3281-3306`` and cites ``:3283-3284`` for the magnetic injection.
    Against this tree those numbers are nine lines short, because the integrated
    sources' withdraw loop landed between them. A board whose central structural
    claim is a line number nobody re-checks is a board that will eventually describe
    a timestep the driver no longer runs.
    """
    source = DRIVER.read_text()
    text = source.splitlines()

    # THE TIMESTEP IS FOUND BY ast, NOT BY A LINE GUESS, AND THAT MATTERS: the driver
    # carries TWO dispatch-guarded step_B call sites. One is the timestep (`step`,
    # the method the run loop calls); the other is `synchronize_magnetic_fields`,
    # a half-step the driver runs to co-locate H with E for a monitor. Their seams
    # differ — the synchronizing copy carries no electric injection — so measuring
    # the wrong one would move the D->E ceiling without changing any other number,
    # which is precisely the kind of error a board reports confidently.
    tree = ast.parse(source)
    spans = {node.name: (node.lineno, node.end_lineno or node.lineno)
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "step" not in spans:
        raise SystemExit("driver.py declares no function named `step`; this board "
                         "locates the timestep by that name and cannot proceed")
    start, end = spans["step"]

    def find_all(needle: str) -> List[int]:
        return [i + 1 for i, line in enumerate(text)
                if needle in line and start <= i + 1 <= end]

    guards = find_all('fast.dispatch("step_B"')
    if len(guards) != 1:
        raise SystemExit(
            f"driver.py's `step` (lines {start}-{end}) has {len(guards)} "
            f"dispatch-guarded step_B call sites {guards}; this board locates the "
            f"seams inside a single timestep and cannot choose between two.")
    ordered: Dict[str, dict] = {}
    for seam, curl, const, between in SEAM_SHAPE:
        curl_lines = [n for n in find_all(f'{curl}(self.fields') if n >= start]
        const_lines = [n for n in find_all(f'{const}(self.fields') if n >= start]
        if not curl_lines or not const_lines:
            raise SystemExit(
                f"{seam}: could not locate {curl}/{const} in driver.py at or after "
                f"line {start}; the timestep's shape has changed and this board's "
                f"seam definition no longer describes it")
        curl_at, const_at = curl_lines[0], const_lines[0]
        inside = []
        for name in between:
            if name.endswith("source.inject"):
                hits = [n for n in find_all("source.inject(self.fields")
                        if curl_at < n < const_at]
                hits += [n for n in find_all("_inject_electric_through_conductivity")
                         if curl_at < n < const_at]
            else:
                hits = [n for n in find_all(f"{name}(self.fields")
                        if curl_at < n < const_at]
            if not hits:
                raise SystemExit(
                    f"{seam}: {name} is named in this board's seam shape but no call "
                    f"site sits between driver.py:{curl_at} and :{const_at}. Either "
                    f"the pass moved out of the seam — in which case this board is "
                    f"over-counting what a fused launch would have to absorb — or "
                    f"it was renamed. Neither may pass silently.")
            inside.append((name, sorted(hits)))
        ordered[seam] = {
            "curl_call": f"driver.py:{curl_at}", "constitutive_call":
                f"driver.py:{const_at}",
            "between": [{"pass": name, "driver_lines": lines} for name, lines in inside],
            "between_summary": (" | ".join(
                f"{name} {','.join(str(x) for x in lines)}" for name, lines in inside)
                or "nothing"),
        }
    return ordered


# ---------------------------------------------------------------------------
# THE CUDA SURFACE — READ FROM THE MODULES BY ast, NEVER IMPORTED
# ---------------------------------------------------------------------------
# cupy is not installed on the authoring host and importing these modules would
# fail; more to the point, a board that IMPORTS the thing it audits can be moved by
# an import side effect. Everything below is a parse.

def module_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target == name and value is not None:
            try:
                return ast.literal_eval(value)
            except Exception:
                return "<not a literal>"
    return None


def cuda_surface() -> Dict[str, Any]:
    """Every ``CERTIFIED_KERNELS`` / ``UNCERTIFIED_KERNELS`` in the package, live."""
    certified: Dict[str, str] = {}
    uncertified: Dict[str, str] = {}
    per_module: Dict[str, dict] = {}
    for path in sorted(CUDA_KERNELS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        cert = module_constant(path, "CERTIFIED_KERNELS")
        uncert = module_constant(path, "UNCERTIFIED_KERNELS")
        if cert is None and uncert is None:
            continue
        cert_names = list(cert or ())
        uncert_names = ([] if not uncert or isinstance(uncert, str)
                        else list(uncert))
        per_module[path.name] = {"certified": cert_names,
                                 "uncertified": uncert_names}
        for kernel in cert_names:
            certified[kernel] = path.name
        for kernel in uncert_names:
            uncertified[kernel] = path.name
    overlap = sorted(set(certified) & set(uncertified))
    if overlap:
        raise SystemExit(
            f"{overlap} appear in BOTH CERTIFIED_KERNELS and UNCERTIFIED_KERNELS. "
            f"The two sets are a partition by construction and a name in both makes "
            f"every count below ambiguous.")
    return {"certified": certified, "uncertified": uncertified,
            "per_module": per_module}


def composer_layer() -> Dict[str, Any]:
    """The composer must be PRESENT — this board reads what it selected.

    THE CHECK RUNS THE OTHER WAY NOW. Until 2026-08-28 this function refused if any
    composer module APPEARED, because a cell was then the family whose predicate
    admitted the row and a composer arriving would have made that walk wrong. It
    arrived, the board refused, and this is the re-point: the cell is the arm, so the
    refusal is now for ABSENCE. Without ``arms.py`` and ``registry.py`` in this tree,
    the ``plan_step.selected`` the census recorded could not have been produced by the
    package the fitness verdict is measured from, and the two halves of every cell
    would describe different engines.
    """
    missing = sorted(name for name in COMPOSER_MODULES
                     if not (CUDA_KERNELS / name).exists())
    if missing:
        raise SystemExit(
            f"cuda_kernels/ is missing {missing}. A cell on this board is the ARM "
            f"THE COMPOSER SELECTS, read from the census's plan_step.selected; with "
            f"no composer in the tree that column describes a package this one is "
            f"not, and the POINTWISE/STENCIL verdict measured from this tree's "
            f"device code would be paired with it. Point --out at a tree that ships "
            f"the composer, or use a board that derives a cell from the predicates.")
    return {"required": list(COMPOSER_MODULES), "missing": missing,
            "consequence": ("a cell here is 'the arm the composer selects at this "
                            "slot', read from plan_step.selected/selected_family/"
                            "selected_kernel, not 'the family whose predicate "
                            "admits this row'")}


# ---------------------------------------------------------------------------
# THE FITNESS VERDICT — POINTWISE vs STENCIL, MEASURED FROM THE DEVICE CODE
# ---------------------------------------------------------------------------

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_CUDA_MARKERS = ("__global__", "__device__", "blockIdx")
_FLAT = re.compile(r"\b(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x")
_SUBSCRIPT = re.compile(r"([A-Za-z_]\w*)\s*\[\s*([^\[\]]{1,100}?)\s*\]")
_BARE = re.compile(r"^(?:[A-Za-z_]\w*|\d+)$")
_DEVICE_FN = re.compile(r"__device__[^\n]*?\b(\w+)\s*\(", re.S)


def _strip_comments(source: str) -> str:
    """C comments OUT before anything is measured.

    NOT COSMETIC. ``offdiag_emitter.py`` transcribes MEEP's own expression
    ``0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])`` into a COMMENT
    beside the code that implements it. A subscript scan that does not strip comments
    reports those four neighbour reads as device code — which happens to give the
    right verdict for that family and the WRONG evidence, and would give the wrong
    verdict for any family whose comment quotes an expression it does not implement.
    """
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", source))


def cuda_literals(path: Path) -> List[Tuple[int, str]]:
    """Every string constant in a module that carries CUDA device code.

    ONE-LINE LITERALS ARE EXCLUDED, and the exclusion is a measurement rather than a
    convenience: ``offdiag_emitter.py:449`` is the single line
    ``extern "C" __global__ void (\\w+)\\(`` — a PYTHON REGEX the module uses to read
    its own kernel names back out. It carries the ``__global__`` marker and is not
    device code. No kernel body in this package is one line.
    """
    tree = ast.parse(path.read_text())
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and any(marker in node.value for marker in _CUDA_MARKERS)
            and len(node.value.splitlines()) > 1]


def _balanced_body(source: str, at: int) -> str:
    """The ``{...}`` block starting at or after ``at``, brace-matched."""
    start = source.find("{", at)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return source[start:]


def _units(source: str, body: str) -> Dict[str, dict]:
    """The kernel body and every ``__device__`` helper, as callable units.

    A unit is ``{params, body}``. The kernel body is the unit named ``__kernel__``.
    Helpers are units named by their C function name. This is what makes the
    same-cell analysis TRANSITIVE, and the transitivity is not optional: the
    off-diagonal family's neighbour reads are three calls deep — the kernel body
    computes ``down``/``up``/``corner`` from ``coord_dn``/``coord_up``/``flat``,
    hands them to ``offdiag_term``, which hands them to ``ghosted``, which is the
    only place an array is actually subscripted with them. A one-level call-site
    check finds ``offdiag_term``'s ``g[home]``, sees ``home`` bound to the flat
    index, and pronounces the kernel POINTWISE. It was MEASURED doing exactly that
    on this corpus before this function existed.
    """
    units: Dict[str, dict] = {"__kernel__": {"params": [], "body": body}}
    for match in _DEVICE_FN.finditer(source):
        name = match.group(1)
        open_paren = source.find("(", match.start(1))
        close = _matching(source, open_paren)
        if open_paren < 0 or close is None:
            continue
        params = [p.strip() for p in _split_args(source[open_paren + 1:close])]
        names = [p.split()[-1].lstrip("*&") for p in params if p.split()]
        units[name] = {"params": names, "body": _balanced_body(source, close)}
    return units


def _matching(source: str, at: int) -> Optional[int]:
    """Index of the ``)`` closing the ``(`` at ``at``."""
    if at < 0 or at >= len(source) or source[at] != "(":
        return None
    depth = 0
    for index in range(at, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


#: ``K * safe`` or ``K * safe + c`` with integer literals — the INTERLEAVED-STORAGE
#: form, not a neighbour. See :func:`_interleaved`.
_INTERLEAVED = re.compile(r"^(\d+)\s*\*\s*([A-Za-z_]\w*)(?:\s*\+\s*(\d+))?$")

#: THE E->P SEAM COSTS MORE THAN ITS BUCKET SUGGESTS, and the board does not price it.
#: MEASURED 2026-08-31, on CUDA's own sources rather than inherited from the Metal board:
#: ``update_E`` runs "in one launch" for all three components
#: (cuda_kernels/constitutive_kernels.py:506), while ``update_P`` runs ONE LAUNCH PER
#: COMPONENT because its three buffers rotate per component INSIDE one call and "the
#: retired history becomes the NEXT component's scratch" (ade_kernels.py:414-424); the
#: launcher re-reads every slot after the previous component's rotation, and a launcher
#: that cached the views "would be stale from the second component of the first step,
#: and stale in a way that still computes".
#:
#: So a fused E->P is NOT structurally refused the way an off-diagonal stencil is -- it
#: is a SHAPE MISMATCH. Welding them means splitting update_E into three per-component
#: launches and fusing each with its update_P: 3 launches replacing 4. It saves exactly
#: ONE launch per step and gives up update_E's whole-grid form to do it.
#:
#: The three E->P cells (7 + 4 + 3 = 14 seam-instances) are therefore scored
#: POINTWISE-BUILDABLE here, which is arithmetically right and economically misleading.
#: Anyone taking them should know they are the worst value on the board: the most
#: restructuring for the least saving. Grade them BUILD, not HARD, and rank them last.
#:
#: ADDENDUM 2026-09-01: the restructuring was done under the completeness directive.
#: ``fused_polarization_pair.py`` welds the 7-instance dispersive x ADE cell and the
#: 3-instance no-PML sibling exactly as measured above -- three per-component
#: launches, host rotation preserved between them -- and its module docstring carries
#: the word-for-word driver-order equivalence argument. Two cells stay unserved: the
#: 4-instance complex no-PML cell (its update_P half is emitted under substitution
#: and stays UNDETERMINED on this board), and a FOURTH E->P cell the count above
#: missed -- 1 instance, ``cuda_dispersive_offdiag x cuda_ade``
#: (absorbed_power_density.py) -- which the per-component split REFUSES outright:
#: the off-diagonal row product reads the OTHER components' ``D - sum P`` at
#: neighbours, so the second component's launch would read a P slot the first
#: component's rotation had already advanced. The corrected census is 7 + 4 + 3 + 1
#: = 15 seam-instances across FOUR cells, of which the shipped products serve 10.


def _interleaved(expr: str, safe: set) -> Optional[Tuple[int, int, str]]:
    """Is ``expr`` a read inside the thread's own K-wide slot rather than a neighbour?

    ``complex_emitter.py``'s ``cf_load`` reads ``g[2*idx]`` and ``g[2*idx + 1]``:
    complex storage is INTERLEAVED, two float32 per cell, so thread ``idx`` owns
    words ``2*idx`` and ``2*idx+1`` and neither belongs to another thread. Refusing
    that as a neighbour would report every complex family as STENCIL-blocked, which
    is false and would be the WRONG direction of error — it would hide buildable
    work rather than invent it.

    The carve-out is narrow on purpose: literal stride, literal offset, offset
    strictly inside the stride. ``idx + sx`` (sx = ny*nz) does not match, and must
    not: that IS a neighbour.
    """
    match = _INTERLEAVED.match(expr)
    if not match:
        return None
    stride, base, offset = int(match.group(1)), match.group(2), int(match.group(3) or 0)
    if base not in safe or stride < 1 or not 0 <= offset < stride:
        return None
    return stride, offset, base


#: What an f-string interpolation collapses to. It is treated as a SAME-CELL name
#: everywhere below, and that is a stated limitation with a checked justification:
#: every interpolation in these emitters substitutes an IDENTIFIER — a volume name,
#: a coefficient-parameter name, or a per-axis index letter — never index
#: arithmetic. The one place an emitter builds an index, ``offdiag_emitter.
#: _flat_index``, emits the call ``flat(di, j, k, nyz, nz)`` in full, so the
#: neighbour SURVIVES as a call — except where the emitter builds the whole index
#: in Python and interpolates it whole, which ``offdiag_emitter._term_lines`` does:
#: ``f"        {down},"`` collapses to a single token and the four-corner read
#: becomes invisible to a subscript walk. That is why the subscript fixpoint is not
#: the only detector here: :func:`_shifted_coordinate_helpers` catches exactly that
#: case, from the coordinate arithmetic the helpers perform, which no interpolation
#: can hide. Left OPAQUE instead of same-cell, every ``kps_{axis}[{index}]``
#: coefficient read would be reported as a neighbour and the off-diagonal family
#: would earn the right verdict for entirely the wrong reason — MEASURED doing
#: exactly that before this token was made same-cell.
_FSTRING_TOKEN = "FSTRING_INTERPOLATION"


def _docstring_ids(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _flatten_fstring(node: ast.JoinedStr) -> str:
    """An f-string with every interpolation replaced by the opaque token ``X``.

    The interpolations in these emitters are volume NAMES and coefficient NAMES —
    ``f"    float term_{tag} = offdiag_term("`` — never index arithmetic. Replacing
    them with a token keeps the CALL STRUCTURE, which is what the read analysis
    needs, and loses only identifiers it would not have resolved anyway.
    """
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append(_FSTRING_TOKEN)
    return "".join(parts)


def device_text(module: str) -> Dict[str, Any]:
    """A module's device code, split into preludes, kernel bodies and fragments.

    THE THIRD BUCKET IS WHY THIS FUNCTION EXISTS. ``offdiag_emitter.py`` does not
    hold its kernel as one string: the template carries ``__SRC_Ex__`` placeholders
    and ``_component_source`` assembles the component bodies LINE BY LINE out of
    f-strings, so the only place ``offdiag_term(volume, coefficient, idx, flat(di,
    j, k, ...), ...)`` ever appears is a Python list comprehension. A collector that
    looks only for string constants carrying ``__global__`` sees the template, misses
    every neighbour read, and reports the off-diagonal family — the one family whose
    stencil is the whole reason this column exists — as POINTWISE. It was MEASURED
    doing exactly that before the fragment bucket was added.

    Docstrings are excluded by identity, not by heuristic: this module's own prose
    quotes MEEP expressions like ``g[i+s]`` and would otherwise be read as code.
    """
    path = CUDA_KERNELS / module
    tree = ast.parse(path.read_text())
    docstrings = _docstring_ids(tree)
    preludes: List[Tuple[int, str]] = []
    bodies: List[Tuple[int, str]] = []
    fragments: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            text, lineno = node.value, node.lineno
        elif isinstance(node, ast.JoinedStr):
            text, lineno = _flatten_fstring(node), node.lineno
        else:
            continue
        multiline = len(text.splitlines()) > 1
        marked = any(marker in text for marker in _CUDA_MARKERS)
        if marked and multiline and "__global__" in text:
            bodies.append((lineno, text))
        elif marked and multiline:
            preludes.append((lineno, text))
        elif ";" in text or re.search(r"\b\w+\s*\($", text.strip()):
            # A C STATEMENT FRAGMENT. The trailing-open-paren form is the first line
            # of a multi-line call the emitter splits across list entries, which is
            # how offdiag_term's four index arguments arrive.
            fragments.append((lineno, text))
    names = sorted({name for _lineno, text in bodies
                    for name in re.findall(
                        r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', text)})
    return {"preludes": preludes, "bodies": bodies, "fragments": fragments,
            "kernel_names": names}


#: Placeholders a template may still carry when its body is analysed ALONE, with the
#: reason each splices no array subscript. Anything else forces UNDETERMINED.
PLACEHOLDER_SPLICES_NO_SUBSCRIPT: Dict[str, str] = {
    "__NAME__": "the kernel's C name; substituted from a table, carries no code",
    "__EPS_PARAMS__": ("three read-only parameter declarations "
                       "(complex_emitter.py:535-536); a signature, not a body"),
}


#: ``return <parameter> + 1;`` / ``- 1;`` — a coordinate SHIFTED off this thread's.
_SHIFTED_RETURN = re.compile(r"\breturn\s+(\w+)\s*([+-])\s*(\d+)\s*;")


def _shifted_coordinate_helpers(units: Dict[str, dict],
                                reachable: set) -> List[dict]:
    """Reachable helpers that CONSTRUCT a coordinate off this thread's own.

    THE SECOND DETECTOR, AND ON ONE FAMILY IT IS THE ONLY ONE THAT WORKS. The
    subscript fixpoint follows an index expression to the array it subscripts; it
    cannot follow one that a PYTHON emitter built and interpolated whole.
    ``offdiag_emitter._term_lines`` does precisely that — ``f"        {down},"``,
    where ``down`` is the string ``flat(di, j, k, nyz, nz)`` computed in
    ``_flat_index`` — so the four-corner read reaches the fixpoint as an opaque
    token and the family measures POINTWISE. It DID, before this function existed.

    What no emitter can hide is the arithmetic itself. ``coord_up(int a, int n, int
    bc) { if (a + 1 < n) return a + 1; ... }`` and its ``a - 1`` twin are in the
    device prelude, in full, and a kernel that only ever touches the cell its thread
    owns has no reason to compute the coordinate next to it. A reachable helper that
    returns ``<parameter> +/- <literal>`` is therefore a neighbour ADDRESS
    computation, reported here with the return it was found on.

    This is a sufficient condition, not a necessary one: a family with no such
    helper is still put through the subscript fixpoint, and either detector firing
    is enough for STENCIL.
    """
    found: List[dict] = []
    for name in sorted(reachable):
        if name == "__kernel__":
            continue
        unit = units[name]
        for parameter, sign, literal in _SHIFTED_RETURN.findall(unit["body"]):
            if parameter not in unit["params"]:
                continue
            found.append({
                "helper": name, "parameter": parameter,
                "returns": f"{parameter} {sign} {literal}",
                "why": (f"{name}() returns its coordinate parameter {parameter!r} "
                        f"shifted by {sign}{literal}. A kernel that reads only the "
                        f"cell its thread owns never needs the coordinate beside "
                        f"it; this is a neighbour ADDRESS being built")})
    return found


#: THE DETECTOR'S OWN CONTROLS, run every time before any verdict is reported.
#: A read-pattern detector that silently degrades is worse than none: this board's
#: whole ranking is downstream of it, and a false POINTWISE invites work that cannot
#: be built byte-identically. Each control has an answer that is settled
#: independently of this script, and the run REFUSES if one does not reproduce.
DETECTOR_CONTROLS: Tuple[Tuple[str, str, str], ...] = (
    ("step_curl_kernels.py", "STENCIL",
     "a PML curl is a neighbour stencil by construction — it is the finite "
     "difference. If the detector calls THIS pointwise it is measuring nothing"),
    ("cylindrical_kernels.py", "STENCIL",
     "the cylindrical curl, same argument, on a module whose neighbour read is a "
     "direct `pfx[idx + sx]` in the kernel body rather than inside a helper"),
    ("nonlinear_constitutive.py", "STENCIL",
     "the nonlinear constitutive reads four corners of two partner volumes "
     "(nonlinear_constitutive.py:435, `four_point_sum(g, o_c, o_d, o_u, o_ud, ...)`) "
     "— a CONSTITUTIVE that is a stencil, which is the case a name-based rule would "
     "get wrong"),
    ("constitutive_kernels.py", "POINTWISE",
     "the diagonal real constitutive reads each cell it writes and nothing else "
     "(constitutive_kernels.py:277-283) — the negative control, without which every "
     "control above would pass on a detector that answered STENCIL always"),
)


def run_detector_controls() -> List[dict]:
    results = []
    for module, expected, why in DETECTOR_CONTROLS:
        got = analyze_kernel_reads({"module": module, "mode": "module"})
        results.append({"module": module, "expected": expected,
                        "measured": got["verdict"], "why_it_is_settled": why,
                        "evidence": (got.get("reads_off_this_thread_cell") or [])[:2],
                        "coordinate_helpers":
                            got.get("neighbour_coordinate_helpers") or []})
        if got["verdict"] != expected:
            raise SystemExit(
                f"DETECTOR CONTROL FAILED: {module} must measure {expected} and "
                f"measured {got['verdict']}. {why}. Every POINTWISE/STENCIL verdict "
                f"in this board comes from the same function, so none of them may "
                f"be reported until this reproduces.")
    return results


_EMITTED_GLOBAL = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(')


def _analyze_emitted(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fitness for a family whose kernel exists only as EMITTED text.

    Some families hold no ``__global__`` string literal at all: the dispersive
    pair is assembled line by line out of f-strings (``dispersive_source``), the
    complex off-diagonal and complex no-PML templates are ``__NAME__``/
    ``__SIGMA_*__`` substitutions. ``mode="module"`` refuses them (no kernel
    names) and ``mode="body"`` cannot fill their placeholders from fragment
    literals. This mode analyses what NVRTC would actually see: it IMPORTS the
    module (every module named here is CuPy-free at module scope -- the same
    property that lets the backend-free census ask its predicates), calls the
    module's own emitter once per declared variant, and runs the SAME detector
    over each emitted source. The verdict is accepted only when EVERY variant
    agrees; a disagreement is UNDETERMINED with both verdicts named, because a
    fitness that depends on the variant is not a fact about the cell.

    The variant list is part of the spec and is deliberately explicit: each
    entry is a real specialization the corpus can drive (a pole arity, a row
    mask, an expansion arm), so what is analysed is a shipped kernel body and
    never a synthetic one.
    """
    import importlib  # noqa: PLC0415

    module = spec["module"]
    try:
        loaded = importlib.import_module(spec["import"])
        emitter = getattr(loaded, spec["function"])
    except BaseException as exc:  # noqa: BLE001 - an unaskable emitter is a refusal
        return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                "why": (f"{spec['import']}.{spec['function']} could not be reached "
                        f"on this host: {type(exc).__name__}: {exc}")}
    per_variant: List[Dict[str, Any]] = []
    for arguments in spec["variants"]:
        label = f"{spec['function']}{tuple(arguments)!r}"
        try:
            source = emitter(*arguments)
        except BaseException as exc:  # noqa: BLE001
            return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                    "why": f"{label} raised: {type(exc).__name__}: {exc}"}
        names = _EMITTED_GLOBAL.findall(source)
        if len(names) != 1:
            return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                    "why": (f"{label} emitted {len(names)} __global__ entry "
                            f"points, not 1; the prelude/body split has no anchor")}
        match = _EMITTED_GLOBAL.search(source)
        prelude = source[:match.start()]
        open_paren = source.find("(", match.start())
        close = _matching(source, open_paren)
        if close is None:
            return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                    "why": f"{label}: the emitted signature does not close"}
        brace = source.find("{", close)
        if brace < 0:
            return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                    "why": f"{label}: the emitted signature has no opening brace"}
        body = source[brace + 1:]
        if body.rstrip().endswith("}"):
            body = body.rstrip()[:-1]
        result = analyze_kernel_reads({"module": module, "mode": "_text",
                                       "_prelude": prelude, "_body": body})
        result["variant"] = label
        result["emitted_kernel"] = names[0]
        per_variant.append(result)
    verdicts = sorted({entry["verdict"] for entry in per_variant})
    if verdicts != ["POINTWISE"] and verdicts != ["STENCIL"]:
        if len(verdicts) > 1:
            return {"verdict": "UNDETERMINED", "module": module, "mode": "emitted",
                    "why": (f"the emitted variants disagree ({verdicts}); a fitness "
                            f"that depends on the variant is not a fact about the "
                            f"cell"),
                    "per_variant": [{"variant": v["variant"],
                                     "verdict": v["verdict"]}
                                    for v in per_variant]}
        return per_variant[0]
    first = dict(per_variant[0])
    first["mode"] = "emitted"
    first["module"] = module
    first["variants_measured"] = [entry["variant"] for entry in per_variant]
    first["variants_agree"] = True
    first["resolved_by"] = (f"emitted text of {spec['import']}.{spec['function']} "
                            f"({len(per_variant)} variants, all {verdicts[0]})")
    return first


def analyze_kernel_reads(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Does this family's device code read any volume OFF the cell its thread owns?

    THE MEASUREMENT, in the order it is made:

    1. Split the module's device code (:func:`device_text`) into ``__device__``
       preludes, ``__global__`` kernel bodies, and the C-statement fragments the
       emitters splice into a template.
    2. Assemble the family's kernel text. ``mode="module"`` takes every body and
       every fragment — allowed ONLY where the module's kernels are all of one KIND
       (all ``step_*`` or all ``update_*``), checked here, because a module holding
       both a curl and a constitutive would otherwise credit the curl's neighbour
       reads to the constitutive. ``mode="body"`` takes named bodies and named
       fragments, and REFUSES if the result still carries a placeholder that is not
       declared to splice no subscript.
    3. Strip C comments (see :func:`_strip_comments`).
    4. Find the thread's own flat index — the variable assigned
       ``blockIdx.x * blockDim.x + threadIdx.x`` — and the per-axis coefficient
       indices derived from it by ``%`` and ``/``.
    5. Build the call graph (:func:`_units`) and take the units REACHABLE from the
       kernel text. An unreachable helper in a shared prelude belongs to a sibling
       kernel and its reads are not this family's.
    6. Solve for which index expressions are provably the thread's own cell, by a
       FIXPOINT that starts optimistic and refutes. A helper's parameter is same-cell
       only while every reachable call site binds it to a same-cell expression in the
       caller; one call that does not demotes it, and the demotion propagates through
       the helpers that parameter is handed to. Iterated to stability. That
       transitivity is not optional: the off-diagonal family's neighbour reads are
       three calls deep — body computes ``flat(coord_dn(...), ...)``, hands it to
       ``offdiag_term``, which hands it to ``ghosted``, which is the only place an
       array is subscripted with it.
    7. Report every array subscript, in any reachable unit, whose index is not
       same-cell at the fixpoint — with the unit it sits in as evidence.

    Returns the verdict and the evidence, never a bare boolean.
    """
    module = spec["module"]
    if not (CUDA_KERNELS / module).exists():
        return {"verdict": "UNDETERMINED", "why": f"{module} is not in the package"}
    mode = spec.get("mode", "module")
    if mode == "emitted":
        return _analyze_emitted(spec)
    text = device_text(module)
    if mode == "_text":
        # PRIVATE: one EMITTED variant's exact text, handed in by _analyze_emitted.
        # The prelude/body split was made there (one __global__ per emitted source,
        # asserted); this branch only routes it into the shared tail below.
        chosen_bodies = [(0, spec["_body"])]
        chosen_fragments = []
        kernel_text = spec["_body"]
        text = dict(text)
        text["preludes"] = [(0, spec["_prelude"])]
    elif mode == "module":
        kinds = {("curl" if name.startswith("step_") else "constitutive")
                 for name in text["kernel_names"]}
        if len(text["kernel_names"]) == 0 or len(kinds) != 1:
            return {"verdict": "UNDETERMINED", "module": module,
                    "why": (f"{module} declares kernels {text['kernel_names']} "
                            f"spanning {sorted(kinds) or 'no'} kind(s); module-wide "
                            f"analysis would credit one kernel's reads to another, "
                            f"so this board refuses it and wants an explicit body "
                            f"selection")}
        chosen_bodies = list(text["bodies"])
        chosen_fragments = list(text["fragments"])
    else:
        chosen_bodies = [(lineno, body) for lineno, body in text["bodies"]
                         if any(marker in body for marker in spec["bodies"])]
        if len(chosen_bodies) != 1:
            return {"verdict": "UNDETERMINED", "module": module,
                    "why": (f"{module}: {len(chosen_bodies)} __global__ literals "
                            f"match {spec['bodies']}; this board refuses to guess "
                            f"which body a family ships")}
        chosen_fragments = []
    if mode == "module":
        # EVERY string in the module is in the text, so every splice point's code is
        # present by construction and an unsubstituted placeholder token measures
        # nothing. The floor below belongs to body mode alone.
        kernel_text = "\n".join(
            body for _lineno, body in chosen_bodies + chosen_fragments)
    else:
        # BODY MODE RECONSTRUCTS THE TEMPLATE. Each declared splice is filled from
        # the named fragment literal, so what is analysed is the text NVRTC would
        # see rather than a template with holes in it. Any hole left unfilled and
        # undeclared refuses below: code spliced into it was not measured, and a
        # subscript could be hiding there.
        kernel_text = chosen_bodies[0][1]
        for placeholder, marker in (spec.get("fills") or {}).items():
            filler = [frag for _lineno, frag in text["fragments"] + text["preludes"]
                      if marker in frag]
            if len(filler) != 1:
                return {"verdict": "UNDETERMINED", "module": module,
                        "why": (f"{module}: {len(filler)} literals match the filler "
                                f"marker {marker!r} declared for {placeholder}; "
                                f"this board refuses to guess which one the "
                                f"template splices")}
            kernel_text = kernel_text.replace(placeholder, filler[0])
            chosen_fragments.append((0, filler[0]))
        leftover = sorted(set(re.findall(r"__[A-Za-z][A-Za-z0-9_]*__", kernel_text))
                          - set(PLACEHOLDER_SPLICES_NO_SUBSCRIPT)
                          - {"__restrict__", "__device__", "__global__",
                             "__forceinline__", "__fdiv_rn"})
        if leftover:
            return {"verdict": "UNDETERMINED", "module": module,
                    "why": (f"{module}: the reconstructed kernel text still carries "
                            f"{leftover}. Each is a splice point, and code spliced "
                            f"there was NOT measured — a subscript could be hiding "
                            f"in it. Declare it in "
                            f"PLACEHOLDER_SPLICES_NO_SUBSCRIPT with its reason, or "
                            f"name the fragment literal that fills it in `fills`.")}

    body_c = _strip_comments(kernel_text)
    all_c = _strip_comments(
        "\n".join(text_ for _lineno, text_ in text["preludes"])) + "\n" + body_c

    flat_names = set(_FLAT.findall(body_c))
    if not flat_names:
        return {"verdict": "UNDETERMINED", "module": module,
                "why": (f"{module}: the assembled kernel text declares no "
                        f"blockIdx.x*blockDim.x+threadIdx.x flat index, so this "
                        f"board cannot say which cell a thread owns")}
    flat = sorted(flat_names)[0]
    axis = set()
    for pattern in (rf"\b(\w+)\s*=\s*{flat}\s*%\s*\w+",
                    rf"\b(\w+)\s*=\s*\(\s*{flat}\s*/\s*\w+\s*\)\s*%\s*\w+",
                    rf"\b(\w+)\s*=\s*{flat}\s*/\s*\("):
        axis.update(re.findall(pattern, body_c))

    units = _units(all_c, body_c)
    reachable = {"__kernel__"}
    frontier = ["__kernel__"]
    while frontier:
        unit = units.get(frontier.pop())
        if not unit:
            continue
        for callee in set(re.findall(r"\b(\w+)\s*\(", unit["body"])):
            if callee in units and callee not in reachable:
                reachable.add(callee)
                frontier.append(callee)

    def same_cell_set(unit_name: str, safe_params: Dict[str, set]) -> set:
        body = units[unit_name]["body"]
        local = ({flat} | set(axis) if unit_name == "__kernel__"
                 else set(safe_params.get(unit_name, ())))
        local.add(_FSTRING_TOKEN)
        for _ in range(4):
            grown = set(local)
            for name, expr in re.findall(
                    r"\b(?:const\s+)?(?:int|long)\s+(\w+)\s*=\s*([^;]{1,60});", body):
                if " ".join(expr.split()) in local:
                    grown.add(name)
            if grown == local:
                break
            local = grown
        return local

    safe_params: Dict[str, set] = {
        name: set(units[name]["params"]) for name in reachable if name != "__kernel__"}
    for _iteration in range(len(units) + 4):
        changed = False
        for caller in sorted(reachable):
            body = units[caller]["body"]
            local = same_cell_set(caller, safe_params)
            for callee in sorted(reachable):
                if callee == "__kernel__":
                    continue
                for call in re.finditer(rf"\b{re.escape(callee)}\s*\(", body):
                    close = _matching(body, call.end() - 1)
                    if close is None:
                        continue
                    args = _split_args(body[call.end():close])
                    params = units[callee]["params"]
                    if len(args) != len(params):
                        continue
                    for param, argument in zip(params, args):
                        if param not in safe_params.get(callee, ()):
                            continue
                        normalized = " ".join(argument.split())
                        if normalized in local or normalized.isdigit():
                            continue
                        if _interleaved(normalized, local):
                            continue
                        safe_params[callee].discard(param)
                        changed = True
        if not changed:
            break

    evidence: List[dict] = []
    interleaving: List[dict] = []
    for unit in sorted(reachable):
        body = units[unit]["body"]
        local = same_cell_set(unit, safe_params)
        for base, expr in _SUBSCRIPT.findall(body):
            normalized = " ".join(expr.split())
            if normalized in local or normalized.isdigit():
                continue
            slot = _interleaved(normalized, local)
            if slot:
                interleaving.append({
                    "where": unit, "read": f"{base}[{normalized}]",
                    "why": (f"interleaved storage: stride {slot[0]}, offset "
                            f"{slot[1]} on the same-cell index {slot[2]!r}; every "
                            f"word this reads is inside the slot this thread owns")})
                continue
            evidence.append({
                "where": ("kernel body" if unit == "__kernel__"
                          else f"{unit}() — a __device__ helper the body reaches"),
                "read": f"{base}[{normalized}]",
                "why": (f"{normalized!r} is not the thread's own cell: not the flat "
                        f"index {flat!r}, not an axis index {sorted(axis)}, and not "
                        f"a same-cell name at the fixpoint {sorted(local)}")})

    shifted = _shifted_coordinate_helpers(units, reachable)
    return {
        "verdict": "STENCIL" if (evidence or shifted) else "POINTWISE",
        "verdict_from": (
            ["a subscript off this thread's cell"] * bool(evidence)
            + ["a reachable helper that builds a shifted coordinate"] * bool(shifted)
            or ["neither detector fired"]),
        "neighbour_coordinate_helpers": shifted,
        "module": module, "mode": mode,
        "bodies_measured": [f"{module}:{lineno}" for lineno, _b in chosen_bodies],
        "fragments_measured": len(chosen_fragments),
        "kernel_names_in_module": text["kernel_names"],
        "flat_index": flat, "axis_indices": sorted(axis),
        "reachable_device_helpers": sorted(reachable - {"__kernel__"}),
        "same_cell_parameters_at_the_fixpoint": {
            name: sorted(params) for name, params in sorted(safe_params.items())},
        "reads_off_this_thread_cell": evidence,
        "reads_inside_this_thread_interleaved_slot": interleaving,
        "what_was_measured": (
            "every array subscript in the comment-stripped kernel text and in every "
            "__device__ helper reachable from it, with helper index parameters "
            "resolved to their call sites by fixpoint"),
    }


def _balanced_args(source: str, at: int) -> Optional[str]:
    depth = 0
    for index in range(at, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return source[at + 1:index]
    return None


def _split_args(text: str) -> List[str]:
    out, depth, current = [], 0, []
    for character in text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        if character == "," and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(character)
    if current:
        out.append("".join(current))
    return [a.strip() for a in out]


# ---------------------------------------------------------------------------
# THE FITNESS SOURCE — WHICH DEVICE CODE A SELECTED KERNEL IS
# ---------------------------------------------------------------------------

#: The composer's arm table, read out of ``registry.py`` by ``ast``.
#:
#: DERIVED, NOT TRANSCRIBED, and for the reason the seam locator is: the table moves.
#: Returned as ``{(family, arm_label, slot): {noun, kernel, predicate_module}}`` with
#: the ``{slot}`` format placeholder resolved, which is exactly the string the
#: composer records as ``plan_step.selected_kernel``. That equality is CHECKED
#: against the census below rather than trusted — a census cut against a different
#: registry would otherwise be read as if it described this one.
#:
#: THE ARM LABEL IS PART OF THE KEY, and (family, slot) alone is not: registry.py:288
#: and :297 register TWO arms of ``cuda_complex_offdiag`` at ``update_E`` — a PML
#: tail and a no-PML tail — shipping different kernels. Keyed on the pair, one would
#: silently shadow the other and the cross-check would report drift that is not there.
def registry_arms() -> Dict[Tuple[str, str, str], dict]:
    tree = ast.parse(REGISTRY.read_text())
    slot_names: Dict[str, Tuple[str, ...]] = {}
    table = None
    for node in ast.walk(tree):
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target is None:
            continue
        if target == "_TABLE":
            table = value
        elif target.endswith("_SLOTS"):
            try:
                slot_names[target] = tuple(ast.literal_eval(value))
            except Exception:
                pass
    if table is None or not isinstance(table, (ast.Tuple, ast.List)):
        raise SystemExit(
            f"{REGISTRY} declares no literal _TABLE; this board reads the arm table "
            f"by ast (registry imports every family module, and several of those "
            f"import the device toolchain) and cannot proceed without it.")
    out: Dict[Tuple[str, str, str], dict] = {}
    for element in table.elts:
        entry: Dict[str, Any] = {}
        for key, value in zip(element.keys, element.values):
            name = getattr(key, "value", None)
            if isinstance(value, ast.Name):
                # ONLY `slots` IS RESOLVED THROUGH A NAME. The other keys that carry
                # one (`license`, which names a LICENSE_* constant) are not read by
                # this board and are kept as the identifier rather than resolved, so
                # a new constant cannot stop the run over a field nothing here uses.
                if name != "slots":
                    entry[name] = f"<name {value.id}>"
                    continue
                entry[name] = slot_names.get(value.id)
                if entry[name] is None:
                    raise SystemExit(
                        f"{REGISTRY}: arm row `slots` is the name {value.id!r} and "
                        f"no module-level *_SLOTS literal defines it; refusing to "
                        f"guess which slots this arm serves.")
                continue
            try:
                entry[name] = ast.literal_eval(value)
            except Exception:
                entry[name] = None
        for slot in entry.get("slots") or ():
            kernel = entry.get("kernel")
            label = (kernel.get(slot) if isinstance(kernel, dict)
                     else (kernel or "").replace("{slot}", slot))
            out[(entry["family"], entry.get("label"), slot)] = {
                "noun": entry.get("noun"), "kernel": label,
                "predicate_module": entry.get("module"),
                "has_launch_resolver": bool(entry.get("resolve"))}
    return out


#: Kernel labels whose device body is emitted through a SIBLING module under
#: substitution, so no module declares the label as an ``extern "C" __global__`` name
#: and the derivation below cannot find it. One entry per label, each carrying the
#: same source spec the predicate-era board used for that body, so this re-point
#: changes which cell a row lands in and NOT what the fitness detector is pointed at.
#:
#: ``complex_emitter`` emits its kernel names through a ``__NAME__`` substitution
#: (there is no literal to match), which is why the complex families need the
#: explicit body selection rather than a name lookup.
FITNESS_SOURCE_BY_LABEL: Dict[str, dict] = {
    "fused_update_H_pml_complex_bloch": {
        "module": "complex_emitter.py", "mode": "body",
        "bodies": ("constitutive_apply(f0, w0, idx",),
        "fills": {"__SCALE__": "mul_field_left(s0, inv_eps_0[idx]"}},
    "fused_update_E_pml_complex_bloch": {
        "module": "complex_emitter.py", "mode": "body",
        "bodies": ("constitutive_apply(f0, w0, idx",),
        "fills": {"__SCALE__": "mul_field_left(s0, inv_eps_0[idx]"}},
    # THE FIVE EMITTED-TEXT LABELS, 2026-09-02 residue round (audit §1.5 stage 1).
    # Each names a family whose kernel exists only as EMITTED text -- f-string
    # assembly (dispersive_source) or __NAME__/__SIGMA__ substitution -- so
    # neither mode="module" (no declared kernel names) nor mode="body" (no
    # fillable fragment literals) can reach it; mode="emitted" analyses the text
    # NVRTC would compile, per shipped variant, and requires every variant to
    # agree. The two composite labels are the registry's own spellings
    # (registry.py: `fused_update_P_no_pml_complex[_uniform]`), and the variant
    # lists cover BOTH specializations the bracket denotes, so the verdict is a
    # fact about the cell whichever the launcher selects.
    "fused_update_P_no_pml_complex[_uniform]": {
        "module": "complex_no_pml_kernels.py", "mode": "emitted",
        "import": "meep_gpu.cuda_kernels.complex_no_pml_kernels",
        "function": "kernel_source",
        "variants": (("update_P", "NAIVE"), ("update_P", "FMA_V1"),
                     ("update_P_uniform", "NAIVE"),
                     ("update_P_uniform", "FMA_V1"))},
    "fused_update_E_pml_real_dispersive": {
        "module": "dispersive_kernels.py", "mode": "emitted",
        "import": "meep_gpu.cuda_kernels.dispersive_kernels",
        "function": "dispersive_source",
        "variants": (("pml", (0, 0, 0)), ("pml", (1, 1, 1)), ("pml", (2, 0, 3)),
                     ("pml", (6, 6, 6)))},
    "update_E_no_pml_real_dispersive": {
        "module": "dispersive_kernels.py", "mode": "emitted",
        "import": "meep_gpu.cuda_kernels.dispersive_kernels",
        "function": "dispersive_source",
        "variants": (("no_pml", (0, 0, 0)), ("no_pml", (1, 1, 1)),
                     ("no_pml", (2, 0, 3)), ("no_pml", (6, 6, 6)))},
    "update_E_pml_complex_offdiag": {
        "module": "complex_offdiag_update_e.py", "mode": "emitted",
        "import": "meep_gpu.cuda_kernels.complex_offdiag_update_e",
        "function": "complex_offdiag_source",
        "variants": (("pml", (1, 0, 0, 0, 0, 0), "NAIVE"),
                     ("pml", (1, 1, 1, 1, 1, 1), "NAIVE"),
                     ("pml", (1, 1, 1, 1, 1, 1), "FMA_V1"))},
    "update_E_no_pml_complex_offdiag": {
        "module": "complex_offdiag_update_e.py", "mode": "emitted",
        "import": "meep_gpu.cuda_kernels.complex_offdiag_update_e",
        "function": "complex_offdiag_source",
        "variants": (("no_pml", (1, 0, 0, 0, 0, 0), "NAIVE"),
                     ("no_pml", (1, 1, 1, 1, 1, 1), "NAIVE"),
                     ("no_pml", (1, 1, 1, 1, 1, 1), "FMA_V1"))},
}

#: Labels that name a plan launching NO device work at all. Not a fitness question:
#: there is no second kernel for a curl to be welded to.
NULL_PLAN_LABELS = ("fused_update_H_no_pml_null", "fused_update_E_no_pml_null")


def declared_kernel_modules() -> Dict[str, List[str]]:
    """Every ``extern "C" __global__`` name in the package, to the modules declaring it."""
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for path in sorted(CUDA_KERNELS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        for name in device_text(path.name)["kernel_names"]:
            out[name].append(path.name)
    return dict(out)


def fitness_source(label: Optional[str],
                   declared: Dict[str, List[str]]) -> Optional[dict]:
    """The device code a selected kernel label IS, or ``None`` if it is not settled.

    THE RESOLUTION, IN ORDER, AND IT REFUSES RATHER THAN GUESSING:

    1. A label in :data:`NULL_PLAN_LABELS` launches nothing — handled by the caller.
    2. A label in :data:`FITNESS_SOURCE_BY_LABEL` takes that spec. These are the
       bodies a sibling emitter substitutes a name into, where no literal carries the
       label.
    3. Otherwise the label is matched against the ``extern "C" __global__`` names
       actually declared in the package, with a leading ``fused_`` stripped: the
       registry's own docstring records that its labels name the PYTHON WRAPPER
       (``fused_update_H_pml_real``) while the declared entry point drops the prefix
       (``update_H_pml_real``). A unique declaring module is taken with
       ``mode="module"``; zero or several is UNDETERMINED.

    ``mode="module"`` unions every kernel body in the module. Where a module declares
    more than one body that union is CONSERVATIVE and cannot invent a weld: adding
    another kernel's text can only add reads that land off this thread's cell, so the
    verdict can move POINTWISE -> STENCIL and never the reverse. The analyzer refuses
    a module outright if its kernels span a curl and a constitutive.
    """
    if not label:
        return None
    spec = FITNESS_SOURCE_BY_LABEL.get(label)
    if spec is not None:
        return dict(spec)
    candidates = [label]
    if label.startswith("fused_"):
        candidates.append(label[len("fused_"):])
    for candidate in candidates:
        modules = declared.get(candidate) or []
        if len(modules) == 1:
            return {"module": modules[0], "mode": "module",
                    "resolved_by": f"declares {candidate}"}
        if len(modules) > 1:
            return None
    return None


def _installable_declaration(column: str) -> Dict[str, Any]:
    """What a spanning product's own module says about being INSTALLED.

    READ OFF THE SOURCE, not by import: the family modules import CuPy at scope and
    this board runs on a host with none. A product may hold a released device gate and
    still be refused by the composer on every configuration -- that is a statement
    about the COMPOSITION, and a board that credited its predicate without carrying it
    would read as a claim that the product runs.
    """
    import ast as _ast  # noqa: PLC0415

    from meep_gpu.cuda_kernels import fused_pairs as _fused  # noqa: PLC0415

    product = _fused.FUSED_PRODUCTS.get(column) or {}
    module = product.get("module")
    if not module:
        return {"module": None, "installable": None,
                "reason": "the board's product table names no module for this column"}
    # BY NAME, never by parents[N]: this file lives at parity/meep_gpu/, so the API
    # root is two levels up, and a wrong count reads as "not in this checkout" -- a
    # refusal that looks like a finding. Resolved by looking for the package itself.
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "meep_gpu" / "cuda_kernels").is_dir())
    path = root / "meep_gpu" / "cuda_kernels" / f"{module}.py"
    if not path.is_file():
        return {"module": module, "installable": None,
                "reason": f"{path} is not in this checkout"}
    tree = _ast.parse(path.read_text(encoding="utf-8"))
    found: Dict[str, Any] = {"module": module, "installable": True, "reason": None}
    for node in tree.body:
        if not (isinstance(node, _ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], _ast.Name)):
            continue
        if node.targets[0].id == "INSTALLABLE":
            found["installable"] = bool(_ast.literal_eval(node.value))
        elif node.targets[0].id == "INSTALLABLE_REASON":
            found["reason"] = _ast.literal_eval(node.value)
    return found


SLOTS = ("step_B", "step_D", "update_H", "update_E", "update_P")

#: Which slot each seam takes its two halves from.
SEAM_SLOTS = {"B_to_H": ("step_B", "update_H"),
              "D_to_E": ("step_D", "update_E"),
              "E_to_P": ("update_E", "update_P")}

#: THE FOURTH SEAM IS SPANNABLE BUT IS NOT IN THE LEDGER, and holding it in a second
#: table is the whole point rather than an oversight. :data:`SEAM_SLOTS` is the
#: THREE-SEAM LEDGER's own table -- every per-cell walk, every ceiling, every bucket
#: and the 403-instance denominator in this file are built over it, cell by cell as
#: (curl arm, constitutive arm) -- and ``H_to_D`` is priced ROW BY ROW by the shared
#: ``h_to_d_seam.price`` instead, which is why the board's PRICED denominator is
#: ``403 + one instance per row``. Adding a fourth row to SEAM_SLOTS would put the
#: seam through the cell walk as well and count it twice.
#:
#: What the two tables DO share is the span resolution: a product declares the slots
#: it spans, and :func:`_seams_of_span` has to be able to name the seam those slots
#: are, or it refuses the column by name. So the fourth seam is joined at exactly that
#: one place, from ``h_to_d_seam``'s own constants rather than a second spelling.
SPANNABLE_SEAM_SLOTS = dict(
    SEAM_SLOTS, **{h_to_d_seam.SEAM: tuple(h_to_d_seam.HALVES)})


# ---------------------------------------------------------------------------
# THE SOURCE-SEAM CEILING
# ---------------------------------------------------------------------------

def no_magnetic_source(facts: dict) -> bool:
    return all(kind != "B" for kind in facts["source_field_types"])


def no_electric_source(facts: dict) -> bool:
    types = facts["source_field_types"]
    return bool(types) and all(kind == "B" for kind in types)


SOURCE_SEAM_CLAUSE: Dict[str, Callable[[dict], bool]] = {
    "B_to_H": no_magnetic_source, "D_to_E": no_electric_source}


# --- THE IN-SEAM DEPOSIT, READ FROM THE SHIPPED FLAG RATHER THAN TRANSCRIBED ---
#
# The two clauses above are a DRIVER FACT and they stay one: the driver deposits
# between the halves, so a launch that spans the deposit consumes a pre-injection
# field. What they are NOT is the whole answer, and this board printed them as if
# they were until 2026-08-28. `meep_gpu/deposit_repair.py` removes the refusal for a
# product that BRACKETS its launch -- the deposit points are saved immediately before
# it (`LeadingRepairPlan`) and recomputed after the driver has injected, filled
# symmetry and cleared walls (`TrailingRepairPlan`) -- and every fused-pair predicate
# routes its source question through `deposit_repair.seam_source_reasons(...,
# carries_repair=<module>.CARRIES_DEPOSIT_REPAIR)`.
#
# THE SIBLING BOARD ALREADY MADE THIS MISTAKE AND FIXED IT: its 2026-08-27 02:38 cut
# transcribed `no_magnetic_source` for every B-side product and kept printing that
# refusal after `metal_kernels/fused_magnetic_pair.py:204` had stopped making it
# (`build_fusion_matrix.py:524-539`). So the credit here IMPORTS the shipped module
# and asks, and it is attached to the CELL rather than to the seam: a deposit is
# carried by a PRODUCT, and a cell whose arms no shipped product implements gets no
# credit no matter what some other product declares.

#: (seam, curl arm key, constitutive arm key) -> the shipped module that fuses that
#: cell. ONE ROW, and it is the board's own rank-1 B_to_H cell: the CUDA product's
#: predicate is literally `covers_real_pml_curl(..., "step_B")` and
#: `covers_real_pml_constitutive(..., "H")`, which are the `PML` and `ordinary` arms'
#: own predicates, and `cuda_kernels/fused_pairs.FUSED_PAIR_ARMS` records the same
#: two labels as the pair it may absorb. A cell with no row here is priced on the
#: driver fact alone.
PRODUCT_ON_CELL: Dict[Tuple[str, str, str], str] = {
    ("B_to_H", "cuda_curl/PML", "cuda_constitutive/ordinary"):
        "meep_gpu.cuda_kernels.fused_magnetic_pair",
    # THE TWO COMPLEX CELLS, ADDED 2026-08-30 WITH THE PRODUCTS THAT OCCUPY THEM.
    # Each row is read from that product's own FUSED_PAIR_ARMS entry -- the arm
    # labels the composer must already have selected before the pair may absorb the
    # two slots -- so a row here can never name a cell the installer would refuse.
    ("B_to_H", "cuda_complex/complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.complex_fused_magnetic_pair",
    ("B_to_H", "cuda_cyl_complex/cylindrical complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.cylindrical_fused_magnetic_pair",
    # THE FIRST ELECTRIC ROW, ADDED 2026-08-30 WITH ITS PRODUCT, and the one whose
    # effect on the ceiling is the point rather than a side effect: 75 of this cell's
    # 79 rows carry an electric deposit, so before the row the whole cell was priced
    # at 4 reachable on the driver fact alone. `covers_fused_electric_pair` opens with
    # `covers_real_pml_curl(..., "step_D")` and `covers_real_pml_constitutive(...,
    # "E")` -- the same `PML` and `ordinary` labels its magnetic sibling conjoins, and
    # the same pair `fused_pairs.FUSED_PAIR_ARMS` records for it.
    ("D_to_E", "cuda_curl/PML", "cuda_constitutive/ordinary"):
        "meep_gpu.cuda_kernels.fused_electric_pair",
    # THE TWO COMPLEX ELECTRIC ROWS, ADDED 2026-08-31 WITH THEIR PRODUCTS, and the
    # pair whose effect on the ceiling is the entire point. Before them these two
    # cells were priced on the driver fact alone at 1 and 0 reachable of 16 rows each
    # -- 31 of the 32 seam-instances carry an electric deposit -- while both were
    # POINTWISE-BUILDABLE. A fitness verdict buys nothing on a cell no product
    # occupies, because the board has no product to ask about the repair.
    #
    # Each row is read from that product's own FUSED_PAIR_ARMS entry, which is itself
    # read from the predicate it conjoins, so a row here can never name a cell the
    # installer would refuse: `covers_complex_fused_electric_pair` opens with
    # `covers_real_pml_complex_curl(..., "step_D")` and
    # `covers_real_pml_complex_constitutive(..., "E")` -- both the `cuda_complex`
    # family's `complex` arm -- and `covers_cylindrical_fused_electric_pair` with
    # `covers_pml_cylindrical_complex_curl(..., "step_D")` and the same constitutive.
    ("D_to_E", "cuda_complex/complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.complex_fused_electric_pair",
    ("D_to_E", "cuda_cyl_complex/cylindrical complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.cylindrical_fused_electric_pair",
    # THE NO-ABSORBER ELECTRIC ROW, ADDED 2026-08-31 WITH ITS PRODUCT, and the one row
    # here whose credit is ZERO by construction rather than large: all four of this
    # cell's rows declare `source_field_types == ['B']`, so the cell already priced at
    # 4 of 4 reachable on the driver fact alone and the module declares
    # `CARRIES_DEPOSIT_REPAIR = False`. The row is added anyway, and that is the point:
    # `served_by_a_shipped_product` reads off the census's own record of the shipped
    # predicate, and without a row here the board would report a POINTWISE-BUILDABLE
    # cell with a shipped product on it as served by nothing.
    #
    # Read from that product's own FUSED_PAIR_ARMS entry, which is itself read from
    # the predicate it conjoins, so a row here can never name a cell the installer
    # would refuse: `covers_no_pml_complex_fused_electric_pair` opens with
    # `complex_no_pml_kernels.covers_complex_no_pml_curl(..., "step_D")` and
    # `covers_complex_no_pml_stored_e` -- the `cuda_complex_no_pml` family's own two
    # arms on these slots.
    # REPOINTED 2026-09-04 TO THE COMPLEX THREE-SLOT WELD, which is the product the
    # composer now installs on this cell: its span STRICTLY CONTAINS the pair's, so
    # `fused_pairs._superseded_by_a_longer_span` gives it the slots. THE CREDIT DOES
    # NOT MOVE and that is checkable: both modules declare CARRIES_DEPOSIT_REPAIR
    # False and the seam is empty on every row of the cell (magnetic source only),
    # so the number this row yields is the same whichever module it names.
    ("D_to_E", "cuda_complex_no_pml/complex no-PML curl",
     "cuda_complex_no_pml/complex no-PML stored E"):
        "meep_gpu.cuda_kernels.complex_no_pml_three_slot_dispersive_weld",
    # THE TWO E->P ROWS, ADDED 2026-09-01 WITH THEIR PRODUCTS, and the rows whose
    # deposit credit is zero BY DRIVER CONSTRUCTION rather than by measurement:
    # `SOURCE_SEAM_CLAUSE` has no E_to_P entry because the driver injects nothing
    # between update_E and update_P (driver.py:3313/:3315), so every instance
    # already clears on the driver fact and `clears_source_seam` returns before
    # reading these rows. They are added anyway, for the reason the no-absorber
    # electric row was: `served_by_a_shipped_product` reads off the census's own
    # record of the shipped predicate, and without the module on this table the
    # board would report a POINTWISE-BUILDABLE cell with a shipped product on it
    # as priced against nothing.
    #
    # Each row is read from that product's own FUSED_PAIR_ARMS entry, which is
    # itself read from the predicates it conjoins:
    # `covers_fused_polarization_pair` opens with
    # `covers_real_pml_dispersive_constitutive` and
    # `covers_real_pml_ade_update_p` -- the `cuda_dispersive/dispersive` and
    # `cuda_ade/ADE` labels of this cell -- and its no-absorber twin with the
    # `cuda_no_pml_dispersive` and `cuda_no_pml_ade` families' own predicates.
    # THE FIVE RESIDUAL MAGNETIC ROWS, ADDED 2026-09-02 WITH THEIR PRODUCTS --
    # the last buildable cells on this board. Each row is read from that
    # product's own FUSED_PAIR_ARMS entry, which is itself read from the
    # predicate it conjoins, so a row here can never name a cell the installer
    # would refuse. The folded-complex product's repair credit is the point of
    # its row: the cell's fifth seam-instance (TestHoleyWvgBands.test_fields_at_kx)
    # deposits a plain point B Source between the halves, and the module's
    # CARRIES_DEPOSIT_REPAIR (imported live, never transcribed) is what clears
    # it. The other four declare False, measured off their cells.
    ("B_to_H", "cuda_complex_folded/folded complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.complex_folded_fused_magnetic_pair",
    ("B_to_H", "cuda_complex_beta/complex beta", "cuda_complex_beta/complex beta"):
        "meep_gpu.cuda_kernels.complex_beta_fused_magnetic_pair",
    ("B_to_H", "cuda_cylindrical/cylindrical", "cuda_constitutive/ordinary"):
        "meep_gpu.cuda_kernels.cylindrical_real_fused_magnetic_pair",
    ("B_to_H", "cuda_special_kz/real beta", "cuda_special_kz/real beta"):
        "meep_gpu.cuda_kernels.special_kz_fused_magnetic_pair",
    ("B_to_H", "cuda_bfast/BFAST", "cuda_bfast/BFAST"):
        "meep_gpu.cuda_kernels.bfast_fused_magnetic_pair",
    ("E_to_P", "cuda_dispersive/dispersive", "cuda_ade/ADE"):
        "meep_gpu.cuda_kernels.fused_polarization_pair",
    ("E_to_P", "cuda_no_pml_dispersive/no-PML dispersive store",
     "cuda_no_pml_ade/no-PML ADE"):
        "meep_gpu.cuda_kernels.fused_polarization_pair",
    # THE FIVE ELECTRIC-TWIN ROWS AND THE THIRD E->P ROW, ADDED 2026-09-02 WITH
    # THEIR PRODUCTS (the residue round; fusion-residue audit §1.3's 12
    # electric-twin instances and §1.5's determined E->P cell). Each row is
    # read from that product's own FUSED_PAIR_ARMS entry, which is itself read
    # from the predicate it conjoins, so a row here can never name a cell the
    # installer would refuse. On every ELECTRIC row the repair credit is the
    # entire point: every reachable row of every one of the five cells declares
    # an electric deposit inside the seam (measured off the residualwelds
    # census), so before these rows the board priced all 12 on the driver fact
    # alone. The E->P row's credit is zero by driver construction (nothing is
    # injected between its two consults) and it is added for the reason the
    # first two E->P rows were: without it the board would report a
    # POINTWISE-BUILDABLE cell with a shipped product on it as served by
    # nothing.
    ("D_to_E", "cuda_complex_folded/folded complex", "cuda_complex/complex"):
        "meep_gpu.cuda_kernels.complex_folded_fused_electric_pair",
    ("D_to_E", "cuda_complex_beta/complex beta",
     "cuda_complex_beta/complex beta"):
        "meep_gpu.cuda_kernels.complex_beta_fused_electric_pair",
    ("D_to_E", "cuda_cylindrical/cylindrical", "cuda_constitutive/ordinary"):
        "meep_gpu.cuda_kernels.cylindrical_real_fused_electric_pair",
    ("D_to_E", "cuda_special_kz/real beta", "cuda_special_kz/real beta"):
        "meep_gpu.cuda_kernels.special_kz_fused_electric_pair",
    ("D_to_E", "cuda_bfast/BFAST", "cuda_bfast/BFAST"):
        "meep_gpu.cuda_kernels.bfast_fused_electric_pair",
    ("E_to_P", "cuda_complex_no_pml/complex no-PML stored E",
     "cuda_complex_no_pml/complex no-PML ADE"):
        "meep_gpu.cuda_kernels.complex_no_pml_fused_polarization_pair",
    # THE TWO STENCIL ROWS, ADDED 2026-09-02 WITH THEIR PRODUCTS -- and the only
    # rows on this table whose cells this board had recorded UNBUILDABLE. The
    # verdict is NOT relaxed and the `fitness_rule` below is NOT edited: STENCIL
    # stays the fitness of both constitutive kernels, because it is a true reading
    # of their device text (they read `u[up]`, `g[2*idx]`, `g[o_d]` -- cells the
    # thread does not own). What changes is that a product now occupies the cell
    # whose weld does not require those reads to come from the curl half's
    # in-place output: it writes a launch-local scratch, recomputes each foreign
    # sample from pre-launch D/fu/H through the same __device__ function, and
    # rotates the D bindings after the launch. The board therefore reports these
    # two cells as SERVED BY A PRODUCT while still reporting their constitutive
    # fitness as STENCIL, which is the honest pair of facts.
    #
    # THE REPAIR CREDIT IS ZERO ON BOTH, and by two independent routes that agree:
    # each module declares CARRIES_DEPOSIT_REPAIR = False, and
    # `_deposit_is_repairable` above refuses the D seam on
    # `has_offdiagonal_epsilon` -- which every row of both cells carries, since
    # the constitutive arms REQUIRE a surviving off-diagonal row. So the ceiling
    # is exactly the rows that clear the injection on the driver fact alone: 8 on
    # the unfolded cell and 9 on the folded one.
    #
    # Each row is read from that product's own FUSED_PAIR_ARMS entry, which is
    # itself read from the predicate it conjoins, so a row here can never name a
    # cell the installer would refuse: `covers_offdiag_fused_electric_pair` opens
    # with `covers_real_pml_curl(..., "step_D")` and
    # `covers_real_pml_offdiag_constitutive`, and
    # `covers_folded_offdiag_fused_electric_pair` with the same curl and
    # `covers_folded_offdiag_composition`.
    ("D_to_E", "cuda_curl/PML", "cuda_offdiag/off-diagonal"):
        "meep_gpu.cuda_kernels.offdiag_fused_electric_pair",
    ("D_to_E", "cuda_curl/PML", "cuda_folded_offdiag/folded off-diagonal"):
        "meep_gpu.cuda_kernels.folded_offdiag_fused_electric_pair",
    # THE LAST UNSERVED E->P CELL, ADDED 2026-09-02 WITH ITS PRODUCT. The board
    # already priced it POINTWISE-BUILDABLE and recorded a `hazard` beside the
    # verdict; that hazard names the per-component SPLIT (launch y reading a P
    # slot launch x's rotation had already moved), and this product does not
    # split -- update_E stays monolithic and one component's recurrence rides
    # inside it, so no rotation happens within the launch. Its deposit credit is
    # zero BY DRIVER CONSTRUCTION, like the three E->P rows above it: nothing is
    # injected between driver.py:3313 and :3315.
    ("E_to_P", "cuda_dispersive_offdiag/dispersive off-diagonal", "cuda_ade/ADE"):
        "meep_gpu.cuda_kernels.dispersive_offdiag_fused_polarization_pair",
    # THE DISPERSIVE ELECTRIC ROW, ADDED 2026-09-02 WITH ITS PRODUCT, and the
    # largest single cell this table has ever gained: `D_to_E (cuda_curl/PML,
    # cuda_dispersive/dispersive)` carries 7 corpus rows and was the biggest
    # POINTWISE-BUILDABLE cell no product occupied on ANY backend.
    #
    # THE REPAIR CREDIT IS THE WHOLE CELL, not part of it. All 7 rows declare an
    # electric deposit inside the seam (`source_field_types == ['D']` on six and ten
    # D sources on stochastic_emitter.py, measured off the stamped census), so
    # `demand_rows_clearing_the_source_seam` was 0 of 7 before this row: the
    # product-aware rule requires a shipped product to occupy the cell AND declare
    # CARRIES_DEPOSIT_REPAIR, and neither held. The module declares it True and the
    # constant is imported live by `_carries_deposit_repair`, never transcribed here.
    #
    # `_deposit_is_repairable` clears all 7: none carries an off-diagonal chi1inv
    # (the dispersive constitutive arm refuses one by name -- that shape belongs to
    # `cuda_dispersive_offdiag`) and none carries an instantaneous chi2/chi3.
    #
    # Read from that product's own FUSED_PAIR_ARMS entry, which is itself read from
    # the predicate it conjoins, so this row can never name a cell the installer
    # would refuse: `covers_dispersive_fused_electric_pair` opens with
    # `coverage.covers_real_pml_curl(..., "step_D")` and
    # `dispersive_kernels.covers_real_pml_dispersive_constitutive`.
    # REPOINTED 2026-09-02 TO THE THREE-SLOT WELD, which is the product the composer
    # now installs on this cell: its span STRICTLY CONTAINS the pair's, so
    # `fused_pairs._superseded_by_a_longer_span` gives it the slots and refuses the
    # pair by name. THE CREDIT DOES NOT MOVE and that is checkable rather than
    # asserted -- both modules declare CARRIES_DEPOSIT_REPAIR True and both declare
    # the SPLIT-FIELD path, and the weld's flag IS the pair's, inherited with the
    # D->E half it welds. What changes is only which module this table names, which
    # is the one thing it is for.
    ("D_to_E", "cuda_curl/PML", "cuda_dispersive/dispersive"):
        "meep_gpu.cuda_kernels.three_slot_dispersive_weld",
    # THE TWO NO-ABSORBER DISPERSIVE-STORE CELLS, ADDED 2026-09-02, AND THEIR ABSENCE
    # UNTIL THEN WAS A MEASUREMENT RATHER THAN AN OMISSION -- one this round closes.
    #
    # `meep_gpu.cuda_kernels.no_pml_dispersive_fused_electric_pair` exists, is gated
    # and is RELEASED under both float32 subnormal policies
    # (certification.json: cuda_no_pml_dispersive_fused_electric_pair_2026-09-02b).
    # What it could NOT do was occupy these two cells, because occupying them cost
    # exactly what it gained:
    #
    #   the product owns step_D AND update_E;
    #   `cuda_no_pml_fused_polarization_pair` owns update_E AND update_P;
    #   `install_fused_pairs` gives a slot to ONE product and walks D_to_E first.
    #
    # ALL THREE ROWS of these two cells (absorber-1d.py, TestAbsorber.test_absorber,
    # material-dispersion.py) are EXACTLY the three rows the E_to_P cell
    # `cuda_no_pml_dispersive x cuda_no_pml_ade` is served on. So a row naming the
    # PAIR was +1 at D_to_E and -1 at E_to_P: NET ZERO, paid for by displacing a
    # released product.
    #
    # WHAT THE THREE-SLOT WELD CHANGES IS THAT THERE IS NO LONGER A TRADE. It owns
    # step_D, update_E AND update_P, so the same three rows are served at BOTH seams,
    # and `_loses_the_shared_slot` withholds nothing from a product that serves both
    # sides itself. The rows below therefore name the weld -- read from its own
    # FUSED_PAIR_ARMS entry, whose first two labels are these cells' two arms and
    # whose third is the ADE arm one slot later, with the `no-PML curl` variant
    # declared in FUSED_PAIR_EXTRA_ARMS exactly as its D->E half declares it.
    #
    # The repair credit is the whole of both cells: all three rows declare an
    # electric deposit inside the seam, the module declares CARRIES_DEPOSIT_REPAIR
    # True (imported live by `_carries_deposit_repair`, never transcribed here) and
    # `_deposit_is_repairable` clears all three -- no off-diagonal chi1inv, no
    # instantaneous chi2/chi3.
    ("D_to_E", "cuda_conductive/conductive",
     "cuda_no_pml_dispersive/no-PML dispersive store"):
        "meep_gpu.cuda_kernels.no_pml_three_slot_dispersive_weld",
    ("D_to_E", "cuda_no_pml_curl/no-PML curl",
     "cuda_no_pml_dispersive/no-PML dispersive store"):
        "meep_gpu.cuda_kernels.no_pml_three_slot_dispersive_weld",
    # THE TWO E_to_P ROWS ABOVE ARE NOT REPOINTED, and that is deliberate. The
    # three-slot welds serve those cells too, on exactly the rows they admit -- but
    # `cuda_(no_pml_)fused_polarization_pair` still serves every OTHER row of them,
    # and this table maps a cell to ONE module. The E->P seam has no
    # SOURCE_SEAM_CLAUSE entry (the driver injects nothing between driver.py:3332
    # and :3334), so the module named there is never read for a repair credit and
    # naming either is the same number; the pair is kept because it is the product
    # that covers more of the cell.
    # THE CONDUCTIVE x ORDINARY CELL, ADDED 2026-09-02 WITH ITS PRODUCT -- the
    # SIGNED-ZERO row, 1 corpus row (tests:TestAdjointSolver.test_damping), which
    # declares BOTH a D and a B source and so was 0 of 1 clearing the seam before
    # this row.
    #
    # IT WAS UNPRICEABLE RATHER THAN UNBUILT UNTIL 2026-09-01. The driver applied its
    # condinv scaling as three WHOLE-VOLUME passes, which rewrite -0.0 to +0.0 at
    # every cell of the component and not only at the deposit; a point repair cannot
    # reconstruct that, so no product could have occupied this cell and a row here
    # would have been an over-count. The driver now replays the rescale sparsely at
    # the published deposit indices, and the product's own predicate keeps the
    # refusal for the one case that still takes the dense branch -- a scaled source
    # publishing no deposit table, which no in-tree source class is.
    ("D_to_E", "cuda_conductive/conductive", "cuda_constitutive/ordinary"):
        "meep_gpu.cuda_kernels.conductive_fused_electric_pair",
    # THE TWO COMPLEX STENCIL CELLS, ADDED 2026-09-02 WITH THEIR PRODUCTS -- the
    # LAST two rows on this table whose cells this board had recorded UNBUILDABLE,
    # and the same treatment the two REAL stencil rows above got: the verdict is NOT
    # relaxed and `fitness_rule` is NOT edited. STENCIL stays the fitness of
    # `update_E_pml_complex_offdiag` and `update_E_no_pml_complex_offdiag`, because
    # it is a true reading of their device text (`g[2*idx]`, `g[2*idx+1]`, `u[up]`
    # -- cells the thread does not own). What changes is that a product now occupies
    # each cell whose weld does not require those reads to come from the curl half's
    # IN-PLACE output: it writes a launch-local scratch, recomputes each foreign
    # sample from pre-launch D/fu/H through the same __device__ function, and rotates
    # the D bindings after the launch. The board therefore reports these two cells as
    # SERVED BY A PRODUCT while still reporting their constitutive fitness as
    # STENCIL, which is the honest pair of facts.
    #
    # THE REPAIR CREDIT IS ZERO ON BOTH, by two independent routes that agree: each
    # module declares CARRIES_DEPOSIT_REPAIR = False, and `_deposit_is_repairable`
    # refuses the D seam on `has_offdiagonal_epsilon` -- which every row of both
    # cells carries, since the constitutive arms REQUIRE a surviving off-diagonal
    # row. So the ceiling is exactly the rows that clear the injection on the driver
    # fact alone: 1 of the folded cell's 3 and 1 of the no-absorber cell's 2.
    #
    # Each row is read from that product's own FUSED_PAIR_ARMS entry, which is itself
    # read from the predicate it conjoins, so a row here can never name a cell the
    # installer would refuse: `covers_folded_complex_offdiag_fused_electric_pair`
    # opens with `complex_folded_kernels.covers_complex_folded_curl(..., "step_D")`
    # and `complex_offdiag_update_e.covers_complex_offdiag_pml_update_e`, and
    # `covers_complex_no_pml_offdiag_fused_electric_pair` with
    # `complex_no_pml_kernels.covers_complex_no_pml_curl(..., "step_D")` and
    # `covers_complex_no_pml_offdiag_update_e`.
    ("D_to_E", "cuda_complex_folded/folded complex",
     "cuda_complex_offdiag/complex off-diagonal PML"):
        "meep_gpu.cuda_kernels.folded_complex_offdiag_fused_electric_pair",
    ("D_to_E", "cuda_complex_no_pml/complex no-PML curl",
     "cuda_complex_offdiag/complex off-diagonal no-PML"):
        "meep_gpu.cuda_kernels.complex_no_pml_offdiag_fused_electric_pair",
}


def _carries_deposit_repair(module_path: str) -> bool:
    """The SHIPPED module's ``CARRIES_DEPOSIT_REPAIR``, imported live."""
    import importlib  # noqa: PLC0415

    return bool(getattr(importlib.import_module(module_path),
                        "CARRIES_DEPOSIT_REPAIR", False))


def _deposit_is_repairable(seam: str, configuration: dict) -> bool:
    """``deposit_repair.repairable``'s own refusals, read from the census block.

    Only the D seam carries any. An off-diagonal chi1inv row makes ``update_E`` a
    stencil over the PARTNER components' volumes at shifted indices
    (stepping.py:1228-1251); an instantaneous chi2/chi3 is not the linear
    accumulation the repair inverts. The third refusal -- ``f_w_<component>``
    unallocated -- needs an engine object, so every count credited here is an UPPER
    BOUND, which is what the board says about itself as a whole.
    """
    if seam == "D_to_E":
        if bool(configuration.get("has_offdiagonal_epsilon")):
            return False
        if bool(configuration.get("has_nonlinearity")):
            return False
    return True


def product_on_cell(seam: str, curl_arm_key: Optional[str],
                    const_arm_key: Optional[str]) -> Optional[str]:
    if curl_arm_key is None or const_arm_key is None:
        return None
    return PRODUCT_ON_CELL.get((seam, curl_arm_key, const_arm_key))


def clears_source_seam(seam: str, facts: dict, configuration: dict,
                       product_module: Optional[str]) -> bool:
    """Does this seam-instance clear its injection, for the product on its cell?

    The driver fact first, unchanged. Then, and only for a cell a shipped product
    implements, the repair: the module's own constant decides, never a boolean typed
    here, because a second copy is exactly what went stale on the sibling board.

    THIS IS THE PRODUCT-AWARE READING and it stays. It answers "can a product that
    EXISTS TODAY serve this instance", which is the right question for this board's
    ranked build list and the only reading ``probe_cuda_source_seam_blockage`` can
    split into its A/B/C groups. It is NOT the right question for a ceiling, and
    reporting it as one is what made this board's 342 incomparable with the sibling
    boards' 363 -- see :func:`clears_source_seam_like_for_like`.
    """
    clause = SOURCE_SEAM_CLAUSE.get(seam)
    if clause is None or clause(facts):
        return True
    if product_module is None or not _carries_deposit_repair(product_module):
        return False
    return _deposit_is_repairable(seam, configuration)


def _deposit_is_repairable_like_for_like(seam: str, configuration: dict) -> bool:
    """``deposit_repair.repairable``'s refusals, ALL FOUR that are readable from a
    census row -- the clause the Metal and Triton boards ask.

    :func:`_deposit_is_repairable` above asks only the two D-side ones. That is
    sound for the product-aware reading (a cell with no product is refused a rung
    earlier, so the fold clauses never decide anything there), and it is NOT sound
    for a ceiling, where the fold refusals are the difference between "no repair can
    carry this" and "no product carries it yet".

    TWO ARE D-SIDE ONLY. An off-diagonal chi1inv row makes ``update_E`` a stencil
    over the PARTNER components' volumes at shifted indices (stepping.py:1228-1251),
    so no repair over the deposit points can reconstruct it; an instantaneous
    chi2/chi3 is not the linear accumulation the repair inverts.

    TWO ARE ABOUT THE FOLD and apply to either seam: the cylindrical r = 0 axis,
    whose below-axis ghost is the r_to_minus_r image rather than the near fill's
    cell 0 <- cell 2, and a folded axis storing no more than
    :data:`NEAR_SOURCE_INDEX` cells, where the row the near fill images does not
    exist.

    MEASURED ON THIS CORPUS (2026-09-02): the two fold clauses refuse NOTHING the
    two D-side clauses do not already refuse -- 24 blocked either way, per seam as
    well as in total. They are carried anyway, because a clause that decides nothing
    on today's corpus is exactly the clause a later corpus needs and nobody
    remembers to add.

    The remaining refusal -- ``f_w_<component>`` unallocated -- needs an engine
    object, so every count this admits is an UPPER BOUND, which is what this board
    says about itself as a whole.
    """
    if seam == "D_to_E":
        if bool(configuration.get("has_offdiagonal_epsilon")):
            return False
        if bool(configuration.get("has_nonlinearity")):
            return False
    if bool(configuration.get("has_symmetry")):
        if any(bool(flag) for flag in configuration["is_axis"]):
            return False
        for axis in range(3):
            if configuration["mirrored"][axis] and \
                    int(configuration["shape"][axis]) <= NEAR_SOURCE_INDEX:
                return False
    return True


def clears_source_seam_like_for_like(seam: str, facts: dict,
                                     configuration: dict) -> bool:
    """THE RECONCILED SOURCE-SEAM RULE (release decision R3), identical on all three
    boards: does the DEPOSIT ITSELF put this instance beyond any fused product?

    THE COUNTING-RULE DELTA, AND WHY IT IS ONE. Until 2026-09-02 this board answered
    that question with :func:`clears_source_seam`, which additionally requires that a
    product ALREADY OCCUPY the cell and declare ``CARRIES_DEPOSIT_REPAIR``. The
    sibling boards ask only ``deposit_repair.repairable``. Both are defensible
    questions and they are DIFFERENT questions, so their answers -- 45 blocked here
    against 24 there -- were never a capability difference between the backends, and
    summing or comparing the three ceilings was reading three different measurements
    as one. The 2026-09-02 audit established that; this function is the like-for-like
    rule it called for.

    WHAT IT COSTS TO GET IT WRONG IN EITHER DIRECTION. Reading the product-aware
    number as a ceiling says an instance can NEVER be served when what is true is
    that nobody has built its product yet -- it prices future work as impossible.
    Reading the seam-level number as today's reach says the opposite. So both are
    reported: this one bounds what any product could ever do at the seam, and
    :func:`clears_source_seam` bounds what the products that exist do now.

    Product occupancy is a COVERAGE fact and under the reconciled taxonomy it belongs
    to ``served`` and ``buildable_not_built``, never to the ceiling.
    """
    clause = SOURCE_SEAM_CLAUSE.get(seam)
    if clause is None or clause(facts):
        return True
    return _deposit_is_repairable_like_for_like(seam, configuration)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# THE BINDING WALK THIS BOARD DID NOT HAVE
# ---------------------------------------------------------------------------
#
# WHAT IT CLOSES, MEASURED. The Triton board runs fifty-two binding checks and
# ``build_fusion_matrix._assert_every_credit_is_bound_to_released_bytes`` refuses to
# credit a product whose released bytes have moved; this board ran ZERO.
#
# RE-MEASURED 2026-09-11 BY RUNNING THIS WALK, which is the only number that should
# be quoted here. Against ``results/fusion_matrix_cuda_2026-09-09_close``, whose
# ``headline.served_by_product`` carries 532 instances over 32 columns (the seven
# H->D columns INCLUDED at the same counts the ``h_to_d_seam`` block re-reports):
# **144 of the 532 credited seam-instances sit on 8 columns** no registry binds —
# the fused electric pair's 82, the complex pairs' 17 + 17, the two three-slot welds'
# 14 + 6, and the H->D share's complex-beta 4, special_kz 2, conductive/BFAST 2.
# Of the 24 columns that ARE bound, 18 (341 instances) carry a ledger weld and 6
# (47 instances) are bound only at the weaker GATE tier — an artifact whose legs
# still hash to the tree, with no entry in the ledger the weld contract walks.
#
# TWO EARLIER NUMBERS HERE WERE WRONG, EACH IN ITS OWN WAY, and both are recorded
# rather than quietly replaced because each was published somewhere a reader may
# still be holding. "218 of the 705" added the 173 H->D instances to a 532 that
# already contained them and assembled its numerator by hand from a column list.
# "191 of the 532 on 17 of 38" fixed the denominator but was produced while the
# certification half of this walk was VACUOUS: it looked a block up by the board
# column name, and every block key in that file is DATED, so the lookup matched
# nothing and six columns with a clean artifact were reported as binding-less.
# 38 is also the size of the product table rather than of the served set; the
# columns this walk is about are the 32 that are credited. A board crediting a
# product whose evidence it cannot locate publishes a number no reader can
# reconstruct — and so does a board quoting a withheld count from a check that
# could not fail.
#
# IT WITHHOLDS WITH A NAMED REASON AND NEVER RAISES, which is the Triton pattern and
# is load-bearing on a FIRST bound cut: a refusal would make the board un-buildable
# until every column had a weld, which is a campaign away.


def _tree_digest(relative: str) -> Optional[str]:
    """sha256 of a repo-relative file, or None when the checkout does not have it."""
    import hashlib  # noqa: PLC0415

    path = REPO_ROOT / relative
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product_module(column: str) -> str:
    """The repo-relative file a board column's product lives in.

    Read off ``fused_pairs.FUSED_PRODUCTS`` — the composer's own declaration —
    rather than derived from the column name, because the two differ (the column
    ``cuda_three_slot_no_pml_dispersive_weld`` lives in
    ``no_pml_three_slot_dispersive_weld.py``) and a derived path would silently find
    no ledger entry and report a clean product as unbound.
    """
    from meep_gpu.cuda_kernels import fused_pairs as _fp  # noqa: PLC0415

    product = _fp.FUSED_PRODUCTS.get(column)
    name = product["module"] if product else column
    return f"meep_gpu/cuda_kernels/{name}.py"


def _artifact_binding(module: str) -> Optional[Dict[str, Any]]:
    """The SECOND registry's answer for one product module, or ``None``.

    ``certification.json``'s blocks are the narrative record; what they carry that
    is a byte binding is the artifact DIRECTORY, whose released policy legs each
    stamp ``imported_source_sha256``. This walks the blocks that declare this
    module, takes the imports every leg agrees on, curates them to the same key set
    ``rebind_cuda_welds`` would curate for an entry it is creating — the module, the
    shared compile seam and harness, the gate script — and rehashes each against
    the checkout.

    THE HELPERS ARE THE REBIND TOOL'S OWN, imported rather than re-implemented. Two
    readers of one artifact layout that drift apart is how a board starts crediting
    a product the tool that binds it refuses: this board's verdict and that tool's
    refusal must be the same question asked twice, not two questions that usually
    agree. The import is function-local so a board cut on a tree without the tool
    degrades to the ledger tier with a named reason rather than failing to load.

    Returns ``{"ok": bool, "block": key, "records": dir, "drifted": [...],
    "checked": n}`` for the FIRST block that declares the module and can be read,
    preferring a clean one so a superseded first cut cannot withhold a product its
    re-gate bound. ``None`` means no block declares this module at all.
    """
    try:
        import rebind_cuda_welds as _rebind  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - the board still cuts, one tier lower
        return {"ok": False, "block": None, "records": None, "checked": 0,
                "drifted": [],
                "unreadable": f"rebind_cuda_welds is not importable here: {exc!r}"}

    import json as _json  # noqa: PLC0415

    try:
        record = _json.loads(
            (REPO_ROOT / "meep_gpu" / "cuda_kernels" / "certification.json").read_text(
                encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    leaf = module.split("/")[-1]
    answers: List[Dict[str, Any]] = []
    for name, block in sorted(_rebind.device_blocks(record).items()):
        declared = block.get("kernel_module") or record.get("kernel_module") or ""
        modules = [declared] if isinstance(declared, str) else list(declared)
        if leaf not in {str(one).split("/")[-1] for one in modules}:
            continue
        directory = _rebind.artifact_directory(block)
        if directory is None or not directory.is_dir():
            answers.append({"ok": False, "block": name, "records": None,
                            "checked": 0, "drifted": [],
                            "why": "the block names no readable artifact directory"})
            continue
        legs = _rebind.canonical_legs(directory)
        imports = (_rebind.agreed_imports(_rebind.leg_payloads(directory, legs))
                   if legs else {})
        curated = [key for key in _rebind.seed_key_set(record, name, block, imports)
                   if key in imports]
        if not curated:
            answers.append({
                "ok": False, "block": name, "checked": 0, "drifted": [],
                "records": directory.relative_to(REPO_ROOT).as_posix(),
                "why": ("the legs record no agreed imported_source_sha256, so the "
                        "artifact names no bytes to rehash")})
            continue
        moved = sorted(key for key in curated
                       if _tree_digest(key) not in (None, imports[key]))
        answers.append({
            "ok": not moved, "block": name, "checked": len(curated),
            "drifted": moved,
            "records": directory.relative_to(REPO_ROOT).as_posix()})
    if not answers:
        return None
    clean = [one for one in answers if one["ok"]]
    return clean[0] if clean else answers[0]


def release_binding_report(products: Sequence[str]) -> Dict[str, Any]:
    """Per served product: which record binds its bytes, and does that record hold?

    TWO REGISTRIES ARE WALKED, because this track has two and they pin different
    things. ``cuda_kernels/fingerprints.json`` is the WELD ledger — one entry per
    released campaign, pinning the source it executed; ``certification.json`` is the
    NARRATIVE record, whose blocks name an artifact DIRECTORY, and it is the legs in
    that directory that carry ``imported_source_sha256`` — the bytes the gate
    actually imported. A product bound by neither is reported as UNBOUND; a product
    whose ledger entry or whose artifact legs name a file the tree has moved is
    reported as DRIFTED. Both are WITHHELD states, and both carry the list of files
    that moved rather than a bare verdict.

    THE SECOND REGISTRY IS JOINED ON ``kernel_module``, NOT ON THE COLUMN NAME, and
    that is the whole reason this walk finds anything. Every block key in
    ``certification.json`` is DATED (``cuda_conductive_fused_electric_pair_2026-09-02b``)
    while a board column is not, so a ``blocks.get(column)`` lookup matches nothing
    and reports "no certification block binds this" about products that have one —
    a check that cannot fail is not a check. The block declares which file it
    certified; joining on that, and curating the legs' agreed imports the way
    ``rebind_cuda_welds.seed_key_set`` curates them for an entry it is creating,
    asks the same question of both registries: do the bytes this evidence names
    still hash to what the checkout holds?

    THE DISPATCH BINDING IS THE SECOND HALF. The CUDA ``driver_dispatch`` block pins
    the seam (``fastpath.py``, ``driver.py``, ``fields.py``, ``fastpath_cuda.py``,
    the composer and every released arm's module); a drift there means the record
    describes a seam the tree no longer has, and ``served_in_dispatch`` is reported
    withheld with that reason rather than published.
    """
    import json as _json  # noqa: PLC0415

    package = REPO_ROOT / "meep_gpu" / "cuda_kernels"
    try:
        ledger = _json.loads((package / "fingerprints.json").read_text(
            encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        ledger = {"_unreadable": repr(exc)}
    try:
        blocks = _json.loads((package / "certification.json").read_text(
            encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        blocks = {"_unreadable": repr(exc)}

    # WHICH LEDGER ENTRY BINDS WHICH PRODUCT, joined on the entry's own
    # ``kernel_module`` rather than on a name: the entry declares which file it
    # certified, and a name match would credit ``cuda_fused_hd_pair`` to
    # ``cuda_fused_magnetic_pair``'s weld on a shared prefix.
    by_module: Dict[str, List[str]] = {}
    for key, entry in ledger.items():
        if not isinstance(entry, dict):
            continue
        for module in entry.get("kernel_module") or ():
            by_module.setdefault(str(module), []).append(key)

    bound: Dict[str, Any] = {}
    withheld: Dict[str, str] = {}
    for column in sorted(set(products)):
        module = _product_module(column)
        keys = sorted(by_module.get(module, ()))
        if not keys:
            # NO WELD — SO ASK THE OTHER REGISTRY, which on this track pins files the
            # ledger pins zero times. A block whose artifact legs still hash to the
            # checkout is a byte binding: the gate imported those bytes and released
            # on them, and the only thing the missing ledger entry costs is the
            # entry. It is reported as the WEAKER tier ("gate") rather than promoted
            # to a weld, because nothing has recomputed it into the ledger the weld
            # contract walks; the campaign's ``rebind_cuda_welds --seed`` is what
            # moves it up a tier.
            artifact = _artifact_binding(module)
            if artifact is None:
                withheld[column] = (
                    f"no ledger entry and no certification block declares {module}; "
                    "this board can locate no released bytes for the product it is "
                    "crediting")
            elif artifact.get("ok"):
                bound[column] = {"kind": "gate", "block": artifact["block"],
                                 "records": artifact["records"],
                                 "pins_checked": artifact["checked"],
                                 "module": module,
                                 "why_not_a_weld": (
                                     "cuda_kernels/fingerprints.json binds no entry "
                                     "to this module; the artifact's own legs are "
                                     "what hold, and seeding the entry is what moves "
                                     "this to the weld tier")}
            elif artifact.get("drifted"):
                withheld[column] = (
                    f"{artifact['block']} is the only binding and its legs pin "
                    f"{len(artifact['drifted'])} file(s) the tree has moved: "
                    f"{artifact['drifted']}; the credit would be for bytes no gate "
                    "ran")
            else:
                withheld[column] = (
                    f"{artifact['block']} declares {module} but "
                    f"{artifact.get('why') or artifact.get('unreadable')}, and "
                    "cuda_kernels/fingerprints.json binds no entry to it either")
            continue
        drifted: List[str] = []
        for key in keys:
            pinned = (ledger[key].get("source_sha256") or {})
            for name, digest in pinned.items():
                live = _tree_digest(name)
                if live is not None and live != digest and name not in drifted:
                    drifted.append(name)
        if drifted:
            withheld[column] = (
                f"{', '.join(keys)} pins {len(drifted)} file(s) the tree has moved: "
                f"{sorted(drifted)}; the credit would be for bytes no gate ran")
            continue
        bound[column] = {"kind": "fingerprint", "keys": keys, "module": module}

    dispatch = ledger.get("driver_dispatch") if isinstance(ledger, dict) else None
    dispatch_block: Dict[str, Any]
    if not isinstance(dispatch, dict):
        dispatch_block = {
            "bound": False,
            "why": ("cuda_kernels/fingerprints.json carries no driver_dispatch "
                    "entry, so no served_in_dispatch number this board prints is "
                    "bound to a run; cut it with recut_driver_dispatch_record.py "
                    "--backend cuda FROM the campaign's legs"),
        }
    else:
        moved = sorted(name for name, digest
                       in (dispatch.get("source_sha256") or {}).items()
                       if _tree_digest(name) not in (None, digest))
        dispatch_block = {
            "bound": not moved,
            "drifted": moved,
            "why": ("the dispatch record binds the seam the number describes"
                    if not moved else
                    f"the dispatch record pins {moved}, which the tree has moved; "
                    "served_in_dispatch is withheld until the campaign is re-run "
                    "and the record re-cut"),
        }
    return {
        "what_this_is": (
            "per served product, WHICH record binds its bytes and whether that "
            "record still holds against this checkout. The Triton board has run "
            "this walk since 2026-09-02; this board ran zero binding checks until "
            "2026-09-11, which is why the withheld list below is long on its first "
            "cut rather than a regression. TWO TIERS OF BOUND: `fingerprint` is a "
            "ledger weld, `gate` is an artifact whose released legs still hash to "
            "the tree but which the ledger carries no entry for — evidence, one "
            "tier weaker, and `rebind_cuda_welds --seed` is what promotes it."),
        "bound": bound,
        "credit_withheld_stale_binding": withheld,
        "products_bound": len(bound),
        "products_withheld": len(withheld),
        "dispatch_binding": dispatch_block,
        "applied_to_the_served_headline": False,
        "why_not_applied_yet": (
            "THE WALK IS PUBLISHED BEFORE IT IS DEDUCTED, deliberately and for one "
            "cut only. `served` on this board is cross-checked three ways (the "
            "per-cell ledger, the row walk and the taxonomy's own grouping) and "
            "deducting a withheld product silently would make those three disagree "
            "and stop the board. So this cut MEASURES the withheld set and names "
            "it; the campaign that seeds the missing welds is what empties it, and "
            "the deduction lands with the shared `credit_withheld` mechanism "
            "`h_to_d_seam.price` already accepts once the residue is small enough "
            "to read."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write fusion_matrix_cuda.json into")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 78, flush=True)
    print("THE CUDA FUSION BOARD", flush=True)
    print("=" * 78, flush=True)
    print(f"census      : {CENSUS}", flush=True)

    composer = composer_layer()
    print(f"\ncomposer layer: required {list(COMPOSER_MODULES)} in cuda_kernels/ — "
          f"all present", flush=True)

    surface = cuda_surface()
    print(f"CUDA surface  : {len(surface['certified'])} certified, "
          f"{len(surface['uncertified'])} uncertified, read from "
          f"{len(surface['per_module'])} modules by ast", flush=True)

    seams = locate_the_seams()
    print("\n--- THE SEAMS, LOCATED IN driver.py ---------------------------------",
          flush=True)
    for seam, spec in seams.items():
        print(f"  {seam:8s} {spec['curl_call']} -> [{spec['between_summary']}] -> "
              f"{spec['constitutive_call']}", flush=True)

    # ------------------------------------------------- rows, and the composer column
    record, engine_facts = rows()
    pin = subject_pin(record)
    print(f"\n--- {len(record)} ROWS ---------------------------------------------",
          flush=True)
    print(f"  subject pin  : {pin['manifest_sha256'][:16]}… over "
          f"{pin['subject_files']} files — the census's own subject digest, "
          f"RE-COMPUTED against this tree", flush=True)
    print(f"  engine facts : source_field_types and plan_step.live read from THESE "
          f"rows; no join with the Metal census", flush=True)

    # THE COMPOSER'S OWN LABELS, CROSS-CHECKED AGAINST THE ARM TABLE IN THIS TREE.
    # The census recorded selected_kernel; registry.py declares the same string for
    # the same (family, slot). If they disagree the census was cut against a
    # different registry and its selections do not describe the package whose device
    # code the fitness verdict is measured from.
    arms = registry_arms()
    drift: Dict[str, str] = {}
    unregistered: Dict[str, str] = {}
    for row in record:
        plan = row.get("plan_step") or {}
        for slot, family in (plan.get("selected_family") or {}).items():
            arm_label = (plan.get("selected") or {}).get(slot)
            entry = arms.get((family, arm_label, slot))
            recorded = (plan.get("selected_kernel") or {}).get(slot)
            if entry is None:
                unregistered[f"{family}/{arm_label}@{slot}"] = (
                    f"{row['leg']}:{row['row']}")
            elif entry["kernel"] != recorded:
                drift[f"{family}/{arm_label}@{slot}"] = (
                    f"census says {recorded!r}, registry.py says "
                    f"{entry['kernel']!r}")
    if unregistered or drift:
        raise SystemExit(
            f"the census's selection and this tree's arm table disagree. "
            f"{len(unregistered)} (family, arm, slot) arms the census selected are not "
            f"in registry.py's _TABLE ({dict(list(unregistered.items())[:3])}); "
            f"{len(drift)} name a different kernel ({dict(list(drift.items())[:3])}). "
            f"The cells below are that census's selection and the fitness verdict is "
            f"measured from THIS tree — they must describe the same composer.")
    print(f"  arm table    : {len(arms)} (family, arm, slot) arms read from "
          f"registry.py by ast; every selected arm matches its recorded kernel "
          f"label", flush=True)

    # ------------------------------------------------- what the composer SELECTED
    print("\n--- WHAT THE COMPOSER SELECTS PER SLOT (the cell) -------------------",
          flush=True)
    slot_pick: Dict[str, Dict[str, Optional[dict]]] = {}
    slot_refusal: Dict[str, Dict[str, dict]] = {}
    for row in record:
        label = f"{row['leg']}:{row['row']}"
        picked: Dict[str, Optional[dict]] = {}
        refused: Dict[str, dict] = {}
        for slot in SLOTS:
            chosen = selection(row, slot)
            picked[slot] = chosen
            if chosen is None:
                refused[slot] = refusal(row, slot)
        slot_pick[label] = picked
        slot_refusal[label] = refused
    for slot in SLOTS:
        counts = collections.Counter(
            (v[slot]["family"] if v[slot] else "NO ARM SELECTED")
            for v in slot_pick.values())
        print(f"  {slot:9s} " + "  ".join(f"{k}={n}" for k, n in counts.most_common()),
              flush=True)

    # THE REFUSAL CENSUS REPLACES THE DISJOINTNESS PROOF. With no composer a cell was
    # only well defined if at most one family admitted a slot, so this board measured
    # that and raised on an overlap. The composer is fail-closed — arms.py refuses an
    # ambiguous slot rather than picking by table order — so an overlap now shows up
    # as a REFUSAL of kind `ambiguous` instead of a silently-arbitrary cell. It is
    # still fatal at a slot a seam needs, for the same reason it always was.
    print("\n--- THE COMPOSER'S REFUSALS (its own words; the old disjointness proof "
          "is now this) ---", flush=True)
    refusal_kinds: Dict[str, collections.Counter] = {
        slot: collections.Counter() for slot in SLOTS}
    ambiguous: List[str] = []
    for label, refused in slot_refusal.items():
        for slot, entry in refused.items():
            refusal_kinds[slot][entry["kind"]] += 1
            if entry["kind"] == "ambiguous":
                ambiguous.append(f"{label} @{slot}: {entry['admitters']}")
    for slot in SLOTS:
        if not refusal_kinds[slot]:
            continue
        print(f"  {slot:9s} " + "  ".join(
            f"{k}={n}" for k, n in refusal_kinds[slot].most_common()), flush=True)
        example = next((entry for refused in slot_refusal.values()
                        for name, entry in refused.items() if name == slot), None)
        if example and example["reasons"]:
            print(f"            \"{example['reasons'][0]}\"", flush=True)
    if ambiguous:
        raise SystemExit(
            f"the composer refused {len(ambiguous)} slot(s) as AMBIGUOUS "
            f"({ambiguous[:3]}). Two arms admit the same configuration and arms.py "
            f"declines to pick by table order, so there is no cell to report. "
            f"Settle the arms' disjointness before the board is used.")
    print("  no slot is ambiguous: where the composer refuses, it is a named engine "
          "fact, not an undefined cell", flush=True)

    # ------------------------------------------------- the per-kernel fitness
    # KEYED ON THE SELECTED KERNEL, NOT ON THE FAMILY. A family may ship a different
    # body per slot: cuda_nonlinear plans update_E_pml_real_nonlinear on the electric
    # side and reuses the ordinary update_H_pml_real on the magnetic one, and a
    # family-keyed verdict would give one of the two the other's read pattern.
    print("\n--- THE FITNESS VERDICT, MEASURED FROM THE DEVICE CODE --------------",
          flush=True)
    print("  POINTWISE = the constitutive half reads the curl's in-place output ONLY",
          flush=True)
    print("  at the cell this thread owns, so a weld is a register hand-off.",
          flush=True)
    print("  STENCIL   = it reads a neighbour another block writes in the same "
          "launch, and", flush=True)
    print("  CUDA has no grid-wide barrier inside an ordinary launch.", flush=True)
    controls = run_detector_controls()
    print("  DETECTOR CONTROLS (the run refuses if one does not reproduce):",
          flush=True)
    for control in controls:
        print(f"    {control['module']:30s} expected {control['expected']:9s} "
              f"measured {control['measured']:9s} OK", flush=True)
    declared = declared_kernel_modules()
    constitutive_slots = ("update_H", "update_E", "update_P")
    selected_labels: Dict[str, set] = collections.defaultdict(set)
    for label, picked in slot_pick.items():
        for slot in constitutive_slots:
            if picked[slot]:
                selected_labels[picked[slot]["kernel"]].add(picked[slot]["family"])
    fitness: Dict[str, dict] = {}
    for kernel in sorted(selected_labels):
        if kernel in NULL_PLAN_LABELS:
            fitness[kernel] = {
                "verdict": "NO SECOND KERNEL",
                "why": ("the no-PML plan performs no device work at this slot; the "
                        "sub-step launches nothing, so there is no constitutive half "
                        "to weld the curl to"),
                "reads_off_this_thread_cell": []}
        else:
            spec = fitness_source(kernel, declared)
            if spec is None:
                fitness[kernel] = {
                    "verdict": "UNDETERMINED",
                    "why": (f"no module in cuda_kernels/ declares an extern \"C\" "
                            f"__global__ named {kernel!r} (or that name without its "
                            f"`fused_` wrapper prefix), and no explicit source is "
                            f"declared for it in FITNESS_SOURCE_BY_LABEL. Its body is "
                            f"emitted through a sibling module under substitution and "
                            f"this board will not guess which one; the cells it "
                            f"appears in are reported UNDETERMINED rather than "
                            f"priced"),
                    "reads_off_this_thread_cell": []}
            else:
                fitness[kernel] = analyze_kernel_reads(spec)
                fitness[kernel]["source"] = spec
        fitness[kernel]["families"] = sorted(selected_labels[kernel])
        verdict = fitness[kernel]["verdict"]
        detail = fitness[kernel].get("reads_off_this_thread_cell") or []
        print(f"    {kernel:48s} {verdict}   "
              f"{','.join(fitness[kernel]['families'])}", flush=True)
        if fitness[kernel].get("body_at"):
            print(f"        body {fitness[kernel]['body_at']}   flat index "
                  f"{fitness[kernel].get('flat_index')!r}, axis indices "
                  f"{fitness[kernel].get('axis_indices')}", flush=True)
        for item in detail[:4]:
            print(f"        EVIDENCE {item['read']}  — {item['why']}", flush=True)
        if len(detail) > 4:
            print(f"        ... and {len(detail) - 4} more reads off this thread's "
                  f"cell", flush=True)
        if verdict == "UNDETERMINED":
            print(f"        {fitness[kernel]['why']}", flush=True)

    # THE E->P SEAM EXISTS ONLY WHERE A POLARIZATION DOES.
    has_polarization = {f"{r['leg']}:{r['row']}":
                        int(r["configuration"]["n_polarizations"]) > 0
                        for r in record}
    #: The census configuration block per row, so the deposit clause can ask
    #: `deposit_repair.repairable`'s own refusals without a second walk.
    # The lift facts ride with the configuration (2026-09-13): dispatch_reachability
    # derives `dimensions` by MEEP's rule from them, not from the extent count.
    configuration_of = {f"{r['leg']}:{r['row']}": dict(r["configuration"], facts=r.get("facts") or {})
                        for r in record}
    n_polarized = sum(1 for v in has_polarization.values() if v)

    seam_instances = 0
    per_row: List[dict] = []
    by_cell: Dict[Tuple[str, Optional[str], Optional[str], Optional[str]],
                  List[str]] = collections.defaultdict(list)
    for row in record:
        label = f"{row['leg']}:{row['row']}"
        picked = slot_pick[label]
        entry = {"row": label, "seams": {}}
        for seam, (curl_slot, const_slot) in SEAM_SLOTS.items():
            if seam == "E_to_P" and not has_polarization[label]:
                continue
            seam_instances += 1
            curl_arm, const_arm = picked[curl_slot], picked[const_slot]
            cell = (seam,
                    f"{curl_arm['family']}/{curl_arm['arm']}" if curl_arm else None,
                    f"{const_arm['family']}/{const_arm['arm']}" if const_arm else None,
                    curl_arm["kernel"] if curl_arm else None,
                    const_arm["kernel"] if const_arm else None)
            by_cell[cell].append(label)
            live = tuple(name for name in IN_SEAM_PASSES.get(seam, ())
                         if name in engine_facts[label]["live_driver_passes"])
            entry["seams"][seam] = {
                "curl_arm": curl_arm, "constitutive_arm": const_arm,
                "curl_refusal": slot_refusal[label].get(curl_slot),
                "constitutive_refusal": slot_refusal[label].get(const_slot),
                "live_in_seam_passes": list(live),
            }
        per_row.append(entry)

    # ------------------------------------------- THE FOURTH SEAM, update_H -> step_D
    # One instance per row (2026-09-04). The two halves are the composer's own picks
    # this walk already read — update_H is B_to_H's constitutive half, step_D is
    # D_to_E's curl half — and the null verdict is the NULL_PLAN_LABELS test the
    # fitness ledger applies, asked of the FIRST half here. Priced by
    # h_to_d_seam.price below from the probe artifact (the withdraw is measured per
    # lifted row, not read off a name). `seam_instances` stays the three-seam
    # ledger's own denominator; the board's PRICED denominator is `priced`.
    def _arm_label(arm: Optional[dict]) -> Optional[str]:
        return f"{arm['family']}/{arm['arm']}" if arm else None

    h_to_d_entries = [
        {"row": e["row"],
         "update_H": _arm_label(e["seams"]["B_to_H"]["constitutive_arm"]),
         "step_D": _arm_label(e["seams"]["D_to_E"]["curl_arm"]),
         "update_H_is_null": bool(
             e["seams"]["B_to_H"]["constitutive_arm"]
             and e["seams"]["B_to_H"]["constitutive_arm"]["kernel"] in NULL_PLAN_LABELS)}
        for e in per_row]
    priced = seam_instances + len(h_to_d_entries)

    print(f"\n  denominator: {len(record)} rows x 2 curl->constitutive seams + "
          f"{n_polarized} rows with a susceptibility x 1 E->P seam = "
          f"{seam_instances} seam-instances on the three-seam ledger; + {len(record)} "
          f"rows x 1 H_to_D seam = {priced} PRICED", flush=True)

    # ------------------------------------------------- the source-seam ceiling
    kinds = collections.Counter(engine_facts[f"{r['leg']}:{r['row']}"]
                                ["source_field_types"] for r in record)
    def _row_clears(entry: dict, seam: str) -> bool:
        """One seam-instance's source verdict, product-aware. ONE definition, three
        readers -- the ceiling, the per-cell reach and the ledger -- because the
        ledger RAISES if its walk and the ceiling disagree, and two spellings of the
        same clause is how they would come to."""
        block = entry["seams"][seam]
        curl_arm, const_arm = block["curl_arm"], block["constitutive_arm"]
        return clears_source_seam(
            seam, engine_facts[entry["row"]], configuration_of[entry["row"]],
            product_on_cell(
                seam,
                f"{curl_arm['family']}/{curl_arm['arm']}" if curl_arm else None,
                f"{const_arm['family']}/{const_arm['arm']}" if const_arm else None))

    # ------------------------------------------------- WHAT A SHIPPED PRODUCT SERVES
    #
    # THE COLUMN IS THE SHIPPED PREDICATE'S OWN ANSWER, RECORDED PER ROW. It is read
    # here and not re-derived: ``covers_fused_magnetic_pair`` decides what one launch
    # of ``fused_magnetic_pair_pml_real`` may span, and a board that restated the
    # clauses could admit a row the launch cannot serve -- which is a silent wrong
    # answer, the worst outcome this board has available.
    #
    # THIS COLUMN READ ZERO UNTIL 2026-08-28, TWICE OVER AND FOR TWO DIFFERENT
    # REASONS. Until the product existed there was nothing to count. Then the product
    # existed but imported CuPy at module scope, so the census could not ASK it on a
    # NumPy host and recorded ``askable: false`` -- and this board still printed a
    # hard-coded 0 with a note saying no module emits such a kernel, which had stopped
    # being true. Both are now closed: the module takes CuPy defensively, the census
    # asks the predicate on every row, and the number below is that answer counted.
    #
    # ``covered_modulo_backend`` IS THE RIGHT COLUMN and the raw one is carried
    # beside it. Every CUDA predicate's first clause is "is the array module CuPy",
    # which a NumPy census host fails by construction; the composer cells above are
    # read through the same shim, so mixing the two readings would price the seam
    # against a different question than the cells.
    def _fused_verdicts() -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for column in FUSED_PRODUCT_COLUMNS:
            per_row_block: Dict[str, Dict[str, Any]] = {}
            for row in record:
                label = f"{row['leg']}:{row['row']}"
                block = row.get(column)
                if not isinstance(block, dict):
                    raise SystemExit(
                        f"the census carries no {column!r} column on {label}; this "
                        f"board reports what each shipped fused product serves and "
                        f"cannot infer that from a census that never asked")
                if not block.get("askable"):
                    raise SystemExit(
                        f"the census could not ask {column!r} on {label}: "
                        f"{block.get('unaskable_reason')}. A board that reported 0 "
                        f"served here would be reporting IGNORANCE as a measurement")
                per_row_block[label] = block
            out[column] = per_row_block
        return out

    fused_verdicts = _fused_verdicts()

    def _seams_of_span(column: str, span: Tuple[str, ...]) -> Tuple[str, ...]:
        """Every seam a declared span covers, in ``SPANNABLE_SEAM_SLOTS`` order.

        A TWO-SLOT SPAN IS EXACTLY ONE SEAM and that was the only shape this board
        had until 2026-09-02. A THREE-SLOT span -- ``step_D``/``update_E``/
        ``update_P`` -- covers TWO, because ``update_E`` is the second consult of
        ``D_to_E`` and the first of ``E_to_P``. It is read as CONSECUTIVE PAIRS of
        the declared span rather than as "every seam whose slots are a subset",
        because the seams are ordered runs of the driver and a subset test would
        credit a span that skipped a consult in the middle.

        A span matching no seam is a refusal by name. Silently crediting it to
        nothing would drop a shipped product off the board with no line saying so.
        """
        seams = []
        for index in range(len(span) - 1):
            pair = (span[index], span[index + 1])
            match = [seam for seam, slots in SPANNABLE_SEAM_SLOTS.items()
                     if slots == pair]
            if not match:
                raise SystemExit(
                    f"{column} declares the span {span}, whose consecutive pair "
                    f"{pair} is no seam this board knows "
                    f"({sorted(SPANNABLE_SEAM_SLOTS)}); a "
                    f"product is priced against the seams it spans and this one "
                    f"could be priced against none")
            seams.extend(match)
        return tuple(seams)

    #: Which seamS each shipped fused product occupies, read off the column the census
    #: recorded rather than assumed: the product declares the slots it spans. A TUPLE
    #: since 2026-09-02, when the first THREE-slot products landed -- one per seam was
    #: right for every product that spans one, and would have had to pick one of two
    #: for a product that serves both, which is precisely the choice those products
    #: exist to remove.
    fused_seams_of = {
        column: _seams_of_span(column,
                               tuple(next(iter(blocks.values()))["spans"]))
        for column, blocks in fused_verdicts.items()}
    #: Every seam a shipped product occupies. ONE UNTIL 2026-08-30 and two since, so
    #: the overlap check below walks this set rather than the first column's seam --
    #: a check keyed on ``FUSED_PRODUCT_COLUMNS[0]`` would have stopped looking at the
    #: electric seam the moment a product landed on it, which is exactly when it
    #: started needing to be looked at.
    fused_seams = tuple(sorted({seam for seams in fused_seams_of.values()
                                for seam in seams}))

    def _serving_products(label: str, seam: str) -> Tuple[str, ...]:
        """Which shipped products admit this row at this seam. USUALLY AT MOST ONE.

        The composer refuses a seam TWO products claim rather than picking by table
        order (``fused_pairs.install_fused_pairs``), so an overlap here is not a
        richer answer -- it is a seam that would be left UNFUSED. It is returned as a
        tuple rather than collapsed so the overlap can be counted and raised on.

        WITH ONE EXCEPTION, AND IT IS READ FROM THE COMPOSER RATHER THAN INVENTED
        HERE, 2026-09-02. A three-slot weld's predicate is the CONJUNCTION of a D->E
        predicate and an E->P one, so wherever it admits, the two-slot D->E product
        it supersedes admits too -- two columns, one seam, on every row the weld
        serves. ``fused_pairs._superseded_by_a_longer_span`` refuses the SHORTER span
        by name, so exactly one product installs; this applies the same rule to the
        same declared spans. Without it the board would raise on ten rows that the
        composer resolves cleanly, which is a board reporting a decision it did not
        read as an ambiguity that does not exist.

        STRICT CONTAINMENT, as in the composer: two spans that merely overlap are
        still an overlap, and still raise below.
        """
        admitting = tuple(column for column in FUSED_PRODUCT_COLUMNS
                          if seam in fused_seams_of[column]
                          and bool(fused_verdicts[column][label].get(
                              "covered_modulo_backend")))
        spans = {column: set(next(iter(fused_verdicts[column].values()))["spans"])
                 for column in admitting}
        return tuple(column for column in admitting
                     if not any(spans[column] < spans[other]
                                for other in admitting if other != column))

    #: The two seams that SHARE a slot: ``update_E`` is the SECOND consult of D->E and
    #: the FIRST of E->P. A D->E product and an E->P product that both admit one row
    #: are asking for the same slot and ``install_fused_pairs`` can give it to only
    #: one, so the board has to model that or it counts a launch that cannot happen.
    _SHARED_SLOT_SEAMS: Tuple[Tuple[str, str], ...] = (("D_to_E", "E_to_P"),)

    def _loses_the_shared_slot(label: str, seam: str) -> bool:
        """Does a LATER seam's product take this row's shared slot from ``seam``?

        THE BOARD DID NOT ASK THIS UNTIL 2026-09-02 AND WAS OVER-COUNTING BY IT.
        ``_serving_products`` is per seam, and the overlap check above is per seam
        too, so a row admitted by a D->E product AND an E->P product was credited
        TWICE -- once at each seam -- while the composer installs exactly one.

        MEASURED WHEN THIS WAS ADDED, on the stamped census:

          cuda_dispersive_fused_electric_pair            7 rows
          cuda_fused_polarization_pair                   the SAME 7
          cuda_no_pml_complex_fused_electric_pair        4 rows
          cuda_complex_no_pml_fused_polarization_pair    the SAME 4
          cuda_no_pml_dispersive_fused_electric_pair     3 rows (landed this round)
          cuda_no_pml_fused_polarization_pair            the SAME 3

        Fourteen seam-instances of apparent coverage that are one-for-one trades.

        WHICH SIDE LOSES IS READ FROM THE COMPOSER, NOT DECIDED HERE.
        ``fused_pairs._later_seam_claimant`` leaves the shared slot with the product
        that already serves it -- the E->P one -- so the D->E side is what this
        withholds. A board that picked its own winner would be re-deriving a composer
        decision instead of reading it.

        A PRODUCT THAT SERVES BOTH SEAMS ITSELF MAKES NO TRADE, 2026-09-02, and that
        is the whole point of the three-slot welds. This asks whether any product
        serving the LATER seam is one that ALSO serves the earlier one; if the same
        column is on both sides there is nothing to withhold, and withholding anyway
        would credit such a product at one seam instead of two -- reporting exactly
        the collision it was built to remove. Read off ``fused_seams_of``, which is
        the column's own declared span, so this cannot drift from what the composer
        installs.
        """
        for earlier, later in _SHARED_SLOT_SEAMS:
            if seam != earlier:
                continue
            claimants = _serving_products(label, later)
            if any(earlier not in fused_seams_of[column] for column in claimants):
                return True
        return False

    def _served(label: str, seam: str) -> bool:
        return bool(_serving_products(label, seam)) and not _loses_the_shared_slot(
            label, seam)

    # TWO PRODUCTS ADMITTING ONE ROW AT ONE SEAM IS A REFUSAL, NOT A GAIN, and it is
    # checked here rather than assumed from the predicates being disjoint by design:
    # the real pair refuses complex64 storage, the complex pair refuses a Dcyl grid
    # and a real run, and the cylindrical pair REQUIRES Dcyl at |m| >= 1 -- but a
    # board that trusted that would report served counts the composer would never
    # realise.
    overlaps = {f"{label}@{seam}": products
                for label in fused_verdicts[FUSED_PRODUCT_COLUMNS[0]]
                for seam in fused_seams
                if len(products := _serving_products(label, seam)) > 1}
    if overlaps:
        raise SystemExit(
            f"{len(overlaps)} rows are admitted by MORE THAN ONE fused product at "
            f"one seam: {dict(list(overlaps.items())[:5])}. install_fused_pairs "
            f"leaves such a seam UNFUSED naming both, so counting either as served "
            f"would be reporting a launch that cannot happen")

    entry_of = {e["row"]: e for e in per_row}
    ceiling: Dict[str, int] = {}
    driver_ceiling: Dict[str, int] = {}
    for seam in SEAM_SLOTS:
        clause = SOURCE_SEAM_CLAUSE.get(seam)
        ceiling[seam] = sum(1 for e in per_row if seam in e["seams"]
                            and _row_clears(e, seam))
        driver_ceiling[seam] = sum(
            1 for e in per_row if seam in e["seams"]
            and (clause is None or clause(engine_facts[e["row"]])))
    print("\n--- THE SOURCE-SEAM CEILING (a driver fact, not a kernel clause) -----",
          flush=True)
    print(f"  source field types across the corpus: "
          f"{ {str(k): v for k, v in kinds.items()} }", flush=True)
    for seam in SEAM_SLOTS:
        instances = sum(1 for e in per_row if seam in e["seams"])
        carried = ceiling[seam] - driver_ceiling[seam]
        print(f"  {seam:8s} at most {ceiling[seam]:3d} / {instances:3d} "
              f"seam-instances can EVER be fused there  "
              f"[{seams[seam]['between_summary'] or 'nothing between the halves'}]",
              flush=True)
        if carried:
            print(f"           of which {carried:3d} clear it ONLY because a shipped "
                  f"product carries the deposit; {driver_ceiling[seam]} clear it on "
                  f"the driver fact alone", flush=True)

    # ------------------------------------------------- the in-seam carry
    print("\n--- WHAT A FUSED LAUNCH WOULD HAVE TO CARRY IN REGISTERS -------------",
          flush=True)
    print("  liveness READ from the SAME census's plan_step.live (an engine fact); "
          "the CUDA", flush=True)
    print("  kernel column is the live CERTIFIED/UNCERTIFIED partition of "
          "in_seam_passes.py.", flush=True)
    carry: Dict[str, dict] = {}
    for seam, names in IN_SEAM_PASSES.items():
        for name in names:
            live = sum(1 for e in per_row if seam in e["seams"]
                       and name in e["seams"][seam]["live_in_seam_passes"])
            kernel = IN_SEAM_CUDA_KERNEL[name]
            state = ("CERTIFIED" if kernel in surface["certified"]
                     else "UNCERTIFIED" if kernel in surface["uncertified"]
                     else "NOT PORTED")
            carry[f"{seam}:{name}"] = {
                "live_seam_instances": live, "cuda_kernel": kernel,
                "cuda_kernel_state": state,
                "module": surface["certified"].get(kernel)
                          or surface["uncertified"].get(kernel)}
            print(f"    {seam:7s} {name:26s} live on {live:4d} seam-instances   "
                  f"CUDA {kernel:20s} {state}", flush=True)

    # ------------------------------------------------- the cells, ranked
    print("\n" + "=" * 78, flush=True)
    print("THE BOARD — every (curl arm, constitutive arm) pair the corpus drives",
          flush=True)
    print("=" * 78, flush=True)
    live_of = {(e["row"], seam): tuple(block["live_in_seam_passes"])
               for e in per_row for seam, block in e["seams"].items()}
    refusals_of = {(e["row"], seam): {
        "curl": block["curl_refusal"], "constitutive": block["constitutive_refusal"]}
        for e in per_row for seam, block in e["seams"].items()}
    cells: List[dict] = []
    for (seam, curl_arm_key, const_arm_key, curl_kernel,
         const_kernel), labels in by_cell.items():
        reaching = [lab for lab in labels if _row_clears(entry_of[lab], seam)]
        reachable = len(reaching)
        # WHAT THE WELD WOULD HAVE TO CARRY, PER CELL. The reachable count is the
        # market; it is NOT the count of rows a first product could take. A fused
        # launch swallows whatever driver passes sit inside the seam, so a row
        # running two of them needs those two carried in registers before it can be
        # served. The subset running NONE is the cheapest possible first product and
        # is reported beside the market rather than instead of it — a rank on the
        # market alone would point at a weld that has to absorb three passes.
        bare = sum(1 for lab in reaching if not live_of[(lab, seam)])
        needs = collections.Counter()
        for lab in reaching:
            for name in live_of[(lab, seam)]:
                needs[name] += 1
        cell: Dict[str, Any] = {
            "seam": seam, "curl_arm": curl_arm_key,
            "constitutive_arm": const_arm_key,
            "curl_kernel": curl_kernel, "constitutive_kernel": const_kernel,
            "demand_rows": len(labels),
            "demand_rows_clearing_the_source_seam": reachable,
            "reachable_with_nothing_to_carry_in_the_seam": bare,
            "reachable_rows_needing_each_in_seam_pass": dict(needs),
            "example_rows": sorted(labels)[:3],
            "example_rows_with_nothing_to_carry": sorted(
                lab for lab in reaching if not live_of[(lab, seam)])[:3],
            # WHAT A SHIPPED PRODUCT SERVES AT THIS CELL, read off the census's own
            # record of the shipped predicate. Counted over the REACHING rows, so a
            # row the source seam already excluded cannot be claimed here.
            "served_by_a_shipped_product": sum(
                1 for lab in reaching if _served(lab, seam)),
            # THE PRODUCT NAMED HERE IS THE ONE THE CENSUS SAYS ADMITS THESE ROWS,
            # never a guess from the cell's arm labels: a cell is served by whichever
            # shipped predicate answered yes on its reaching rows, and the overlap
            # check above has already refused the case where that is more than one.
            "served_by": next(
                (product for lab in reaching
                 for product in _serving_products(lab, seam)), None),
        }
        if curl_arm_key is None or const_arm_key is None:
            # THE COMPOSER'S OWN WORDS, NOT A DESCRIPTION OF THEM. A cell with an
            # unselected half groups every row the composer refused there, and the
            # refusals need not share a kind, so all of them are carried.
            missing = []
            if curl_arm_key is None:
                missing.append(SEAM_SLOTS[seam][0])
            if const_arm_key is None:
                missing.append(SEAM_SLOTS[seam][1])
            kinds_seen = collections.Counter()
            quoted: List[str] = []
            for lab in labels:
                for side in ("curl", "constitutive"):
                    entry = refusals_of[(lab, seam)][side]
                    if not entry:
                        continue
                    kinds_seen[entry["kind"]] += 1
                    for reason in entry["reasons"]:
                        if reason not in quoted:
                            quoted.append(reason)
            cell["verdict"] = "NO ARM SELECTED"
            cell["refusal_kinds"] = dict(kinds_seen)
            cell["composer_reasons"] = quoted[:6]
            cell["why"] = (
                f"the composer selects no arm at {', '.join(missing)} on these rows, "
                f"so there is no second kernel to weld — the array path steps them. "
                f"This is a COVERAGE gap, not a fusion one, and closing it is a "
                f"sub-step kernel's job. Refusal kinds: {dict(kinds_seen)}")
            cells.append(cell)
            continue
        verdict = fitness.get(const_kernel, {})
        kind = verdict.get("verdict", "UNDETERMINED")
        cell["constitutive_fitness"] = kind
        # WHICH SHIPPED MODULE, IF ANY, OCCUPIES THIS CELL. Read from the one table
        # that maps a cell to a product, the same reading `clears_source_seam` makes
        # for the repair credit, so the verdict and the ceiling cannot disagree
        # about whether there is a product here.
        product_module = product_on_cell(seam, curl_arm_key, const_arm_key)
        cell["fitness_evidence"] = verdict.get("reads_off_this_thread_cell") or []
        cell["fitness_measured_on"] = verdict.get("body_at")
        cell["fitness_source"] = verdict.get("source")
        # THE HALVES ARE THE COMPOSER'S KERNEL LABELS, and the registry's own
        # docstring records that several of them are DESCRIPTIONS rather than entry
        # points — `fused_step_B_(no_)pml_conductive` names two kernels and a choice.
        # A label with no counterpart in the certified/uncertified partition is
        # therefore reported as NOT AN ENTRY POINT, never as a missing kernel.
        halves = [k for k in (curl_kernel, const_kernel) if k]
        entry_points = {}
        for half in halves:
            plain = half[len("fused_"):] if half.startswith("fused_") else half
            for candidate in (half, plain):
                if candidate in surface["certified"]:
                    entry_points[half] = ("CERTIFIED", candidate)
                    break
                if candidate in surface["uncertified"]:
                    entry_points[half] = ("UNCERTIFIED", candidate)
                    break
            else:
                entry_points[half] = ("NOT AN ENTRY POINT IN THIS TREE", None)
        uncertified_half = [h for h, (state, _) in entry_points.items()
                            if state == "UNCERTIFIED"]
        unknown_half = [h for h, (state, _) in entry_points.items()
                        if state.startswith("NOT AN ENTRY POINT")]
        cell["halves"] = halves
        cell["halves_state"] = {h: state for h, (state, _) in entry_points.items()}
        cell["halves_uncertified"] = uncertified_half
        cell["halves_not_in_either_set"] = unknown_half
        if kind == "NO SECOND KERNEL":
            cell["verdict"] = "NOT A FUSION CANDIDATE"
            cell["why"] = verdict["why"]
        elif kind == "STENCIL":
            cell["verdict"] = "STENCIL-BLOCKED"
            cell["why"] = (
                "the constitutive half reads the curl half's in-place output at a "
                "cell this thread does not own. In one launch that cell is written "
                "by another block, and CUDA offers no grid-wide barrier inside an "
                "ordinary launch — only cooperative groups, which caps the grid at "
                "the resident-block count. Not weldable byte-identically as an "
                "ordinary launch. MEASURED from the device code, not inferred from "
                "the family's name")
            # WHERE A PRODUCT OCCUPIES THE CELL, THE BOARD SAYS BOTH FACTS. The
            # fitness verdict above is NOT relaxed and the constitutive kernel is
            # still STENCIL — that is a true reading of its device text, and this
            # board's `fitness_rule` is unchanged. What the sentence overreached on
            # was the CELL: it describes the IN-PLACE weld, whose two premises a
            # SCRATCH-OUTPUT weld removes (the curl half writes a launch-local
            # allocation, so nothing this launch reads is a word this launch wrote;
            # and the foreign sample is RECOMPUTED from pre-launch state through the
            # same __device__ function rather than loaded from another block's
            # output). Splitting the bucket rather than moving the cell into
            # POINTWISE keeps the read-pattern measurement and the weld's existence
            # as two separate, separately checkable facts.
            if product_module is not None:
                cell["verdict"] = "STENCIL-BLOCKED (served by a scratch-output weld)"
                cell["why"] += (
                    f"; and that describes the IN-PLACE weld. {product_module} "
                    f"occupies this cell with a SCRATCH-OUTPUT weld, which has "
                    f"neither premise: D_new/fu_new go to a launch-local allocation "
                    f"the launcher rotates in afterwards, and every foreign sample "
                    f"is recomputed from pre-launch D/fu/H — a pure function of "
                    f"memory no thread writes, so there is no ordering to depend on. "
                    f"The constitutive half's fitness is unchanged and is still "
                    f"STENCIL")
                cell["scratch_output_weld"] = {
                    "module": product_module,
                    "what_the_stencil_verdict_still_says": (
                        "the constitutive kernel reads cells the thread does not "
                        "own — true, unchanged, and measured from its device text"),
                    "what_the_weld_changes": (
                        "where those cells' values come from: a recompute from "
                        "unwritten pre-launch state, not a load of another block's "
                        "in-place output"),
                    "host_arithmetic": (
                        "parity/meep_gpu/results/"
                        "cuda_offdiag_scratch_weld_2026-09-02/probe.json"),
                }
        elif kind == "POINTWISE":
            cell["verdict"] = "POINTWISE-BUILDABLE"
            cell["why"] = (
                "every read the constitutive half makes of the curl half's in-place "
                "output lands on the cell this thread just wrote, so the weld is a "
                "register hand-off and needs no barrier")
            cell["hazard"] = (
                "__restrict__ ALIASING: the curl's output volume and the "
                "constitutive's input volume are THE SAME ALLOCATION. Binding it as "
                "two __restrict__ pointers is undefined behaviour and NVRTC may "
                "miscompile it silently. The fused signature must bind the shared "
                "volume ONCE. complex_emitter.py:509-511 already declines "
                "__restrict__ on the inverse-epsilon parameters for the sibling "
                "reason — an isotropic run hands the same pointer three times")
            if uncertified_half:
                cell["verdict"] = "POINTWISE-BUILDABLE (a half is UNCERTIFIED)"
                cell["why"] += (
                    f"; but {uncertified_half} is in UNCERTIFIED_KERNELS, so a fused "
                    f"product would inherit an ungated half and its gate would be "
                    f"the first byte-identity evidence either kernel has")
        else:
            cell["verdict"] = "UNDETERMINED"
            cell["why"] = verdict.get("why", "the fitness measurement did not resolve")
        cells.append(cell)

    cells.sort(key=lambda c: (-c["demand_rows_clearing_the_source_seam"],
                              -c["demand_rows"], c["seam"], str(c["curl_arm"])))
    for cell in cells:
        print(f"\n  reach {cell['demand_rows_clearing_the_source_seam']:4d}  (of "
              f"{cell['demand_rows']:3d} rows)  {cell['seam']:8s} "
              f"({cell['curl_arm']}, {cell['constitutive_arm']})", flush=True)
        print(f"        carries nothing in the seam on "
              f"{cell['reachable_with_nothing_to_carry_in_the_seam']:4d} of those; "
              f"the rest need "
              f"{cell['reachable_rows_needing_each_in_seam_pass'] or 'nothing'}",
              flush=True)
        print(f"        {cell['verdict']}", flush=True)
        print(f"        {cell['why']}", flush=True)
        for item in (cell.get("fitness_evidence") or [])[:2]:
            print(f"        EVIDENCE {item['read']} — {item['why']}", flush=True)

    # ------------------------------------------------- the ledger
    buckets: Dict[str, collections.Counter] = {
        seam: collections.Counter() for seam in SEAM_SLOTS}
    verdict_of = {(c["seam"], c["curl_arm"], c["constitutive_arm"],
                   c["curl_kernel"], c["constitutive_kernel"]): c["verdict"]
                  for c in cells}
    for entry in per_row:
        for seam, block in entry["seams"].items():
            if not _row_clears(entry, seam):
                buckets[seam]["blocked by the source injection in the seam"] += 1
                continue
            curl_arm, const_arm = block["curl_arm"], block["constitutive_arm"]
            buckets[seam][verdict_of[(
                seam,
                f"{curl_arm['family']}/{curl_arm['arm']}" if curl_arm else None,
                f"{const_arm['family']}/{const_arm['arm']}" if const_arm else None,
                curl_arm["kernel"] if curl_arm else None,
                const_arm["kernel"] if const_arm else None)]] += 1
    print("\n" + "=" * 78, flush=True)
    print("THE LEDGER — every seam-instance that clears its source injection",
          flush=True)
    print("=" * 78, flush=True)
    total_reachable = 0
    for seam in SEAM_SLOTS:
        instances = sum(1 for e in per_row if seam in e["seams"])
        blocked = buckets[seam]["blocked by the source injection in the seam"]
        reach = instances - blocked
        total_reachable += reach
        if reach != ceiling[seam]:
            raise SystemExit(
                f"{seam}: the ledger reaches {reach} but the source-seam clause says "
                f"{ceiling[seam]}; the walk and the clause disagree and no number "
                f"here may be used until they do not")
        print(f"\n  {seam}: {instances} instances, {blocked} blocked by the source "
              f"seam, CEILING {reach}", flush=True)
        accounted = 0
        for bucket, count in sorted(buckets[seam].items(), key=lambda kv: -kv[1]):
            if bucket.startswith("blocked"):
                continue
            accounted += count
            print(f"      {count:4d}  {bucket}", flush=True)
        if accounted != reach:
            raise SystemExit(
                f"{seam}: buckets sum to {accounted}, ceiling is {reach}")

    pointwise = sum(v for seam in buckets for k, v in buckets[seam].items()
                    if k.startswith("POINTWISE-BUILDABLE"))
    stencil = sum(v for seam in buckets for k, v in buckets[seam].items()
                  if k.startswith("STENCIL-BLOCKED"))
    stencil_served = sum(v for seam in buckets for k, v in buckets[seam].items()
                         if k.startswith("STENCIL-BLOCKED (served"))
    no_pair = sum(v for seam in buckets for k, v in buckets[seam].items()
                  if k == "NO ARM SELECTED")
    null_plan = sum(v for seam in buckets for k, v in buckets[seam].items()
                    if k == "NOT A FUSION CANDIDATE")
    undetermined = sum(v for seam in buckets for k, v in buckets[seam].items()
                       if k == "UNDETERMINED")
    # SERVED, TOTALLED THE SAME WAY THE BUCKETS ARE: over the cells' own per-cell
    # counts, which were taken over the REACHING rows. It is a SUBSET of the
    # reachable ceiling and never a bucket beside it -- a served seam-instance is
    # still POINTWISE-BUILDABLE, and adding it to the decomposition would double-count.
    served_by_product: Dict[str, int] = collections.Counter()
    for cell in cells:
        if cell.get("served_by"):
            served_by_product[cell["served_by"]] += cell["served_by_a_shipped_product"]
    served_by_product = dict(served_by_product)
    served_total = sum(served_by_product.values())
    # THE OTHER HALF OF "SERVED": whether the DISPATCHER could select any of it.
    # Zero on CUDA, and the zero is a structural measurement rather than this
    # board's own ``wired: false`` restated — ``fastpath.plan_fast_path`` imports
    # ``triton_kernels`` and nothing else, so no CUDA product is reachable from the
    # driver's consults however complete its predicate coverage or its device
    # gates. Passing the served instances anyway, so the day that changes the
    # number moves on its own — EVERY served instance since 2026-09-06, the H->D
    # rows included, so the measurement's `served_counted` is the served headline
    # and not 0 beside it. The H->D verdict is this board's own, given here (the
    # shared rule cannot evaluate a per-backend predicate; see the block below the
    # taxonomy walk for what the verdict is and is not).
    for entry in h_to_d_entries:
        entry["served_by"] = next(
            iter(_serving_products(entry["row"], h_to_d_seam.SEAM)), None)
    # The seam rides along; see build_fusion_matrix.py's note at the same call.
    dispatch_reach = _reach.served_in_dispatch("cuda", [
        (e["row"], next(iter(_serving_products(e["row"], seam))),
         configuration_of[e["row"]], seam)
        for e in per_row for seam in e["seams"] if _served(e["row"], seam)
    ] + [
        (e["row"], e["served_by"], configuration_of[e["row"]], h_to_d_seam.SEAM)
        for e in h_to_d_entries if e["served_by"]])
    served_cross_check = sum(
        1 for e in per_row for seam in e["seams"]
        if _served(e["row"], seam) and _row_clears(e, seam))
    if served_total != served_cross_check:
        raise SystemExit(
            f"the per-cell served counts total {served_total} but the row walk finds "
            f"{served_cross_check}; two spellings of one question disagree and no "
            f"number here may be used until they do not")
    # THE FLOOR IS THE TWO BUCKETS A PRODUCT MAY OCCUPY, and it was one until
    # 2026-09-02. The old form compared SERVED against POINTWISE alone, on the
    # premise that "a shipped product cannot serve a seam this board prices as
    # unweldable" -- true of the IN-PLACE weld the STENCIL verdict measures, and
    # false of a SCRATCH-OUTPUT weld, which is why the stencil bucket now splits.
    # A served instance in a cell NO product occupies is still an inconsistency
    # and still stops the board, which is the half of the check that was doing the
    # work.
    if served_total > pointwise + stencil_served:
        raise SystemExit(
            f"{served_total} seam-instances are reported SERVED but only "
            f"{pointwise} are POINTWISE-BUILDABLE and {stencil_served} sit in a "
            f"STENCIL cell a scratch-output weld occupies; a shipped product cannot "
            f"serve a seam no product is recorded on, so one of the two is wrong")

    classified = pointwise + stencil + no_pair + null_plan + undetermined
    if classified != total_reachable:
        raise SystemExit(
            f"the verdict buckets total {classified} but {total_reachable} "
            f"seam-instances are reachable; {total_reachable - classified} fall into "
            f"no bucket and would have vanished from the decomposition. Ledger: "
            f"{ {s: dict(c) for s, c in buckets.items()} }")

    # ------------------------------------------------- THE RECONCILED TAXONOMY
    # THE SAME EIGHT NAMES THE METAL AND TRITON BOARDS EMIT (release decision R1; the
    # eighth, withdraw_seam, arrived with the H_to_D seam on 2026-09-04), filed
    # over the SAME instances the ledger above walks. The ledger is kept unchanged
    # and is not replaced: it is this board's own vocabulary -- cell verdicts read
    # from device text -- and its sentences are what the sub-reasons below carry.
    #
    # TWO THINGS DIFFER FROM THE LEDGER, BOTH RELEASE DECISIONS, BOTH COUNTED BELOW
    # RATHER THAN ASSERTED:
    #
    #   R3. The source seam is asked LIKE FOR LIKE -- `clears_source_seam_like_for_
    #       like`, the seam-level rule the sibling boards ask -- instead of the
    #       product-aware `clears_source_seam` the ledger uses. That is the whole of
    #       this board's 342-against-363 divergence and it was never a capability
    #       difference. The product-aware number is NOT discarded: it is what the
    #       ledger above still reports and what the ranked build list is ordered by.
    #
    #   R4. `not_fusion_surface` is asked FIRST, ahead of the source seam. This board
    #       folded null-constitutive instances into its source bucket wherever the
    #       deposit blocked them, which prices an obstruction that obstructs nothing:
    #       where the constitutive half launches nothing there is no weld for a
    #       deposit to sit inside.
    #
    # SERVED IS ASKED THE WAY THE LEDGER ASKS IT and is not re-derived: `_served` is
    # the census's record of the shipped predicate's own answer. Widening the source
    # clause cannot add a served instance -- a predicate that admits a row already
    # asked `deposit_repair` about it through `seam_source_reasons` -- and the floor
    # inside `Taxonomy.finish` is what turns that reasoning into a check.
    taxonomy = fusion_taxonomy.Taxonomy(
        backend="cuda",
        priced=priced,
        rows=len(record),
        platform_note=(
            "CUDA HAS NO ANALOGUE OF METAL'S ARGUMENT-TABLE CEILING, and this zero "
            "is measured rather than skipped: CUDA passes kernel arguments in a "
            "32,764-byte parameter space and the widest kernel in this tree spends "
            "256 bytes of it, so no cell on this board is ever refused for the "
            "number of volumes its fused signature would bind. This board's fitness "
            "rule is POINTWISE-vs-STENCIL and explicitly NOT a binding count; the "
            "one genuine per-launch limit here -- no grid-wide barrier inside an "
            "ordinary launch -- is a STRUCTURAL fact shared with the other "
            "backends and is counted under structurally_unweldable, never here"),
        seam_instances={**{seam: sum(1 for e in per_row if seam in e["seams"])
                           for seam in SEAM_SLOTS},
                        h_to_d_seam.SEAM: len(h_to_d_entries)})
    lfl_blocked = collections.Counter()
    r4_moved_rows: List[dict] = []
    for entry in per_row:
        configuration = configuration_of[entry["row"]]
        for seam, block in entry["seams"].items():
            curl_arm, const_arm = block["curl_arm"], block["constitutive_arm"]
            verdict = verdict_of[(
                seam,
                f"{curl_arm['family']}/{curl_arm['arm']}" if curl_arm else None,
                f"{const_arm['family']}/{const_arm['arm']}" if const_arm else None,
                curl_arm["kernel"] if curl_arm else None,
                const_arm["kernel"] if const_arm else None)]
            clears_lfl = clears_source_seam_like_for_like(
                seam, engine_facts[entry["row"]], configuration)
            if not clears_lfl:
                lfl_blocked[seam] += 1
            if verdict == "NOT A FUSION CANDIDATE":
                if not clears_lfl or not _row_clears(entry, seam):
                    # R4's move, recorded per instance rather than as a total: the
                    # ledger prices this one as source-blocked and it has no weld
                    # for the deposit to obstruct.
                    r4_moved_rows.append({"row": entry["row"], "seam": seam})
                taxonomy.add(
                    "not_fusion_surface", seam, entry["row"],
                    "NOT A FUSION CANDIDATE — the constitutive sub-step launches "
                    "nothing, so there is no second kernel to weld",
                    {"constitutive_arm":
                        f"{const_arm['family']}/{const_arm['arm']}"
                        if const_arm else None})
            elif _served(entry["row"], seam):
                taxonomy.add(
                    "served", seam, entry["row"],
                    f"SERVED by a shipped product "
                    f"({next(iter(_serving_products(entry['row'], seam)), None)})",
                    {"product": next(
                        iter(_serving_products(entry["row"], seam)), None)})
            elif not clears_lfl:
                taxonomy.add(
                    "source_seam_unbracketable", seam, entry["row"],
                    "blocked by the source injection in the seam — the driver "
                    "deposits between the halves (driver.py:3283-3284 magnetic, "
                    ":3294-3299 electric) and deposit_repair.repairable REFUSES to "
                    "reconstruct this row's constitutive half BY NAME",
                    {"refused_by": sorted(
                        k for k, v in (
                            ("off-diagonal chi1inv",
                             bool(configuration.get("has_offdiagonal_epsilon"))),
                            ("instantaneous chi2/chi3",
                             bool(configuration.get("has_nonlinearity"))),
                            ("an unreadable fold map",
                             bool(configuration.get("has_symmetry"))))
                        if v)})
            elif verdict == "NO ARM SELECTED":
                curl_slot, const_slot = SEAM_SLOTS[seam]
                missing_slot = const_slot if const_arm is None else curl_slot
                partner = (f"{curl_arm['family']}/{curl_arm['arm']}"
                           if const_arm is None and curl_arm else
                           f"{const_arm['family']}/{const_arm['arm']}"
                           if const_arm else None)
                # NOT `refusal` — that name is a module-level function this
                # walk's enclosing scope calls, and binding it here would make it
                # a local for the whole of main().
                composer_refusal = block["constitutive_refusal"] \
                    if const_arm is None else block["curl_refusal"]
                taxonomy.add(
                    "missing_half", seam, entry["row"],
                    f"A KERNEL COVERAGE GAP, NOT A FUSION REFUSAL: the composer "
                    f"selects no arm at {missing_slot} on this row, so there is no "
                    f"second kernel to weld and the array path steps the slot",
                    {"missing_slot": missing_slot,
                     "missing_kernel":
                         f"a cuda_kernels arm on {missing_slot} admitting this row — "
                         f"the seam's other half is "
                         + (f"already covered by {partner!r}, so what is absent is "
                            f"the {missing_slot} body for that cell"
                            if partner else
                            "unselected too, so BOTH bodies are absent")
                         + ". Closing it is a SUB-STEP kernel's job, not a fusion "
                           "one",
                     "partner_arm": partner,
                     "composer_refusal_kind":
                         (composer_refusal or {}).get("kind"),
                     "composer_reasons":
                         (composer_refusal or {}).get("reasons", [])[:4]})
            elif verdict.startswith("STENCIL-BLOCKED") and "scratch-output" \
                    not in verdict:
                taxonomy.add(
                    "structurally_unweldable", seam, entry["row"],
                    "STENCIL-BLOCKED — the constitutive half reads the curl half's "
                    "in-place output at a cell this thread does not own. In one "
                    "launch that cell is written by another block, and CUDA offers "
                    "no grid-wide barrier inside an ordinary launch — only "
                    "cooperative groups, which caps the grid at the resident-block "
                    "count. MEASURED from the device code by analyze_kernel_reads, "
                    "not inferred from the family's name. A SCRATCH-OUTPUT weld is "
                    "not blocked by this and none occupies this cell yet",
                    {"kind": "stencil"})
            else:
                taxonomy.add(
                    "buildable_not_built", seam, entry["row"],
                    f"reachable and NOT BUILT — {verdict}: every precondition this "
                    f"board measures is met and no shipped product's predicate "
                    f"admits this row at this cell",
                    {"cell_verdict": verdict,
                     "clears_the_product_aware_source_clause":
                         _row_clears(entry, seam)})
    # The fourth seam, filed by the shared rule. What this board hands it is its own
    # product columns: which span update_H+step_D, and -- since 2026-09-06, when one
    # does -- a per-row `served_by` verdict, which is the only thing that can carry a
    # per-backend predicate into a rule three boards share. h_to_d_seam.price NAMES the
    # rows a board leaves unanswered rather than pricing them as unbuilt.
    #
    # THE VERDICT IS THE PREDICATE'S, AND THE SHARED RULE DEFINES IT THAT WAY: a served
    # H_to_D instance is a shipped product whose OWN predicate admits that row. It is
    # NOT a claim that the product installs, and on this board it does not: the one
    # product spanning this seam declares INSTALLABLE = False and
    # fused_pairs._declared_uninstallable refuses it on every configuration, which the
    # block records beside the count. `_loses_the_shared_slot` is deliberately NOT
    # applied here -- it models which of two products gets a slot the composer will
    # actually give to one of them, and this seam's product is given no slot at all, so
    # withholding on that ground would be modelling an arbitration that never runs.
    #
    # NO `credit_withheld` KEY ON THIS BOARD (2026-09-10). `h_to_d_seam.price` accepts
    # a withheld credit (the board's reason sentence) beside served_by, and this board
    # has nothing to withhold on -- it carries no per-product byte binding at all
    # (`subject_pin` and the registry arm cross-check are its only tree-vs-record
    # checks; that absence is this board's own recorded gap). Absent means credited,
    # by the shared contract, and the dispatch input above hands over every row with
    # a `served_by` for the same reason.
    h_to_d_spanning = [column for column, seams in fused_seams_of.items()
                       if h_to_d_seam.SEAM in seams]
    # (`served_by` was written onto each entry above, ahead of the dispatch
    # measurement, by exactly this rule.)
    h_to_d_block = h_to_d_seam.price(
        taxonomy, "cuda", "cuda_kernels", h_to_d_entries,
        products_spanning_the_seam=h_to_d_spanning)
    h_to_d_block["installable_declared_by_each_spanning_product"] = {
        column: _installable_declaration(column) for column in h_to_d_spanning}
    taxonomy_block = taxonomy.finish(
        served_expected=served_total + h_to_d_block["served"],
        served_expected_source=(
            f"`served_total` — the per-cell served counts, cross-checked against the "
            f"row walk, plus the H_to_D seam's served ({h_to_d_block['served']} of "
            f"{h_to_d_block['instances_priced']}; {h_to_d_block['served_note']})"))
    # THE PER-PRODUCT LEDGER OVER ALL FOUR SEAMS, read off the taxonomy's own grouping
    # of the served instances — and cross-checked against the two walks it must
    # agree with: the three-seam cell ledger product by product, and the H_to_D
    # block's per-product rows. A product credited by the taxonomy that neither walk
    # names, or a count the walks do not reproduce, stops the board here.
    served_ledger = {product: len(items) for product, items
                     in taxonomy_block["served_by_product"].items()}
    _expected_ledger = dict(served_by_product)
    for product, rows_ in h_to_d_block["served_by_product"].items():
        _expected_ledger[product] = _expected_ledger.get(product, 0) + len(rows_)
    if served_ledger != _expected_ledger:
        raise SystemExit(
            f"the taxonomy's served ledger {served_ledger} does not reproduce the "
            f"three-seam cell ledger plus the H_to_D block's per-product rows "
            f"{_expected_ledger}; the served instances and the walks that credit "
            f"them disagree about which product serves what.")

    # WHAT R3 AND R4 EACH MOVED, COUNTED AND NOT ESTIMATED. Two rulings act on the
    # same bucket and their effects overlap, so reporting only the net change would
    # hide which ruling did what. The ledger's own source bucket is the baseline.
    _ledger_blocked = sum(c["blocked by the source injection in the seam"]
                          for c in buckets.values())
    _lfl_total = sum(lfl_blocked.values())
    reconciled = {
        "r3_like_for_like_source_seam": {
            "the_board_s_own_product_aware_rule_blocks": _ledger_blocked,
            "the_reconciled_seam_level_rule_blocks": _lfl_total,
            "instances_freed": _ledger_blocked - _lfl_total,
            "per_seam": {"product_aware": {
                seam: buckets[seam]["blocked by the source injection in the seam"]
                for seam in SEAM_SLOTS},
                "like_for_like": {seam: lfl_blocked[seam] for seam in SEAM_SLOTS}},
            "why_they_differ": (
                "the product-aware rule additionally requires that a shipped "
                "product ALREADY OCCUPY the cell and declare "
                "CARRIES_DEPOSIT_REPAIR; the seam-level rule asks only "
                "deposit_repair.repairable, which is what the Metal and Triton "
                "boards ask. Every freed instance is one whose deposit a repair CAN "
                "carry and whose cell no product occupies yet — a coverage fact, "
                "not a ceiling one"),
            "both_are_kept": (
                "the product-aware number still prices the ledger and orders the "
                "ranked build list, where 'can a product that exists serve this' is "
                "the right question. The reconciled tiers use the seam-level rule, "
                "where 'could any product ever' is"),
        },
        "r4_precedence_move": {
            "instances_moved_from_source_seam_to_not_fusion_surface":
                len(r4_moved_rows),
            "instances": r4_moved_rows,
            "what_it_is": (
                "instances this board's own ledger filed as blocked by the source "
                "injection and whose constitutive sub-step launches NOTHING. Under "
                "R4 an instance is counted in its most fundamental bucket: where "
                "there is no second kernel to weld, the deposit obstructs nothing"),
            "the_ledger_still_says": {
                "source_blocked": _ledger_blocked,
                "null_constitutive_outside_the_source_bucket": null_plan},
        },
    }
    # The H_to_D seam's null rows are the same null update_H the ledger counts at
    # B_to_H, filed once more at the fourth seam; the ledger prices three seams and
    # the taxonomy four, so the difference is exactly that count.
    _h_to_d_null = h_to_d_block["buckets"].get("not_fusion_surface", 0)
    if taxonomy_block["buckets"]["not_fusion_surface"] != (
            null_plan + len(r4_moved_rows) + _h_to_d_null):
        raise SystemExit(
            f"the ledger and the reconciled taxonomy do not reconcile: the ledger "
            f"reports {null_plan} null-constitutive outside its source bucket, "
            f"R4 moves {len(r4_moved_rows)} into it and the H_to_D seam's null "
            f"update_H adds {_h_to_d_null}, but the taxonomy files "
            f"{taxonomy_block['buckets']['not_fusion_surface']}. One of the two "
            f"walks is reading a different corpus.")
    taxonomy_block.update(reconciled)
    taxonomy.report(lambda line: print(line, flush=True))
    print(f"\n  R3 LIKE-FOR-LIKE SOURCE SEAM: this board's product-aware rule "
          f"blocks {_ledger_blocked}; the reconciled seam-level rule blocks "
          f"{_lfl_total}, freeing {_ledger_blocked - _lfl_total} instance(s). The "
          f"ceiling moves {seam_instances - _ledger_blocked} -> "
          f"{seam_instances - _lfl_total} and becomes comparable with the sibling "
          f"boards for the first time.", flush=True)
    print(f"  R4 PRECEDENCE: {len(r4_moved_rows)} instance(s) move from "
          f"source_seam_unbracketable to not_fusion_surface.", flush=True)

    # ------------------------------------- what this board cannot be asked
    # DERIVED FROM THE SELECTION, NOT FROM A TABLE. A kernel this board never prices
    # is one no arm the composer selected on any corpus row names. That set moves
    # with the corpus and with the registry, so it is computed rather than listed:
    # the entry points reachable from the selected labels (with the `fused_` wrapper
    # prefix stripped, which is how the registry spells them) minus the in-seam
    # passes, subtracted from the shipped surface.
    selected_kernels = {plan["kernel"] for picked in slot_pick.values()
                        for plan in picked.values() if plan and plan["kernel"]}
    asked_kernels = set()
    for kernel_label in selected_kernels:
        asked_kernels.add(kernel_label)
        if kernel_label.startswith("fused_"):
            asked_kernels.add(kernel_label[len("fused_"):])
    in_seam = set(IN_SEAM_CUDA_KERNEL.values())
    unasked = sorted(set(surface["certified"]) - asked_kernels - in_seam)
    unasked_uncertified = sorted(set(surface["uncertified"]) - asked_kernels)
    print("\n" + "=" * 78, flush=True)
    print("WHAT THIS BOARD CANNOT BE ASKED", flush=True)
    print("=" * 78, flush=True)
    print(f"  the composer selected {len(selected_kernels)} distinct kernel labels "
          f"across the corpus: {sorted(selected_kernels)}", flush=True)
    print(f"  CERTIFIED kernels shipped today that NO selected arm names "
          f"({len(unasked)}):", flush=True)
    for kernel in unasked:
        print(f"      {kernel:38s} {surface['certified'][kernel]}", flush=True)
    print(f"  UNCERTIFIED kernels no selected arm names "
          f"({len(unasked_uncertified)}):", flush=True)
    for kernel in unasked_uncertified:
        print(f"      {kernel:38s} {surface['uncertified'][kernel]}", flush=True)
    # A SEAM WHOSE EVERY CELL IS "NO ARM SELECTED" IS NOT NECESSARILY A SEAM WITH
    # NO CANDIDATE — it may be a seam whose slot no arm is registered on at all.
    # Derived, so it moves with the registry rather than being a maintained sentence.
    unpriceable: Dict[str, dict] = {}
    for seam, (curl_slot, const_slot) in SEAM_SLOTS.items():
        seam_cells = [c for c in cells if c["seam"] == seam]
        if not seam_cells or any(c["verdict"] != "NO ARM SELECTED"
                                 for c in seam_cells):
            continue
        missing = sorted({slot for c in seam_cells for slot, arm in
                          ((curl_slot, c["curl_arm"]),
                           (const_slot, c["constitutive_arm"]))
                          if arm is None})
        candidates = sorted(
            kernel for kernel in unasked
            if any(kernel.startswith(slot) for slot in missing))
        unpriceable[seam] = {
            "every_cell_is_NO_ARM_SELECTED": True,
            "slots_with_no_selected_arm": missing,
            "certified_kernels_at_those_slots_no_selected_arm_names": candidates,
            "reading": (
                f"the {seam} seam prices as zero here because the composer selects "
                f"no arm at {', '.join(missing)} on any row that has the seam — "
                f"NOT because no candidate exists. "
                + (f"{candidates} are certified and shipped today and no selected "
                   f"arm names them, so this seam's demand is UNMEASURED rather "
                   f"than measured-zero." if candidates else
                   "No certified kernel at those slots is outside the selected set "
                   "either, so measured-zero is the reading.")),
        }
        print(f"  {seam}: every cell is NO ARM SELECTED, no arm at {missing}; "
              f"certified-but-unselected candidates there: {candidates or 'none'}",
              flush=True)
    print("  Their demand is UNMEASURED here: no arm the composer selects names "
          "them on any corpus row.", flush=True)
    print("  The run that would answer it: register an arm for them in "
          "cuda_kernels/registry.py and", flush=True)
    print("  re-cut parity/meep_gpu/results/cuda_predicate_coverage_*/ so the "
          "composer is asked with", flush=True)
    print("  that arm in the table.", flush=True)

    result = {
        "board": "hand-CUDA fusion board",
        "census": str(CENSUS),
        "engine_facts_read_from_the_same_census": [
            "configuration.source_field_types", "plan_step.live"],
        "engine_facts_joined_from": None,
        "engine_facts_join_retired": (
            "until 2026-08-28 source_field_types and plan_step.live were joined out "
            "of metal_coverage_tranche6_2026-08-19 because the CUDA census recorded "
            "neither. The promoted battery records both on every CUDA row, so the "
            "join is gone; rows() refuses unless every measured row carries them"),
        "cell_source": "plan_step.selected / selected_family / selected_kernel",
        "cell_source_justification": (
            "the composer (cuda_kernels/arms.py + registry.py) selects one arm per "
            "slot and refuses fail-closed rather than picking by table order, so the "
            "arm it selects IS what the engine would run. The predicate walk this "
            "board used before the composer shipped reported which family ADMITS a "
            "row, which is the same thing only while the admissions stay disjoint "
            "and no arm sits behind another clause"),
        "selection_asked_behind_the_backend_shim": (
            "every shipped hand-CUDA predicate opens with a non-CuPy refusal and the "
            "census ran on a NumPy host; the census records the real-grid "
            "composition beside the shimmed one under plan_step.unfactored and a "
            "per-row plan_step.shim soundness probe. Reading the unshimmed column "
            "would report a board of near-total refusal"),
        "subject_pin": pin,
        "composer_layer": composer,
        "rows": len(record),
        "rows_with_a_susceptibility": n_polarized,
        "denominator": priced,
        "denominator_three_seam_ledger": seam_instances,
        "denominator_justification": taxonomy.priced_justification,
        "denominator_located_not_transcribed": (
            f"{len(record)} rows x 2 curl->constitutive seams (B->H, D->E) + "
            f"{n_polarized} rows carrying a susceptibility x 1 E->P seam = "
            f"{seam_instances} seam-instances on the three-seam ledger, + "
            f"{len(record)} rows x 1 constitutive->curl seam (H->D) = {priced} "
            f"PRICED, with the seam line numbers LOCATED in driver.py at run time "
            f"rather than transcribed (the H->D span by h_to_d_seam.driver_seam_fact)."),
        # THE FOURTH SEAM'S OWN BLOCK: what sits between the halves (located in
        # driver.py at cut time), the probe it was priced from, its buckets and its
        # instances. The ledger, cells and headline counts above stay three-seam.
        "h_to_d_seam": h_to_d_block,
        # THE BINDING WALK, published beside the credits it is about rather than in
        # a companion artifact: a reader who takes `served` away from this file must
        # take the withheld list with it. See :func:`release_binding_report` for
        # what "bound" means on each of this track's TWO registries and why the
        # first bound cut reports rather than deducts.
        "release_binding": release_binding_report(
            sorted(set(served_by_product) | set(h_to_d_block["served_by_product"]))),
        "fused_products_that_exist": len(FUSED_PRODUCT_COLUMNS),
        "fused_products_note": (
            f"{len(FUSED_PRODUCT_COLUMNS)}: {', '.join(FUSED_PRODUCT_COLUMNS)}, "
            f"across the seams {', '.join(fused_seams)}. Three span B_to_H in a "
            f"single launch. The real magnetic pair performs FIVE driver passes there "
            f"(step_B, fill_symmetry_bc_B, zero_metal_B, "
            f"fill_folded_far_ghosts_B, update_H), carrying both mirror fills through "
            f"the ownership inversion; the two complex pairs added on 2026-08-30 "
            f"perform THREE (step_B, zero_metal_B, update_H) and refuse every folded "
            f"grid, so the two fills are no-ops on every row they admit rather than "
            f"passes they silently skip. cuda_fused_electric_pair, added the same "
            f"day, is the first product on D_to_E and performs THREE there (step_D, "
            f"zero_metal_D, update_E) on the same refusal. The 'served' column is "
            f"each product's OWN "
            f"predicate, asked by the census on every row and counted here -- not a "
            f"restatement of its clauses, which could admit a row the launch cannot "
            f"serve. certification.json's dispatch block still records wired=false, "
            f"so SERVED means 'this product's predicate admits the row', never "
            f"'anything runs it'"),
        "seams_located_in_driver": seams,
        "cuda_surface": {
            "certified": surface["certified"],
            "uncertified": surface["uncertified"],
            "per_module": surface["per_module"],
            "certified_count": len(surface["certified"]),
            "uncertified_count": len(surface["uncertified"]),
        },
        "fitness_rule": (
            "POINTWISE vs STENCIL, not a binding count. CUDA's kernel parameter "
            "space (4KB) is never the binding constraint for these signatures; what "
            "decides a weld is whether the constitutive half reads the curl half's "
            "in-place output at a cell this thread does not own. It cannot, "
            "byte-identically, inside one ordinary launch: there is no grid-wide "
            "barrier. Measured from the emitted device code by "
            "analyze_kernel_reads, with its four DETECTOR_CONTROLS reproduced "
            "before any verdict here is reported"),
        "fitness_per_selected_kernel": fitness,
        "detector_controls": controls,
        "source_seam_ceiling": {
            **{seam: ceiling[seam] for seam in SEAM_SLOTS},
            "on_the_driver_fact_alone": {seam: driver_ceiling[seam]
                                         for seam in SEAM_SLOTS},
            "deposit_carrying_products": {
                "/".join(key): module for key, module in PRODUCT_ON_CELL.items()
                if _carries_deposit_repair(module)},
            "what_it_is": (
                "the largest number of seam-instances a fused product can serve at "
                "that seam. The driver deposits a magnetic source between step_B and "
                "update_H and an electric source between step_D and update_E, and "
                "one launch cannot be on both sides of a deposit -- so a product "
                "that does not bracket its launch is capped at "
                "`on_the_driver_fact_alone`. A product that DOES bracket it (its own "
                "CARRIES_DEPOSIT_REPAIR, imported live) clears the injection on the "
                "cell it implements, because the deposit points are saved before the "
                "launch and recomputed after the driver has injected. E->P carries "
                "no injection, so its ceiling is simply the rows that have an "
                "update_P pass"),
            "source_field_types": {str(k): v for k, v in kinds.items()},
        },
        "in_seam_carry": carry,
        "per_slot_arm_selection": {
            slot: dict(collections.Counter(
                (f"{v[slot]['family']}/{v[slot]['arm']}" if v[slot]
                 else "NO ARM SELECTED") for v in slot_pick.values()))
            for slot in SLOTS},
        "per_slot_refusal_kinds": {
            slot: dict(refusal_kinds[slot]) for slot in SLOTS
            if refusal_kinds[slot]},
        "composer_refusals_measured": {
            "ambiguous_slots": ambiguous,
            "what_it_replaces": (
                "with no composer a cell was well defined only if at most one family "
                "admitted a row at a slot, and this board measured that disjointness "
                "and raised on an overlap. The composer refuses an ambiguous slot "
                "rather than picking by table order, so an overlap now surfaces as a "
                "refusal of kind `ambiguous` — still fatal, measured on all rows and "
                "all slots"),
        },
        "cells_ranked_by_demand": cells,
        "ledger": {seam: dict(counter) for seam, counter in buckets.items()},
        # PER PRODUCT OVER ALL FOUR SEAMS since 2026-09-06: the three-seam cell
        # ledger plus the H_to_D block's own per-product rows, so this sums to the
        # served headline. Read from the taxonomy's served ledger, which groups the
        # very instances the headline counts.
        "served_ledger": served_ledger,
        "served_ledger_three_seam": served_by_product,
        "reachable_seam_instances": total_reachable,
        # THE RECONCILED DECOMPOSITION, in full and under the key the Metal and
        # Triton boards use. The ledger above is unchanged and stays: this is the
        # translation, not a replacement, and every reconciled bucket carries this
        # board's own prose beneath it as the sub-reason.
        "taxonomy": taxonomy_block,
        "headline": {
            "denominator": priced,
            "denominator_three_seam_ledger": seam_instances,
            "reachable_after_the_source_seam": total_reachable,
            # THE RECONCILED CEILING (R3), beside the product-aware one this board
            # has always published. They answer different questions and are labelled
            # with them; only the reconciled one is comparable with the sibling
            # boards, and comparing the other one is what the 2026-09-02 audit found
            # three boards doing.
            "reachable_after_the_source_seam_like_for_like":
                seam_instances - sum(lfl_blocked.values()),
            "source_seam_rule_reconciled": reconciled["r3_like_for_like_source_seam"],
            "taxonomy": taxonomy_block["buckets"],
            "tiers": {name: block["value"]
                      for name, block in taxonomy_block["tiers"].items()},
            "pointwise_buildable": pointwise,
            "stencil_blocked": stencil,
            # THE SPLIT, 2026-09-02. The fitness verdict is unchanged on every one
            # of these cells -- the constitutive kernel really does read cells the
            # thread does not own -- and what the split records is that a
            # SCRATCH-OUTPUT weld occupies some of them, which the original
            # sentence's "not weldable byte-identically as an ordinary launch"
            # ruled out only for the in-place shape it was measuring.
            "stencil_blocked_served_by_a_scratch_output_weld": stencil_served,
            "stencil_blocked_with_no_product": stencil - stencil_served,
            "no_arm_selected_at_one_slot": no_pair,
            "null_constitutive_no_second_kernel": null_plan,
            "undetermined": undetermined,
            "buckets_total_the_reachable_ceiling": classified == total_reachable,
            # DERIVED FROM THE INSTANCE LEDGER, 2026-09-06. Through the `_hd` cut
            # this key carried `served_total`, the three-seam cell ledger's total,
            # while the taxonomy summed four seams — 359 beside 483 in one artifact.
            # It is now the count of instances the taxonomy filed served (and
            # lists), and `fusion_taxonomy.headline_agreement` refuses the cut if any
            # served key in this artifact says otherwise. The three-seam total keeps
            # its own name beside it: a different fact, not a second spelling.
            "served_by_a_fused_product_today": taxonomy_block["buckets"]["served"],
            "served_by_predicate_per_seam": {
                seam: counts.get("served", 0)
                for seam, counts in taxonomy_block["per_seam"].items()},
            "served_three_seam_ledger": served_total,
            "served_by_product": served_ledger,
            "served_in_dispatch": dispatch_reach["served_in_dispatch"],
        },
        "served_in_dispatch_measurement": dispatch_reach,
        "kernel_labels_the_composer_selected": sorted(selected_kernels),
        "unanswerable_from_this_board": {
            "certified_kernels_no_selected_arm_names": {
                k: surface["certified"][k] for k in unasked},
            "uncertified_kernels_no_selected_arm_names": {
                k: surface["uncertified"][k] for k in unasked_uncertified},
            "seams_this_board_prices_as_zero_without_measuring_zero": unpriceable,
            "the_run_that_would_answer_it": (
                "register an arm for these kernels in cuda_kernels/registry.py and "
                "re-cut parity/meep_gpu/results/cuda_predicate_coverage_* so the "
                f"composer is asked with that arm in the table on the same {len(record)} "
                "lifted rows. A kernel with no arm cannot be selected, so its demand is "
                "unmeasured here rather than measured zero"),
            "not_claimed_here": [
                "whether any candidate fused body is CORRECT. This board asks only "
                "whether a pair is demanded, whether both halves exist, and whether "
                "the constitutive half's reads permit one launch. Nothing here is a "
                "byte-parity claim and no gate is implied",
                "the E->P seam's launch sequencing. ade_update_p launches once per "
                "driven component with the host rotating buffers between launches; "
                "a POINTWISE verdict at that seam is a read-pattern verdict only",
                "register pressure and occupancy of any welded kernel — not "
                "measured, and a pointwise weld can still lose to two launches",
            ],
        },
        "elapsed_s": round(time.time() - started, 1),
    }

    print("\n" + "=" * 78, flush=True)
    print("THE HEADLINE", flush=True)
    print("=" * 78, flush=True)
    print(f"  denominator (PRICED, four seams)          {priced}   "
          f"(three-seam ledger {seam_instances} + {len(h_to_d_entries)} H_to_D)",
          flush=True)
    print(f"  reachable after the source seam           {total_reachable}  "
          f"({total_reachable / seam_instances:.1%} of the three-seam ledger)",
          flush=True)
    print(f"  POINTWISE-BUILDABLE                       {pointwise}", flush=True)
    print(f"  STENCIL-BLOCKED                           {stencil}  "
          f"(of which served by a scratch-output weld: {stencil_served})", flush=True)
    print(f"  composer selects no arm at one slot        {no_pair}", flush=True)
    print(f"  constitutive launches nothing             {null_plan}", flush=True)
    print(f"  UNDETERMINED                              {undetermined}", flush=True)
    print(f"  served by a fused product TODAY           "
          f"{taxonomy_block['buckets']['served']}  (derived from the instance "
          f"ledger: {served_total} on the three-seam ledger + "
          f"{h_to_d_block['served']} H_to_D; per product {served_ledger})", flush=True)
    print(f"  BUCKET FLOOR: {classified} == {total_reachable} reachable", flush=True)
    print("\n  THE RANKED BUILDABLE LIST:", flush=True)
    rank = 0
    for cell in cells:
        if not cell["verdict"].startswith("POINTWISE-BUILDABLE"):
            continue
        rank += 1
        print(f"    {rank}. reach {cell['demand_rows_clearing_the_source_seam']:4d}  "
              f"(carrying nothing: "
              f"{cell['reachable_with_nothing_to_carry_in_the_seam']:3d})  "
              f"{cell['seam']:8s} ({cell['curl_arm']}, "
              f"{cell['constitutive_arm']})", flush=True)

    # FLOOR 7: every served total this artifact publishes, at any depth, is the
    # instance ledger's count — refused, not recorded, where one is not.
    result["headline_agreement"] = fusion_taxonomy.headline_agreement(
        result, taxonomy_block, "cuda")
    # AND THE SAME FLOOR ON THE DISPATCH AXIS. `served` and `served_in_dispatch` are
    # different numbers, and until 2026-09-11 only the first had a floor -- which is
    # how the Metal board came to publish a hardcoded dispatch 0 beside its own
    # measured 176 in one artifact. `dispatch_agreement` reads the measurement as the
    # authority and refuses any other number for the same question, listing (never
    # comparing) the totals that sit under a declared different CONDITION.
    result["dispatch_agreement"] = dispatch_agreement.dispatch_agreement(
        result, dispatch_reach, "cuda")
    print(f"  DISPATCH AGREEMENT: "
          f"{result['dispatch_agreement']['served_in_dispatch']} in dispatch at "
          f"{len(result['dispatch_agreement']['paths_checked'])} published paths, "
          f"all read from the measurement", flush=True)
    print(f"  HEADLINE AGREEMENT: {result['headline_agreement']['served']} served at "
          f"{len(result['headline_agreement']['paths_checked'])} published paths, "
          f"all derived from the instance ledger", flush=True)
    out = out_dir / "fusion_matrix_cuda.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}  ({result['elapsed_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
