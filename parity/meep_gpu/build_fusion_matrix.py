#!/usr/bin/env python3
"""THE METAL FUSION MATRIX — which (curl, constitutive) pair each corpus row drives
at each fusable seam, whether a fused product exists for it, and whether that
product's SHIPPED PREDICATE admits the row.

WHAT THIS ANSWERS THAT THE PER-PRODUCT CENSUSES DO NOT
======================================================

``results/triton_folded_fused_magnetic_pair_census_2026-08-19`` and
``results/triton_complex_fused_magnetic_pair_census_2026-08-19T2355`` each price ONE
product against ONE census. Neither says what fraction of the corpus a fused path can
step on a backend, and neither enumerates the pairs no product covers. This does both,
for Metal.

THE SEAM, NOT THE SLOT
======================

The sub-step censuses count SLOTS: 759 = 186 rows x 4 sub-steps + 15 rows with a
polarization, on the 186-row basis those censuses were cut over (the corpus is 194
rows since 2026-09-03). Fusion is not measured in slots. A fused product collapses a CURL, the
driver passes that follow it, and the CONSTITUTIVE that closes it into one launch —
it serves a ROW at a SEAM. The driver's timestep (driver.py:3281-3306) has exactly
three seams a fused product could close, and the two source injections bound them:

    step_B (3282) -> [magnetic inject 3283-3284 | fill_symmetry_bc_B 3285 |
                      zero_metal_B 3286 | fill_folded_far_ghosts_B 3287] -> update_H (3289)
    step_D (3293) -> [electric inject 3294-3299 | fill_symmetry_bc_D 3300 |
                      zero_metal_D 3301 | fill_folded_far_ghosts_D 3302] -> update_E (3304)
    update_E (3304) -> [NOTHING] -> update_P (3306)

so the denominator, on the 194-row basis priced since 2026-09-03, is

    194 rows x 2 curl->constitutive seams          = 388
  + 15 rows carrying a susceptibility x 1 E->P seam =  15
                                                     ---
                                                      403 seam-instances

(387 = 372 + 15 on the 186-row basis every board before 2026-09-03 priced.) Both
counts are read from the census at run time; the arithmetic above is what they mean.

The third seam is counted only on the 15 rows that HAVE an ``update_P`` pass, exactly
as the 759 slot denominator adds 15 rather than the row count for that sub-step. The
seam-instance total is NOT a partition of the slot total and is not meant to be: a
B->H seam consumes two slots, and ``update_E`` is shared between the D->E seam and
the E->P chain.

WHERE THE ROWS AND THE VERDICTS COME FROM
=========================================

``results/metal_coverage_2026-09-03_complete`` — the standing METAL census since
2026-09-03: the 186 rows of ``metal_coverage_tranche6_2026-08-19`` (the record commit
``e97c3eb``'s "757 of 759 corpus slots" is measured from) plus the eight example
scripts the 2026-08-09 corpus campaign accepted in its _blocked/_gdsii/_sigma legs,
194 in all. It lifted every engine-accepted row, called every live Metal predicate
on each, AND RECORDED
``plan_step.selected``: which arm the composer actually picks at each slot. That last
field is why this round does not have to reconstruct the routing — the planner's own
answer is in the record.

DERIVED, NOT RE-LIFTED, and that is stated rather than implied. Each fused product's
predicate is a CONJUNCTION of its two halves' own certified predicates plus seam
clauses (fused_magnetic_pair.py:465-527, folded_fused_pair.py:679-770,
fused_dispersive_pair.py:471-522, folded_fused_magnetic_pair.py:853-959,
complex_fused_magnetic_pair.py:676-742). The halves were evaluated by the census on
the lifted engine object; the seam clauses are evaluated here from the census's
``configuration`` block. Every clause that could NOT be evaluated is named in
``clauses_not_evaluable_from_the_census_block`` and the result is marked
``is_an_upper_bound``.

THE BINDING CEILING IS MEASURED HERE, NOT TRANSCRIBED
=====================================================

``device.MAX_BUFFER_BINDINGS`` is 31 (device.py:72) — buffer attribute indices run
0..30 and a 32nd is a compile error. This script EMITS every shipped kernel's source
through its own generator and counts the distinct ``[[buffer(N)]]`` attributes,
splitting pointers (``device T*``) from scalars (``constant T&``), because the ceiling
constrains the SIGNATURE and packing scalars into one ``constant Params&`` buys slots
back. Measured on this host 2026-08-19 and recorded at fused_dispersive_pair.py:44-62:

    35 separate bindings                     -> FAILED, 'buffer' attribute out of bounds
    30 pointers + 5 scalars in one Params&   -> COMPILES (31 bindings)
    30 pointers                              -> COMPILES

so the fitness test for a candidate fused product is POINTERS <= 30.

The shipped fused pairs calibrate how many pointers a fusion SHARES between its two
halves, and that calibration is computed here rather than assumed: for each shipped
pair, ``shared = pointers(curl) + pointers(constitutive) - pointers(fused)``. A
candidate's fused pointer count is reported as the BRACKET
``[sum - max_shared, sum - min_shared]`` over the measured shared values. Where the
bracket's LOW end already exceeds 30 the pair cannot be bound on Metal at all, and
that is a first-class finding.

Usage (from the repository root)::

    PYTHONPATH=. python -u \
      parity/meep_gpu/results/fusion_matrix_metal_2026-08-20/build_fusion_matrix.py
"""

from __future__ import annotations

import collections
import argparse
import contextlib
import os
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

#: The results tree this cut reads its censuses from and writes its board into.
#: Named explicitly rather than derived from ``HERE.parent`` because this script
#: is CANONICAL and lives beside the gates, not inside the board it produces --
#: that is the whole reason it was promoted out of ``results/``. Nineteen
#: divergent copies under ``results/`` is how the 2026-08-25 board forked off a
#: pre-carry ancestor and silently re-applied a retired fold clause.
RESULTS = HERE / "results"
#: ``the repository root`` — this file sits at ``parity/meep_gpu/results/<dir>/``.
#: Derived from ``HERE`` rather than from the working directory so the release
#: binding below is checked against the same tree the census came from, whatever
#: directory the script is launched from.
#: the repository root -- the package root, not the repo root (named for what it is:
#: the depth-vs-name confusion here already cost one silent bad census).
API_ROOT = HERE.parents[1]
#: The Metal census an argument-free run prices. Selectable so a fresh cut does not
#: need a fresh copy of this file; older records stay selectable by name for
#: reproducing an old cut.
#:
#: THE DEFAULT MOVED 2026-09-03 from ``metal_coverage_tranche6_2026-08-19`` — the
#: record every Metal board up to ``fusion_matrix_metal_2026-09-02_tierfix`` (343/387)
#: was cut over — to ``metal_coverage_2026-09-03_complete``, and what moved it is the
#: CORPUS, not the subject: eight example scripts the 2026-08-09 corpus campaign
#: accepted in its _blocked/_gdsii/_sigma legs had gone unpriced on every board
#: because the census driver named only the stock leg's lift record. The new record
#: prices them — 194 rows, 60 examples + 134 tests (24 of them the parameterised leg's
#: matched rows), every row stamped with one subject manifest and one battery digest
#: — and the board cut over it (``fusion_matrix_metal_2026-09-03_complete``) reads
#: 403 PRICED / 384 FUSABLE / 359 ATTAINABLE and 356/403 served, with 0 status changes
#: on the 186 shared labels. NOT ``metal_coverage_2026-09-03_extended``, cut the same
#: day and superseded: its five gdsii/sigma rows were evaluated under interpreters
#: with no torch, every Metal predicate refused for that one process-level reason,
#: and the driver still stamped them measured — nine phantom missing halves on the
#: board cut over it. The ``_complete`` re-cut is the one with the runtime preflight
#: that refuses such a row by name.
#:
#: THE DEFAULT MOVED AGAIN 2026-09-06, to ``metal_coverage_2026-09-04_m0complex`` —
#: the census EVERY Metal board from ``fusion_matrix_metal_2026-09-04_m0`` on was cut
#: over while this default still named ``_2026-09-03_complete``, so an argument-free
#: run answered a different question from the standing board for two days. What
#: separates the two records is the SUBJECT, measured by diffing them row by row:
#: identical 194 rows, identical configurations, and exactly ONE row whose
#: ``plan_step.selected`` differs — ``examples:dipole_in_vacuum_cyl_off_axis.py``,
#: unselected at all four slots on ``_complete`` and ``cylindrical complex`` at all
#: four on ``_m0complex`` (four ``predicates`` columns False -> True on that row,
#: nothing else). That is the m = 0 complex cylindrical landing of 2026-09-04
#: (``cylindrical_complex.py`` gained the M_ZERO arm), which ``_complete`` predates;
#: cutting over ``_complete`` today prices that row's B_to_H and D_to_E instances as
#: ``missing_half`` where the shipped tree serves them — two regressions that are a
#: census difference, not a coverage one — and its H_to_D instance as ``missing_half``
#: where the tree has both halves. An argument-free cut over ``_m0complex``
#: reproduces the standing board with zero instances changed
#: (``fusion_matrix_metal_2026-09-06_consolidated`` against ``_2026-09-06_hd``).
#: ``_complete`` stays selectable by name for reproducing a 2026-09-03 cut.
#: THE DEFAULT MOVED AGAIN 2026-09-23, to ``metal_coverage_2026-09-23_sparse``: the
#: sparse-transport pass moved `stepping.py` and seven `metal_kernels/` subjects (eight
#: of 82), so `_m0complex` no longer hashes to this tree and the subject pin refuses it
#: by name. Same battery and probe basis as `_m0complex`; the corpus is complete now
#: (194 rows, 179 measured against the 170 the 09-04 cut could reach), so this census
#: prices nine rows more, and the board cut over it after the Metal sparse round is
#: the first Metal board on the full corpus.
#: THE DEFAULT MOVED AGAIN 2026-09-25, to ``metal_coverage_2026-09-25_night``: the census cut on
#: the tree carrying the citation re-point and the batched wall clears.
#: THE DEFAULT MOVED AGAIN 2026-09-24, to ``metal_coverage_2026-09-24_round3``: the night
#: round moved `metal_kernels/device.py` and `no_pml_constitutive.py`, both subjects, and
#: THIS builder does not check its census's subject manifest (only the CUDA board does),
#: so a stale default would cut silently. Cut on the round-3 tree, 60 + 119 + 24 rows
#: measured, 0 of 94 subjects moved against the final tree.
CENSUS = RESULTS / (os.environ.get("MEEP_GPU_METAL_CENSUS")
                    or "metal_coverage_2026-10-02_arch3")
RECLOSE = RESULTS / "metal_coverage_special_kz_reclose_2026-08-19"

#: ``device.MAX_BUFFER_BINDINGS``. Read from the package rather than spelled, so a
#: platform change moves this file's arithmetic with it.
# The package lives at ``API_ROOT/meep_gpu``. Put it on the path EXPLICITLY rather
# than inheriting it from the caller's working directory: the copies under
# ``results/`` only ever worked when invoked from one particular cwd, and a
# canonical script that silently depends on that is the same class of trap as a
# canonical script that silently reads a stale ancestor's clauses.
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
except Exception:  # pragma: no cover - the ceiling is the one number we refuse to guess
    raise SystemExit("meep_gpu.metal_kernels.device is not importable; the binding "
                     "ceiling cannot be read and this script refuses to assume it")

# THE RUN RECORDS, one per GPU architecture: a weld's artifact and timestamp are
# read from its live runs, never from top-level fields the ledger no longer carries.
from meep_gpu import metal_runs  # noqa: E402

# SERVED IN DISPATCH, measured rather than asserted. This board wrote a literal
# ``0`` until 2026-08-29; see ``dispatch_reachability`` for why a constant is the
# wrong shape for that answer even when the answer is zero.
import dispatch_reachability as _reach  # noqa: E402

# THE RECONCILED BUCKET NAMES AND THE THREE TIERS, shared with the Triton and CUDA
# boards (release decision R1/R2, 2026-09-02). One name list in one file is what stops a
# fourth vocabulary growing back; the module also carries the floors that make the
# decomposition and the served total REFUSABLE rather than merely printed.
import dispatch_agreement  # FLOOR 8; see its docstring for why it is not in fusion_taxonomy
import fusion_taxonomy  # noqa: E402
# THE FOURTH SEAM, update_H -> step_D (2026-09-04): one pricing rule for the three
# boards, off the probe artifact and the driver's own text. See h_to_d_seam.py.
import h_to_d_seam  # noqa: E402

#: With every scalar packed into one ``constant Params&``, a kernel may bind this many
#: POINTERS. Measured, fused_dispersive_pair.py:52-58.
MAX_POINTERS = MAX_BUFFER_BINDINGS - 1

# THE NEAR FILL'S SOURCE ROW IS READ, NOT SPELLED, for the same reason the binding
# ceiling above is. `_deposit_is_repairable` mirrors `deposit_repair.repairable`'s
# too-short-fold refusal, and `deposit_repair._folded_seam_reasons` compares against
# `stepping.MIRROR_SOURCE_INDEX` (deposit_repair.py:286). A board that spelled the
# number would keep answering the old threshold after the engine moved and would ADMIT
# folds the shipped predicate refuses -- over-counting, which is the one direction that
# reads as success. `triton_kernels.symmetry` carries its own copy that the folded
# products' fill carries use; the two are required to agree here rather than assumed to,
# since a split would put the board and the kernels on different rows.
try:
    from meep_gpu.stepping import MIRROR_SOURCE_INDEX as NEAR_SOURCE_INDEX
    from meep_gpu.triton_kernels.symmetry import (
        MIRROR_SOURCE_INDEX as _SYMMETRY_NEAR_SOURCE_INDEX)
except Exception:  # pragma: no cover - the row the near fill images is not guessable
    raise SystemExit("meep_gpu.stepping.MIRROR_SOURCE_INDEX is not importable; the row "
                     "the near fill images -- and so which folds the deposit repair "
                     "refuses -- cannot be read and this script refuses to assume it")
if NEAR_SOURCE_INDEX != _SYMMETRY_NEAR_SOURCE_INDEX:
    raise SystemExit(
        f"stepping.MIRROR_SOURCE_INDEX is {NEAR_SOURCE_INDEX} and "
        f"triton_kernels.symmetry.MIRROR_SOURCE_INDEX is "
        f"{_SYMMETRY_NEAR_SOURCE_INDEX}; the array path's near fill and the kernels' "
        f"carry image different rows, so no single board clause is right for both")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rows() -> List[dict]:
    record = load(CENSUS / "examples.jsonl") + load(CENSUS / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load(CENSUS / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return _apply_reclose([r for r in record if r.get("measured")])


def _apply_reclose(record: List[dict]) -> List[dict]:
    """Substitute the RECLOSE artifact's re-measured rows for their stale twins.

    THE TRANCHE-6 CENSUS PREDATES THE 2026-08-19 special_kz/folded_beta ARMS BY
    THREE HOURS, and the reclose battery re-ran the affected rows with the current
    tree — the staleness block below has carried that as a NOTE while nothing on
    the stale value's pair had a product on either side, so record and source
    agreed and the correction cost nothing. The RESIDUE ROUND built the fused
    pair for exactly that pair (`beta_complex_fused_*`), so carrying the stale
    value forward would now MISCOUNT SERVED — the one thing this board may not
    do — and the correction is APPLIED rather than annotated: a row is replaced
    wholesale by the reclose record of the same name (a MEASUREMENT by the same
    battery on the same host, not an edit), keeping the census `leg`.

    MEASURED SCOPE, so the substitution cannot quietly widen: exactly ONE census
    row matches a reclose record by name (tests:TestSpecialKz.test_special_kz —
    the reclose's other five rows carry unparametrised names the census does not
    use), its predicate diff is exactly the two 2026-08-19 arms
    (beta_run_complex_constitutive@update_H/E False -> True), and its selected
    diff is exactly those two slots gaining `special_kz complex beta`. The floor
    below refuses a substitution outside that scope.
    """
    reclose: Dict[str, dict] = {}
    for name in ("test_special_kz.json", "test_refl_angular.json"):
        path = RECLOSE / "smoke" / name
        if not path.is_file():
            raise SystemExit(
                f"the reclose artifact {path} is missing; the tranche-6 census "
                f"cannot be corrected and this board would miscount served")
        for entry in json.loads(path.read_text()):
            if entry.get("measured") and "predicates" in entry:
                reclose[entry["row"]] = entry
    substituted: List[str] = []
    out: List[dict] = []
    for row in record:
        fresh = reclose.get(row.get("row"))
        if fresh is None:
            out.append(row)
            continue
        replacement = dict(fresh)
        replacement["leg"] = row.get("leg")
        substituted.append(f"{row.get('leg')}:{row.get('row')}")
        out.append(replacement)
    if substituted != ["tests:TestSpecialKz.test_special_kz"]:
        raise SystemExit(
            f"the reclose substitution matched {substituted!r} where exactly "
            f"['tests:TestSpecialKz.test_special_kz'] was measured to match; the "
            f"correction's scope has changed and must be re-measured before this "
            f"board is cut")
    return out


def covered(row: dict, key: str) -> bool:
    """The Metal census ran on THIS host with MPS live, so ``covered`` is the verdict.

    Unlike the Triton rounds — whose predicates all refuse a NumPy host and whose
    censuses therefore have to read ``covered_modulo_backend`` — no Metal predicate
    carries a host clause that this corpus trips, and the census's own precondition
    block records a resolved subnormal policy of ``flush`` on every row (measured on
    the 2026-08-19 and the 2026-09-03 census alike). Reading
    the raw column is correct here and reading a residual would be the error.

    THE KEY MUST EXIST. A half key the census never measured used to score ZERO —
    ``.get(key, {})`` — which prices the product at zero admitted rows and reads
    exactly like a product nothing admits. That is the silent under-coverage this
    board exists to prevent, and it happened: the two complex stencil welds were
    first written against ``complex_no_pml_curl@step_D`` and
    ``folded_complex_bloch_curl@step_D``, neither of which the census carries
    (they are ``complex_plain_curl@step_D`` and ``folded_complex_pml_curl@step_D``),
    and the board printed 0 of 2 and 0 of 3 without a word. It raises now.
    """
    predicates = row["predicates"]
    if key not in predicates:
        raise SystemExit(
            f"the census row {row.get('row')!r} carries no predicate {key!r}; a "
            f"half key nothing measured would price its product at ZERO admitted "
            f"rows and read as a product nothing admits. The measured keys are "
            f"{sorted(predicates)}")
    return bool(predicates[key].get("covered"))


# ---------------------------------------------------------------------------
# THE MEASURED SIGNATURES
# ---------------------------------------------------------------------------

_PTR = re.compile(r"device\s+(?:const\s+)?\w+\s*\*\s*\w+\s*\[\[buffer\((\d+)\)\]\]")
_SCA = re.compile(r"constant\s+(?:const\s+)?[\w:]+\s*&\s*\w+\s*\[\[buffer\((\d+)\)\]\]")
_ANY = re.compile(r"\[\[buffer\((\d+)\)\]\]")
#: The WRITABLE pointers only — `device T*` without `const`. A body's component
#: arity is read off these and nothing else (`_component_arity`).
_RW_PTR = re.compile(
    r"device\s+(?!const\b)[\w:]+\s*\*\s*(\w+)\s*\[\[buffer\(\d+\)\]\]")
_TRAILING_INDEX = re.compile(r"\d+$")

_PERIODIC = (0, 0, 0)
_MIRROR_METALLIC = (2, 0, 0)   # triton_kernels/symmetry.py:111-114
#: The folded H->D weld's fingerprint triple: a folded PERIODIC y axis, which
#: is the cell's 10-row shape and the ONLY termination that emits a top-plane
#: mask. On an unfolded triple the folded emitter reduces to the certified one
#: and the digest would be the plain product's plus a comment.
_FOLDED_PERIODIC = (0, 3, 0)   # triton_kernels/symmetry.py:111-114
_EXPANSION = "FMA_V1"
_LIVE_ROWS = (1, 1, 1, 1, 1, 1)
_ZEROS = (0, 0, 0)
_ONES = (1, 1, 1)
_UNWALLED = (False, False, False)


def _generators() -> Dict[str, Callable[[], str]]:
    """One emitter per shipped kernel body, called with a representative
    specialisation. The COUNT is what is being measured and it does not vary with the
    boundary triple: every one of these signatures is fixed and the constexprs are
    baked into the BODY."""
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        ade_update_p, bfast_curl, complex_conductive_pml,
        complex_folded_offdiag_update_e, complex_fields, complex_fused_magnetic_pair,
        complex_no_pml_conductive, complex_no_pml_curl,
        complex_no_pml_offdiag_update_e, complex_no_pml_stored_e,
        complex_dispersive_update_e, conductive_pml, cylindrical_complex,
        cylindrical_real, dispersive_update_e, folded_beta, folded_complex,
        cylindrical_fused_magnetic_pair, cylindrical_real_fused_magnetic_pair,
        cylindrical_real_fused_electric_pair, cylindrical_fused_electric_pair,
        cylindrical_real_fused_hd_pair, folded_fused_hd_pair,
        bfast_fused_hd_pair, conductive_fused_hd_pair,
        beta_complex_fused_hd_pair, beta_real_fused_hd_pair,
        folded_complex_fused_hd_pair,
        fused_ade_chain, complex_fused_ade_chain, fused_electric_pair, fused_hd_pair,
        cylindrical_complex_fused_hd_pair,
        complex_conductive_fused_pair,
        beta_fused_magnetic_pair, bfast_fused_magnetic_pair,
        bfast_fused_electric_pair, conductive_fused_electric_pair,
        beta_complex_fused_electric_pair, beta_complex_fused_magnetic_pair,
        nonlinear_fused_magnetic_pair, folded_complex_fused_magnetic_pair,
        folded_beta_complex_fused_magnetic_pair,
        beta_fused_electric_pair, complex_fused_electric_pair,
        complex_fused_hd_pair,
        folded_complex_fused_pair, folded_beta_complex_fused_pair,
        folded_beta_real_fused_magnetic_pair, folded_beta_real_fused_pair,
        folded_fused_dispersive_pair,
        folded_fused_magnetic_pair, folded_fused_pair,
        folded_offdiag_dispersive_update_e, folded_offdiag_update_e,
        fused_dispersive_pair, fused_magnetic_pair, no_pml_conductive,
        no_pml_conductive_fused_electric_pair, no_pml_curl,
        no_pml_fused_electric_pair,
        no_pml_stored_e, nonlinear_update_e, offdiag_update_e, shaders, special_kz,
        symmetry, templates,
        offdiag_fused_electric_pair, folded_offdiag_fused_electric_pair,
        complex_no_pml_offdiag_fused_electric_pair,
        folded_complex_offdiag_fused_electric_pair,
    )
    return {
        # --- curl bodies -----------------------------------------------------
        "shaders.curl": lambda: shaders.curl_source(_PERIODIC, False),
        "symmetry.folded_curl":
            lambda: symmetry.folded_curl_source(_MIRROR_METALLIC, False),
        "complex_fields.bloch_curl":
            lambda: complex_fields.bloch_curl_source(_PERIODIC, False, _ZEROS,
                                                     _EXPANSION),
        "folded_complex.bloch_curl":
            lambda: folded_complex.folded_bloch_curl_source(_MIRROR_METALLIC, False,
                                                            _ZEROS, _EXPANSION),
        "cylindrical_complex.curl":
            lambda: cylindrical_complex.cylindrical_curl_source(1, False, 1,
                                                                _EXPANSION),
        "cylindrical_real.curl":
            lambda: cylindrical_real.cylindrical_curl_source("step_B", (1, 0, 1)),
        "cylindrical_real.prefix":
            lambda: cylindrical_real.cylindrical_prefix_source("step_B"),
        "no_pml_curl.plain_curl":
            lambda: no_pml_curl.plain_curl_source(_PERIODIC, False, False),
        "no_pml_conductive.curl":
            lambda: no_pml_conductive.conductive_plain_curl_source(
                _PERIODIC, False, False, (True, True, True)),
        "conductive_pml.curl":
            lambda: conductive_pml.conductive_pml_curl_source(_PERIODIC, False,
                                                              (True, True, True)),
        "complex_no_pml_curl.curl":
            lambda: complex_no_pml_curl.complex_no_pml_curl_source(
                _PERIODIC, False, _ZEROS, _EXPANSION),
        "complex_no_pml_conductive.curl":
            lambda: complex_no_pml_conductive.complex_conductive_no_pml_curl_source(
                _PERIODIC, False, _ZEROS, (True, True, True), _EXPANSION),
        "complex_conductive_pml.curl":
            lambda: complex_conductive_pml.complex_conductive_pml_curl_source(
                _PERIODIC, False, _ZEROS, (True, True, True), _EXPANSION),
        "special_kz.beta_curl":
            lambda: special_kz.beta_curl_source(_PERIODIC, False),
        "special_kz.beta_bloch_curl":
            lambda: special_kz.beta_bloch_curl_source(_PERIODIC, False, _ZEROS,
                                                      _EXPANSION),
        "folded_beta.curl":
            lambda: folded_beta.folded_beta_curl_source(_MIRROR_METALLIC, False),
        "folded_beta.bloch_curl":
            lambda: folded_beta.folded_beta_bloch_curl_source(
                _MIRROR_METALLIC, False, _ZEROS, _EXPANSION),
        "bfast_curl.curl": lambda: bfast_curl.bfast_curl_source(_PERIODIC, False),
        # --- constitutive bodies ---------------------------------------------
        "shaders.constitutive_H": lambda: shaders.constitutive_source("H"),
        "shaders.constitutive_E": lambda: shaders.constitutive_source("E"),
        "complex_fields.bloch_constitutive_H":
            lambda: complex_fields.bloch_constitutive_source("H", _EXPANSION),
        "complex_fields.bloch_constitutive_E":
            lambda: complex_fields.bloch_constitutive_source("E", _EXPANSION),
        "offdiag_update_e.offdiag":
            lambda: offdiag_update_e.offdiag_source(_LIVE_ROWS, _PERIODIC, _ZEROS),
        "folded_offdiag_update_e.offdiag":
            lambda: folded_offdiag_update_e.folded_offdiag_source(
                _LIVE_ROWS, _MIRROR_METALLIC, _ZEROS, _ZEROS),
        "complex_folded_offdiag_update_e.offdiag":
            lambda: complex_folded_offdiag_update_e.complex_folded_offdiag_source(
                _LIVE_ROWS, _MIRROR_METALLIC, _ZEROS, _ZEROS, _ZEROS, _EXPANSION),
        "complex_no_pml_offdiag_update_e.offdiag":
            lambda: complex_no_pml_offdiag_update_e.complex_no_pml_offdiag_source(
                _LIVE_ROWS, _PERIODIC, _ZEROS, _ZEROS, _EXPANSION),
        "dispersive_update_e.e": lambda: dispersive_update_e.dispersive_e_source(1, 0),
        "complex_dispersive_update_e.e":
            lambda: complex_dispersive_update_e.complex_dispersive_e_source(
                1, 0, _EXPANSION),
        "folded_offdiag_dispersive_update_e.e":
            lambda: folded_offdiag_dispersive_update_e.folded_offdiag_dispersive_source(
                _LIVE_ROWS, _MIRROR_METALLIC, _ZEROS, _ZEROS, _ONES),
        "no_pml_stored_e.e": lambda: no_pml_stored_e.stored_e_source(1),
        "complex_no_pml_stored_e.e":
            lambda: complex_no_pml_stored_e.complex_stored_e_source(1, _EXPANSION),
        "nonlinear_update_e.e":
            lambda: nonlinear_update_e.nonlinear_source(_ONES, _ZEROS, _ZEROS,
                                                        _PERIODIC),
        "ade_update_p.p": lambda: ade_update_p.ade_source("float32", True),
        # --- the five SHIPPED fused products, for the sharing calibration -----
        "FUSED.fused_magnetic_pair":
            lambda: fused_magnetic_pair.fused_magnetic_pair_source(_PERIODIC,
                                                                   _UNWALLED),
        # THE D-SIDE TWIN, added 2026-08-28. Same two specialisation arguments as
        # the magnetic pair above, because it is the same shape of product on the
        # other seam -- so the measured difference between the two counts is the
        # three inverse-epsilon volumes and nothing else, which is exactly the
        # claim its docstring makes.
        "FUSED.fused_electric_pair":
            lambda: fused_electric_pair.fused_electric_pair_source(_PERIODIC,
                                                                   _UNWALLED),
        # THE FOURTH SEAM'S BODY, added 2026-09-06 with the product. ONE
        # specialisation argument -- the boundary triple -- because this family has no
        # wall clear to carry: `zero_metal_B` runs one seam earlier and `zero_metal_D`
        # one seam later, so neither is inside `update_H` -> `step_D` and a mask here
        # would be a pass the driver runs again. Its signature is the platform ceiling
        # as an EQUALITY (31 of 31), which the family pins and its gate bisects.
        "FUSED.fused_hd_pair":
            lambda: fused_hd_pair.fused_hd_pair_source(_PERIODIC),
        # THE FOURTH SEAM'S TWO SMALL CELLS, added 2026-09-08 with their
        # RELEASE_BINDING rows, each at the specialisation its OWN gate compiles.
        # The conductive triple is `bc_ppp`'s -- every target lossy, the flags that
        # case carries (gate_metal_conductive_fused_hd_pair.CASES) -- so this row is
        # the same emitter call the gate's binding-ceiling leg makes.
        #
        # THE COUNT DOES NOT MOVE WITH EITHER ARGUMENT, measured before these rows
        # were written rather than assumed: the conductive body emits 27 buffer
        # attributes on (T,T,T), on (F,F,F) and on (T,F,F) alike, and the BFAST body
        # 26 on the periodic triple and on the fully metallic one. That is what the
        # docstring above claims for the boundary triple, checked here for the lossy
        # triple as well, and 27 is the number the gate's ceiling leg bisects.
        # THE THREE 2026-09-08 EARNED WELDS. One representative specialisation each,
        # at the same boundary/phase/arm constants the sibling complex and folded rows
        # use -- the COUNT is what this measures and the constexprs are baked into the
        # body, so the triple does not vary it.
        "FUSED.beta_complex_fused_hd_pair":
            lambda: (beta_complex_fused_hd_pair
                     .beta_complex_fused_hd_pair_source(
                         "plain", _PERIODIC, _ONES, _EXPANSION)),
        "FUSED.beta_real_fused_hd_pair":
            lambda: (beta_real_fused_hd_pair
                     .beta_real_fused_hd_pair_source("plain", _PERIODIC)),
        "FUSED.folded_complex_fused_hd_pair":
            lambda: (folded_complex_fused_hd_pair
                     .folded_complex_fused_hd_pair_source(
                         # NO BLOCH PHASE ON A FOLDED AXIS: a mirror plane reflects
                         # rather than repeating, so stepping._bloch_phases raises.
                         # The folded-complex siblings above specialise the same way.
                         _FOLDED_PERIODIC, _ZEROS, _EXPANSION)),
        "FUSED.conductive_fused_hd_pair":
            lambda: (conductive_fused_hd_pair
                     .conductive_fused_hd_pair_source(_PERIODIC, (True, True, True))),
        "FUSED.bfast_fused_hd_pair":
            lambda: bfast_fused_hd_pair.bfast_fused_hd_pair_source(_PERIODIC),
        # THE FOLDED TWIN'S SOURCE IS TAKEN ON A FOLDED TRIPLE, deliberately. On
        # (PERIODIC, PERIODIC, PERIODIC) the folded emitter reduces to the plain one
        # and this entry would digest the row above's bytes plus a comment -- which is
        # a real property (the transcription leg measures exactly that reduction) and
        # a useless fingerprint. (P, MIRROR_PERIODIC, P) is the cell's 10-row shape
        # and carries both the folded ghost and the top-plane mask.
        "FUSED.folded_fused_hd_pair":
            lambda: folded_fused_hd_pair.folded_fused_hd_pair_source(_FOLDED_PERIODIC),
        # THE Dcyl m = 0 H->D PAIR IS TWO LAUNCHES, and the signature measured here is
        # LAUNCH 1's -- the new kernel (the pointwise update_H into scratch plus the
        # column-leader radial prefix, 24 pointers + Params). Launch 2 is the
        # certified cylindrical curl, already in this table as
        # `cylindrical_real.curl`. READ THE CALIBRATION ROW THIS PRODUCES WITH CARE:
        # its "shared_pointers" (halves minus fused) is NOT a sharing measurement,
        # because the two halves never sit in one signature -- the D/fu_D group and
        # the curl's sinv vectors do not enter launch 1 at all. The count is recorded
        # because the board requires one per product; it does not license a
        # predicted pointer range for an unbuilt two-launch cell.
        "FUSED.cylindrical_real_fused_hd_pair":
            lambda: (cylindrical_real_fused_hd_pair
                     .cylindrical_real_fused_hd_pair_constitutive_source()),
        # THE COMPLEX Dcyl H->D LEAD LAUNCH (2026-09-06): the pointwise complex H
        # constitutive plus the column-leader radial scan, 24 pointers + Params.
        # Its second launch is the certified `cylindrical_complex.curl` byte for
        # byte and is already enumerated above under that name; the product is
        # excluded from the pointer-sharing calibration below because its two
        # launches share no signature (TWO_LAUNCH_NOT_A_SPLICE).
        "FUSED.cylindrical_complex_fused_hd_pair":
            lambda: cylindrical_complex_fused_hd_pair.lead_source(_EXPANSION),
        "FUSED.complex_fused_magnetic_pair":
            lambda: complex_fused_magnetic_pair.complex_fused_magnetic_pair_source(
                _PERIODIC, _ZEROS, _UNWALLED, _EXPANSION),
        "FUSED.folded_fused_pair":
            lambda: folded_fused_pair.folded_fused_pair_source(
                _MIRROR_METALLIC, _ONES, _UNWALLED),
        "FUSED.folded_fused_magnetic_pair":
            lambda: folded_fused_magnetic_pair.folded_fused_magnetic_pair_source(
                _MIRROR_METALLIC, _ONES, _UNWALLED),
        "FUSED.fused_dispersive_pair":
            lambda: fused_dispersive_pair.fused_dispersive_pair_source(
                _PERIODIC, 0, 1, _UNWALLED),
        # THE TWO NO-ABSORBER STORED-E D->E PAIRS, added 2026-09-01 with the
        # products -- the last cells the 2026-08-31_plainrepair board scored
        # "FITS ... NOT BUILT". Per component like `fused_dispersive_pair` above,
        # and for the pole-budget reason the conductive module's docstring spells:
        # its own corpus rows carry FIVE poles per component, so the
        # all-three-component signature needs 33 pointers against the 30 ceiling.
        # The pole count and sigma flag change the BODY only -- the signature
        # declares all eight pole slots either way -- so any specialisation
        # measures the same count; the representative here is one pole on axis 0,
        # the same convention as the dispersive row above.
        "FUSED.no_pml_fused_electric_pair":
            lambda: (no_pml_fused_electric_pair
                     .no_pml_fused_electric_pair_source(
                         _PERIODIC, 0, 1, _UNWALLED)),
        "FUSED.no_pml_conductive_fused_electric_pair":
            lambda: (no_pml_conductive_fused_electric_pair
                     .no_pml_conductive_fused_electric_pair_source(
                         _PERIODIC, 0, 1, True, _UNWALLED)),
        # The Dcyl m = 0 pair, added 2026-08-20 with the product. Its r axis is
        # pinned METALLIC by the certified curl emitter, so the triple below is the
        # only one it will emit for.
        "FUSED.cylindrical_real_fused_magnetic_pair":
            lambda: (cylindrical_real_fused_magnetic_pair
                     .cylindrical_real_fused_magnetic_pair_source(
                         (1, 0, 1), _UNWALLED)),
        # THE Dcyl m = 0 D->E PAIR, added 2026-08-31 with the product. Same two
        # specialisation arguments as its magnetic twin above, because it is the same
        # cell on the other seam -- so the measured difference between the two counts
        # is the three inverse-epsilon volumes, the prefix and the PACK, and nothing
        # else. THIS IS THE ONE ROW ON THIS TABLE WHOSE POINTER COUNT IS NOT THE SUM
        # OF ITS HALVES: the curl half's six per-axis PML coefficient vectors ride in
        # ONE buffer with six element offsets in the Params struct
        # (`metal_kernels/coefficient_pack.py`), so the census reads 26 pointers where
        # the unpacked shape needs 31 -- which is one over the ceiling and is why the
        # cell scored UNFUSABLE ON METAL until this round. Both numbers are COMPILED
        # by `gate_metal_cylindrical_real_fused_electric_pair.py`'s `binding_ceiling`
        # leg (`metal_cylindrical_real_fused_electric_pair_2026-08-31_pack`), which
        # also bisects the ceiling on this host rather than reading it off a constant.
        "FUSED.cylindrical_real_fused_electric_pair":
            lambda: (cylindrical_real_fused_electric_pair
                     .cylindrical_real_fused_electric_pair_source(
                         (1, 0, 1), _UNWALLED)),
        # THE Dcyl |m| >= 1 COMPLEX D->E PAIR, added 2026-08-31 with the product, and
        # the SECOND row on this table whose pointer count is not the sum of its
        # halves. Same specialisation arguments as its magnetic twin above — bcz
        # PERIODIC with the UNWALLED triple, M_ONE — so the measured difference
        # between the two counts is the three inverse-epsilon volumes MINUS the five
        # pointers the coefficient pack saves. Unpacked it needs 33 against a ceiling
        # of 30, which is the arithmetic behind the UNFUSABLE ON METAL verdict this
        # cell carried; `gate_metal_cylindrical_fused_electric_pair.py`'s
        # `binding_ceiling` leg COMPILES all four signatures and bisects the ceiling.
        "FUSED.cylindrical_complex_fused_electric_pair":
            lambda: (cylindrical_fused_electric_pair
                     .cylindrical_fused_electric_pair_source(
                         templates.PERIODIC, cylindrical_complex.M_ONE,
                         _UNWALLED, _EXPANSION)),
        # The Dcyl |m| >= 1 COMPLEX pair (module cylindrical_fused_magnetic_pair.py,
        # arm family cylindrical_complex_fused_magnetic_pair). Carried verbatim from
        # ../fusion_matrix_metal_2026-08-20c/. bcz PERIODIC pairs with the UNWALLED
        # triple exactly as the module's own enumerate_sources pairs them, so this is
        # one of the four sources a shipped plan can emit. M_ONE is the |m| = 1 arm.
        "FUSED.cylindrical_complex_fused_magnetic_pair":
            lambda: (cylindrical_fused_magnetic_pair
                     .cylindrical_fused_magnetic_pair_source(
                         templates.PERIODIC, cylindrical_complex.M_ONE,
                         _UNWALLED, _EXPANSION)),
        # The folded COMPLEX B/H pair, carried verbatim from
        # ../fusion_matrix_metal_2026-08-20_foldedcomplex/.
        "FUSED.folded_complex_fused_magnetic_pair":
            lambda: (folded_complex_fused_magnetic_pair
                     .folded_complex_fused_magnetic_pair_source(
                         _MIRROR_METALLIC, _ZEROS, _UNWALLED, _EXPANSION)),
        # THE FOLDED BETA COMPLEX B/H pair, added 2026-08-27. Same specialisation
        # arguments as the folded COMPLEX parent above (this weld is that module with
        # one emitter swapped), so a divergence in the measured count would mean the
        # beta insert had cost a pointer -- which is exactly the question.
        "FUSED.folded_beta_complex_fused_magnetic_pair":
            lambda: (folded_beta_complex_fused_magnetic_pair
                     .folded_beta_complex_fused_magnetic_pair_source(
                         _MIRROR_METALLIC, _ZEROS, _UNWALLED, _EXPANSION)),
        # --- THE SEVEN PRODUCTS OF THE 2026-08-30 TRANCHE ---------------------
        # EVERY SPECIALISATION BELOW IS THE ONE ITS OWN GATE COMPILED. Not chosen
        # by resemblance to a sibling entry: `gate_metal_tranche7_fused_pairs.py`
        # compiled each of these seven signatures on this Mac's MPS and recorded
        # the binding count it got (`metal_tranche7_fused_pairs_2026-08-30/
        # binding_counts.json`), and the arguments below are the ones that
        # reproduce those seven numbers exactly -- 31, 31, 31, 31, 28, 31, 25.
        # That is what makes these entries a measurement of the SHIPPED kernel
        # rather than of a signature nothing launches: a specialisation that
        # emitted a different pointer set would disagree with the gate's number,
        # and this table would be calibrating a bracket against bytes no plan
        # builds. Four of the seven sit ON the 31-binding ceiling, which is why
        # the gate also records that appending one more buffer is REFUSED.
        #
        # Each pairs with the sibling it was spliced from, and takes that
        # sibling's arguments for the reason the D-side twin above takes the
        # magnetic pair's: the measured difference between the two counts is then
        # the product's own claim and nothing else.
        "FUSED.beta_fused_electric_pair":            # splice of fused_electric_pair
            lambda: beta_fused_electric_pair.beta_fused_electric_pair_source(
                _PERIODIC, _UNWALLED),
        # THE CARTESIAN COMPLEX/BLOCH H->D WELD (2026-09-07): the certified
        # complex `update_H` body lifted as `h_cell` into write-only scratch and
        # RECOMPUTED for each of the curl's three backward taps, spliced ahead
        # of the certified complex `step_D` curl. 30 pointers plus one packed
        # `Params` is the platform ceiling EXACTLY, which the family pins as an
        # EQUALITY and its gate bisects on the host.
        "FUSED.complex_fused_hd_pair":
            lambda: complex_fused_hd_pair.complex_fused_hd_pair_source(
                _PERIODIC, _ONES, _EXPANSION),
        "FUSED.complex_fused_electric_pair":         # D-side complex twin
            lambda: complex_fused_electric_pair.complex_fused_electric_pair_source(
                _PERIODIC, _ZEROS, _UNWALLED, _EXPANSION),
        "FUSED.folded_complex_fused_pair":           # D-side folded-complex twin
            lambda: folded_complex_fused_pair.folded_complex_fused_pair_source(
                _MIRROR_METALLIC, _ZEROS, _UNWALLED, _EXPANSION),
        "FUSED.folded_beta_complex_fused_pair":      # D-side folded-beta-complex twin
            lambda: (folded_beta_complex_fused_pair
                     .folded_beta_complex_fused_pair_source(
                         _MIRROR_METALLIC, _ZEROS, _UNWALLED, _EXPANSION)),
        "FUSED.folded_beta_real_fused_magnetic_pair":  # beta head in folded_fused_magnetic_pair
            lambda: (folded_beta_real_fused_magnetic_pair
                     .folded_beta_real_fused_magnetic_pair_source(
                         _MIRROR_METALLIC, _ONES, _UNWALLED)),
        "FUSED.folded_beta_real_fused_pair":         # beta head in folded_fused_pair
            lambda: folded_beta_real_fused_pair.folded_beta_real_fused_pair_source(
                _MIRROR_METALLIC, _ONES, _UNWALLED),
        # The folded dispersive D->E pair. ONE POLE on component 0, the same
        # specialisation `FUSED.fused_dispersive_pair` above is measured at, with
        # the folded codes and parities its emitter additionally takes.
        "FUSED.folded_fused_dispersive_pair":
            lambda: folded_fused_dispersive_pair.folded_fused_dispersive_pair_source(
                _MIRROR_METALLIC, 0, 1, _ONES, _UNWALLED),
        # --- THE THREE BELOW-THE-CUT WELDS, carried verbatim from
        # ../fusion_matrix_metal_2026-08-20_belowcut/. nonlinear emits
        # `fused_magnetic_pair`'s OWN bytes (it is an admission port), so its
        # signature is measured through its own forwarding entry point rather than
        # assumed equal — a divergence would mean the two had stopped being the same
        # kernel.
        "FUSED.nonlinear_fused_magnetic_pair":
            lambda: nonlinear_fused_magnetic_pair
            .nonlinear_fused_magnetic_pair_source(_PERIODIC, _UNWALLED),
        "FUSED.beta_fused_magnetic_pair":
            lambda: beta_fused_magnetic_pair.beta_fused_magnetic_pair_source(
                _PERIODIC, _UNWALLED),
        "FUSED.bfast_fused_magnetic_pair":
            lambda: bfast_fused_magnetic_pair.bfast_fused_magnetic_pair_source(
                _PERIODIC, _UNWALLED),
        # The three E->P chain arms, added 2026-08-20 with the product. ONE POLE
        # with a VOLUME sigma, axis 0 — the same specialisation the two certified
        # E emitters above are measured at (`stored_e_source(1)`,
        # `dispersive_e_source(1, 0)`) and the same sigma kind as
        # `ade_source("float32", True)`, so the three numbers this feeds the
        # sharing calibration are taken at one specialisation across all three
        # bodies rather than at three.
        "FUSED.fused_ade_chain":
            lambda: fused_ade_chain.fused_ade_chain_source("no_pml", 0, 1, (True,)),
        "FUSED.fused_ade_chain_dispersive":
            lambda: fused_ade_chain.fused_ade_chain_source(
                "dispersive", 0, 1, (True,)),
        # The FOLDED dispersive arm emits the SAME source as the unfolded one —
        # the fold lives in the E half's PREDICATE (folded_dispersive_e_coverage),
        # not in its body, which is why `dispersive_update_e.e` already serves both
        # ("update_E", "dispersive PML E") and ("update_E", "folded dispersive PML
        # E") in CONSTITUTIVE_BODY above. Emitted separately anyway so the
        # calibration measures a number per PRODUCT rather than assuming two
        # products share a signature.
        "FUSED.folded_fused_ade_chain":
            lambda: fused_ade_chain.fused_ade_chain_source(
                "dispersive", 0, 1, (True,)),
        # --- THE TWO PRODUCTS OF THIS CUT -----------------------------------
        # The complex conductive D->E pair, at the specialisation the corpus's
        # four rows actually take: all-periodic, all three axes PHASED (their
        # k_point is (0.4, -1.3, 0.7)), all three targets conductive (an Absorber
        # on every face), ONE pole per component.
        "FUSED.complex_conductive_fused_pair":
            lambda: (complex_conductive_fused_pair
                     .complex_conductive_fused_pair_source(
                         _PERIODIC, _ONES, (True, True, True), (1, 1, 1),
                         _EXPANSION)),
        # The COMPLEX E->P chain, at ONE POLE with a VOLUME sigma — the same
        # specialisation `complex_no_pml_stored_e.e` and `ade_update_p.p` are
        # measured at above, so the sharing number this feeds is taken at one
        # specialisation across all three bodies rather than at three.
        "FUSED.complex_fused_ade_chain":
            lambda: (complex_fused_ade_chain
                     .complex_fused_ade_chain_source(1, (True,), _EXPANSION)),
        # --- THE FOUR PRODUCTS OF THE RESIDUE ROUND -------------------------
        # The two packed D->E welds close the board's last two CANNOT-BIND cells;
        # their emitted signatures carry the pack (28 pointers where the halves
        # sum to 33 and 39), and the calibration adds each family's declared
        # savings back so the sharing number stays a shared-VOLUME count.
        "FUSED.bfast_fused_electric_pair":
            lambda: bfast_fused_electric_pair.bfast_fused_electric_pair_source(
                _PERIODIC, _UNWALLED),
        "FUSED.conductive_fused_electric_pair":
            lambda: (conductive_fused_electric_pair
                     .conductive_fused_electric_pair_source(
                         _PERIODIC, (True, True, True), _UNWALLED)),
        # The two special_kz complex-beta welds, at the one-phased-axis
        # specialisation their corpus row takes (an in-plane k on x, beta != 0).
        "FUSED.beta_complex_fused_electric_pair":
            lambda: (beta_complex_fused_electric_pair
                     .beta_complex_fused_electric_pair_source(
                         _PERIODIC, (1, 0, 0), _UNWALLED, _EXPANSION)),
        "FUSED.beta_complex_fused_magnetic_pair":
            lambda: (beta_complex_fused_magnetic_pair
                     .beta_complex_fused_magnetic_pair_source(
                         _PERIODIC, (1, 0, 0), _UNWALLED, _EXPANSION)),
        # THE FOUR SCRATCH-OUTPUT STENCIL WELDS. Their emitted signatures carry
        # the SCRATCH pointers (pre-launch D/fu and their twins are different
        # buffers, so the unshared count is six over the halves' sum) and, on
        # three of the four, TWO packs; the calibration below adds each family's
        # declared savings back so the sharing number stays a shared-VOLUME count.
        "FUSED.offdiag_fused_electric_pair":
            lambda: (offdiag_fused_electric_pair
                     .offdiag_fused_electric_pair_source(
                         _LIVE_ROWS, _PERIODIC, _ZEROS)),
        "FUSED.folded_offdiag_fused_electric_pair":
            lambda: (folded_offdiag_fused_electric_pair
                     .folded_offdiag_fused_electric_pair_source(
                         _LIVE_ROWS, _MIRROR_METALLIC, _ZEROS, _ZEROS,
                         {0: 1}, {})),
        "FUSED.complex_no_pml_offdiag_fused_electric_pair":
            lambda: (complex_no_pml_offdiag_fused_electric_pair
                     .complex_no_pml_offdiag_fused_electric_pair_source(
                         _LIVE_ROWS, _PERIODIC, _ZEROS, _ZEROS, _EXPANSION)),
        "FUSED.folded_complex_offdiag_fused_electric_pair":
            lambda: (folded_complex_offdiag_fused_electric_pair
                     .folded_complex_offdiag_fused_electric_pair_source(
                         _LIVE_ROWS, _MIRROR_METALLIC, _ZEROS, _ZEROS, _ZEROS,
                         {0: 1}, {}, _EXPANSION)),
    }


def _component_arity(name: str, source: str) -> int:
    """How many FIELD COMPONENTS one launch of this body writes — MEASURED.

    Not read off an arm label and not inferred from the emitter's argument list: it
    is counted from the emitted Metal text, from the WRITABLE pointers alone. A
    `device const T*` is an input; a `device T*` is a volume this launch owns. The
    engine names a per-component volume with the component index glued to a role
    prefix (`f0/f1/f2` the target triple, `u0/u1/u2` the auxiliary triple, `w0/w1/w2`
    the split-field `f_w`), so the arity is the largest number of writable pointers
    that share a role — three for a body that steps the whole vector in one launch,
    one for a body the host launches once per driven component.

    THE TABLES SAY THREE INDEPENDENTLY, which is what makes this a measurement of a
    known quantity rather than a new definition: ``coverage.CURL_SUB_STEPS['step_D']``
    is ``('Dx', 'Dy', 'Dz')`` and ``coverage.CONSTITUTIVE_SIDES['E']['sources']`` is
    the same triple, so an ordinary D->E pair is three-against-three and the volumes
    the two halves have in common are exactly those three D volumes.

    Refuses anything that is neither 1 nor 3: this engine has three components, and a
    body that appears to write some other number means the parse is wrong, not that
    the kernel is exotic. A wrong arity here would mis-calibrate a bracket, and a
    quietly wrong bracket is the failure this whole script exists to prevent.
    """
    roles: Dict[str, int] = collections.Counter()
    for pointer in set(_RW_PTR.findall(source)):
        roles[_TRAILING_INDEX.sub("", pointer)] += 1
    if not roles:
        raise SystemExit(
            f"{name}: the emitted body binds no WRITABLE pointer at all, so this "
            f"script cannot say how many components one launch of it writes")
    arity = max(roles.values())
    if arity not in (1, 3):
        raise SystemExit(
            f"{name}: the writable pointers group as {dict(roles)}, giving a "
            f"component arity of {arity}; this engine steps one or three components "
            f"per launch, so the signature parse is wrong and every bracket derived "
            f"from it would be wrong with it")
    return arity


def measure_signatures() -> Dict[str, Dict[str, int]]:
    generators = _generators()
    # THE FLOOR THAT WAS A KeyError. Every priced product's fused body must have an
    # emitter here, because the sharing calibration reads `signatures[f"FUSED.
    # {family}"]` for all of PRODUCTS. Until 2026-08-30 a product landing in PRODUCTS
    # without an entry in `_generators` got no such row and the run died ~100 lines
    # later inside the calibration loop, on a bare `KeyError: 'FUSED.<family>'` with
    # no statement of what was wrong or how many were missing -- which is how the
    # seven products of that tranche presented. It is asked HERE, before any
    # measurement, and it names every missing family at once: a floor this script
    # relies on should refuse by name like the rest of them do.
    unmeasured = sorted(f for f in PRODUCTS if f"FUSED.{f}" not in generators)
    if unmeasured:
        raise SystemExit(
            f"PRODUCTS prices {unmeasured} but `_generators` emits no FUSED body for "
            f"them, so this cut can measure no signature for those products and the "
            f"pointer-sharing calibration has nothing to read. Add one emitter entry "
            f"per family, at the specialisation its own gate compiled.")
    out: Dict[str, Dict[str, int]] = {}
    for name, emit in generators.items():
        source = emit()
        pointers = set(_PTR.findall(source))
        scalars = set(_SCA.findall(source))
        every = set(_ANY.findall(source))
        if len(pointers) + len(scalars) != len(every):
            raise SystemExit(
                f"{name}: {len(every)} buffer attributes but {len(pointers)} pointers "
                f"+ {len(scalars)} scalars; the signature parse is incomplete and a "
                f"binding count derived from it would be wrong")
        arity = _component_arity(name, source)
        out[name] = {"pointers": len(pointers), "scalars": len(scalars),
                     "bindings": len(every), "component_arity": arity}
        print(f"  signature {name:44s} pointers={len(pointers):3d} "
              f"scalars={len(scalars):3d} bindings={len(every):3d} "
              f"components={arity}", flush=True)
    return out


#: arm LABEL -> the kernel body it emits, per slot kind. The reuse entries are
#: transcribed from the tree with their citation, because "same kernel" is a fact the
#: modules state and not one this script may infer from a label.
CURL_BODY: Dict[str, str] = {
    "PML": "shaders.curl",
    "folded": "symmetry.folded_curl",
    "complex/Bloch": "complex_fields.bloch_curl",
    "folded complex": "folded_complex.bloch_curl",
    "cylindrical complex": "cylindrical_complex.curl",
    "cylindrical m=0": "cylindrical_real.curl",
    "no-PML curl": "no_pml_curl.plain_curl",
    "conductive no-PML curl": "no_pml_conductive.curl",
    "conductive PML curl": "conductive_pml.curl",
    "complex no-PML curl": "complex_no_pml_curl.curl",
    "complex conductive no-PML curl": "complex_no_pml_conductive.curl",
    "complex conductive PML curl": "complex_conductive_pml.curl",
    "special_kz real beta": "special_kz.beta_curl",
    "special_kz complex beta": "special_kz.beta_bloch_curl",
    "folded beta real": "folded_beta.curl",
    "folded beta complex": "folded_beta.bloch_curl",
    "BFAST": "bfast_curl.curl",
    # registry.py:106-117 — B/D/H reuse the certified ordinary PML shaders through
    # nonlinear-only spine arms; only update_E has a dedicated Pade body.
    "nonlinear PML curl": "shaders.curl",
}

CONSTITUTIVE_BODY: Dict[Tuple[str, str], Optional[str]] = {
    # update_H -------------------------------------------------------------
    ("update_H", "ordinary"): "shaders.constitutive_H",
    # symmetry.plan_folded_constitutive builds through launch.ConstitutivePlan /
    # _constitutive_functions — the certified body (symmetry.py:"A CERTIFIED
    # constitutive plan for a folded run").
    ("update_H", "folded"): "shaders.constitutive_H",
    ("update_H", "complex/Bloch"): "complex_fields.bloch_constitutive_H",
    ("update_H", "folded complex"): "complex_fields.bloch_constitutive_H",
    # cylindrical_complex.py:21 / :1197 — the CERTIFIED bloch_constitutive_step.
    ("update_H", "cylindrical complex"): "complex_fields.bloch_constitutive_H",
    # cylindrical_real.py:1312/1335-1338 — launch.ConstitutivePlan, certified real.
    ("update_H", "cylindrical m=0"): "shaders.constitutive_H",
    ("update_H", "special_kz real beta"): "shaders.constitutive_H",
    ("update_H", "special_kz complex beta"): "complex_fields.bloch_constitutive_H",
    ("update_H", "folded beta real"): "shaders.constitutive_H",
    ("update_H", "folded beta complex"): "complex_fields.bloch_constitutive_H",
    ("update_H", "BFAST"): "shaders.constitutive_H",
    ("update_H", "nonlinear PML magnetic"): "shaders.constitutive_H",
    # THE NULL ARM LAUNCHES NOTHING (no_pml_constitutive.py:457
    # performs_device_work = False): the array path's update_H returns before its
    # first statement (stepping.py:944-945 / :983-984, MEASURED by poisoning every
    # volume). There is no second kernel to fuse the curl WITH.
    ("update_H", "no-PML null"): None,
    # update_E -------------------------------------------------------------
    ("update_E", "ordinary"): "shaders.constitutive_E",
    ("update_E", "folded"): "shaders.constitutive_E",
    ("update_E", "complex/Bloch"): "complex_fields.bloch_constitutive_E",
    ("update_E", "folded complex"): "complex_fields.bloch_constitutive_E",
    ("update_E", "cylindrical complex"): "complex_fields.bloch_constitutive_E",
    ("update_E", "cylindrical m=0"): "shaders.constitutive_E",
    ("update_E", "special_kz real beta"): "shaders.constitutive_E",
    ("update_E", "special_kz complex beta"): "complex_fields.bloch_constitutive_E",
    ("update_E", "folded beta real"): "shaders.constitutive_E",
    ("update_E", "folded beta complex"): "complex_fields.bloch_constitutive_E",
    ("update_E", "BFAST"): "shaders.constitutive_E",
    ("update_E", "offdiag"): "offdiag_update_e.offdiag",
    ("update_E", "folded offdiag"): "folded_offdiag_update_e.offdiag",
    ("update_E", "complex folded off-diagonal PML E"):
        "complex_folded_offdiag_update_e.offdiag",
    ("update_E", "complex no-PML off-diagonal"):
        "complex_no_pml_offdiag_update_e.offdiag",
    ("update_E", "dispersive PML E"): "dispersive_update_e.e",
    # folded_dispersive_update_e imports dispersive_update_e as _base and reuses its
    # kernel body (folded_dispersive_update_e.py:20 `from . import
    # dispersive_update_e as _base`).
    ("update_E", "folded dispersive PML E"): "dispersive_update_e.e",
    ("update_E", "folded off-diagonal dispersive PML E"):
        "folded_offdiag_dispersive_update_e.e",
    ("update_E", "complex dispersive PML E"): "complex_dispersive_update_e.e",
    ("update_E", "no-PML stored E"): "no_pml_stored_e.e",
    ("update_E", "complex no-PML stored E"): "complex_no_pml_stored_e.e",
    ("update_E", "nonlinear"): "nonlinear_update_e.e",
    ("update_E", "no-PML null"): None,
    # update_P -------------------------------------------------------------
    ("update_P", "ADE update_P"): "ade_update_p.p",
}


# ---------------------------------------------------------------------------
# THE SHIPPED FUSED PRODUCTS THIS CUT ASKS ABOUT
# (the count is printed from len(PRODUCTS) rather than spelled, so a cut that adds
#  a product cannot leave a stale numeral describing its own table)
# ---------------------------------------------------------------------------

def _folds(configuration: dict) -> List[int]:
    return [axis for axis in range(3) if configuration["mirrored"][axis]]


def _source_types(configuration: dict) -> Tuple[str, ...]:
    return tuple(str(kind) for kind in (configuration.get("source_field_types") or ()))


def _no_magnetic_source(configuration: dict) -> bool:
    return all(kind != "B" for kind in _source_types(configuration))


def _no_electric_source(configuration: dict) -> bool:
    types = _source_types(configuration)
    return bool(types) and all(kind == "B" for kind in types)


# --- THE IN-SEAM DEPOSIT, READ FROM THE SHIPPED FLAG RATHER THAN TRANSCRIBED ---
# `meep_gpu/deposit_repair.py` removes the refusal the two clauses above encode: the
# fused launch runs over the whole grid, the deposit points are saved immediately
# before it (`LeadingRepairPlan`, deposit_repair.py:249-274) and recomputed after the
# driver has injected, filled symmetry and cleared walls (`TrailingRepairPlan`,
# :276-297). Every fused-pair predicate routes its source question through
# `deposit_repair.seam_source_reasons(..., carries_repair=<module>.
# CARRIES_DEPOSIT_REPAIR)` (deposit_repair.py:210-247), so THE PRODUCT'S OWN CONSTANT
# is the answer and a boolean typed here would be a second, unchecked copy of it —
# which is exactly what the 2026-08-27 02:38 cut of this board was: it transcribed
# `no_magnetic_source` for every B-side product and kept printing that refusal after
# `metal_kernels/fused_magnetic_pair.py:204` had stopped making it. The clauses below
# IMPORT the shipped module and ask.


def _deposit_is_repairable(pair: str, configuration: dict) -> bool:
    """`deposit_repair.repairable`'s own refusals, read from the census
    configuration block.

    TWO OF THEM ARE D-SIDE ONLY. An off-diagonal chi1inv row makes ``update_E`` a
    stencil over the PARTNER components' volumes at shifted indices
    (stepping.py:1228-1251), so no repair over the deposit points can reconstruct it;
    an instantaneous chi2/chi3 is not the linear accumulation the repair inverts.

    TWO ARE ABOUT THE FOLD AND APPLY TO EITHER SEAM, added 2026-08-28 with
    ``repair_cells``. The repair now carries a deposit across a folded seam by
    restoring the cells ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` image
    it into, and it refuses any fold whose fill map it cannot read: the cylindrical
    r = 0 axis, whose below-axis ghost is the r_to_minus_r image rather than the near
    fill's cell 0 <- cell 2, and a folded axis storing no more than
    ``MIRROR_SOURCE_INDEX`` cells, where the row the near fill images does not exist.
    A board that admitted either would be over-counting against the shipped predicate,
    which is the one direction that reads as success.

    The remaining refusal — ``f_w_<component>`` unallocated — needs an engine object
    and is named in every affected product's ``not_evaluable``.
    """
    if pair == "D":
        if bool(configuration["has_offdiagonal_epsilon"]):
            return False
        if bool(configuration["has_nonlinearity"]):
            return False
    if bool(configuration["has_symmetry"]):
        if any(bool(flag) for flag in configuration["is_axis"]):
            return False
        for axis in _folds(configuration):
            if int(configuration["shape"][axis]) <= NEAR_SOURCE_INDEX:
                return False
    return True


def _carries_deposit_repair(module_path: str) -> bool:
    """The SHIPPED module's ``CARRIES_DEPOSIT_REPAIR``, imported live."""
    import importlib  # noqa: PLC0415
    return bool(getattr(importlib.import_module(module_path),
                        "CARRIES_DEPOSIT_REPAIR", False))


def _module_dotted_path(family: str) -> str:
    """The importable module behind a PRODUCTS key, parsed from its ``module`` field.

    The two differ for the Dcyl complex pair (family
    ``cylindrical_complex_fused_magnetic_pair``, module
    ``cylindrical_fused_magnetic_pair.py``), which is why this reads the field rather
    than assuming the family names the file.
    """
    import re as _re  # noqa: PLC0415
    field = PRODUCTS[family]["module"]
    match = _re.search(r"(meep_gpu/metal_kernels/[a-z0-9_]+)\.py", field)
    if not match:
        raise SystemExit(f"{family}: 'module' does not name a metal_kernels file: "
                         f"{field!r}")
    return match.group(1).replace("/", ".")


def _seam_deposit_clause(pair: str, module_path: str) -> Callable[[dict], bool]:
    """Does THIS product's shipped predicate clear the seam's source injection?

    No in-seam source of this seam's field type: cleared, exactly as before. One or
    more: cleared only if the module declares ``CARRIES_DEPOSIT_REPAIR`` AND
    `deposit_repair.repairable` does not refuse the configuration. That is
    `seam_source_reasons` evaluated over the columns a census block carries.
    """
    absent = _no_magnetic_source if pair == "B" else _no_electric_source

    def test(configuration: dict) -> bool:
        if absent(configuration):
            return True
        if not _carries_deposit_repair(module_path):
            return False
        return _deposit_is_repairable(pair, configuration)
    return test


#: What a census block cannot answer about a CARRIED deposit, stated once and added to
#: every product whose source clause now depends on the repair. Both are refusals the
#: shipped predicate can still make on an engine object, so every count this board
#: prints for a repair-carrying product is an UPPER BOUND — which `is_an_upper_bound`
#: already says for the board as a whole.
_DEPOSIT_NOT_EVALUABLE: Tuple[str, ...] = (
    "each in-seam source publishes the index it writes — `deposit_repair."
    "_deposit_index` reads `_point_ix/_point_iy/_point_iz` (deposit_repair.py:79-85) "
    "and `seam_source_reasons` refuses a source that returns None. All four source "
    "classes DECLARE those fields (sources.py:1616-1618, :1794-1796, :1900-1902, "
    ":2022-2024) but each sets them to None when the row's indices are empty on this "
    "chunk (:1660, :1827, :1941, :2067), which needs a built source to observe; the "
    "census records `n_sources` and `source_field_types` and stops there",
    "`f_w_<component>` is allocated for all three of the seam's constitutive targets "
    "— the third refusal in `deposit_repair.repairable` (deposit_repair.py:108-111)",
)


def _every_fold_is_mirror_metallic(configuration: dict) -> bool:
    """``folded_axis_kinds`` resolves MIRROR_METALLIC exactly where ``is_metallic``.

    triton_kernels/symmetry.py:465-473 — the split comes from
    ``stepping._stored_past_owned`` and ``grid.is_metallic`` must AGREE with it, a
    disagreement being refused. So the configuration's per-axis ``metallic`` triple
    IS the split, which is the derivation both 2026-08-19 censuses used.
    """
    return all(bool(configuration["metallic"][axis])
               for axis in _folds(configuration))


def _fold_stores_the_source_row(configuration: dict) -> bool:
    return all(int(configuration["shape"][axis]) > NEAR_SOURCE_INDEX
               for axis in _folds(configuration))


def _fold_and_wall_are_disjoint(configuration: dict) -> bool:
    """``zero_metal_axes`` (coverage.py:408-428) is ``has_metallic and is_metallic(a)
    and not is_mirrored(a)`` — FALSE on every folded axis by construction, so
    folded_fused_magnetic_pair's drift check is satisfied on every configuration and
    is evaluable exactly rather than approximately."""
    return True


def _no_polarization(configuration: dict) -> bool:
    return int(configuration["n_polarizations"]) == 0


def _no_scaled_electric_injection_on_a_conductive_run(configuration: dict) -> bool:
    """``_scaled_conductive_injection_reasons``' census-evaluable face — LIFTED.

    THE CLAUSE THIS FUNCTION USED TO EVALUATE PRICED THE CONDUCTIVE CELL AT 0 OF
    2: the driver's whole-volume condinv rescale (``array -= before; array *=
    condinv; array += before``) canonicalised every ``-0.0`` in the target D at
    cells no deposit closure can name, MEASURED as 2417 Dz words moved and 1231
    Ez words divergent on step 1 of a signed-zero seed. The driver now replays
    the rescale SPARSELY, at the deposit cells the sources publish
    (``_inject_electric_through_conductivity`` — snapshot at the published
    indices, inject, replay per deposit cell in the retired passes' exact
    operand order), leaving every other word untouched; the lift was taken on a
    re-measurement of the same signed-zero walk (byte-identical at every step,
    with the retired whole-volume passes replayed as an armed control that still
    diverges — the gate's ``lifted_refusal`` leg).

    WHAT THE SHIPPED PREDICATE STILL REFUSES is the driver's own fallback: a
    scaled source publishing NO deposit table (no ``_point_ix``), which no
    in-tree electric source class is — every source publishes its indices at
    setup, a fact the corpus-facing host suite pins. The census records
    ``source_field_types`` and stops — no per-source table column — so this
    clause returns True and the fallback question is carried in the product's
    ``not_evaluable`` block: on this corpus the answer is exact (all sources are
    in-tree classes and publish), and a future out-of-tree table-less source
    would be OVER-counted here and refused by the shipped predicate at plan
    time, which is the direction a board can carry when the refusal is loud.
    """
    del configuration  # every in-tree source publishes its table; see docstring
    return True


# --- the complex conductive D->E pair's own seam clauses --------------------
# complex_conductive_fused_pair_coverage conjoins the two halves' certified
# predicates — read from the census columns like every other product here — and
# then adds clauses of its own. Three are evaluable from the configuration block
# and are below; the rest are the halves' and are already read as columns.

def _pole_budget_fits(row: dict) -> bool:
    """No E component is driven by more than ``MAX_POLES`` susceptibilities.

    ``folded_fused_dispersive_pair`` BAKES the pole count into its source, so a
    component over the budget is not a slower kernel -- it is a kernel that subtracts
    the wrong number of arrays. The E half already refuses it; the clause is asked
    again here because on that family the count is a SOURCE specialisation.

    Read off the census's own ``polarization`` block (each state's ``driven`` list),
    which is the only column that carries it.
    """
    counts: Dict[str, int] = collections.Counter()
    for state in (row.get("polarization") or ()):
        for component in (state.get("driven") or ()):
            counts[str(component)] += 1
    return all(value <= 8 for value in counts.values())


def _has_a_polarization(configuration: dict) -> bool:
    """The INVERSE of :func:`_no_polarization` -- this family exists for pole runs.

    ``folded_fused_dispersive_pair``'s E half is ``folded dispersive PML E``, whose
    own predicate refuses a run with NO susceptibility by name ("folded ordinary
    constitutive owns update_E"). Stated here as its own clause because it is an
    inversion rather than an omission.
    """
    return int(configuration["n_polarizations"]) > 0


_H_TO_D_PROBE: Optional[Dict[str, dict]] = None


def _h_to_d_probe() -> Dict[str, dict]:
    """The H->D seam probe, read once. ``h_to_d_seam`` owns the campaign name."""
    global _H_TO_D_PROBE  # noqa: PLW0603 - read once from disk, then held
    if _H_TO_D_PROBE is None:
        _H_TO_D_PROBE = h_to_d_seam.probe_rows()
    return _H_TO_D_PROBE


def _no_standing_electric_withdraw(row: dict) -> bool:
    """``HOISTS_THE_WITHDRAW`` is False, so a row whose seam holds one is refused.

    A ROW CLAUSE, NOT A CONFIGURATION CLAUSE, and that is forced rather than stylistic:
    whether an integrated electric source's withdraw DOES WORK on a given row is a
    per-source measurement the seam probe took on the lifted simulation
    (``h_to_d_seam.probe_rows``'s ``withdraw_in_seam``), and the census's
    ``configuration`` block carries only ``source_field_types``. Inferring it from a
    kind name would be the over-covering guess the family's own predicate refuses to
    make (``withdraw_hoist.seam_withdraw_reasons``' ``undeclared`` branch).
    """
    probe = _h_to_d_probe()
    label = f"{row['leg']}:{row['row']}"
    if label not in probe:
        raise SystemExit(
            f"the H->D seam probe measured no row named {label!r}; a row it did not "
            f"lift may not be priced as admitted or as refused")
    # ``probe_rows`` returns the ``h_to_d_seam`` BLOCK per label, not the census row
    # that holds it, so ``withdraw_in_seam`` is a direct key here -- the same read
    # ``h_to_d_seam.price`` makes of the same mapping.
    return not probe[label]["withdraw_in_seam"]


#: A CELL's label, and the one spelling of it the ladder, the JSON and the console
#: all use, so a reader can grep a cell across the three without translating.
def _cell_label(cell: Tuple[str, str]) -> str:
    return f"{cell[0]} -> {cell[1]}"


def _selected_pair_is(cell: Tuple[str, str]) -> Tuple[str, Callable[[dict], bool]]:
    """The clause that admits exactly the rows whose ``plan_step`` SELECTED ``cell``.

    DERIVED FROM THE CELL KEY, NOT WRITTEN BESIDE IT. Every cell of a ``cells`` table
    gets this clause automatically (:func:`_cells_of`), which is what makes a per-cell
    credit honest: a product may serve more than one cell, but a ROW is credited only
    on the cell the census says the planner selected for it, so two cells of one family
    can never both claim the same row and the family's admitted count is a union of
    disjoint sets rather than a sum with an overlap hidden inside it.
    """
    pair = (cell[0], cell[1])

    def clause(row: dict) -> bool:
        selected = (row.get("plan_step") or {}).get("selected") or {}
        return (selected.get("update_H"), selected.get("step_D")) == pair

    return (f"the_{cell[0]}_to_{cell[1]}_cell".replace(" ", "_"), clause)


#: Parsed ``lift/rows.jsonl`` files, keyed by path. A cell clause is evaluated once per
#: census row; re-reading the file each time would be the same answer at a cost.
_DRIVEN_ROWS_CACHE: Dict[Path, Dict[str, dict]] = {}


def _driven_rows(path: Path) -> Dict[str, dict]:
    """The bound gate's own per-row lift records, indexed by census label.

    RAISES on an absent file rather than scoring zero. "The gate drove no rows" and
    "this board cannot find the gate's rows" are different measurements, and the
    second one silently costs a product every instance it earned.
    """
    if path not in _DRIVEN_ROWS_CACHE:
        if not path.is_file():
            raise SystemExit(
                f"a credited cell names {path} for its driven rows and the file is "
                f"absent, so the credit rests on nothing this board can read")
        records: Dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[row["label"]] = row
        _DRIVEN_ROWS_CACHE[path] = records
    return _DRIVEN_ROWS_CACHE[path]


def _driven_bit_identical_by(
        cell: Tuple[str, str], binding: Tuple[str, str],
        driven_rows: Tuple[str, str]) -> Tuple[str, Callable[[dict], bool]]:
    """A cell's row clause: the BOUND gate drove THIS row and got the same bytes.

    WHY A SECOND CELL OWES MORE THAN A FIRST ONE. A family's primary cell is credited
    on the strength of its predicate plus the family's own released weld, and the weld
    walk (:func:`_assert_every_credit_is_bound_to_released_bytes`) re-hashes what that
    run imported. A SECOND cell is a different configuration of the same module, and a
    release on the first says nothing about it -- the plain ``fused_hd_pair`` artifact
    holds neither nonlinear row. So an extra cell is credited row by row, against the
    lift records of the artifact that actually drove it: the row must be MEASURED, the
    named block must record that the predicate admitted it, that it was DRIVEN, and
    that the bytes were identical. Anything less is a coverage claim resting on a text
    identity, which is what held the nonlinear cell back for three weeks.

    The identity flag is spelled ``identical`` by the ``hd_tail_gate_common`` gates and
    ``bit_identical`` by ``gate_metal_fused_hd_pair_nonlinear``; both are read, and a
    block carrying neither is refused rather than admitted.

    THE RECORD MUST SAY WHICH CELL IT DROVE. A lift record is keyed by row label, and
    a row can select different arms under different censuses (the gates lift against
    the census they were written for; the board is cut over the newest). So the
    record's own ``census_selected`` is read and must equal ``cell`` -- a record that
    does not say is refused, and a record driven at another cell is not this cell's
    evidence however identical its bytes were.
    """
    artifact = RESULTS / binding[1]
    directory = artifact.parent if artifact.suffix == ".json" else artifact
    path = directory / driven_rows[0]
    block_key = driven_rows[1]

    def clause(row: dict) -> bool:
        record = _driven_rows(path).get(f"{row['leg']}:{row['row']}")
        if record is None or not record.get("measured"):
            return False
        block = record.get(block_key) or {}
        identity = block.get("identical")
        if identity is None:
            identity = block.get("bit_identical")
        if identity is None:
            raise SystemExit(
                f"{path}: the {block_key!r} block of {record['label']} records "
                f"neither `identical` nor `bit_identical`, so this board cannot tell "
                f"whether the row it would credit was driven to the same bytes")
        selected = record.get("census_selected")
        if not isinstance(selected, dict):
            raise SystemExit(
                f"{path}: the lift record for {record['label']} does not say which "
                f"arms the census selected when it was driven, so this board cannot "
                f"tell whether it was driven at {_cell_label(cell)}")
        if (selected.get("update_H"), selected.get("step_D")) != tuple(cell):
            return False
        return bool(block.get("predicate_admits") and block.get("driven")
                    and identity and block.get("passed", True))

    return (f"driven_bit_identical_by_{Path(binding[1]).parts[0]}", clause)


def _cells_of(spec: dict) -> Tuple[dict, ...]:
    """The cells a product is priced on -- one per (update_H arm, step_D arm) pair.

    ONE ENTRY PER ARM FAMILY IS THE INVARIANT THIS BOARD IS BUILT ON: ``_wiring``
    refuses a PRODUCTS key that is not a registered family and refuses any count
    mismatch between the two, and a dozen other sites key on the family name. So a
    family that serves a SECOND cell declares it INSIDE its entry rather than as a
    second entry, and every clause a cell is judged by -- its census half keys, its
    seam clauses, its row clauses and its binding -- belongs to THAT cell.

    THAT IS THE WHOLE OF WHY THE EARLIER ``extra_cells`` ATTEMPT FAILED. It unioned a
    second cell's rows into the first cell's ladder, so the primary cell's ``halves``
    (``constitutive@update_H``, false on every nonlinear row) refused the second cell's
    rows before its own clause was ever asked, and the union came out empty. Here no
    cell's clauses are ever evaluated against another cell's rows.

    A single-cell entry keeps writing ``halves``/``seam_clauses``/``row_clauses`` at
    the top level and is normalised here, so 44 of the 47 products are untouched.
    """
    declared = spec.get("cells")
    if declared is not None:
        if any(key in spec for key in ("halves", "seam_clauses", "row_clauses")):
            raise SystemExit(
                "a PRODUCTS entry declares `cells` AND a top-level halves/clause set; "
                "one of the two would be silently ignored, which is how a clause stops "
                "being applied without anyone deleting it")
        return tuple(declared)
    return ({"cell": tuple(spec["pair"]),
             "halves": tuple(spec["halves"]),
             "seam_clauses": tuple(spec["seam_clauses"]),
             "row_clauses": tuple(spec.get("row_clauses", ())),
             "binding": None,
             "driven_rows": None,
             "fingerprint": None},)


def ladder(record: Sequence[dict],
           spec: dict) -> "collections.OrderedDict[Tuple[str, str], Tuple[List[Tuple[str, int]], List[dict]]]":
    """Walk one product's clause ladder, per cell, over the census.

    Returns cell -> (the ladder as (clause, surviving) steps, the surviving rows).
    Declaration order is precedence order, and the first cell is the family's
    ``pair`` -- so a single-cell product's ladder is byte-for-byte what this board
    printed and filed before cells existed.

    THE CELL CLAUSE IS APPLIED TO EVERY CELL OF A MULTI-CELL PRODUCT, including the
    first: without it the primary cell would keep any row a second cell's arms select
    but whose primary halves happen to be covered, and the two cells' row sets would
    overlap. Disjointness is then ASSERTED rather than assumed.
    """
    cells = _cells_of(spec)
    multi = len(cells) > 1
    out: "collections.OrderedDict[Tuple[str, str], Tuple[List[Tuple[str, int]], List[dict]]]" = collections.OrderedDict()
    for cell in cells:
        surviving = list(record)
        steps: List[Tuple[str, int]] = []
        for key in cell["halves"]:
            surviving = [r for r in surviving if covered(r, key)]
            steps.append((key, len(surviving)))
        for label, test in cell["seam_clauses"]:
            surviving = [r for r in surviving if test(r["configuration"])]
            steps.append((label, len(surviving)))
        if multi:
            label, test = _selected_pair_is(cell["cell"])
            surviving = [r for r in surviving if test(r)]
            steps.append((label, len(surviving)))
        # ROW CLAUSES read the WHOLE census row rather than its `configuration`
        # block, and the split is not cosmetic: the pole budget lives in the
        # `polarization` block, which `configuration` does not carry. A clause that
        # needed it and was written as a configuration clause would raise on every
        # row, which reads as "the product admits nothing" rather than as a bug.
        row_clauses = list(cell["row_clauses"])
        if cell["driven_rows"]:
            # DERIVED FROM THE CELL'S OWN BINDING, never spelled beside it: the
            # artifact a cell is credited against is written once, in `binding`, and
            # the per-row clause reads that same run -- two spellings could disagree
            # and nothing would notice.
            if not cell["binding"]:
                raise SystemExit(
                    f"a cell {_cell_label(tuple(cell['cell']))} names driven rows "
                    f"{cell['driven_rows']} but no binding to read them from")
            row_clauses.append(_driven_bit_identical_by(
                tuple(cell["cell"]), cell["binding"], cell["driven_rows"]))
        for label, test in row_clauses:
            surviving = [r for r in surviving if test(r)]
            steps.append((label, len(surviving)))
        out[tuple(cell["cell"])] = (steps, surviving)
    seen: Dict[str, Tuple[str, str]] = {}
    for cell_key, (_steps, rows) in out.items():
        for row in rows:
            label = f"{row['leg']}:{row['row']}"
            if label in seen:
                raise SystemExit(
                    f"{label} is credited on two cells of one product "
                    f"({_cell_label(seen[label])} and {_cell_label(cell_key)}); a "
                    f"seam-instance served twice by one family would be counted twice")
            seen[label] = cell_key
    return out


def _the_gated_cylindrical_m0_cell(row: dict) -> bool:
    """The ``(update_H cylindrical m=0 -> step_D cylindrical m=0)`` cell, and only it.

    Read off the census's own ``plan_step.selected``, exactly as
    :func:`_selected_pair_is` reads a Cartesian cell. The Dcyl m = 0 product's
    predicate is the conjunction of the two ``cylindrical m=0`` arm predicates, so the
    two clauses say the same thing from two sides; the row clause is what keeps a
    complex-storage or Cartesian row out of this product's credit by name.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == ("cylindrical m=0",
                                                                  "cylindrical m=0")


def _the_conductive_h_to_d_cell(row: dict) -> bool:
    """The ``(update_H ordinary -> step_D conductive PML curl)`` cell, and only it.

    ONE ROW on this board (``tests:TestAdjointSolver.test_damping``) -- the SAME row
    whose D->E seam ``conductive_fused_electric_pair`` serves, which is what makes the
    arbitration on this cell a loss rather than a tie.

    THE TWO HALVES COME FROM DIFFERENT FAMILIES and that is the composer's own answer:
    an ELECTRIC conductivity changes the CURL recurrence only, and the curl predicate
    is sub-step aware about which curl, so ``step_B`` and both constitutive slots stay
    on their ordinary products. Read off the census's own ``plan_step.selected``,
    exactly as :func:`_selected_pair_is` reads a declared cell; a MAGNETIC
    conductivity puts the conductive curl on ``step_B`` and is kept out of this
    product's credit by this clause rather than by an argument.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == (
        "ordinary", "conductive PML curl")


def _the_bfast_h_to_d_cell(row: dict) -> bool:
    """The ``(update_H BFAST -> step_D BFAST)`` cell, and only it.

    ONE ROW on this board (``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``),
    and it carries no standing in-seam electric withdraw, so the product's served set is
    the one. A FOLDED BFAST run selects the ``folded BFAST`` arm on both slots and is a
    different cell and a different product; this clause is what keeps it out by name.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == ("BFAST", "BFAST")


def _the_folded_h_to_d_cell(row: dict) -> bool:
    """The ``(update_H folded -> step_D folded)`` cell, and only it.

    THE LARGEST CELL ON THIS BACKEND: 78 rows, of which 3 carry a standing in-seam
    electric withdraw and are refused beside this clause by
    :func:`_no_standing_electric_withdraw`, so the product's served set is 75.

    Read off the census's own ``plan_step.selected``, exactly as
    :func:`_selected_pair_is` reads a Cartesian cell. The folded product's
    predicate is the conjunction of the two ``folded`` arm predicates, so the two
    clauses say the same thing from two sides; the row clause is what keeps an
    UNFOLDED row out of this product's credit by name -- and the plain H->D product's
    clause keeps a FOLDED row out of its credit, which is the same inversion the two
    predicates carry.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == ("folded", "folded")


def _the_cylindrical_complex_h_to_d_cell(row: dict) -> bool:
    """The ``(update_H cylindrical complex -> step_D cylindrical complex)`` cell.

    Sixteen rows on this board; the two ``withdraw_seam`` rows of the same cell
    are refused by ``_no_standing_electric_withdraw`` beside this clause, so the
    product's served set is the sixteen and not the eighteen.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == (
        "cylindrical complex", "cylindrical complex")


def _the_complex_h_to_d_cell(row: dict) -> bool:
    """The ``(update_H complex/Bloch -> step_D complex/Bloch)`` cell.

    Seventeen rows on this board, and NOT ONE of them carries a standing in-seam
    electric withdraw (``h_to_d_seam_2026-09-04``: 0 of 17), which is what
    separates this cell from the Cartesian real one's 2 of 49. The withdraw
    clause sits beside this one anyway, because a row that grows an integrated
    source later must be refused rather than silently served.
    """
    selected = (row.get("plan_step") or {}).get("selected") or {}
    return (selected.get("update_H"), selected.get("step_D")) == (
        "complex/Bloch", "complex/Bloch")


def _no_folded_axis(configuration: dict) -> bool:
    """The mirror, refused BY NAME: `fill_symmetry_bc_D` (driver.py:3300) and
    `fill_folded_far_ghosts_D` (:3302) both sit inside this seam on a folded grid
    and this family carries neither."""
    return not _folds(configuration)


def _has_folded_axis(configuration: dict) -> bool:
    """The INVERSE of :func:`_no_folded_axis` — the two folded stencil welds exist
    for the fold and refuse an unfolded grid BY NAME, so that they and their
    unfolded twins do not overlap on one cell."""
    return bool(_folds(configuration))


def _has_offdiagonal_epsilon(configuration: dict) -> bool:
    """The off-diagonal chi1inv row the four stencil welds exist for.

    The certified off-diagonal constitutive emitter REFUSES an all-dead row mask
    by name ("that configuration belongs to the certified plain constitutive
    kernel"), so a row without one is not this product's even though the arm
    labels would place it in the cell.
    """
    return bool(configuration["has_offdiagonal_epsilon"])


def _no_metallic_axis(configuration: dict) -> bool:
    """The wall, refused BY NAME: `zero_metal_D` (driver.py:3301) writes stored
    cell 0 of every D component whose Yee shift on a walled axis is 0 and
    `update_E` then reads it. The family refuses rather than carrying, and the
    module docstring PRICES that: a metallic axis buys ZERO seam-instances on this
    corpus, which the ledger below reproduces."""
    return not any(bool(flag) for flag in configuration["metallic"])


def _fits_the_conductive_pair_ceiling(configuration: dict) -> bool:
    """complex_conductive_fused_pair.binding_count(pole_counts) <= 31.

    Evaluated at the WORST CASE the census can express — every one of the row's
    susceptibilities driving every one of the three components — so the clause is
    CONSERVATIVE: it can refuse a row the product would admit, never admit one the
    product would refuse. The corpus's rows in this cell carry ONE state, which
    lands at 22 bindings against a 31 ceiling.
    """
    from meep_gpu.metal_kernels import complex_conductive_fused_pair  # noqa: PLC0415
    poles = int(configuration["n_polarizations"])
    return complex_conductive_fused_pair.binding_count(
        (poles, poles, poles)) <= MAX_BUFFER_BINDINGS


def _fits_the_complex_chain_ceiling(configuration: dict) -> bool:
    """complex_fused_ade_chain.binding_count(poles, volume_sigmas) <= 31.

    Same conservatism as `_fits_the_fused_binding_ceiling`: the per-component pole
    count is at most the state count and the sigma KIND is not in the census, so
    both are taken at their maximum.
    """
    from meep_gpu.metal_kernels import complex_fused_ade_chain  # noqa: PLC0415
    poles = int(configuration["n_polarizations"])
    return complex_fused_ade_chain.binding_count(poles, poles) <= MAX_BUFFER_BINDINGS


# --- the E->P chain's own seam clauses -------------------------------------
# fused_ade_chain_coverage (meep_gpu/metal_kernels/fused_ade_chain.py) conjoins the
# two halves' certified predicates — read from the census columns like every other
# product here — and then adds exactly three clauses of its own. Two are evaluable
# from the configuration block and are below; the third is not and is named in
# `not_evaluable`.

def _no_offdiagonal_epsilon(configuration: dict) -> bool:
    """fused_ade_chain.py's first seam clause.

    An off-diagonal chi1inv row makes ``update_E(c)`` read the OTHER components'
    ``D - sum P`` volumes (stepping.py:991-997), so the per-component interleave
    this product performs is no longer the driver's order. Both E-half predicates
    already refuse it; the clause is asked again here because the FACT is
    different — this one is about the interleave, not about the arm's arithmetic.
    """
    return not bool(configuration["has_offdiagonal_epsilon"])


def _no_chi2_chi3(configuration: dict) -> bool:
    """fused_ade_chain.py's second seam clause, same reason as the row above."""
    return not bool(configuration["has_nonlinearity"])


def _fits_the_fused_binding_ceiling(arm: str) -> Callable[[dict], bool]:
    """fused_ade_chain.py's third: `binding_count(arm, poles, sigmas) <= 31`.

    Evaluated at the WORST CASE the configuration can express — every one of the
    row's susceptibilities driving this component, and every sigma a VOLUME. The
    census records `n_polarizations` (the state count) and, per state, which
    components it drives; the per-component pole count is at most the state count,
    and the sigma KIND is not in the census at all. Taking both at their maximum
    makes this clause CONSERVATIVE: it can refuse a row the product would admit,
    never admit one the product would refuse.

    Measured on this corpus the bracket never bites either way — the widest
    configuration (`stochastic_emitter*.py`, 6 states, dispersive arm) lands at
    exactly 31 bindings with every sigma a volume and 25 with none, so all ten
    rows this family claims admit at BOTH ends and the conservatism costs nothing
    here. That is a property of this corpus and is stated rather than assumed.
    """
    def test(configuration: dict) -> bool:
        from meep_gpu.metal_kernels import fused_ade_chain  # noqa: PLC0415
        poles = int(configuration["n_polarizations"])
        return fused_ade_chain.binding_count(arm, poles, poles) <= MAX_BUFFER_BINDINGS
    return test


#: WHERE A CELL'S CARRY OBSTRUCTION HAS ACTUALLY BEEN MEASURED, keyed by cell so a
#: citation cannot drift onto a different one. The carry column above says WHICH
#: cells run an uncarried in-seam pass; these rows say what it was measured to cost,
#: and only for the cells where someone has measured it.
CARRY_EVIDENCE: Dict[Tuple[str, str, str], str] = {
    ("B_to_H", "folded beta complex", "folded beta complex"): (
        "results/metal_folded_beta_complex_seam_2026-08-20/ (PASS, 7 legs): both "
        "reachable rows are MIRROR_PERIODIC folds, so fill_folded_far_ghosts_B "
        "(driver.py:3287) runs inside the seam and update_H reads what it writes. "
        "Dropping ONLY that fill moves H by 192 uint32 words at one seam and 5,235 "
        "over 12 complete steps, in every one of the 24 volumes, while the "
        "predicted 27-pointer signature COMPILES on the device at 28 bindings — so "
        "the obstruction is the CARRY, not the binding count"),
}


#: What every E->P product cannot check from a census block, stated once.
_ADE_CHAIN_NOT_EVALUABLE = (
    "the two halves' POLE ORDERS agree per component — the E chain's `_poles(fields)` "
    "partition against the ADE plan's entry order (fused_ade_chain.py, the pole-set "
    "clause). Both are `fields.polarizations` order filtered by `drives(component)`, "
    "so a disagreement needs an engine object to observe and the census carries only "
    "the state list; the plan builder CHECKS the identity rather than inheriting it",
)


#: FUSED FAMILIES THE LIVE ARM TABLE REGISTERS AND THIS CUT DOES NOT PRICE, named
#: so the completeness floor still bites for anything else. Both landed on this tree
#: on 2026-08-20 while this cut was being built — file mtimes 21:37 and 21:43,
#: against the 2026-08-19 census this walk scores against — and pricing another
#: family's clause ladder from its module without its own admission census is how a
#: wrong number gets published with a floor's blessing.
#:
#: WHAT THAT COSTS THE HEADLINE, stated rather than buried: whatever these two serve
#: is counted here as a GAP, so ``served_by_a_fused_product`` is a LOWER BOUND on the
#: board. It does not touch this cut's DELTA, which is measured against
#: ``../fusion_matrix_metal_2026-08-20_closed/`` — the same 14 products, the same
#: census, the same method, one clause different.
#: EMPTY IN THIS CUT, AND THE EMPTINESS IS THE POINT. The 2026-08-20 farcarry cut
#: named these two here because they landed in a SIBLING fork while it was being
#: built, which made its served total a LOWER BOUND. This cut merges the two forks,
#: so both are priced in PRODUCTS and the hatch has nothing left in it. `_wiring()`
#: now FAILS on a non-empty tuple rather than printing a lower-bound warning: a
#: warning is how a lower bound gets read as a measurement.
NOT_PRICED_HERE: Tuple[str, ...] = ()


PRODUCTS: Dict[str, dict] = {
    "fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("PML", "ordinary"),
        "module": "meep_gpu/metal_kernels/fused_magnetic_pair.py:465-527",
        "halves": ("pml_curl@step_B", "constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.fused_magnetic_pair")),
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline (fused_magnetic_pair.py:520-527)",
        ),
    },
    "complex_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("complex/Bloch", "complex/Bloch"),
        "module": "meep_gpu/metal_kernels/complex_fused_magnetic_pair.py:676-742",
        "halves": ("complex_pml_curl@step_B", "complex_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.complex_fused_magnetic_pair")),
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored "
            "(complex_fused_magnetic_pair.py:735-742)",
        ),
    },
    # ---- THE ONE PRODUCT THIS CUT CHANGES ------------------------------------
    # `every_fold_is_mirror_metallic` IS GONE, because the kernel now carries
    # `fill_folded_far_ghosts_B` inside the fused launch. The clause was never about
    # the fold: it was the far fill, refused by name because no product imaged the
    # top stored slot. Gate:
    # results/metal_folded_far_carry_2026-08-20/gate.json (verdict PASS, released).
    "folded_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("folded", "folded"),
        "module": ("meep_gpu/metal_kernels/folded_fused_magnetic_pair.py:"
                   "metal_folded_fused_magnetic_pair_coverage"),
        "halves": ("folded_pml_curl@step_B", "folded_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.folded_fused_magnetic_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("fold_and_wall_are_disjoint", _fold_and_wall_are_disjoint),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis "
            "(folded_fused_magnetic_pair.py:_mirror_phase_reasons) — but "
            "folded_axis_kinds, which the folded CURL half already calls, carries "
            "the same question (triton_kernels/symmetry.py:513-517), so a row "
            "failing it would already have failed the half",
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "the far carry's reflect-row clauses (_far_carry_reasons): the row "
            "0 <= n_full - stored + 2 < stored - 1, not 0, and stored - 1 != 2. The "
            "census block carries `shape` but not `shape_full`, so the row itself is "
            "not evaluable here. MEASURED INSTEAD, over every folded PERIODIC extent "
            "Grid will build (58 of them, cells 3..60): stored == ceil(n_full/2) + 2 "
            "and reflect == floor(n_full/2) with ZERO violations, minimum stored 4 — "
            "so all three clauses hold on every grid this engine can construct and "
            "the carry adds no admission question. "
            "(results/metal_folded_far_carry_2026-08-20/grid_reflect_row_law.json)",
            "the code and stepping._stored_past_owned agree per axis — "
            "folded_axis_kinds DERIVES the code from that predicate "
            "(triton_kernels/symmetry.py:465-473), so the clause can only fail if "
            "the grid's own two routes to the fold split disagree, which that "
            "function already refuses",
        ),
    },
    # ---- THE PRODUCT THIS CUT CHANGES, the D-side twin of the one above --------
    # `every_fold_is_mirror_metallic` IS GONE HERE TOO, and for the same reason and
    # in the same order: the hazard was CARRIED FIRST and the refusal retired after.
    # `metal_kernels/folded_fused_pair.py` now emits `fill_folded_far_ghosts_D`
    # inside the fused launch (REPLACES:306-307 is the five-tuple naming it;
    # docstring:64 "CARRIED INLINE as of 2026-08-21"), and the shipped predicate no
    # longer contains a folded-PERIODIC refusal at all: :1161-1163 calls
    # `_far_carry_reasons`, which asks what the carry's OWNERSHIP MOVE needs rather
    # than saying no. The 2026-08-21 cut of this script predates that edit and still
    # transcribed the pre-carry clause, so it priced this product at 2 where the
    # shipped predicate admits more.
    #
    # Gate: ../folded_far_carry_d_2026-08-21b/gate.json — VERDICT PASS, released,
    # 35/35 legs, 48 recorded sources, device legs including `xyz_all_folded_3d`.
    #
    # What replaces the clause is NOT EVALUABLE from the census block (it reads
    # grid.shape, stepping._stored_past_owned and stepping._far_reflect_rows), so it
    # is listed below — and, as on the B side, a not-evaluable entry is only honest
    # if the question behind it is SETTLED. It was settled by MEASUREMENT, one step
    # stronger than the B side's: ./measure_d_side_far_carry_law.py CALLS THE SHIPPED
    # `_far_carry_reasons` on every folded PERIODIC extent Grid can construct — 58 of
    # them, cells 3..60 — and records what it returns. ZERO reasons on all 58, zero
    # violations, minimum stored 4 (./d_side_far_carry_law.json).
    "folded_fused_pair": {
        "seam": "D_to_E",
        "pair": ("folded", "folded"),
        "module": ("meep_gpu/metal_kernels/folded_fused_pair.py:"
                   "metal_folded_fused_pair_coverage (:1115-1195)"),
        "halves": ("folded_pml_curl@step_D", "folded_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause("D", "meep_gpu.metal_kernels.folded_fused_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis "
            "(folded_fused_pair.py:_mirror_phase_reasons) — carried by the folded "
            "curl half too",
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "the far carry's own clauses (_far_carry_reasons, :1050-1113): that "
            "stepping._stored_past_owned AGREES with the axis code both ways, that a "
            "folded PERIODIC axis has a reflect row, and that the row satisfies "
            "0 <= row < stored - 1. The census block carries `shape` but not "
            "`shape_full` and holds no grid object, so none is evaluable here. "
            "MEASURED INSTEAD by calling the SHIPPED function on every folded "
            "PERIODIC extent Grid will build (58, cells 3..60): ZERO reasons "
            "returned on all 58, zero violations (./d_side_far_carry_law.json). "
            "Note the shipped D-side clause deliberately OMITS two of the magnetic "
            "twin's — reflect row != 0 and stored - 1 != 2 — because on the D side "
            "the near destination never lands on the far axis (:1063-1073); those "
            "two were measured true anyway on the same sweep.",
        ),
    },
    # =====================================================================
    # THE FOUR SCRATCH-OUTPUT STENCIL WELDS, 2026-09-01.
    #
    # These four close the cells this board has printed as STRUCTURALLY UNFUSABLE
    # ON ANY BACKEND since the off-diagonal round of 2026-08-20. That verdict was
    # measured on a fused kernel spliced VERBATIM from the two shipped bodies —
    # a splice in which the curl half writes D IN PLACE and the constitutive half
    # then reads its neighbours — and for that shape it is still exactly right.
    # It is NOT a property of the seam: these products write D and fu_D to
    # LAUNCH-LOCAL SCRATCH, re-derive every foreign tap from PRE-LAUNCH state
    # through the curl's own body as an inline function, and rotate the buffers
    # after the launch returns. Nothing written is ever read, so the grid-wide
    # barrier the verdict asked for is not wanted.
    #
    # THE SOURCE CLAUSE IS THE WHOLE DEMAND STORY and it does not move: an
    # off-diagonal chi1inv is one of the two shapes `deposit_repair.repairable`
    # refuses BY NAME, so every one of these families declares
    # CARRIES_DEPOSIT_REPAIR False and `_seam_deposit_clause` scores a row with an
    # in-seam electric source as UNCLEARED. That is why these entries admit 8 of
    # 16, 9 of 19, 1 of 2 and 1 of 3 rather than their cells outright.
    "offdiag_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("PML", "offdiag"),
        "module": ("meep_gpu/metal_kernels/offdiag_fused_electric_pair.py:"
                   "metal_offdiag_fused_electric_pair_coverage"),
        "halves": ("pml_curl@step_D", "offdiag_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.offdiag_fused_electric_pair")),
            ("has_offdiagonal_epsilon", _has_offdiagonal_epsilon),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (the wall-readability clause)",
            "which of the six chi1inv ROW SLOTS are live — the census records that "
            "an off-diagonal row exists, not which; the emitter specialises per "
            "row mask either way and the constitutive arm is certified on all of "
            "them",
        ),
    },
    "folded_offdiag_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("folded", "folded offdiag"),
        "module": ("meep_gpu/metal_kernels/folded_offdiag_fused_electric_pair.py:"
                   "metal_folded_offdiag_fused_electric_pair_coverage"),
        "halves": ("folded_pml_curl@step_D",
                   "folded_offdiag_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D",
                 "meep_gpu.metal_kernels.folded_offdiag_fused_electric_pair")),
            ("has_offdiagonal_epsilon", _has_offdiagonal_epsilon),
            ("has_folded_axis", _has_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis, and "
            "stepping._stored_past_owned / ._far_reflect_rows resolve the far "
            "slot — the census block carries `shape` but no grid object, so the "
            "far arm's inputs are not evaluable here. The closed form they feed "
            "is MEASURED against the array path's own three passes over 17 "
            "fixtures including an odd full count and a two-axis mixed-phase "
            "fold (results/metal_scratch_weld_closed_form_2026-09-01T2)",
            "which of the six chi1inv ROW SLOTS are live, and which folded axis "
            "carries the NEGATED ghost lane — both are emitter specialisations "
            "and both arms are certified across them",
        ),
    },
    "complex_no_pml_offdiag_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("complex no-PML curl", "complex no-PML off-diagonal"),
        "module": ("meep_gpu/metal_kernels/"
                   "complex_no_pml_offdiag_fused_electric_pair.py:"
                   "metal_complex_no_pml_offdiag_fused_electric_pair_coverage"),
        "halves": ("complex_plain_curl@step_D",
                   "complex_no_pml_offdiag@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D",
                 "meep_gpu.metal_kernels."
                 "complex_no_pml_offdiag_fused_electric_pair")),
            ("has_offdiagonal_epsilon", _has_offdiagonal_epsilon),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "the complex-multiply EXPANSION probe artifact is present and loads — "
            "which arm the numpy reference takes is a measured platform fact both "
            "halves already refuse to guess",
            "which of the six chi1inv ROW SLOTS are live",
        ),
    },
    "folded_complex_offdiag_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("folded complex", "complex folded off-diagonal PML E"),
        "module": ("meep_gpu/metal_kernels/"
                   "folded_complex_offdiag_fused_electric_pair.py:"
                   "metal_folded_complex_offdiag_fused_electric_pair_coverage"),
        "halves": ("folded_complex_pml_curl@step_D",
                   "complex_folded_offdiag@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D",
                 "meep_gpu.metal_kernels."
                 "folded_complex_offdiag_fused_electric_pair")),
            ("has_offdiagonal_epsilon", _has_offdiagonal_epsilon),
            ("has_folded_axis", _has_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "the FOLDED-COMPLEX expansion probe artifact is present and loads",
            "the far arm's grid inputs, as for the real folded weld above",
            "which of the six chi1inv ROW SLOTS are live",
        ),
    },
    "cylindrical_real_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("cylindrical m=0", "cylindrical m=0"),
        "module": "meep_gpu/metal_kernels/cylindrical_real_fused_magnetic_pair.py",
        "halves": ("cylindrical_real_curl@step_B",
                   "cylindrical_real_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.cylindrical_real_fused_magnetic_pair")),
            # The two fill passes must stay INERT: both return before touching a cell
            # unless grid.has_symmetry() (stepping.py:1481-1483, :1565-1566), which is
            # why this product's REPLACES names three driver passes and not five.
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline "
            "(cylindrical_real_fused_magnetic_pair.py, the wall-readability clause)",
        ),
    },
    # THE D->E TWIN OF THE ROW ABOVE, added 2026-08-31, and the ONLY product on this
    # board whose cell was refused by the BINDING CEILING rather than by a clause.
    # The 2026-08-30 cut scored it `UNFUSABLE ON METAL -- even at the largest sharing
    # measured on this seam at this arity (3) the fused signature needs 31 pointers,
    # over the 30 the platform allows`. Over by EXACTLY ONE, and the one is the radial
    # prefix the cylindrical curl binds and the Cartesian one does not.
    #
    # WHAT MOVED IS THE SIGNATURE AND NOT THE CEILING. `metal_kernels/
    # coefficient_pack.py` puts the curl half's six read-only per-axis PML coefficient
    # vectors in ONE buffer with six element offsets carried in the Params struct the
    # six scalars already ride in, so the shipped shape is 26 pointers. The gate
    # COMPILES both: the unpacked 32-binding signature is refused with the platform's
    # own "'buffer' attribute parameter is out of bounds" and the packed 27-binding one
    # compiles, launches and reads every struct field back including the six offsets.
    #
    # THE PREFIX IS DELIBERATELY NOT IN THE PACK. It is the one member of that
    # read-only set the DEVICE writes (the radial scan fills it in the launch before
    # this one), and folding a device-written volume in beside constants would let a
    # wrong offset in the SCAN corrupt an absorber permanently and silently.
    #
    # THE SOURCE CLAUSE IS THE PRODUCT HERE. All three corpus rows declare an ELECTRIC
    # source in the seam, so with CARRIES_DEPOSIT_REPAIR at False this row would price
    # at ZERO -- the exact reverse of its magnetic twin, where the same three rows are
    # electric-only and the clause costs nothing.
    "cylindrical_real_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("cylindrical m=0", "cylindrical m=0"),
        "module": "meep_gpu/metal_kernels/cylindrical_real_fused_electric_pair.py",
        "halves": ("cylindrical_real_curl@step_D",
                   "cylindrical_real_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.cylindrical_real_fused_electric_pair")),
            # The two fill passes must stay INERT: both return before touching a cell
            # unless grid.has_symmetry() (stepping.py:1481-1483, :1565-1566), which is
            # why this product's REPLACES names three driver passes and not five.
            ("no_folded_axis", lambda c: not _folds(c)),
            # RESTATED, not inherited, exactly as the plain D/E row restates it: this
            # family's kernel bakes `source = D`, and a board that read the clause off
            # the constitutive half would be inferring one module's coverage from
            # another's guard.
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline "
            "(cylindrical_real_fused_electric_pair.py, the wall-readability clause)",
        ),
    },
    # ---- carried verbatim from ../fusion_matrix_metal_2026-08-20c/ -----------
    # THE |m| >= 1 Dcyl CELL. Note the KEY is the arm FAMILY name and the MODULE is
    # cylindrical_fused_magnetic_pair.py; the two differ for this one product, which
    # is what the _wiring() floor below exists to catch.
    "cylindrical_complex_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("cylindrical complex", "cylindrical complex"),
        "module": "meep_gpu/metal_kernels/cylindrical_fused_magnetic_pair.py:666-853",
        "halves": ("cylindrical_complex_pml_curl@step_B",
                   "cylindrical_complex_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.cylindrical_fused_magnetic_pair")),
            # Both symmetry fills must be INERT. Grid refuses a mirror plane on a
            # Dcyl cell outright (grid.py:648-653), so unlike the Cartesian twins
            # there is no configuration on which this family could be handed a live
            # fill; the predicate re-checks anyway rather than reading its coverage
            # off another module's guard.
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline (the wall-readability clause)",
            "zero_metal_axes reports no wall on r or phi (the FORBIDDEN_WALL_AXES "
            "clause): on a Dcyl cell neither can be walled, and the r = 0 clear "
            "ZM_X would compile is a row the per-|m| axis rules own",
        ),
    },
    # THE D->E TWIN OF THE ROW ABOVE, added 2026-08-31, and the LARGEST cell this
    # board has ever priced as a gap: 16 seam-instances, more than any other.
    # The 2026-08-30 cut scored it `UNFUSABLE ON METAL -- ... the fused signature
    # needs 33 pointers, over the 30 the platform allows`. Over by THREE, and the
    # three are the E side's inverse-epsilon volumes: the MAGNETIC twin sits at
    # EXACTLY 31 bindings with zero headroom because the H constitutive brings none.
    #
    # WHAT MOVED IS THE SIGNATURE AND NOT THE CEILING, and it is the SAME move that
    # cleared the m = 0 row above: `metal_kernels/coefficient_pack.py` puts the curl
    # half's six read-only per-axis PML coefficient vectors in ONE buffer with six
    # element offsets in the Params struct the eight non-pointer arguments already
    # ride in, so the shipped shape is 28 pointers. ONE TECHNIQUE, TWO CELLS -- which
    # is what makes packing a tool on this backend rather than a rescue for one
    # signature. The gate COMPILES four signatures (41 separate-scalar, 34 unpacked,
    # a synthetic 32 and the shipped 29) and BISECTS the ceiling on this host.
    #
    # THE SOURCE CLAUSE IS THE PRODUCT HERE, exactly as on the m = 0 row: all sixteen
    # corpus rows declare an ELECTRIC source in the seam, so with
    # CARRIES_DEPOSIT_REPAIR at False this row would price at ZERO.
    "cylindrical_complex_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("cylindrical complex", "cylindrical complex"),
        "module": "meep_gpu/metal_kernels/cylindrical_fused_electric_pair.py",
        "halves": ("cylindrical_complex_pml_curl@step_D",
                   "cylindrical_complex_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.cylindrical_fused_electric_pair")),
            # Both symmetry fills must be INERT. Grid refuses a mirror plane on a
            # Dcyl cell outright (grid.py:648-653), so unlike the Cartesian twins
            # there is no configuration on which this family could be handed a live
            # fill; the predicate re-checks anyway.
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (the wall-readability clause)",
            "zero_metal_axes reports no wall on r or phi (the FORBIDDEN_WALL_AXES "
            "clause): on a Dcyl cell neither can be walled, and the r = 0 clear "
            "would be a row the per-|m| axis rules own",
        ),
    },
    # ---- carried verbatim from ../fusion_matrix_metal_2026-08-20_foldedcomplex/
    # The folded COMPLEX B/H pair, the intersection of folded and complex. Its
    # clause list is IDENTICAL to folded_fused_magnetic_pair's because the seam is
    # the same seam: the fold reaches this kernel through the stored extent and the
    # parity word, neither of which is a new admission question.
    "folded_complex_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("folded complex", "folded complex"),
        "module": ("meep_gpu/metal_kernels/folded_complex_fused_magnetic_pair.py:"
                   "metal_folded_complex_fused_magnetic_pair_coverage"),
        "halves": ("folded_complex_pml_curl@step_B",
                   "folded_complex_constitutive@update_H"),
        # THE THIRD CARRY OF 2026-08-21, and the module ASKED FOR THIS RUN BY NAME.
        # `folded_complex_fused_magnetic_pair.py` now emits `fill_folded_far_ghosts_B`
        # inside the fused launch (REPLACES:284-285 is the five-tuple naming it;
        # docstring:180 "CARRIED INLINE as of 2026-08-21") and its shipped predicate
        # calls `_far_carry_reasons` (:1512) where the blanket refusal used to sit.
        # Its own docstring (:18-29) leaves the pre-carry ladder in place and says
        # why: "a new cell count is a matrix RUN, not an edit to this table, and none
        # has been made since". This is that run. It also states the size of the move
        # in advance — "the two rows it cost are no longer refused BY IT" — which this
        # walk reproduces rather than assumes.
        #
        # Gate: ../metal_folded_complex_far_carry_2026-08-21e/gate.json — VERDICT
        # PASS, released, 72/72 legs, 46 recorded sources, planted defect flips.
        # Hazard carried FIRST, refusal retired after; the only legitimate order.
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.folded_complex_fused_magnetic_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("fold_and_wall_are_disjoint", _fold_and_wall_are_disjoint),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis "
            "(folded_complex_fused_magnetic_pair.py:_mirror_phase_reasons) — but "
            "folded_axis_kinds, which the folded complex CURL half already calls, "
            "carries the same question, so a row failing it would already have "
            "failed the half",
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "the far carry's own clauses (_far_carry_reasons, :1372ff), which read "
            "grid.shape, stepping._stored_past_owned and stepping._far_reflect_rows "
            "— none of them in the census block. MEASURED INSTEAD by calling the "
            "SHIPPED function on every folded PERIODIC extent Grid will build (58, "
            "cells 3..60): ZERO reasons on all 58 "
            "(./d_side_far_carry_law.json, column "
            "shipped_folded_complex_B_far_carry_reasons).",
        ),
    },
    # ---- ADDED 2026-08-27: the rank-1 buildable gap of
    # results/metal_buildable_gaps_2026-08-27/gaps.json, built and gated the same day.
    # It is folded_complex_fused_magnetic_pair with ONE EMITTER SWAPPED (the
    # beta_fused_magnetic_pair idiom applied to the folded complex seam), so its clause
    # ladder is folded_beta's two SHIPPED sub-step predicates plus the folded seam
    # clauses -- the same three the folded complex parent carries, and for the same
    # reasons. Gate: ../metal_folded_beta_complex_fused_pair_2026-08-27/
    # gate_metal_folded_beta_complex_fused_magnetic_pair.json (PASS, released, 71/71).
    # NOT welded in fingerprints.json: that edit is reserved, so the credit is bound
    # through the gate artifact in RELEASE_BINDING below instead.
    "folded_beta_complex_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("folded beta complex", "folded beta complex"),
        "module": ("meep_gpu/metal_kernels/folded_beta_complex_fused_magnetic_pair.py:"
                   "metal_folded_beta_complex_fused_magnetic_pair_coverage"),
        "halves": ("folded_beta_bloch_pml_curl@step_B",
                   "folded_beta_complex_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.folded_beta_complex_fused_magnetic_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("fold_and_wall_are_disjoint", _fold_and_wall_are_disjoint),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis "
            "(folded_beta_complex_fused_magnetic_pair.py:_mirror_phase_reasons) -- "
            "but folded_axis_kinds, which the folded beta CURL half already calls, "
            "carries the same question, so a row failing it would already have "
            "failed the half",
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline "
            "(folded_beta_complex_fused_magnetic_pair.py:1028-1031)",
            "the far carry's own clauses (_far_carry_reasons, :1023), which read "
            "grid.shape, stepping._stored_past_owned and stepping._far_reflect_rows "
            "-- none of them in the census block. The folded COMPLEX parent measured "
            "the SAME shipped function over every folded PERIODIC extent Grid will "
            "build (58, cells 3..60) with ZERO reasons "
            "(../fusion_matrix_metal_2026-08-26_canonical/../"
            "fusion_matrix_metal_2026-08-21_farcarry/d_side_far_carry_law.json)",
            "the two expansion artifacts (_expansion_reasons): this product needs "
            "BOTH folded_beta's and folded_complex's, because the fill arithmetic it "
            "carries inline is folded_complex's while the curl is folded_beta's; the "
            "census block carries neither probe",
        ),
    },
    # ---- carried verbatim from ../fusion_matrix_metal_2026-08-20_belowcut/ ----
    # THE THREE BELOW-THE-CUT WELDS. Each is the shipped magnetic weld's own
    # construction with ONE emitter swapped (or, for the nonlinear one, with nothing
    # swapped at all), so each product's clause ladder is the two halves' SHIPPED
    # sub-step predicates plus the same seam clauses.
    "nonlinear_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("nonlinear PML curl", "nonlinear PML magnetic"),
        "module": "meep_gpu/metal_kernels/nonlinear_fused_magnetic_pair.py",
        "halves": ("nonlinear_run_pml_curl@step_B",
                   "nonlinear_run_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.nonlinear_fused_magnetic_pair")),
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline (inherited from fused_magnetic_pair.py:520-527)",
        ),
    },
    "beta_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("special_kz real beta", "special_kz real beta"),
        "module": "meep_gpu/metal_kernels/beta_fused_magnetic_pair.py",
        "halves": ("beta_pml_curl@step_B", "beta_run_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.beta_fused_magnetic_pair")),
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored",
        ),
    },
    "bfast_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("BFAST", "BFAST"),
        "module": "meep_gpu/metal_kernels/bfast_fused_magnetic_pair.py",
        "halves": ("bfast_pml_curl@step_B", "bfast_run_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause("B", "meep_gpu.metal_kernels.bfast_fused_magnetic_pair")),
            ("no_folded_axis", lambda c: not _folds(c)),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "f_bfast_Bx/By/Bz are allocated — this weld binds the Tustin state "
            "directly and the census records bfast_active, not the allocation",
        ),
    },
    # ---- ADDED 2026-08-28: THE LARGEST UNSERVED CELL ON THIS BOARD ------------
    # D->E (PML, ordinary), 26 of 26 rows. It sat in the BINDING-MARGINAL bucket
    # until the arity block above narrowed its bracket from 28..31 to exactly 30,
    # and 30 is the ceiling: `meep_gpu/metal_kernels/fused_electric_pair.py` binds
    # 30 pointers plus one packed `constant Params&` for 31 of the 31 attributes the
    # platform allows, with zero headroom. THE PREDICTION WAS CONFIRMED BY
    # COMPILATION rather than by re-deriving it: 24 specialisations compiled on this
    # Mac's MPS, and the gate's `binding_ceiling` leg additionally compiles one
    # pointer PAST the shipped shape and requires the failure.
    #
    # Its clause ladder is the two halves' SHIPPED sub-step predicates
    # (`pml_curl_coverage(..., "step_D")` and `constitutive_coverage(..., "E")`,
    # which are the `PML` and `ordinary` arms' own bodies) plus the seam clauses
    # below -- the same construction the magnetic twin uses on the other seam.
    #
    # Gate: ../metal_fused_electric_pair_2026-08-28/gate.json -- VERDICT PASS,
    # released, 48/48 legs, 91 recorded sources, 16 shader + 2 host mutations all
    # caught, the byte-neutral control uncaught, and the deposit null control
    # diverging.
    "fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("PML", "ordinary"),
        "module": ("meep_gpu/metal_kernels/fused_electric_pair.py:"
                   "metal_fused_electric_pair_coverage"),
        "halves": ("pml_curl@step_D", "constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause("D", "meep_gpu.metal_kernels.fused_electric_pair")),
            ("no_folded_axis", _no_folded_axis),
            # RESTATED, not inherited. `constitutive_coverage("E")` already refuses a
            # registered susceptibility, and the census column for that half carries
            # the refusal -- but the family's own predicate asks again because ITS
            # kernel bakes `source = D`, and a board that read this clause off the
            # half would be inferring one module's coverage from another's guard.
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (fused_electric_pair.py, the wall-readability "
            "clause)",
        ),
    },
    # ---- THE FOURTH SEAM'S FIRST PRODUCT, released 2026-09-06 -----------------
    # Device gate: ../metal_fused_hd_pair_2026-09-07_wired/gate.json (VERDICT PASS,
    # released). Legs on this Mac's MPS: the synthetic fixture's eight boundary
    # specialisations at the full 60-step budget against four references (the array
    # path, the certified singles, the composition the composer installs today, and
    # the same slots dispatched unfused), 21 shader mutations and four host mutations
    # all caught, a byte-neutral control uncaught, a binding ceiling pinned as an
    # EQUALITY at 31 of 31, and the corpus cell driven row by row.
    #
    # WHAT `served` MEANS FOR THIS ROW OF THE TABLE, said plainly because it is unlike
    # every other product here: `INSTALLABLE = False` on a measured arbitration
    # verdict, so the composer refuses this product BY NAME on every configuration.
    # The board credits the instances its predicate admits, and the kernel executes
    # NOWHERE. `served_in_dispatch` is unaffected -- it is 0 on this backend for the
    # structural reason the whole Metal board reports, and it would be 0 for this
    # product even on a backend the fast path planned against.
    "fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("ordinary", "PML"),
        "module": ("meep_gpu/metal_kernels/fused_hd_pair.py:"
                   "metal_fused_hd_pair_coverage"),
        # THE DRIVER'S ORDER, WHICH IS THE PREDICATE'S: the constitutive half runs
        # first (driver.py:3311) and the curl second (:3315), so the ladder refuses in
        # the order a reader would meet the refusals.
        # TWO CELLS, EACH JUDGED BY ITS OWN CLAUSES AND BOUND TO THE ARTIFACT THAT
        # DROVE IT. The predicate admits a chi2/chi3 run through
        # `nonlinear_update_e`'s spine arms -- the SAME two kernels under a scope view
        # -- and `launch.FUSED_PAIR_EXTRA_ARMS` carries the matching absorb row. The
        # plain family artifact holds NEITHER nonlinear row, so that cell is credited
        # against its own gate's lift records rather than on the family's weld.
        "cells": (
            {"cell": ("ordinary", "PML"),
             "halves": ("constitutive@update_H", "pml_curl@step_D"),
             "seam_clauses": (),
             "row_clauses": (
                 ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
             ),
             "binding": None, "driven_rows": None, "fingerprint": None},
            {"cell": ("nonlinear PML magnetic", "nonlinear PML curl"),
             "halves": ("nonlinear_run_constitutive@update_H",
                        "nonlinear_run_pml_curl@step_D"),
             "seam_clauses": (),
             "row_clauses": (
                 ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
             ),
             "binding": ("gate", "metal_fused_hd_pair_nonlinear_2026-10-04_g13s"),
             "driven_rows": ("lift/rows.jsonl", "metal_fused_hd_pair_nonlinear_gate"),
             "fingerprint": "metal_fused_hd_pair_nonlinear_device_gate"},
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults -- the magnetic deposit is "
            "one seam earlier (driver.py:3283-3284) and the electric one is one seam "
            "later (:3317-3322) -- so this family sets CARRIES_DEPOSIT_REPAIR False "
            "and consults deposit_repair not at all. There is therefore no deposit "
            "clause here to be unevaluable",
            "the nonlinear widening (nonlinear_update_e's spine arms on both slots, "
            "FUSED_PAIR_EXTRA_ARMS) is admitted by the shipped predicate AND, since "
            "2026-09-07, driven: gate_metal_fused_hd_pair_nonlinear.py takes a "
            "complete driver step on six chi2/chi3 fixtures and both corpus rows to "
            "zero differing words "
            "(first driven 2026-09-07; the run this cell is bound to is named in the "
            "`cells` table above and in release_binding_per_cell). The `cells` table "
            "credits it row by row against that artifact's own lift records; what is "
            "still NOT evaluable from the census block is the composer's arbitration, "
            "which that gate measures separately (zero installed pairs on every "
            "nonlinear fixture) and which changes no coverage number here",
        ),
    },
    # ---- THE FOURTH SEAM'S LAST TWO SMALL CELLS, gated 2026-09-07 ---------------
    # Two products, not one, and that was MEASURED rather than assumed. Both lift
    # `fused_hd_pair`'s `h_cell` whole on the constitutive side -- the census gives
    # both cells `shaders.constitutive_H` there -- and they differ only in the CURL
    # emitter. What makes them two products is that the two curls bring different
    # signatures: the conductive one adds three in-place histories and six read-only
    # conductivity VOLUMES, the BFAST one adds three Tustin states and six host-rounded
    # SCALARS carried in `Params`. One `resolve()` spanning them would have to switch
    # signature, pack layout and struct shape, which is two products wearing one name.
    #
    # THE CONDUCTIVE CELL IS THE ONE WHOSE TWO HALVES COME FROM DIFFERENT FAMILIES:
    # `constitutive_coverage(..., "H")` (the `ordinary` arm) beside
    # `metal_conductive_pml_curl_coverage(..., "step_D")`. That pairing is the shipped
    # composer's own answer -- `metal_composition_matrix`'s
    # `cart_pml_conductive_electric` row pins it, and the gate's `arm_pairing` leg
    # re-derives it from `plan_step` and requires the ORDINARY curl and the CONDUCTIVE
    # step_B to refuse the same fixture, which is the other half of the partition.
    #
    # Device gates: ../metal_conductive_fused_hd_pair_2026-09-07/flush/gate.json and
    # ../metal_bfast_fused_hd_pair_2026-09-07/flush/gate.json (each also run under a
    # resolved `keep` from a policy-separated cache, where every fixture is refused BY
    # NAME because the MPS executor cannot honour that policy -- recorded in the
    # `keep/` artifact, which releases nothing because there is nothing to release).
    #
    # `INSTALLABLE = False` on both, on measured arbitrations that DIFFER: on the
    # conductive cell BOTH neighbours install (`fused_magnetic_pair` +
    # `conductive_fused_electric_pair`), so the product is a LOSS; on the BFAST cell
    # only `bfast_fused_electric_pair` does -- `bfast_fused_magnetic_pair` carries no
    # FUSED_PAIR_ARMS row -- so it is a TIE, resolved for the incumbent. `served` here
    # is a PREDICATE verdict and the kernels execute NOWHERE outside their own gates.
    # THE THREE H_to_D CELLS THE 2026-09-07 WIRING ROUND BUILT AND THE 2026-09-08 WELD
    # ROUND EARNED. Each was priced only after its gate ran against the WIRED bytes and
    # `mint_metal_weld.py` wrote its ledger entry from that run -- the board refuses to
    # credit a family that declares WELD_OWED, which is what held these back.
    # The halves are the census keys the module's own coverage helpers conjoin, read off
    # the source rather than inferred from the cell labels.
    "beta_complex_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("special_kz complex beta", "special_kz complex beta"),
        "module": ("meep_gpu/metal_kernels/beta_complex_fused_hd_pair.py:"
                   "metal_beta_complex_fused_hd_pair_coverage"),
        # TWO CELLS: the module emits BOTH the plain and the folded variant (its own
        # VARIANTS/VARIANT_CELLS), and the released artifact drove all four rows --
        # three folded, one plain -- to zero differing words. The folded cell carries
        # the per-row check against those lift records; the plain cell is the family's
        # own binding.
        "cells": (
            {"cell": ("special_kz complex beta", "special_kz complex beta"),
             "halves": ("beta_run_complex_constitutive@update_H",
                        "beta_bloch_pml_curl@step_D"),
             "seam_clauses": (), "row_clauses": (),
             "binding": None, "driven_rows": None, "fingerprint": None},
            {"cell": ("folded beta complex", "folded beta complex"),
             "halves": ("folded_beta_complex_constitutive@update_H",
                        "folded_beta_bloch_pml_curl@step_D"),
             "seam_clauses": (),
             "row_clauses": (
                 ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
             ),
             "binding": ("gate", "metal_beta_complex_fused_hd_pair_2026-10-04_g13s"),
             "driven_rows": ("lift/rows.jsonl", "gate"),
             "fingerprint": "metal_beta_complex_fused_hd_pair_device_gate"},
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults -- the magnetic deposit is "
            "one seam earlier (driver.py:3283-3284) and the electric one is one seam "
            "later (:3317-3322) -- so this family sets CARRIES_DEPOSIT_REPAIR False "
            "and consults deposit_repair not at all. There is no deposit clause here "
            "to be unevaluable",
        ),
    },
    "beta_real_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("special_kz real beta", "special_kz real beta"),
        "module": ("meep_gpu/metal_kernels/beta_real_fused_hd_pair.py:"
                   "metal_beta_real_fused_hd_pair_coverage"),
        # TWO CELLS, the same shape as the complex sibling: the artifact drove one
        # plain row and one folded row, both to zero differing words.
        "cells": (
            {"cell": ("special_kz real beta", "special_kz real beta"),
             "halves": ("beta_run_constitutive@update_H", "beta_pml_curl@step_D"),
             "seam_clauses": (), "row_clauses": (),
             "binding": None, "driven_rows": None, "fingerprint": None},
            {"cell": ("folded beta real", "folded beta real"),
             "halves": ("folded_beta_constitutive@update_H",
                        "folded_beta_pml_curl@step_D"),
             "seam_clauses": (),
             "row_clauses": (
                 ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
             ),
             "binding": ("gate", "metal_beta_real_fused_hd_pair_2026-10-04_g13s"),
             "driven_rows": ("lift/rows.jsonl", "gate"),
             "fingerprint": "metal_beta_real_fused_hd_pair_device_gate"},
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults -- the magnetic deposit is "
            "one seam earlier (driver.py:3283-3284) and the electric one is one seam "
            "later (:3317-3322) -- so this family sets CARRIES_DEPOSIT_REPAIR False "
            "and consults deposit_repair not at all. There is no deposit clause here "
            "to be unevaluable",
        ),
    },
    "folded_complex_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("folded complex", "folded complex"),
        "module": ("meep_gpu/metal_kernels/folded_complex_fused_hd_pair.py:"
                   "metal_folded_complex_fused_hd_pair_coverage"),
        "halves": ("folded_complex_constitutive@update_H",
                   "folded_complex_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (),
        "not_evaluable": (
            "nothing is INJECTED between the two consults -- the magnetic deposit is "
            "one seam earlier (driver.py:3283-3284) and the electric one is one seam "
            "later (:3317-3322) -- so this family sets CARRIES_DEPOSIT_REPAIR False "
            "and consults deposit_repair not at all. There is no deposit clause here "
            "to be unevaluable",
        ),
    },
    "conductive_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("ordinary", "conductive PML curl"),
        "module": ("meep_gpu/metal_kernels/conductive_fused_hd_pair.py:"
                   "metal_conductive_fused_hd_pair_coverage"),
        # THE DRIVER'S ORDER, WHICH IS THE PREDICATE'S: constitutive first
        # (driver.py:3311), curl second (:3315). `conductive_pml_curl` is the key the
        # census MEASURES the conductive curl arm as on `step_D`; `constitutive` is the
        # ordinary arm's own key on `update_H`, the same one `fused_hd_pair` uses.
        "halves": ("constitutive@update_H", "conductive_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_conductive_h_to_d_cell", _the_conductive_h_to_d_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults -- the magnetic deposit is "
            "one seam earlier (driver.py:3283-3284) and the electric one is one seam "
            "later (:3317-3322) -- so this family sets CARRIES_DEPOSIT_REPAIR False "
            "and consults deposit_repair not at all. There is no deposit clause here "
            "to be unevaluable",
            "the per-component conductivity allocation (f_cond_D*, condfac_for, "
            "condinv_for): the census records that a target is conductive, not which "
            "volumes the engine allocated for it. The plan aliases a LOSSLESS "
            "target's history slot to that target's own displacement -- "
            "conductive_pml's own construction -- and asserts the emitted text names "
            "that pointer nowhere",
            "the two coefficient packs' contents. The census carries no PML profile, "
            "so 'the packed buffer is the separate vectors' words end to end' is the "
            "gate's pack_bytes leg and not a clause here",
        ),
    },
    "bfast_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("BFAST", "BFAST"),
        "module": ("meep_gpu/metal_kernels/bfast_fused_hd_pair.py:"
                   "metal_bfast_fused_hd_pair_coverage"),
        "halves": ("bfast_run_constitutive@update_H", "bfast_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_bfast_h_to_d_cell", _the_bfast_h_to_d_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults, so this family sets "
            "CARRIES_DEPOSIT_REPAIR False and consults deposit_repair not at all",
            "f_bfast_Dx/By/Bz are allocated -- this weld binds the Tustin state "
            "directly and the census records bfast_active, not the allocation "
            "(the same note bfast_fused_magnetic_pair carries)",
            "the six host-rounded k1/k2 scalars: the census carries "
            "bfast_scaled_k but not the per-target term tables, so 'the baked words "
            "are stepping's own, D-side negated' is the gate's tustin_scalars leg "
            "and not a clause here",
        ),
    },
    # ---- THE FOURTH SEAM'S LARGEST CELL, the fold, gated 2026-09-07 -------------
    # Device gate: ../metal_folded_fused_hd_pair_2026-09-07_wired/gate.json. ONE
    # dispatch, the plain H->D product's shape exactly (the fold adds no kernel
    # argument on this backend), over the folded curl emitter and the certified
    # constitutive body. 78 rows in the cell -- the largest on this backend, against
    # the Cartesian weld's 49 -- of which 3 carry a standing in-seam electric withdraw
    # and are refused BY NAME, so the credit is 75.
    #
    # `INSTALLABLE = False` on a measured arbitration, and on this cell it is not even
    # close: the released `folded_fused_magnetic_pair` serves `B_to_H` on 78 of 78
    # rows, so installing this product is a LOSS on the 67 rows where `D_to_E` is also
    # served and a TIE on the other 11, and a gain on none. `served` here is therefore
    # a PREDICATE verdict and nothing more -- the same reading `fused_hd_pair` carries
    # above -- and the kernel executes NOWHERE outside its own gate and tests.
    "folded_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("folded", "folded"),
        "module": ("meep_gpu/metal_kernels/folded_fused_hd_pair.py:"
                   "metal_folded_fused_hd_pair_coverage"),
        # THE DRIVER'S ORDER, WHICH IS THE PREDICATE'S: constitutive first
        # (driver.py:3311), curl second (:3315).
        # `folded_pml_curl`, which is what the census MEASURES the folded curl arm as
        # on both curl slots (the same key the folded D->E products above use on
        # `step_D`). A `folded_curl@step_D` spelling stood here on the way in and the
        # board refused it BY NAME -- a half key nothing measured would price this
        # product at zero admitted rows and read as a product nothing admits.
        "halves": ("folded_constitutive@update_H", "folded_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_folded_h_to_d_cell", _the_folded_h_to_d_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults, and on a FOLD that is a "
            "stronger statement than on an unfolded grid: fill_symmetry_bc_B, "
            "zero_metal_B and fill_folded_far_ghosts_B all close BEFORE the update_H "
            "consult (driver.py:3305-3310) and their three D twins all open AFTER the "
            "step_D consult (:3325-3330). So this family sets CARRIES_DEPOSIT_REPAIR "
            "False, consults deposit_repair not at all, and carries no fill -- there "
            "is no deposit clause here to be unevaluable and no carried-fill clause "
            "either",
        ),
    },
    # ---- THE FOURTH SEAM'S SECOND PRODUCT, the Dcyl m = 0 cell, gated 2026-09-06 ---
    # Device gate: ../metal_cylindrical_real_fused_hd_pair_2026-09-07_cyl/gate.json (the
    # wiring round's fleet re-cut; first released at ../..._2026-09-06_cyl/).
    # TWO LAUNCHES (the constitutive + the radial prefix over the RECOMPUTED Hy, then
    # the certified cylindrical curl), replacing the three the certified singles need at
    # the seam. `INSTALLABLE = False` on a measured arbitration TIE: the released D->E
    # cylindrical pair installs first and holds step_D on every row of the cell, so the
    # board credits the instances the predicate admits and the composer installs the
    # product nowhere -- the same reading `fused_hd_pair` carries above.
    "cylindrical_real_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("cylindrical m=0", "cylindrical m=0"),
        "module": ("meep_gpu/metal_kernels/cylindrical_real_fused_hd_pair.py:"
                   "metal_cylindrical_real_fused_hd_pair_coverage"),
        # THE DRIVER'S ORDER, WHICH IS THE PREDICATE'S: constitutive first, curl second.
        "halves": ("cylindrical_real_constitutive@update_H",
                   "cylindrical_real_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_gated_cylindrical_m0_cell", _the_gated_cylindrical_m0_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults, so this family sets "
            "CARRIES_DEPOSIT_REPAIR False and consults deposit_repair not at all; "
            "there is no deposit clause here to be unevaluable",
        ),
    },
    # ---- THE FOURTH SEAM'S THIRD PRODUCT, the complex Dcyl cell (2026-09-06) ---
    # TWO launches, not one: launch 1 folds the pointwise complex H constitutive
    # with the column-leader device scan of the recomputed Hy (the radial prefix
    # the array path scans on the HOST for this family), launch 2 is the certified
    # complex cylindrical curl unchanged. Device gate:
    # ../metal_cylindrical_complex_fused_hd_pair_2026-09-07_cyl/gate.json (the wiring
    # round's fleet re-cut; first released at ../..._2026-09-06_cyl3/); its scan
    # prerequisite: ../metal_cylindrical_complex_scan_2026-09-07_cyl/gate.json (first
    # released at ../..._2026-09-06_cyl4/).
    # INSTALLABLE = False on the fused_hd_pair precedent (see the module's
    # INSTALLABLE_REASON: a tied launch count with the released D->E pair that
    # does not price the host round trip this product removes); `served` here is
    # a PREDICATE verdict, as on the row above.
    "cylindrical_complex_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("cylindrical complex", "cylindrical complex"),
        "module": ("meep_gpu/metal_kernels/cylindrical_complex_fused_hd_pair.py:"
                   "metal_cylindrical_complex_fused_hd_pair_coverage"),
        "halves": ("cylindrical_complex_constitutive@update_H",
                   "cylindrical_complex_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_cylindrical_complex_h_to_d_cell",
             _the_cylindrical_complex_h_to_d_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults, so CARRIES_DEPOSIT_REPAIR "
            "is False and no deposit clause exists to be unevaluable",
            "the product is TWO launches (lead, then the certified curl) and is "
            "excluded from the pointer-sharing calibration: its 'fused' source is "
            "the constitutive plus the scan, not a splice of the two halves",
        ),
    },
    "fused_dispersive_pair": {
        "seam": "D_to_E",
        "pair": ("PML", "dispersive PML E"),
        "module": "meep_gpu/metal_kernels/fused_dispersive_pair.py:471-522",
        "halves": ("pml_curl@step_D", "dispersive_e@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause("D", "meep_gpu.metal_kernels.fused_dispersive_pair")),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored "
            "(fused_dispersive_pair.py:511-521)",
        ),
    },
    # ---- THE 2026-08-30 TRANCHE: seven fused pairs, one round -----------------
    # Device gate: ../metal_tranche7_fused_pairs_2026-08-30/gate.json (VERDICT PASS).
    # Legs, all on this Mac's MPS: 24 identity cases at 0 differing uint32 words per
    # COMPLETE driver step over 12 steps with the declared launch counts; 20 armed
    # source mutations, every one caught; 5 DECLARED equivalences, every one
    # bit-identical; 7 deposit cases carried bit-identically with their unbracketed
    # null controls all diverging; 5 refuted signatures all failing to compile.
    #
    # EVERY LADDER BELOW IS THE SHIPPED PREDICATE'S, transcribed clause for clause:
    # the two halves are read as census columns and the seam clauses are the ones the
    # family's own coverage function adds. Nothing is widened -- the clauses a census
    # block cannot answer are listed in `not_evaluable`, which is what makes every
    # count here an UPPER BOUND on that product rather than a claim.
    # ---- THE FOURTH SEAM ON THE COMPLEX CELL, 2026-09-07 ---------------------
    # ONE dispatch spanning complex `update_H` and complex `step_D`: the
    # certified complex constitutive body into write-only scratch for the
    # thread's own cell and RECOMPUTED for each of the curl's three backward
    # taps, then the H/f_w_H rotation. Device gate:
    # ../metal_complex_fused_hd_pair_2026-09-07_wired/gate.json (17 of 17 corpus
    # rows of the cell driven and bit-identical). INSTALLABLE = False on a
    # MEASURED arbitration -- both neighbouring released pairs install on all 17
    # rows, so this product is a LOSS on 17 and a GAIN on 0 -- so `served` here
    # is a PREDICATE verdict, as on the H->D rows above.
    "complex_fused_hd_pair": {
        "seam": "H_to_D",
        "pair": ("complex/Bloch", "complex/Bloch"),
        "module": ("meep_gpu/metal_kernels/complex_fused_hd_pair.py:"
                   "metal_complex_fused_hd_pair_coverage"),
        "halves": ("complex_constitutive@update_H", "complex_pml_curl@step_D"),
        "seam_clauses": (),
        "row_clauses": (
            ("the_complex_h_to_d_cell", _the_complex_h_to_d_cell),
            ("no_standing_electric_withdraw", _no_standing_electric_withdraw),
        ),
        "not_evaluable": (
            "nothing is INJECTED between the two consults, so CARRIES_DEPOSIT_REPAIR "
            "is False and no deposit clause exists to be unevaluable",
        ),
    },
    "complex_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("complex/Bloch", "complex/Bloch"),
        "module": ("meep_gpu/metal_kernels/complex_fused_electric_pair.py:"
                   "metal_complex_fused_electric_pair_coverage"),
        "halves": ("complex_pml_curl@step_D", "complex_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.complex_fused_electric_pair")),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline",
        ),
    },
    "folded_fused_dispersive_pair": {
        "seam": "D_to_E",
        "pair": ("folded", "folded dispersive PML E"),
        "module": ("meep_gpu/metal_kernels/folded_fused_dispersive_pair.py:"
                   "metal_folded_fused_dispersive_pair_coverage"),
        "halves": ("folded_pml_curl@step_D", "folded_dispersive_e@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.folded_fused_dispersive_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("has_a_polarization", _has_a_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) reads +1/-1 on every folded axis, and the far "
            "carry's own clauses (_far_carry_reasons) -- both carried by the folded "
            "curl half this ladder already reads as a column, and both measured over "
            "every folded PERIODIC extent Grid will build "
            "(../fusion_matrix_metal_2026-08-21_farcarry/d_side_far_carry_law.json)",
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "no E component is driven by more than MAX_POLES susceptibilities -- "
            "asked as a ROW clause below rather than a configuration one, because "
            "the count lives in the census's `polarization` block",
        ),
        "row_clauses": (("pole_budget_fits", _pole_budget_fits),),
    },
    "folded_complex_fused_pair": {
        "seam": "D_to_E",
        "pair": ("folded complex", "folded complex"),
        "module": ("meep_gpu/metal_kernels/folded_complex_fused_pair.py:"
                   "metal_folded_complex_fused_pair_coverage"),
        "halves": ("folded_complex_pml_curl@step_D",
                   "folded_complex_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.folded_complex_fused_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) and the far carry's own clauses -- both carried "
            "by the folded complex curl half this ladder reads as a column",
            "grid exposes has_metallic / is_metallic / is_mirrored",
        ),
    },
    "folded_beta_complex_fused_pair": {
        "seam": "D_to_E",
        "pair": ("folded beta complex", "folded beta complex"),
        "module": ("meep_gpu/metal_kernels/folded_beta_complex_fused_pair.py:"
                   "metal_folded_beta_complex_fused_pair_coverage"),
        "halves": ("folded_beta_bloch_pml_curl@step_D",
                   "folded_beta_complex_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.folded_beta_complex_fused_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) and the far carry's own clauses -- both carried "
            "by the folded beta curl half this ladder reads as a column",
            "grid exposes has_metallic / is_metallic / is_mirrored",
            "the two expansion artifacts (_expansion_reasons): this product needs "
            "BOTH folded_beta's and folded_complex's, and the census block carries "
            "neither probe",
        ),
    },
    "beta_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("special_kz real beta", "special_kz real beta"),
        "module": ("meep_gpu/metal_kernels/beta_fused_electric_pair.py:"
                   "metal_beta_fused_electric_pair_coverage"),
        "halves": ("beta_pml_curl@step_D", "beta_run_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.beta_fused_electric_pair")),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored",
        ),
    },
    "folded_beta_real_fused_pair": {
        "seam": "D_to_E",
        "pair": ("folded beta real", "folded beta real"),
        "module": ("meep_gpu/metal_kernels/folded_beta_real_fused_pair.py:"
                   "metal_folded_beta_real_fused_pair_coverage"),
        "halves": ("folded_beta_pml_curl@step_D", "folded_beta_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.folded_beta_real_fused_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) and the far carry's own clauses -- both carried "
            "by the folded beta curl half this ladder reads as a column",
            "grid exposes has_metallic / is_metallic / is_mirrored",
        ),
    },
    "folded_beta_real_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("folded beta real", "folded beta real"),
        "module": ("meep_gpu/metal_kernels/folded_beta_real_fused_magnetic_pair.py:"
                   "metal_folded_beta_real_fused_magnetic_pair_coverage"),
        "halves": ("folded_beta_pml_curl@step_B", "folded_beta_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause(
                 "B",
                 "meep_gpu.metal_kernels.folded_beta_real_fused_magnetic_pair")),
            ("fold_stores_the_source_row", _fold_stores_the_source_row),
            ("fold_and_wall_are_disjoint", _fold_and_wall_are_disjoint),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid.mirror_phase(axis) and the far carry's own clauses -- both carried "
            "by the folded beta curl half this ladder reads as a column",
            "grid exposes has_metallic / is_metallic / is_mirrored",
        ),
    },
    # --- THE E->P CHAIN, three arms under three family names ----------------
    # meep_gpu/metal_kernels/fused_ade_chain.py, released 2026-08-20 at
    # ../metal_fused_ade_chain_2026-08-20T2150/. THREE families rather than one
    # because the arm table holds at most one row per (family, slot) and all three
    # register on `update_E`; the split is forced, not stylistic (fused_ade_chain.py,
    # the FAMILY / DISPERSIVE_FAMILY / FOLDED_FAMILY block).
    #
    # THERE IS NO SOURCE CLAUSE ON ANY OF THEM, and the absence is the seam's own
    # fact rather than a scope choice: `driver.step` runs update_E at :3304 and
    # update_P at :3306 with NOTHING between — no injection, no fill, no wall
    # clear — which is why SOURCE_SEAM_CLAUSE below carries no E_to_P entry and
    # this seam's ceiling is its full instance count.
    "fused_ade_chain": {
        "seam": "E_to_P",
        "pair": ("no-PML stored E", "ADE update_P"),
        "module": "meep_gpu/metal_kernels/fused_ade_chain.py (arm 'no_pml')",
        "halves": ("stored_e@update_E", "ade_update_p@update_P"),
        "seam_clauses": (
            ("no_offdiagonal_epsilon", _no_offdiagonal_epsilon),
            ("no_chi2_chi3", _no_chi2_chi3),
            ("fits_the_binding_ceiling", _fits_the_fused_binding_ceiling("no_pml")),
        ),
        "not_evaluable": _ADE_CHAIN_NOT_EVALUABLE,
    },
    "fused_ade_chain_dispersive": {
        "seam": "E_to_P",
        "pair": ("dispersive PML E", "ADE update_P"),
        "module": "meep_gpu/metal_kernels/fused_ade_chain.py (arm 'dispersive')",
        "halves": ("dispersive_e@update_E", "ade_update_p@update_P"),
        "seam_clauses": (
            ("no_offdiagonal_epsilon", _no_offdiagonal_epsilon),
            ("no_chi2_chi3", _no_chi2_chi3),
            ("fits_the_binding_ceiling",
             _fits_the_fused_binding_ceiling("dispersive")),
        ),
        "not_evaluable": _ADE_CHAIN_NOT_EVALUABLE,
    },
    "complex_conductive_fused_pair": {
        "seam": "D_to_E",
        "pair": ("complex conductive no-PML curl", "complex no-PML stored E"),
        "module": "meep_gpu/metal_kernels/complex_conductive_fused_pair.py",
        "halves": ("complex_conductive_plain_curl@step_D",
                   "complex_stored_e@update_E"),
        "seam_clauses": (
            # THE POLARITY IS THE WHOLE CELL. The four corpus rows carry exactly
            # one source and it is MAGNETIC, so the electric injection — and with
            # it the condinv-scaled path at driver.py:3296 — cannot run. That is
            # MEASURED, not read off the guard: gate_metal_complex_conductive_
            # fused_pair leg `seam_is_empty` instruments the real FdtdDriver on
            # this configuration, with a control leg where it does fire.
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause("D", "meep_gpu.metal_kernels.complex_conductive_fused_pair")),
            ("no_folded_axis", _no_folded_axis),
            ("no_metallic_axis", _no_metallic_axis),
            ("no_offdiagonal_epsilon", _no_offdiagonal_epsilon),
            ("no_chi2_chi3", _no_chi2_chi3),
            ("fits_the_binding_ceiling", _fits_the_conductive_pair_ceiling),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "the per-target CONDUCTIVITY split (`conductive_no_pml_targets`) — the "
            "census records that a conductivity exists, not which of Dx/Dy/Dz "
            "carries one; the emitter specialises the tail per target either way "
            "and both arms are certified, so the split changes which text is "
            "emitted, never whether the row is admitted",
        ),
    },
    # ---- THE TWO PRODUCTS OF THE 2026-09-01 CUT: the last "FITS ... NOT BUILT"
    # cells of the 2026-08-31_plainrepair board, and the first Metal products on
    # `deposit_repair.PLAIN_PATH` — the second repair, which inverts `update_E`'s
    # plain overwrite (stepping.py:1019-1022) rather than the split-field recurrence.
    # One construction over the two certified no-absorber curl arms, split into two
    # modules the way the arms are split: `no_pml_conductive._any_curl_conductivity`
    # is the partition, so exactly one family answers for any run.
    #
    # THE REAL-STORAGE LOSSLESS D->E WELD. Its one corpus row
    # (examples:material-dispersion.py — all-periodic, TWO Lorentz poles per
    # component, ONE non-integrated in-seam electric source, no conductivity) is
    # served: with no conductivity the driver takes the plain per-source injection
    # loop (driver.py:3307) and the PLAIN repair carries the deposit across the
    # launch. The conductive-rescale clause is consulted anyway — imported from the
    # conductive sibling, one home — and is vacuously true on every row this
    # family's curl half admits, because that half refuses every D-target sigma
    # by name.
    "no_pml_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("no-PML curl", "no-PML stored E"),
        "module": "meep_gpu/metal_kernels/no_pml_fused_electric_pair.py",
        "halves": ("plain_curl@step_D", "stored_e@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.no_pml_fused_electric_pair")),
            ("no_scaled_electric_injection_on_a_conductive_run",
             _no_scaled_electric_injection_on_a_conductive_run),
            # The two fills are INERT on every admitted grid: the predicate
            # refuses a folded axis by name (fill_symmetry_bc_D and
            # fill_folded_far_ghosts_D both run inside this seam,
            # driver.py:3300-3302, and neither is carried), which is why REPLACES
            # names three driver passes and not five.
            ("no_folded_axis", _no_folded_axis),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (no_pml_fused_electric_pair.py, the "
            "wall-readability clause)",
        ),
    },
    # THE CONDUCTIVE SIBLING, and it now serves BOTH OF ITS CELL'S 2 ROWS — the
    # refusal that priced it at 0 of 2 was LIFTED ON A MEASUREMENT. Both rows
    # (examples:absorber-1d.py, tests:TestAbsorber.test_absorber — an mp.Absorber,
    # meep.materials.Al at five poles per component, metallic on z) declare a
    # NON-integrated in-seam electric source, which the clause refused while the
    # driver rescaled the WHOLE target volume (the -0.0 canonicalisation, measured
    # as 1231 divergent Ez words at step 1 of a signed-zero seed). The driver now
    # replays the condinv rescale SPARSELY at the deposit cells the sources
    # publish, and the same signed-zero walk re-run against it reads
    # byte-identical at every step while the retired whole-volume passes replayed
    # as an armed control still diverge (the gate's lifted_refusal leg). The
    # `no_scaled_electric_injection_on_a_conductive_run` clause stays in the
    # ladder recording what the shipped predicate still refuses — a scaled source
    # publishing NO deposit table, the driver's dense fallback, which no in-tree
    # source class is — and is census-vacuous for that reason.
    "no_pml_conductive_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("conductive no-PML curl", "no-PML stored E"),
        "module": "meep_gpu/metal_kernels/no_pml_conductive_fused_electric_pair.py",
        "halves": ("conductive_plain_curl@step_D", "stored_e@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D",
                 "meep_gpu.metal_kernels.no_pml_conductive_fused_electric_pair")),
            ("no_scaled_electric_injection_on_a_conductive_run",
             _no_scaled_electric_injection_on_a_conductive_run),
            ("no_folded_axis", _no_folded_axis),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "whether a scaled in-seam electric source PUBLISHES its deposit table "
            "— the census records `source_field_types` and stops, so the "
            "`no_scaled_electric_injection_on_a_conductive_run` clause counts "
            "every conductive row served where the shipped predicate would still "
            "refuse a table-less source (the driver's whole-volume fallback); "
            "none exists on this corpus — every in-tree source class publishes "
            "`_point_ix/_point_iy/_point_iz` at setup, pinned by the host suite",
            "the per-target CONDUCTIVITY split (`conductive_no_pml_targets`) — "
            "the census records that a conductivity exists, not which of Dx/Dy/Dz "
            "carries one; the emitter specialises the tail per target either way "
            "and both arms are certified",
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (no_pml_conductive_fused_electric_pair.py, the "
            "wall-readability clause)",
        ),
    },
    # ---- THE FOUR PRODUCTS OF THE RESIDUE ROUND ------------------------------
    # The last two CANNOT-BIND cells of the 2026-09-01 board, closed by the
    # SHIPPED coefficient_pack technique (the same move that flipped the two Dcyl
    # D->E cells), and the two seams of the one slot-UNSELECTED row, closed by
    # welding the special_kz complex-beta arms whose constitutive companion landed
    # 2026-08-19 (after the tranche-6 census; the reclose record substituted in
    # `rows()` carries the re-measured selections).
    #
    # THE BFAST D->E WELD. Its one corpus row (tests:TestReflectanceAngular.
    # test_reflectance_angular_2_35_7 — electric source, all-periodic) was scored
    # `UNFUSABLE ON METAL ... 33 pointers, over the 30 the platform allows`. The
    # curl half's six read-only PML coefficient vectors ride ONE buffer with six
    # element offsets in Params, so the shipped shape binds 28 pointers; the gate
    # COMPILES the refused 34-binding unpacked signature and the shipped 29 and
    # bisects the ceiling on this host. The cross-backend fact the residue audit
    # recorded stands here: CUDA's same cell serves 0, so this weld is the FIRST
    # service of the instance on any backend.
    "bfast_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("BFAST", "BFAST"),
        "module": "meep_gpu/metal_kernels/bfast_fused_electric_pair.py",
        "halves": ("bfast_pml_curl@step_D", "bfast_run_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.bfast_fused_electric_pair")),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (the wall-readability clause)",
            "the three f_bfast_D* state volumes are allocated — the curl half's "
            "own predicate checks them on the lifted object and the census reads "
            "that column; restated in the weld's predicate because the weld binds "
            "them directly",
        ),
    },
    # THE CONDUCTIVE-PML D->E WELD. Its one corpus row (tests:TestAdjointSolver.
    # test_damping — electric + magnetic sources, conductivity on D, active PML)
    # was scored `UNFUSABLE ON METAL ... 39 pointers, over by NINE`. ONE pack
    # holds BOTH halves' twelve read-only per-axis vectors (the curl's kms/sinv
    # and the constitutive's kps/kms), so the shipped shape binds 28 pointers —
    # the widest pack in the tree, measured by the gate against the refused
    # 40-binding unpacked form. The conductive-rescale clause is consulted through
    # the same one-home function the no-PML sibling ships, lifted as above.
    "conductive_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("conductive PML curl", "ordinary"),
        "module": "meep_gpu/metal_kernels/conductive_fused_electric_pair.py",
        "halves": ("conductive_pml_curl@step_D", "constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.conductive_fused_electric_pair")),
            ("no_scaled_electric_injection_on_a_conductive_run",
             _no_scaled_electric_injection_on_a_conductive_run),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "whether a scaled in-seam electric source PUBLISHES its deposit table "
            "— the same census gap the no-PML conductive row names, vacuous on "
            "this corpus for the same measured reason",
            "the per-target CONDUCTIVITY split (`conductive_targets`) — the "
            "census records that a conductivity exists, not which of Dx/Dy/Dz "
            "carries one; the emitter specialises the tail per target either way "
            "and both arms are certified",
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (the wall-readability clause)",
        ),
    },
    # THE TWO SPECIAL_KZ COMPLEX-BETA WELDS — the one formerly slot-UNSELECTED
    # row (tests:TestSpecialKz.test_special_kz), both seams. The curl arm won all
    # along; the complex-beta constitutive arm landed 2026-08-19 (the certified
    # complex bloch body under a restated predicate — no new arithmetic), and the
    # reclose artifact records plan_step selecting `special_kz complex beta` at
    # ALL FOUR slots. CUDA serves this row's B->H through its own special_kz
    # magnetic weld, which is the cross-backend proof of the weld shape.
    "beta_complex_fused_electric_pair": {
        "seam": "D_to_E",
        "pair": ("special_kz complex beta", "special_kz complex beta"),
        "module": "meep_gpu/metal_kernels/beta_complex_fused_electric_pair.py",
        "halves": ("beta_bloch_pml_curl@step_D",
                   "beta_run_complex_constitutive@update_E"),
        "seam_clauses": (
            ("in_seam_electric_deposit_clears",
             _seam_deposit_clause(
                 "D", "meep_gpu.metal_kernels.beta_complex_fused_electric_pair")),
            ("no_folded_axis", _no_folded_axis),
            ("no_polarization", _no_polarization),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "the beta expansion artifact (_beta_complex_grid_reasons): the census "
            "block carries no probe, and both halves already read it on the "
            "lifted object — the census columns carry their verdicts",
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_D "
            "can be carried inline (the wall-readability clause)",
        ),
    },
    "beta_complex_fused_magnetic_pair": {
        "seam": "B_to_H",
        "pair": ("special_kz complex beta", "special_kz complex beta"),
        "module": "meep_gpu/metal_kernels/beta_complex_fused_magnetic_pair.py",
        "halves": ("beta_bloch_pml_curl@step_B",
                   "beta_run_complex_constitutive@update_H"),
        "seam_clauses": (
            ("in_seam_magnetic_deposit_clears",
             _seam_deposit_clause(
                 "B", "meep_gpu.metal_kernels.beta_complex_fused_magnetic_pair")),
            ("no_folded_axis", _no_folded_axis),
        ),
        "not_evaluable": _DEPOSIT_NOT_EVALUABLE + (
            "the beta expansion artifact — as on the electric twin",
            "grid exposes has_metallic / is_metallic / is_mirrored so zero_metal_B "
            "can be carried inline (the wall-readability clause)",
        ),
    },
    "complex_fused_ade_chain": {
        "seam": "E_to_P",
        "pair": ("complex no-PML stored E", "ADE update_P"),
        "module": "meep_gpu/metal_kernels/complex_fused_ade_chain.py",
        "halves": ("complex_stored_e@update_E", "ade_update_p@update_P"),
        "seam_clauses": (
            ("no_offdiagonal_epsilon", _no_offdiagonal_epsilon),
            ("no_chi2_chi3", _no_chi2_chi3),
            ("fits_the_binding_ceiling", _fits_the_complex_chain_ceiling),
        ),
        "not_evaluable": _ADE_CHAIN_NOT_EVALUABLE + (
            "the expansion ARM the probe licenses equals the arm the certified ADE "
            "half bakes in (ade_update_p.py:105) — a PLATFORM fact, not a row one; "
            "the predicate refuses by name when they disagree and "
            "gate_metal_complex_fused_ade_chain leg `arm_binding` measures both "
            "the agreement and the refusal",
        ),
    },
    "folded_fused_ade_chain": {
        "seam": "E_to_P",
        "pair": ("folded dispersive PML E", "ADE update_P"),
        "module": "meep_gpu/metal_kernels/fused_ade_chain.py (arm 'dispersive', folded)",
        "halves": ("folded_dispersive_e@update_E", "ade_update_p@update_P"),
        "seam_clauses": (
            ("no_offdiagonal_epsilon", _no_offdiagonal_epsilon),
            ("no_chi2_chi3", _no_chi2_chi3),
            ("fits_the_binding_ceiling",
             _fits_the_fused_binding_ceiling("dispersive")),
        ),
        "not_evaluable": _ADE_CHAIN_NOT_EVALUABLE,
    },
}

#: Each product's arm-table state, read live rather than transcribed.
#: The driver passes that sit INSIDE each seam, in ``coverage.RESIDENCY_ORDER``'s
#: spelling. ``E_to_P`` has none (driver.py:3303-3306), which is why it is absent
#: rather than empty.
IN_SEAM_PASSES: Dict[str, Tuple[str, str, str]] = {
    "B_to_H": ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B"),
    "D_to_E": ("fill_D", "zero_metal_D", "fill_folded_far_ghosts_D"),
}


def live_in_seam_passes(row: dict, seam: str) -> Tuple[str, ...]:
    """Which of a seam's in-seam passes RUN on this row — READ, NOT DERIVED.

    THE 2026-08-20 CARRY CUTS DERIVED THIS FROM THE BOUNDARY TRIPLE AND THE NUMBER
    THEY PUBLISHED FOR THE NEAR FILLS IS WRONG. That rule said the near fill is live
    iff some axis is mirrored. The census records the ENGINE'S OWN answer —
    ``plan_step.live``, captured by ``launch.plan_step`` with the run's real source
    list — and measured against it on these same 186 rows the derived rule disagrees
    on ``fill_B`` on 32 rows and on ``fill_D`` on 85, because a near fill is also
    live on an UNFOLDED row that carries a source. It agrees exactly (0 rows) on the
    two wall clears and on both far-ghost passes.

    So this reads the recorded list. Nothing is derived here, and the disagreement
    is reported below rather than quietly repaired.
    """
    names = IN_SEAM_PASSES.get(seam)
    if names is None:
        return ()
    live = set((row.get("plan_step") or {}).get("live") or ())
    return tuple(name for name in names if name in live)


def derived_in_seam_passes(configuration: dict, seam: str) -> Tuple[str, ...]:
    """The 2026-08-20 carry cuts' DERIVED rule, kept only to measure it against the
    engine's recorded answer. Never used to score a cell."""
    names = IN_SEAM_PASSES.get(seam)
    if names is None:
        return ()
    near, wall, far = names
    mirrored = [bool(v) for v in configuration["mirrored"]]
    metallic = [bool(v) for v in configuration["metallic"]]
    live: List[str] = []
    if any(mirrored):
        live.append(near)
    if any(metallic[a] and not mirrored[a] for a in range(3)):
        live.append(wall)
    if any(mirrored[a] and not metallic[a] for a in range(3)):
        live.append(far)
    return tuple(live)


def _families_declaring_a_weld_is_owed() -> Dict[str, Dict[str, Any]]:
    """``{family: {"module": …, "installable": …, "reason": …}}`` for the withdrawn.

    READ OUT OF THE SOURCE, keyed by the module's own ``FAMILY`` constant rather than
    by its filename -- the same trap ``_wiring``'s docstring records, where the Dcyl
    complex pair's module name and family name differ and a filename key silently
    disabled the check it fed.
    """
    import ast  # noqa: PLC0415

    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted((API_ROOT / "meep_gpu" / "metal_kernels").glob("*.py")):
        declared: Dict[str, Any] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            name = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)):
                name = node.targets[0].id
            if name in ("FAMILY", "WELD_OWED", "INSTALLABLE") and node.value is not None:
                with contextlib.suppress(ValueError):
                    declared[name] = ast.literal_eval(node.value)
        if declared.get("WELD_OWED") and declared.get("FAMILY"):
            out[declared["FAMILY"]] = {"module": path.name,
                                       "installable": declared.get("INSTALLABLE"),
                                       "reason": declared["WELD_OWED"]}
    return out


def _wiring() -> Dict[str, bool]:
    """Every product's arm-table state, read live rather than transcribed.

    THE KEYS OF :data:`PRODUCTS` ARE ARM FAMILY NAMES, not module names, and the two
    differ for the Dcyl complex pair: the module is
    ``cylindrical_fused_magnetic_pair.py`` and the family it registers is
    ``cylindrical_complex_fused_magnetic_pair`` (that module's ``FAMILY``).
    MEASURED ON THE 08-20c CUT: keyed by the module name, ``wired.get(family)``
    returned ``None`` and the product's wiring state was reported as unknown while
    the run still printed a headline — a name that matches nothing disabling the
    check it feeds, silently. The assertion below (carried from that cut) is what
    turns that from a thing a reader must notice into a thing the script refuses to
    run past.

    THE SECOND FLOOR IS NEW IN THE CLOSED CUT, and it is the one this round's whole
    job depends on. A matrix that does not ASK about a landed product cannot report
    it, and the failure is silent in the other direction too: a product registered
    in the live arm table but ABSENT from ``PRODUCTS`` is invisible here, its cell
    reads "no fused product covers this pair", and its seam-instances are counted as
    a gap. That is exactly how five landed families went uncounted across the
    2026-08-20 forks. So the completeness check runs in BOTH directions and raises.
    """
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415
    families = {spec.family for spec in arms.registered()}
    unknown = sorted(set(PRODUCTS) - families)
    if unknown:
        raise SystemExit(
            f"PRODUCTS names {unknown} which the live arm table does not register. "
            f"A PRODUCTS key must be an arm FAMILY name (the module's FAMILY "
            f"constant), because _wiring() looks the wiring flag up by it; a key "
            f"that matches nothing would report wired=None and measure nothing. "
            f"Registered families on step_B: "
            f"{sorted(s.family for s in arms.registered() if s.slot == 'step_B')}")
    # A fused family is one whose name carries `fused` or `chain`; every one of the
    # 14 registered on this tree does, and no sub-step arm does. Derived from the
    # live table rather than listed, so a family added tomorrow is asked about
    # tomorrow instead of being silently dropped.
    fused_families = sorted(f for f in families
                            if "fused" in f or "chain" in f)
    # AND A FAMILY THAT DECLARES ITS WELD IS OWED IS NOT A LANDED PRODUCT. The floor
    # above exists because an absent product's cells read as GAPS and its served
    # instances vanish; a family whose gate has RUN AND REFUSED serves nothing to
    # lose. `WELD_OWED` is that declaration (metal_kernels/fused_hd_pair.py, the
    # Metal spelling of the CUDA track's UNCERTIFIED_KERNELS), and it is not the
    # NOT_PRICED_HERE hatch under another name: that hatch was for a LANDED product
    # this cut had not yet priced, which made the headline a lower bound, and it is
    # deliberately closed. This exclusion costs the headline nothing, and the two
    # premises that make that true are ASSERTED rather than assumed -- the family
    # must be uninstallable, and it must own no weld in fingerprints.json. A family
    # that is unwelded but installable, or welded but unpriced, still raises.
    owed = _families_declaring_a_weld_is_owed()
    for family, module in sorted(owed.items()):
        if family in PRODUCTS:
            raise SystemExit(
                f"{family} declares WELD_OWED and is priced in PRODUCTS. A product "
                f"whose gate refused may not be credited by this board")
        if module.get("installable") is not False:
            raise SystemExit(
                f"{family} declares WELD_OWED but does not declare INSTALLABLE = "
                f"False. An unwelded product the composer may install serves rows "
                f"this board would then be failing to count; price it or refuse it")
        welded = json.loads(
            (API_ROOT / "meep_gpu" / "metal_kernels" / "fingerprints.json").read_text())
        if f"metal_{family}_device_gate" in welded:
            raise SystemExit(
                f"{family} declares WELD_OWED and fingerprints.json carries "
                f"metal_{family}_device_gate. Delete whichever is false")
    unasked = sorted(set(fused_families) - set(PRODUCTS) - set(NOT_PRICED_HERE)
                     - set(owed))
    if unasked:
        raise SystemExit(
            f"the live arm table registers fused families {unasked} which this "
            f"matrix does not ask about. A landed product missing from PRODUCTS is "
            f"invisible: its cell reads 'no fused product covers this pair' and its "
            f"seam-instances are reported as a GAP. Wire it into PRODUCTS or this "
            f"census under-reports the board.")
    absent = sorted(set(NOT_PRICED_HERE) - set(fused_families))
    if absent:
        raise SystemExit(
            f"NOT_PRICED_HERE names {absent}, which the live arm table does not "
            f"register. An exception list that outlives its exception silences the "
            f"floor it was carved out of; delete the entry.")
    if NOT_PRICED_HERE:
        raise SystemExit(
            f"NOT_PRICED_HERE is non-empty: {sorted(NOT_PRICED_HERE)}. This cut "
            f"merges every fork and reports a MEASUREMENT, not a lower bound. "
            f"Price the family or withdraw the headline.")
    # BOTH COUNTS, PRINTED, BECAUSE ONE COUNT CANNOT SHOW A MISMATCH. The forward
    # direction ("every registered fused family is priced") is `unasked` above; the
    # reverse ("every priced product is a registered family") is `unknown` above.
    # Neither raises here, so the only way a reader learns the two sets are the SAME
    # SIZE is to be told both sizes. A verifier found a Metal cut pricing 7 while the
    # table registered 14 — a one-directional check that passes on a subset.
    # The WELD_OWED families are subtracted on the table's side of this equality for
    # the reason the clause above states and asserts: they are registered, refused by
    # their own gate, uninstallable and unwelded, so they are not a coverage surplus
    # this cut cannot see. The count of them is printed below rather than folded away.
    fused_families = sorted(set(fused_families) - set(owed))
    if len(fused_families) != len(PRODUCTS):
        raise SystemExit(
            f"the fused product set is asymmetric: the live arm table registers "
            f"{len(fused_families)} fused families {fused_families} while PRODUCTS "
            f"prices {len(PRODUCTS)} {sorted(PRODUCTS)}. Neither set may exceed the "
            f"other — a surplus in the table is coverage this cut cannot see, and a "
            f"surplus in PRODUCTS is a credit against a family nothing registers.")
    print(f"  COMPLETENESS, BOTH DIRECTIONS:", flush=True)
    print(f"    fused families the live arm table registers : "
          f"{len(fused_families)}", flush=True)
    print(f"    fused products PRODUCTS prices              : "
          f"{len(PRODUCTS)}", flush=True)
    print(f"    registered but unpriced (forward direction) : {len(unasked)}",
          flush=True)
    print(f"    priced but unregistered (reverse direction) : {len(unknown)}",
          flush=True)
    print(f"    NOT_PRICED_HERE escape hatch                : "
          f"{len(NOT_PRICED_HERE)} (a non-empty hatch now RAISES)", flush=True)
    print(f"    registered, gate REFUSED, weld owed         : "
          f"{len(owed)} {sorted(owed)} (uninstallable and unwelded, both asserted; "
          f"serves nothing, so its absence from PRODUCTS costs the headline nothing)",
          flush=True)
    # The lower-bound banner that stood here is GONE with the hatch it reported on:
    # a non-empty NOT_PRICED_HERE now raises above, so there is no state in which
    # this cut's served total is a lower bound and a banner saying so could fire.
    #
    # THE DEPOSIT-REPAIR WIRING FLOOR, NEW IN THIS CUT AND THE ONE THAT MATTERS MOST.
    # `CARRIES_DEPOSIT_REPAIR` is the product's declaration that its launch is wrapped
    # in `LeadingRepairPlan`/`TrailingRepairPlan`; `deposit_repair.seam_source_reasons`
    # says so in its own docstring — "a product that passes True without those wrappers
    # computes the constitutive half against a pre-injection field and REPORTS SUCCESS".
    # On Metal the wrappers are built by `metal_kernels.launch._install_fused_pair`
    # (launch.py:645-701), which `_install_fused_pairs` (:705-800) reaches ONLY for a
    # family holding a row in `FUSED_PAIR_ARMS` (:640-642) — every other family is
    # refused BY NAME before its predicate is asked. So a family that declares True
    # without such a row would be credited here for seam-instances nothing brackets.
    # That is the precise defect this board would otherwise publish, so it raises.
    from meep_gpu.metal_kernels import launch as _metal_launch  # noqa: PLC0415
    declared = sorted(f for f in PRODUCTS
                      if _carries_deposit_repair(_module_dotted_path(f)))
    unbracketed = [f for f in declared if f not in _metal_launch.FUSED_PAIR_ARMS]
    if unbracketed:
        raise SystemExit(
            f"{unbracketed} declare CARRIES_DEPOSIT_REPAIR but hold no "
            f"FUSED_PAIR_ARMS row, so metal_kernels.launch._install_fused_pairs "
            f"refuses them by name (launch.py:733-738) and NOTHING would wrap their "
            f"launch in LeadingRepairPlan/TrailingRepairPlan. Crediting an in-seam "
            f"deposit to such a product is a fused launch computing its constitutive "
            f"half against a pre-injection field and reporting success.")
    print(f"    declare CARRIES_DEPOSIT_REPAIR              : "
          f"{len(declared)} {declared}", flush=True)
    print(f"    ... and hold a FUSED_PAIR_ARMS row          : "
          f"{len(declared) - len(unbracketed)} "
          f"(table: {sorted(_metal_launch.FUSED_PAIR_ARMS)})", flush=True)

    # ------------------------------------------- THE CELLS A PRODUCT IS PRICED ON
    # A `cells` TABLE AND THE COMPOSER'S OWN ABSORB ROWS MUST MOVE TOGETHER. The
    # composer decides which arm pairs a fused family absorbs -- `FUSED_PAIR_ARMS`
    # for the primary pair, `FUSED_PAIR_EXTRA_ARMS` for every further one -- and this
    # board may price exactly those. A cell here that the composer does not absorb
    # would credit a configuration nothing routes; an absorb row with no cell here
    # would leave measured coverage unpriced and unnamed. Both are refused, so the
    # two tables cannot drift apart silently in either direction.
    registered_slots = collections.defaultdict(set)
    for spec in arms.registered():
        registered_slots[spec.slot].add(spec.label)
    multi: List[str] = []
    for family in sorted(PRODUCTS):
        spec_entry = PRODUCTS[family]
        cells = _cells_of(spec_entry)
        if tuple(cells[0]["cell"]) != tuple(spec_entry["pair"]):
            raise SystemExit(
                f"{family}: the first cell {cells[0]['cell']} is not the entry's own "
                f"pair {spec_entry['pair']}; the primary cell is what every ladder, "
                f"print and cross-check on this board reads as the product's own")
        keys = [tuple(cell["cell"]) for cell in cells]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{family}: declares the same cell twice: {keys}")
        extra = {tuple(pair) for pair in
                 _metal_launch.FUSED_PAIR_EXTRA_ARMS.get(family, ())}
        # BOTH DIRECTIONS: a single-cell entry that the composer absorbs on a second
        # pair falls through to the equality below and is refused there, rather than
        # being skipped as "not multi-cell" -- measured 2026-09-10: dropping the
        # nonlinear cell from the table raised nothing until this guard read `extra`.
        if len(cells) == 1 and not extra:
            continue
        multi.append(family)
        if spec_entry["seam"] != h_to_d_seam.SEAM:
            raise SystemExit(
                f"{family}: only the H->D seam has a per-cell price today; a "
                f"multi-cell product on {spec_entry['seam']} would need that seam's "
                f"walk taught the same union before it could be credited")
        for update_h, step_d in keys:
            for arm, slot in ((update_h, "update_H"), (step_d, "step_D")):
                if arm not in registered_slots[slot]:
                    raise SystemExit(
                        f"{family}: cell arm {arm!r} is registered on no {slot} arm, "
                        f"so no census row can ever select it")
        absorbed = ({tuple(_metal_launch.FUSED_PAIR_ARMS[family])}
                    if family in _metal_launch.FUSED_PAIR_ARMS else set())
        absorbed |= {tuple(pair) for pair in
                     _metal_launch.FUSED_PAIR_EXTRA_ARMS.get(family, ())}
        if set(keys) != absorbed:
            raise SystemExit(
                f"{family}: this board prices the cells {sorted(keys)} and the "
                f"composer absorbs {sorted(absorbed)} "
                f"(launch.FUSED_PAIR_ARMS + FUSED_PAIR_EXTRA_ARMS). A cell priced "
                f"here that nothing absorbs credits a configuration the composer "
                f"never routes; an absorbed pair with no cell here leaves measured "
                f"coverage unpriced.")
        for index, cell in enumerate(cells[1:], start=1):
            if not (cell["binding"] and cell["driven_rows"] and cell["fingerprint"]):
                raise SystemExit(
                    f"{family}: cell {_cell_label(keys[index])} carries no binding, "
                    f"driven-row block and ledger fingerprint. A family's release "
                    f"certifies its PRIMARY cell; a further cell is credited only "
                    f"against the run that drove it, row by row, and the ledger "
                    f"must weld that run.")
            if cell["binding"][0] != "gate":
                raise SystemExit(
                    f"{family}: cell {_cell_label(keys[index])} is bound by "
                    f"{cell['binding'][0]!r}; a per-cell credit needs a GATE artifact "
                    f"-- the run's own lift records are the evidence, and a "
                    f"fingerprint entry carries none.")
    print(f"    price more than one cell                    : "
          f"{len(multi)} {multi}", flush=True)
    for family in multi:
        for cell in _cells_of(PRODUCTS[family]):
            bound = cell["binding"][1] if cell["binding"] else "(the family's weld)"
            print(f"        {family:34s} {_cell_label(tuple(cell['cell'])):48s} "
                  f"{bound}", flush=True)
    state: Dict[str, bool] = {}
    for spec in arms.registered():
        if spec.family in PRODUCTS:
            state[spec.family] = bool(spec.wired)
    return state


# ---------------------------------------------------------------------------
# THE SEAMS
# ---------------------------------------------------------------------------

#: WHAT EACH SEAM'S SOURCE INJECTION FORBIDS *A FUSED LAUNCH ON ANY BACKEND*.
#:
#: THIS IS NO LONGER "no source in the seam", AND THE CHANGE IS THE WHOLE POINT OF
#: THIS CUT. Every earlier board read the deposit as an absolute driver fact — one
#: launch cannot be on both sides of `driver.py:3283-3284` / `:3294-3299` — and so
#: put 215 of the 387 seam-instances beyond ANY product's reach, capping the board at
#: 172. That reading was a property of the PRODUCTS THEN SHIPPING, not of the seam:
#: `meep_gpu/deposit_repair.py` straddles the deposit by saving the affected points
#: before the fused launch and recomputing the constitutive half at exactly those
#: points afterwards, and `probe_deposit_repair.py` byte-compares that shape against
#: the driver's own order (results/deposit_repair_device_2026-08-23/probe.json:
#: 6/6 identical with the real fused kernel on an RTX A6000, 6/6 null controls
#: diverging).
#:
#: So the ceiling clause now asks what `deposit_repair.repairable` asks
#: (deposit_repair.py:87-111) rather than whether a source exists. A row whose seam
#: carries a deposit is UNREACHABLE only where the constitutive half is not the
#: linear pointwise accumulation the repair inverts — an off-diagonal chi1inv or an
#: instantaneous chi2/chi3, both on the D side, both refused BY NAME there. The B
#: seam has no such refusal, so its ceiling is now every row.
#:
#: A seam with no injection between the halves (E->P) carries no entry.
SOURCE_SEAM_CLAUSE: Dict[str, Callable[[dict], bool]] = {
    "B_to_H": lambda c: _no_magnetic_source(c) or _deposit_is_repairable("B", c),
    "D_to_E": lambda c: _no_electric_source(c) or _deposit_is_repairable("D", c),
}


SEAMS = (
    # name,      curl slot,  constitutive slot,  what sits between (driver.py)
    ("B_to_H", "step_B", "update_H",
     "magnetic inject 3283-3284 | fill_symmetry_bc_B 3285 | zero_metal_B 3286 | "
     "fill_folded_far_ghosts_B 3287"),
    ("D_to_E", "step_D", "update_E",
     "electric inject 3294-3299 | fill_symmetry_bc_D 3300 | zero_metal_D 3301 | "
     "fill_folded_far_ghosts_D 3302"),
    ("E_to_P", "update_E", "update_P", "nothing (driver.py:3303-3306)"),
)


#: The fingerprint entry each E->P family's release binds, and the digests it
#: recorded. Checked at run time — see :func:`_assert_the_credited_bytes_are_the
#: _released_ones`.
E_TO_P_RELEASE_FINGERPRINT = "metal_fused_ade_chain_device_gate"

#: WHERE EACH PRODUCT'S RELEASE LIVES. This matrix credits 118 seam-instances to
#: fourteen products on the strength of device gates that ran against particular
#: bytes; if a module has moved since, the credit rests on nothing. Thirteen
#: families' gates recorded their digests in ``metal_kernels/fingerprints.json``.
#: The fourteenth — ``cylindrical_real_fused_magnetic_pair`` — is RELEASED and
#: PASSING but has no fingerprints.json entry (a bookkeeping gap in that round, not
#: a missing gate), so it is bound through its own gate's weld manifest instead.
#: Every product must appear here, and the floor below raises if one does not.
# RE-BOUND AGAIN 2026-08-27, THE FOLDED ROUTING ROUND, AND FOR THE SAME REASON THE
# deposit round re-bound it: the routing edit moved `metal_kernels/launch.py`
# (FUSED_PAIR_ARMS gained the two folded rows, a56395f27ba1 -> b82cf8b8dac9),
# `triton_kernels/launch.py` (the has_fold branch now dispatches,
# d65ec7a8ae9d -> 2550edbc5dd7) and the two folded family modules, and this cut's
# FIRST run REFUSED TO BUILD on exactly that drift -- 30 digest mismatches across
# ten of the seventeen products. Both launch.py files sit in every Metal gate's
# import set, so a table edit in either invalidates every binding at once. The
# credit was re-earned by RE-RUNNING EACH GATE on this Mac's MPS against the moved
# bytes rather than by editing any record. The two folded families are bound to the
# ROUTING ROUND's own gate artifacts, which already ran after those edits. The owed
# fingerprints.json digest edits are reported by the round, not applied here.
#
# RE-BOUND 2026-08-27, THE IN-SEAM DEPOSIT ROUND. `metal_kernels/launch.py` gained
# `_install_fused_pair` and the shared `deposit_repair` import (launch.py:81-97,
# :645-701), so the shipped bytes moved a2c2eff660ef -> a56395f27ba1 and this cut's
# FIRST run REFUSED TO BUILD on exactly that drift, for eleven of the seventeen
# products at once. The credit was re-earned by RE-RUNNING EACH GATE on this Mac's
# MPS against the moved bytes rather than by editing any record: ten gate scripts,
# every verdict PASS, every artifact release.released=True, and every recorded
# source digest matching the worktree (drift 0 on all ten). They are bound through
# their GATE ARTIFACTS rather than through `fingerprints.json` because writing that
# weld record is RESERVED in this round -- the same reason the
# `folded_beta_complex_fused_magnetic_pair` row above gives. The owed
# fingerprints.json digest edits are reported by the round, not applied here.
#
# RE-BOUND 2026-08-28, THE DEPOSIT-CARRY ROUND, by the same rule and for the same
# cause as the two rounds above. `metal_kernels/launch.py`'s FUSED_PAIR_ARMS gained
# two rows (`complex_fused_magnetic_pair` and `fused_dispersive_pair`, b82cf8b8dac9
# -> 457601314f20) and those two family modules flipped CARRIES_DEPOSIT_REPAIR in
# the SAME edit, so the shipped bytes moved and this cut's FIRST run REFUSED TO
# BUILD on exactly that drift -- 21 digest mismatches across twelve of the
# seventeen products, launch.py being in every Metal gate's import set. The credit
# was re-earned by RE-RUNNING EVERY GATE on this Mac's MPS against the moved bytes
# rather than by editing any record: 43 of 43 green, every artifact
# release.released=True, drift 0 on all ten bound manifests. ONE GATE HAD TO MOVE
# FIRST and it is the interesting part rather than bookkeeping:
# gate_metal_complex_fused_magnetic_pair's `refusal` leg PINNED the pre-flip answer
# ("a magnetic source must be REFUSED, naming driver.py:3283-3284") and failed on
# the first pass -- the gate catching the flip, which is what it is for. The leg now
# measures BOTH directions: the source is CARRIED with the shipped flag (4 deposit
# points, plan built), the pre-flip prose comes back verbatim with the flag held
# False, and a source publishing no deposit index is still refused by name. Unlike
# the two rounds above, `fingerprints.json` was NOT reserved this round: all 43
# welds were rebound through `rebind_metal_welds.py --stamp 2026-08-28_deposit`.
#
# RE-BOUND AGAIN 2026-08-28 (stamp `2026-08-28_carry`), and for the same reason one
# more time. `deposit_repair.py` gained `repair_cells` -- the closure of the cells a
# post-injection fill images a deposit into -- and the two FOLDED families flipped
# `CARRIES_DEPOSIT_REPAIR` on the strength of it. Every Metal fused-pair gate records
# `deposit_repair.py` and `metal_kernels/launch.py`, so EVERY credit stopped binding
# the moment those bytes moved and this floor refused to build. Re-earned by re-running
# all 43 gates on this Mac's MPS in one pass -- 43 of 43 exit 0, every artifact
# release.released=True -- then `rebind_metal_welds.py --stamp 2026-08-28_carry
# --write` (43 rebound, 0 skipped; re-cut once more as `2026-08-28_carry2` after the
# last comment edits landed, since a gate certifies the bytes it ran against and not
# the ones that follow it). TWO GATES HAD TO MOVE FIRST, and both were the
# gates catching the flip: `gate_metal_folded_fused_pair`'s `refusal` leg and
# `gate_metal_folded_fused_magnetic_pair`'s `carry` leg each PINNED the pre-flip
# answer. Both now measure BOTH directions and each gained a `deposit` leg that steps
# the kernel with a REAL in-seam deposit placed on a row a fill reads from -- four
# cases per seam (near, far, metallic near, two-fold corner), 12 complete driver steps,
# 24 arrays, ZERO differing uint32 words -- with two controls per case that must
# diverge: no repair at all, and the repair cut back to the deposit index.
# `fingerprints.json`'s `kernel_source_sha256` is byte-identical across the whole
# round: not one character of what the MPS compiler received changed.
#
# RE-EARNED AGAIN 2026-08-28 AS `_arity`, and this round's cause is the round's own
# edit rather than a shared file moving underneath it. `metal_kernels/launch.py`
# gained the absorb row `fused_electric_pair: ("PML", "ordinary")` -- which is what
# lets the composer bracket the new product's launch with the deposit repair -- and
# `metal_kernels/registry.py` gained the family's import line and list entry. Both
# are pinned by 39 of the 44 weld entries and reached by every gate through
# `registry.py`, so every credit here stopped binding the moment the table moved and
# this floor refused to build. Re-earned by RE-RUNNING all 44 gates on this Mac's MPS
# in one pass (44 of 44 exit 0, every artifact `release.released=True`), then
# rebinding `fingerprints.json` to the fresh round. `kernel_source_sha256` is
# byte-identical across the whole round -- only host-file digests moved -- so not one
# character of what the MPS compiler received changed for any family that already
# shipped.
#
# RE-EARNED AGAIN 2026-08-29 AS `_release`, and this round's cause is a file no
# Metal product executes: `meep_gpu/fastpath.py`. Releasing the first three fused
# arms through the driver seam moved it (d3a0a8ed3da5 -> 5fbf9b903aba: the fusion
# opt-in, `RELEASED_FUSED_ARMS`, `FUSED_RELEASE_ENVELOPE` and the narrowed clause
# (8)), and every Metal gate IMPORTS the package, so `fastpath.py` sits in all
# thirteen gate-bound weld manifests and every credit here stopped binding at once
# -- this cut's FIRST run refused to build on exactly that drift, 13 mismatches,
# every one of them the same file. NOTHING METAL CHANGED, and the round says so by
# measurement rather than by assertion: `kernel_source_sha256` is byte-identical
# across it, so not one character of what the MPS compiler received moved. The
# credit was re-earned the only way this file accepts -- by RE-RUNNING each gate on
# this Mac's MPS against the moved bytes (13 of 13 `verdict: PASS`,
# `release.released: true`, and 0 drift over 49-91 recorded sources each) -- and
# then repointing the rows below at the fresh artifacts. No record was edited.
#
# THE STANDING COST, worth naming because it will recur: `fastpath.py` is imported
# by every Metal gate and by nothing Metal executes, so a dispatch-side edit taxes
# the whole Metal board with a re-gate campaign it learns nothing from. Narrowing
# what the manifests pin is the fix, and it is a change to how gates record their
# imports, not to this table.
#
# AND IT RECURRED TWICE THE SAME DAY, which is why the sentence above is worth
# keeping. Two more `fastpath.py` rounds followed the release: `_fusionveto` (the
# fused route was given its opt-out, `fastpath.FUSE_ARMS_VETO`, without which the
# driver-route gate's substitution baseline could not be asked for at all and that
# gate reported `launch_drop_per_step 0.0`), and `_2026-08-29_gatename` below (that
# gate then ran and released, and `DRIVER_ROUTE_FUSED_GATE` was repointed at the
# run that measured the SHIPPED route rather than the opt-in). The second is a
# ONE-STRING edit whose whole executable content is that constant -- proved by
# reverting it and reproducing the as-run `code_digest` exactly -- and it still
# costs a full round, which is the standing cost above, measured.
# Same shape as the rounds before them: 15 of 15 gates re-run on this Mac's MPS,
# `verdict: PASS`, `release.released: true`, zero drift over 43-91 recorded sources
# each, and `kernel_source_sha256` byte-identical across the edit, so not one
# character of what the MPS compiler received moved. The rows were repointed at the
# fresh artifacts; no record was edited.
#
# AND THEN IT WENT UNPAID, which is the round `_2026-08-30_bind` below closes and
# the reason this floor earns its cost. `fastpath.py` moved twice more after
# `_gatename` -- to 87f740ee and then to d11454ba -- and nothing repointed these
# rows, so EVERY ONE of the thirteen credited artifacts recorded 06cdbf37 and this
# function raised on all thirteen at once: the Metal board could not be cut at all.
# It is the same defect the Triton record carried in the same hours (the release
# citing a driver-route run that had executed none of the shipping bytes), one
# board over, and the fix is the same one this comment has now recorded four
# times: re-run the gates, repoint the rows, never re-type a digest. 14 of 14
# credited products bound, zero drift.
#
# AND IT WENT UNPAID AGAIN WITHIN THIRTY HOURS -- `_2026-08-30_launchcarry` below.
# THE FILE THAT MOVED IS NOT THIS BACKEND'S. `meep_gpu/triton_kernels/launch.py`
# went 2550edbc5dd7 -> 453733074a83 at 2026-08-30T01:19, and every one of the
# thirteen `_bind` artifacts records importing it, because a Metal gate process
# imports the shared planner. So this function raised on all thirteen at once and
# the Metal board could not be cut, while `test_metal_weld_contract.py` stayed
# GREEN at zero drift -- correctly, because only 3 of the 44 ledger entries PIN
# that path and all three had been rebound to `metal_regate_2026-08-30_carry/`.
# THE TWO RECORDS WERE MEASURING DIFFERENT SETS: the ledger checks the paths each
# weld CURATES, this floor checks every path the gate PROCESS imported, which is
# the larger set and the reason these rows bind through the artifact. A green
# ledger is therefore not evidence that a board can be cut, and neither is a
# cuttable board evidence that the ledger is bound. Both are needed, which is why
# both are kept.
#
# AND IT WENT UNPAID A THIRD TIME, in the same day -- `_2026-08-30_armsrows`
# below, and this time the file that moved is the SAME `metal_kernels/launch.py`
# and the cause was this round's own work. Landing the nine new fused products
# meant landing nine `FUSED_PAIR_ARMS` rows, and that table lives in `launch.py`,
# which 39 of the 44 ledger entries pin and which EVERY Metal gate process
# imports. Two family modules moved with it -- `folded_complex_fused_magnetic_
# pair.py` and `folded_beta_complex_fused_magnetic_pair.py`, whose
# `CARRIES_DEPOSIT_REPAIR` flipped False -> True once those rows gave them a
# route into `_install_fused_pair`. THREE PATHS, 39 ENTRIES, ONE CAUSE.
#
# THE LESSON THIS ROUND ADDS to the four already recorded above: the drift is not
# an accident that befalls a campaign, it is the PRICE of touching `launch.py`,
# and it is owed in the same change that touches it. A round that adds an arm row
# has re-gated the whole Metal board whether or not it planned to.
#
# Paid the same way, on this Mac's MPS: every gate whose weld pins a moved path,
# plus the tranche7 gate that had never had a releasable artifact at all, re-run
# with the `2026-08-30_armsrows` stamp and the rows below repointed at those runs.
# No digest was typed and no record was edited.
#
# The previous round's note, kept: paid on this Mac's MPS, 2026-08-30T21:52-21:55:
# eleven gate
# scripts covering all thirteen credited artifacts, ~3 minutes end to end, every
# one `verdict: PASS` / `release.released: true`, and 647 recorded sources across
# them with ZERO drift against the worktree. The rows below are repointed at
# those runs. No digest was typed and no record was edited; the `_bind` artifacts
# are left exactly where they are, because they are the evidence for what the
# 2026-08-29T23:45 cut credited and deleting them would destroy that.
# RE-BOUND 2026-08-31, THE COEFFICIENT-PACK ROUND, and for the fifth time by the
# rule the notes above establish: the drift is the PRICE of touching `launch.py`,
# and it is owed in the same change that touches it. This round's two Dcyl D->E
# products each need a `FUSED_PAIR_ARMS` row before the seam loop will ask their
# predicate, so that table went 15 rows -> 17, and `registry.py` gained the two
# imports. Both files sit in every Metal gate's import set, so EVERY credit stopped
# binding at once and this floor refused the board on its first run -- 30 digest
# mismatches across seventeen of the twenty-seven products.
#
# Paid the same way, on this Mac's MPS: `recut_metal_gates.sh <root> 2026-08-31_pack`
# re-ran all three expansion probes and all 46 gates in one pass, ALL GREEN, into the
# `results/metal_<family>_2026-08-31_pack/` layout `rebind_metal_welds.py` reads (the
# script gained that optional stamp argument this round, because a campaign that
# re-cut every gate and then could not rebind the ledger has done half the work).
# The two new welds were MINTED from their own released artifacts by
# `mint_metal_weld.py` -- each pinning `coefficient_pack.py` alongside the family
# module, because on these two products the claim rests on both -- and the other 45
# were rebound with `rebind_metal_welds.py --stamp 2026-08-31_pack --write`: 47
# rebound, 0 skipped, 0 without an artifact. No digest was typed and no record was
# edited. The rows below are repointed at those runs.
# RE-BOUND 2026-08-31 A SECOND TIME, THE PLAIN-REPAIR ROUND, and for the SIXTH time by
# the rule above -- with one difference worth stating: the two files that moved are not
# Metal's. `meep_gpu/deposit_repair.py` gained `PLAIN_PATH`, the second repair, which
# inverts `update_E`'s plain overwrite rather than the split-field recurrence; and
# `meep_gpu/triton_kernels/launch.py`'s `_install_fused_pair` gained the `repair_paths`
# argument that lets a product declare WHICH repair its two slots install. Every Metal
# gate reaches both through its own import set -- `deposit_repair` because the Metal
# composer builds the same two repair plans, and the Triton composer because the shared
# engine modules pull it in -- so EVERY credit stopped binding at once and this floor
# refused the board on its first run: the same 2 digests across all 27 products.
#
# Paid the same way, on this Mac's MPS: `recut_metal_gates.sh <root> 2026-08-31_plainrepair`
# re-ran all three expansion probes and all 48 gates in one pass, ALL GREEN (51 rows,
# zero non-zero exits), and `rebind_metal_welds.py --stamp 2026-08-31_plainrepair --write`
# rebound 47, skipped 0, with 0 without an artifact. No digest was typed and no record
# was edited. The rows below are repointed at those runs; the `_pack` artifacts are left
# exactly where they are, because they are the evidence for what the 2026-08-31
# coefficient-pack cut credited.
# REPOINTED 2026-09-02 (the folded-release round), and the cause is the widest so
# far: `fastpath.py` gained the two-half release predicate and the `fuse_labels`
# offer it hands the composer, `triton_kernels/launch.py` and its package
# `__init__` gained the argument that carries it, and this package's own
# `launch.py`/`__init__.py` docstrings were corrected where they still claimed
# dispatch returns None on every branch. Every Metal gate imports all of them
# through its own import set, so every credit stopped binding at once and this
# floor refused the board on its first run -- 6 digests across all 27 products.
#
# Paid the same way, on this Mac's MPS: `recut_metal_gates.sh <root>
# 2026-09-02_foldedrelease` re-ran all three expansion probes and all 50 gates in
# one pass, ALL GREEN (54 rows, zero non-zero exits), and
# `rebind_metal_welds.py --stamp 2026-09-02_foldedrelease --write` rebound 50,
# skipped 0, with 0 without an artifact; `launch.write_fingerprints()` then re-cut
# the two host digests the docstring edits moved. No digest was typed and no record
# was edited by hand.
# REPOINTED 2026-09-04 (the m = 0 complex cylindrical round), and the cause is the
# narrowest so far in the sources and the widest in the credits: the four Dcyl host
# modules moved -- `cylindrical_complex.py` gained the M_ZERO arm,
# `cylindrical_real.py` its refusal message, and both cylindrical fused pairs their
# m = 0 admission -- plus a `launch.py` docstring. Every Metal gate imports all four
# through its own import set, so all 32 gate-bound credits stopped binding at once
# even though only two products' arithmetic changed, and this floor refused the board
# on its first run: 6 digests (the four Dcyl modules, `launch.py`, and
# `metal_composition_matrix.py`) across the credited products.
#
# Paid the same way, on this Mac's MPS: the fleet re-gate wrote 49 gate families plus
# `metal_whole_step` to `results/metal_*_2026-09-04_arbfix/`, EVERY ONE released on
# both `canonical_verdict.released` and `release.released`, and
# `rebind_metal_welds.py --stamp 2026-09-04_m0complex --write` rebound 50, skipped 0,
# with 0 without an artifact; `launch.write_fingerprints()` then re-cut the five host
# digests the kernel edits moved. Every one of those 50 artifacts records EXACTLY the
# live bytes of the five moved files -- checked, zero disagreements -- so what these
# rows are repointed at is a measurement of the tree that ships. No digest was typed
# and no record was edited by hand.
#
# REPOINTED AGAIN 2026-09-05, at `2026-09-05_hdweld`, and the cause is one line rather
# than a kernel: `metal_kernels/registry.py` gained the import and the FAMILY_MODULES
# row for `fused_hd_pair` -- the H->D weld, the fourth fusion seam's first Metal
# product -- and every Metal gate reaches that module through `arms.ensure_registered()`,
# so every `_arbfix` credit stopped binding at once. NOTHING ABOUT THE COMPOSITION
# MOVED: the product registers `wired=False`, holds no `FUSED_PAIR_ARMS` row and
# declares `INSTALLABLE = False`, so `plan_step` cannot select it and the seam loop
# refuses it by name before any predicate is asked -- which is exactly what the
# re-cut whole-step gate is the evidence for. Paid the same way, on this Mac's MPS:
# `recut_metal_gates.sh <root> 2026-09-05_hdweld`, 54 of 54 legs exit 0, VERDICT: ALL
# GREEN; `rebind_metal_welds.py --stamp 2026-09-05_hdweld --write` rebound 50, skipped
# 0, 0 without an artifact; `launch.write_fingerprints()` then re-cut the host digests
# the new module and the registry row moved. THE H->D PRODUCT IS NOT CREDITED ON THIS
# BOARD and this repoint does not credit it -- it has no released gate, no ledger
# entry and no seam row here; what changed is which artifacts the EXISTING credits
# bind to.
#
# REPOINTED ONCE MORE 2026-09-05, at `2026-09-05_hdwithdraw`, and this cause is PROSE:
# `metal_kernels/fused_hd_pair.py` gained `WELD_OWED` -- the declaration that its gate
# has run to completion and REFUSED, which is what lets the weld contract's
# completeness test be green on honest grounds rather than red -- and corrected the
# tail of `INSTALLABLE_REASON`, which used to say the family's own gate certifies its
# arithmetic. The gate certifies a synthetic fixture and has never certified the
# corpus cell. Every Metal gate reaches that module through `arms.ensure_registered()`,
# so every `_hdweld` credit stopped binding at once: the drift is the price of touching
# a file the fleet imports and it is paid by re-running, never by declaring.
#
# Paid the same way, on this Mac's MPS: `recut_metal_gates.sh <root>
# 2026-09-05_hdwithdraw` re-ran all three expansion probes and all 51 gates in one
# pass -- 54 of 55 exits zero, and the ONE non-zero is `gate_metal_fused_hd_pair`
# itself, which is the withdrawal's own subject and MUST NOT release (verdict FAIL,
# `release.released` false, 49 of its 50 legs pass);
# `rebind_metal_welds.py --stamp 2026-09-05_hdwithdraw --write` rebound 50, skipped 0,
# 0 without an artifact; `launch.write_fingerprints()` then re-cut the one host digest
# the prose moved. THE H->D PRODUCT IS STILL NOT CREDITED, and now the board says why
# rather than refusing to build: `_families_declaring_a_weld_is_owed` excludes it from
# the completeness floor and ASSERTS the two premises that make the exclusion free --
# it is uninstallable and it owns no weld. Re-cut on the standing census
# (`metal_coverage_2026-09-04_m0complex`): 358 / 597 served by predicate, 0 in
# dispatch, ZERO instances moved against `fusion_matrix_metal_2026-09-04_arbfix`.
#
# REPOINTED 2026-09-08, at `2026-09-08_regate`, and the cause is the wiring merge
# rather than any kernel: it moved files every Metal gate reaches through its own
# import set, so 37 of this table's 42 rows stopped binding at once. MEASURED before
# one character was changed, by re-hashing each row's recorded digests against the
# checkout: 37 rows drifted on their `_2026-09-07_wired` artifact (3 to 5 files each,
# out of 45 to 126 recorded), and the five `fingerprint` rows drifted on nothing,
# because the mint had already rebound them.
#
# Paid the same way every repoint above was, on this Mac's MPS, and paid BEFORE this
# table was touched: the fleet re-gate wrote 62 artifact directories at the stamp,
# `rebind_metal_welds.py --stamp 2026-09-08_regate --write` rebound 56 with 0 skipped
# and 0 without an artifact, and each of the 37 artifacts named below reads
# `release.released` true with ZERO of its recorded digests disagreeing with the tree.
# No digest was typed and no record was edited by hand; a row whose gate had not been
# re-run was left pointing at what it had.
#
# TWO ROWS ARE ADDED RATHER THAN REPOINTED, and the board could not be cut without
# them: `bfast_fused_hd_pair` and `conductive_fused_hd_pair` are credited by
# :data:`PRODUCTS` (44 entries) and were named in no row here (42), which
# `_assert_every_credit_is_bound_to_released_bytes` refuses by name. Each names its
# gate's FLUSH leg explicitly, because that is the leg that released: both `keep`
# legs record `verdict: INCOMPLETE` with `release.released` false (the gate could not
# certify every leg on this host, exit 75), and a leg that did not release is not
# evidence. Both flush artifacts carry 126 recorded digests and drift zero.
# REPOINTED 2026-09-11 to the dispatch round's own fleet. Every name below that
# had a `_2026-09-11_dispatch` twin on disk was moved to it, because the Phase 1-3
# dispatch work edited `fastpath.py` and the board re-hashes each product's FULL
# import closure -- so the `_2026-09-08_close` and `_2026-09-10_cells` artifacts
# certify bytes the tree no longer ships and the board refuses to credit them. The
# five `*_device_gate` rows are NOT dated and were already matching; they are left
# alone rather than renamed to a directory that does not exist.
# REPOINTED 2026-09-15 from `_2026-09-13_batch` to `_2026-09-15_offdiag`, 45 references
# over 28 gate directories (42 here, 3 in PRODUCTS rows), for the same reason: the
# off-diagonal fix moved `fastpath.py`. Every one of the 28 has a released twin from
# that fleet (ALL GREEN, 0 non-zero exits), and neither gate that imports this file
# records its digest, so the repoint does not drift the artifacts it names.
# REPOINTED 2026-09-19 from `_2026-09-15_gapclose` to `_2026-09-17_allpaths`, 45
# references over 28 gate directories, for the reason every repoint above gives: the
# all-paths round moved files in each product's import closure, the Metal fleet re-ran
# (`results/metal_campaign_2026-09-17_allpaths/fleet_b`, every gate exit 0) and the
# ledger was re-welded from it, so the board refused the first cell it reached --
# "the ledger's metal_beta_complex_fused_hd_pair_device_gate welds 51c96626f94d and
# this cell is bound to 0d7669afd034". Every one of the 28 twins exists and records
# `release.released` true; the board re-hashes each and refuses any that drifts.
# REPOINTED 2026-09-19 (later the same day) from `_2026-09-17_allpaths` to
# `_2026-09-19_witness`, 45 references over 28 gate directories: the overhead round
# moved `deposit_repair.py`, which every one of these products' import closure
# re-hashes; the fleet re-ran (`results/metal_campaign_2026-09-19_witness/fleet`, ALL
# GREEN, 69 min) and the ledger was re-welded from it (62 rebound, 0 skipped). Every
# one of the 28 twins exists and records its release true.
# REPOINTED 2026-09-21 from `_2026-09-19_witness` to `_2026-09-20_restrict`, 45
# references over 28 gate directories: the deposit repair now repairs only the
# component the source writes, which every one of these products' import closure
# re-hashes; the fleet re-ran (`results/metal_campaign_2026-09-20_restrict/fleet`,
# ALL GREEN) and the ledger was re-welded from it (62 rebound, 0 skipped). Every
# twin exists and records its release true, checked before this repoint.
# REPOINTED 2026-09-24 from `_2026-09-20_restrict` to `_2026-09-24_round3`, 45
# references over 28 gate directories, by `repoint_and_cut_metal_board.sh`: the night
# round's held-step fixes (the conditional per-launch drain, the sparse doors as Metal
# kernels, the PML-auxiliary deposit through the doors and the deposit difference formed
# on the device -- the design notes (metal-residency-fix-plan) section 10.8) moved `device.py`,
# `host_writes.py` and `sources.py`, which every one of these products' import closure
# re-hashes; the fleet re-ran (`results/metal_campaign_2026-09-24_round3/fleet`, ALL
# GREEN, 66 min), the ledger was re-welded from it (62 rebound, 0 skipped),
# `launch.write_fingerprints()` re-cut the host digests (device.py, launch.py,
# no_pml_constitutive.py, and barrier.py added), and the board was cut on the fresh
# census `metal_coverage_2026-09-24_round3`: 327 of 597 in dispatch, 531 served, the
# same as `_2026-09-20_restrict` -- a transport change moves no dispatch decision.
RELEASE_BINDING: Dict[str, Tuple[str, str]] = {
    "fused_magnetic_pair": ("gate", "metal_fused_magnetic_pair_2026-10-04_g13s"),
    "nonlinear_fused_magnetic_pair":
        ("gate", "metal_below_the_cut_fused_pairs_2026-10-04_g13s"),
    "beta_fused_magnetic_pair":
        ("gate", "metal_below_the_cut_fused_pairs_2026-10-04_g13s"),
    "bfast_fused_magnetic_pair":
        ("gate", "metal_below_the_cut_fused_pairs_2026-10-04_g13s"),
    "complex_fused_magnetic_pair":
        ("gate", "metal_complex_fused_magnetic_pair_2026-10-04_g13s"),
    # BOUND TO THE ROUTING ROUND'S OWN GATE, which ran AFTER the routing edits
    # (results/folded_routing_metal_2026-08-27/, verdict PASS, released=True,
    # 122/122 rows, 0 stale of 92 recorded sources). No re-run was needed for this
    # family or for `folded_fused_pair`.
    "folded_fused_magnetic_pair": (
        "gate", "metal_folded_fused_magnetic_pair_2026-10-04_g13s"),
    "folded_complex_fused_magnetic_pair":
        ("fingerprint", "metal_folded_complex_fused_magnetic_pair_device_gate"),
    "cylindrical_complex_fused_magnetic_pair":
        ("fingerprint", "metal_cylindrical_fused_magnetic_pair_device_gate"),
    # RE-GATED 2026-08-21T16:46 and RE-BOUND HERE, for the same reason the
    # `complex_conductive_fused_pair` row below was re-bound in the 02:02 cut: the
    # 08-20b artifact stopped binding when the GATE SCRIPT itself moved
    # (`gate_metal_cylindrical_real_fused_magnetic_pair.py`
    # 77a76aeb424a -> cb1a646cbfb8) after that gate ran. This cut's first run
    # REFUSED TO BUILD on exactly that drift. The credit was re-earned by RE-RUNNING
    # THE GATE on this Mac's MPS rather than by editing either record:
    # `..._2026-08-21_recut2/gate.json`, 42/42 legs PASS, released=True, 44 recorded
    # sources, and every one of the 44 matches the worktree (drift 0). The gate's own
    # `gate_reported_source_sha256.gate` in that artifact IS cb1a646cbfb8 — the value
    # the floor named as unmatched — so the re-run is bound to the moved script.
    "cylindrical_real_fused_magnetic_pair":
        ("gate", "metal_cylindrical_real_fused_magnetic_pair_2026-10-05_092_g13s"),
    # ADDED 2026-08-31 WITH THE PRODUCT, bound through its GATE ARTIFACT rather than
    # through fingerprints.json for the reason the rows above give: the gate artifact
    # records every module the gate PROCESS imported, which is a LARGER set of bytes
    # than a fingerprint entry -- and on this product that matters, because the claim
    # rests on TWO modules (the family and `coefficient_pack.py`) rather than one.
    "cylindrical_real_fused_electric_pair":
        ("gate", "metal_cylindrical_real_fused_electric_pair_2026-10-04_g13s"),
    # ADDED 2026-08-31 WITH THE PRODUCT, bound through its GATE ARTIFACT for the same
    # reason: the claim rests on TWO modules (the family and `coefficient_pack.py`)
    # and on the cylindrical-complex EXPANSION PROBE the gate's leg 0 binds, and a
    # gate artifact records every module the process imported.
    "cylindrical_complex_fused_electric_pair":
        ("gate", "metal_cylindrical_fused_electric_pair_2026-10-04_g13s"),
    # ADDED 2026-08-27 and bound through its GATE ARTIFACT rather than through
    # fingerprints.json, because writing that weld record is reserved in this round.
    # The gate artifact records every module the gate PROCESS imported, so this binds
    # a LARGER set of bytes than a fingerprint entry would.
    # `_armsrows2`, NOT `_armsrows`, AND THE SECOND STAMP IS THE RECORD OF A DEFECT
    # THIS ROUND FOUND. This family is one of the two "clause discharges": its
    # `CARRIES_DEPOSIT_REPAIR` was flipped True with its absorb row, but its GATE was
    # left asserting the pre-flip contract -- that a real in-seam magnetic
    # `VolumeSource` is REFUSED -- so the first campaign run returned verdict FAIL and
    # this floor refused the board. The flip itself is sound and was measured both
    # ways before anything was edited: the real indexed source is admitted and its
    # plan builds, while a source publishing no deposit index is still refused BY NAME
    # by `deposit_repair.seam_source_reasons`. The GATE was brought up to the contract
    # -- both directions now asserted where one was -- and RE-RUN into a fresh
    # directory, which its own manifest required: `metal_gate_runner` refuses to
    # release against a `source_sha256.txt` that disagrees with the gate script it
    # just executed, and editing the script is exactly that disagreement.
    "folded_beta_complex_fused_magnetic_pair": (
        "gate", "metal_folded_beta_complex_fused_magnetic_pair_2026-10-04_g13s"),
    "folded_fused_pair": (
        "gate", "metal_folded_fused_pair_2026-10-04_g13s"),
    "fused_dispersive_pair": (
        "gate", "metal_fused_dispersive_pair_2026-10-04_g13s"),
    # ADDED 2026-08-28 WITH THE PRODUCT, bound through its GATE ARTIFACT rather than
    # through `fingerprints.json`: writing a weld record is reserved this round, and
    # the gate artifact binds a LARGER set of bytes anyway -- every module the gate
    # PROCESS imported, 91 paths.
    "fused_electric_pair": (
        "gate", "metal_fused_electric_pair_2026-10-04_g13s"),
    "fused_ade_chain": ("fingerprint", E_TO_P_RELEASE_FINGERPRINT),
    "fused_ade_chain_dispersive": ("fingerprint", E_TO_P_RELEASE_FINGERPRINT),
    "folded_fused_ade_chain": ("fingerprint", E_TO_P_RELEASE_FINGERPRINT),
    # THE TWO PRODUCTS OF THIS CUT, bound through their GATE ARTIFACT rather than
    # through `fingerprints.json`, and the choice is for STRENGTH rather than
    # necessity: both are welded there too (`metal_complex_fused_ade_chain_device_
    # gate`, `metal_complex_conductive_fused_pair_device_gate`, both PASS), and
    # `test_metal_weld_contract` already checks those six curated paths against the
    # tree. The gate artifact records every module the gate PROCESS imported — 49
    # and 46 paths — so binding here checks the larger set, and the two routes
    # together check both. The key carries the FILE, not just the directory,
    # because these gates name their artifact after the product rather than
    # `gate.json`.
    "complex_fused_ade_chain": (
        "gate", "metal_complex_fused_ade_chain_2026-10-04_g13s"),
    # RE-GATED 2026-08-21, and the re-gate is what this cut's credit rests on.
    # The 2026-08-20T5 artifact no longer binds: a sibling round added a
    # `_boundary_kinds is None` refusal to `complex_no_pml_conductive._base_reasons`
    # (:213-225, the shared base of four complex families), so the shipped bytes
    # moved from e9f27a4b3af8 to ad4a813fb8b9 AFTER that gate ran. This floor
    # caught it — the merged cut refused to build until the credit was re-bound.
    # The change is a coverage predicate, not arithmetic, and the re-gate says so
    # by measurement rather than by inspection: 42/42 legs PASS, released=True,
    # and the verdict FLIPS to FAIL / released=False against the planted
    # `conductive_tail_drops_the_condinv_pass` defect on the SAME bytes
    # (`../metal_complex_conductive_fused_pair_2026-08-21_flip/pair.json`).
    "complex_conductive_fused_pair": (
        "gate", "metal_complex_conductive_fused_pair_2026-10-05_092_g13s"),
    # THE SEVEN PRODUCTS OF THE 2026-08-30 TRANCHE, all bound to the ONE gate that
    # measures them. `gate_metal_tranche7_fused_pairs.py` runs four legs per family
    # -- identity, mutation, deposit and the binding ceiling -- so a single artifact
    # carries the evidence for all seven, exactly as `below_the_cut_fused_pairs`
    # does for its three. Sharing an artifact does not weaken the binding: this
    # floor checks the FULL set of paths the gate PROCESS imported, and every one of
    # the seven family modules is in that set because the gate builds a plan from
    # each.
    #
    # WHY THEY WERE UNBOUND UNTIL NOW, and it was not that the gate had not run. It
    # HAD run and recorded `verdict: PASS` (results/metal_tranche7_fused_pairs_
    # 2026-08-30/). But that gate called `main()` directly instead of routing
    # through `metal_gate_runner.run_current_measurement`, and never called
    # `gate_provenance.stamp`, so its artifact carried NO `release` block and NO
    # source digests at all -- `n source_sha256: 0`. This floor asks for released
    # bytes and there were none to check, so it refused all seven by name and the
    # board could not be cut. The gate was given the two lines every sibling gate
    # has and RE-RUN; no record was edited to make this bind.
    "beta_fused_electric_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "complex_fused_electric_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "folded_complex_fused_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "folded_beta_complex_fused_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "folded_beta_real_fused_magnetic_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "folded_beta_real_fused_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    "folded_fused_dispersive_pair": (
        "gate", "metal_tranche7_fused_pairs_2026-10-04_g13s"),
    # THE TWO PRODUCTS OF THE 2026-09-01 CUT, bound through the ONE gate that
    # measures both (`gate_metal_no_pml_fused_electric_pairs.py`, six legs per the
    # module docstring: identity, separate_control, mutation, deposit,
    # lifted_refusal, binding_ceiling) — the same one-artifact-many-families shape
    # as `below_the_cut` and `tranche7` above, and the binding checks the FULL set
    # of paths the gate PROCESS imported, which covers both family modules, both
    # certified halves and `launch.py`. Both are ALSO welded in
    # `fingerprints.json` (`metal_no_pml_fused_electric_pairs_device_gate`), so
    # the two routes together check both path sets.
    "no_pml_fused_electric_pair": (
        "gate", "metal_no_pml_fused_electric_pairs_2026-10-04_g13s"),
    "no_pml_conductive_fused_electric_pair": (
        "gate", "metal_no_pml_fused_electric_pairs_2026-10-04_g13s"),
    # THE FOUR PRODUCTS OF THE RESIDUE ROUND, bound to the ONE gate that measures
    # them (`gate_metal_residue_fused_pairs.py`, re-cut at residue2 after the
    # no_pml gate script's final edit invalidated the first cut's manifest —
    # identity, mutation, deposit and
    # binding-ceiling legs per family, with the two packed families' ceilings
    # BISECTED on this host as the cylindrical precedent's gate did). The same
    # one-artifact-many-families shape as `tranche7` above; the binding checks
    # the FULL set of paths the gate PROCESS imported, which covers all four
    # family modules, `coefficient_pack.py`, both certified curl emitters and
    # `launch.py`.
    "bfast_fused_electric_pair": (
        "gate", "metal_residue_fused_pairs_2026-10-04_g13s"),
    "conductive_fused_electric_pair": (
        "gate", "metal_residue_fused_pairs_2026-10-04_g13s"),
    "beta_complex_fused_electric_pair": (
        "gate", "metal_residue_fused_pairs_2026-10-04_g13s"),
    "beta_complex_fused_magnetic_pair": (
        "gate", "metal_residue_fused_pairs_2026-10-04_g13s"),
    # THE FOUR STENCIL WELDS, bound to the ONE gate that measures them
    # (`gate_metal_offdiag_stencil_welds.py`): identity against the array path,
    # identity against the CERTIFIED DEVICE KERNELS each weld replaces, mutation
    # and rotation nulls, the in-seam-source refusal with its admitting control,
    # and the binding ceiling bisected on this host. The binding check covers the
    # FULL set of paths the gate PROCESS imported, which includes all four family
    # modules, `offdiag_weld_common.py`, `coefficient_pack.py`, both certified
    # curl emitters, both certified off-diagonal emitters and `launch.py`.
    "offdiag_fused_electric_pair": (
        "gate", "metal_offdiag_stencil_welds_2026-10-04_g13s"),
    "folded_offdiag_fused_electric_pair": (
        "gate", "metal_offdiag_stencil_welds_2026-10-04_g13s"),
    "complex_no_pml_offdiag_fused_electric_pair": (
        "gate", "metal_offdiag_stencil_welds_2026-10-04_g13s"),
    "folded_complex_offdiag_fused_electric_pair": (
        "gate", "metal_offdiag_stencil_welds_2026-10-04_g13s"),
    # THE FOURTH SEAM'S FIRST PRODUCT, added 2026-09-06 with its release and bound
    # through its GATE ARTIFACT rather than through fingerprints.json, for the reason
    # every `gate` row above gives: the artifact records every module the gate PROCESS
    # imported, which is the larger set of bytes -- and on this product that matters,
    # because the claim rests on the family module, `offdiag_weld_common.py` (whose
    # `ScratchWeldPairPlan` choreography it reuses), both certified emitters in
    # `shaders.py` and `launch.py`. It is ALSO welded in `fingerprints.json`
    # (`metal_fused_hd_pair_device_gate`), so the two routes check both path sets.
    "fused_hd_pair": ("gate", "metal_fused_hd_pair_2026-10-04_g13s"),
    "folded_fused_hd_pair": ("gate",
                             "metal_folded_fused_hd_pair_2026-10-04_g13s"),
    # ADDED 2026-09-06 WITH THE PRODUCT, bound through its GATE ARTIFACT: the claim
    # rests on the family module, `fused_hd_pair.py` (whose `h_cell_function` it
    # lifts), `cylindrical_real.py` (whose scan text and curl it lifts and launches)
    # and `offdiag_weld_common.py` (whose scratch-twin registration it reuses), and a
    # gate artifact records every module the process imported.
    "cylindrical_real_fused_hd_pair":
        ("gate", "metal_cylindrical_real_fused_hd_pair_2026-10-04_g13s"),
    # ADDED 2026-09-06 WITH THE PRODUCT, bound through its GATE ARTIFACT from the
    # post-wire fleet re-cut: the wiring edited bytes the pre-wire gate imported
    # (launch.py, the product module's WELD_OWED), so the pre-wire stamp
    # (_2026-09-06_cyl3) could not be bound and the fleet stamp is.
    # ADDED 2026-09-07 WITH THE PRODUCT, bound through its GATE ARTIFACT. REPOINT
    # at the post-wire fleet re-cut: this stamp is the PRE-WIRE run, and this
    # wiring edits bytes that gate imported (launch.py, the product module's
    # WELD_OWED and its register_arms), so the recorded digests disagree with the
    # tree until the gate is re-run.
    "complex_fused_hd_pair":
        ("gate", "metal_complex_fused_hd_pair_2026-10-04_g13s"),
    "cylindrical_complex_fused_hd_pair":
        ("gate", "metal_cylindrical_complex_fused_hd_pair_2026-10-04_g13s"),
    # EVERY H->D FAMILY IS BOUND TO `metal_<family>_2026-09-10_cells`, a DIRECTORY
    # whose `gate.json` is the artifact. Those eleven runs were cut on 2026-09-10
    # because the withheld-credit answer landed in parity/meep_gpu/h_to_d_seam.py, a
    # file every H->D gate manifest pins (10 of the 42 gate-bound families, and no
    # other); each ran under MEEP_GPU_SUBNORMAL_POLICY=flush -- the only policy
    # metal_kernels/subnormal.ATTAINABLE admits on this executor -- and records that
    # policy inside the artifact, so there is no flush/keep split to choose between
    # here. (The 2026-09-08_close round they replace had run both policies and bound
    # the flush leg by file; the keep leg recorded the refusal.)
    "conductive_fused_hd_pair": (
        "gate", "metal_conductive_fused_hd_pair_2026-10-04_g13s"),
    # THE THREE FAMILIES FIRST EARNED ON 2026-09-08, re-cut with the rest on
    # 2026-09-10. Two of them (beta complex, beta real) also price a SECOND cell from
    # the same artifact's lift records -- see their `cells` tables in PRODUCTS.
    "beta_complex_fused_hd_pair": (
        "gate", "metal_beta_complex_fused_hd_pair_2026-10-04_g13s"),
    "beta_real_fused_hd_pair": (
        "gate", "metal_beta_real_fused_hd_pair_2026-10-04_g13s"),
    "folded_complex_fused_hd_pair": (
        "gate", "metal_folded_complex_fused_hd_pair_2026-10-04_g13s"),
    "bfast_fused_hd_pair": (
        "gate", "metal_bfast_fused_hd_pair_2026-10-04_g13s"),
}


def _assert_every_credit_is_bound_to_released_bytes() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """EVERY product's credited bytes must still BE the bytes its gate certified.

    The `_eop` cut checked this for the E->P family alone. It is exactly as
    load-bearing for the other eleven, so it is asked for all fourteen here, and it
    RAISES: an unverifiable credit is worse than an absent one.
    """
    import hashlib  # noqa: PLC0415
    missing = sorted(set(PRODUCTS) - set(RELEASE_BINDING))
    if missing:
        raise SystemExit(
            f"{missing} are credited by this matrix but named in no RELEASE_BINDING "
            f"row, so their seam-instances rest on bytes nothing here can check")
    fingerprints = json.loads(
        (API_ROOT / "meep_gpu" / "metal_kernels" / "fingerprints.json").read_text())
    out: Dict[str, Any] = {}
    drift: List[str] = []

    def recorded_digests(kind: str, key: str, who: str):
        """(digests, status, when, artifact path or None) for one binding."""
        if kind == "fingerprint":
            entry = fingerprints.get(key)
            if not entry:
                raise SystemExit(f"fingerprints.json carries no {key!r} entry")
            return ({API_ROOT / rel: want
                     for rel, want in (entry.get("source_sha256") or {}).items()},
                    entry.get("status"), _live_recorded_utc(entry, key), None)
        # `key` is either a results DIRECTORY (whose artifact is `gate.json`)
        # or an explicit `directory/artifact.json`. Both forms are accepted so
        # a gate that names its artifact after its product can still be bound.
        relative = key if key.endswith(".json") else f"{key}/gate.json"
        artifact = RESULTS / relative
        gate = json.loads(artifact.read_text())
        if not (gate.get("release") or {}).get("released"):
            raise SystemExit(f"{relative} does not record release.released ({who})")
        return ({Path(rel): want
                 for rel, want in (gate.get("source_sha256") or {}).items()},
                gate.get("verdict"), key, artifact)

    def rehash(digests, label):
        for target, want in digests.items():
            got = (hashlib.sha256(target.read_bytes()).hexdigest()
                   if target.exists() else "MISSING")
            if got != want:
                drift.append(f"{label}: {target} {got[:12]} != {want[:12]}")

    for family in sorted(PRODUCTS):
        kind, key = RELEASE_BINDING[family]
        digests, status, when, _artifact = recorded_digests(kind, key, family)
        if not digests:
            raise SystemExit(f"{family}: {key} records no source digests")
        rehash(digests, f"{family} / {key}")
        out[family] = {"bound_by": kind, "key": key, "status": status,
                       "files_checked": len(digests), "recorded": when}

    # ------------------------------------------------------ THE PER-CELL PASS
    # A SECOND CELL IS A SECOND CLAIM AND CARRIES ITS OWN EVIDENCE. The family's
    # binding above certifies the family's PRIMARY cell; a released run on that cell
    # says nothing about another configuration of the same module -- the plain
    # `fused_hd_pair` artifact holds neither nonlinear row. So every extra cell names
    # the artifact that DROVE it, and that artifact is put through the same three
    # questions: does it still hash to the tree, does it say it drove THIS cell, and
    # does the ledger agree it is the run that was welded.
    cell_bindings: Dict[str, Any] = {}
    for family in sorted(PRODUCTS):
        cells = _cells_of(PRODUCTS[family])
        if len(cells) == 1:
            continue
        for index, cell in enumerate(cells):
            label = _cell_label(tuple(cell["cell"]))
            kind, key = cell["binding"] or RELEASE_BINDING[family]
            who = f"{family} / {label}"
            digests, status, _when, artifact = recorded_digests(kind, key, who)
            if not digests:
                raise SystemExit(f"{who}: {key} records no source digests")
            rehash(digests, who)
            if index:
                if artifact is None:
                    raise SystemExit(
                        f"{who}: bound by {kind!r}; an extra cell must be bound to a "
                        f"gate artifact (see _wiring), and the walk will not skip the "
                        f"cell, rows and ledger checks for a binding it cannot read")
                # THE ARTIFACT MUST NAME THE CELL IT IS CREDITED FOR. Read off the
                # run's own subject block rather than assumed from the directory
                # name, which is a label a campaign chose.
                gate = json.loads(artifact.read_text())
                named = [tuple(v) for v in
                         ((gate.get("product") or {}).get("cells") or {}).values()]
                for block in ("corpus", "subject", "product"):
                    arms = (gate.get(block) or {}).get("cell_arms")
                    if arms:
                        named.append(tuple(arms))
                if tuple(cell["cell"]) not in named:
                    raise SystemExit(
                        f"{who}: {key} names the cells {named} and not this one, so "
                        f"the run bound to this credit measured something else")
                rows = artifact.parent / cell["driven_rows"][0]
                if not rows.is_file():
                    raise SystemExit(
                        f"{who}: {key} carries no {cell['driven_rows'][0]}, so the "
                        f"per-row evidence this cell is credited on is unreadable")
                if cell["fingerprint"]:
                    entry = fingerprints.get(cell["fingerprint"])
                    if not entry:
                        raise SystemExit(
                            f"{who}: fingerprints.json carries no "
                            f"{cell['fingerprint']!r} entry")
                    got = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    # THE RUN THAT CERTIFIED THIS CELL IS ONE OF THE ENTRY'S LIVE RUNS,
                    # one per GPU architecture since the per-architecture records. A
                    # comparison against a top-level field that no longer exists would
                    # read None and pass without comparing anything, so an entry with
                    # no live run recording an artifact refuses by name.
                    welded = {name: run.get("artifact_sha256") for name, run in
                              metal_runs.live_runs(entry).items()}
                    if not any(welded.values()):
                        raise SystemExit(
                            f"{who}: the ledger's {cell['fingerprint']} has no live run "
                            f"recording an artifact_sha256 ({welded}), so nothing says "
                            f"which run certified this cell")
                    if got not in welded.values():
                        raise SystemExit(
                            f"{who}: the ledger's {cell['fingerprint']} welds "
                            f"{ {name: str(value)[:12] for name, value in welded.items()} } "
                            f"and this cell is bound to {got[:12]}; two registries "
                            f"disagree about which run certified this cell")
            cell_bindings[who] = {"bound_by": kind, "key": key, "status": status,
                                  "files_checked": len(digests),
                                  "cell": list(cell["cell"]),
                                  "driven_rows": (list(cell["driven_rows"])
                                                  if cell["driven_rows"] else None),
                                  "fingerprint": cell["fingerprint"]}
    if drift:
        raise SystemExit("the bytes this cut credits are NOT the bytes the gates "
                         "certified:" + "".join(f"\n    {d}" for d in drift))
    kinds = collections.Counter(v["bound_by"] for v in out.values())
    print(f"  RELEASE BINDING: all {len(out)} credited products match their gate's "
          f"recorded bytes ({kinds['fingerprint']} via fingerprints.json, "
          f"{kinds['gate']} via a gate weld manifest)", flush=True)
    if cell_bindings:
        print(f"  RELEASE BINDING, PER CELL: {len(cell_bindings)} cells across "
              f"{len({k.split(' / ')[0] for k in cell_bindings})} multi-cell products, "
              f"each bound to the run that drove it", flush=True)
        for who, detail in sorted(cell_bindings.items()):
            print(f"      {who:64s} {detail['key']}", flush=True)
    return out, cell_bindings


def _live_recorded_utc(entry: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """``{architecture: recorded_utc}`` over the entry's live runs. Refuses with none.

    The run's timestamp lives in ``runs[<architecture>]`` since the per-architecture
    records (``meep_gpu.metal_runs``); an entry with no live run has no run whose time
    could be quoted, and saying ``None`` would read as an unrecorded time rather than
    as a missing certification.
    """
    runs = metal_runs.live_runs(entry)
    if not runs:
        raise SystemExit(f"fingerprints.json's {key!r} has no live run "
                         f"({metal_runs.shape_reasons(entry) or 'every run is stale'})")
    return {name: run.get("recorded_utc") for name, run in runs.items()}


def _assert_the_credited_bytes_are_the_released_ones() -> Dict[str, str]:
    """The E->P product's source must still BE the source its gate certified.

    This matrix credits ten seam-instances to `fused_ade_chain` on the strength of
    a device gate that ran against particular bytes
    (`../metal_fused_ade_chain_2026-08-20T2150/`, PASS, 40/40). The gate recorded
    those bytes' digests in `metal_kernels/fingerprints.json`. If the module has
    moved since, this cut would be crediting a product nothing has certified — the
    exact failure a census is supposed to prevent, silently.

    Checked rather than trusted, and it RAISES: an unverifiable credit is worse
    than an absent one.
    """
    import hashlib  # noqa: PLC0415
    path = API_ROOT / "meep_gpu" / "metal_kernels" / "fingerprints.json"
    entry = json.loads(path.read_text()).get(E_TO_P_RELEASE_FINGERPRINT)
    if not entry:
        raise SystemExit(
            f"{path} carries no {E_TO_P_RELEASE_FINGERPRINT!r} entry, so the ten "
            f"E->P credits below rest on nothing this script can check")
    recorded = entry.get("source_sha256") or {}
    if not recorded:
        raise SystemExit(
            f"{E_TO_P_RELEASE_FINGERPRINT} records no source_sha256; the credit "
            f"cannot be bound to any bytes")
    drift = []
    for relative, want in sorted(recorded.items()):
        target = API_ROOT / relative
        if not target.exists():
            drift.append(f"{relative}: MISSING")
            continue
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != want:
            drift.append(f"{relative}: {got[:12]} != released {want[:12]}")
    if drift:
        raise SystemExit(
            "the bytes this cut credits are NOT the bytes the E->P gate certified:"
            + "".join(f"\n    {line}" for line in drift))
    recorded_utc = _live_recorded_utc(entry, E_TO_P_RELEASE_FINGERPRINT)
    print(f"  RELEASE BINDING: {len(recorded)} files match "
          f"{E_TO_P_RELEASE_FINGERPRINT} ({entry.get('status')}, "
          f"{recorded_utc})", flush=True)
    return {"fingerprint": E_TO_P_RELEASE_FINGERPRINT,
            "files_checked": len(recorded),
            "status": entry.get("status"),
            "recorded_utc": recorded_utc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="directory to write fusion_matrix.json into")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 78, flush=True)
    print("THE METAL FUSION MATRIX — 2026-08-20", flush=True)
    print("=" * 78, flush=True)
    print(f"census : {CENSUS}", flush=True)
    print(f"ceiling: MAX_BUFFER_BINDINGS={MAX_BUFFER_BINDINGS} "
          f"(pointers <= {MAX_POINTERS} with the scalars packed)", flush=True)

    release_binding = _assert_the_credited_bytes_are_the_released_ones()
    all_credits_bound, cell_credits_bound = (
        _assert_every_credit_is_bound_to_released_bytes())

    print("\n--- MEASURING EVERY SHIPPED SIGNATURE -------------------------------",
          flush=True)
    signatures = measure_signatures()

    # THE SHARING CALIBRATION, computed rather than assumed.
    #
    # ONE CORRECTION IS APPLIED FIRST, AND IT IS THE THIRD OF ITS KIND ON THIS BOARD.
    # `shared = a + b - fused` is a SHARED-VOLUME count only when the fused signature
    # binds the same POINTERS the two halves bind, one each for the volumes they have
    # in common. Two distortions of that were already written down below — the E->P
    # eight-pole pad, and the arity-crossed rows. A THIRD arrived on 2026-08-31 with
    # the two Dcyl D->E products: their curl half's six read-only per-axis PML
    # coefficient vectors ride in ONE buffer with six element offsets in the Params
    # struct (`metal_kernels/coefficient_pack.py`), so the fused signature binds FIVE
    # fewer pointers than the halves do FOR A REASON THAT IS NOT SHARING. Left
    # uncorrected each would enter this calibration as `shared = 8`, and the arity
    # class's largest measured sharing is what the UNFUSABLE / BINDING-MARGINAL /
    # FITS brackets are computed against — so a packed product would have moved
    # ANOTHER family's cell across a bucket boundary without one byte of that family
    # changing. MEASURED on the first cut of this round: the BFAST D->E cell (18 + 18
    # pointers, no pack) went from UNFUSABLE ON METAL to BINDING-MARGINAL on exactly
    # that inflation, and moving a bucket by redefining it is the one thing this
    # board may not do.
    #
    # SO THE SAVINGS ARE ADDED BACK, from the product module's OWN declaration rather
    # than from a number typed here, and the declaration is cross-checked against the
    # pack's length: packing N vectors into one buffer saves exactly N - 1 pointers.
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        bfast_fused_electric_pair as _bfast_pack,
        conductive_fused_electric_pair as _conductive_pack,
        cylindrical_fused_electric_pair, cylindrical_real_fused_electric_pair,
        folded_complex_offdiag_fused_electric_pair as _folded_complex_stencil,
        folded_offdiag_fused_electric_pair as _folded_stencil,
        offdiag_fused_electric_pair as _stencil,
    )

    #: The stencil welds carry TWO packs, so their saving is the sum of the two
    #: (N - 1) terms rather than one. The complex no-PML weld packs NOTHING — it
    #: has no per-axis coefficient vector and fits nine under the ceiling — so it
    #: is absent here and its saving is the +0 every unpacked product takes.
    _TWO_PACK_MEMBERS = (len(_stencil.PACKED_VECTORS)
                         + len(_stencil.PACKED_VOLUMES))

    packed_savings: Dict[str, int] = {}
    for family, module in ((_stencil.FAMILY, _stencil),
                           (_folded_stencil.FAMILY, _folded_stencil),
                           (_folded_complex_stencil.FAMILY,
                            _folded_complex_stencil)):
        saved = module.UNPACKED_POINTERS - module.PACKED_POINTERS
        if saved != _TWO_PACK_MEMBERS - 2:
            raise SystemExit(
                f"{family} declares {saved} pointers saved by packing but its two "
                f"packs hold {_TWO_PACK_MEMBERS} members between them, which saves "
                f"{_TWO_PACK_MEMBERS - 2}; the calibration would add back the wrong "
                f"number and price every unbuilt cell of that arity class against it")
        packed_savings[family] = saved

    for family, module in ((cylindrical_real_fused_electric_pair.FAMILY,
                            cylindrical_real_fused_electric_pair),
                           (cylindrical_fused_electric_pair.FAMILY,
                            cylindrical_fused_electric_pair),
                           (_bfast_pack.FAMILY, _bfast_pack),
                           (_conductive_pack.FAMILY, _conductive_pack)):
        saved = module.UNPACKED_POINTERS - module.PACKED_POINTERS
        if saved != len(module.PACKED_VECTORS) - 1:
            raise SystemExit(
                f"{family} declares {saved} pointers saved by packing but its pack "
                f"holds {len(module.PACKED_VECTORS)} vectors, which saves "
                f"{len(module.PACKED_VECTORS) - 1}; the calibration would add back "
                f"the wrong number and price every unbuilt cell of that arity class "
                f"against it")
        packed_savings[family] = saved
    print(f"\n--- PACKED-POINTER SAVINGS ADDED BACK BEFORE SHARING IS MEASURED ----",
          flush=True)
    for family in sorted(packed_savings):
        print(f"  {family:45s} +{packed_savings[family]} (an ARRAY PACK, not a "
              f"shared volume)", flush=True)
    print(f"  every other product: +0", flush=True)

    # A TWO-LAUNCH PRODUCT WHOSE LAUNCHES SHARE NO SIGNATURE is not a splice, so
    # "halves minus fused" is not a sharing count for it: its FUSED source is the
    # constitutive plus the radial scan (24 pointers) and its curl is the certified
    # kernel unchanged. Named here rather than filtered by shape, so the exclusion
    # is a decision a reader can see.
    TWO_LAUNCH_NOT_A_SPLICE = {"cylindrical_complex_fused_hd_pair"}
    calibration = []
    for family, spec in PRODUCTS.items():
        if family in TWO_LAUNCH_NOT_A_SPLICE:
            continue
        if spec["seam"] == "E_to_P":
            # E->P's FIRST half is an update_E CONSTITUTIVE arm, not a curl — the
            # same substitution the gap ranking below already makes
            # (`curl_slot_body = ... if seam != "E_to_P" else CONSTITUTIVE_BODY`).
            curl_body = CONSTITUTIVE_BODY[("update_E", spec["pair"][0])]
            const_body = CONSTITUTIVE_BODY[("update_P", spec["pair"][1])]
        elif spec["seam"] == h_to_d_seam.SEAM:
            # THE ONE SEAM WHOSE HALVES RUN CONSTITUTIVE-THEN-CURL. `pair` is always
            # spelled in the DRIVER'S order, so on H->D `pair[0]` is the update_H
            # constitutive arm and `pair[1]` is the step_D curl arm -- the reverse of
            # every other row here. The two names below stay TRUE to what they hold
            # (`curl_body` is the curl) rather than to the tuple's positions, and the
            # sharing arithmetic below is symmetric in the two, so nothing else moves.
            curl_body = CURL_BODY[spec["pair"][1]]
            const_body = CONSTITUTIVE_BODY[("update_H", spec["pair"][0])]
        else:
            curl_body = CURL_BODY[spec["pair"][0]]
            const_body = CONSTITUTIVE_BODY[
                (SEAMS[0][2] if spec["seam"] == "B_to_H" else SEAMS[1][2],
                 spec["pair"][1])]
        fused = signatures[f"FUSED.{family}"]["pointers"]
        halves = signatures[curl_body]["pointers"] + signatures[const_body]["pointers"]
        calibration.append({
            "family": family, "curl_body": curl_body, "constitutive_body": const_body,
            "curl_pointers": signatures[curl_body]["pointers"],
            "constitutive_pointers": signatures[const_body]["pointers"],
            "fused_pointers": fused, "fused_bindings":
                signatures[f"FUSED.{family}"]["bindings"],
            "packed_pointer_savings": packed_savings.get(family, 0),
            # THE PACK IS ADDED BACK, so this stays a SHARED-VOLUME count. See the
            # block above the loop for what it costs to leave it in.
            "shared_pointers": halves - (fused + packed_savings.get(family, 0)),
            "curl_arity": signatures[curl_body]["component_arity"],
            "constitutive_arity": signatures[const_body]["component_arity"],
            "fused_arity": signatures[f"FUSED.{family}"]["component_arity"],
        })
    shared_values = sorted({entry["shared_pointers"] for entry in calibration})
    print(f"\n--- POINTER SHARING, MEASURED ON THE {len(PRODUCTS)} SHIPPED PRODUCTS "
          f"----------", flush=True)
    for entry in calibration:
        entry["seam"] = PRODUCTS[entry["family"]]["seam"]
        print(f"  {entry['seam']:7s} {entry['family']:30s} curl "
              f"{entry['curl_pointers']:2d} + const "
              f"{entry['constitutive_pointers']:2d} = "
              f"{entry['curl_pointers'] + entry['constitutive_pointers']:2d} -> fused "
              f"{entry['fused_pointers']:2d} ({entry['fused_bindings']:2d} bindings)  "
              f"SHARED {entry['shared_pointers']}"
              + (f" (+{entry['packed_pointer_savings']} packed)"
                 if entry["packed_pointer_savings"] else "") + "  "
              f"arity {entry['curl_arity']}x{entry['constitutive_arity']}"
              f"->{entry['fused_arity']}", flush=True)
    print(f"  measured shared-pointer range, pooled: {shared_values}", flush=True)
    lo_shared, hi_shared = min(shared_values), max(shared_values)

    # PER SEAM, because the two sides do NOT share the same number and the pooled
    # bracket blurs a real difference: all three shipped B->H products share exactly
    # 6, while the D->E side shares 3 and 5. Reported as measured, not explained —
    # the reason lives in which volumes each side carries and this script does not
    # open the bodies.
    seam_shared: Dict[str, List[int]] = collections.defaultdict(list)
    for entry in calibration:
        seam_shared[entry["seam"]].append(entry["shared_pointers"])
    for seam in sorted(seam_shared):
        print(f"  measured shared-pointer values on {seam}: "
              f"{sorted(seam_shared[seam])} "
              f"(n={len(seam_shared[seam])})", flush=True)
    # The E->P chain took the POOLED bracket while it had no shipped product. It has
    # three now, so the bracket is its own measurement — but ONLY if one exists: the
    # fallback stays for a cut that asks about no E->P product.
    #
    # READ THAT NUMBER WITH ITS CAVEAT. "halves minus fused" over-reports sharing at
    # this seam, because BOTH certified E bodies declare EIGHT pole slots always and
    # bind the displacement into the unused ones (no_pml_stored_e.py:213,
    # dispersive_update_e.py:213) while the fused signature declares exactly
    # `pole_count` (fused_ade_chain.py, THE POLE SLOTS ARE NOT PADDED TO EIGHT). At
    # the one-pole specialisation measured here that pad is SEVEN pointers of the
    # difference and they are not shared with anything — they are simply absent.
    # Nothing downstream of this bracket is a correctness claim: it feeds the
    # PREDICTED pointer range printed for cells nobody has built, and at this seam
    # those are the two cells this family does not claim.
    if not seam_shared["E_to_P"]:
        seam_shared["E_to_P"] = list(shared_values)

    # ---------------------------------------------------------------- ARITY
    # THE PER-SEAM POOL IS STILL TOO WIDE, AND ON D->E IT IS WIDE FOR A REASON THAT
    # IS NOT A REASON. Its three values [2, 3, 5] do not disagree about how many
    # volumes two halves share; they disagree about WHAT WAS BEING COUNTED, because
    # the three products are not the same shape of fusion:
    #
    #   folded_fused_pair            curl 3 components x constitutive 3 -> fused 3
    #   fused_dispersive_pair        curl 3            x constitutive 1 -> fused 1
    #   complex_conductive_fused_pair curl 3           x constitutive 1 -> fused 3
    #
    # `shared = a + b - fused` is a SHARED-VOLUME COUNT only when the fused kernel
    # does the same amount of work as the two halves it replaces. On the two
    # arity-crossed rows it does not:
    #   * fused_dispersive_pair takes ONE component of a THREE-component curl, so its
    #     5 is three D volumes it never binds (2 of the curl's target triple, and the
    #     one it keeps is shared) plus the auxiliary triple it likewise drops — an
    #     ABSENCE, not a sharing. Exactly the caveat the E->P block above already
    #     writes down for the eight-pole pad, arrived at from the other direction.
    #   * complex_conductive_fused_pair does the reverse: it absorbs THREE launches of
    #     a one-component constitutive into one launch, so `b` is counted once while
    #     the fused signature pays for it three times, and its 2 is that triple-count
    #     showing through, not a small overlap.
    # Pooling those two against the arity-matched row is comparing a per-component
    # kernel, a triple-count and a shared-volume count as if they measured one thing.
    #
    # SO THE CALIBRATION IS KEYED BY (seam, curl arity, constitutive arity), and a
    # candidate is priced only against shipped products of ITS OWN shape. The bucket
    # boundaries are untouched — FITS, BINDING-MARGINAL and UNFUSABLE ON METAL still
    # mean exactly what they meant against the same MAX_POINTERS. What changes is the
    # measured input, from a pool that mixed three quantities to the one that measures
    # the candidate's.
    #
    # AND THE ARITY-MATCHED NUMBER IS INDEPENDENTLY PREDICTED, which is what stops it
    # being a curve fitted to itself. For a three-against-three D->E pair the shared
    # volumes are the D triple and nothing else:
    #   coverage.CURL_SUB_STEPS['step_D']        == ('Dx', 'Dy', 'Dz')
    #   coverage.CONSTITUTIVE_SIDES['E']['sources'] == ('Dx', 'Dy', 'Dz')
    # — three, stated by the shipped tables with no reference to any binding count.
    # The measurement agrees: folded_fused_pair shares 3.
    #
    # THE MODEL IS THEN CHECKED, NOT ASSUMED. `fused = a + b - shared(class)` with ONE
    # constant per arity class must reproduce every shipped product in a class whose
    # members agree, and the assertion below requires it product by product.
    arity_shared: Dict[Tuple[str, int, int], List[int]] = collections.defaultdict(list)
    arity_members: Dict[Tuple[str, int, int], List[str]] = collections.defaultdict(list)
    for entry in calibration:
        key = (entry["seam"], entry["curl_arity"], entry["constitutive_arity"])
        arity_shared[key].append(entry["shared_pointers"])
        arity_members[key].append(entry["family"])
    print("\n--- THE SAME MEASUREMENT, KEYED BY COMPONENT ARITY ------------------",
          flush=True)
    for key in sorted(arity_shared):
        seam, curl_arity, const_arity = key
        values = sorted(set(arity_shared[key]))
        agree = "ONE VALUE" if len(values) == 1 else "SPREAD — no single constant"
        print(f"  {seam:7s} curl {curl_arity}c x const {const_arity}c : "
              f"shared {values} over {len(arity_shared[key])} products  ({agree})",
              flush=True)
        print(f"          {sorted(arity_members[key])}", flush=True)
    # THE 11 THE MODEL IS CHECKED ON, product by product. Every arity class whose
    # members agree on one shared value is required to reproduce each member's fused
    # pointer count EXACTLY from its two halves. A class that does not agree (D->E
    # 3c x 1c, the two arity-crossed products above) keeps a bracket and is excluded
    # here by measurement rather than by choice — it is named in the print either way.
    exact, checked = [], []
    for key in sorted(arity_shared):
        values = sorted(set(arity_shared[key]))
        if len(values) != 1:
            continue
        for entry in calibration:
            if (entry["seam"], entry["curl_arity"], entry["constitutive_arity"]) != key:
                continue
            predicted = (entry["curl_pointers"] + entry["constitutive_pointers"]
                         - values[0])
            # THE PACK IS ADDED BACK ON THIS SIDE TOO, and for the model check that
            # is not bookkeeping but the whole point: `fused = a + b - shared` is a
            # statement about which VOLUMES the two halves have in common, and an
            # ARRAY PACK is a statement about how many BUFFERS one of them spends on
            # the volumes it does not share. A packed product satisfies the model on
            # its unpacked pointer count and beats it on its bound one, so the check
            # is made against the unpacked count and the pack is reported beside it.
            # Comparing the bound count directly would make every packed product fail
            # a model it does not contradict — and the tempting repair, dropping the
            # class or widening the check, would retire the one assertion that stops
            # the bracket below being a curve fitted to itself.
            bound = entry["fused_pointers"] + entry["packed_pointer_savings"]
            checked.append(entry["family"])
            if predicted == bound:
                exact.append(entry["family"])
            else:
                raise SystemExit(
                    f"the arity-matched model does not reproduce {entry['family']}: "
                    f"{entry['curl_pointers']} + {entry['constitutive_pointers']} - "
                    f"{values[0]} = {predicted} but the emitted fused signature binds "
                    f"{entry['fused_pointers']} pointers"
                    + (f" (+{entry['packed_pointer_savings']} saved by its array "
                       f"pack, added back to {bound})"
                       if entry["packed_pointer_savings"] else "")
                    + "; the calibration below would be a number nothing checks")
    print(f"  MODEL CHECK: fused = curl + constitutive - shared(class) reproduces "
          f"{len(exact)}/{len(checked)} shipped products EXACTLY", flush=True)
    _three_by_three = sorted(
        f for f, k in ((e["family"], (e["seam"], e["curl_arity"],
                                      e["constitutive_arity"])) for e in calibration)
        if k[1] == 3 and k[2] == 3)
    print(f"    of which three-against-three: {len(_three_by_three)} "
          f"({_three_by_three})", flush=True)

    record = rows()
    print(f"\n--- {len(record)} ROWS x {len(SEAMS)} SEAMS -------------------------",
          flush=True)
    # THE LIFT FACTS RIDE WITH THE CONFIGURATION (2026-09-13): dispatch_reachability
    # derives `dimensions` by MEEP's rule from `dimensions_attr`, `cell_size` and the
    # lifted k_point/beta, not from the extent count, which the `complex` round measured
    # wrong on the kz_2d="3d" row and on every (0, 0, L) cell declared 3-D.
    configuration_of = {f"{r['leg']}:{r['row']}": dict(r["configuration"], facts=r.get("facts") or {})
                        for r in record}

    # THE CEILING, BEFORE ANY KERNEL EXISTS. A source injected inside a seam
    # (driver.py:3283-3284, :3294-3299) no longer bars a fused launch by itself: the
    # launch is bracketed and the deposit points are recomputed after the injection
    # (deposit_repair.py). What remains barred is the configuration `deposit_repair.
    # repairable` refuses — see SOURCE_SEAM_CLAUSE. This is the largest number of rows
    # ANY fused product, built or not, can ever serve there; every gap below is
    # bounded by it. The PRE-REPAIR ceiling is computed beside it so the delta this
    # module bought is a printed number rather than a subtraction a reader must do.
    source_ceiling = {
        seam: sum(1 for r in record if clause(r["configuration"]))
        for seam, clause in SOURCE_SEAM_CLAUSE.items()}
    pre_repair_ceiling = {
        "B_to_H": sum(1 for r in record if _no_magnetic_source(r["configuration"])),
        "D_to_E": sum(1 for r in record if _no_electric_source(r["configuration"]))}
    source_ceiling["E_to_P"] = sum(
        1 for r in record if (r.get("plan_step") or {}).get("selected", {}).get("update_P"))
    pre_repair_ceiling["E_to_P"] = source_ceiling["E_to_P"]
    print("\n--- THE SOURCE-SEAM CEILING (a driver fact + what the repair carries) -",
          flush=True)
    kinds = collections.Counter(_source_types(r["configuration"]) for r in record)
    print(f"  source field types across the corpus: {dict(kinds)}", flush=True)
    for seam in ("B_to_H", "D_to_E", "E_to_P"):
        print(f"  {seam:8s} at most {source_ceiling[seam]:3d} / "
              f"{len(record) if seam != 'E_to_P' else source_ceiling['E_to_P']} rows "
              f"can EVER be fused there "
              f"(was {pre_repair_ceiling[seam]:3d} with no deposit repair; "
              f"+{source_ceiling[seam] - pre_repair_ceiling[seam]})", flush=True)

    wired = _wiring()

    by_pair: Dict[Tuple[str, Optional[str], Optional[str]], List[str]] = (
        collections.defaultdict(list))
    served: Dict[str, List[str]] = collections.defaultdict(list)
    product_detail: Dict[str, dict] = {}
    seam_instances = 0
    unfused_reason: Dict[str, collections.Counter] = {
        name: collections.Counter() for name, _c, _s, _w in SEAMS}
    per_row: List[dict] = []

    # PRE-COMPUTE the clause ladders per product so the cumulative counts come out.
    ladders: Dict[str, List[Tuple[str, int]]] = {}
    for family, spec in PRODUCTS.items():
        # ONE WALK, SHARED WITH ``cut_fusion_matrix_served.py``: the clause ladder is
        # per CELL (a product may serve more than one), and a single-cell product's
        # ladder is exactly what this loop computed before cells existed.
        per_cell = ladder(record, spec)
        cells = _cells_of(spec)
        steps = per_cell[tuple(cells[0]["cell"])][0]
        # THE FAMILY'S ROWS ARE THE UNION, TAKEN IN CENSUS ORDER rather than per cell,
        # so a single-cell product's list is byte-for-byte the old one and a two-cell
        # product's reads in the same order as every other list on this board. The
        # cells are disjoint by assertion inside ``ladder``, so this is a union with
        # nothing to resolve.
        admitted_labels = {f"{row['leg']}:{row['row']}"
                           for _steps, rows in per_cell.values() for row in rows}
        rows_admitted = [label for label in
                         (f"{r['leg']}:{r['row']}" for r in record)
                         if label in admitted_labels]
        ladders[family] = steps
        product_detail[family] = {
            "seam": spec["seam"], "pair": list(spec["pair"]),
            "module": spec["module"], "wired_in_the_arm_table": wired.get(family),
            "clause_ladder": [{"clause": k, "surviving": v} for k, v in steps],
            "admitted": len(rows_admitted),
            "rows": rows_admitted,
            "clauses_not_evaluable_from_the_census_block": list(spec["not_evaluable"]),
        }
        if spec.get("cells"):
            # PRINTED ONLY WHERE A PRODUCT DECLARES CELLS, so the 44 single-cell
            # blocks keep the shape every reader and test of this file already knows.
            product_detail[family]["cells"] = {
                _cell_label(tuple(cell["cell"])): {
                    "cell": list(cell["cell"]),
                    "clause_ladder": [{"clause": k, "surviving": v}
                                      for k, v in per_cell[tuple(cell["cell"])][0]],
                    "admitted": len(per_cell[tuple(cell["cell"])][1]),
                    "rows": [f"{r['leg']}:{r['row']}"
                             for r in per_cell[tuple(cell["cell"])][1]],
                    "binding": (list(cell["binding"]) if cell["binding"]
                                else RELEASE_BINDING[family][1]),
                    "bound_by": "cell" if cell["binding"] else "family",
                    "driven_rows": (list(cell["driven_rows"])
                                    if cell["driven_rows"] else None),
                }
                for cell in cells}
        served[family] = product_detail[family]["rows"]

    for index, row in enumerate(record, start=1):
        label = f"{row['leg']}:{row['row']}"
        configuration = row["configuration"]
        selected = (row.get("plan_step") or {}).get("selected", {})
        entry = {"row": label, "seams": {}}
        for seam, curl_slot, const_slot, between in SEAMS:
            if seam == "E_to_P" and not selected.get("update_P"):
                continue          # the row has no update_P pass; the seam does not exist
            seam_instances += 1
            curl_arm = selected.get(curl_slot)
            const_arm = selected.get(const_slot)
            by_pair[(seam, curl_arm, const_arm)].append(label)
            product = next(
                (f for f, s in PRODUCTS.items()
                 if s["seam"] == seam and tuple(s["pair"]) == (curl_arm, const_arm)),
                None)
            admitted = bool(product and label in served[product])
            if not product:
                if curl_arm is None or const_arm is None:
                    unfused_reason[seam]["a slot is UNSELECTED (array path)"] += 1
                elif CONSTITUTIVE_BODY.get((const_slot, const_arm)) is None:
                    unfused_reason[seam]["NOT A FUSION CANDIDATE — the constitutive sub-step launches nothing, so there is no second kernel to weld"] += 1
                else:
                    unfused_reason[seam]["no fused product covers this pair"] += 1
            elif not admitted:
                unfused_reason[seam]["a product exists; its predicate REFUSES"] += 1
            entry["seams"][seam] = {
                "curl_arm": curl_arm, "constitutive_arm": const_arm,
                "between": between, "product": product, "served": admitted,
                # READ off plan_step.live, not derived — see live_in_seam_passes.
                "live_in_seam_passes": list(live_in_seam_passes(row, seam))}
        per_row.append(entry)
        if index % 20 == 0 or index == len(record):
            print(f"  row {index:3d}/{len(record)}  {label[:56]:56s} "
                  f"({time.time() - started:5.1f} s)", flush=True)

    total_served = sum(1 for e in per_row for s in e["seams"].values() if s["served"])

    # ------------------------------------------- THE FOURTH SEAM, update_H -> step_D
    # One instance per row (2026-09-04). The two halves are arms this walk already
    # selected — update_H is B_to_H's constitutive half, step_D is D_to_E's curl half
    # — and the null verdict is the CONSTITUTIVE_BODY rule B_to_H applies, asked of
    # the FIRST half here. Priced by h_to_d_seam.price below, from the probe artifact
    # (the withdraw is measured per lifted row, not read off a name), into the same
    # taxonomy. `seam_instances` stays the three-seam ledger's own denominator; the
    # board's PRICED denominator is `priced`.
    #
    # AND, FROM 2026-09-06, A PER-ROW `served_by`. `h_to_d_seam.price` cannot evaluate
    # a per-backend predicate -- it is the one rule three boards share -- so a board
    # that names a product spanning this seam must answer FOR EVERY ROW, and a row left
    # unanswered is an absence rather than a gap. The answer is the same clause ladder
    # every other product on this board is priced by (`served[family]`, built above
    # from the census columns and this product's own clauses), so the H->D verdict and
    # the three-seam verdicts are one walk and not two.
    #
    # NO `credit_withheld` KEY ON THIS BOARD, BY MEASUREMENT NOT OMISSION (2026-09-10).
    # `h_to_d_seam.price` accepts a third answer (a withheld credit, the board's reason
    # sentence) beside served_by = name / None, and this board has no withheld table
    # to answer from: `_assert_every_credit_is_bound_to_released_bytes` refuses a
    # drifted credit before any walk runs, so a Metal cut either binds every credit or
    # does not build. Absent means credited, by the shared contract. The day this
    # board grows a table, the key is written here, per row, from it -- AND two sites
    # below that count ADMITS via `served_by` truthiness would have to count CREDITED
    # rows instead (bucket == served, or `not credit_withheld`): `h_to_d_credits` in
    # the disjointness cross-check and `seam_credited_per_seam.H_to_D` in its block.
    # Until then the cross-check RAISES on the disagreement rather than mis-publishing
    # it, because the taxonomy's served ledger it compares against would already
    # exclude the withheld rows.
    h_to_d_spanning = sorted(f for f, s in PRODUCTS.items()
                             if s["seam"] == h_to_d_seam.SEAM)
    h_to_d_entries = [
        {"row": e["row"],
         "update_H": e["seams"]["B_to_H"]["constitutive_arm"],
         "step_D": e["seams"]["D_to_E"]["curl_arm"],
         "update_H_is_null": (
             e["seams"]["B_to_H"]["constitutive_arm"] is not None
             and CONSTITUTIVE_BODY.get(
                 ("update_H", e["seams"]["B_to_H"]["constitutive_arm"])) is None),
         "served_by": next((f for f in h_to_d_spanning if e["row"] in served[f]), None)}
        for e in per_row]
    priced = seam_instances + len(h_to_d_entries)

    # ---------------------------------------------- WHICH IN-SEAM PASSES ARE CARRIED
    # MEASURED ON THE CORPUS, NOT READ OFF ``REPLACES``. A product's ``REPLACES``
    # lists the driver call sites one launch takes over INCLUDING INERT ONES, so
    # reading it as a carry reports a pass as carried on the strength of a product
    # that never executes it. A pass counts as CARRIED only where some product
    # ADMITS an instance on which that pass is LIVE.
    #
    # FIRST, THE CORRECTION THIS CUT OWES THE TWO 2026-08-20 CARRY CUTS. They derived
    # liveness from the boundary triple; this one reads plan_step.live. The two are
    # compared here rather than one quietly replacing the other.
    disagreement: Dict[str, int] = collections.Counter()
    for row in record:
        for seam in IN_SEAM_PASSES:
            measured = set(live_in_seam_passes(row, seam))
            guessed = set(derived_in_seam_passes(row["configuration"], seam))
            for name in IN_SEAM_PASSES[seam]:
                if (name in measured) != (name in guessed):
                    disagreement[name] += 1
    print("\n--- IN-SEAM PASS LIVENESS: THE ENGINE'S RECORD vs THE DERIVED RULE ---",
          flush=True)
    print("  the 2026-08-20 carry cuts DERIVED this from the boundary triple; this "
          "cut READS plan_step.live", flush=True)
    for seam, names in IN_SEAM_PASSES.items():
        for name in names:
            live = sum(1 for r in record if name in live_in_seam_passes(r, seam))
            guess = sum(1 for r in record
                        if name in derived_in_seam_passes(r["configuration"], seam))
            mark = ("AGREES" if not disagreement[name]
                    else f"DISAGREES ON {disagreement[name]} ROWS — the derived rule "
                         f"is WRONG and the carry cuts published its number")
            print(f"    {seam:7s} {name:26s} live on {live:4d} / {len(record)} rows "
                  f"(derived rule said {guess:4d})  {mark}", flush=True)

    carried_in_practice: Dict[str, set] = {seam: set() for seam in IN_SEAM_PASSES}
    for entry in per_row:
        for seam, block_ in entry["seams"].items():
            if seam in carried_in_practice and block_["served"]:
                carried_in_practice[seam].update(block_["live_in_seam_passes"])
    # A SEAM WITH NOTHING SERVED CANNOT ANSWER THE QUESTION. Reported as a third
    # state rather than as "not carried", which would be an inference from ignorance.
    seam_has_evidence = {
        seam: any(b["served"] for e in per_row for s, b in e["seams"].items()
                  if s == seam)
        for seam in IN_SEAM_PASSES}
    print("\n--- WHICH IN-SEAM PASSES A PRODUCT ACTUALLY CARRIES ------------------",
          flush=True)
    for seam, names in IN_SEAM_PASSES.items():
        for name in names:
            if not seam_has_evidence[seam]:
                mark = "UNDETERMINED — nothing is served at this seam"
            elif name in carried_in_practice[seam]:
                mark = "CARRIED"
            else:
                mark = "NOT CARRIED BY ANY PRODUCT"
            print(f"    {seam:7s} {name:26s} {mark}", flush=True)

    # THE CARRY GAP: reachable, unserved, and running a pass nothing carries.
    carry_gap: Dict[str, List[dict]] = collections.defaultdict(list)
    ceiling_clause = SOURCE_SEAM_CLAUSE
    for entry, row in zip(per_row, record):
        for seam, block_ in entry["seams"].items():
            if seam not in IN_SEAM_PASSES or block_["served"]:
                continue
            clause = ceiling_clause.get(seam)
            if clause is not None and not clause(row["configuration"]):
                continue                    # blocked by the source seam; not reachable
            if not seam_has_evidence[seam]:
                continue
            for name in block_["live_in_seam_passes"]:
                if name in carried_in_practice[seam]:
                    continue
                carry_gap[name].append({
                    "row": entry["row"], "seam": seam,
                    "cell": [block_["curl_arm"], block_["constitutive_arm"]],
                    "has_a_product": bool(block_["product"])})
    print("\n--- THE CARRY GAP — reachable, unserved, waiting on ONE uncarried pass",
          flush=True)
    if not carry_gap:
        print("    none", flush=True)
    for name, entries in sorted(carry_gap.items(), key=lambda kv: -len(kv[1])):
        inside = sum(1 for e in entries if e["has_a_product"])
        print(f"    {name:26s} {len(entries):4d} seam-instances "
              f"({inside} inside a cell a SHIPPED product already covers and REFUSES)",
              flush=True)
        by_cell = collections.Counter(
            (e["seam"], str(e["cell"][0]), str(e["cell"][1])) for e in entries)
        for (seam, ca, cb), count in by_cell.most_common():
            print(f"          {count:4d}  {seam:7s}  ({ca}, {cb})", flush=True)

    # A MEASURED CONSISTENCY CHECK, not a comment. Each product's clause ladder is
    # evaluated over ALL census rows, with no reference to which arm the planner picked.
    # The seam walk credits a product only where the planner ALSO selected that
    # product's (curl, constitutive) pair. If any row were admitted by a product whose
    # pair the planner does not select, the two totals would differ — and they would
    # differ because the arm predicates had stopped being disjoint, which is the
    # composer's whole contract (arms.py, `_select_slot`). They must agree exactly.
    #
    # ASKED PER SEAM FROM 2026-09-06, AND THAT IS STRICTLY MORE THAN IT WAS. The
    # three-seam walk above (`total_served`) does not visit H->D at all -- that seam is
    # priced by `h_to_d_seam` from `h_to_d_entries` -- so a single pooled equality
    # would compare a ladder total that INCLUDES the H->D product against a credit
    # total that EXCLUDES it, and the only way to keep it green would be to stop
    # asking. Split, each side answers for its own walk and the two can no longer
    # offset one another: an over-admission on one seam and an under-credit on another
    # used to cancel in the pooled sum.
    three_seam_ladder = sum(detail["admitted"] for family, detail
                            in product_detail.items()
                            if PRODUCTS[family]["seam"] != h_to_d_seam.SEAM)
    if three_seam_ladder != total_served:
        raise SystemExit(
            f"the three-seam clause ladders admit {three_seam_ladder} rows but only "
            f"{total_served} seam-instances credit a product: some row is admitted by "
            f"a product whose pair plan_step does not select, so the arm predicates "
            f"are no longer disjoint and this matrix cannot be trusted")
    h_to_d_ladder = sum(detail["admitted"] for family, detail
                        in product_detail.items()
                        if PRODUCTS[family]["seam"] == h_to_d_seam.SEAM)
    h_to_d_credits = sum(1 for entry in h_to_d_entries if entry["served_by"])
    if h_to_d_ladder != h_to_d_credits:
        raise SystemExit(
            f"the H->D clause ladders admit {h_to_d_ladder} rows but only "
            f"{h_to_d_credits} H->D seam-instances carry a `served_by`: the two walks "
            f"disagree about which rows a spanning product admits, and the seam's "
            f"served count is derived from the second")
    ladder_total = three_seam_ladder + h_to_d_ladder
    print(f"\n  DISJOINTNESS CROSS-CHECK: three-seam clause ladders "
          f"{three_seam_ladder} == seam credits {total_served}; H->D ladders "
          f"{h_to_d_ladder} == H->D credits {h_to_d_credits}", flush=True)

    def disjointness_cross_check(taxonomy_block: dict) -> dict:
        """The JSON block, DERIVED from the taxonomy's served ledger — and a floor.

        UNTIL 2026-09-06 THIS BLOCK'S CREDIT SIDE WAS THE THREE-SEAM WALK'S TOTAL
        while its ladder side pooled all four seams, so the first board with an H->D
        product (``fusion_matrix_metal_2026-09-06_hd``) wrote ``clause_ladder_total:
        405, seam_credited_total: 358, agree: false`` — and wrote it, because the
        block recorded the verdict instead of refusing on it. Recomputed from that
        artifact: three-seam ladders 358 == three-seam credits 358, H->D ladders 47 ==
        H->D credits 47, every product's admitted row set identical to the rows the
        seam walk credits it, and 0 served instances whose product pair is not the
        pair plan_step selected. The CHECK was wrong (a stale credit denominator), the
        credits were not. It now reads the credit side off the same instance ledger
        the headline derives from, compares every product's ladder rows to its
        credited rows as SETS rather than counts, and raises rather than records.
        """
        credited = taxonomy_block["buckets"]["served"]
        by_product = {product: sorted({row for _seam, row in items})
                      for product, items in taxonomy_block["served_by_product"].items()}
        row_set_disagreements = {}
        for family, detail in product_detail.items():
            ladder_rows = sorted(detail["rows"])
            if ladder_rows != by_product.get(family, []):
                row_set_disagreements[family] = {
                    "ladder_admits_not_credited":
                        sorted(set(ladder_rows) - set(by_product.get(family, []))),
                    "credited_not_admitted":
                        sorted(set(by_product.get(family, [])) - set(ladder_rows))}
        unknown = sorted(set(by_product) - set(product_detail))
        block = {
            "clause_ladder_total": ladder_total,
            "clause_ladder_per_seam": {"three_seam_ledger": three_seam_ladder,
                                       "H_to_D": h_to_d_ladder},
            "seam_credited_total": credited,
            "seam_credited_per_seam": {"three_seam_ledger": total_served,
                                       "H_to_D": h_to_d_credits},
            "credited_read_from": "taxonomy.buckets.served / taxonomy.served_by_product",
            "products_whose_ladder_rows_differ_from_their_credited_rows":
                row_set_disagreements,
            "products_credited_but_not_in_the_product_table": unknown,
            "agree": (ladder_total == credited and three_seam_ladder == total_served
                      and h_to_d_ladder == h_to_d_credits
                      and not row_set_disagreements and not unknown),
            "what_it_proves": (
                "every row a product admits is a row whose (curl, constitutive) pair "
                "plan_step independently selects, which is the arm table's "
                "disjointness contract holding on this corpus — asked per seam AND per "
                "product as row sets, over all four seams"),
        }
        if not block["agree"]:
            raise SystemExit(
                f"DISJOINTNESS CROSS-CHECK FAILED: clause ladders admit "
                f"{ladder_total} rows ({three_seam_ladder} three-seam + "
                f"{h_to_d_ladder} H->D) and the instance ledger credits {credited} "
                f"({total_served} + {h_to_d_credits}); products whose ladder rows are "
                f"not their credited rows: {row_set_disagreements}; credited products "
                f"the table does not list: {unknown}. Some row is admitted by a "
                f"product whose pair plan_step does not select, or credited by one "
                f"whose ladder refuses it, and this matrix cannot be trusted.")
        print(f"  DISJOINTNESS CROSS-CHECK (derived): ladders {ladder_total} == "
              f"credited {credited} over {len(product_detail)} products, row sets "
              f"identical", flush=True)
        return block

    # ------------------------------------------------------------------ gaps
    gaps = []
    for (seam, curl_arm, const_arm), labels in by_pair.items():
        product = next((f for f, s in PRODUCTS.items()
                        if s["seam"] == seam and tuple(s["pair"]) == (curl_arm, const_arm)),
                       None)
        if product is not None:
            continue
        const_slot = {"B_to_H": "update_H", "D_to_E": "update_E",
                      "E_to_P": "update_P"}[seam]
        curl_slot_body = (CURL_BODY.get(curl_arm) if seam != "E_to_P"
                          else CONSTITUTIVE_BODY.get(("update_E", curl_arm)))
        const_body = CONSTITUTIVE_BODY.get((const_slot, const_arm))
        gap = {"seam": seam, "curl_arm": curl_arm, "constitutive_arm": const_arm,
               "rows": len(labels), "example_rows": sorted(labels)[:3]}
        # THE STENCIL REFUSAL, AND IT OUTRANKS EVERY BINDING VERDICT BELOW.
        #
        # `stepping._offdiagonal_terms` (stepping.py:1235-1253) reads each partner
        # component's D volume at FOUR indices — its own, one down the partner axis,
        # one up the component's own axis, and the corner — and those D volumes are
        # exactly what `step_D` writes, IN PLACE, in the first half of the same
        # launch. A fused launch would have program A reading a cell program B is
        # concurrently writing, and no grid-wide barrier exists inside one launch on
        # Metal (nor on Triton; on CUDA cooperative groups is one, at the cost of
        # capping the grid at the resident-block count and unreachable from Triton).
        # Its only two callers are inside `update_E` (stepping.py:1006 and, via
        # `_nonlinear_constitutive`, :1099), so this is a D->E fact and NOT a B->H
        # one: `update_H` never calls it.
        #
        # MEASURED FROM THE CENSUS, NOT FROM AN ARM NAME: a cell earns this verdict
        # when EVERY row the planner routes into it declares has_offdiagonal_epsilon.
        # Established on device on Triton (RTX A6000, both float32 subnormal cuts) by
        # the 2026-08-20 off-diagonal round, which built the fused kernel spliced
        # verbatim from the two shipped bodies and measured it failing.
        #
        # THIS CORRECTS THE STANDING METAL RANKING. It printed
        # (complex no-PML curl, complex no-PML off-diagonal) as "FITS — 22..24
        # pointers against a 30 ceiling; NOT BUILT", which reads as an invitation.
        # It fits on bindings and is structurally refused anyway.
        offdiagonal_cell = (
            seam == "D_to_E" and bool(labels)
            and all(bool(configuration_of[l]["has_offdiagonal_epsilon"])
                    for l in labels))
        # THE SAME MATERIAL REFUSES THE E->P SEAM TOO, FOR A DIFFERENT REASON, AND
        # THE STANDING CUT DID NOT SAY SO. At D->E the off-diagonal row is a
        # STENCIL over the curl half's in-place output. At E->P there is no curl
        # half and no stencil — the refusal is the INTERLEAVE. Every fused E->P
        # product on this backend is one launch PER COMPONENT, `update_E(c)` then
        # every `update_P(state, c)`, and that is the driver's order only while
        # `update_E(c)` reads nothing but component c's own volumes. Under a
        # chi1inv row it reads the PARTNERS' `D - sum P`, so the partner's
        # constitutive would consume a polarization `update_P` had already
        # advanced.
        #
        # MEASURED, NOT ARGUED: results/metal_eop_interleave_dependency_2026-08-20T3
        # perturbs ONE component's P by one word on the array path and counts which
        # components' E move. Diagonal, both storage widths: only the perturbed
        # component moves (1 word) — which is the licence the four served E->P
        # cells stand on. Off-diagonal, both storage widths: BOTH partners move
        # (4 words each). So the fifth cell's standing verdict, `FITS — 23..23
        # pointers against a 30 ceiling; NOT BUILT`, is a binding verdict and not
        # the operative one, exactly as at D->E.
        interleave_cell = (
            seam == "E_to_P" and bool(labels)
            and all(bool(configuration_of[l]["has_offdiagonal_epsilon"])
                    for l in labels))
        gap["structurally_unfusable"] = bool(offdiagonal_cell or interleave_cell)
        gap["refused_at_the_interleave"] = bool(interleave_cell)
        if curl_arm is None or const_arm is None:
            gap["verdict"] = "UNFUSABLE — a slot is UNSELECTED; there is no pair"
        elif const_body is None:
            gap["verdict"] = ("NOT A FUSION CANDIDATE — the constitutive arm is the "
                              "NULL plan (no_pml_constitutive.py:457 "
                              "performs_device_work = False); the sub-step launches "
                              "nothing, so there is no second kernel to fuse with")
        elif curl_slot_body is None or curl_slot_body not in signatures \
                or const_body not in signatures:
            gap["verdict"] = ("UNDETERMINED — this script could not emit one of the "
                              "two bodies, so no binding count is claimed")
        else:
            a = signatures[curl_slot_body]["pointers"]
            b = signatures[const_body]["pointers"]
            curl_arity = signatures[curl_slot_body]["component_arity"]
            const_arity = signatures[const_body]["component_arity"]
            seam_share = sorted(seam_shared[seam])
            # PRICED AGAINST ITS OWN SHAPE. `arity_shared` is keyed by (seam, curl
            # components, constitutive components) precisely so a three-against-three
            # candidate is never calibrated against a product that fused a
            # three-component curl to a one-component constitutive — see the ARITY
            # block above for what those rows actually measure.
            share = sorted(arity_shared.get((seam, curl_arity, const_arity), []))
            gap.update({"curl_body": curl_slot_body, "constitutive_body": const_body,
                        "curl_pointers": a, "constitutive_pointers": b,
                        "curl_arity": curl_arity, "constitutive_arity": const_arity,
                        "shared_pointers_measured_on_this_arity_class": share,
                        "arity_class_members": sorted(
                            arity_members.get((seam, curl_arity, const_arity), [])),
                        "shared_pointers_measured_on_this_seam": seam_share,
                        "seam_pooled_bracket": [a + b - max(seam_share),
                                                a + b - min(seam_share)],
                        "pooled_bracket": [a + b - hi_shared, a + b - lo_shared],
                        "ceiling_pointers": MAX_POINTERS})
            if not share:
                # IGNORANCE IS A THIRD STATE. No shipped product fuses this shape, so
                # there is nothing to calibrate against; falling back to the seam pool
                # would be reinstating exactly the comparison this block refuses.
                gap["fused_pointer_bracket"] = None
                gap["verdict"] = (
                    f"NO ARITY-MATCHED CALIBRATION — this candidate fuses a "
                    f"{curl_arity}-component first half to a {const_arity}-component "
                    f"second, and no shipped product on {seam} has that shape (the "
                    f"only shapes shipped there are "
                    f"{sorted((k[1], k[2]) for k in arity_members if k[0] == seam)}), "
                    f"so this script states NO pointer bracket; the seam-pooled "
                    f"range would be "
                    f"{a + b - max(seam_share)}..{a + b - min(seam_share)} and it "
                    f"would be measuring a different quantity")
            else:
                low, high = a + b - max(share), a + b - min(share)
                gap["fused_pointer_bracket"] = [low, high]
                if low > MAX_POINTERS:
                    gap["verdict"] = (
                        f"UNFUSABLE ON METAL — even at the largest sharing measured "
                        f"on this seam at this arity ({max(share)}) the fused "
                        f"signature needs {low} pointers, over the {MAX_POINTERS} the "
                        f"platform allows (device.py:72)")
                    # THIS REFUSAL IS THIS PLATFORM'S ALONE, and saying so is the
                    # difference between "nobody can build it" and "the wrong track
                    # is holding it". MAX_BUFFER_BINDINGS = 31 is a Metal argument-
                    # table limit; CUDA passes its arguments in a 32,764-byte
                    # parameter space and the widest kernel in that tree spends 256
                    # bytes of it, so no CUDA candidate is ever refused on this
                    # ground. A cell that clears every OTHER clause and is over here
                    # by a handful of pointers is a CUDA-track cell, not a dead one.
                    if not gap["structurally_unfusable"]:
                        gap["over_by_binding_alone"] = (
                            f"OVER BY {low - MAX_POINTERS} POINTER"
                            f"{'S' if low - MAX_POINTERS != 1 else ''} AND BY NOTHING "
                            f"ELSE HERE — the ceiling refusing it is Metal's argument "
                            f"table (MAX_BUFFER_BINDINGS={MAX_BUFFER_BINDINGS}, "
                            f"device.py:72), which has no CUDA analogue: that backend "
                            f"passes arguments in a 32,764-byte parameter space and "
                            f"its widest kernel spends 256 bytes. This cell belongs "
                            f"to the CUDA track")
                elif high <= MAX_POINTERS:
                    exact = (" EXACTLY ON THE CEILING, zero headroom"
                             if low == high == MAX_POINTERS else "")
                    gap["verdict"] = (f"FITS — {low}..{high} pointers against a "
                                      f"{MAX_POINTERS} ceiling{exact}; NOT BUILT")
                else:
                    gap["verdict"] = (
                        f"BINDING-MARGINAL — {low}..{high} pointers against a "
                        f"{MAX_POINTERS} ceiling; which end it lands on turns on how "
                        f"many volumes the two halves share, and the "
                        f"{curl_arity}c x {const_arity}c products on this seam "
                        f"measure {share}")
        if gap.get("refused_at_the_interleave"):
            gap["binding_verdict_had_it_been_buildable"] = gap.pop("verdict", None)
            gap["verdict"] = (
                "STRUCTURALLY REFUSED AT THE INTERLEAVE — every fused E->P product "
                "on this backend is ONE LAUNCH PER COMPONENT (update_E(c), then "
                "every update_P(state, c)), which is the driver's order only while "
                "update_E(c) reads nothing but component c's own volumes. MEASURED "
                "by perturbation on the array path "
                "(results/metal_eop_interleave_dependency_2026-08-20T3): diagonal, "
                "perturbing P[c] moves ONLY E_c (1 word, both storage widths); "
                "off-diagonal, it moves BOTH partners (4 words each). The only "
                "remaining fused shape takes all three components in one launch, "
                "which reintroduces the intra-launch alias fused_ade_chain exists "
                "to avoid. NOT a binding verdict — the bracket this script computed "
                "is kept alongside and does not change the answer")
        elif gap["structurally_unfusable"]:
            gap["binding_verdict_had_it_been_buildable"] = gap.pop("verdict", None)
            gap["verdict"] = (
                "UNFUSABLE AS AN IN-PLACE SPLICE — the off-diagonal constitutive "
                "arm is a STENCIL over the curl arm's D output "
                "(stepping.py:1235-1253 reads the partner D volume at four "
                "indices), so a kernel in which the curl half writes D IN PLACE "
                "would need a grid-wide barrier that does not exist inside one "
                "launch. MEASURED on Triton (2026-08-20) on exactly that splice. "
                "THIS IS NOT A PROPERTY OF THE SEAM, and this board said it was "
                "until 2026-09-01: the four Metal SCRATCH-OUTPUT welds "
                "(offdiag_fused_electric_pair and its three siblings) write D and "
                "fu_D to launch-local scratch, re-derive every foreign tap from "
                "PRE-LAUNCH state through the curl's own body as an inline "
                "function, and rotate the buffers after the launch returns — "
                "nothing written is ever read, so no barrier is wanted. A cell "
                "still carrying this verdict is one no such product covers YET, "
                "not one nothing can. NOT a binding verdict — the binding bracket "
                "this script computed is kept alongside and does not change the "
                "answer")

        # THE E->P CHAIN CARRIES A CONSTRAINT THE BINDING COUNT DOES NOT SEE, and it
        # is not a source seam. `ade_update_p` launches ONCE PER DRIVEN COMPONENT with
        # its buffer arguments RE-RESOLVED between launches: `PolarizationState.update`
        # rotates P, P_prev and a shared scratch after EACH component, and the plan
        # "applies the reference rotation only after that launch succeeds"
        # (ade_update_p.py:13-18, launches_per_run = len(entries) at :353). A single
        # update_E launch writes all three components; a fused E->P kernel would have
        # to either absorb only the FIRST component's P launch or bake a rotation the
        # host currently performs between launches. NEITHER HAS BEEN MEASURED. The
        # bracket below is therefore a binding-fitness verdict ONLY and must not be
        # read as "buildable".
        if seam == "E_to_P":
            gap["sequencing_constraint"] = (
                "ade_update_p launches once per driven component and the host rotates "
                "P / P_prev / scratch between launches (ade_update_p.py:13-18, :353); "
                "a fused E->P kernel must either take one component or bake that "
                "rotation. NOT MEASURED — the pointer bracket is a fitness verdict, "
                "not a buildability one")
        # A GAP WHOSE ROWS RUN AN UNCARRIED IN-SEAM PASS IS NOT AN INVITATION, and
        # the binding bracket above does not say so. The carry column already knows
        # which cells those are; joining it here is what stops a reader taking
        # `FITS — NOT BUILT` at face value on a cell whose real obstruction has
        # already been measured. The join is by CELL, from the carry gap computed
        # above, so it moves with the census rather than being a hand-kept list.
        uncarried = sorted({name for name, entries in carry_gap.items()
                            for e in entries
                            if e["seam"] == seam
                            and tuple(e["cell"]) == (curl_arm, const_arm)})
        if uncarried:
            gap["in_seam_passes_no_product_carries"] = uncarried
            citation = CARRY_EVIDENCE.get((seam, curl_arm, const_arm))
            if citation:
                gap["carry_measured_by"] = citation
        # THE SOURCE SEAM IS A CEILING ON THIS GAP, not a clause of a future kernel.
        # A row whose seam carries a source injection can never be fused there,
        # however the kernel is written — the driver deposits into the field between
        # the halves (driver.py:3283-3284 magnetic, :3294-3299 electric). So the
        # DEMAND that a built product could actually reach is the subset that clears
        # it, and a gap's raw row count overstates it wherever they differ.
        clears = SOURCE_SEAM_CLAUSE.get(seam)
        gap["rows_clearing_the_source_seam"] = (
            len(labels) if clears is None
            else sum(1 for lab in labels if clears(configuration_of[lab])))
        gaps.append(gap)
    gaps.sort(key=lambda g: (-g["rows_clearing_the_source_seam"], -g["rows"],
                             g["seam"], str(g["curl_arm"])))

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 78, flush=True)
    print("THE MATRIX — every (curl, constitutive) pair the corpus drives", flush=True)
    print("=" * 78, flush=True)
    for seam, _c, _s, between in SEAMS:
        print(f"\n--- {seam}   [between the halves: {between}]", flush=True)
        entries = sorted(((k, v) for k, v in by_pair.items() if k[0] == seam),
                         key=lambda kv: -len(kv[1]))
        for (_seam, curl_arm, const_arm), labels in entries:
            product = next((f for f, s in PRODUCTS.items()
                            if s["seam"] == seam
                            and tuple(s["pair"]) == (curl_arm, const_arm)), None)
            n_served = sum(1 for lab in labels if product and lab in served[product])
            mark = f"{product} -> {n_served}/{len(labels)} admitted" if product else "-"
            print(f"  {len(labels):4d}  ({str(curl_arm)}, {str(const_arm)})"
                  f"{'':>{max(0, 58 - len(str(curl_arm)) - len(str(const_arm)))}}{mark}",
                  flush=True)

    print("\n" + "=" * 78, flush=True)
    print("THE AGGREGATE", flush=True)
    print("=" * 78, flush=True)
    print(f"  denominator (three-seam ledger)     : {seam_instances}  (the board's "
          f"PRICED denominator is {priced}: + {len(h_to_d_entries)} H_to_D)",
          flush=True)
    print(f"  SERVED BY A FUSED PRODUCT           : {total_served}", flush=True)
    print(f"  fraction (of the three-seam ledger) : "
          f"{total_served / seam_instances:.1%}", flush=True)
    for seam, _c, _s, _w in SEAMS:
        n = sum(1 for e in per_row if seam in e["seams"])
        k = sum(1 for e in per_row if e["seams"].get(seam, {}).get("served"))
        print(f"    {seam:8s} {k:4d} / {n:4d}", flush=True)
    print("  why the rest are not served:", flush=True)
    for seam, _c, _s, _w in SEAMS:
        for reason, count in unfused_reason[seam].most_common():
            print(f"    {seam:8s} {count:4d}  {reason}", flush=True)
    # DERIVED FROM THE LIVE READ, NOT SPELLED. `wired` came from arms.registered()
    # a few hundred lines above; a numeral or an adjective typed here would be a
    # second, unchecked claim about it. If some product ever IS wired this prints
    # the count that is, and the dispatch total below stops being trivially zero.
    _unwired = sorted(f for f, state in wired.items() if not state)
    _are_wired = sorted(f for f, state in wired.items() if state)
    print(f"\n  DISPATCH IS A SEPARATE AND SMALLER NUMBER: {len(_unwired)} of "
          f"{len(wired)} products register wired=False"
          + (f"; WIRED: {_are_wired}" if _are_wired else ""), flush=True)
    print(f"    live arm-table wiring: {wired}", flush=True)
    print("    arms.arms_for skips an unwired arm (arms.py:'arms_for'), so plan_step "
          "cannot select any", flush=True)
    print("    of them, and meep_gpu.fastpath.plan_fast_path returns None on every "
          "branch. SEAM-INSTANCES", flush=True)
    print("    ACTUALLY STEPPED BY A FUSED KERNEL TODAY: 0 / "
          f"{seam_instances}.", flush=True)

    # ---------------------------------------------------------------- the ledger
    # THE BOOKS, CLOSED. Every seam-instance that clears its source injection is in
    # exactly one bucket, and the buckets must sum to the ceiling. If they do not,
    # something in the walk above is double-counting and the ranked list is wrong.
    ledger: Dict[str, collections.Counter] = {
        name: collections.Counter() for name, _c, _s, _w in SEAMS}
    for entry in per_row:
        configuration = configuration_of[entry["row"]]
        for seam, cell in entry["seams"].items():
            clause = SOURCE_SEAM_CLAUSE.get(seam)
            if clause is not None and not clause(configuration):
                ledger[seam]["blocked by the source injection in the seam"] += 1
                continue
            if cell["served"]:
                ledger[seam]["SERVED by a shipped product"] += 1
                continue
            const_slot = {"B_to_H": "update_H", "D_to_E": "update_E",
                          "E_to_P": "update_P"}[seam]
            if cell["product"] is not None:
                ledger[seam]["reachable; a product exists and REFUSES on another "
                             "clause"] += 1
            elif cell["curl_arm"] is None or cell["constitutive_arm"] is None:
                ledger[seam]["reachable; a slot is UNSELECTED"] += 1
            elif CONSTITUTIVE_BODY.get((const_slot, cell["constitutive_arm"])) is None:
                ledger[seam]["NOT A FUSION CANDIDATE — the constitutive sub-step launches nothing, so there is no second kernel to weld"] += 1
            else:
                verdict = next(
                    (g["verdict"] for g in gaps
                     if g["seam"] == seam and g["curl_arm"] == cell["curl_arm"]
                     and g["constitutive_arm"] == cell["constitutive_arm"]), "")
                head = verdict.split(" —")[0].split(" (")[0]
                ledger[seam][f"reachable; NOT BUILT — {head}"] += 1

    print("\n" + "=" * 78, flush=True)
    print("THE LEDGER — every seam-instance that clears its source injection",
          flush=True)
    print("=" * 78, flush=True)
    total_ceiling = 0
    for seam, _c, _s, _w in SEAMS:
        instances = sum(1 for e in per_row if seam in e["seams"])
        blocked = ledger[seam]["blocked by the source injection in the seam"]
        reach = instances - blocked
        total_ceiling += reach
        if reach != source_ceiling[seam] and seam != "E_to_P":
            raise SystemExit(f"{seam}: the ledger reaches {reach} but the source "
                             f"ceiling is {source_ceiling[seam]}; the walk and the "
                             f"clause disagree")
        print(f"\n  {seam}: {instances} instances, {blocked} blocked by the source "
              f"seam, CEILING {reach}", flush=True)
        accounted = 0
        for bucket, count in sorted(ledger[seam].items(), key=lambda kv: -kv[1]):
            if bucket.startswith("blocked"):
                continue
            accounted += count
            print(f"      {count:4d}  {bucket}", flush=True)
        if accounted != reach:
            raise SystemExit(f"{seam}: buckets sum to {accounted}, ceiling is {reach}")
    print(f"\n  THE ABSOLUTE CEILING ON THIS CORPUS: {total_ceiling} / "
          f"{seam_instances} seam-instances ({total_ceiling / seam_instances:.1%}) "
          f"could EVER", flush=True)
    print("  be closed by a fused kernel, on any backend, however many are built — "
          "the rest carry a", flush=True)
    print("  source deposit inside the seam (driver.py:3283-3284, :3294-3299) whose "
          "constitutive half", flush=True)
    print("  `deposit_repair.repairable` REFUSES to reconstruct (an off-diagonal "
          "chi1inv or an", flush=True)
    print("  instantaneous chi2/chi3, deposit_repair.py:98-108); a deposit the repair "
          "CAN carry no", flush=True)
    print(f"  longer bounds anything. {total_served} of those {total_ceiling} are "
          f"reached today.",
          flush=True)

    # ------------------------------------------------- THE RECONCILED TAXONOMY
    # THE SAME EIGHT NAMES ALL THREE BOARDS NOW EMIT (release decision R1; the eighth,
    # withdraw_seam, arrived with the H_to_D seam on 2026-09-04), filed over
    # the SAME instances the ledger above walks. The ledger is kept and is NOT
    # replaced: it is this board's own vocabulary and its prose is what the
    # sub-reasons below carry. What the taxonomy adds is a decomposition a reader
    # can put beside the Triton and CUDA boards without translating three
    # vocabularies in their head, and the three tiers the headline now reads
    # against.
    #
    # THE PRECEDENCE IS NOT THIS LEDGER'S. The ledger puts the source seam first;
    # the taxonomy puts `not_fusion_surface` first, because an instance whose
    # constitutive half launches nothing has no weld for the deposit to obstruct
    # (R4). On THIS board the two orders agree on every instance — all 19
    # null-constitutive instances clear the source clause — and the agreement is
    # measured below rather than assumed, by requiring the two decompositions to
    # report the same served total and the same null count.
    taxonomy = fusion_taxonomy.Taxonomy(
        backend="metal",
        priced=priced,
        rows=len(record),
        platform_note=(
            f"METAL IS THE BACKEND WHERE THIS BUCKET IS REAL: MAX_BUFFER_BINDINGS = "
            f"{MAX_BUFFER_BINDINGS} (device.py:72) caps one dispatch's argument "
            f"table, so a fused signature needing more than {MAX_POINTERS} pointers "
            f"cannot be bound however it is written. MEASURED at "
            f"metal_kernels/fused_dispersive_pair.py: 30 pointers COMPILES, 30 "
            f"pointers + 5 packed scalars COMPILES at 31 bindings, 35 separate "
            f"bindings FAILS. Neither sibling backend has an analogue — CUDA passes "
            f"arguments in a 32,764-byte parameter space whose widest kernel here "
            f"spends 256 bytes, and Triton passes pointers as kernel arguments with "
            f"no argument-table ceiling — so a cell over by pointers alone is a CUDA "
            f"or Triton cell, not a dead one"),
        seam_instances={**{seam: sum(1 for e in per_row if seam in e["seams"])
                           for seam, _c, _s, _w in SEAMS},
                        h_to_d_seam.SEAM: len(h_to_d_entries)})
    verdict_of = {(g["seam"], g["curl_arm"], g["constitutive_arm"]): g["verdict"]
                  for g in gaps}
    for entry in per_row:
        configuration = configuration_of[entry["row"]]
        for seam, cell in entry["seams"].items():
            const_slot = {"B_to_H": "update_H", "D_to_E": "update_E",
                          "E_to_P": "update_P"}[seam]
            curl_slot = {"B_to_H": "step_B", "D_to_E": "step_D",
                         "E_to_P": "update_E"}[seam]
            clause = SOURCE_SEAM_CLAUSE.get(seam)
            verdict = verdict_of.get(
                (seam, cell["curl_arm"], cell["constitutive_arm"]), "")
            head = verdict.split(" —")[0].split(" (")[0]
            # 1. NOT A FUSION SURFACE — ahead of everything, R4. Guarded on the arm
            #    being SELECTED: CONSTITUTIVE_BODY has no row for None either, and
            #    reading an unselected slot as a null arm would file a coverage gap
            #    as an absence of the object.
            if (cell["constitutive_arm"] is not None
                    and CONSTITUTIVE_BODY.get(
                        (const_slot, cell["constitutive_arm"])) is None):
                taxonomy.add(
                    "not_fusion_surface", seam, entry["row"],
                    "NOT A FUSION CANDIDATE — the constitutive sub-step launches "
                    "nothing, so there is no second kernel to weld",
                    {"constitutive_slot": const_slot,
                     "constitutive_arm": cell["constitutive_arm"]})
            elif cell["served"]:
                taxonomy.add("served", seam, entry["row"],
                             f"SERVED by a shipped product ({cell['product']})",
                             {"product": cell["product"]})
            elif clause is not None and not clause(configuration):
                taxonomy.add(
                    "source_seam_unbracketable", seam, entry["row"],
                    "blocked by the source injection in the seam — the driver "
                    "deposits between the halves (driver.py:3283-3284 magnetic, "
                    ":3294-3299 electric) and deposit_repair.repairable REFUSES to "
                    "reconstruct this row's constitutive half BY NAME "
                    "(deposit_repair.py:98-108)",
                    {"refused_by": sorted(
                        k for k, v in (
                            ("off-diagonal chi1inv",
                             bool(configuration["has_offdiagonal_epsilon"])),
                            ("instantaneous chi2/chi3",
                             bool(configuration["has_nonlinearity"])),
                            ("an unreadable fold map",
                             bool(configuration["has_symmetry"])))
                        if v)})
            elif cell["curl_arm"] is None or cell["constitutive_arm"] is None:
                missing_slot = (curl_slot if cell["curl_arm"] is None
                                else const_slot)
                partner = (cell["constitutive_arm"] if cell["curl_arm"] is None
                           else cell["curl_arm"])
                taxonomy.add(
                    "missing_half", seam, entry["row"],
                    f"A KERNEL COVERAGE GAP: the census records NO admitting arm at "
                    f"{missing_slot}, so there is no pair to fuse — the array path "
                    f"steps this slot",
                    {"missing_slot": missing_slot,
                     "missing_kernel":
                         f"a metal_kernels arm on {missing_slot} admitting this row "
                         f"(the seam's other half is already covered by "
                         f"{partner!r}); until it exists the pair has no second "
                         f"body to weld and closing it is a SUB-STEP kernel's job, "
                         f"not a fusion one",
                     "partner_arm": partner})
            elif head.startswith("UNFUSABLE AS AN IN-PLACE SPLICE"):
                taxonomy.add("structurally_unweldable", seam, entry["row"],
                             verdict, {"kind": "stencil"})
            elif head.startswith("STRUCTURALLY REFUSED AT THE INTERLEAVE"):
                taxonomy.add("structurally_unweldable", seam, entry["row"], verdict,
                             {"kind": "interleave",
                              # CROSS-BACKEND NOTE, NOT A RECLASSIFICATION. This
                              # refusal is MEASURED and stands: perturbing P[c] on
                              # an off-diagonal row moves BOTH partners, so the
                              # PER-COMPONENT split shape every Metal E->P product
                              # takes is refused here. What the CUDA board shows is
                              # that the OTHER shape Metal's own verdict names --
                              # all three components in one launch -- was built
                              # there. That is a finding about the shape not yet
                              # attempted on Metal, and moving the instance on the
                              # strength of another backend's product would be
                              # crediting Metal for a kernel Metal does not have.
                              "the_shape_this_refuses": (
                                  "one launch per component (update_E(c) then every "
                                  "update_P(state, c)) — the driver's order only "
                                  "while update_E(c) reads nothing but component "
                                  "c's own volumes"),
                              "the_shape_it_does_not_refuse": (
                                  "all three components in one launch, which this "
                                  "verdict's own text names and which the CUDA "
                                  "board reports built as a monolithic update_E "
                                  "carrying one component's recurrence. NOT BUILT "
                                  "ON METAL, so this instance stays here")})
            elif head.startswith("UNFUSABLE ON METAL"):
                taxonomy.add("platform_specific", seam, entry["row"], verdict,
                             {"ceiling": MAX_POINTERS,
                              "bindings": MAX_BUFFER_BINDINGS})
            elif cell["product"] is not None:
                taxonomy.add(
                    "buildable_not_built", seam, entry["row"],
                    f"a product exists on this cell ({cell['product']}) and REFUSES "
                    f"on a clause other than the source seam — what is missing is "
                    f"coverage inside a built kernel, not a kernel",
                    {"product": cell["product"]})
            else:
                taxonomy.add(
                    "buildable_not_built", seam, entry["row"],
                    verdict or "reachable; no fused product covers this pair",
                    {"cell": [cell["curl_arm"], cell["constitutive_arm"]]})
    # The fourth seam, filed by the shared rule. What this board hands it is its own
    # PRODUCTS table: which entries span update_H+step_D (none today), and — the day
    # one does — a per-row `served_by` verdict, which is the only thing that can carry
    # a per-backend predicate into a rule three boards share. h_to_d_seam.price NAMES
    # the rows a board leaves unanswered rather than pricing them as unbuilt.
    h_to_d_block = h_to_d_seam.price(
        taxonomy, "metal", "metal_kernels", h_to_d_entries,
        products_spanning_the_seam=[f for f, spec in PRODUCTS.items()
                                    if spec["seam"] == h_to_d_seam.SEAM])
    taxonomy_block = taxonomy.finish(
        served_expected=total_served + h_to_d_block["served"],
        served_expected_source=(
            f"the per-row walk's own `served` flag, totalled at "
            f"build_fusion_matrix.py `total_served`, plus the H_to_D seam's served "
            f"({h_to_d_block['served']} of {h_to_d_block['instances_priced']}; "
            f"{h_to_d_block['served_note']})"))

    # THE TWO DECOMPOSITIONS MUST RECONCILE EXACTLY, and the only licensed
    # difference between them is R4's precedence. The ledger files an instance under
    # the source seam first; the taxonomy files it under `not_fusion_surface` first.
    # Every instance the two disagree about is therefore one that is BOTH
    # null-constitutive and source-blocked, and the count of those is what R4 moves.
    # Anything else disagreeing means one of the two walks is reading a different
    # corpus, so this raises rather than reporting a delta.
    _ledger_blocked = sum(c["blocked by the source injection in the seam"]
                          for c in ledger.values())
    _ledger_null = sum(
        v for seam in ledger for k, v in ledger[seam].items()
        if k.endswith("NOT A FUSION CANDIDATE — the constitutive sub-step launches "
                      "nothing, so there is no second kernel to weld"))
    _moved = _ledger_blocked - taxonomy_block["buckets"]["source_seam_unbracketable"]
    # The H_to_D seam's null rows are the same null update_H the ledger counts at
    # B_to_H, filed once more at the fourth seam; the ledger prices three seams and
    # the taxonomy four, so the difference is exactly that count.
    _h_to_d_null = h_to_d_block["buckets"].get("not_fusion_surface", 0)
    if taxonomy_block["buckets"]["not_fusion_surface"] != (
            _ledger_null + _moved + _h_to_d_null):
        raise SystemExit(
            f"the ledger and the reconciled taxonomy do not reconcile: the ledger "
            f"reports {_ledger_null} null-constitutive and {_ledger_blocked} "
            f"source-blocked, the taxonomy "
            f"{taxonomy_block['buckets']['not_fusion_surface']} and "
            f"{taxonomy_block['buckets']['source_seam_unbracketable']}, R4's "
            f"precedence accounts for only {_moved} of the difference and the H_to_D "
            f"seam's null update_H for {_h_to_d_null}. One of the two walks is "
            f"reading a different corpus.")
    taxonomy_block["r4_precedence_move"] = {
        "instances_moved_from_source_seam_to_not_fusion_surface": _moved,
        "what_it_is": (
            "instances this board's own ledger filed as blocked by the source "
            "injection and whose constitutive sub-step launches NOTHING. Under R4 "
            "an instance is counted in its most fundamental bucket: where there is "
            "no second kernel to weld, the deposit between the halves obstructs "
            "nothing"),
        "the_ledger_still_says": {"source_blocked": _ledger_blocked,
                                  "null_constitutive": _ledger_null},
    }
    taxonomy.report(lambda line: print(line, flush=True))
    print(f"\n  R4 PRECEDENCE: {_moved} instance(s) move from "
          f"source_seam_unbracketable to not_fusion_surface on this board "
          f"(the ledger's own order files them the other way)", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("THE GAPS, RANKED BY DEMAND", flush=True)
    print("=" * 78, flush=True)
    print("  ranked by REACHABLE demand — the rows that clear the seam's source "
          "injection —", flush=True)
    print("  with the raw row count beside it. Where they differ, the raw count is "
          "not buildable demand.", flush=True)
    for gap in gaps:
        print(f"  reach {gap['rows_clearing_the_source_seam']:4d}  (of "
              f"{gap['rows']:3d} rows)  {gap['seam']:8s} "
              f"({gap['curl_arm']}, {gap['constitutive_arm']})", flush=True)
        if gap.get("fused_pointer_bracket"):
            print(f"          pointers {gap['curl_pointers']} + "
                  f"{gap['constitutive_pointers']} -> "
                  f"{gap['fused_pointer_bracket'][0]}..{gap['fused_pointer_bracket'][1]}"
                  f" (ceiling {MAX_POINTERS}; the "
                  f"{gap['curl_arity']}c x {gap['constitutive_arity']}c products on "
                  f"this seam share "
                  f"{gap['shared_pointers_measured_on_this_arity_class']}, from "
                  f"{gap['arity_class_members']}; the seam POOLED across arities "
                  f"would have said "
                  f"{gap['seam_pooled_bracket'][0]}..{gap['seam_pooled_bracket'][1]})",
                  flush=True)
        print(f"          {gap['verdict']}", flush=True)
        if gap.get("over_by_binding_alone"):
            print(f"          {gap['over_by_binding_alone']}", flush=True)
        if gap.get("in_seam_passes_no_product_carries"):
            print(f"          CARRY: these rows run "
                  f"{', '.join(gap['in_seam_passes_no_product_carries'])} INSIDE the "
                  f"seam and no fused product on either backend carries it, so the "
                  f"binding verdict above is not the operative constraint",
                  flush=True)
            if gap.get("carry_measured_by"):
                print(f"          CARRY EVIDENCE: {gap['carry_measured_by']}",
                      flush=True)
        if "sequencing_constraint" in gap:
            print(f"          SEQUENCING: {gap['sequencing_constraint']}", flush=True)

    # ------------------------------------------------- against the standing records
    # THE ONLY REASON TO TRUST A DERIVATION IS THAT IT REPRODUCES ONE. Three of the
    # five products were priced independently on 2026-08-19 by scripts that walked
    # the same census with their own clause lists. Those numbers are asserted here,
    # not quoted: a drift means this walk has a bug, and a silent drift is worse than
    # no cross-check at all.
    standing = {
        "folded_fused_magnetic_pair": (
            RESULTS / "metal_folded_fused_magnetic_pair_2026-08-19"
            / "corpus_admission.json", ("folded_fused_magnetic_pair_B_to_H",)),
        "folded_fused_pair": (
            RESULTS / "metal_folded_fused_pair_2026-08-19"
            / "corpus_admission.json", ("folded_fused_pair_D_to_E",)),
    }
    print("\n" + "=" * 78, flush=True)
    print("CROSS-CHECK AGAINST THE STANDING PER-PRODUCT RECORDS", flush=True)
    print("=" * 78, flush=True)
    # THE ONE PRODUCT THAT MUST NOT REPRODUCE ITS STANDING RECORD, and it is checked
    # HARDER than the ones that must. `folded_fused_magnetic_pair` lost a clause this
    # cut, so an unchanged number would mean the clause never bound anything and the
    # carry bought nothing. The check therefore evaluates BOTH ladders over the same
    # census rows: the PRE-CARRY one (the clause list of ../fusion_matrix_metal_
    # 2026-08-20_closed, restored here) must reproduce the standing record exactly —
    # which is what says the METHOD is unchanged — and the shipped one must exceed it
    # by exactly the rows the retired clause was the ONLY thing refusing.
    # BOTH folded products lost this clause now, and both are checked the hard way:
    # the pre-carry ladder must reproduce the standing record EXACTLY (which is what
    # says the method did not move) and the shipped ladder must exceed it by exactly
    # the rows the retired clauses were the ONLY thing refusing (which is what says the
    # carry bought something). An unchanged number would mean the clause never bound.
    #
    # THE STANDING RECORDS WERE WALKED ON THE 186-ROW BASIS. Since 2026-09-03 the
    # census holds 194 rows, so the pre-carry ladder reproduces them only if none of
    # the eight added rows survives it — asserted here on every cut, never assumed
    # (the 2026-09-03 board reproduced both: 45 and 2).
    #
    # TWO CLAUSES, NOT ONE, SINCE 2026-08-28. The standing records were cut while BOTH
    # `every_fold_is_mirror_metallic` AND the blanket in-seam source refusal were in the
    # ladder. `deposit_repair.repair_cells` retired the second one for these two
    # families, so reconstructing the standing number now needs BOTH put back --
    # restoring only the fold clause reconstructs a ladder nobody ever ran and the check
    # correctly refused to reconcile. The reconstruction restores every clause retired
    # SINCE the standing record, and each one is required to have refused at least one
    # row on its own: a retirement that bound nothing is a retirement that bought
    # nothing, and the delta would be decoration.
    WIDENED: Dict[str, Tuple[Tuple[str, Callable[[dict], bool]], ...]] = {
        "folded_fused_magnetic_pair": (
            ("every_fold_is_mirror_metallic", _every_fold_is_mirror_metallic),
            ("no_magnetic_source", _no_magnetic_source)),
        "folded_fused_pair": (
            ("every_fold_is_mirror_metallic", _every_fold_is_mirror_metallic),
            ("no_electric_source", _no_electric_source)),
    }
    reproduced = {}
    for family, (path, keys) in standing.items():
        prior = json.loads(path.read_text())
        for key in keys:
            expected = prior[key]["admitted"]
            got = product_detail[family]["admitted"]
            retired = WIDENED.get(family)
            clause = None if retired is None else "+".join(n for n, _ in retired)
            if retired is None:
                reproduced[f"{path.parent.name}:{key}"] = {"expected": expected,
                                                           "measured": got}
                status = "MATCH" if expected == got else "*** DOES NOT REPRODUCE"
                print(f"  {path.parent.name}/{key}: {expected} vs {got}  {status}",
                      flush=True)
                if expected != got:
                    raise SystemExit(
                        f"{family}: the standing record says {expected}, this walk "
                        f"says {got}; one of the two clause lists is wrong and "
                        f"neither number may be used until that is settled")
                continue
            # Re-run the pre-carry ladder, by the SAME construction the shipped one
            # uses above (both halves' certified predicates, then the seam clauses),
            # with the retired clause put back at the end.
            spec = PRODUCTS[family]
            # THE PRIMARY CELL, through the same normaliser the ladder uses. Both
            # families walked here are single-cell (folded B->H and folded D->E), so
            # this is the entry's own halves either way; reading it through
            # `_cells_of` is what keeps the two walks one construction.
            primary = _cells_of(spec)[0]
            base = list(record)
            for key in primary["halves"]:
                base = [r for r in base if covered(r, key)]
            shipped_rows = [r for r in base
                            if all(test(r["configuration"])
                                   for _, test in primary["seam_clauses"])]
            before = sum(1 for r in shipped_rows
                         if all(test(r["configuration"]) for _, test in retired))
            delta = got - before
            only_this_clause = sum(
                1 for r in shipped_rows
                if not all(test(r["configuration"]) for _, test in retired))
            # PER CLAUSE, so a retirement that bound nothing cannot hide inside a
            # nonzero total. Each retired clause must have refused at least one row.
            per_clause = {
                name: sum(1 for r in shipped_rows if not test(r["configuration"]))
                for name, test in retired}
            reproduced[f"{path.parent.name}:{key}"] = {
                "standing_record": expected, "pre_carry_ladder": before,
                "measured": got, "delta": delta,
                "rows_the_retired_clause_alone_refused": only_this_clause,
                "rows_each_retired_clause_refused": per_clause,
                "retired_clause": clause}
            ok = (before == expected and delta == only_this_clause and delta > 0
                  and all(count > 0 for count in per_clause.values()))
            print(f"  {path.parent.name}/{key}: standing {expected}, pre-carry "
                  f"ladder {before}, this walk {got} (+{delta} on the retired "
                  f"{clause}; per clause {per_clause})  "
                  f"{'MATCH' if ok else '*** DOES NOT RECONCILE'}", flush=True)
            if not ok:
                raise SystemExit(
                    f"{family}: standing {expected}, pre-carry ladder {before}, "
                    f"shipped ladder {got}, rows the retired clauses alone refused "
                    f"{only_this_clause}, per clause {per_clause}. The widening does "
                    f"not reconcile with the standing record and no number here may "
                    f"be used until it does")
    # The Triton track priced the SAME construction with its own predicates. Not an
    # identity — different backend, different halves — so it is REPORTED, not asserted.
    print("  cross-track, reported not asserted: the Triton censuses of 2026-08-19 "
          "measured 45", flush=True)
    print("  (folded B->H), 2 (folded D->E) and 12 (complex B->H) on the same corpus "
          "with Triton", flush=True)
    print("  predicates. This walk measures 45, 2 and 12 on METAL predicates.",
          flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"THE {len(PRODUCTS)} SHIPPED PRODUCTS, CLAUSE BY CLAUSE", flush=True)
    print("=" * 78, flush=True)
    for family, detail in product_detail.items():
        print(f"\n  {family}  [{detail['seam']}  {tuple(detail['pair'])}]", flush=True)
        for step in detail["clause_ladder"]:
            print(f"      after {step['clause']:38s} {step['surviving']:4d}", flush=True)
        # A MULTI-CELL PRODUCT SHOWS ITS SUM, so the console says 47 + 2 = 49 rather
        # than an ADMITTED that no printed ladder reaches. The first ladder above IS
        # the primary cell's; the extra cells print their own beneath it.
        for label, cell in list(detail.get("cells", {}).items())[1:]:
            print(f"      cell {label}  [bound: {cell['binding']}]", flush=True)
            for step in cell["clause_ladder"]:
                print(f"          after {step['clause']:34s} {step['surviving']:4d}",
                      flush=True)
        if detail.get("cells"):
            per_cell = " + ".join(str(c["admitted"]) for c in detail["cells"].values())
            print(f"      ADMITTED {detail['admitted']} / {len(record)}  "
                  f"({per_cell} across {len(detail['cells'])} cells)", flush=True)
        else:
            print(f"      ADMITTED {detail['admitted']} / {len(record)}", flush=True)

    # The rows special_kz and folded_beta own, READ from plan_step.selected rather than
    # spelled: the staleness note below states them against the census's row count.
    beta_owned = sorted({
        f"{r['leg']}:{r['row']}" for r in record
        for name in ((r.get("plan_step") or {}).get("selected") or {}).values()
        if "special_kz" in str(name) or "folded beta" in str(name)})
    result = {
        "census": str(CENSUS),
        "coverage_column": "covered",
        "rows": len(record),
        "denominator": priced,
        "denominator_three_seam_ledger": seam_instances,
        "denominator_justification": taxonomy.priced_justification,
        # THE FOURTH SEAM'S OWN BLOCK: what sits between the halves (located in
        # driver.py at cut time), the probe it was priced from, its buckets and its
        # instances.
        "h_to_d_seam": h_to_d_block,
        # DERIVED FROM THE INSTANCE LEDGER, 2026-09-06. Through the `_hd` cut this key
        # carried `total_served`, the three-seam walk's total, while the taxonomy
        # summed four seams — 358 beside 405 in one artifact. The served total is now
        # the count of instances the taxonomy filed served (and lists), and
        # `fusion_taxonomy.headline_agreement` refuses the cut if any served key in
        # this artifact says otherwise. The three-seam walk's own total keeps its own
        # name beside it: it is a different fact, not a second spelling of this one.
        "served_by_a_fused_product": taxonomy_block["buckets"]["served"],
        "served_three_seam_ledger": total_served,
        # THE DISPATCH TOTAL IS NOT SET HERE, and the empty comment is deliberate: see
        # the assignment after ``dispatch`` is measured, ~200 lines below. Putting
        # ``dispatch["served_in_dispatch"]`` in this literal is what the first repair
        # did, and it raised UnboundLocalError on every cut because this dict is built
        # BEFORE the measurement runs -- the Metal board could not produce an artifact
        # at all, and the test written beside it grepped this file's text instead of
        # executing the board, so the suite stayed green over a board that was dead.
        "source_seam_ceiling": {
            "B_to_H": source_ceiling["B_to_H"],
            "D_to_E": source_ceiling["D_to_E"],
            "E_to_P": source_ceiling["E_to_P"],
            "what_it_is": (
                "the largest number of rows ANY fused product, built or not, can ever "
                "serve at that seam. The driver deposits a magnetic source between "
                "step_B and update_H (driver.py:3283-3284) and an electric source "
                "between step_D and update_E (driver.py:3294-3299). A launch is no "
                "longer barred from straddling that deposit: meep_gpu/deposit_repair.py "
                "saves the affected points before the fused launch and recomputes the "
                "constitutive half at exactly those points after the driver has "
                "injected, filled symmetry and cleared walls. What still bars a row is "
                "a constitutive half `deposit_repair.repairable` refuses to "
                "reconstruct — an off-diagonal chi1inv (update_E is then a stencil "
                "over the partner volumes, stepping.py:1228-1251) or an instantaneous "
                "chi2/chi3 — both on the D side only. E->P carries no injection "
                "(driver.py:3303-3306), so its ceiling is simply the rows that have "
                "an update_P pass."),
            "pre_repair": {
                "B_to_H": pre_repair_ceiling["B_to_H"],
                "D_to_E": pre_repair_ceiling["D_to_E"],
                "E_to_P": pre_repair_ceiling["E_to_P"],
                "what_it_is": (
                    "the same ceiling under the rule every board before this cut used "
                    "— no launch may be on both sides of a deposit at all. The "
                    "difference between the two is what deposit_repair.py buys the "
                    "CEILING, which is a separate question from what any shipped "
                    "product reaches."),
            },
            "source_field_types": {str(k): v for k, v in kinds.items()},
            "consequence": (
                "the D->E seam's addressable market is no longer the "
                f"{pre_repair_ceiling['D_to_E']} magnetic-source-only rows but "
                f"{source_ceiling['D_to_E']}: every electric-source row whose "
                "constitutive half is the diagonal linear accumulation the repair "
                "inverts is now reachable. The B->H seam's is "
                f"{source_ceiling['B_to_H']} (was {pre_repair_ceiling['B_to_H']}) — "
                "the B seam has no repairability refusal at all, so its ceiling is "
                "every row."),
        },
        "disjointness_cross_check": disjointness_cross_check(taxonomy_block),
        # THE MEASUREMENT'S OWN WORDS, and the sentence this replaces was FALSE.
        # It said "arms.arms_for skips an unwired arm so plan_step cannot compose
        # one" — but `_install_fused_pairs` iterates `arms.registered(curl_name)`,
        # NOT `arms_for`, so `wired=False` does not block fusion at all, and the
        # shipped composer was measured selecting fused products on twelve of the
        # fourteen driver-route cases (2026-09-10). It also said `plan_fast_path`
        # "still returns None on every branch" and named a hard-coded `fuse=False`
        # at a line number, both of which the release round retired. Three wrong
        # clauses in one string, beside a headline that was right.
        #
        # So the reason is no longer typed here at all: `dispatch_reachability`
        # measures it off the tree on every cut and reports either `why_zero` (the
        # structural refusal) or the arms it credited. The wiring flag stays beside
        # it as the fact it actually is — a registration state — with its own
        # sentence rather than a causal claim it does not support.
        # RENAMED WITH THE NUMBER IT DESCRIBES. While the key was `why_dispatch_is_zero`
        # it asked the reader to consult a `why_zero` that has been None since the
        # Metal table became reachable -- a question about a zero that is not there.
        # What a reader needs instead is where the number comes from and what it is
        # NOT: an upper bound, conditioned on a run that asked for dispatch.
        "how_dispatch_is_measured": (
            "MEASURED, not asserted: served_by_a_fused_product_in_dispatch is read "
            "from dispatch_reachability.served_in_dispatch('metal', ...) "
            "(parity/meep_gpu/dispatch_reachability.py), which walks "
            "meep_gpu/fastpath.py's parse tree and the meep_gpu siblings it imports "
            "to decide whether a Metal product is reachable from the dispatch seam at "
            "all, then joins every served instance to an arm label and asks the "
            "release table whether that arm dispatches on that row. It is an UPPER "
            "BOUND -- see served_in_dispatch_measurement.clauses_below_this_one for "
            "the three clauses it does not evaluate -- and the enable condition it "
            f"is counted under is: {_reach.dispatch_condition()} (read off "
            "fastpath.DISPATCH_BY_DEFAULT on this cut). A zero here "
            "would carry its own reason in that block's `why_zero`; a non-zero one "
            "carries its per-arm `arms` table."),
        "wired_flag_is_not_the_reason": (
            f"{sum(1 for v in wired.values() if not v)} of {len(wired)} Metal fused "
            f"families register wired=False, measured at run time. That flag does "
            f"NOT block fusion: _install_fused_pairs iterates arms.registered("
            f"curl_name), not arms_for, so an unwired weld is still composed — "
            f"measured on 2026-09-10, the shipped composer selects a fused product "
            f"on twelve of the fourteen driver-route cases"),
        "wired_true_families": sorted(f for f, v in wired.items() if v),
        "per_seam": {**{seam: {
            "instances": sum(1 for e in per_row if seam in e["seams"]),
            "served": sum(1 for e in per_row if e["seams"].get(seam, {}).get("served")),
            "between_the_halves": between,
        } for seam, _c, _s, between in SEAMS},
            h_to_d_seam.SEAM: {
                "instances": h_to_d_block["instances_priced"],
                "served": h_to_d_block["served"],
                "between_the_halves":
                    h_to_d_block["what_sits_between_the_halves"]["between_summary"]}},
        "matrix": [
            {"seam": seam, "curl_arm": curl_arm, "constitutive_arm": const_arm,
             "rows": len(labels),
             "fused_product": next(
                 (f for f, s in PRODUCTS.items() if s["seam"] == seam
                  and tuple(s["pair"]) == (curl_arm, const_arm)), None),
             "admitted": sum(
                 1 for lab in labels
                 for f, s in PRODUCTS.items()
                 if s["seam"] == seam and tuple(s["pair"]) == (curl_arm, const_arm)
                 and lab in served[f])}
            for (seam, curl_arm, const_arm), labels in sorted(
                by_pair.items(), key=lambda kv: (kv[0][0], -len(kv[1])))],
        "gaps_ranked_by_demand": gaps,
        "ledger": {seam: dict(counter) for seam, counter in ledger.items()},
        "absolute_ceiling_seam_instances": total_ceiling,
        "reproduces_the_standing_records": reproduced,
        "products": product_detail,
        "signature_census": signatures,
        "pointer_sharing_calibration": calibration,
        "binding_ceiling": {
            "MAX_BUFFER_BINDINGS": MAX_BUFFER_BINDINGS,
            "max_pointers_with_packed_scalars": MAX_POINTERS,
            "measured_at": "meep_gpu/metal_kernels/fused_dispersive_pair.py:44-62",
            "measurement": {
                "35 separate bindings": "FAILED, 'buffer' attribute out of bounds",
                "30 pointers + 5 scalars in one Params&": "COMPILES (31 bindings)",
                "30 pointers": "COMPILES"},
        },
        "unfused_reasons_per_seam": {k: dict(v) for k, v in unfused_reason.items()},
        "in_seam_pass_liveness_read_from_plan_step_live": {
            seam: {name: sum(1 for r in record
                             if name in live_in_seam_passes(r, seam))
                   for name in names}
            for seam, names in IN_SEAM_PASSES.items()},
        "in_seam_pass_liveness_the_derived_rule_the_carry_cuts_used": {
            seam: {name: sum(1 for r in record
                             if name in derived_in_seam_passes(r["configuration"],
                                                               seam))
                   for name in names}
            for seam, names in IN_SEAM_PASSES.items()},
        "in_seam_pass_liveness_rows_the_derived_rule_gets_wrong": dict(disagreement),
        "in_seam_passes_carried_by_some_product": {
            seam: sorted(names) for seam, names in carried_in_practice.items()},
        "carry_gap_totals": {name: len(v) for name, v in carry_gap.items()},
        "carry_gap": {name: v for name, v in carry_gap.items()},
        "per_row": per_row,
        "is_an_upper_bound": True,
        "clauses_not_evaluable_from_the_census_block": sorted({
            clause for spec in PRODUCTS.values() for clause in spec["not_evaluable"]
        } | {
            "layout clauses (dtype / C-contiguity / grid.shape) beyond what the two "
            "halves already check on the lifted object",
            "PML coefficient-table lengths beyond what the two halves already check",
            "the per-launch residency declaration, which plan_step builds fresh and "
            "the census asked with an empty device.Residency()",
            "whether a candidate fused body is CORRECT. This round asks only whether "
            "the pair is demanded, whether a product covers it, and whether the "
            "signature can be bound. Nothing here is a byte-parity claim",
            "how many volumes a candidate's two halves actually share. The bracket is "
            "calibrated from the shipped products' measured sharing on the same "
            "seam; a candidate whose curl is larger than any calibrating one is "
            "reported as a range and not resolved",
            "the E->P chain's launch sequencing — ade_update_p launches per driven "
            "component with the host rotating buffers between launches "
            "(ade_update_p.py:13-18); its 15 seam-instances are priced for BINDING "
            "FITNESS only",
        }),
        "staleness": {
            "census_finished": "2026-08-19 02:15",
            "modules_edited_after_it": [
                "meep_gpu/metal_kernels/special_kz.py (2026-08-19 05:17)",
                "meep_gpu/metal_kernels/folded_beta.py (2026-08-19 05:16)"],
            "measured_consequence": (
                "results/metal_coverage_special_kz_reclose_2026-08-19/smoke/"
                "test_special_kz.json re-ran the affected rows with the current tree "
                "and records plan_step selecting 'special_kz complex beta' at ALL FOUR "
                "slots on TestSpecialKz.test_special_kz, where tranche 6 left "
                "update_H/update_E UNSELECTED. That is the ONE row whose pair the "
                "tranche-6 record gets wrong."),
            "correction": (
                "APPLIED since the residue round, not merely noted: `_apply_reclose` "
                "substitutes the reclose record for that one row (a measurement by "
                "the same battery on the same host), with the substitution's scope "
                "asserted at exactly ['tests:TestSpecialKz.test_special_kz'] — the "
                "prior cuts carried the stale value UNCORRECTED because no fused "
                "product existed on either side of the pair, and the "
                "beta_complex_fused_* welds of this round removed that licence: a "
                "stale (curl, None) would now miscount served."),
            "bound": (f"special_kz and folded_beta together own {len(beta_owned)} of "
                      f"{len(record)} rows at each curl->constitutive seam (read from "
                      "plan_step.selected on this census; 6 of 186 on the 2026-08-19 "
                      "census, 1+1+1+3); the one row the record got "
                      "wrong is substituted, and the reclose's other five rows carry "
                      "unparametrised names the census does not use, so they can "
                      "substitute nothing (asserted by _apply_reclose's scope floor)"),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    print("\n" + "=" * 78, flush=True)
    print("THE HEADLINE", flush=True)
    print("=" * 78, flush=True)
    print(f"  denominator                                 {priced} "
          f"seam-instances ({seam_instances} on the three-seam ledger + "
          f"{len(h_to_d_entries)} H_to_D)", flush=True)
    served_derived = taxonomy_block["buckets"]["served"]
    print(f"  SERVED BY A FUSED PRODUCT (predicate)       {served_derived}  "
          f"({served_derived / priced:.1%} of {priced}; derived from the instance "
          f"ledger: {total_served} on the three-seam ledger + "
          f"{h_to_d_block['served']} H_to_D)", flush=True)
    # SERVED IN DISPATCH — MEASURED off the shipped ladder since 2026-08-29, not
    # read off this board's own wiring register. The register answers "did anyone
    # wire this product into the Metal planner"; the question here is whether the
    # DISPATCHER can select it.
    #
    # THE ANSWER IS NO LONGER STRUCTURALLY ZERO, which is the 2026-09-10 change and
    # the reason the comment that stood here is gone. The Metal ladder does not live
    # in `fastpath.py` — its rungs and its release table are in
    # `meep_gpu/metal_dispatch.py`, deliberately, so a Metal release edit never
    # re-drifts the Triton driver_dispatch record — so a walk of `fastpath.py`'s
    # parse tree ALONE would report the Metal table unreachable while it dispatched.
    # `dispatch_reachability` now follows the meep_gpu siblings `fastpath` imports
    # and counts a sibling only when it CALLS a composer, which is what separates
    # the Metal ladder from `subnormal_policy`'s lazy import of the mps executor arm.
    #
    # HANDED EVERY SERVED INSTANCE since 2026-09-06 rather than an empty list, so its
    # `served_counted` is the served headline and not 0. When the seam is not
    # reachable the measurement still returns before the per-instance walk, with its
    # structural reason; when it is, `instances` carries the label the composer
    # SELECTED per row so the credit can be attributed rather than inferred.
    # THE SEAM RIDES ALONG so served_in_dispatch can evaluate the slot-contention half
    # of _pair_may_absorb instead of declaring it. Four-tuples; a three-tuple would get
    # the old count with slot_contention.evaluated False.
    dispatch = _reach.served_in_dispatch("metal", [
        (e["row"], s["product"], configuration_of[e["row"]], seam)
        for e in per_row for seam, s in e["seams"].items() if s["served"]
    ] + [
        (i["row"], i["served_by"], configuration_of[i["row"]], h_to_d_seam.SEAM)
        for i in h_to_d_block["instances"] if i["bucket"] == "served"])
    # READ, NOT TYPED, AND ASSIGNED HERE BECAUSE IT CANNOT BE ASSIGNED EARLIER. This was
    # the literal ``0`` until 2026-09-11, left from the round when no Metal arm was
    # released -- and by then the board's own measurement in the same artifact read 176
    # over 96 rows, with ``why_zero`` None and a non-empty ``arms`` table. Two numbers
    # for one question in one file, and no floor on the dispatch axis to refuse it.
    # ``dispatch_agreement.dispatch_agreement`` is that floor now and compares THIS key
    # against that measurement on every cut; ``result`` is built above, so the value
    # lands once the measurement exists rather than in the literal that cannot see it.
    result["served_by_a_fused_product_in_dispatch"] = dispatch["served_in_dispatch"]

    print(f"  served IN DISPATCH                          "
          f"{dispatch['served_in_dispatch']}   "
          f"({dispatch.get('why_zero') or 'released arms: ' + ', '.join(dispatch['arms'])})",
          flush=True)
    print(f"  the absolute ceiling, any backend           {total_ceiling}  "
          f"({total_ceiling / seam_instances:.1%})", flush=True)
    # THE BUCKETS ARE CLASSIFIED BY VERDICT HEAD AND THE HEADS ARE A NAME LIST, so a
    # verdict class added upstream and not added here would silently vanish from the
    # decomposition while the run still printed a headline — the same name-drift
    # failure the _wiring floor catches. MEASURED WHEN THE STENCIL VERDICT WAS ADDED:
    # the buckets fell to 153 of 172 and nothing complained. The floor at the bottom
    # of this block is what turns that into a refusal.
    fits = sum(v for seam in ledger for k, v in ledger[seam].items()
               if k.startswith("reachable; NOT BUILT — FITS")
               or k.startswith("reachable; NOT BUILT — BINDING-MARGINAL")
               or k.startswith("reachable; NOT BUILT — UNDETERMINED"))
    blocked_ceiling = sum(v for seam in ledger for k, v in ledger[seam].items()
                          if k.startswith("reachable; NOT BUILT — UNFUSABLE ON METAL"))
    # A CELL WITH NO ARITY-MATCHED PRODUCT TO CALIBRATE AGAINST GETS ITS OWN BUCKET,
    # and deliberately NOT a place inside "would fit": no bracket was computed for it,
    # so counting it as fitting would be reading a verdict out of an absence. Zero on
    # this corpus — the only such cell is the E->P off-diagonal one, whose operative
    # verdict is the interleave refusal — and it is written anyway, because a bucket
    # that appears for the first time on some later corpus must land somewhere named
    # rather than fall through the floor below.
    uncalibrated = sum(v for seam in ledger for k, v in ledger[seam].items()
                       if k.startswith("reachable; NOT BUILT — NO ARITY-MATCHED"))
    # TWO STRUCTURAL REFUSALS, COUNTED SEPARATELY, because they are different
    # facts and one label over both would misdescribe the E->P instance. The
    # STENCIL is a D->E fact: the off-diagonal constitutive reads partner D volumes
    # the curl half writes in place in the same launch. The INTERLEAVE is an E->P
    # fact: there is no curl half at all, and what breaks is that update_E(c) reads
    # partner polarizations a per-component fusion has already advanced. Both are
    # measured; both are structural; neither is a binding verdict.
    #
    # THE STENCIL BUCKET NARROWED ON 2026-09-01. Its rows are cells no
    # SCRATCH-OUTPUT weld covers yet, not cells nothing can cover: four such welds
    # shipped that day and emptied the four cells that had held every one of its
    # instances. The label moved with the finding.
    stencil = sum(v for seam in ledger for k, v in ledger[seam].items()
                  if k.startswith("reachable; NOT BUILT — UNFUSABLE AS AN IN-PLACE"))
    interleave = sum(v for seam in ledger for k, v in ledger[seam].items()
                     if k.startswith("reachable; NOT BUILT — STRUCTURALLY REFUSED"))
    structural = stencil + interleave
    null_arm = sum(v for seam in ledger for k, v in ledger[seam].items()
                   if k.endswith("NOT A FUSION CANDIDATE — the constitutive sub-step launches nothing, so there is no second kernel to weld"))
    refused = sum(v for seam in ledger for k, v in ledger[seam].items()
                  if k.endswith("a product exists and REFUSES on another clause"))
    unselected = sum(v for seam in ledger for k, v in ledger[seam].items()
                     if k.endswith("a slot is UNSELECTED")
                     or k.startswith("reachable; NOT BUILT — UNFUSABLE — a slot"))
    buckets = {"served": total_served, "would fit, not built": fits,
               "cannot be bound on Metal": blocked_ceiling,
               "no arity-matched calibration": uncalibrated,
               "structurally unfusable (stencil)": stencil,
               "structurally refused (interleave)": interleave,
               "null constitutive": null_arm, "product refuses": refused,
               "no admitting arm": unselected}
    if sum(buckets.values()) != total_ceiling:
        unclassified = {f"{seam}: {k}": v for seam in ledger
                        for k, v in ledger[seam].items()
                        if not k.startswith("blocked by the source")}
        raise SystemExit(
            f"the buckets total {sum(buckets.values())} but {total_ceiling} "
            f"seam-instances are reachable, so {total_ceiling - sum(buckets.values())} "
            f"fall into no bucket and would have vanished from the decomposition. "
            f"Buckets: {buckets}. Ledger keys: {unclassified}")
    print(f"  BUCKET FLOOR: the {len(buckets)} buckets total "
          f"{sum(buckets.values())} == the {total_ceiling} reachable", flush=True)
    print(f"  of the {total_ceiling} reachable:", flush=True)
    print(f"      {total_served:4d}  served today", flush=True)
    print(f"      {fits:4d}  a fused product WOULD FIT under the binding ceiling and "
          f"is NOT BUILT", flush=True)
    print(f"      {blocked_ceiling:4d}  CANNOT be bound on Metal at any sharing "
          f"(over 30 pointers) — a ceiling with NO CUDA analogue", flush=True)
    print(f"      {uncalibrated:4d}  NO ARITY-MATCHED CALIBRATION — no shipped "
          f"product fuses this shape, so no bracket is claimed", flush=True)
    print(f"      {interleave:4d}  STRUCTURALLY REFUSED at the per-component "
          f"interleave — update_E(c) reads partner polarizations a fused E->P "
          f"launch has already advanced (measured by perturbation)", flush=True)
    print(f"      {stencil:4d}  UNFUSABLE AS AN IN-PLACE SPLICE — the "
          f"constitutive arm is a stencil over the curl arm's output; a "
          f"SCRATCH-OUTPUT weld is not blocked (four shipped 2026-09-01)",
          flush=True)
    print(f"      {null_arm:4d}  the constitutive sub-step launches nothing — no "
          f"fusion exists to build", flush=True)
    print(f"      {refused:4d}  a product covers the pair and refuses on a clause "
          f"other than the source seam", flush=True)
    print(f"      {unselected:4d}  a slot is UNSELECTED, so there is no pair",
          flush=True)
    result["e_to_p_release_binding"] = release_binding
    result["release_binding_per_product"] = all_credits_bound
    # PER CELL, and only where a product declares more than one: the artifact each
    # extra cell is credited against, which is not always the family's own weld.
    result["release_binding_per_cell"] = cell_credits_bound
    result["headline"] = {
        "denominator": priced,
        "denominator_three_seam_ledger": seam_instances,
        # DERIVED from the instance ledger (see `served_by_a_fused_product` above);
        # the per-seam split is the taxonomy's own, so the four numbers sum to it.
        "served_by_predicate": taxonomy_block["buckets"]["served"],
        "served_by_predicate_per_seam": {
            seam: counts.get("served", 0)
            for seam, counts in taxonomy_block["per_seam"].items()},
        "served_three_seam_ledger": total_served,
        "served_in_dispatch": dispatch["served_in_dispatch"],
        "served_in_dispatch_measurement": dispatch,
        "absolute_ceiling": total_ceiling,
        "reachable_would_fit_not_built": fits,
        "reachable_cannot_be_bound_on_metal": blocked_ceiling,
        # THE KEY NAME WAS WRONG UNTIL 2026-09-02 AND THE NUMBER WAS NOT. One key
        # called `reachable_structurally_unfusable_stencil` published the SUM of two
        # different measurements, and on this corpus that sum is entirely the OTHER
        # one: the stencil bucket has been 0 since the four scratch-output welds of
        # 2026-09-01, and the 1 it reported is an E->P INTERLEAVE refusal. A reader
        # comparing it against the sibling boards' stencil counts was comparing two
        # different facts. Split, with each half under its own name; the total is
        # kept beside them so nothing that read the old key loses the number.
        "reachable_structurally_unfusable_stencil": stencil,
        "reachable_structurally_refused_at_the_interleave": interleave,
        "reachable_structurally_unweldable_total": structural,
        "buckets_total_the_reachable_ceiling": sum(buckets.values()) == total_ceiling,
        "reachable_null_constitutive_no_fusion_exists": null_arm,
        "reachable_product_exists_but_refuses": refused,
        "reachable_slot_unselected": unselected,
        "reachable_no_arity_matched_calibration": uncalibrated,
        # THE RECONCILED HEADLINE (R1/R2) — the same key names on all three boards.
        "taxonomy": taxonomy_block["buckets"],
        "tiers": {name: block["value"]
                  for name, block in taxonomy_block["tiers"].items()},
    }
    # THE RECONCILED DECOMPOSITION, in full and under the shared key. The board's
    # own ledger above is unchanged and stays: this is the translation, not a
    # replacement, and every bucket here carries the ledger's own prose beneath it.
    result["taxonomy"] = taxonomy_block
    # FLOOR 7: every served total this artifact publishes, at any depth, is the
    # instance ledger's count — refused, not recorded, where one is not.
    result["headline_agreement"] = fusion_taxonomy.headline_agreement(
        result, taxonomy_block, "metal")
    # AND THE SAME FLOOR ON THE DISPATCH AXIS. `served` and `served_in_dispatch` are
    # different numbers, and until 2026-09-11 only the first had a floor -- which is
    # how the Metal board came to publish a hardcoded dispatch 0 beside its own
    # measured 176 in one artifact. `dispatch_agreement` reads the measurement as the
    # authority and refuses any other number for the same question, listing (never
    # comparing) the totals that sit under a declared different CONDITION.
    result["dispatch_agreement"] = dispatch_agreement.dispatch_agreement(
        result, dispatch, "metal")
    print(f"  DISPATCH AGREEMENT: "
          f"{result['dispatch_agreement']['served_in_dispatch']} in dispatch at "
          f"{len(result['dispatch_agreement']['paths_checked'])} published paths, "
          f"all read from the measurement", flush=True)
    print(f"  HEADLINE AGREEMENT: {result['headline_agreement']['served']} served at "
          f"{len(result['headline_agreement']['paths_checked'])} published paths, "
          f"all derived from the instance ledger", flush=True)
    out = out_dir / "fusion_matrix.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}  ({result['elapsed_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
